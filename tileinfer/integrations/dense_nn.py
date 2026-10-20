"""H79 dense b1 权重物理预排 v2=weight-swap(opt-in,默认关;H77 v1 的机制重做)。

背景(E297/E269/E255/E100/E320):
  c1 档 dense GEMV(MMV3 族)是最大单一设备域。E255×E269 交叉:三大形状引擎内
  有效带宽缺口 1.24-1.3x;E100/H77 微基准 NN 连续布局在 skinny-M 下更快。
  v1(H77)=实例级替换 quant_method.apply 为 python 闭包——E320 定谳:vllm 0.23
  默认 cudagraph_mode=FULL_AND_PIECEWISE,FULL graph decode 全程不经 python,
  shim 被整体旁路(forward=0),A/B 无效。
  v2(H79)=同一假设的零分支实现:PWAL 期把 weight.data 物理替换为
  ``w.t().contiguous().t()``(逻辑形状 [N,K] 不变、存储 [K,N] 连续,stride=(1,N)),
  vendor apply(torch.ops.vllm.unquantized_gemm→F.linear)按 stride 自动走 NN
  内核——零热路径分支、零 python 计数依赖,图 capture 前定形(地址稳定)。

E321 stride 敏感性微基准(h79_stride_bench,引擎精确链,HBM-cold):
  o_proj[2048×4096]:M=1 x1.169 / M=8 x1.143 / M=128 x1.160(全段 +14~17%)
  qkvz[12288×2048]:M=1 x0.919(M=1 段亏损,排除部署)
  attn_qkv[9216×2048]:M=1 x1.038 / M=128 x0.876(M=128 亏损,排除部署)
  ⇒ 部署白名单=oproj 单族(静态族粒度是图模式唯一安全面;M 级分支=v1 死路);
  A/B 以 ``TILEINFER_DENSE_NN=oproj`` 挂载。o_proj 收益估计 ~80μs/步(c1 ≈−0.9%)。

实现要点:
  - 钩子双类同挂(leaf ``AscendUnquantizedLinearMethod``+core
    ``UnquantizedLinearMethod``)且后置执行(orig 完成后再 swap)——leaf 末尾有
    ``layer.weight.data = maybe_trans_nz(layer.weight.data)``(bf16 mode=1 恒等,
    E43),swap 必须是最后写者;幂等 marker+stride 自愈复核兜底。
  - 幂等:层属性 ``_tileinfer_dense_nn_swapped`` 标记;与 H73/H28 的 PWAL 包装
    独立 marker,互不干扰。
  - 生效探针(图模式可见面):stats() 惰性复核已换层 weight.data 的
    shape/stride(1,N)——物理属性断言,不依赖任何 forward 计数。

红线核对:
  - 不动 MoE 收官结论(不触碰 GroupedMatmul 路径);
  - vendor 已最优(E255)的推翻走双侧证据:微基准门槛(E321 o_proj ≥1.05x)+
    权威口径引擎 A/B(判据 c1 TPOT ≥−1%、其余 |Δ|≤3%);
  - TP>1 时 RowParallel/Column 分片语义不改;安装期强制 world_size==1;
  - HBM 增量:oproj 单族 40×16MB≈0.64GB。

开关:
  ``TILEINFER_DENSE_NN``   未设/0=关;1=all(三族);qkvz,attnqkv,oproj 子集
                           (部署推荐 oproj,E321 证据白名单)
  ``TILEINFER_DENSE_NN_DRYRUN=1``  只记日志不实际替换(部署前自检)
"""

from __future__ import annotations

import logging
import os
import weakref

import torch

logger = logging.getLogger(__name__)

_INSTALLED = False
_REGISTERED: list[weakref.ref] = []  # 已换层的弱引用(探针面)
_STATS: dict[str, int] = {
    "dense_nn_modules": 0,    # 已物理换层的线性层数
    "dense_nn_stride_ok": 0,  # 探针:stride/shape 仍为 NN 视图的层数
    "dense_nn_stride_bad": 0, # 探针:被外部改动(如权重更新流)的层数
    "dense_nn_skipped": 0,    # 形状/条件不满足而跳过的次数
}

# (N, K) -> family;形状在本模型内唯一(qkvz/attn qkv 是仅有的两个 K=2048 大 N 族,
# o/out_proj 是仅有的 K=4096 族;router/ba/shared/lm_head 形状不同不会误命中)。
_FAMILY_BY_SHAPE: dict[tuple[int, int], str] = {
    (12288, 2048): "qkvz",
    (9216, 2048): "attnqkv",
    (2048, 4096): "oproj",
}

_VALID_FAMILIES = {"qkvz", "attnqkv", "oproj"}


def _enabled_families() -> set[str] | None:
    raw = (os.environ.get("TILEINFER_DENSE_NN") or "").strip()
    if raw in ("", "0", "false", "False"):
        return None
    if raw in ("1", "true", "True", "all"):
        return set(_VALID_FAMILIES)
    fams = {s.strip() for s in raw.split(",") if s.strip()}
    fams &= _VALID_FAMILIES
    return fams or None


def _dryrun() -> bool:
    return (os.environ.get("TILEINFER_DENSE_NN_DRYRUN") or "").strip() not in (
        "", "0", "false", "False")


def stats() -> dict[str, int]:
    """计数 + 物理探针(惰性复核 stride;1Hz dumper 下 80 次属性读=纳秒级)。"""
    ok = bad = 0
    for ref in _REGISTERED:
        layer = ref()
        if layer is None:
            continue
        w = layer.weight.data
        if w.ndim == 2 and tuple(w.shape) in _FAMILY_BY_SHAPE \
                and tuple(w.stride()) == (1, w.shape[0]):
            ok += 1
        else:
            bad += 1
    _STATS["dense_nn_stride_ok"] = ok
    _STATS["dense_nn_stride_bad"] = bad
    return dict(_STATS)


def is_installed() -> bool:
    return _INSTALLED


def _swap_layer(layer) -> bool:
    """对单个 Linear 层做 weight 物理换形(存储 NN+逻辑 NT);返回是否生效。

    幂等且自愈:已换层若 stride 仍为 (1,N) 直接跳过;若被后续权重处理改写
    (如 NZ 转换路径),重新换形——保证 swap 是 PWAL 全链的最后写者。
    """
    fams = _enabled_families()
    if not fams:
        return False
    w = getattr(layer, "weight", None)
    if w is None or w.data.ndim != 2:
        return False
    if getattr(layer, "_tileinfer_dense_nn_swapped", False):
        if tuple(w.data.stride()) == (1, w.data.shape[0]):
            return False  # 已换形且未被改写
        logger.warning("H79: swapped weight clobbered, re-swapping %s",
                       getattr(layer, "prefix", "?"))
    if w.data.dtype not in (torch.bfloat16, torch.float16):
        _STATS["dense_nn_skipped"] += 1
        return False
    fam = _FAMILY_BY_SHAPE.get(tuple(w.data.shape))
    if fam is None or fam not in fams:
        return False
    if getattr(layer, "bias", None) is not None:
        # 带 bias 的形态不做(现模型三族均 bias=False);保持零风险面。
        _STATS["dense_nn_skipped"] += 1
        return False
    if getattr(layer, "quant_method", None) is None:
        _STATS["dense_nn_skipped"] += 1
        return False
    if not _tp1():
        _STATS["dense_nn_skipped"] += 1
        return False
    if _dryrun():
        logger.info("H79 dryrun: would weight-swap %s (%s)", fam,
                    getattr(layer, "prefix", "?"))
        return False

    # 存 [K,N] 连续、逻辑 [N,K] 视图(stride=(1,N));vendor F.linear 走 NN 内核。
    w.data = w.data.t().contiguous().t()
    layer._tileinfer_dense_nn_swapped = True
    if not any(r() is layer for r in _REGISTERED):
        _REGISTERED.append(weakref.ref(layer))
        _STATS["dense_nn_modules"] += 1
    return True


# 兼容别名(v1 命名;E320 起语义=swap)
_shim_layer = _swap_layer


def _tp1() -> bool:
    try:
        from vllm.distributed import get_tensor_model_parallel_world_size

        return get_tensor_model_parallel_world_size() == 1
    except Exception:
        # 取不到 TP 上下文(如单测)按 TP1 处理;引擎内必然取得到。
        return True


def _wrap_pwal(cls) -> bool:
    """给指定 quant-method 类挂 PWAL 后置钩子;已包装返回 False。"""
    orig = cls.process_weights_after_loading
    if getattr(orig, "_tileinfer_dense_nn_wrapped", False):
        return False

    def process_weights_after_loading(self, layer, *args, **kwargs):
        result = orig(self, layer, *args, **kwargs)
        try:
            _swap_layer(layer)
        except Exception:  # pragma: no cover - 优化模块不得影响在役路径
            _STATS["dense_nn_skipped"] += 1
        return result

    process_weights_after_loading._tileinfer_dense_nn_wrapped = True
    # 留存包装前原函数,供测试/回滚恢复类属性(与 H73/H28 的 marker 语义独立)。
    process_weights_after_loading._tileinfer_dense_nn_orig = orig
    cls.process_weights_after_loading = process_weights_after_loading
    return True


def install() -> bool:
    """挂 PWAL 后置钩子(权重加载后、图 capture 前;swap=最后写者)。

    双类同挂:leaf ``AscendUnquantizedLinearMethod``(若可导入)与 core
    ``UnquantizedLinearMethod``——引擎在役 qm 类两链证据(E320)均兼容,双挂+
    幂等 marker 对任一实态都命中;leaf 链 mid-way 的 maybe_trans_nz 为 bf16
    恒等(E43 mode=1),后置自愈复核兜底。
    """
    global _INSTALLED
    if _INSTALLED:
        return True
    if _enabled_families() is None:
        return False
    try:
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    except Exception:
        logger.warning("H79 not installed: vllm linear unavailable")
        return False
    classes = [UnquantizedLinearMethod]
    try:
        from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

        if AscendUnquantizedLinearMethod not in classes:
            classes.append(AscendUnquantizedLinearMethod)
    except Exception:
        pass

    wrapped_any = False
    for cls in classes:
        try:
            wrapped_any |= _wrap_pwal(cls)
        except Exception:  # pragma: no cover - 优化模块不得影响在役路径
            logger.warning("H79 pwal wrap failed on %s", cls.__name__)
    if not wrapped_any and not any(
            getattr(c.process_weights_after_loading,
                    "_tileinfer_dense_nn_wrapped", False) for c in classes):
        return False
    _INSTALLED = True
    logger.info(
        "TileInfer dense NN weight-swap installed (H79, TILEINFER_DENSE_NN; "
        "cls=%s; dryrun=%s)",
        "+".join(c.__name__ for c in classes), _dryrun(),
    )
    return True
