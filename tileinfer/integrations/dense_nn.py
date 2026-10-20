"""H77 dense b1 大形状权重物理预排 NT→NN(opt-in,默认关)。

背景(E297/E269/E255/E100):
  c1 档 dense GEMV(MMV3 族)是最大单一设备域(3.532ms/步,37.5%)。E255×E269 交叉:
  qkvz[2048→12288]/attn_qkv[2048→9216]/o_proj[4096→2048] 三大形状引擎内有效带宽
  1.30/1.28/0.94TB/s,而同形状微基准流式 1.62-1.65TB/s——缺口 1.24-1.3x,聚合
  ~450μs/步(≈c1 设备链 4.8%)。换后端(E255 matmul 视图 0.877-1.047 全败)与
  NZ 布局(E42/E100 零收益;weight_nz_mode=2 引擎启动 abort)均已排除;
  E100 M=16 微基准显示 **ND 物理连续 NN 布局**(w_nn=[K,N],forward=x@w_nn)
  在 skinny-M 下比现役 NT 形态快 ~11%,且 M=1 从未测过——本模块把该轴搬进引擎。

引擎事实(源码级):在役权重实态=ND [N,K](weight_nz_mode 默认=1,bf16 不转 NZ,
  E43:mode=2 启动 abort);unquantized_gemm 本体=F.linear 包装
  (vllm_ascend/ops/linear.py)。预排=加载期 PWAL 钩子内做一次
  ``w_nn = weight.t().contiguous()`` 存层属性,实例级替换 quant_method.apply:
  ``x.shape[0] <= TILEINFER_DENSE_NN_MAXM(默认 16)`` 走 ``torch.matmul(x, w_nn)``,
  否则回落原 apply(prefill 大 M 保持 compute-optimal NT 路径)。

红线核对:
  - 不动 MoE 收官结论(不触碰 GroupedMatmul 路径);
  - vendor 已最优(E255)的推翻走双侧证据:本模块仅在微基准门槛
    (h77_nn_bench.py 任一大形状 M=1 ≥1.05x 且 allclose)通过后接入引擎 A/B;
  - TP>1 时 RowParallel 的 apply 含 all-reduce、Column 的 N 已分片,matmul 直替
    会破坏通信/分片语义 ⇒ 安装期强制 world_size==1,否则整族跳过。

与 H28/H73 的关系:三族形状 (12288,2048)/(9216,2048)/(2048,4096) 与 H28 折叠族
  shared gate_up/down([1024,2048]/[2048,512])及 H73 融合 gate_up(2I+1)零交集;
  H73 与本模块共用的 PWAL 钩子各自独立包装(独立 marker),互不干扰。

开关:
  ``TILEINFER_DENSE_NN``            未设/0=关;1=all;qkvz,attnqkv,oproj 子集
  ``TILEINFER_DENSE_NN_MAXM``       NN 路径的 M 上限(默认 16;E100 证据域)
  ``TILEINFER_DENSE_NN_DRYRUN=1``   只记日志不实际替换(部署前自检)
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

_INSTALLED = False
_STATS: dict[str, int] = {
    "dense_nn_modules": 0,   # 已 shim 的线性层数
    "dense_nn_forward": 0,   # 走 NN matmul 路径的次数
    "dense_nn_fallback": 0,  # 超 MAXM 回落原 apply 的次数
    "dense_nn_skipped": 0,   # 形状/条件不满足而跳过的次数
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


def _max_m() -> int:
    try:
        return max(1, int(os.environ.get("TILEINFER_DENSE_NN_MAXM", "16")))
    except ValueError:
        return 16


def stats() -> dict[str, int]:
    return dict(_STATS)


def is_installed() -> bool:
    return _INSTALLED


def _shim_layer(layer) -> bool:
    """对单个 Linear 层做 NN 预排 + apply 实例替换;返回是否生效。"""
    fams = _enabled_families()
    if not fams:
        return False
    w = getattr(layer, "weight", None)
    if w is None or w.data.ndim != 2 or w.data.dtype not in (torch.bfloat16, torch.float16):
        _STATS["dense_nn_skipped"] += 1
        return False
    fam = _FAMILY_BY_SHAPE.get(tuple(w.data.shape))
    if fam is None or fam not in fams:
        return False
    if getattr(layer, "bias", None) is not None:
        # 带 bias 的形态不做(现模型三族均 bias=False);保持零风险面。
        _STATS["dense_nn_skipped"] += 1
        return False
    qm = getattr(layer, "quant_method", None)
    if qm is None or getattr(qm, "_tileinfer_dense_nn", False):
        return False
    if not _tp1():
        _STATS["dense_nn_skipped"] += 1
        return False
    if _dryrun():
        logger.info("H77 dryrun: would preformat %s (%s)", fam,
                    getattr(layer, "prefix", "?"))
        return False

    w_nn = w.data.t().contiguous()
    orig_apply = qm.apply
    maxm = _max_m()

    def apply(layer_arg, x, bias=None, _w=w_nn, _orig=orig_apply, _maxm=maxm):
        if x.shape[0] <= _maxm:
            _STATS["dense_nn_forward"] += 1
            return torch.matmul(x, _w)
        _STATS["dense_nn_fallback"] += 1
        return _orig(layer_arg, x, bias)

    qm.apply = apply
    qm._tileinfer_dense_nn = True
    _STATS["dense_nn_modules"] += 1
    return True


def _tp1() -> bool:
    try:
        from vllm.distributed import get_tensor_model_parallel_world_size

        return get_tensor_model_parallel_world_size() == 1
    except Exception:
        # 取不到 TP 上下文(如单测)按 TP1 处理;引擎内必然取得到。
        return True


def install() -> bool:
    """挂 PWAL 基类钩子(权重加载后、NZ no-op 前后均可,ND 实态下与顺序无关)。"""
    global _INSTALLED
    if _INSTALLED:
        return True
    if _enabled_families() is None:
        return False
    try:
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    except Exception:
        logger.warning("H77 not installed: vllm linear unavailable")
        return False

    orig = UnquantizedLinearMethod.process_weights_after_loading
    if getattr(orig, "_tileinfer_dense_nn_wrapped", False):
        _INSTALLED = True
        return True

    def process_weights_after_loading(self, layer, *args, **kwargs):
        try:
            _shim_layer(layer)
        except Exception:  # pragma: no cover - 优化模块不得影响在役路径
            _STATS["dense_nn_skipped"] += 1
        return orig(self, layer, *args, **kwargs)

    process_weights_after_loading._tileinfer_dense_nn_wrapped = True
    # 留存包装前原函数,供测试/回滚恢复类属性(与 H73 的 marker 语义独立)。
    process_weights_after_loading._tileinfer_dense_nn_orig = orig
    UnquantizedLinearMethod.process_weights_after_loading = process_weights_after_loading
    _INSTALLED = True
    logger.info(
        "TileInfer dense NN preformat installed (H77, TILEINFER_DENSE_NN; dryrun=%s)",
        _dryrun(),
    )
    return True
