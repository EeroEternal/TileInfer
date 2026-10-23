"""MoE 权重 prefetch 阈值对齐(H37/S-MoE 域;依据 E105/E109 #19033 第 2 项)。

已装 vllm-ascend 0.23.0 自带完整 MoE 权重预取管道(E109 管道核查):

* worker model_runner init 时按 ``additional_config.weight_prefetch_config`` 建立
  ``WeightPrefetchMethod``(默认 ``enabled=False``,管道整体失活);
* ``experts_selector.select_experts`` 前置 hook 预取**上一层** ``experts.w13_weight``
  入 L2(独立 prefetch 流,``torch_npu.npu_prefetch``),``moe_mlp`` 后置流同步;
* 门槛 ``MOE_PREFETCH_TOKEN_THRESHOLD = 96`` 把 decode(M<=32)全部排除。

本模块(TILEINFER_OPS=moeprefetch opt-in)做三件事:

1. 阈值对齐 96→1(模块常量运行期 patch;上游 #19033 同款改动,commit 05187a0,
   c=1 TPOT 约 -2.4%,FULL_DECODE_ONLY 图模式实证可跑,E109);
2. 管道武装兜底:plugin 安装期早于 worker init,``_WEIGHT_PREFETCH_METHOD`` 尚为
   None → 安装惰性替位对象(首次 hook 调用时按 ``get_ascend_config()``
   weight_prefetch_config 优先物化,否则自建 enabled 配置;attn 臂恒置零保单变量),
   runner 的 ``set_weight_prefetch_method`` 因非 None 守卫被跳过,武装权归本模块;
   若安装时 method 已存在(worker init 先行)则直接就地武装;
3. 计数器 ``moeprefetch_active``/``moeprefetch_fallbacks`` 供在役探针断言。

物化失败(异常)时替位对象按 falsy 处理,全部 hook 回到 stock 失活行为,引擎不炸。

风险双录(E109):上游 main 已于 e77502e 移除该功能(仅 0.23.0 分支自持);
950PR 小 M GEMM 权重 L2-bypass(DeepGEMM-Ascend #11)可能架空 L2 预取假设——
均以权威口径引擎 A/B 实测裁决,不以上游数字外推。

阈值 patch 与武装均为安装期语义(与 gdnproj 权重融合一致):翻转 TILEINFER_OPS
需重启生效;计数器随安装进程存在。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

_logger = logging.getLogger("tileinfer.moe.prefetch")

#: 探针计数器(安装时同步进后端类 _stats,metrics dumper 读取)
COUNTERS: dict[str, int] = {
    "moeprefetch_active": 0,
    "moeprefetch_fallbacks": 0,
}

_stats_sink: Callable[[str], None] | None = None

#: 安装幂等标记(模块级,测试可复位)
_installed = False

#: 默认参数:#19033 终态阈值与管道默认比例(E109)
DEFAULT_TOKEN_THRESHOLD = 1
DEFAULT_GATE_UP_RATIO = 0.8


def _inc(name: str) -> None:
    COUNTERS[name] = COUNTERS.get(name, 0) + 1
    if _stats_sink is not None:
        _stats_sink(name)


def _fallback_cfg(gate_up_ratio: float) -> dict[str, Any]:
    """自建武装配置:enabled + moe gate_up 比例;attn 臂恒置零(单变量纪律)。"""
    return {
        "enabled": True,
        "prefetch_ratio": {
            "attn": {"qkv": 0.0, "o": 0.0},
            "moe": {"gate_up": gate_up_ratio},
        },
    }


def _resolve_cfg_dict(gate_up_ratio: float) -> dict[str, Any]:
    """配置真值优先(ADDCF enabled),否则回退比例配置(enabled=True)。"""
    try:
        from vllm_ascend.ascend_config import get_ascend_config

        addcfg = get_ascend_config().weight_prefetch_config
        if getattr(addcfg, "enabled", False):
            return {
                "enabled": True,
                "prefetch_ratio": dict(getattr(addcfg, "prefetch_ratio", {}) or {}),
            }
    except Exception:  # noqa: BLE001 - ascend config 未初始化属常态(安装期早)
        pass
    return _fallback_cfg(gate_up_ratio)


def _construct_armed(gate_up_ratio: float) -> Any:
    """构造并武装真实 ``WeightPrefetchMethod``(E409 H37 修复核心)。

    ``WeightPrefetchMethod.__init__`` 断言 ``is_moe_model(get_current_vllm_config())``
    ——必须在 ``set_current_vllm_config`` 作用域内构造(worker_base.init_device:
    321-323 包裹 runner 构造,model_runner_v1:338 在其中)。当前进程无 ctx 时按
    原样构造(旧行为;AssertionError 交由调用方 dead-path 兜底)。构造成功即
    就地武装(比例+计数包装),返回真实 method。
    """
    from vllm.config import get_current_vllm_config_or_none, set_current_vllm_config
    from vllm_ascend.ascend_config import WeightPrefetchConfig
    from vllm_ascend.ops.weight_prefetch import WeightPrefetchMethod

    cfg_dict = _resolve_cfg_dict(gate_up_ratio)
    ctx = get_current_vllm_config_or_none()
    if ctx is not None:
        with set_current_vllm_config(ctx):
            real = WeightPrefetchMethod(WeightPrefetchConfig(cfg_dict))
    else:
        real = WeightPrefetchMethod(WeightPrefetchConfig(cfg_dict))
    return _arm(real, gate_up_ratio)


def _install_runner_binding(gate_up_ratio: float) -> bool:
    """接管 runner init 的 ``set_weight_prefetch_method``(E409 H37 修复)。

    惰性替位路径的物化时点(model build 后首次 hook 触碰)在
    ``set_current_vllm_config`` 作用域之外 ⇒ 构造断言炸 ⇒ 管道永久失活
    (E409 探针实证:moeprefetch_fallbacks=1+日志 AssertionError)。本绑定把
    构造交还 runner init 的 ctx 时点(worker_base.py:321→worker.py:518→
    model_runner_v1:338),构造后就地武装;runner 侧 V1 绑定被替换,惰性对象
    不再被触碰。import 失败(CPU 测试/非昇腾)返回 False,维持惰性兜底。
    """
    try:
        from vllm_ascend import utils as vutils
        from vllm_ascend.worker import model_runner_v1 as mrv1
    except Exception:  # noqa: BLE001 - CPU 测试/非昇腾环境
        return False
    orig = mrv1.__dict__.get("set_weight_prefetch_method")
    if orig is None:
        return False
    if getattr(orig, "_tileinfer_force_construct", False):
        return True

    def _set_and_arm(weight_prefetch_config: Any) -> Any:
        # 构造权自持(ctx 内构造+enabled 真值优先),槽位落真实武装 method;
        # 不调 _orig——其槽位非 None 判据只会跳过构造,且 stock cfg(enabled
        # 随 ADDCF)与我们 enabled=True 的武装语义冲突。
        vutils._WEIGHT_PREFETCH_METHOD = _construct_armed(gate_up_ratio)
        return vutils._WEIGHT_PREFETCH_METHOD

    _set_and_arm._tileinfer_force_construct = True  # type: ignore[attr-defined]
    mrv1.set_weight_prefetch_method = _set_and_arm
    return True


def _arm(real: Any, gate_up_ratio: float) -> Any:
    """就地武装真实 method:moe 臂强制使能 + 比例设置 + MoE hook 计数包装。

    attn 臂置零仅当比例为空(ADDCF 已显式置零则尊重其值);mlp 臂对 MoE 模型
    天然失活(enable=enabled and not is_moe),不动。
    """
    real.moe.enable = True
    if not real.moe.prefetch_ratio.get("gate_up"):
        real.moe.prefetch_ratio = {"gate_up": gate_up_ratio}
    if not real.attn.prefetch_ratio or any(v > 0 for v in real.attn.prefetch_ratio.values()):
        # 单变量纪律:除非 ADDCFG 显式置零,否则强制 attn 臂失活(我栈 bf16 GDN
        # 本无 attn hook 调用点,此处防御 W8A8 量化路径意外引入第二变量)
        real.attn.enable = False

    orig_pre = real.maybe_prefetch_moe_weight_preprocess
    orig_post = real.maybe_prefetch_moe_weight_postprocess

    def _pre(hidden_states: Any, prefix: str) -> None:
        try:
            orig_pre(hidden_states, prefix)
            if getattr(real.moe, "is_active_this_forward", False):
                _inc("moeprefetch_active")
        except Exception:  # noqa: BLE001 - 计数后原样上抛,不改变 stock 失败语义
            _inc("moeprefetch_fallbacks")
            raise

    def _post(stop_flag: Any) -> None:
        try:
            orig_post(stop_flag)
        except Exception:  # noqa: BLE001
            _inc("moeprefetch_fallbacks")
            raise

    real.maybe_prefetch_moe_weight_preprocess = _pre
    real.maybe_prefetch_moe_weight_postprocess = _post
    return real


class _LazyPrefetchMethod:
    """``vllm_ascend.utils._WEIGHT_PREFETCH_METHOD`` 的惰性替位对象。

    首次任一访问时物化真实 ``WeightPrefetchMethod``:配置优先取
    ``get_ascend_config().weight_prefetch_config``(ADDCF 真值,enabled 时),
    否则回退 :func:`_fallback_cfg`;物化成功即就地武装(计数包装)。物化失败
    则本对象永久 falsy——全部 hook 短路,行为等于 stock 失活管道。
    """

    def __init__(self, gate_up_ratio: float) -> None:
        self._gate_up_ratio = gate_up_ratio
        self._real: Any = None
        self._dead = False

    def _materialize(self) -> Any:
        if self._real is not None:
            return self._real
        if self._dead:
            return None
        try:
            real = _construct_armed(self._gate_up_ratio)
            self._real = real
            return real
        except Exception as exc:  # noqa: BLE001 - 物化失败=管道失活,不破引擎
            # 首次失活一次性告警(H37 A/B 归因:E402 fallbacks=1 零 active 的静默盲区)
            _logger.warning_once(
                "moeprefetch pipeline materialization failed, prefetch permanently "
                "inactive this process: %s: %s",
                type(exc).__name__,
                exc,
            )
            self._dead = True
            _inc("moeprefetch_fallbacks")
            return None

    def __bool__(self) -> bool:
        return self._materialize() is not None

    def __getattr__(self, name: str) -> Any:
        # 仅未显式定义的属性走委托(_real/_dead/_gate_up_ratio 走实例字典)
        real = self._materialize()
        if real is None:
            raise AttributeError(
                f"weight prefetch pipeline unavailable (moeprefetch materialization failed); "
                f"attribute {name!r} requested"
            )
        return getattr(real, name)


def install_moe_weight_prefetch(
    token_threshold: int = DEFAULT_TOKEN_THRESHOLD,
    gate_up_ratio: float = DEFAULT_GATE_UP_RATIO,
) -> bool:
    """武装 MoE 权重 prefetch(H37,``TILEINFER_OPS=moeprefetch``)。

    返回 True 表示安装完成(幂等);阈值 patch 恒生效,管道武装视时序落位:
    worker init 先行→就地武装既有 method;否则安装惰性替位对象接管 runner 的
    ``set_weight_prefetch_method``(非 None 守卫使其跳过)。任一 import 失败
    (无 vllm-ascend 环境)返回 False 且不留半装状态。
    """
    global _installed
    if _installed:
        return True
    from ...integrations.vllm_ascend import ops_enabled

    if not ops_enabled("moeprefetch"):
        return False
    try:
        from vllm_ascend import utils as vutils
        from vllm_ascend.ops import weight_prefetch as wp
    except Exception:  # noqa: BLE001 - 无 vllm-ascend 环境(CPU 测试/非昇腾)
        return False

    if getattr(vutils, "_WEIGHT_PREFETCH_METHOD", None) is None:
        vutils._WEIGHT_PREFETCH_METHOD = _LazyPrefetchMethod(gate_up_ratio)
    else:
        _arm(vutils._WEIGHT_PREFETCH_METHOD, gate_up_ratio)
    # E409 修复:优先把构造交还 runner init 的 ctx 时点(worker_base 包裹),
    # 惰性替位降级为兜底(绑定接管失败=测试/CPU 环境才走)。
    _install_runner_binding(gate_up_ratio)
    wp.MOE_PREFETCH_TOKEN_THRESHOLD = int(token_threshold)
    _installed = True
    return True
