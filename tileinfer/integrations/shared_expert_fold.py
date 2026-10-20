"""H28 shared expert 恒选 E+1 折叠(opt-in):把 shared expert 并进 routed grouped 路径。

背景(E79/E91 画像 + E300 vendor 审查,iter-210):
  在役(multistream 被 vendor ``platform.py:32`` 硬编码关闭)下每层 shared expert
  走 ``_shared_experts_part1``(gate_up GEMM)→ ``_shared_experts_part2``(act+down
  +gate)串行于主流 ≈ 1.2ms/步(80 GEMM/步,dense 21% 份额的主体)。

机制(AITER 式恒选 E+1):
  ① 权重拼接:shared 的 gate_up/down 权重并进 routed ``w13_weight``/``w2_weight``
    作第 E 号专家(0-based;必须在 ``process_weights_after_loading`` 的
    pad/transpose/NZ 之前完成);
  ② 路由扩展:``select_experts`` 之后为每 token 追加专家 E,scale = sigmoid(gate_logit)
    ——门控语义与原 ``sigmoid(expert_gate) * down(...)`` 逐点等价,且不受 routed
    renormalize 污染(renorm 在 select_experts 内部完成);
  ③ gate_logit 免费:router gate 权重扩 1 行,internal-router 路径一次 GEMM 同时
    产出 [M,E] logits 与 [M,1] gate_logit,运行时切片;
  ④ 原 shared 串行链整体 bypass(``_forward_shared_experts`` 返回零张量,合加语义不变)。

开关:
  ``TILEINFER_SHARED_EXPERT_FOLD=1``   启用(默认关 = 零行为变化)
  ``TILEINFER_SHARED_EXPERT_FOLD_DRYRUN=1``  只记日志不改权重

实现红线(E199/E200):
  fused 专家计数语义以 ``w13_weight``/``w2_weight`` 的第 0 维为准;折叠后
  E_eff = E+1 只存在于权重与 topk 列,routed 选择语义(num_experts、expert_map、
  capturer、zero_expert)全部保持 E,不得出现双重拆分。

残留待审(E300-5,设备期验证):
  - ``w13_weight_list`` 条件分支(per-expert unbind clone)若激活须同步扩 list;
  - 非 internal router 模型走 GEMV 降级路径;
  - eplb/tid2eid 在 TP=1 下恒等映射确认。
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

_INSTALLED = False
_STATS: dict[str, int] = {
    "fold_layers": 0,  # 完成权重折叠的 MoE 层数
    "fold_forward": 0,  # 走折叠路径的 forward 次数
    "fold_skipped": 0,  # 条件不满足回退原路径次数
}

_CTX = threading.local()


def _enabled() -> bool:
    return os.environ.get("TILEINFER_SHARED_EXPERT_FOLD", "").strip() not in (
        "",
        "0",
        "false",
        "False",
    )


def _dryrun() -> bool:
    return os.environ.get("TILEINFER_SHARED_EXPERT_FOLD_DRYRUN", "").strip() not in (
        "",
        "0",
        "false",
        "False",
    )


def stats() -> dict[str, int]:
    return dict(_STATS)


def is_installed() -> bool:
    return _INSTALLED


# ---------------------------------------------------------------- 纯函数(契约可测)


def slice_router_logits(router_logits, num_experts: int):
    """internal-router 扩列后 [M, E+1] → (routed 用 [M, E], gate_logit [M, 1])。"""
    return router_logits[:, :num_experts], router_logits[:, num_experts:]


def append_shared_column(topk_ids, topk_weights, expert_id: int, gate_logit, dtype):
    """topk 追加恒选列:ids 末列=expert_id,weights 末列=sigmoid(gate_logit)。"""
    import torch

    m = topk_ids.shape[0]
    ids_col = torch.full((m, 1), expert_id, dtype=topk_ids.dtype, device=topk_ids.device)
    w_col = torch.sigmoid(gate_logit.to(dtype))
    return torch.cat([topk_ids, ids_col], dim=1), torch.cat([topk_weights, w_col], dim=1)


def concat_shared_weights(w13, w2, shared_w13, shared_w2):
    """第 0 维(专家维)拼接;调用方保证发生在 pad/NZ 之前(2-D 批量权重)。"""
    if w13.dim() != 3 or w2.dim() != 3:
        raise ValueError(f"expect 3-D stacked expert weights, got {w13.shape}/{w2.shape}")
    if shared_w13.shape[1:] != w13.shape[1:] or shared_w2.shape[1:] != w2.shape[1:]:
        raise ValueError(
            f"shared expert shape mismatch: w13 {shared_w13.shape} vs {w13.shape}, "
            f"w2 {shared_w2.shape} vs {w2.shape}"
        )
    import torch

    return torch.cat([w13, shared_w13], dim=0), torch.cat([w2, shared_w2], dim=0)


def extend_gate_weight(gate_weight, shared_gate_row):
    """router gate 权重扩 1 行(gate_logit 的 free col);fp32 连续新张量。"""
    import torch

    if gate_weight.dim() != 2:
        raise ValueError(f"expect 2-D gate weight (NZ 前), got {tuple(gate_weight.shape)}")
    row = shared_gate_row.reshape(1, -1).to(dtype=gate_weight.dtype)
    return torch.cat([gate_weight, row], dim=0).contiguous()


# ---------------------------------------------------------------- 接线(install)


_OWNER_REGISTRY: dict[int, object] = {}


def _wrap_moe_init(orig_init):
    """登记 AscendFusedMoE 实例与其权重载体的从属关系(权重拼接期查表)。

    0.23.0 在役结构(E309 审查):AscendFusedMoE(FusedMoE 子类)本体即持有
    w13_weight/w2_weight/quant_method/_shared_experts,pwals 以 layer=层本体调用;
    非 0.23.0 分支则经 routed_experts 子模块。两键都登记,查表按 pwals 实参命中。
    """

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            if getattr(self, "_shared_experts", None) is None:
                return
            _OWNER_REGISTRY[id(self)] = self
            routed = getattr(self, "routed_experts", None)
            if routed is not None:
                _OWNER_REGISTRY[id(routed)] = self
        except Exception:  # pragma: no cover
            pass

    return __init__


def _weight_holder(owner):
    """0.23.0:层本体持权重;防御式回落 routed_experts 子模块。"""
    if getattr(owner, "w13_weight", None) is not None:
        return owner
    routed = getattr(owner, "routed_experts", None)
    if routed is not None and getattr(routed, "w13_weight", None) is not None:
        return routed
    return None


_DISPATCHER_BUMPED = {"e": None, "k": None}


def _bump_dispatcher_top_k(owner_k: int) -> int:
    """折叠后 topk 列 k→k+1:npu_moe_init_routing 的 ``active_num=num_tokens×td.top_k``
    (token_dispatcher.py:405)按构造期 k 取值 ⇒ ids [M,k+1] 与 active_num M×k 失配
    (EZ0002:16384 vs 18432,iter-213 on 臂 attempt3 实证)。对已构造单例补升
    top_k k→k+1;层本体 self.top_k 保持 k(select_experts 仍选 k 个真实专家)。

    单例跨层共享 ⇒ 全局一次性幂等。前置=H28 全层折叠:若存在未折叠层,其 ids [M,k]
    与补升后 active_num M×(k+1) 失配 ⇒ 半开不可用(fail-closed)。
    """
    if _DISPATCHER_BUMPED["k"] == owner_k:
        return 1
    try:
        from vllm_ascend.ops.fused_moe import moe_comm_method as mcm

        bumped = 0
        for impl in list(getattr(mcm, "_MoECommMethods", {}).values()):
            td = getattr(impl, "token_dispatcher", None)
            if td is None:
                continue
            cur = getattr(td, "top_k", None)
            if isinstance(cur, int) and cur == owner_k:
                td.top_k = owner_k + 1
                bumped += 1
        if bumped:
            _DISPATCHER_BUMPED["k"] = owner_k
            print(f"[H28] dispatcher top_k bumped {owner_k}->{owner_k + 1} "
                  f"on {bumped} singleton(s)", flush=True)
        return bumped
    except Exception as exc:  # pragma: no cover - 优化不得影响在役路径
        print(f"[H28] dispatcher top_k bump failed: {exc}", flush=True)
        return 0


def _bump_dispatcher_expert_num(owner_e: int) -> int:
    """折叠后 id=E 越界:npu_moe_init_routing(E309 实证)会把越界 token 丢弃
    (counts sum < rows ⇒ GMM 错乱)。comm 方法单例在层 __init__ 期构造(expert_num
    按值捕获),PWAL 折叠晚于它 ⇒ 对已构造单例补升 expert_num E→E+1。

    单例跨层共享 ⇒ 全局一次性幂等(首层补升后其余层 no-op 即成功)。
    """
    if _DISPATCHER_BUMPED["e"] == owner_e:
        return 1
    try:
        from vllm_ascend.ops.fused_moe import moe_comm_method as mcm

        bumped = 0
        for impl in list(getattr(mcm, "_MoECommMethods", {}).values()):
            td = getattr(impl, "token_dispatcher", None)
            if td is None:
                continue
            cur = getattr(td, "num_experts_local", None)
            if isinstance(cur, int) and cur == owner_e:
                td.num_experts_local = owner_e + 1
                bumped += 1
        if bumped:
            _DISPATCHER_BUMPED["e"] = owner_e
            print(f"[H28] dispatcher expert_num bumped {owner_e}->{owner_e + 1} "
                  f"on {bumped} singleton(s)", flush=True)
        return bumped
    except Exception as exc:  # pragma: no cover - 优化不得影响在役路径
        print(f"[H28] dispatcher bump failed: {exc}", flush=True)
        return 0


def _fold_weights(owner) -> bool:
    """load 后、pad/NZ 前把 shared 权重拼进 routed 专家维 + gate 扩列。返回是否成功。"""
    import torch

    mlp = owner._shared_experts
    holder = _weight_holder(owner)
    if holder is None:
        _STATS["fold_skipped"] += 1
        return False
    w13 = holder.w13_weight.data
    w2 = holder.w2_weight.data
    if w13.dim() != 3 or w13.shape[0] < 1:
        _STATS["fold_skipped"] += 1
        return False
    # H73 融合形态下 gate_up 权重末行是 shared_gate,须剔除后再拼。
    shared_w13 = mlp.gate_up_proj.weight.data
    if getattr(mlp, "_tileinfer_gate_fused", False):
        if shared_w13.shape[0] < 2:
            _STATS["fold_skipped"] += 1
            return False
        shared_w13 = shared_w13[:-1, :]
    shared_w2 = mlp.down_proj.weight.data
    try:
        new_w13, new_w2 = concat_shared_weights(
            w13, w2, shared_w13.unsqueeze(0), shared_w2.unsqueeze(0)
        )
    except ValueError:
        _STATS["fold_skipped"] += 1
        return False
    holder.w13_weight = torch.nn.Parameter(new_w13, requires_grad=False)
    holder.w2_weight = torch.nn.Parameter(new_w2, requires_grad=False)
    owner._tileinfer_fold_expert_id = w13.shape[0]  # 0-based,原 E 即新末位
    bumped = _bump_dispatcher_expert_num(int(w13.shape[0]))
    if bumped == 0:
        # 路由计数未补升 ⇒ 越界 id 必被 init_routing 丢弃(E309),折叠无效即回退
        _STATS["fold_skipped"] += 1
        return False
    owner_k = int(getattr(owner, "top_k", 0) or 0)
    if owner_k <= 0 or _bump_dispatcher_top_k(owner_k) == 0:
        # top_k 未补升 ⇒ active_num=num_tokens×k 与 ids [M,k+1] 失配
        # (EZ0002 iter-213 实证),折叠无效即回退
        _STATS["fold_skipped"] += 1
        return False
    _STATS["fold_layers"] += 1
    print(f"[H28] folded shared expert into layer id={id(owner):#x} "
          f"E+1={w13.shape[0] + 1} (E300 注入面;层内共享单例补升幂等)", flush=True)
    return True


def _wrap_quant_pwals(orig_pwals):
    """quant_method.process_weights_after_loading 前置折叠(pad/transpose/NZ 之前)。"""

    def process_weights_after_loading(self, layer, *args, **kwargs):
        owner = _OWNER_REGISTRY.get(id(layer))
        if owner is not None and _enabled() and not getattr(owner, "_tileinfer_fold_done", False):
            if _dryrun():
                logger.info("H28 dryrun: would fold shared expert into routed experts")
                owner._tileinfer_fold_done = True
            elif _fold_weights(owner):
                owner._tileinfer_fold_done = True
        return orig_pwals(self, layer, *args, **kwargs)

    return process_weights_after_loading


def _set_ctx(owner) -> None:
    _CTX.owner = owner


def _clear_ctx() -> None:
    _CTX.owner = None


def _ensure_gate_extended(owner) -> None:
    """首个 forward(warmup,先于图 capture)把 router gate 权重扩 1 列。

    扩列必须在 capture 前的 warmup 完成且此后张量对象不再更换;非 internal
    router(gate 无 weight_fp32)保持未扩,select 包装走 GEMV 降级。
    """
    if getattr(owner, "_tileinfer_fold_ext", False):
        return
    try:
        import torch

        gate = owner.gate
        w = getattr(gate, "weight_fp32", None)
        if w is None or w.dim() != 2:
            print(f"[H28] gate extension skipped: weight_fp32={type(w)} dim="
                  f"{getattr(w, 'dim', lambda: -1)()}", flush=True)
            return
        row = owner._shared_experts.expert_gate.weight.data.reshape(1, -1).to(dtype=w.dtype)
        new_w = torch.cat([w.data, row], dim=0).contiguous()
        gate.weight_fp32 = torch.nn.Parameter(new_w, requires_grad=False)
        owner._tileinfer_fold_ext = True
        print(f"[H28] gate extended {tuple(w.shape)}->{tuple(new_w.shape)}", flush=True)
    except Exception:  # pragma: no cover - 降级为 GEMV 路径
        logger.warning("H28: gate weight extension failed; gate_logit falls back to GEMV")


def _wrap_shared_forward(orig_shared_forward):
    """shared_forward_impl 包装:置上下文(select_experts 扩列用),bypass 由 part2 补齐。"""

    def shared_forward_impl(self, hidden_states, router_logits):
        if not (_enabled() and getattr(self, "_tileinfer_fold_done", False)):
            return orig_shared_forward(self, hidden_states, router_logits)
        if not getattr(self, "_tileinfer_ctx_seen", False):
            self._tileinfer_ctx_seen = True
            print(f"[H28] shared_forward_impl ctx armed on folded layer "
                  f"id={id(self):#x} (forward 链触达)", flush=True)
        _ensure_gate_extended(self)
        _set_ctx(self)
        try:
            return orig_shared_forward(self, hidden_states, router_logits)
        finally:
            _clear_ctx()

    return shared_forward_impl


def _wrap_select_experts(orig_select_experts):
    """select_experts 包装:折叠层上追加恒选专家与 sigmoid 门控。

    双形态(E310 定谳):
    - internal router(gate.weight_fp32 扩列):logits [M, E+1] → 切片 [M, E] + 免费门控列;
    - 非 internal(Qwen3.6 在役:router_logits 由模型外部 gate 产出,无扩列):
      logits [M, E] 原样透传,门控经 shared expert 的 expert_gate GEMV 现算
      (原语义 ``sigmoid(expert_gate(h)) * shared_out``,0_23_0 L355-357)。
    """

    _DIAG = {"entry": 0, "pass": 0}

    def select_experts(*args, **kwargs):
        owner = getattr(_CTX, "owner", None)
        logits = kwargs.get("router_logits")
        if owner is None or logits is None or logits.ndim != 2:
            if owner is not None and _DIAG["entry"] < 2:
                _DIAG["entry"] += 1
                print(f"[H28] select passthrough(no logits): owner={owner is not None} "
                      f"logits={type(logits)} args={len(args)} "
                      f"kw={sorted(kwargs)[:6]}", flush=True)
            return orig_select_experts(*args, **kwargs)
        num_experts = kwargs.get("num_experts")
        n_cols = int(logits.shape[1])
        eid = getattr(owner, "_tileinfer_fold_expert_id", None)
        extended = eid is not None and n_cols == eid + 1
        plain = eid is not None and n_cols == eid
        if not (extended or plain):
            if _DIAG["pass"] < 3:
                _DIAG["pass"] += 1
                print(f"[H28] select passthrough(shape): logits{tuple(logits.shape)} "
                      f"num_experts={num_experts}", flush=True)
            return orig_select_experts(*args, **kwargs)
        try:
            import torch

            kwargs = dict(kwargs)
            kwargs["router_logits"] = logits[:, :num_experts] if extended else logits
            topk_weights, topk_ids = orig_select_experts(*args, **kwargs)
            dtype = topk_weights.dtype
            gate_logit = logits[:, num_experts:] if extended else None
            if not getattr(owner, "_tileinfer_fold_ext", False):
                # 扩列 gate 权重的拼接在首个 forward 前完成;未扩列则 GEMV 降级。
                gate_logit = None
            if gate_logit is None:
                mlp = owner._shared_experts
                gate_mod = getattr(mlp, "expert_gate", None)
                if gate_mod is None:
                    raise ValueError("no gate source")
                hidden = args[0] if args else kwargs["hidden_states"]
                gate_logit = gate_mod(hidden)[0]
            ids, ws = append_shared_column(
                topk_ids, topk_weights, owner._tileinfer_fold_expert_id, gate_logit, dtype
            )
            if _STATS["fold_forward"] == 0:
                print(f"[H28] first append: logits{tuple(logits.shape)} "
                      f"ids{tuple(ids.shape)} expert_id={owner._tileinfer_fold_expert_id} "
                      f"extended={extended}", flush=True)
            _STATS["fold_forward"] += 1
            return ws, ids
        except Exception:  # pragma: no cover - 优化不得影响在役路径
            _STATS["fold_skipped"] += 1
            return orig_select_experts(*args, **kwargs)

    return select_experts


def _wrap_shared_bypass(orig_forward_shared):
    """折叠层 bypass 原 shared 串行链:返回零张量保持外层合加语义。"""

    def _forward_shared_experts(self, hidden_states, fused_moe_evts):
        if not (_enabled() and getattr(self, "_tileinfer_fold_done", False)):
            return orig_forward_shared(self, hidden_states, fused_moe_evts)
        try:
            import torch

            return torch.zeros_like(hidden_states)
        except Exception:  # pragma: no cover
            return orig_forward_shared(self, hidden_states, fused_moe_evts)

    return _forward_shared_experts


def install() -> bool:
    """模型构建前挂包装;默认关(开关每步读,装后 =0 仍零行为变化)。"""
    global _INSTALLED
    if _INSTALLED:
        return True
    try:
        from vllm_ascend.ops.fused_moe import fused_moe as fm
    except Exception:
        logger.warning("H28 not installed: vllm_ascend fused_moe unavailable")
        return False

    cls = getattr(fm, "AscendFusedMoE", None)
    qm = getattr(fm, "AscendUnquantizedFusedMoEMethod", None)
    if cls is None or qm is None or not hasattr(cls, "shared_forward_impl"):
        logger.warning("H28 not installed: AscendFusedMoE seams missing")
        return False

    cls.__init__ = _wrap_moe_init(cls.__init__)
    cls.shared_forward_impl = _wrap_shared_forward(cls.shared_forward_impl)
    if hasattr(cls, "_forward_shared_experts"):
        cls._forward_shared_experts = _wrap_shared_bypass(cls._forward_shared_experts)
    if hasattr(qm, "process_weights_after_loading"):
        qm.process_weights_after_loading = _wrap_quant_pwals(qm.process_weights_after_loading)
    if hasattr(fm, "select_experts"):
        fm.select_experts = _wrap_select_experts(fm.select_experts)
    _INSTALLED = True
    logger.info("TileInfer shared expert fold installed (H28, TILEINFER_SHARED_EXPERT_FOLD)")
    return True
