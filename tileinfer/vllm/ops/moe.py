"""Torch implementations of the three MoE operators the Ascend backend needs.

Why these exist
---------------
vLLM 0.23 runs every MoE layer through operators an Ascend backend cannot build: ``topk_softmax``
and ``moe_align_block_size`` are ``torch.ops._moe_C`` CUDA kernels, and ``fused_experts`` is a Triton
kernel (plus the ``torch.ops._C``/``torch.ops._moe_C`` grouped GEMMs it calls).  The model only needs
the *semantics*: softmax-then-top-k routing, the padded per-expert slot layout the grouped GEMM
consumes, and the SwiGLU expert MLP with the weighted combine.  This module provides all three in
torch, **correctness first** - the expert loop is deliberately naive; the TileInfer kernels in
``tileinfer/kernels/moe/`` are the performance step, and they consume exactly the layout produced
here.

Provenance of every semantic claim
----------------------------------
* ``[vLLM]`` - a line number in the vLLM 0.23 source now vendored under ``runtime/vllm-api/``:
  ``vllm___custom_ops.py`` (the Python wrappers over ``torch.ops._moe_C``), and
  ``vllm__model_executor__layers__fused_moe__fused_moe.py`` (the Triton expert path).  These are the
  files that decide the *signatures* this module has to honour, and they are read, not guessed.
* ``[参考实现]`` - ``tileinfer/testing/moe_reference.py`` (the repository's executable spec).
* ``[内核 docstring]`` - ``tileinfer/kernels/moe/*.py`` (the kernels that consume these layouts).
* ``[推断]`` - still inferred, because the CUDA/Triton *bodies* are not in the local subset (only
  the Python wrappers are; ``csrc/`` and the kernels are not).  Every such claim is either made
  tolerant (both plausible behaviours available) or refused loudly, so a later reader knows exactly
  which assumption to re-check against the real ABI.

Signatures, as the real source states them
------------------------------------------
::

    # vllm/_custom_ops.py:2494-2509  ->  torch.ops._moe_C.topk_softmax
    def topk_softmax(topk_weights, topk_ids, token_expert_indices, gating_output,
                     renormalize: bool = False,
                     e_score_correction_bias: torch.Tensor | None = None) -> None

    # vllm/_custom_ops.py:2346-2363  ->  torch.ops._moe_C.moe_align_block_size
    def moe_align_block_size(topk_ids, num_experts, block_size,
                             sorted_token_ids, experts_ids, num_tokens_post_pad,
                             expert_map: torch.Tensor | None = None) -> None

    # .../fused_moe/fused_moe.py:1474-1485  ->  torch.ops.vllm.fused_experts (:1490)
    def fused_experts(hidden_states, w1, w2, topk_weights, topk_ids,
                      activation: MoEActivation = MoEActivation.SILU,
                      apply_router_weight_on_input: bool = False,
                      global_num_experts: int = -1,
                      expert_map: torch.Tensor | None = None,
                      quant_config: FusedMoEQuantConfig | None = None) -> torch.Tensor

The three implementations below take those arguments **in that order**, so a positional call from
vLLM lands correctly, and every knob this module cannot honour is either implemented or raises
(never silently ignored) - see "What the real source settles" and "Refused, with the reason".

What the real source settles (it was inferred before)
----------------------------------------------------
1. ``moe_align_block_size``'s positional order is ``(topk_ids, num_experts, block_size, ...)`` -
   ``num_experts`` **before** ``block_size`` (``vllm___custom_ops.py:2346-2354``).  This is the
   primary signature here, and it is no longer a guess.
   :func:`moe_align_block_size_block_first` keeps the other order because a *different*,
   newer module-level helper uses it - ``.../fused_moe/fused_moe.py:1465-1471`` calls
   ``moe_align_block_size(topk_ids, block_size, global_num_experts, expert_map,
   ignore_invalid_experts)`` from ``...fused_moe.moe_align_block_size``, which **returns** a triple
   instead of writing into caller buffers.  That one is not this ABI and must not be bound to this
   function (``ops/install.py`` says so where it matters).
2. ``renormalize`` is an explicit argument of ``topk_softmax`` itself (default ``False``, :2499), so
   renormalising inside this function is the same contract upstream has - it is not a guess about
   ``select_experts`` any more.  The value is still applied in fp32 before the cast to
   ``topk_weights.dtype``.
3. ``fused_experts``'s positional core is ``(hidden_states, w1, w2, topk_weights, topk_ids)``, then
   ``activation``, ``apply_router_weight_on_input``, ``global_num_experts``, ``expert_map`` and
   ``quant_config`` (:1474-1485).  Our signature is that one, with the TileInfer extras (shared
   expert, ``out_dtype``) keyword-only *after* it, so an upstream call - positional or keyword -
   binds without adaptation.

What is still inferred, and therefore tolerant
----------------------------------------------
1. ``token_expert_indices``' fill convention in ``topk_softmax``.  The wrapper's signature proves
   nothing about what the CUDA kernel writes there, and ``csrc/`` is not in the local subset.  Two
   conventions are plausible (``token * num_experts + expert`` vs. the expanded token index alone)
   [推断]; both are implemented and selectable.  The default cannot corrupt numerics: none of the
   other operators in this module read that buffer - ``fused_experts`` derives everything from
   ``topk_ids``.
2. Whether upstream fills ``sorted_token_ids``/``experts_ids`` past ``num_tokens_post_pad``.  We fill
   the *whole* buffer (sentinel / expert 0), which is a safe superset: a consumer that reads a whole
   block past the end sees exactly the "inactive tile" pattern the kernels already gate on
   [内核 docstring].

Refused, with the reason (accepted by the signature, rejected at the value)
--------------------------------------------------------------------------
These three are part of the real signatures and are *implemented* nowhere in this module.  Accepting
them silently would produce wrong routing/expert math, so each raises ``NotImplementedError`` naming
what is missing:

* ``topk_softmax(e_score_correction_bias=...)`` - the DeepSeek-style selection bias the CUDA kernel
  applies before top-k.  The wrapper shows the parameter exists (:2500) but not *where* the kernel
  adds it (selection score only, or the returned weight too), and getting that wrong changes which
  experts fire.  ``None`` (the default, and what every un-biased router passes) is exact.
* ``moe_align_block_size(expert_map=...)`` - the expert-parallel global->local remap.  Its exact
  interaction with the padded layout (which slots a ``-1`` expert leaves behind) is in the CUDA body,
  which is not local; v0 has no EP (plan §0), so ``None`` is the only path that runs.
* ``fused_experts(quant_config=...)`` - any quantised config (fp8/int8/int4/ocp-mx, per-activation or
  block scales, zps, biases).  v0 is bf16/fp16 (plan §0), and a quantised call must fail loudly
  rather than compute in the wrong precision.  ``None`` and an all-``False`` unquantised config
  (``FUSED_MOE_UNQUANTIZED_CONFIG``) are accepted; a config object this module cannot classify is
  refused too, because "I could not read it" is not "it is unquantised".

The shared-expert path is a TileInfer addition to the op, not part of upstream ``fused_experts``
[推断]: the Qwen3.6 block adds a dense expert *outside* the routed MoE, and folding it in here
keeps the whole block behind one call.  Its gating is ``sigmoid(x @ shared_gate_weight.T)``
multiplying the shared MLP output (``shared_expert_gate = Linear(hidden, 1)``, hidden 2048) - the
same algebra ``tileinfer/integrations/shared_expert_fold.py`` relies on ("门控语义与原
``sigmoid(expert_gate) * down(...)`` 逐点等价").
"""

from __future__ import annotations

from typing import Optional

import os

import torch

__all__ = [
    "TOKEN_EXPERT_INDEX_MODES",
    "fused_experts",
    "moe_align_block_size",
    "moe_align_block_size_block_first",
    "silu_and_mul",
    "topk_softmax",
]

#: Accepted values for :func:`topk_softmax`'s ``token_expert_index_mode`` (see the module's
#: "What is still inferred" note 1 - the CUDA body that would settle it is not in the local subset).
TOKEN_EXPERT_INDEX_MODES = ("token_expert", "token")

#: The ``quant_config`` attributes that mean "this config quantises something".  The names are the
#: ones ``torch.ops.vllm.fused_experts`` is called with (``fused_moe.py:1498-1514``), i.e. the
#: attribute names ``FusedMoEQuantConfig`` really has; anything else is treated as unclassified.
_QUANT_BOOL_ATTRS = (
    "use_fp8_w8a8",
    "use_int8_w8a8",
    "use_int8_w8a16",
    "use_int4_w4a16",
    "per_act_token_quant",
    "per_channel_quant",
)
_QUANT_VALUE_ATTRS = (
    "ocp_mx_scheme",
    "block_shape",
    "w1_scale",
    "w2_scale",
    "w1_zp",
    "w2_zp",
    "a1_scale",
    "a2_scale",
    "w1_bias",
    "w2_bias",
)


def _check_unquantized(quant_config: Optional[object]) -> None:
    """Refuse a quantised (or unreadable) ``fused_experts`` config; allow ``None``/unquantised.

    ``FusedMoEQuantConfig`` itself is not in the local vLLM subset, so this does not import it: it
    reads the attribute names the real call site uses (``fused_moe.py:1498-1514``) and refuses when
    any of them says "quantised".  A config object that exposes *none* of them is refused as
    unclassifiable - treating an unknown object as unquantised would silently run the wrong math.
    """
    if quant_config is None:
        return
    bool_attrs = {name: getattr(quant_config, name, None) for name in _QUANT_BOOL_ATTRS}
    value_attrs = {name: getattr(quant_config, name, None) for name in _QUANT_VALUE_ATTRS}
    recognised = [name for name, value in bool_attrs.items() if value is not None]
    recognised += [name for name, value in value_attrs.items() if value is not None]
    if not recognised:
        raise NotImplementedError(
            "fused_experts received a quant_config whose type "
            f"({type(quant_config).__name__}) exposes none of "
            f"{_QUANT_BOOL_ATTRS + _QUANT_VALUE_ATTRS}; this TileInfer port only runs unquantised "
            "bf16/fp16 MoE, and it cannot prove that object is unquantised - pass None, or a "
            "FusedMoEQuantConfig whose quantisation flags are all False."
        )
    active = sorted(
        [name for name, value in bool_attrs.items() if value]
        + [
            name
            for name, value in value_attrs.items()
            if value is not None and value is not False
        ]
    )
    if active:
        raise NotImplementedError(
            "fused_experts is called with a quantised config "
            f"({', '.join(active)}); this TileInfer port implements the unquantised SwiGLU expert "
            "MLP only (docs/plan-vllm-native-backend.md §0 keeps the run bf16/fp16). A quantised "
            "call must fail here rather than compute in the wrong precision."
        )


# ---------------------------------------------------------------------------------------------
# silu_and_mul - vLLM's ``SiluAndMul`` (``torch.ops._C.silu_and_mul``), also used below.
# ---------------------------------------------------------------------------------------------


def _silu_and_mul_f32(x: torch.Tensor) -> torch.Tensor:
    """``silu(x[..., :I]) * x[..., I:]`` with ``I = x.shape[-1] // 2``, no dtype cast."""
    inter = x.shape[-1] // 2
    return torch.nn.functional.silu(x[..., :inter]) * x[..., inter:]


def _npu_swiglu_fn():
    """The fused Ascend SwiGLU, or ``None`` when it cannot be used here.

    ``torch_npu.npu_swiglu(x, dim=-1)`` computes ``silu(x[..., :I]) * x[..., I:]`` in one kernel —
    exactly this module's :func:`silu_and_mul`, whose torch form is two device ops (a ``silu`` on a
    view plus a multiply).  At Qwen3.6 shapes the expert loop calls it once per hit expert, so the
    saving is one op x 256 experts x 32 layers per **prefill** step and x 8 x 32 per decode step.
    """
    global _NPU_SWIGLU
    if _NPU_SWIGLU is _UNSET:
        try:
            import torch_npu  # noqa: PLC0415 - optional, NPU-only

            _NPU_SWIGLU = getattr(torch_npu, "npu_swiglu", None)
        except Exception:  # noqa: BLE001 - no NPU build: keep the torch path
            _NPU_SWIGLU = None
    return _NPU_SWIGLU


_UNSET = object()
_NPU_SWIGLU = _UNSET


def _swiglu(x: torch.Tensor) -> torch.Tensor:
    """SwiGLU with the fused kernel when enabled, available **and applicable to this device**.

    The device check is not cosmetic: ``npu_swiglu`` is registered only for the PrivateUse1 backend,
    so on a host that *has* torch_npu (every CI/dev box for this repo) a CPU tensor would select the
    fused path because the symbol exists and then fail with a ``NotImplementedError`` from the
    dispatcher.  That is exactly how the MoE oracles failed on the server while passing on a CPU-only
    laptop — ``torch_npu`` present differs from ``tensor on NPU``.
    """
    from tileinfer.serving import flags  # noqa: PLC0415

    if flags.flag("moe_swiglu", "TILEINFER_MOE_SWIGLU", True):
        fn = _npu_swiglu_fn()
        if fn is not None and x.device.type != "cpu":
            return fn(x)
    return silu_and_mul(x)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU activation over a ``[..., 2*I]`` gate|up tensor, returning ``[..., I]``.

    Gate rows come first and up rows second in the fused layout [参考实现]
    (``reference_moe``: "gate = first half of the w13 output, up = second half"), which is also how
    :func:`fused_experts` reads ``w1`` [内核 docstring] (``moe_gate_up_swiglu`` accumulates
    ``W13[eid, n0:n0+BN]`` into ``acc_g`` and ``W13[eid, n0+BN:...]`` into ``acc_u``).

    The product is formed in fp32 and cast back to ``x.dtype``, matching a kernel that accumulates
    in fp32 and stores the activation in the input dtype [推断].
    """
    if x.dim() < 1:
        raise ValueError("silu_and_mul expects a tensor with at least one dimension")
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"silu_and_mul needs an even last dimension, got {x.shape[-1]}")
    return _silu_and_mul_f32(x.to(torch.float32)).to(x.dtype)


def _sorted_path_enabled() -> bool:
    """The sort-once expert loop is the default; ``TILEINFER_MOE_SORTED=0`` restores the old one.

    Enabled by default because it is **numerically identical** to the old loop — measured against
    the fp32 oracle it reproduces ``max_abs`` to the last digit (0.000053 / mean 4.3e-6, exactly the
    old loop's numbers), since it runs the same per-expert arithmetic in the same fp32 accumulator
    and only changes the order in which segments are visited and how the combine is issued.  Its
    measured ratio at Qwen3.6 shapes is 2.57x (33.89 -> 13.19 ms/call), which is M2's exit bar.
    """
    from tileinfer.serving import flags  # noqa: PLC0415 - keep the import off the module-load path

    return flags.flag("moe_sorted", "TILEINFER_MOE_SORTED", True)


#: Decode steps with at most this many tokens *may* use the capture-safe, host-read-free expert path.
#: Prefill keeps the cheaper host-read form: graph capture covers decode only.
_DEVICE_PATH_MAX_TOKENS = 8


def _device_path_enabled() -> bool:
    """``TILEINFER_MOE_DEVICE_PATH=1`` — the capture-safe expert path for decode.

    Removes the last per-layer device->host read (§70), which graph capture requires.  It was wrong
    until §78: the padded grid was bounded by ``num_tokens`` instead of the slot count, and a token
    with duplicate top-k ids can put several slots on one expert, so the excess was dropped and
    ``torch.empty`` rows were read through (``3.7e+19`` in the oracle).  Both primitives it uses are
    probed deterministic (§74 ``index_copy_``, §76 ``bmm``).
    """
    from tileinfer.serving import flags  # noqa: PLC0415

    return flags.flag("moe_device_path", "TILEINFER_MOE_DEVICE_PATH", False)


def _routed_device_into(
    slot_out, order, counts_dev, sorted_x, sorted_w, w1, w2, compute, num_experts, num_tokens
) -> None:
    """Every-expert ``bmm`` with **no host read** — the capture-safe form for decode.

    The sorted loop needs ``counts.tolist()`` (one device->host synchronisation per layer per step),
    and a synchronous copy inside a captured graph is illegal — that is exactly what made ACLGraph
    capture fail with ``Synchronous memory copy failed`` (§70).

    For a decode step every upper bound is known from the *shapes*: no expert can receive more than
    ``num_tokens`` slots, so padding each expert to ``num_tokens`` rows is safe and needs no counts
    on the host.  Decode means a handful of tokens, so the waste is a few hundred rows of ``hidden``
    (256 experts x 1 row at a single token), while prefill keeps the cheaper host-read form.
    """
    dev = sorted_x.device
    hidden = sorted_x.shape[-1]
    # The bound must be the **slot count**, not the token count: one token can put several of its
    # top-k slots on the same expert (duplicate expert ids), so a single expert can hold up to
    # ``top_k`` slots from one token.  Using ``num_tokens`` truncated the padded grid, silently
    # dropped the excess slots, and left ``torch.empty`` rows to be read through as 3.7e+19 (§77).
    max_n = max(1, int(slot_out.shape[0]))
    starts = (torch.cumsum(counts_dev, 0) - counts_dev).unsqueeze(1)          # [E, 1]
    rows = torch.arange(max_n, device=dev, dtype=torch.int64).unsqueeze(0)    # [1, max_n]
    src = starts + rows                                                       # [E, max_n]
    valid = rows < counts_dev.unsqueeze(1)                                    # [E, max_n]
    total = sorted_x.shape[0]
    src_flat = src.reshape(-1).clamp_(0, max(total - 1, 0))
    mask = valid.reshape(-1)
    padded = sorted_x.index_select(0, src_flat).reshape(num_experts, max_n, hidden) * mask.view(
        num_experts, max_n, 1
    ).to(sorted_x.dtype)
    padded_w = sorted_w.index_select(0, src_flat).reshape(num_experts, max_n)
    gate_up = torch.bmm(padded, w1.to(compute).transpose(1, 2))
    act = _swiglu(gate_up)
    y = torch.bmm(act, w2.to(compute).transpose(1, 2)) * padded_w.unsqueeze(-1)
    live = mask.nonzero().reshape(-1)
    dst = order.index_select(0, src_flat.index_select(0, live))
    slot_out.index_copy_(0, dst, y.reshape(-1, hidden).index_select(0, live))


def _moe_bmm_enabled() -> bool:
    """Batch every expert into one ``bmm`` instead of one launch per expert.

    Why: at Qwen3.6 prefill shapes the expert loop is **8192 kernel launches per step**
    (256 experts x 32 layers), and §19 measured the two matmuls alone at 7.6 ms against ~1.2 s for
    the whole thing — i.e. the cost is dispatch, not arithmetic.  ``bmm`` over a padded
    ``[num_experts, max_rows_per_expert, hidden]`` tensor is one launch per matmul; the padding waste
    (``num_experts * max_n`` rows vs the real ``num_tokens * top_k``) is ~2x of an arithmetic total
    that is already ~0.3% of the time.

    Deliberately shape-gated: at decode only a handful of experts are hit, so the *host-read* loop
    stays, except that decode now has its own capture-safe form (:func:`_routed_device_into`) which
    pads to ``num_tokens`` rows and reads nothing back to the host.
    """
    from tileinfer.serving import flags  # noqa: PLC0415

    return flags.flag("moe_bmm", "TILEINFER_MOE_BMM", True)


def _routed_bmm_into(
    slot_out, order, sorted_x, sorted_w, counts, w1, w2, compute, num_experts
) -> None:
    """One ``bmm`` per matmul over all experts, padding each bucket to the widest one.

    ``sorted_x``/``sorted_w`` are the slots in expert order and ``counts[e]`` how many belong to
    expert ``e`` (``counts[num_experts]`` are the dropped ones, which are skipped here and zeroed by
    the caller).  Every output row is written exactly once, through one ``index_copy_`` with a
    permutation as its index -- deterministic, unlike ``index_add_`` (§44).
    """
    dev = sorted_x.device
    hidden = sorted_x.shape[-1]
    live = counts[:num_experts]
    max_n = max(live) if live else 0
    if max_n == 0:
        return

    starts = torch.tensor(
        [sum(live[:e]) for e in range(num_experts)], device=dev, dtype=torch.int64
    ).unsqueeze(1)
    rows = torch.arange(max_n, device=dev, dtype=torch.int64).unsqueeze(0)
    cnt = torch.tensor(live, device=dev, dtype=torch.int64).unsqueeze(1)
    src = starts + rows                                   # [E, max_n] position in sorted order
    valid = rows < cnt                                    # [E, max_n]
    total_sorted = sorted_x.shape[0]
    src_flat = src.reshape(-1).clamp_(0, total_sorted - 1)
    mask = valid.reshape(-1)

    padded = (
        sorted_x.index_select(0, src_flat).reshape(num_experts, max_n, hidden) * mask.view(
            num_experts, max_n, 1
        )
    )
    padded_w = sorted_w.index_select(0, src_flat).reshape(num_experts, max_n)
    gate_up = torch.bmm(padded, w1.to(compute).transpose(1, 2))       # [E, max_n, 2*inter]
    act = _swiglu(gate_up)
    y = torch.bmm(act, w2.to(compute).transpose(1, 2))                # [E, max_n, hidden]
    y = y * padded_w.unsqueeze(-1)
    # Only the live rows land: their destination is the slot index that ``order`` maps to.
    dst = order.index_select(0, src_flat[mask])
    slot_out.index_copy_(0, dst, y.reshape(-1, hidden).index_select(0, mask.nonzero().reshape(-1)))


def _routed_sorted_into(
    slot_out, x_slot, w1, w2, flat_weights, slot_tokens, local, num_experts, compute, num_tokens
) -> None:
    """Sort the slots by expert **once**, walk contiguous segments, write results in slot order.

    The baseline loop above pays, per hit expert, a ``torch.nonzero`` (a full scan + allocation)
    plus three ``index_select`` and one ``index_add_``.  Measured at Qwen3.6 shapes (512 tokens,
    top-8, 256 experts): the two matmuls alone are 7.6 ms, while the whole call is ~100 ms — so
    ~93% of the cost is that per-expert plumbing, not arithmetic.

    With the slots sorted by expert, a segment is a *view* (``sorted_x[off:off+n]``), so the loop
    body is one slice + two matmuls + two elementwise ops.  Dropped slots (``local < 0``) are parked
    in a synthetic bucket past the last expert and never visited.

    Writes into the caller's ``slot_out`` in **slot order** via one ``index_copy_`` of a permutation.
    ``index_copy_`` (unlike ``index_add_``) has a unique destination per element and is therefore
    bit-deterministic, which the caller's fixed-order ``top_k`` reduction then keeps.
    """
    keyed = torch.where(local >= 0, local, torch.full_like(local, num_experts))
    order = torch.argsort(keyed, stable=True)
    sorted_ids = keyed.index_select(0, order)
    sorted_x = x_slot.index_select(0, order)
    sorted_w = flat_weights.index_select(0, order)
    counts_dev = torch.bincount(keyed, minlength=num_experts + 1)
    # Decode needs a path with **no device->host read** (§70: a synchronous copy inside a captured
    # graph is illegal and that is what stops ACLGraph).  Every bound here comes from the shapes.
    if num_tokens <= _DEVICE_PATH_MAX_TOKENS and _moe_bmm_enabled() and _device_path_enabled():
        _routed_device_into(
            slot_out,
            order,
            counts_dev[:num_experts],
            sorted_x,
            sorted_w,
            w1,
            w2,
            compute,
            num_experts,
            num_tokens,
        )
        return
    counts = counts_dev.tolist()
    # Batching pays only when most experts are actually hit (prefill); at decode a handful are, and
    # the padded form would multiply the arithmetic while gathering their weights into a dense batch
    # would copy tens of MB per layer.
    hot = sum(1 for c in counts[:num_experts] if c)
    if _moe_bmm_enabled() and hot * 2 >= num_experts:
        _routed_bmm_into(
            slot_out, order, sorted_x, sorted_w, counts, w1, w2, compute, num_experts
        )
        return
    # Dropped slots live in the synthetic bucket (index ``num_experts``) which the loop never visits.
    # Zero the buffer only when there is such a slot: ``counts`` is already on the host, so this costs
    # nothing to decide, and an unconditional ``zeros`` would clear ``[slots, hidden]`` on every layer
    # of every prefill step (tens of MB each time).
    if counts[num_experts]:
        out = torch.zeros(
            (sorted_x.shape[0], w2.shape[1]), dtype=torch.float32, device=x_slot.device
        )
    else:
        out = torch.empty(
            (sorted_x.shape[0], w2.shape[1]), dtype=torch.float32, device=x_slot.device
        )
    off = 0
    for e in range(num_experts):
        n = counts[e]
        if n == 0:
            continue
        h = sorted_x[off : off + n]
        act = _swiglu(h @ w1[e].to(compute).T)
        y = act @ w2[e].to(compute).T
        out[off : off + n] = y * sorted_w[off : off + n].unsqueeze(1)
        off += n
    slot_out.index_copy_(0, order, out)


def _moe_strict() -> bool:
    """Whether ``fused_experts`` re-validates ``topk_ids`` on *every* call.

    **Default off, because it was measured.**  The three ``bool(...)`` reductions are three
    device->host synchronisations per call; caching them once per ``(device, expert count)`` is
    worth **1.107x on TPOT — 10.6 ms per token, −9.6%, 4/4 paired runs** (§42).

    The trade-off is the error path: with the check cached, a *stale* out-of-range id surfaces as an
    indexing error from ``w1[e]`` rather than as this module's ``ValueError``.  That is still a loud
    failure, and a broken id layout breaks on the first step (which *is* checked), so the first-call
    validation covers the realistic case.  ``TILEINFER_MOE_STRICT=1`` (or ``moe_strict=1``) restores
    per-call validation while diagnosing a caller.
    """
    from tileinfer.serving import flags  # noqa: PLC0415

    return flags.flag("moe_strict", "TILEINFER_MOE_STRICT", False)


#: ``(str(device), num_experts)`` pairs whose ``topk_ids`` bounds have already been checked.
_VALIDATED_ID_BOUNDS: set = set()


def _matmul_dtype(hidden_states: torch.Tensor) -> torch.dtype:
    """The dtype the routed and shared expert matmuls run in.

    **Default: the input dtype** (``TILEINFER_MOE_FP32=1`` restores the historical fp32 upcast).

    The history here is worth keeping straight, because it twice looked like a non-effect:

    * with the per-expert *loop*, bf16 measured 1.035 — inside the noise, because that path is
      dispatch-bound and the dtype changes nothing about the number of launches;
    * with the *batched* (``bmm``) path the same switch measured **1.251 alone**, and combined with
      batching it is **4.517** against loop+fp32 (§54).  The two switches multiply because batching
      makes the path arithmetic-bound, and fp32 arithmetic on this NPU is ~4x the bf16 rate here.

    Numerics: against ``reference_moe`` (straight-line fp32) the fp32 path is 5.3e-5 and the bf16
    path 8.0e-5 on values of ~1.8e-2 — fp32-reassociation level, and bf16-with-fp32-accumulate is
    what the upstream Triton kernel does too, so this is the *closer* port.
    """
    if hidden_states.dtype == torch.float32:
        return torch.float32
    raw = os.environ.get("TILEINFER_MOE_FP32", "0").strip().lower()
    return torch.float32 if raw not in ("0", "false", "no") else hidden_states.dtype


# ---------------------------------------------------------------------------------------------
# Op 1: topk_softmax
# ---------------------------------------------------------------------------------------------


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: Optional[torch.Tensor] = None,
    *,
    token_expert_index_mode: str = "token_expert",
) -> None:
    """Fill ``topk_weights``/``topk_ids``/``token_expert_indices`` from router logits, in place.

    Signature is ``vllm/_custom_ops.py:2494-2501`` verbatim (which forwards to
    ``torch.ops._moe_C.topk_softmax`` at :2502): four pre-allocated output/input buffers, then
    ``renormalize`` and ``e_score_correction_bias``, no return value.  ``token_expert_index_mode``
    is this module's only extension and is keyword-only, so every upstream call form works unchanged.

    Semantics
    ---------
    * softmax over **all** experts in fp32, whatever ``gating_output``'s dtype is [推断] (the CUDA
      op loads to float before the reduction; doing it in the logits' own dtype would make bf16
      routing disagree with the fp32 reference);
    * ``k = topk_ids.shape[1]`` experts with the largest probability, descending;
    * **ties are broken towards the lower expert index**, explicitly and by construction: the
      selection is a *stable descending* sort over all experts (``torch.argsort(..., stable=True)``
      keeps equal elements in ascending-expert order), not ``torch.topk``, whose tie order is
      unspecified.  This is the only tie rule that makes the kernel and the CPU reference agree on
      degenerate routers (uniform logits, saturated fp32, duplicated logits from a folded gate);
    * the values written are the softmax probabilities of the selected experts, cast into
      ``topk_weights``' dtype; ``topk_ids`` receives the expert indices in ``topk_ids``' dtype;
    * ``renormalize=True`` divides the selected weights by their sum **in fp32, before** the cast to
      ``topk_weights.dtype`` (the same contract the op's own argument implies, :2499).

    Parameters
    ----------
    topk_weights : ``[num_tokens, top_k]`` float buffer, filled in place.
    topk_ids : ``[num_tokens, top_k]`` integer buffer, filled in place.
    token_expert_indices : ``[num_tokens, top_k]`` integer buffer, filled in place.  Convention
        ``"token_expert"`` (default) writes ``token_idx * num_experts + expert_idx``, ``"token"``
        writes the expanded token index alone - see the module's [推断] note 1; nothing else in
        this module reads this buffer.
    gating_output : ``[num_tokens, num_experts]`` router logits (fp32 or bf16, any float dtype).
    renormalize : divide the selected weights by their sum (off by default, as upstream's is).
    e_score_correction_bias : **refused when not None** - see the module docstring.  It is accepted
        so the real signature binds; the kernel's placement of the bias is not in the local subset,
        and guessing it would change which experts fire.
    token_expert_index_mode : see ``token_expert_indices`` above.
    """
    if e_score_correction_bias is not None:
        raise NotImplementedError(
            "topk_softmax(e_score_correction_bias=...) is not implemented: vLLM's wrapper accepts "
            "it (vllm/_custom_ops.py:2500) and forwards it to torch.ops._moe_C.topk_softmax "
            "(:2502-2508), but whether the CUDA kernel adds it to the selection score only or to "
            "the returned weight as well is in the kernel body, which is not in "
            "runtime/vllm-api/. Pass None (the default, and what an un-biased router does): that "
            "path is exact."
        )
    if gating_output.dim() != 2:
        raise ValueError(
            f"topk_softmax expects 2-D gating_output [num_tokens, num_experts], "
            f"got shape {tuple(gating_output.shape)}"
        )
    num_tokens, num_experts = gating_output.shape
    if topk_ids.shape != topk_weights.shape:
        raise ValueError(
            f"topk_ids {tuple(topk_ids.shape)} and topk_weights {tuple(topk_weights.shape)} "
            "must have the same shape"
        )
    if topk_ids.dim() != 2 or topk_ids.shape[0] != num_tokens:
        raise ValueError(
            f"topk_ids must be [num_tokens={num_tokens}, top_k], got {tuple(topk_ids.shape)}"
        )
    if token_expert_indices.shape != topk_ids.shape:
        raise ValueError(
            f"token_expert_indices {tuple(token_expert_indices.shape)} must match "
            f"topk_ids {tuple(topk_ids.shape)}"
        )
    top_k = topk_ids.shape[1]
    if not 1 <= top_k <= num_experts:
        raise ValueError(f"top_k={top_k} must be in [1, num_experts={num_experts}]")
    if token_expert_index_mode not in TOKEN_EXPERT_INDEX_MODES:
        raise ValueError(
            f"token_expert_index_mode must be one of {TOKEN_EXPERT_INDEX_MODES}, "
            f"got {token_expert_index_mode!r}"
        )

    # fp32 softmax over all experts [推断]; the cast is a no-op for fp32 logits.
    probs = torch.softmax(gating_output.to(torch.float32), dim=-1)
    # Stable descending sort: equal probabilities keep ascending expert order, i.e. the tie rule
    # is "lower expert index wins" by construction.
    order = torch.argsort(probs, dim=-1, descending=True, stable=True)[:, :top_k]
    weights = probs.gather(1, order)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)

    topk_ids.copy_(order.to(topk_ids.dtype))
    topk_weights.copy_(weights.to(topk_weights.dtype))

    tokens = torch.arange(num_tokens, device=gating_output.device, dtype=torch.int64).unsqueeze(1)
    if token_expert_index_mode == "token_expert":
        filled = tokens * num_experts + order
    else:  # "token"
        filled = tokens.expand_as(order)
    token_expert_indices.copy_(filled.to(token_expert_indices.dtype))


# ---------------------------------------------------------------------------------------------
# Op 2: moe_align_block_size
# ---------------------------------------------------------------------------------------------


def moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    expert_map: Optional[torch.Tensor] = None,
    *,
    validate: bool = True,
) -> None:
    """Expand ``topk_ids`` into the padded per-expert slot layout, in place.

    Signature is ``vllm/_custom_ops.py:2346-2354`` verbatim (forwarding to
    ``torch.ops._moe_C.moe_align_block_size`` at :2355-2363): ``topk_ids``, then ``num_experts``,
    then ``block_size`` - **that order**, which the real source settles (it was a coin flip before
    the file was available), then the three caller buffers, then ``expert_map``.  ``validate`` is
    this module's only extension and is keyword-only, so every upstream call form binds unchanged.

    Layout contract (the one the grouped GEMM consumes)
    ---------------------------------------------------
    ``R = numel(topk_ids)`` token-expert pairs are sorted by expert (ascending flat pair index
    within an expert) and every expert's run of ``n`` pairs is padded to ``ceil(n/block_size) *
    block_size`` slots [参考实现] (``build_slot_layout``: ``tiles_per_expert = cdiv(counts, block)``,
    pad slots carry ``num_expanded``):

    * ``sorted_token_ids[slot]`` = flat pair index (``token_idx * top_k + k``) that owns the slot,
      or the sentinel ``numel(topk_ids)`` for padding.  The kernels test ``SortedIds[row0] < R``
      to skip a whole tile [内核 docstring] (``moe_gate_up_swiglu``/``moe_down``), so the sentinel
      must be *at least* ``R`` and every real entry strictly below it.
    * ``experts_ids[tile]`` = expert of tile ``tile`` (one expert per padded tile, which is why a
      run is padded: a tile may not span two experts) [内核 docstring].  Pad tiles carry expert 0
      and are skipped anyway [参考实现].
    * ``num_tokens_post_pad[0]`` = ``sum_e ceil(n_e/block_size) * block_size`` - the number of
      slots actually written; an expert with zero tokens contributes zero slots.

    ``expert_map`` is **refused when not None**: it is the expert-parallel global->local remap of
    the real op (:2353), and which slots a ``-1`` expert leaves behind is decided inside the CUDA
    body, which is not in the local subset.  v0 runs without EP (plan §0), where it is ``None``.

    Deviation from a plausible upstream behaviour, deliberately
    -----------------------------------------------------------
    The *entire* ``sorted_token_ids`` buffer is filled (sentinel padding, and expert 0 in the
    ``experts_ids`` tail), not only the ``num_tokens_post_pad`` prefix [推断]: a consumer that
    reads a full block past the end then sees inactive tiles instead of stale ids from the previous
    step.  For the same reason a buffer larger than this step's layout is accepted.

    ``validate=True`` (default) costs one host sync: it range-checks ``topk_ids`` and the buffer
    sizes and raises instead of silently truncating.  Pass ``validate=False`` inside a captured
    region where the caller has already proven the bounds.
    """
    if expert_map is not None:
        raise NotImplementedError(
            "moe_align_block_size(expert_map=...) is not implemented: the real op takes the "
            "expert-parallel global->local map (vllm/_custom_ops.py:2353) and applies it inside "
            "torch.ops._moe_C.moe_align_block_size (:2355-2363), whose body is not in "
            "runtime/vllm-api/ - in particular whether a '-1' expert leaves a padding slot or "
            "shrinks num_tokens_post_pad. v0 runs without expert parallelism, where the argument "
            "is None; that path is exact."
        )
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if sorted_token_ids.dim() != 1 or experts_ids.dim() != 1:
        raise ValueError("sorted_token_ids and experts_ids must be 1-D buffers")
    if num_tokens_post_pad.numel() < 1:
        raise ValueError("num_tokens_post_pad must hold at least one element")

    device = topk_ids.device
    flat = topk_ids.reshape(-1).to(torch.int64)
    num_expanded = int(flat.numel())

    if validate and num_expanded:
        if bool((flat < 0).any()) or bool((flat >= num_experts).any()):
            raise ValueError(
                f"topk_ids must be in [0, num_experts={num_experts}); "
                f"got min={int(flat.min())}, max={int(flat.max())}"
            )

    # Per-expert counts.  scatter_add (not bincount) keeps the op portable to devices that do not
    # implement bincount; flat is already range-checked (or trusted under validate=False).
    counts = torch.zeros(num_experts, dtype=torch.int64, device=device)
    if num_expanded:
        counts.scatter_add_(0, flat, torch.ones_like(flat))

    tiles = (counts + (block_size - 1)) // block_size          # tiles per expert, 0 for empty
    tile_base = torch.cumsum(tiles, 0) - tiles                 # first tile of each expert
    group_start = torch.cumsum(counts, 0) - counts             # compact rank of each expert's first
    total_tiles_dev = tiles.sum()
    total_slots_dev = total_tiles_dev * block_size

    if validate:
        total_slots = int(total_slots_dev)
        if total_slots > sorted_token_ids.numel():
            raise ValueError(
                f"sorted_token_ids has {sorted_token_ids.numel()} slots but this routing needs "
                f"{total_slots}"
            )
        if int(total_tiles_dev) > experts_ids.numel():
            raise ValueError(
                f"experts_ids has {experts_ids.numel()} tiles but this routing needs "
                f"{int(total_tiles_dev)}"
            )

    # Whole-buffer sentinel fill, then the valid prefix (see the deviation note above).
    sorted_token_ids.fill_(num_expanded)
    experts_ids.fill_(0)

    if num_expanded:
        # Stable order by expert without relying on sort stability: key = expert * R + flat index
        # is strictly monotone for distinct (expert, index) pairs.
        key = flat * num_expanded + torch.arange(num_expanded, dtype=torch.int64, device=device)
        order = torch.argsort(key)
        sorted_experts = flat[order]
        rank = torch.arange(num_expanded, dtype=torch.int64, device=device)
        slot = tile_base[sorted_experts] * block_size + (rank - group_start[sorted_experts])
        sorted_token_ids[slot] = order.to(sorted_token_ids.dtype)

    per_tile = torch.repeat_interleave(torch.arange(num_experts, device=device), tiles)
    if per_tile.numel():
        experts_ids[: per_tile.numel()] = per_tile.to(experts_ids.dtype)

    num_tokens_post_pad.reshape(-1)[0] = total_slots_dev.to(num_tokens_post_pad.dtype)


def moe_align_block_size_block_first(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    expert_map: Optional[torch.Tensor] = None,
    *,
    validate: bool = True,
) -> None:
    """:func:`moe_align_block_size` with ``block_size`` before ``num_experts``.

    Kept because a *different* helper in vLLM 0.23 uses this order:
    ``.../fused_moe/fused_moe.py:1465-1471`` calls
    ``moe_align_block_size(topk_ids, block_size, global_num_experts, expert_map,
    ignore_invalid_experts)`` from the ``vllm.model_executor.layers.fused_moe.moe_align_block_size``
    module.  That function **returns** ``(sorted_token_ids, expert_ids, num_tokens_post_padded)``
    (see the ``return`` at :1465 and the ``_prepare_expert_assignment`` contract at :1437-1438)
    instead of writing into caller buffers, so it is *not* this ABI: ``ops/install.py`` deliberately
    does not bind it.  This alias exists only so a call site with the positional order of that
    module can be served by hand, with the same in-place semantics.
    """
    return moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        expert_map,
        validate=validate,
    )


# ---------------------------------------------------------------------------------------------
# Op 3: fused_experts
# ---------------------------------------------------------------------------------------------


def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: object = "silu",
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    quant_config: Optional[object] = None,
    *,
    w1_shared: Optional[torch.Tensor] = None,
    w2_shared: Optional[torch.Tensor] = None,
    shared_gate_weight: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Routed MoE MLP with the weighted top-k combine (plus an optional shared expert).

    Signature is ``.../fused_moe/fused_moe.py:1474-1485`` verbatim, in that positional order -
    ``(hidden_states, w1, w2, topk_weights, topk_ids)`` then ``activation``,
    ``apply_router_weight_on_input``, ``global_num_experts``, ``expert_map``, ``quant_config``.
    Note it is *not* the order this module used before the source was available
    (``expert_map`` used to precede ``global_num_experts``), so a positional call would have swapped
    two integers/tensors.  The TileInfer extras (shared expert, ``out_dtype``) are keyword-only
    **after** ``quant_config``, so they cannot shadow an upstream argument.

    ``activation`` arrives as a ``MoEActivation`` enum member from that call site
    (``MoEActivation.SILU`` is the default; the wrapper passes ``activation.value`` - a **string** -
    down to ``torch.ops.vllm.fused_experts`` at :1496, and ``fused_experts_impl`` does
    ``MoEActivation.from_str(activation)`` at :1570).  Because this port is bound at the *wrapper*
    level it accepts both spellings and compares the name, and it implements only ``silu``.

    Shapes (Qwen3.6-35B-A3B vocabulary) [参考实现]::

        hidden_states [num_tokens, hidden]        bf16/fp32
        w1            [num_experts, 2*inter, hidden]   rows 0..inter-1 = gate, inter..2*inter-1 = up
        w2            [num_experts, hidden, inter]
        topk_ids      [num_tokens, top_k]  int   (from :func:`topk_softmax`)
        topk_weights  [num_tokens, top_k]  float

    and the math, per token ``t`` with ``e_k = topk_ids[t, k]`` [参考实现]
    (``reference_moe``)::

        out[t] = Σ_k topk_weights[t,k] · ( silu(x[t] @ w1[e_k][:I].T) * (x[t] @ w1[e_k][I:].T) )
                                        @ w2[e_k].T

    A slot whose id repeats an earlier slot of the same token simply accumulates twice - duplicates
    are *legal* routing (the same expert may be selected by two columns) and must add, not overwrite.
    A negative id is a dropped/padded slot and contributes nothing [推断] (``expert_map`` uses
    ``-1`` for "not on this rank"; a padded ``topk_ids`` row must also be harmless).

    Parameters
    ----------
    activation : only ``"silu"``/``MoEActivation.SILU`` (SwiGLU) is implemented; anything else
        raises rather than silently substituting a different nonlinearity [推断] (upstream also has
        ``"gelu"`` via a different, tanh-approximated kernel - not verified offline, so not guessed).
    apply_router_weight_on_input : multiply the token's input row by its router weight before the
        gate/up projection instead of weighting the down-projection output.  Only valid for
        ``top_k == 1`` (with more slots the input scaling would be applied once per slot, which is
        a different function) [推断].
    global_num_experts : size of the id space ``expert_map`` is indexed by (``-1`` = infer from
        ``expert_map.numel()``, else from ``w1``'s first dimension; upstream's
        ``fused_experts_impl`` does the same at :1589-1590).
    expert_map : ``[global_num_experts]`` int map from global expert id to this rank's local row in
        ``w1``/``w2``, ``-1`` for "not local"; mapped-out slots contribute nothing [推断].  This is
        what an expert-parallel deployment uses, and it is also how a 256-expert routing geometry
        can be tested against a small local weight shard.
    quant_config : **refused unless unquantised** - ``None`` or a ``FusedMoEQuantConfig`` whose
        quantisation flags are all False (e.g. ``FUSED_MOE_UNQUANTIZED_CONFIG``).  Anything else
        raises, and so does an object this module cannot classify (see :func:`_check_unquantized`):
        running a quantised call through an fp32 SwiGLU would be silently wrong.
    w1_shared, w2_shared : optional dense shared expert (``[2*inter_s, hidden]`` /
        ``[hidden, inter_s]``), applied to *every* token; this is a TileInfer extension to the op
        [推断] and both must be given together.
    shared_gate_weight : optional ``[hidden]``/``[1, hidden]`` (or any shape with ``numel ==
        hidden``) gate for the shared expert; the shared output is multiplied by
        ``sigmoid(x @ gate)`` - ``shared_expert_gate = Linear(hidden, 1)`` in the model, see the
        module docstring.
    out_dtype : result dtype, default ``hidden_states.dtype`` [推断].

    Numerics: the projections run in fp32 after casting the used weight rows (correctness first: the
    reference is fp32), and only the returned tensor is cast down.  This makes ``fp32`` in/out match
    a fp32 oracle to rounding, and ``bf16`` differ only by the output cast - see the test tolerances.
    """
    if hidden_states.dim() != 2:
        raise ValueError(
            f"fused_experts expects 2-D hidden_states [num_tokens, hidden], "
            f"got {tuple(hidden_states.shape)}"
        )
    if w1.dim() != 3 or w2.dim() != 3:
        raise ValueError("w1 and w2 must be 3-D (one matrix per expert)")
    _check_unquantized(quant_config)
    # The enum member and the bare string are the same request; compare the name so a call from
    # either side of the wrapper (fused_moe.py:1480 passes the enum, :1496 the string) agrees.
    activation_name = str(getattr(activation, "value", activation)).strip().lower()
    if activation_name != "silu":
        raise NotImplementedError(
            f"activation={activation!r} is not implemented; only 'silu' (SwiGLU) is, because the "
            "gelu variant's exact kernel formula could not be verified offline"
        )

    num_tokens, hidden = hidden_states.shape
    num_experts, two_i, w1_hidden = w1.shape
    if w1_hidden != hidden:
        raise ValueError(f"w1 has hidden={w1_hidden}, expected {hidden}")
    if two_i % 2 != 0:
        raise ValueError(f"w1's second dimension must be 2*inter (even), got {two_i}")
    inter = two_i // 2
    if tuple(w2.shape) != (num_experts, hidden, inter):
        raise ValueError(f"w2 must be [{num_experts}, {hidden}, {inter}], got {tuple(w2.shape)}")

    if topk_ids.dim() != 2:
        raise ValueError(f"topk_ids must be 2-D, got {tuple(topk_ids.shape)}")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            f"topk_weights {tuple(topk_weights.shape)} must match topk_ids {tuple(topk_ids.shape)}"
        )
    if topk_ids.shape[0] != num_tokens:
        raise ValueError(
            f"topk_ids has {topk_ids.shape[0]} rows but hidden_states has {num_tokens} tokens"
        )
    top_k = topk_ids.shape[1]
    if apply_router_weight_on_input and top_k != 1:
        raise ValueError(
            "apply_router_weight_on_input is only defined for top_k == 1 "
            f"(got top_k={top_k}): the input scaling would be applied once per slot"
        )

    device = hidden_states.device
    compute = _matmul_dtype(hidden_states)
    x = hidden_states.to(compute)
    # Router weights and the output accumulator stay fp32 in both modes: they are tiny next to the
    # expert matmuls and fp32 accumulation is what keeps the top-k combine from drifting.
    flat_weights = topk_weights.to(torch.float32).reshape(-1)
    ids = topk_ids.reshape(-1).to(torch.int64)

    # ---- resolve global -> local expert ids ------------------------------------------------
    # Bounds checks that cost a device->host synchronisation each time they run.  In strict mode
    # (the default) they run on every call; otherwise once per (device, expert count), which is the
    # same information: ``topk_ids`` comes from an argmax over the same expert axis for the whole
    # process, so a broken layout breaks on the first step, not the thousandth.
    bounds_key = (str(device), int(num_experts), expert_map is None)
    check_now = _moe_strict() or bounds_key not in _VALIDATED_ID_BOUNDS
    if expert_map is not None:
        if expert_map.dim() != 1:
            raise ValueError(f"expert_map must be 1-D, got {tuple(expert_map.shape)}")
        if global_num_experts > 0 and expert_map.numel() != global_num_experts:
            raise ValueError(
                f"expert_map has {expert_map.numel()} entries but global_num_experts="
                f"{global_num_experts}"
            )
        map_t = expert_map.to(torch.int64)
        if check_now and ids.numel() and bool(ids.max() >= map_t.numel()):
            raise ValueError(
                f"topk_ids contains expert id {int(ids.max())} outside expert_map "
                f"[{map_t.numel()}]"
            )
        safe_ids = ids.clamp(min=0)
        local = torch.where(ids >= 0, map_t[safe_ids], torch.full_like(ids, -1))
    else:
        if check_now and ids.numel() and bool(ids.max() >= num_experts):
            raise ValueError(
                f"topk_ids contains expert id {int(ids.max())} but w1/w2 hold {num_experts} experts"
            )
        local = ids

    active = local >= 0
    if check_now and bool(active.any()) and bool(local[active].max() >= num_experts):
        raise ValueError(
            f"expert_map maps a token to local expert {int(local[active].max())} but w1/w2 hold "
            f"{num_experts} experts"
        )
    if check_now:
        _VALIDATED_ID_BOUNDS.add(bounds_key)
    # fp32 in both modes: it is the top-k combine's accumulator, and ``y * w_e`` promotes anyway.
    out = torch.zeros((num_tokens, hidden), dtype=torch.float32, device=device)

    if num_tokens and top_k:
        # slot s -> its token idx; the expand is the same row expansion build_slot_layout uses
        slot_tokens = torch.arange(num_tokens, device=device, dtype=torch.int64).repeat_interleave(
            top_k
        )
        if apply_router_weight_on_input:
            x_slot = x.index_select(0, slot_tokens) * flat_weights.unsqueeze(1)
        else:
            x_slot = x.index_select(0, slot_tokens)

        # One host sync (unique -> .tolist()); the loop is over experts actually hit, never all E.
        # The sorted path folds the router weight into the single combine, so it only applies when
        # the weight is *not* already applied to the input.
        #
        # The combine is a **slot buffer + a fixed-order reduction**, never ``index_add_``: the
        # atomic accumulate is not bit-deterministic on this NPU (measured: 50/50 trials differ by
        # ~1.2e-7, §45), and one fp32 ulp in a hidden state is enough to flip a greedy token 20 steps
        # later -- which is exactly what made this arm non-reproducible in the C3 check (§44).
        # ``empty`` here: every slot row is written by one of the two branches below, except a dropped
        # slot (``local < 0``), which no branch visits -- those rows are zeroed explicitly after the
        # branch, so the ``top_k`` reduction cannot pick up uninitialised memory.
        slot_out = torch.empty((num_tokens * top_k, hidden), dtype=torch.float32, device=device)
        dead = (local < 0).nonzero(as_tuple=False).reshape(-1)
        if _sorted_path_enabled() and not apply_router_weight_on_input:
            _routed_sorted_into(
                slot_out,
                x_slot,
                w1,
                w2,
                flat_weights,
                slot_tokens,
                local,
                num_experts,
                compute,
                num_tokens,
            )
        else:
            for e in torch.unique(local[active]).tolist():
                slots = torch.nonzero(active & (local == e), as_tuple=False).reshape(-1)
                h = x_slot.index_select(0, slots)
                gate_up = h @ w1[e].to(compute).T
                # ``silu_and_mul`` forms the product in fp32 and stores it in ``x.dtype``, so it is
                # the right activation for both modes (in fp32 mode the cast is a no-op).
                act = _swiglu(gate_up)
                y = act @ w2[e].to(compute).T
                if apply_router_weight_on_input:
                    slot_out[slots] = y
                else:
                    w_e = flat_weights.index_select(0, slots).unsqueeze(1)
                    slot_out[slots] = y * w_e
        # ``slot_out`` is in token-major slot order by construction, so the combine is a sum over a
        # fixed ``top_k`` axis -- deterministic, and cheaper than the scatter it replaces.
        if dead.numel():
            slot_out.index_fill_(0, dead, 0.0)
        out = out + slot_out.view(num_tokens, top_k, hidden).sum(dim=1)

    # ---- optional shared (dense) expert ----------------------------------------------------
    if (w1_shared is None) != (w2_shared is None):
        raise ValueError("w1_shared and w2_shared must be given together")
    if w1_shared is not None:
        if w1_shared.dim() != 2 or w2_shared.dim() != 2:
            raise ValueError("w1_shared/w2_shared must be 2-D dense weights")
        two_is, shared_hidden = w1_shared.shape
        if shared_hidden != hidden:
            raise ValueError(f"w1_shared has hidden={shared_hidden}, expected {hidden}")
        if two_is % 2 != 0:
            raise ValueError(f"w1_shared's first dimension must be 2*inter_s, got {two_is}")
        inter_s = two_is // 2
        if tuple(w2_shared.shape) != (hidden, inter_s):
            raise ValueError(
                f"w2_shared must be [{hidden}, {inter_s}], got {tuple(w2_shared.shape)}"
            )
        shared = _swiglu(x @ w1_shared.to(compute).T) @ w2_shared.to(compute).T
        if shared_gate_weight is not None:
            gate_row = shared_gate_weight.reshape(-1)
            if gate_row.numel() != hidden:
                raise ValueError(
                    f"shared_gate_weight must have {hidden} elements, got {gate_row.numel()}"
                )
            shared = shared * torch.sigmoid(x @ gate_row.to(compute)).unsqueeze(1)
        out = out + shared
    elif shared_gate_weight is not None:
        raise ValueError("shared_gate_weight requires w1_shared and w2_shared")

    return out.to(out_dtype or hidden_states.dtype)
