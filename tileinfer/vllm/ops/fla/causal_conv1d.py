"""Torch port of vLLM 0.23's mamba depthwise ``causal_conv1d`` operators.

Source of truth
---------------
``runtime/vllm-api/vllm__model_executor__layers__mamba__ops__causal_conv1d.py`` (1239 lines, read in
full).  Every semantic below is quoted from it with a line number.  Caller:
``runtime/vllm-api/vllm__model_executor__layers__mamba__gdn__qwen_gdn_linear_attn.py``

* prefill  -> ``causal_conv1d_fn(mixed_qkv.T, conv_weights, bias, activation=..., conv_states=...,
  has_initial_state=..., cache_indices=..., query_start_loc=..., metadata=attn_metadata)`` (l.1365)
* decode   -> ``causal_conv1d_update(mixed_qkv, conv_state, conv_weights, bias, activation,
  conv_state_indices=...)`` (l.1378 and l.1674; ``x`` is ``[num_tokens, dim]``, one token per
  sequence, so the internal ``seqlen`` is 1)
* spec-dec -> ``causal_conv1d_update(..., conv_state_indices=..., num_accepted_tokens=...,
  query_start_loc=spec_query_start_loc, max_query_len=spec_state_indices_tensor.size(-1))`` (l.1344)

What the operator computes
--------------------------
A *depthwise* (per-channel) causal convolution over time, one filter per channel, with the last
``width - 1`` inputs of every sequence kept in a cache line::

    y[t] = act(bias + sum_{i=0}^{width-1} weight[:, i] * x[t - (width - 1) + i])
    x[j] = 0 for j < 0          # zero left padding, i.e. no wrap-around from the cache

and the cache line afterwards holds the last ``width - 1`` inputs in chronological order
(column 0 = oldest, column ``width - 2`` = newest), so that the next call continues the convolution.

Tensor layouts (established, not guessed)
-----------------------------------------
**Both functions are channel-last, and their token axes differ - this is the layout decision:**

* ``causal_conv1d_fn`` (prefill): ``[dim, cu_seqlen]`` - channel axis is **axis 0 and innermost**
  (``x_ptr  # (dim, cu_seqlen)``, l.18; ``is_channel_last = (x.stride(0) == 1) & (x.stride(1) > 1)``,
  l.563; ``assert is_channel_last, "Need to run in channel-last layout"``, l.620).  Tokens run along
  axis 1, sequences are packed along that same axis and delimited by ``query_start_loc``.
* ``causal_conv1d_update`` (decode): ``[batch, dim]``, ``[batch, dim, seqlen]`` or ``[num_tokens, dim]``
  (l.1085-1090) - the **token/batch axis is axis 0** and ``dim`` is axis 1, i.e. token-major; the 3-D
  form puts ``seqlen`` last.  With ``query_start_loc`` the layout is ``[num_tokens, dim]``
  (``stride_x_token, stride_x_dim = x.stride()``, l.1171-1172).

So a decode-step result ``[T, dim]`` must be transposed to be compared with a prefill result
``[dim, T]``; the test suite does exactly that and nothing else (see
``test_update_token_by_token_equals_one_prefill_over_the_whole_sequence``).

* ``causal_conv1d_fn``: ``x`` is ``[dim, cu_seqlen]`` with the *channel* axis innermost:
  ``is_channel_last = (x.stride(0) == 1) & (x.stride(1) > 1)`` (l.563) and
  ``assert is_channel_last, "Need to run in channel-last layout"`` (l.620).  The GDN caller builds it
  as ``mixed_qkv_non_spec.transpose(0, 1)`` from a contiguous ``[T, dim]`` buffer (l.1362), so this is
  ``[dim, T]`` with ``stride == (1, dim)``.  The kernel comment says ``x_ptr  # (dim, cu_seqlen)``
  (l.18).  Batches are *variable length and packed*: sequences are concatenated along the token axis
  and delimited by ``query_start_loc`` (l.489-510, l.569 ``padded_batch = query_start_loc.size(0)-1``).
* ``causal_conv1d_update``: ``x`` is ``[batch, dim]``, ``[batch, dim, seqlen]`` or ``[num_tokens, dim]``
  (l.1085-1090), and the 2-D form is turned into ``[batch, dim, 1]`` (l.1132-1135); with
  ``query_start_loc`` the layout is ``[num_tokens, dim]`` (``stride_x_token, stride_x_dim =
  x.stride()``, l.1171-1172).  Attention: the token axis is axis 0 for the 2-D/varlen form but axis 2
  for the 3-D form - both are supported here.
* ``weight`` is ``[dim, width]`` (``assert (dim, width) == weight.shape``, l.619) with ``width``
  contiguous in memory (``assert weight.stride(1) == 1``, l.618/1160).  The GDN caller builds it as
  ``self.conv1d.weight.view(dim, width)`` (l.1325).
* ``conv_states`` / ``conv_state`` is ``[num_cache_lines, dim, state_len]`` with
  ``state_len >= width - 1`` (l.586-590 ``assert num_cache_lines == conv_states.shape[0] and dim ==
  conv_states.shape[1] and width - 1 <= conv_states.shape[2]``, l.1092, l.1144-1145).  The GDN caller
  passes ``conv_state`` possibly as a ``transpose(-1, -2)`` view (l.1311-1315), i.e. a non-contiguous
  tensor, so this port never assumes contiguity.

Which state columns are live
----------------------------
Only columns ``0 .. width - 2`` take part.  Both kernels hard-code
``state_len = (KERNEL_WIDTH - 1)  # can be passed via argument if it's not the same as this value``
(l.67-69) and (l.1143-1145, l.1181-1185).  The prefill kernel reads
``prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok`` (l.156) and then walks
*backwards* (l.166-172 for width 4: ``col2 = state[2]``, ``col1 = state[1]``, ``col0 = state[0]``), so
column 0 is the oldest prior token and column ``width-2`` the newest.  The update kernel reads forward
``col0 = state[0]``, ``col1 = state[1]``, ``col2 = state[2]`` (l.865-876) - same order.  Every store
is masked with ``idx_tokens_conv < state_len`` (l.233, l.277, l.304, l.932) with ``state_len ==
width-1``, so columns ``>= width-1`` are neither read nor written: they are alignment padding in the
cache line, and this port leaves them untouched.

The convolution itself
----------------------
Both kernels unroll the tap loop as ``acc = acc_preload (bias, fp32)`` then
``for j in static_range(KERNEL_WIDTH): ... acc += matrix_x * matrix_w`` where ``matrix_x`` is the
prior column for ``j < width-1`` and the *current* token for ``j == width-1`` (fn: l.411-452, update:
l.972-1055), and finally shift the window ``col0 = col1; col1 = col2; col2 = matrix_x`` (l.444-452,
l.1036-1055).  So ``y[t] = bias + sum_i w[i] * Z[t+i]`` with
``Z = concat(state[0:width-1], x_seq)`` - exactly the formula in the module docstring of the upstream
interface (``y[t] = sum_i w[i] * x[t - (width-1) + i]`` with zero left padding).  Activation:
``if SILU_ACTIVATION: acc = acc / (1 + tl.exp(-acc))`` (l.454-455, l.1057-1058) applied to the fp32
accumulator before it is stored.  ``SILU_ACTIVATION=activation in ["silu", "swish"]`` (l.736, l.1229),
with ``if isinstance(activation, bool) and activation: activation = "silu"`` (l.538) for
``causal_conv1d_fn`` and ``if isinstance(activation, bool): activation = "silu" if activation is True
else None`` (l.1125-1126) for ``causal_conv1d_update``.  ``activation=None`` therefore stores the raw
convolution.

Prefill state update (``causal_conv1d_fn``)
-------------------------------------------
The new cache line is "the last ``width-1`` samples of ``concat(old_state, x_seq)``", where
``old_state`` is replaced by zeros when ``has_initial_state[seq]`` is false.  Derived from the three
branches of the kernel and identical in all of them:

* ``state_len <= seqlen`` (l.196-235): reads ``idx_tokens_last = (seqlen - state_len) + arange(...)``
  and stores the last ``state_len`` tokens of the sequence.
* ``state_len > seqlen`` with an initial state (l.237-280): ``new_conv_state = tl.where(mask,
  conv_state_shifted_by_seqlen, loaded_x)`` with ``VAL = state_len - seqlen``, i.e. shift the old
  state left by ``seqlen`` and append the new tokens.
* ``state_len > seqlen`` without an initial state (l.281-307): the same shift but the old part is
  replaced by ``loaded_x``, whose mask ``idx_tokens_conv - VAL >= 0`` (l.294) zero-fills everything
  before the new tokens - i.e. left zero padding.

Consequence worth knowing: a sequence of length 0 with ``has_initial_state = False`` *zeroes* its
cache line (``concat(zeros, empty)[-(width-1):] == zeros``); with ``has_initial_state = True`` it keeps
the old one.  This port reproduces both.

The state is written for the *whole* sequence regardless of the kernel's ``BLOCK_M = 8`` chunking
(l.578, l.123-124): ``seqlen = sequence_end_index - sequence_start_index`` (l.84-87) is the full
sequence length and the store uses it (l.202-235, l.255-300), so a chunked launch and a one-shot launch
are numerically identical.

Decode state update (``causal_conv1d_update``) and ``in place`` semantics
------------------------------------------------------------------------
``l.1162-1163: # adopt the strategy in vLLM that overwrite on 'x' directly, rather than creating a new
tensor 'o' -> out = x``.  The *cache* is mutated in place through ``tl.store`` (l.925-933) into
``conv_state[conv_state_indices[seq]]``; ``conv_state`` is therefore both input and output and the
function returns no state.  The new window is ``shift-left-by-seqlen + append``:
``conv_state_ptrs_source`` reads at ``idx_tokens + seqlen`` while ``VAL = state_len - seqlen`` makes the
tail come from ``x`` (l.887-933), i.e. again ``concat(old, x_seq)[-(width-1):]`` for the normal
(``num_accepted_tokens is None``) case.

Speculative decoding (``num_accepted_tokens not None``) changes two things:
``state_len = width - 1 + (seqlen - 1)`` (l.1181-1182) and ``conv_state_token_offset =
num_accepted_tokens[seq] - 1`` (l.850-852), with the source index shifted by one more
(``idx_tokens + (1 if IS_SPEC_DECODING else seqlen)``, l.892) while the store still starts at column 0
(l.925-931).  So the window *grows* to ``width - 2 + seqlen`` columns: ``[history_{2+a} .. history_M,
draft_1 .. draft_N]`` as the comment at l.837-849 says, where ``a`` is the number of accepted tokens.

Varlen trimming (``query_start_loc`` given): ``state_len = state_len - (seqlen - (query_end_index -
query_start_index))`` (l.823-824) trims the *write* window to the actual sequence length, but the read
of the prior columns (l.864-879) is not trimmed - it always uses ``offset + 0 .. offset + width - 2``.
For a non-spec varlen batch whose sequences have different lengths this leaves the short sequences
reading stale columns; the GDN caller only uses the varlen form with ``num_accepted_tokens``
(l.1344-1357), where the trim exactly compensates the spec growth (``width-2+L``).  Both facts are
reproduced verbatim here and pinned by a test, because "fixing" them would change numbers.

Output dtype / precision
------------------------
``causal_conv1d_fn``: ``original_x_dtype = x.dtype; x = x.to(conv_states.dtype); out =
torch.empty_like(x) ... return out.to(original_x_dtype)`` (l.543-545, l.745) - the result is computed in
the *cache* dtype and cast back to the input dtype.  ``causal_conv1d_update`` does the same
(l.1130-1131, l.1239).  This port matches that (so bf16 x + fp32 cache returns bf16), but does the
arithmetic in fp32 with fp32 taps instead of multiplying in the low-precision dtype; the difference is
rounding only (and this side is the *more* accurate one).

Verified by test (tests/test_gdn_causal_conv1d.py)
-------------------------------------------------
1. one prefill call over ``T`` tokens == ``T`` decode calls against the same cache line (the property
   the engine relies on), including the final cache line;
2. ``y[t] = sum_i w[i] x[t-(width-1)+i]`` against a hand-written token loop oracle, width 4, several
   channels, ``activation in {None, "silu", "swish"}``;
3. zero left padding (``y[0]`` only sees ``x[0]``), state carried across calls;
4. varlen prefill and varlen decode with two sequences and distinct cache lines;
5. ``has_initial_state`` True/False, including "False => left zero padding";
6. in-place cache mutation, untouched padding columns ``>= width-1``, ``conv_state`` object identity;
7. dtype preservation fp32/bf16, and that the returned tensor is a fresh tensor;
8. ``null_block_id`` rows are skipped (no output, no state change);
9. spec-decoding window growth ``[history_{2+a} .. history_M, draft_1 .. draft_N]``;
10. APC arguments are rejected instead of silently ignored.

Deliberately *not* implemented (documented, not guessed)
--------------------------------------------------------
* APC / prefix caching (``block_idx_first_scheduled_token``, ``block_idx_last_scheduled_token``,
  ``initial_state_idx``, ``num_computed_tokens``, ``block_size_to_align``).  These only redirect which
  cache line each chunk writes to (l.91-121, l.340-384, l.801-807, l.920-924).  The GDN prefill call
  (l.1365-1375) passes none of them, so instead of guessing their exact semantics this module raises
  ``NotImplementedError`` when any is given.
* ``metadata`` (the pre-built ``nums_dict``/``batch_ptr``/``token_chunk_offset_ptr`` launch plan,
  l.546-550, l.663-687) affects kernel *scheduling* only, never the numbers, so it is accepted and
  ignored.  A test asserts that passing ``metadata`` changes nothing.
* ``pad_slot_id`` is accepted for signature compatibility but has no effect: upstream it only marks the
  Triton programs that carry no work (``if idx_seq == pad_slot_id: return``, l.81-82, and the
  ``PAD_SLOT_ID``-filled ``batch_ptr``, l.556-561), i.e. a launch-geometry artifact of the kernel, not a
  data-plane flag.  (``pad_slot_id = -1``, ``NULL_BLOCK_ID = 0`` per
  ``vllm/v1/attention/backends/utils.py``.)

Assumptions where the source cannot decide
------------------------------------------
* ``conv_states`/``cache_indices`` are required in practice: the wrapper dereferences
  ``conv_states.dtype`` (l.544) and the kernel dereferences both pointers (l.132-134), so ``None``
  would crash upstream.  Here ``conv_states=None`` is an error, and ``cache_indices=None`` falls back
  to the identity mapping ``arange(batch)`` (the only mapping consistent with the docstring's
  "updated inplace if cache_indices are not provided", l.493-499).
* ``has_initial_state=None`` is treated as all-false (upstream would dereference the null pointer,
  l.153); the false branch is exactly "prior tokens are zeros" (l.182-192).
* Out-of-range cache indices (``< 0`` or ``>= num_cache_lines``) are *skipped*; upstream would read and
  write outside the tensor (undefined behaviour, l.140-144).  Since ``NULL_BLOCK_ID == 0`` is skipped
  by upstream too (``if conv_states_input_coord == null_block_id: return``, l.136-139, l.814-817), the
  guard only fires on genuinely corrupt indices.
* ``width < 2`` is rejected: the unrolled kernel only defines taps for ``width >= 2`` (l.399-409) and
  would not compile.  ``width`` up to 6 is unrolled upstream (l.399-409, l.950-966); the generic loop
  here handles any ``width >= 2`` (a superset - a deviation in the safe direction).
* ``causal_conv1d_update`` returns a fresh tensor instead of aliasing ``out = x`` (l.1163): the caller
  only reads the return value (l.1378, l.1674), and skipping the aliasing keeps the input intact.  Rows
  that are skipped (null block, empty sequence) therefore keep the input's values, exactly as upstream
  does by construction.
"""

from __future__ import annotations

from typing import Optional

import torch

# vllm/v1/attention/backends/utils.py: PAD_SLOT_ID = -1, NULL_BLOCK_ID = 0
PAD_SLOT_ID = -1
NULL_BLOCK_ID = 0

__all__ = [
    "PAD_SLOT_ID",
    "NULL_BLOCK_ID",
    "causal_conv1d_fn",
    "causal_conv1d_update",
]


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------


def _silu(acc: torch.Tensor) -> torch.Tensor:
    """``acc = acc / (1 + tl.exp(-acc))`` - verbatim kernel activation (l.454-455, l.1057-1058).

    Written in this form rather than ``torch.nn.functional.silu`` so it matches the source expression
    term by term; the two agree to fp32 rounding.
    """
    return acc / (1.0 + torch.exp(-acc))


def _resolve_activation_fn(activation) -> bool:
    """Map the ``activation`` argument to "is silu".

    ``causal_conv1d_fn`` (l.538, l.736): ``if isinstance(activation, bool) and activation: activation =
    "silu"`` and then ``SILU_ACTIVATION = activation in ["silu", "swish"]``.  Note that a ``True``
    stored in a non-bool container (e.g. ``1``) is *not* silu upstream either.
    """
    if isinstance(activation, bool) and activation:
        activation = "silu"
    return activation in ("silu", "swish")


def _resolve_activation_update(activation) -> bool:
    """``causal_conv1d_update`` variant (l.1125-1128).

    ``if isinstance(activation, bool): activation = "silu" if activation is True else None
    elif activation is not None: assert activation in ["silu", "swish"]``.
    """
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None and activation not in ("silu", "swish"):
        raise ValueError(
            f"activation must be None, True/False, 'silu' or 'swish', got {activation!r}"
        )
    return activation in ("silu", "swish")


def _conv_lib_enabled() -> bool:
    """Decode conv through a depthwise ``conv1d`` (one library call) instead of mul + reduction.

    **Default off, because it was measured**: paired against the vectorised torch form it came out at
    **1.004** — no difference, 4/4 pairs inside ±2.1% (§42).  The two forms cost the same, so the
    default keeps the one with fewer dependencies (no grouped-conv kernel) and this stays available
    as ``TILEINFER_GDN_CONV_LIB=1``.
    """
    from tileinfer.serving import flags  # noqa: PLC0415

    return flags.flag("gdn_conv_lib", "TILEINFER_GDN_CONV_LIB", False)


def _conv_fast_enabled() -> bool:
    """The vectorised decode paths (one reduction instead of ``width`` mul/add rounds; a slice

    instead of ``width-1`` column gathers).  On by default — the values are identical and the
    contract tests run both shapes — and switchable so the end-to-end effect can be measured in one
    process (``TILEINFER_GDN_CONV_FAST=0``, or the ``gdn_conv_fast`` flag file).
    """
    from tileinfer.serving import flags  # noqa: PLC0415

    return flags.flag("gdn_conv_fast", "TILEINFER_GDN_CONV_FAST", True)


def _depthwise_causal_conv(
    x_seq: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    prior: Optional[torch.Tensor],
    width: int,
    use_silu: bool,
) -> torch.Tensor:
    """``y[t] = act(bias + sum_i weight[:, i] * Z[t + i])`` with ``Z = concat(prior, x_seq)``.

    ``x_seq`` is ``[L, dim]`` (token major), ``prior`` is ``[width-1, dim]`` in chronological order or
    ``None`` for "zeros before the sequence", and the result is ``[L, dim]`` fp32.  ``prior=None``
    reproduces the ``load_init_state == False`` branch of the kernel (l.182-192: "prior-tokens are
    zeros"), i.e. plain zero left padding.
    """
    length, dim = x_seq.shape
    taps = width - 1
    if prior is None:
        prior = torch.zeros((taps, dim), dtype=torch.float32, device=x_seq.device)
    z = torch.cat([prior.to(torch.float32), x_seq.to(torch.float32)], dim=0)
    w = weight.to(torch.float32)
    if length == 1 and _conv_fast_enabled() and _conv_lib_enabled():
        # Decode via a **depthwise** ``conv1d`` with ``groups == dim``: one library call computes
        # ``y[d] = sum_i w[d, i] * z[i, d]`` for every channel, i.e. exactly this tap sum, instead of
        # a mul + a reduction over a [dim, width] tensor.  ``conv1d`` is cross-correlation
        # (``out[t] = sum_i w[i] x[t+i]``), which is the same formula with no kernel flip.
        y = torch.nn.functional.conv1d(
            z.transpose(0, 1).unsqueeze(0),          # [1, dim, width]
            w.unsqueeze(1),                          # [dim, 1, width]
            None,                                    # bias is added once, by the shared tail below
            groups=dim,
        ).squeeze(2)                                 # [1, dim]
    elif length == 1 and _conv_fast_enabled():
        # Decode: one row, so the tap loop collapses into a single reduction over a [dim, width]
        # view instead of `width` rounds of (mul, add) on [1, dim].  Decode is dispatch-bound
        # (§22/§30: the per-op cost here is ~1 ms and this operator is the largest single one in a
        # decode step), so the op count is what matters; the arithmetic is the same sum of
        # `width` fp32 products, only accumulated pairwise rather than left to right.
        y = (z[:width].transpose(0, 1) * w).sum(dim=-1, keepdim=True).transpose(0, 1)
    else:
        y = torch.zeros((length, dim), dtype=torch.float32, device=x_seq.device)
        for i in range(width):
            y = y + z[i : i + length] * w[:, i].unsqueeze(0)
    if bias is not None:
        y = y + bias.to(torch.float32).unsqueeze(0)
    if use_silu:
        y = _silu(y)
    return y


def _prior_window(line: torch.Tensor, offset: int, width: int) -> torch.Tensor:
    """Columns ``offset .. offset+width-2`` of a cache line ``[dim, state_len]`` as ``[width-1, dim]``.

    Index order and the ``offset`` are the kernel's (l.864-879).  Reads outside the cache line are
    zero-filled: upstream has no bounds check there (it relies on ``state_len >= width-1`` and on
    ``offset + width - 2`` staying inside the line), so this only fires on corrupt inputs.
    """
    n_state = line.shape[-1]
    if 0 <= offset and offset + (width - 2) < n_state and _conv_fast_enabled():
        # Common case (decode: offset 0, state_len = width-1): the window is one slice, so it is a
        # view instead of `width-1` column gathers plus a stack.
        return line[:, offset : offset + width - 1].transpose(0, 1)
    cols = []
    for j in range(width - 1):
        c = offset + j
        if 0 <= c < n_state:
            cols.append(line[:, c])
        else:
            cols.append(torch.zeros(line.shape[0], dtype=line.dtype, device=line.device))
    return torch.stack(cols, dim=0)


def _cache_index_list(cache_indices, batch: int) -> list[int]:
    """Per-sequence cache line, ``arange(batch)`` when not given (see the module docstring)."""
    if cache_indices is None:
        return list(range(batch))
    ids = cache_indices.detach().to("cpu", dtype=torch.int64).tolist()
    if len(ids) < batch:
        raise ValueError(
            f"cache_indices has {len(ids)} entries but the batch has {batch} sequences"
        )
    return ids[:batch]


def _flag_list(flags, batch: int, name: str) -> list[bool]:
    if flags is None:
        return [False] * batch
    values = flags.detach().to("cpu").tolist()
    if len(values) < batch:
        raise ValueError(f"{name} has {len(values)} entries but the batch has {batch} sequences")
    return [bool(v) for v in values[:batch]]


def _check_no_apc(**kwargs) -> None:
    """APC / prefix caching is out of scope - fail loudly instead of computing wrong numbers."""
    given = [name for name, value in kwargs.items() if value is not None]
    if given:
        raise NotImplementedError(
            "causal_conv1d: prefix-caching / APC arguments are not implemented in this torch port: "
            + ", ".join(given)
            + ". They only choose which cache line a chunk writes to (upstream l.91-121, l.340-384, "
            "l.801-807, l.920-924); the GDN path passes none of them. Extend this module (and its "
            "tests) before enabling APC."
        )


# --------------------------------------------------------------------------------------------------
# prefill: many tokens per sequence, cache line updated from the whole sequence
# --------------------------------------------------------------------------------------------------


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    null_block_id: Optional[int] = NULL_BLOCK_ID,
    block_idx_first_scheduled_token: Optional[torch.Tensor] = None,
    block_idx_last_scheduled_token: Optional[torch.Tensor] = None,
    initial_state_idx: Optional[torch.Tensor] = None,
    num_computed_tokens: Optional[torch.Tensor] = None,
    block_size_to_align: int = 0,
    metadata=None,
    validate_data: bool = False,
) -> torch.Tensor:
    """Causal depthwise conv1d over a packed, variable-length batch, updating ``conv_states`` in place.

    Signature and defaults are upstream's (l.468-486); see the module docstring for the semantics and
    the source lines behind each of them.

    Parameters (upstream docstring, l.487-536, condensed and corrected where the kernel disagrees):
        x: ``[dim, cu_seqlen]``, channel-last.  ``cu_seqlen`` is the sum of all sequence lengths;
            sequences are packed left to right.
        weight: ``[dim, width]``.
        bias: ``[dim]`` or ``None``.
        conv_states: ``[num_cache_lines, dim, state_len >= width-1]``, mutated in place.
        query_start_loc: ``[batch+1]`` cumulative sequence lengths, starting at 0
            (``[0, 5, 6, 7, 8]`` for a 4-sequence batch of lengths ``[5, 1, 1, 1]``).
        cache_indices: ``[batch]`` cache line per sequence (``conv_states[cache_indices[i]]``).
        has_initial_state: ``[batch]`` bool - whether the cache line holds a real prefix.
        activation: ``None``, ``True``/``False``, ``"silu"`` or ``"swish"`` (default ``"silu"``).
        pad_slot_id: accepted, no effect (launch-geometry only; see the module docstring).
        null_block_id: cache lines equal to this are skipped (upstream default ``NULL_BLOCK_ID = 0``);
            ``None`` disables the check, as upstream's ``HAS_NULL_BLOCK = null_block_id is not None``
            (l.738) does.
        metadata: accepted and ignored (scheduling only).
        validate_data: adds upstream's shape assertions (l.602-626).

    Returns:
        ``[dim, cu_seqlen]``, same dtype as ``x``.  Rows of skipped sequences are zeros here, whereas
        upstream leaves them uninitialised (``torch.empty_like``, l.545) - no caller reads them.
    """
    _check_no_apc(
        block_idx_first_scheduled_token=block_idx_first_scheduled_token,
        block_idx_last_scheduled_token=block_idx_last_scheduled_token,
        initial_state_idx=initial_state_idx,
        num_computed_tokens=num_computed_tokens,
    )
    use_silu = _resolve_activation_fn(activation)

    if conv_states is None:
        raise ValueError(
            "conv_states is required: the upstream wrapper reads conv_states.dtype (l.544) and the "
            "kernel indexes it (l.132-134)"
        )
    if x.dim() != 2:
        raise ValueError(f"x must be [dim, cu_seqlen] (channel-last), got shape {tuple(x.shape)}")
    if weight.dim() != 2:
        raise ValueError(f"weight must be [dim, width], got shape {tuple(weight.shape)}")
    dim, cu_seqlen = x.shape
    if weight.shape[0] != dim:
        raise ValueError(f"weight has {weight.shape[0]} channels but x has {dim}")
    width = int(weight.shape[1])
    if width < 2:
        raise ValueError(f"width must be >= 2, got {width} (the kernel unrolls width >= 2, l.399-409)")
    if conv_states.dim() != 3:
        raise ValueError(
            f"conv_states must be [num_cache_lines, dim, state_len], got shape {tuple(conv_states.shape)}"
        )
    num_cache_lines = int(conv_states.shape[0])
    if int(conv_states.shape[1]) != dim:
        raise ValueError(f"conv_states has {conv_states.shape[1]} channels but x has {dim} (l.588)")
    if int(conv_states.shape[2]) < width - 1:
        raise ValueError(
            f"conv_states state_len {conv_states.shape[2]} < width-1 == {width - 1} (l.589)"
        )
    if bias is not None and (bias.dim() != 1 or bias.shape[0] != dim):
        raise ValueError(f"bias must be [dim] == [{dim}], got {tuple(bias.shape)}")

    if query_start_loc.dim() != 1 or query_start_loc.numel() < 1:
        raise ValueError("query_start_loc must be a 1-D cumulative-length tensor of size batch+1")
    padded_batch = int(query_start_loc.numel()) - 1  # l.569
    starts = query_start_loc.detach().to("cpu", dtype=torch.int64).tolist()
    if starts[0] != 0:
        raise ValueError(f"query_start_loc must start at 0, got {starts[0]} (l.502-505)")
    if starts[-1] > cu_seqlen:
        raise ValueError(
            f"query_start_loc ends at {starts[-1]} but x only has {cu_seqlen} tokens"
        )

    if validate_data:  # l.602-626
        assert x.dim() == 2
        assert query_start_loc.dim() == 1
        assert x.stride(0) == 1 or x.stride(1) == 1
        if bias is not None:
            assert bias.dim() == 1
            assert dim == bias.size(0)
        if cache_indices is not None:
            assert cache_indices.dim() == 1
            assert padded_batch == cache_indices.size(0)
        if has_initial_state is not None:
            assert has_initial_state.size() == (padded_batch,)
        assert (dim, width) == weight.shape
        # NOTE: upstream also asserts `is_channel_last` (l.620).  This port indexes logically and is
        # layout agnostic, so the assertion is deliberately dropped (see the module docstring).

    lines = _cache_index_list(cache_indices, padded_batch)
    has_init = _flag_list(has_initial_state, padded_batch, "has_initial_state")

    original_x_dtype = x.dtype
    conv_dtype = conv_states.dtype
    x_work = x.to(conv_dtype)  # l.544
    out = torch.zeros_like(x_work)  # upstream: torch.empty_like(x) (l.545)
    taps = width - 1

    for seq in range(padded_batch):
        start = int(starts[seq])
        end = int(starts[seq + 1])
        if end < start:
            raise ValueError(f"query_start_loc is not monotonic at sequence {seq}: {starts}")
        line = int(lines[seq])
        if null_block_id is not None and line == null_block_id:  # l.136-139
            continue
        if line < 0 or line >= num_cache_lines:  # undefined upstream, skipped here
            continue

        x_seq = x_work[:, start:end].transpose(0, 1)  # [L, dim], token major
        prior = conv_states[line, :, :taps].transpose(0, 1) if has_init[seq] else None
        y = _depthwise_causal_conv(x_seq, weight, bias, prior, width, use_silu)
        out[:, start:end] = y.to(conv_dtype).transpose(0, 1)

        # New cache line = last `taps` samples of concat(old_state, x_seq); the old part is zeros when
        # has_initial_state is false (l.196-307, see the module docstring).
        if prior is not None:
            base = prior.to(conv_dtype)
        else:
            base = torch.zeros((taps, dim), dtype=conv_dtype, device=conv_states.device)
        new_state = torch.cat([base, x_seq], dim=0)[-taps:]  # x_seq is already the cache dtype
        conv_states[line, :, :taps] = new_state.transpose(0, 1)

    return out.to(original_x_dtype)  # l.745


# --------------------------------------------------------------------------------------------------
# decode: one (or a few, for speculative decoding) token per sequence
# --------------------------------------------------------------------------------------------------


def _conv_capture_safe_enabled() -> bool:
    """Plain-decode conv update with **no device->host read**, so graph capture can engage.

    The general path reads ``conv_state_indices`` back to the host (``_cache_index_list``) to get one
    Python ``int`` per sequence, and a synchronous copy inside a captured graph is illegal (§70/§71:
    this is the last one on the decode path, 30 per step, one per GDN layer).

    ``TILEINFER_GDN_CONV_CAPTURESAFE=1`` selects it; it is default **off** until it is measured and
    has run against the oracles, matching every other candidate in this repo (C2).
    """
    from tileinfer.serving import flags  # noqa: PLC0415

    return flags.flag("gdn_conv_capture_safe", "TILEINFER_GDN_CONV_CAPTURESAFE", False)


def _conv_kernel_enabled() -> bool:
    """``TILEINFER_GDN_CONV_KERNEL=1`` — run the decode conv update through the fused kernel (§88).

    Default **off**: the kernel has a 42.5x in-process ratio on the device but has to clear the
    oracle suite and an end-to-end paired measurement before it may be a default (C2/E4).
    """
    from tileinfer.serving import flags  # noqa: PLC0415

    return flags.flag("gdn_conv_kernel", "TILEINFER_GDN_CONV_KERNEL", False)


#: How many fused-kernel launches have failed and fallen back to the torch path (diagnostics only).
_CONV_KERNEL_FALLBACKS = 0

#: ``(channels, device) -> zeros[channels] fp32`` for the ``bias is None`` case.  Without this the
#: fused path allocated a fresh zeros tensor on every layer of every decode step (30 per step).
_ZERO_BIAS_CACHE: dict = {}


#: ``(data_ptr, dtype, numel) -> fp32 bias``.  The GDN conv bias is a bf16 parameter, and converting
#: it on every layer of every decode step was a fresh 32 KB copy 30 times per step (§92).
_BIAS_FP32_CACHE: dict = {}


def _fp32_bias(bias: torch.Tensor) -> torch.Tensor:
    """The fp32 view of ``bias`` the kernel declares, converted once per parameter.

    Keyed on identity *and* holding a strong reference: a freed tensor's address must never be
    recycled into a stale hit (the same argument as ``hostcache``/``recurrent._decode_constants``).
    """
    if bias.dtype == torch.float32:
        return bias
    key = (int(bias.data_ptr()), str(bias.dtype), int(bias.numel()))
    entry = _BIAS_FP32_CACHE.get(key)
    if entry is not None and entry[0] is bias:
        return entry[1]
    converted = bias.to(torch.float32)
    if len(_BIAS_FP32_CACHE) >= 8:
        _BIAS_FP32_CACHE.clear()
    _BIAS_FP32_CACHE[key] = (bias, converted)
    return converted


def _zero_bias(channels: int, device) -> torch.Tensor:
    key = (int(channels), str(device))
    buffer = _ZERO_BIAS_CACHE.get(key)
    if buffer is None:
        buffer = torch.zeros(int(channels), dtype=torch.float32, device=device)
        if len(_ZERO_BIAS_CACHE) >= 8:
            _ZERO_BIAS_CACHE.clear()
        _ZERO_BIAS_CACHE[key] = buffer
    return buffer


def _try_conv_kernel(x_seq, conv_state, weight, bias, line: int, conv_dtype):
    """One fused conv+SiLU+state-shift call, or ``None`` when this shape/branch cannot use it.

    The kernel is built per exact shape and only supports the plain decode case: one token, one
    request, ``width == 4``, bf16 state, channels divisible by the core count.  Everything else
    returns ``None`` so the caller keeps its torch path — a wrong-shape call would either fail or,
    worse, silently compute the wrong window.
    """
    if conv_dtype != torch.bfloat16:
        return None
    try:
        from tileinfer.kernels.gdn.conv_update_ascend950 import (  # noqa: PLC0415
            build_conv_update_kernel,
            gdn_conv_update_kernel_available,
        )
    except Exception:  # pragma: no cover - kernel module missing
        return None
    if not gdn_conv_update_kernel_available():
        return None
    channels = int(weight.shape[0])
    width = int(weight.shape[1])
    lines = int(conv_state.shape[0])
    if width != 4 or channels % 32 != 0 or int(conv_state.shape[2]) != width - 1:
        return None
    # The kernel receives raw pointers and assumes dense ``[L, C, taps]`` / ``[C, W]`` layouts.  A
    # view (a slice of a bigger state pool, a transposed weight) would make it read the wrong
    # addresses and return *plausible* wrong numbers — the worst failure mode.  Refuse those.
    if not conv_state.is_contiguous() or not weight.is_contiguous():
        return None
    if x_seq.numel() != channels or not x_seq.is_contiguous():
        return None
    out = torch.empty(channels, dtype=conv_dtype, device=conv_state.device)
    # The kernel declares the bias as fp32 (matching ``bias``'s eager dtype), so cast rather than
    # letting a bf16 bias raise inside the launch — that raise used to be swallowed by the fallback
    # below and made the integration probe look like "identical output", i.e. "kernel not taken".
    bias_t = _zero_bias(channels, conv_state.device) if bias is None else _fp32_bias(bias)
    try:
        kernel = build_conv_update_kernel(
            channels=channels, width=width, lines=lines, dtype="bfloat16"
        )
        kernel(
            x_seq.reshape(-1).to(conv_dtype).contiguous(),
            conv_state,
            weight,
            bias_t,
            int(line),
            out,
        )
    except Exception as exc:  # noqa: BLE001 - a kernel problem must degrade, not kill the engine
        # Counted, not silent: a silent fallback is indistinguishable from "the kernel ran and gave
        # the same answer", which is exactly the confusion §89 records.
        global _CONV_KERNEL_FALLBACKS
        _CONV_KERNEL_FALLBACKS += 1
        if _CONV_KERNEL_FALLBACKS <= 3:
            import logging  # noqa: PLC0415

            logging.getLogger(__name__).warning(
                "TileInfer: fused conv kernel fell back to torch (%s: %s)",
                type(exc).__name__,
                exc,
            )
        return None
    return out


def _conv_state_single_write() -> bool:
    """Write the new conv window with one contiguous store instead of two strided ones.

    **Default off: the same-process A/B says it is slower.**  A cross-run comparison suggested
    0.72 -> 0.26 ms (2.77x), but this machine's absolute speed drifts by ~2x between windows (§35);
    alternating the two modes inside one process gives ``on`` 0.201 ms vs ``off`` 0.186 ms, i.e.
    **ratio 0.926** with a sign that flips across pairs (§83).  Kept as a switch because the
    *measurement* it triggered is the valuable part.
    """
    from tileinfer.serving import flags  # noqa: PLC0415

    return flags.flag("gdn_conv_state1", "TILEINFER_GDN_CONV_STATE1", False)


#: ``(device, num_cache_lines, batch)`` shapes whose cache indices have been bounds-checked once.
_CONV_INDEX_CHECKED: set = set()


def _capturing() -> bool:
    """Whether a graph capture is in progress (used to keep device->host reads out of it).

    Answers ``False`` whenever the runtime cannot tell, which only makes the caller do the check it
    would have done anyway.
    """
    try:
        import torch_npu  # noqa: PLC0415

        npu = getattr(torch, "npu", None)
        for name in ("is_current_stream_capturing", "is_capturing"):
            fn = getattr(npu, name, None) if npu is not None else None
            if fn is not None:
                return bool(fn())
    except Exception:  # pragma: no cover - no NPU runtime here
        return False
    return False


def _update_capture_safe(
    conv_state,
    x_work,
    weight,
    bias,
    width: int,
    use_silu: bool,
    conv_dtype,
    conv_state_indices,
    batch: int,
    out,
    n_state: int,
    num_cache_lines: int,
    null_block_id=None,
):
    """The plain-decode state update indexed by a **device** tensor.

    Same arithmetic as the loop it replaces (``_prior_window`` + ``_depthwise_causal_conv``, then
    shift-left-and-append), but every cache line is addressed through ``index_select`` /
    ``index_copy_`` instead of a host ``int``.  The bounds check runs **once per shape** — the same
    cached-validation trade-off as ``moe._VALIDATED_ID_BOUNDS`` (§42): a stale out-of-range index
    then fails inside ``index_copy_`` instead of raising our ``ValueError``.
    """
    index = conv_state_indices[:batch].to(torch.int64)
    key = (str(conv_state.device), int(num_cache_lines), int(batch))
    if key not in _CONV_INDEX_CHECKED and not _capturing():
        lowest = int(index.min())
        highest = int(index.max())
        if lowest < 0 or highest >= num_cache_lines:
            raise ValueError(
                f"conv_state_indices addresses cache lines [{lowest}, {highest}] but the state has "
                f"{num_cache_lines} lines; check the mamba state page mapping"
            )
        _CONV_INDEX_CHECKED.add(key)

    taps = width - 1
    lines = conv_state.index_select(0, index)      # [batch, dim, n_state], a copy
    new_lines = lines.clone()
    out_cols = []
    for seq in range(batch):
        line_view = lines[seq]                     # [dim, n_state]
        prior = _prior_window(line_view, 0, width)
        x_seq = x_work[seq, :, :1].transpose(0, 1)  # [1, dim]
        y = _depthwise_causal_conv(x_seq, weight, bias, prior, width, use_silu)
        out_cols.append(y[0])
        if taps > 1:
            new_lines[seq, :, : taps - 1] = line_view[:, 1:taps]
        new_lines[seq, :, taps - 1] = x_seq[0].to(conv_dtype)
    # ``null_block_id`` marks a sequence with no state page: upstream skips it entirely (no read, no
    # write), which the general path does with a Python ``continue``.  Same here, but as a mask so
    # nothing is decided on the host.
    keep = (
        torch.ones(batch, dtype=torch.bool, device=index.device)
        if null_block_id is None
        else (index != int(null_block_id))
    )
    conv_state.index_copy_(0, index[keep], new_lines[keep])
    stacked = torch.stack(out_cols, dim=0).to(out.dtype)
    out[keep, :, 0] = stacked[keep]
    return out


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[bool | str] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    max_query_len: int = -1,
    null_block_id: Optional[int] = NULL_BLOCK_ID,
    block_idx_last_scheduled_token: Optional[torch.Tensor] = None,
    initial_state_idx: Optional[torch.Tensor] = None,
    validate_data: bool = False,
) -> torch.Tensor:
    """Single-token (or few-token) causal conv1d step against a cached window, updated in place.

    Signature and defaults are upstream's (l.1069-1083); see the module docstring for the semantics
    and the source lines behind each of them.

    Parameters:
        x: ``[batch, dim]``, ``[batch, dim, seqlen]``, or ``[num_tokens, dim]`` with
            ``query_start_loc``.  ``[batch, dim]`` is treated as ``seqlen == 1`` (l.1132-1135).
        conv_state: ``[num_cache_lines, dim, state_len >= width-1]``; **mutated in place**.
        weight: ``[dim, width]``.  bias: ``[dim]`` or ``None``.
        activation: ``None`` (default), ``True``/``False``, ``"silu"`` or ``"swish"``.
        conv_state_indices: ``[batch]`` cache line per sequence (continuous batching).
        num_accepted_tokens: ``[batch]``; enables the speculative-decoding sliding window
            (state grows to ``width-2+seqlen`` columns).
        query_start_loc: ``[batch+1]``; varlen form, ``x`` is then ``[num_tokens, dim]``.
        max_query_len: ``seqlen`` used for the spec-decoding state length when varlen (l.1142).
        null_block_id: cache lines equal to this are skipped (default ``NULL_BLOCK_ID = 0``).
        block_idx_last_scheduled_token, initial_state_idx: APC, not implemented -> ``NotImplementedError``.
        validate_data: adds upstream's assertions (l.1122-1160).

    Returns:
        Same shape as ``x`` (minus the added token axis for the 2-D input) and dtype, a fresh tensor.
    """
    _check_no_apc(
        block_idx_last_scheduled_token=block_idx_last_scheduled_token,
        initial_state_idx=initial_state_idx,
    )
    use_silu = _resolve_activation_update(activation)

    if conv_state is None:
        raise ValueError("conv_state is required")
    if weight.dim() != 2:
        raise ValueError(f"weight must be [dim, width], got shape {tuple(weight.shape)}")
    dim = int(weight.shape[0])
    width = int(weight.shape[1])
    if width < 2:
        raise ValueError(f"width must be >= 2, got {width} (the kernel unrolls width >= 2, l.950-966)")
    if conv_state.dim() != 3:
        raise ValueError(
            f"conv_state must be [num_cache_lines, dim, state_len], got shape {tuple(conv_state.shape)}"
        )
    num_cache_lines, conv_dim, n_state = (int(v) for v in conv_state.shape)
    if conv_dim != dim:
        raise ValueError(f"conv_state has {conv_dim} channels but weight has {dim} (l.1151)")
    if n_state < width - 1:
        raise ValueError(
            f"conv_state state_len {n_state} < width-1 == {width - 1} (l.1149); the kernel reads and "
            "writes columns 0..width-2"
        )
    if bias is not None and (bias.dim() != 1 or bias.shape[0] != dim):
        raise ValueError(f"bias must be [dim] == [{dim}], got {tuple(bias.shape)}")

    varlen = query_start_loc is not None
    if varlen:
        if x.dim() != 2:
            raise ValueError(
                "with query_start_loc, x must be [num_tokens, dim] (upstream reads "
                f"`stride_x_token, stride_x_dim = x.stride()`, l.1172); got {tuple(x.shape)}"
            )
        if conv_state_indices is None:
            raise ValueError("query_start_loc requires conv_state_indices (l.1139)")
        if query_start_loc.dim() != 1 or query_start_loc.numel() < 2:
            raise ValueError("query_start_loc must be 1-D with batch+1 entries")
        if max_query_len <= 0:
            raise ValueError(f"max_query_len must be >= 1 with query_start_loc, got {max_query_len}")
        batch = int(conv_state_indices.numel())
        if int(x.shape[1]) != dim:
            raise ValueError(f"x has {x.shape[1]} channels but weight has {dim}")
        seqlen_outer = int(max_query_len)  # l.1142
        starts = query_start_loc.detach().to("cpu", dtype=torch.int64).tolist()
        if len(starts) < batch + 1:
            raise ValueError(
                f"query_start_loc has {len(starts)} entries but the batch has {batch} sequences"
            )
        if starts[-1] > int(x.shape[0]):
            raise ValueError(f"query_start_loc ends at {starts[-1]} but x has {x.shape[0]} tokens")
    else:
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # l.1134-1135
            squeezed = True
        elif x.dim() == 3:
            squeezed = False
        else:
            raise ValueError(
                "x must be [batch, dim], [batch, dim, seqlen] or [num_tokens, dim], got "
                f"{tuple(x.shape)}"
            )
        if int(x.shape[1]) != dim:
            raise ValueError(f"x has {x.shape[1]} channels but weight has {dim}")
        batch, _, seqlen_outer = (int(v) for v in x.shape)
        starts = [seq * seqlen_outer for seq in range(batch + 1)]
        if conv_state_indices is not None and int(conv_state_indices.numel()) < batch:
            raise ValueError(
                f"conv_state_indices has {conv_state_indices.numel()} entries but the batch has "
                f"{batch} sequences (l.1152-1157)"
            )

    if validate_data:  # l.1122-1160
        assert null_block_id is not None
        assert x.stride(1) == 1  # channel innermost (l.1124)
        assert batch <= num_cache_lines
        assert width - 1 <= n_state

    original_x_dtype = x.dtype
    conv_dtype = conv_state.dtype
    x_work = x.to(conv_dtype)  # l.1131
    out = x_work.clone()

    if (
        _conv_capture_safe_enabled()
        and conv_state_indices is not None
        and not varlen
        and num_accepted_tokens is None
        and x.dim() == 3
        and int(x.shape[2]) == 1
    ):
        # Plain decode with a known cache line per sequence: no host read anywhere (§71).
        result = _update_capture_safe(
            conv_state,
            x_work,
            weight,
            bias,
            width,
            use_silu,
            conv_dtype,
            conv_state_indices,
            batch,
            out,
            n_state,
            num_cache_lines,
            null_block_id,
        )
        result = result.to(original_x_dtype)  # the general path casts back on the way out
        return result.squeeze(-1) if squeezed else result

    lines = _cache_index_list(conv_state_indices, batch)
    accepted = None
    if num_accepted_tokens is not None:
        accepted = num_accepted_tokens.detach().to("cpu", dtype=torch.int64).tolist()
        if len(accepted) < batch:
            raise ValueError(
                f"num_accepted_tokens has {len(accepted)} entries but the batch has {batch} sequences"
            )

    # l.1162-1163: upstream aliases `out = x`; this port writes into a copy so the caller's buffer
    # survives (documented deviation).  Unprocessed rows therefore keep the input values, as upstream.
    out = x_work.clone()
    taps = width - 1

    for seq in range(batch):
        start = int(starts[seq])
        end = int(starts[seq + 1])
        length = end - start
        if varlen and length == 0:  # l.833-834
            continue
        line = int(lines[seq])
        if null_block_id is not None and line == null_block_id:  # l.814-817
            continue
        if line < 0 or line >= num_cache_lines:  # undefined upstream, skipped here
            continue

        if varlen:
            x_seq = x_work[start:end, :]  # [L, dim]
        else:
            x_seq = x_work[seq, :, :length].transpose(0, 1)  # [L, dim]

        # l.1181-1185: the spec-decoding window is longer by seqlen-1; l.823-824 trims it back to the
        # actual sequence length for varlen.  l.850-852: the read/write offset.
        if accepted is None:
            base_state_len = taps
            offset = 0
            shift = length
        else:
            base_state_len = taps + (seqlen_outer - 1)
            offset = int(accepted[seq]) - 1
            shift = 1
        state_len_eff = base_state_len - (seqlen_outer - length)

        line_view = conv_state[line]  # [dim, n_state], read before the write (STEP 1 / STEP 2 order)

        # Experimental fused kernel (§88): one launch for conv+bias+SiLU+state shift.  The condition
        # is "one token per sequence" (``length == 1``), **not** ``batch == 1``: serving runs with
        # ``--max-num-seqs 4``, so requiring a single-sequence batch made the kernel unreachable in
        # practice (§91).  The loop below is already per sequence, so each call gets its own line and
        # its own ``[dim]`` row.
        if (
            _conv_kernel_enabled()
            and not varlen
            and accepted is None
            and offset == 0
            and length == 1
        ):
            fused = _try_conv_kernel(x_seq, conv_state, weight, bias, line, conv_dtype)
            if fused is not None:
                out[seq, :, 0] = fused.to(original_x_dtype).reshape(-1)
                continue

        prior = _prior_window(line_view, offset, width)  # [width-1, dim]
        y = _depthwise_causal_conv(x_seq, weight, bias, prior, width, use_silu)
        if varlen:
            out[start:end, :] = y.to(conv_dtype)
        else:
            out[seq, :, :length] = y.to(conv_dtype).transpose(0, 1)

        # l.887-933: new window = shift-left-by-`shift` + append, stored from column 0.
        if state_len_eff > 0:
            take = min(length, state_len_eff)
            keep = state_len_eff - take
            if (
                _conv_fast_enabled()
                and not varlen
                and offset == 0
                and accepted is None
                and shift == 1
                and take == 1
                and keep == state_len_eff - 1
                and state_len_eff <= n_state
            ):
                # Plain decode (one token, no spec window, no offset): the window is exactly
                # "drop the oldest column, append this token".  The general form below costs a
                # device→host sync (`bool(valid.any())`) on **every** call -- 30 of them per decode
                # step, one per GDN layer -- plus an ``arange``, two masks, an advanced-index gather
                # and a ``cat``.  This form is two slice writes and needs no control flow.
                # Two ops instead of three: build the new window with one ``cat`` and write it with
                # one slice assignment.  ``cat`` materialises before the write, so the
                # overlapping-view hazard the in-place form had to clone around cannot arise.
                if state_len_eff > 1 and _conv_state_single_write():
                    new_line = torch.cat(
                        [
                            line_view[:, 1:state_len_eff],
                            x_seq[0].to(conv_dtype).unsqueeze(1),
                        ],
                        dim=1,
                    )
                    conv_state[line, :, :state_len_eff] = new_line
                elif state_len_eff > 1:
                    # Two strided stores (the historical form): each is stride-``n_state`` on the
                    # last axis, which is what made this op expensive on NPU (§82).
                    nxt = line_view[:, 1:state_len_eff].clone()
                    conv_state[line, :, : state_len_eff - 1].copy_(nxt)
                    conv_state[line, :, state_len_eff - 1] = x_seq[0].to(conv_dtype)
                else:
                    conv_state[line, :, state_len_eff - 1] = x_seq[0].to(conv_dtype)
            else:
                parts = []
                if keep > 0:
                    src = torch.arange(keep, device=conv_state.device) + (offset + shift)
                    block = torch.zeros((keep, dim), dtype=conv_dtype, device=conv_state.device)
                    valid = (src >= 0) & (src < n_state)
                    if bool(valid.any()):
                        block[valid] = line_view[:, src[valid]].transpose(0, 1)
                    parts.append(block)
                if take > 0:
                    x_from = length - take
                    parts.append(x_seq[x_from : x_from + take])
                new_state = torch.cat(parts, dim=0)
                write_cols = min(state_len_eff, n_state)
                conv_state[line, :, :write_cols] = new_state[:write_cols].transpose(0, 1)

    if not varlen and squeezed:
        out = out.squeeze(-1)  # l.1237-1238
    return out.to(original_x_dtype)  # l.1239
