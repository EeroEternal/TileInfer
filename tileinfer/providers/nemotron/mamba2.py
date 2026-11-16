"""Correctness-first Mamba2 SSD operators for Ascend (v3 provider).

Why this file exists
--------------------
vLLM 0.23's Mamba2 prefill path calls ``mamba_chunk_scan_combined_varlen`` and its decode path
calls ``selective_state_update``; both are Triton kernels (``ssd_chunk_scan.py`` /
``mamba_ssm.py``).  The Mamba2 mixer's prefill also calls ``causal_conv1d_fn`` before the SSD
scan.  On Ascend the SSD kernel fails in Triton code generation at
``ssd_chunk_scan.py:238`` (``Currently ptr type from different source not supported``), and the
vendor prefill convolution kernel returns all zeros at the target geometry.  This module
implements the three ABI entry points with torch tensor operations, so the Mamba2 mixer has a
real path on the vendor platform.

Scope and honesty
-----------------
* This is the **reference** implementation for correctness, not a performance claim.  It keeps
  the scalar recurrence's arithmetic (softplus dt, exp(dt*A), state update, C read-out, D skip,
  optional z gate) in float32, and only casts at the storage boundary.
* It is an operator replacement, not a model registration and not a claim that the target 30B has
  been through C1/C2.  The provider evidence lives in the Mamba2 bring-up notes.
* No vLLM import at module level: the CPU tests and ``tileinfer doctor`` can import it.
* This is the canonical v3 implementation.  The old v2 module is archived at
  ``legacy/v2/tileinfer/vllm/ops/mamba2.py``; it is no longer part of the installed package.
* No compile, no device sync in the tensor math.  The varlen/chunk loops read host-side metadata
  once per call; a later kernel can replace the inner matmuls without changing this ABI.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "causal_conv1d_fn",
    "mamba_chunk_scan_combined_varlen",
    "selective_state_update",
]


def _as_int_list(value: Optional[Iterable], *, device: torch.device) -> List[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [int(v) for v in value.detach().to("cpu").tolist()]
    return [int(v) for v in value]


def _repeat_groups(x: torch.Tensor, nheads: int) -> torch.Tensor:
    """[T, G, N] -> [T, H, N], head ``h`` using group ``h // (H/G)``."""
    groups = x.shape[1]
    if groups == nheads:
        return x
    if nheads % groups:
        raise ValueError(f"nheads={nheads} is not divisible by ngroups={groups}")
    return x.repeat_interleave(nheads // groups, dim=1)


def _softplus_dt(dt: torch.Tensor, dt_bias: Optional[torch.Tensor], enabled: bool) -> torch.Tensor:
    if dt_bias is not None:
        dt = dt + dt_bias
    if enabled:
        dt = F.softplus(dt)
    return dt


def _chunk_output(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    state: torch.Tensor,
    D: Optional[torch.Tensor],
    z: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    dt_softplus: bool,
    dt_limit: Tuple[float, float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One logical chunk: math from the Mamba2 recurrence, vectorised over its tokens.

    Returns ``(y, state_end)``.  ``state`` is the state *before* this chunk; ``y`` has the same
    dtype as ``x`` and shape [T, H, P].
    """
    t, h, p = x.shape
    n = B.shape[-1]
    device = x.device
    work = torch.float32

    xf = x.to(work)
    dtf = _softplus_dt(dt.to(work), None if dt_bias is None else dt_bias.to(work), dt_softplus)
    dtf = torch.clamp(dtf, float(dt_limit[0]), float(dt_limit[1]))
    af = A.to(work)
    bf = _repeat_groups(B.to(work), h)
    cf = _repeat_groups(C.to(work), h)
    sf = state.to(work)

    # A * dt, and its inclusive prefix sum inside this logical chunk.
    da = dtf * af[None, :]  # [T, H]
    da_cs = torch.cumsum(da, dim=0)

    # x is discretised once; both the intra-chunk term and the state update use it.
    xdt = xf * dtf[..., None]  # [T, H, P]

    # Intra-chunk (diagonal block): G[t,s,h] = <C[t,h], B[s,h]>, masked to s <= t.
    g = torch.einsum("thn,shn->hts", cf, bf)
    diff = da_cs[:, None, :] - da_cs[None, :, :]  # [T, T, H]
    causal = torch.tril(torch.ones(t, t, device=device, dtype=torch.bool))
    # Mask before exp: the strict upper triangle can have a positive exponent that overflows
    # to inf, and ``inf * False`` is NaN, not zero.
    decay = torch.exp(diff.masked_fill(~causal[..., None], float("-inf")))
    m = g * decay.permute(2, 0, 1)
    y_diag = torch.einsum("hts,shp->thp", m, xdt)

    # Inter-chunk (off-diagonal block): read the incoming state with C * exp(dA_cs).
    c_decay = cf * torch.exp(da_cs)[..., None]
    y_off = torch.einsum("thn,hpn->thp", c_decay, sf)

    y = y_diag + y_off
    if D is not None:
        df = D.to(work)
        if df.dim() == 1:
            df = df[:, None]
        y = y + xf * df[None, :, :]
    if z is not None:
        y = y * F.silu(z.to(work))

    # State at the end of this chunk: decay the incoming state, add B-weighted xdt.
    state_decay = torch.exp(da_cs[-1])  # [H]
    weights = torch.exp(da_cs[-1][None, :] - da_cs)  # [T, H]
    state_add = torch.einsum("th,thp,thn->hpn", weights, xdt, bf)
    state_end = sf * state_decay[:, None, None] + state_add
    return y.to(x.dtype), state_end


def _chunk_plan(
    cu_seqlens: Optional[Sequence[int]],
    cu_chunk_seqlens: Optional[Sequence[int]],
    seq_idx: Optional[Sequence[int]],
    total: int,
    chunk_size: int,
) -> Tuple[List[int], List[int], List[int]]:
    """Return ``(chunk_starts, chunk_ends, chunk_seq_ids)`` for the packed token buffer.

    The vLLM metadata builder supplies all three; the fallback is only for CPU tests and synthetic
    calls.  Chunk boundaries are physical (``pos % chunk_size``), matching
    ``compute_varlen_chunk_metadata`` so prefix-caching state indices stay valid.
    """
    starts = list(cu_chunk_seqlens or [])
    if starts:
        return starts[:-1], starts[1:], list(seq_idx or [])

    bounds = list(cu_seqlens or [0, total])
    if len(bounds) < 2:
        bounds = [0, total]
    chunk_starts: List[int] = []
    chunk_ends: List[int] = []
    chunk_ids: List[int] = []
    for b in range(len(bounds) - 1):
        pos, end = int(bounds[b]), int(bounds[b + 1])
        while pos < end:
            room = chunk_size - (pos % chunk_size)
            take = min(room, end - pos)
            chunk_starts.append(pos)
            chunk_ends.append(pos + take)
            chunk_ids.append(b)
            pos += take
    return chunk_starts, chunk_ends, chunk_ids


def mamba_chunk_scan_combined_varlen(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    cu_seqlens: Optional[torch.Tensor],
    cu_chunk_seqlens: Optional[torch.Tensor],
    last_chunk_indices: Optional[torch.Tensor],
    seq_idx: Optional[torch.Tensor],
    out: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    dt_limit: Tuple[float, float] = (0.0, float("inf")),
    return_intermediate_states: bool = False,
    state_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Torch implementation of vLLM's ``mamba_chunk_scan_combined_varlen`` contract.

    The output is written in-place into ``out`` (shape [tokens, nheads, headdim]).  With
    ``return_intermediate_states`` the return value is the state at every logical chunk boundary,
    [nchunks, nheads, headdim, dstate]; otherwise it is the final state per sequence,
    [batch, nheads, headdim, dstate].
    """
    total, nheads, headdim = x.shape
    ngroups = B.shape[1]
    dstate = B.shape[2]
    device = x.device
    bounds = _as_int_list(cu_seqlens, device=device)
    if not bounds:
        bounds = [0, total]
    starts, ends, seqs = _chunk_plan(
        bounds,
        _as_int_list(cu_chunk_seqlens, device=device),
        _as_int_list(seq_idx, device=device),
        total,
        int(chunk_size),
    )

    nseq = len(bounds) - 1
    state = torch.zeros((nseq, nheads, headdim, dstate), device=device, dtype=torch.float32)
    if initial_states is not None:
        if tuple(initial_states.shape) != tuple(state.shape):
            raise ValueError(
                f"initial_states shape {tuple(initial_states.shape)} != {tuple(state.shape)}"
            )
        # Clone: ``_chunk_output`` writes ``state[seq]`` in place, and the caller owns
        # ``initial_states``.  An fp32 input otherwise aliases it and is silently mutated.
        state = initial_states.to(torch.float32).clone()

    records: List[torch.Tensor] = []
    for start, end, seq in zip(starts, ends, seqs):
        if end <= start:
            continue
        if seq < 0 or seq >= nseq:
            raise ValueError(f"chunk seq_idx={seq} outside [0, {nseq})")
        y, state[seq] = _chunk_output(
            x[start:end],
            dt[start:end],
            A,
            B[start:end],
            C[start:end],
            state[seq],
            D,
            None if z is None else z[start:end],
            dt_bias,
            dt_softplus,
            dt_limit,
        )
        out[start:end].copy_(y)
        if return_intermediate_states:
            records.append(state[seq].clone().to(state_dtype or torch.float32))

    if return_intermediate_states:
        if records:
            return torch.stack(records, dim=0)
        return torch.empty(
            (0, nheads, headdim, dstate), device=device, dtype=state_dtype or torch.float32
        )
    return state.to(state_dtype or torch.float32)


def _apply_activation(x: torch.Tensor, activation: object) -> torch.Tensor:
    if activation is True:
        activation = "silu"
    if activation in (None, False):
        return x
    if activation in ("silu", "swish"):
        return x * torch.sigmoid(x)
    raise ValueError(f"unsupported causal-conv activation {activation!r}")


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_states: Optional[torch.Tensor],
    query_start_loc: torch.Tensor,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    activation: object = "silu",
    pad_slot_id: int = -1,
    null_block_id: int = 0,
    block_idx_first_scheduled_token: Optional[torch.Tensor] = None,
    block_idx_last_scheduled_token: Optional[torch.Tensor] = None,
    initial_state_idx: Optional[torch.Tensor] = None,
    num_computed_tokens: Optional[torch.Tensor] = None,
    block_size_to_align: int = 0,
    metadata: object = None,
    validate_data: bool = False,
) -> torch.Tensor:
    """Torch implementation of vLLM's prefill ``causal_conv1d_fn`` contract.

    ``x`` is ``[dim, tokens]``.  For each sequence the incoming ``width - 1`` convolution
    taps are read from ``conv_states[cache_indices[i]]`` when ``has_initial_state[i]`` is true,
    and the last ``width - 1`` taps are written back to that cache line.  This mirrors the
    vendor kernel's state ownership; APC block bookkeeping is accepted but not used by this
    correctness path.

    The vendor Ascend prefill kernel at the target geometry returns all zeros, which turns the
    following SSD scan into a constant/non-finite state.  Keeping the convolution here makes the
    Mamba2 provider's prefill path complete rather than partially vendor-backed.
    """
    del (null_block_id, block_idx_first_scheduled_token, block_idx_last_scheduled_token,
         initial_state_idx, num_computed_tokens, block_size_to_align, metadata, validate_data)
    if x.dim() != 2:
        raise ValueError(f"causal_conv1d_fn expects x [dim, tokens], got {tuple(x.shape)}")
    dim, total = x.shape
    width = int(weight.shape[1])
    if tuple(weight.shape) != (dim, width):
        raise ValueError(f"weight shape {tuple(weight.shape)} != {(dim, width)}")
    starts = _as_int_list(query_start_loc, device=x.device)
    if len(starts) < 2:
        starts = [0, total]
    cache = _as_int_list(cache_indices, device=x.device) if cache_indices is not None else []
    initial = (
        _as_int_list(has_initial_state, device=x.device)
        if has_initial_state is not None
        else []
    )

    original_dtype = x.dtype
    work_dtype = conv_states.dtype if conv_states is not None else x.dtype
    xw = x.to(work_dtype)
    out = torch.zeros((dim, total), device=x.device, dtype=work_dtype)
    taps = width - 1

    for seq in range(len(starts) - 1):
        start, end = int(starts[seq]), int(starts[seq + 1])
        if end <= start:
            continue
        cache_line = int(cache[seq]) if cache else seq
        if cache_line == int(pad_slot_id) or cache_line < 0:
            continue
        if conv_states is not None and cache_line >= conv_states.shape[0]:
            raise IndexError(f"conv cache line {cache_line} outside {conv_states.shape[0]}")
        use_initial = bool(initial[seq]) if initial else conv_states is not None
        if conv_states is not None and use_initial:
            head = conv_states[cache_line, :, :taps].to(work_dtype)
        else:
            head = torch.zeros((dim, taps), device=x.device, dtype=work_dtype)
        full = torch.cat((head, xw[:, start:end]), dim=1)
        # conv1d is cross-correlation: out[:, t] = sum_k weight[:, k] * full[:, t + k].
        y = F.conv1d(
            full.unsqueeze(0),
            weight.to(work_dtype).unsqueeze(1),
            None if bias is None else bias.to(work_dtype),
            groups=dim,
        ).squeeze(0)
        out[:, start:end] = _apply_activation(y, activation)
        if conv_states is not None and taps > 0:
            conv_states[cache_line, :, :taps] = full[:, -taps:].to(conv_states.dtype)

    return out.to(original_dtype)


def _softplus_clamp(dt: torch.Tensor, dt_bias: torch.Tensor, enabled: bool) -> torch.Tensor:
    if enabled:
        dt = F.softplus(dt + dt_bias)
    else:
        dt = dt + dt_bias
    return torch.clamp(dt, 0.0, float("inf"))


def _selective_state_update_varlen(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    z: Optional[torch.Tensor],
    dt_softplus: bool,
    state_batch_indices: Optional[torch.Tensor],
    dst_state_batch_indices: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    num_accepted_tokens: Optional[torch.Tensor],
    cu_seqlens: torch.Tensor,
) -> None:
    """Speculative-decode update: replay every accepted token per sequence.

    vLLM's verification step passes ``x`` as concatenated multi-token sequences and
    ``num_accepted_tokens`` for the prefix that actually committed.  Ignoring those (the
    pre-MTP implementation did) leaves the SSM state one or more tokens behind, so the next
    speculative step disagrees with the target even though the draft was verified.
    """
    if state.dim() == 5:
        state = state.squeeze(1)
    if state.dim() != 4:
        raise ValueError(f"state must be [slots, heads, headdim, dstate], got {tuple(state.shape)}")
    if x.dim() != 3:
        raise ValueError(f"varlen x must be [tokens, heads, headdim], got {tuple(x.shape)}")
    slots, nheads, headdim, dstate = state.shape
    tokens = x.shape[0]
    ngroups = B.shape[1]
    if nheads % ngroups:
        raise ValueError(f"nheads={nheads} is not divisible by ngroups={ngroups}")
    if out is None:
        out = torch.empty_like(x)
    if z is not None and z.dim() != 3:
        raise ValueError(f"z must be [tokens, heads, headdim], got {tuple(z.shape)}")

    a = A.to(torch.float32)
    if a.dim() == 1:
        a = a[:, None, None].expand(nheads, headdim, dstate)
    elif a.dim() == 2:
        a = a[:, None, :].expand(nheads, headdim, -1)
    elif a.dim() != 3:
        raise ValueError(f"A must be 1-D, 2-D or 3-D, got {tuple(A.shape)}")
    df = D.to(torch.float32)
    if df.dim() == 1:
        df = df[:, None]
    dt_bias_f = dt_bias.to(torch.float32)
    if dt_bias_f.dim() == 1:
        dt_bias_f = dt_bias_f[:, None]
    b_rep = _repeat_groups(B.to(torch.float32), nheads)
    c_rep = _repeat_groups(C.to(torch.float32), nheads)

    starts = _as_int_list(cu_seqlens, device=x.device)
    if len(starts) < 2:
        return
    src = (
        state_batch_indices.reshape(-1).to(torch.int64)
        if state_batch_indices is not None
        else torch.arange(len(starts) - 1, device=state.device, dtype=torch.int64)
    )
    dst = (
        dst_state_batch_indices.reshape(-1).to(torch.int64)
        if dst_state_batch_indices is not None
        else src
    )
    accepted = _as_int_list(num_accepted_tokens, device=x.device) if num_accepted_tokens is not None else []

    for seq in range(len(starts) - 1):
        start, end = int(starts[seq]), int(starts[seq + 1])
        if end <= start:
            continue
        count = end - start
        if accepted:
            count = max(0, min(count, int(accepted[seq])))
        if count == 0:
            continue
        src_slot, dst_slot = int(src[seq]), int(dst[seq])
        if src_slot < 0 or src_slot >= slots or dst_slot < 0 or dst_slot >= slots:
            raise IndexError(f"state slot out of range: src={src_slot}, dst={dst_slot}, slots={slots}")
        current = state[src_slot].to(torch.float32)
        for pos in range(start, start + count):
            dtf = _softplus_clamp(dt[pos].to(torch.float32), dt_bias_f[:, 0], dt_softplus)
            current = (
                current * torch.exp(dtf[:, None, None] * a)
                + dtf[:, None, None] * x[pos].to(torch.float32)[:, :, None] * b_rep[pos][:, None, :]
            )
            y = (current * c_rep[pos][:, None, :]).sum(dim=-1) + x[pos].to(torch.float32) * df
            if z is not None:
                y = y * F.silu(z[pos].to(torch.float32))
            out[pos].copy_(y.to(out.dtype))
        state[dst_slot].copy_(current.to(state.dtype))


def selective_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    z: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    state_batch_indices: Optional[torch.Tensor] = None,
    dst_state_batch_indices: Optional[torch.Tensor] = None,
    null_block_id: int = 0,
    out: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    is_blackwell: bool = False,
) -> None:
    """Torch implementation of the Mamba2 decode/state-update entry point.

    The common decode case is one token per sequence; ``state_batch_indices`` selects the source
    block and ``dst_state_batch_indices`` the destination (they are equal without speculative
    decoding).  Speculative decoding passes ``cu_seqlens`` plus ``num_accepted_tokens``; those
    tokens are replayed in state order so the target state stays exact.  The function is in-place
    on ``state`` and ``out`` like vLLM's kernel.
    """
    del null_block_id, is_blackwell
    if cu_seqlens is not None and x.dim() == 3:
        _selective_state_update_varlen(
            state,
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias,
            z,
            dt_softplus,
            state_batch_indices,
            dst_state_batch_indices,
            out,
            num_accepted_tokens,
            cu_seqlens,
        )
        return
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D.dim() == 1:
        D = D.unsqueeze(0)
    if dt_bias.dim() == 1:
        dt_bias = dt_bias[:, None]
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if out is None:
        out = torch.empty_like(x)

    batch, nheads, headdim = x.shape
    ngroups = B.shape[1]
    dstate = state.shape[-1]
    if nheads % ngroups:
        raise ValueError(f"nheads={nheads} is not divisible by ngroups={ngroups}")
    src = (
        state_batch_indices
        if state_batch_indices is not None
        else torch.arange(batch, device=state.device).unsqueeze(1)
    )
    dst = (
        dst_state_batch_indices
        if dst_state_batch_indices is not None
        else src
    )
    src_flat = src.reshape(-1).to(torch.int64)
    dst_flat = dst.reshape(-1).to(torch.int64)
    df = D.to(torch.float32)
    if df.dim() == 1:
        df = df[:, None]

    b_rep = _repeat_groups(B.to(torch.float32), nheads)
    c_rep = _repeat_groups(C.to(torch.float32), nheads)
    a = A.to(torch.float32)
    if a.dim() == 1:
        a = a[:, None, None].expand(nheads, headdim, dstate)
    elif a.dim() == 2:
        a = a[:, None, :].expand(nheads, headdim, -1)
    dt_bias_f = dt_bias.to(torch.float32)
    if dt_bias_f.shape[-1] == 1:
        dt_bias_f = dt_bias_f.expand(nheads, headdim)

    # Vectorised over the batch: no host round-trip for the state indices.  The previous
    # per-row loop did ``src.to('cpu')``/``int(...)`` and therefore failed ACL graph capture with
    # ``rtMemcpy ... stream is capturing``.  Duplicate destination slots keep the same
    # last-writer-wins semantics as the kernel; ordinary decode has one token per sequence.
    dtf = _softplus_clamp(dt.to(torch.float32), dt_bias_f, dt_softplus)
    xf = x.to(torch.float32)
    state_in = state.index_select(0, src_flat).to(torch.float32)
    decay = torch.exp(dtf[..., None] * a)
    state_out = state_in * decay + dtf[..., None] * xf[..., None] * b_rep[:, :, None, :]
    y = (state_out * c_rep[:, :, None, :]).sum(dim=-1) + xf * df
    if z is not None:
        y = y * F.silu(z.to(torch.float32))
    out.copy_(y.to(out.dtype))
    state.index_copy_(0, dst_flat, state_out.to(state.dtype))
