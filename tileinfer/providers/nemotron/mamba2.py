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
    "causal_conv1d_update",
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

    ``x`` is ``[dim, tokens]``.  For each sequence the incoming convolution taps are read from
    the cache line selected by ``cache_indices``/``initial_state_idx`` when
    ``has_initial_state[i]`` is true, and the complete sliding window (``state_len`` columns,
    which is ``width - 1 + num_speculative_tokens`` under speculative decoding) is written back.
    """
    del (null_block_id, block_idx_first_scheduled_token, num_computed_tokens,
         block_size_to_align, metadata, validate_data)
    if x.dim() != 2:
        raise ValueError(f"causal_conv1d_fn expects x [dim, tokens], got {tuple(x.shape)}")
    dim, total = x.shape
    width = int(weight.shape[1])
    if tuple(weight.shape) != (dim, width):
        raise ValueError(f"weight shape {tuple(weight.shape)} != {(dim, width)}")
    starts = _as_int_list(query_start_loc, device=x.device)
    if len(starts) < 2:
        starts = [0, total]
    batch = len(starts) - 1
    initial = (
        _as_int_list(has_initial_state, device=x.device)
        if has_initial_state is not None
        else []
    )
    taps = width - 1
    state_len = int(conv_states.shape[-1]) if conv_states is not None else taps
    if state_len < taps:
        raise ValueError(f"conv_state length {state_len} < width-1={taps}")

    # ``cache_indices`` is 1-D for the ordinary path, but a 2-D block table when
    # ``is_mamba_cache_all`` is on (prefix caching / speculative verify).  The initial
    # state is not necessarily row 0 of that table: vLLM gives us the block pointer in
    # ``initial_state_idx``.
    if cache_indices is None:
        read_lines = write_lines = list(range(batch))
    else:
        read_pointer = initial_state_idx
        write_pointer = (
            block_idx_last_scheduled_token
            if block_idx_last_scheduled_token is not None
            else read_pointer
        )
        if cache_indices.dim() == 2 and read_pointer is None:
            read_pointer = torch.zeros(batch, device=cache_indices.device, dtype=torch.int64)
        if cache_indices.dim() == 2 and write_pointer is None:
            write_pointer = torch.zeros(batch, device=cache_indices.device, dtype=torch.int64)
        read_lines = _conv_lines(batch, cache_indices, read_pointer, x.device).tolist()
        write_lines = _conv_lines(batch, cache_indices, write_pointer, x.device).tolist()

    original_dtype = x.dtype
    work_dtype = conv_states.dtype if conv_states is not None else x.dtype
    xw = x.to(work_dtype)
    out = torch.zeros((dim, total), device=x.device, dtype=work_dtype)

    for seq in range(batch):
        start, end = int(starts[seq]), int(starts[seq + 1])
        if end <= start:
            continue
        read_line = int(read_lines[seq])
        write_line = int(write_lines[seq])
        if write_line == int(pad_slot_id) or write_line < 0:
            continue
        if conv_states is not None and (read_line >= conv_states.shape[0] or write_line >= conv_states.shape[0]):
            raise IndexError(
                f"conv cache line outside {conv_states.shape[0]}: read={read_line}, write={write_line}"
            )
        use_initial = bool(initial[seq]) if initial else conv_states is not None
        if conv_states is not None and use_initial:
            head = conv_states[read_line, :, state_len - taps :].to(work_dtype)
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
        if conv_states is not None:
            # Keep the full window, not only the last ``width - 1`` taps: speculative decode
            # indexes into the extra columns with ``num_accepted_tokens``.
            if full.shape[1] >= state_len:
                new_state = full[:, -state_len:]
            else:
                new_state = torch.cat(
                    (torch.zeros((dim, state_len - full.shape[1]), device=x.device, dtype=work_dtype),
                     full),
                    dim=1,
                )
            conv_states[write_line, :, :state_len] = new_state.to(conv_states.dtype)
    return out.to(original_dtype)

def _is_stream_capturing() -> bool:
    """True while an ACL graph is being captured.

    The speculative entry points keep a host-metadata loop for eager execution (it handles
    arbitrary per-sequence lengths), but that loop cannot run inside capture: reading
    ``query_start_loc``/``num_accepted_tokens`` would issue a synchronous D2H copy.  During
    capture vLLM's decode batch has a uniform token count per sequence, so the dedicated
    tensor-only paths below can be captured instead.  On CPU this is always false.
    """
    npu = getattr(torch, "npu", None)
    if npu is None or not hasattr(npu, "is_current_stream_capturing"):
        return False
    try:
        return bool(npu.is_current_stream_capturing())
    except Exception:  # pragma: no cover - defensive, platform dependent
        return False


def _conv_lines(
    batch: int,
    indices: Optional[torch.Tensor],
    pointer: Optional[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Select ``batch`` cache lines without a host round-trip (graph-capture safe)."""
    if indices is None:
        return torch.arange(batch, device=device, dtype=torch.int64)
    if indices.dim() == 1:
        lines = indices[:batch].to(torch.int64)
    else:
        pos = (
            pointer.reshape(-1)[:batch].to(torch.int64)
            if pointer is not None
            else torch.zeros(batch, device=indices.device, dtype=torch.int64)
        )
        lines = indices.gather(1, pos[:, None]).squeeze(1).to(torch.int64)
    if lines.numel() != batch:
        raise ValueError(f"expected {batch} cache lines, got {lines.numel()}")
    return lines


def _write_conv_lines(
    conv_state: torch.Tensor,
    lines: torch.Tensor,
    window: torch.Tensor,
    null_block_id: int,
) -> None:
    """Write one window per sequence to tensor-selected cache lines.

    ``null_block_id`` entries must not be touched.  Keep the existing line instead of using a
    data-dependent host mask: that makes the write capturable and preserves the eager semantics.
    """
    mask = lines != int(null_block_id)
    safe = lines.clamp(min=0).to(torch.int64)
    keep = conv_state.index_select(0, safe).to(torch.float32)
    src = torch.where(mask[:, None, None], window, keep)
    conv_state.index_copy_(0, safe, src.to(conv_state.dtype))


def _causal_conv1d_update_capture(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    activation: object,
    conv_state_indices: Optional[torch.Tensor],
    num_accepted_tokens: Optional[torch.Tensor],
    query_start_loc: torch.Tensor,
    null_block_id: int,
    block_idx_last_scheduled_token: Optional[torch.Tensor],
    initial_state_idx: Optional[torch.Tensor],
) -> torch.Tensor:
    """Tensor-only speculative convolution for ACL graph capture.

    vLLM's captured decode batch has the same number of scheduled tokens for every sequence.
    Reading that number from ``x``/``query_start_loc`` shapes is free; reading the tensor values
    is not (and is exactly what failed with ``rtMemcpy ... stream is capturing``).  The rolling
    window is therefore built for all sequences at once, with per-token scatter to the 2-D
    destination table used by speculative Mamba2.
    """
    if x.dim() != 2:
        raise RuntimeError("captured speculative conv expects 2-D [tokens, dim] input")
    batch = int(query_start_loc.numel()) - 1
    if batch <= 0 or x.shape[0] % batch:
        raise RuntimeError(
            f"captured speculative conv has non-uniform batch: tokens={x.shape[0]}, "
            f"query_start_loc={query_start_loc.numel()}"
        )
    count = x.shape[0] // batch
    dim = x.shape[1]
    taps = weight.shape[1] - 1
    state_len = int(conv_state.shape[-1])
    if taps > state_len:
        raise ValueError(f"conv_state state_len={state_len} < width-1={taps}")

    dst_table: Optional[torch.Tensor]
    if conv_state_indices is None:
        read_lines = torch.arange(batch, device=x.device, dtype=torch.int64)
        dst_table = None
        write_lines = read_lines
    elif conv_state_indices.dim() == 1:
        read_lines = conv_state_indices[:batch].to(torch.int64)
        write_lines = read_lines
        dst_table = None
    else:
        dst_table = conv_state_indices
        pos = (
            initial_state_idx.reshape(-1)[:batch].to(torch.int64)
            if initial_state_idx is not None
            else (
                (num_accepted_tokens.reshape(-1)[:batch].to(torch.int64) - 1).clamp(min=0)
                if num_accepted_tokens is not None
                else torch.zeros(batch, device=x.device, dtype=torch.int64)
            )
        )
        read_lines = dst_table.gather(1, pos[:, None]).squeeze(1).to(torch.int64)
        write_lines = _conv_lines(batch, dst_table, block_idx_last_scheduled_token, x.device)

    hist = conv_state.index_select(0, read_lines).to(torch.float32)
    hist = hist[:, :, state_len - taps :] if taps else hist[:, :, :0]
    x_seq = x.reshape(batch, count, dim).transpose(1, 2).to(torch.float32)
    full = torch.cat((hist, x_seq), dim=2)
    y = F.conv1d(
        full,
        weight.to(torch.float32).unsqueeze(1),
        None if bias is None else bias.to(torch.float32),
        groups=dim,
    )
    y = _apply_activation(y, activation)
    out = torch.empty_like(x, dtype=conv_state.dtype)
    out.copy_(y.transpose(1, 2).reshape(x.shape).to(out.dtype))

    if dst_table is None:
        window = full[:, :, -state_len:]
        _write_conv_lines(conv_state, write_lines, window, null_block_id)
    else:
        cols = dst_table.shape[1]
        if cols < count:
            raise RuntimeError(f"conv destination table has {cols} columns, need {count}")
        for t in range(count):
            window = torch.cat((hist, x_seq[:, :, : t + 1]), dim=2)[:, :, -state_len:]
            _write_conv_lines(conv_state, dst_table[:, t], window, null_block_id)
    return out.to(x.dtype)


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: object = "silu",
    conv_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    max_query_len: int = -1,
    null_block_id: int = 0,
    block_idx_last_scheduled_token: Optional[torch.Tensor] = None,
    initial_state_idx: Optional[torch.Tensor] = None,
    validate_data: bool = False,
) -> torch.Tensor:
    """Torch ``causal_conv1d_update`` including speculative multi-token rolling.

    Shapes are the vLLM contract: ``x`` is ``[tokens, dim]`` with ``query_start_loc``,
    ``[batch, dim]`` for one token per sequence, or ``[batch, dim, seqlen]``.  The cache line is
    selected by ``conv_state_indices`` (optionally through the APC block pointers).

    With speculative decoding the cache holds ``width - 1 + num_spec`` columns.  ``num_accepted``
    selects the history offset (``num_accepted - 1``), but the convolution and the cache must
    still advance through **all** scheduled tokens: vLLM later reads the per-token SSM/conv state
    at the accepted position.  Limiting the loop to accepted tokens (the old implementation) or
    writing only the last ``width - 1`` columns leaves the target state one or more tokens behind.
    """
    del validate_data
    _, width = weight.shape
    taps = width - 1
    state_len = int(conv_state.shape[-1])
    if state_len < taps:
        raise ValueError(f"conv_state state_len={state_len} < width-1={taps}")
    spec = num_accepted_tokens is not None
    # Graph capture cannot read query_start_loc/num_accepted_tokens on the host.  Captured
    # speculative decode has a uniform per-sequence token count, so use the tensor-only path.
    if spec and _is_stream_capturing():
        if query_start_loc is None:
            raise RuntimeError("captured speculative conv requires query_start_loc")
        return _causal_conv1d_update_capture(
            x,
            conv_state,
            weight,
            bias,
            activation,
            conv_state_indices,
            num_accepted_tokens,
            query_start_loc,
            int(null_block_id),
            block_idx_last_scheduled_token,
            initial_state_idx,
        )
    # Graph capture needs a host-sync-free path.  Ordinary decode is one token per sequence with
    # a compact (width-1 column) state; keep that vectorised.  Speculative batches take the
    # rolling-window path below.
    single = False
    if num_accepted_tokens is None and state_len == taps:
        if x.dim() == 2 and query_start_loc is None:
            single = True
            batch = x.shape[0]
            starts = None
        elif x.dim() == 2:
            batch = query_start_loc.numel() - 1
            single = x.shape[0] == batch
            starts = query_start_loc[:-1].to(torch.int64) if single else None
        elif x.dim() == 3 and x.shape[2] == 1:
            single = True
            batch = x.shape[0]
            starts = None
    if single:
        dim = x.shape[1]
        _, cache_dim, cache_len = conv_state.shape
        if cache_dim != dim or cache_len < taps:
            raise ValueError("conv_state shape is incompatible with x/weight")
        read_lines = _conv_lines(
            conv_state_indices.size(0) if conv_state_indices is not None else batch,
            conv_state_indices,
            initial_state_idx,
            conv_state.device,
        )
        write_lines = _conv_lines(
            conv_state_indices.size(0) if conv_state_indices is not None else batch,
            conv_state_indices,
            block_idx_last_scheduled_token,
            conv_state.device,
        )
        head = conv_state[read_lines, :, state_len - taps :].to(torch.float32)
        tokens = x if starts is None else x.index_select(0, starts)
        if x.dim() == 3:
            tokens = tokens[:, :, 0]
        full = torch.cat((head, tokens.to(torch.float32).unsqueeze(-1)), dim=2)
        y = F.conv1d(
            full,
            weight.to(torch.float32).unsqueeze(1),
            None if bias is None else bias.to(torch.float32),
            groups=dim,
        )
        y = _apply_activation(y, activation)
        out = torch.empty_like(x, dtype=conv_state.dtype)
        if starts is None:
            if x.dim() == 3:
                out[:, :, 0] = y[:, :, 0].to(out.dtype)
            else:
                out[:] = y[:, :, 0].to(out.dtype)
        else:
            out[starts] = y[:, :, 0].to(out.dtype)
        if taps:
            conv_state[write_lines, :, state_len - taps :] = full[:, :, -taps:].to(
                conv_state.dtype
            )
        return out.to(x.dtype)

    if x.dim() == 3:
        if query_start_loc is not None:
            raise ValueError("3-D x and query_start_loc are mutually exclusive")
        batch, dim, seqlen = x.shape
        source = x
        qsl = None
    elif x.dim() == 2:
        dim = x.shape[1]
        if query_start_loc is not None:
            starts = _as_int_list(query_start_loc, device=x.device)
            batch = len(starts) - 1
            source = x
            qsl = starts
        else:
            batch, seqlen = x.shape
            source = x.unsqueeze(-1)
            qsl = None
    else:
        raise ValueError(f"x must be 2-D or 3-D, got {tuple(x.shape)}")

    num_cache_lines, cache_dim, state_len = conv_state.shape
    if cache_dim != dim:
        raise ValueError(f"conv_state dim={cache_dim} != x dim={dim}")
    if state_len < taps:
        raise ValueError(f"conv_state state_len={state_len} < width-1={taps}")
    out = torch.empty_like(x, dtype=conv_state.dtype)
    accepted = _as_int_list(num_accepted_tokens, device=x.device) if spec else []
    cache_indices = conv_state_indices
    write_pointer = (
        block_idx_last_scheduled_token.reshape(-1).to(torch.int64)
        if block_idx_last_scheduled_token is not None
        else None
    )
    read_pointer = (
        initial_state_idx.reshape(-1).to(torch.int64)
        if initial_state_idx is not None
        else write_pointer
    )

    # vLLM's varlen kernel keeps the old part of the cache at a constant
    # ``state_len - max_query_len`` width; shorter sequences write a correspondingly shorter
    # effective window.  Using ``state_len - count`` here shifts mixed-length batches.
    if qsl is not None:
        counts = [int(qsl[i + 1]) - int(qsl[i]) for i in range(batch)]
    elif x.dim() == 3:
        counts = [x.shape[2]] * batch
    else:
        # [batch, dim] is one token per sequence.
        counts = [1] * batch
    max_count = int(max_query_len) if int(max_query_len) > 0 else max(counts or [0])
    if max_count > state_len:
        raise ValueError(f"max_query_len={max_count} exceeds conv state_len={state_len}")
    old_count = state_len - max_count

    def _cache_line(seq: int, pointer: Optional[torch.Tensor]) -> int:
        if cache_indices is None:
            return seq
        if cache_indices.dim() == 1:
            return int(cache_indices[seq])
        # Without APC, vLLM passes a 2-D table whose first column is the sequence's state slot;
        # with APC, ``initial_state_idx``/``block_idx_last_scheduled_token`` select the block.
        pos = int(pointer[seq]) if pointer is not None and seq < len(pointer) else 0
        return int(cache_indices[seq, pos])

    for seq in range(batch):
        if qsl is not None:
            start, end = int(qsl[seq]), int(qsl[seq + 1])
            count = end - start
            tokens = source[start:end].transpose(0, 1)
        else:
            start = 0
            tokens = source[seq]
            count = tokens.shape[1]
            end = count
        if count == 0:
            continue
        read_line = _cache_line(seq, read_pointer)
        write_line = _cache_line(seq, write_pointer)
        if read_line == int(null_block_id) or write_line == int(null_block_id):
            continue
        if not (0 <= read_line < num_cache_lines and 0 <= write_line < num_cache_lines):
            raise IndexError(f"conv cache line out of range: read={read_line}, write={write_line}")

        # Read the state that precedes this verify step.  With speculative decoding the state
        # table has one column per scheduled token; ``num_accepted - 1`` is the column that
        # holds the state after the accepted prefix.  The convolution itself always needs the
        # last ``width - 1`` taps, independently of how the cache window is laid out.
        init_col = max(0, int(accepted[seq]) - 1) if spec and accepted else 0
        if cache_indices is None:
            read_line = write_line = seq
        elif cache_indices.dim() == 1:
            read_line = write_line = int(cache_indices[seq])
        else:
            if read_pointer is not None:
                read_line = int(cache_indices[seq, int(read_pointer[seq])])
            else:
                read_line = int(cache_indices[seq, min(init_col, cache_indices.shape[1] - 1)])
            if write_pointer is not None:
                write_line = int(cache_indices[seq, int(write_pointer[seq])])
            else:
                write_line = int(cache_indices[seq, 0])
        if read_line == int(null_block_id) or write_line == int(null_block_id):
            continue
        if not (0 <= read_line < num_cache_lines):
            raise IndexError(f"conv cache read line out of range: {read_line}")
        window = conv_state[read_line].to(torch.float32)
        if window.shape[1] < state_len:
            window = torch.cat(
                (torch.zeros((dim, state_len - window.shape[1]), device=x.device), window), dim=1
            )
        else:
            window = window[:, -state_len:]
        seq_out = []
        x_tokens = tokens.to(torch.float32)
        for t in range(count):
            hist = window[:, -taps:] if taps else window[:, :0]
            full = torch.cat((hist, x_tokens[:, t : t + 1]), dim=1)
            y = F.conv1d(
                full.unsqueeze(0),
                weight.to(torch.float32).unsqueeze(1),
                None if bias is None else bias.to(torch.float32),
                groups=dim,
            ).squeeze(0)
            y_out = _apply_activation(y[:, -1], activation)
            seq_out.append(y_out)
            window = torch.cat((window[:, 1:], x_tokens[:, t : t + 1]), dim=1)
            if cache_indices is not None and cache_indices.dim() == 2:
                dst_col = min(t, cache_indices.shape[1] - 1)
                dst_line = int(cache_indices[seq, dst_col])
                if dst_line != int(null_block_id):
                    if not (0 <= dst_line < num_cache_lines):
                        raise IndexError(f"conv cache destination line out of range: {dst_line}")
                    conv_state[dst_line] = window.to(conv_state.dtype)

        y = torch.stack(seq_out, dim=1).to(out.dtype)
        if qsl is not None:
            out[start:end] = y.transpose(0, 1)
        elif x.dim() == 3:
            out[seq, :, :count] = y
        else:
            out[seq, :] = y[:, 0]
        if cache_indices is not None and cache_indices.dim() == 1:
            if write_line != int(null_block_id):
                if not (0 <= write_line < num_cache_lines):
                    raise IndexError(f"conv cache write line out of range: {write_line}")
                conv_state[write_line] = window.to(conv_state.dtype)
    return out.to(x.dtype)

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
    null_block_id: int,
) -> None:
    """Speculative-decode update with vLLM's per-token state slots.

    For speculative decoding vLLM passes a 2-D ``state_batch_indices`` of shape
    ``[batch, 1 + num_speculative_tokens]``.  ``num_accepted_tokens`` selects the **initial**
    slot (``max(num_accepted - 1, 0)``), while the recurrence must still run over **every**
    scheduled token and write each intermediate state to ``dst_state_batch_indices[seq, i]``.
    The old implementation flattened the table and wrote only one state, so the target read the
    state of the wrong draft token on the next verify step.
    """
    if state.dim() == 5:
        state = state.squeeze(1)
    if state.dim() != 4:
        raise ValueError(f"state must be [slots, heads, headdim, dstate], got {tuple(state.shape)}")
    if x.dim() != 3:
        raise ValueError(f"varlen x must be [tokens, heads, headdim], got {tuple(x.shape)}")
    slots, nheads, headdim, dstate = state.shape
    ngroups = B.shape[1]
    if nheads % ngroups:
        raise ValueError(f"nheads={nheads} is not divisible by ngroups={ngroups}")
    if out is None:
        out = torch.empty_like(x)
    if z is not None and z.dim() != 3:
        raise ValueError(f"z must be [tokens, heads, headdim], got {tuple(z.shape)}")
    if _is_stream_capturing():
        _selective_state_update_capture(
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
            null_block_id,
        )
        return

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
    if dt_bias_f.dim() != 2:
        raise ValueError(f"dt_bias must be 1-D or 2-D, got {tuple(dt_bias.shape)}")
    if dt_bias_f.shape[-1] == 1:
        # The fused kernel receives a per-head bias expanded to [heads, head_dim]; expand the
        # same way here so the batched verify math is identical to the ordinary decode path.
        dt_bias_f = dt_bias_f.expand(nheads, headdim)
    b_rep = _repeat_groups(B.to(torch.float32), nheads)
    c_rep = _repeat_groups(C.to(torch.float32), nheads)

    starts = _as_int_list(cu_seqlens, device=x.device)
    if len(starts) < 2:
        return
    accepted = _as_int_list(num_accepted_tokens, device=x.device) if num_accepted_tokens is not None else []
    src_table = state_batch_indices
    dst_table = dst_state_batch_indices if dst_state_batch_indices is not None else state_batch_indices
    dst_2d = dst_table is not None and dst_table.dim() == 2

    def slot(table: Optional[torch.Tensor], seq: int, token: int) -> int:
        if table is None:
            return seq
        if table.dim() == 1:
            return int(table[seq])
        if token >= table.shape[1]:
            raise IndexError(f"state slot table has {table.shape[1]} columns, need {token + 1}")
        return int(table[seq, token])

    for seq in range(len(starts) - 1):
        start, end = int(starts[seq]), int(starts[seq + 1])
        count = end - start
        if count <= 0:
            continue
        init_token = max(0, int(accepted[seq]) - 1) if accepted else 0
        src_slot = slot(src_table, seq, init_token)
        if src_slot == int(null_block_id):
            current = torch.zeros((nheads, headdim, dstate), device=state.device, dtype=torch.float32)
        else:
            if src_slot < 0 or src_slot >= slots:
                raise IndexError(f"state source slot out of range: {src_slot}, slots={slots}")
            current = state[src_slot].to(torch.float32)
        # The final destination for a 1-D table (ordinary decode) is written once, after the
        # whole sequence; a 2-D table gets one state per scheduled token.
        final_dst = slot(dst_table, seq, 0)
        for offset, pos in enumerate(range(start, end)):
            # Use the exact batch-one formulation of ``selective_state_update``.  A per-token
            # loop with squeezed tensors is algebraically equivalent, but small broadcast /
            # accumulation-order differences can flip near-tie target logits during verify.
            xf1 = x[pos : pos + 1].to(torch.float32)
            dt_t = dt[pos : pos + 1].to(torch.float32)
            if dt_t.dim() == 2:
                dt_t = dt_t.unsqueeze(-1)  # [1, heads, 1] so the bias broadcasts per head
            dtf1 = _softplus_clamp(dt_t, dt_bias_f, dt_softplus)
            b1 = _repeat_groups(B[pos : pos + 1].to(torch.float32), nheads)
            c1 = _repeat_groups(C[pos : pos + 1].to(torch.float32), nheads)
            state_in1 = current.unsqueeze(0)
            decay1 = torch.exp(dtf1[..., None] * a)
            state_out1 = (
                state_in1 * decay1
                + dtf1[..., None] * xf1[..., None] * b1[:, :, None, :]
            )
            y = (state_out1 * c1[:, :, None, :]).sum(dim=-1) + xf1 * df
            current = state_out1[0]
            y = y[0]
            if z is not None:
                y = y * F.silu(z[pos].to(torch.float32))
            out[pos].copy_(y.to(out.dtype))
            if dst_2d:
                dst_slot = slot(dst_table, seq, offset)
                if dst_slot != int(null_block_id):
                    if dst_slot < 0 or dst_slot >= slots:
                        raise IndexError(f"state destination slot out of range: {dst_slot}, slots={slots}")
                    state[dst_slot].copy_(current.to(state.dtype))
        if not dst_2d:
            if final_dst != int(null_block_id):
                if final_dst < 0 or final_dst >= slots:
                    raise IndexError(f"state destination slot out of range: {final_dst}, slots={slots}")
                state[final_dst].copy_(current.to(state.dtype))


def _write_state_slots(
    state: torch.Tensor,
    slots: torch.Tensor,
    value: torch.Tensor,
    null_block_id: int,
) -> None:
    """Scatter states to tensor-selected slots without a host read (capture-safe)."""
    mask = slots != int(null_block_id)
    safe = slots.clamp(min=0).to(torch.int64)
    keep = state.index_select(0, safe).to(torch.float32)
    src = torch.where(mask[:, None, None, None], value, keep)
    state.index_copy_(0, safe, src.to(state.dtype))


def _selective_state_update_capture(
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
    null_block_id: int,
) -> None:
    """Tensor-only speculative SSM update for ACL graph capture.

    Same arithmetic as the eager batch-one loop, but the batch dimension is vectorised and no
    tensor value is read on the host.  vLLM's captured decode batch has a uniform token count;
    the count comes from shapes, the accepted/source/destination slots from device tensors.
    """
    if state.dim() == 5:
        state = state.squeeze(1)
    if state.dim() != 4:
        raise ValueError(f"state must be [slots, heads, headdim, dstate], got {tuple(state.shape)}")
    if x.dim() != 3:
        raise ValueError(f"captured varlen x must be [tokens, heads, headdim], got {tuple(x.shape)}")
    slots, nheads, headdim, dstate = state.shape
    batch = int(cu_seqlens.numel()) - 1
    if batch <= 0 or x.shape[0] % batch:
        raise RuntimeError(
            f"captured speculative state has non-uniform batch: tokens={x.shape[0]}, "
            f"cu_seqlens={cu_seqlens.numel()}"
        )
    count = x.shape[0] // batch
    if B.shape[1] == 0 or nheads % B.shape[1]:
        raise ValueError(f"nheads={nheads} is not divisible by ngroups={B.shape[1]}")
    if out is None:
        out = torch.empty_like(x)

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
    if dt_bias_f.dim() != 2:
        raise ValueError(f"dt_bias must be 1-D or 2-D, got {tuple(dt_bias.shape)}")
    if dt_bias_f.shape[-1] == 1:
        dt_bias_f = dt_bias_f.expand(nheads, headdim)

    src_table = state_batch_indices
    dst_table = dst_state_batch_indices if dst_state_batch_indices is not None else src_table
    if src_table is None:
        src_slots = torch.arange(batch, device=x.device, dtype=torch.int64)
    elif src_table.dim() == 1:
        src_slots = src_table[:batch].to(torch.int64)
    else:
        pos = (
            (num_accepted_tokens.reshape(-1)[:batch].to(torch.int64) - 1).clamp(min=0)
            if num_accepted_tokens is not None
            else torch.zeros(batch, device=x.device, dtype=torch.int64)
        )
        src_slots = src_table.gather(1, pos[:, None]).squeeze(1).to(torch.int64)

    if dst_table is None:
        dst_slots = src_slots
        dst_2d = False
    elif dst_table.dim() == 1:
        dst_slots = dst_table[:batch].to(torch.int64)
        dst_2d = False
    else:
        if dst_table.shape[1] < count:
            raise RuntimeError(f"state destination table has {dst_table.shape[1]} columns, need {count}")
        dst_slots = dst_table[:, :count].to(torch.int64)
        dst_2d = True

    x_r = x.reshape(batch, count, nheads, headdim).to(torch.float32)
    dt_r = dt.reshape(batch, count, *dt.shape[1:]).to(torch.float32)
    if dt_r.dim() == 3:
        dt_r = dt_r.unsqueeze(-1)  # [B, count, heads, 1]
    b_r = B.reshape(batch, count, *B.shape[1:]).to(torch.float32)
    c_r = C.reshape(batch, count, *C.shape[1:]).to(torch.float32)
    z_r = None
    if z is not None:
        z_r = z.reshape(batch, count, *z.shape[1:]).to(torch.float32)
        if z_r.dim() == 3:
            z_r = z_r.unsqueeze(-1)

    src_mask = (src_slots != int(null_block_id))[:, None, None, None]
    state_in = state.index_select(0, src_slots.clamp(min=0)).to(torch.float32) * src_mask
    out_r = out.reshape(batch, count, nheads, headdim)
    for t in range(count):
        dtf = _softplus_clamp(dt_r[:, t], dt_bias_f, dt_softplus)
        b1 = _repeat_groups(b_r[:, t], nheads)
        c1 = _repeat_groups(c_r[:, t], nheads)
        xf = x_r[:, t]
        decay = torch.exp(dtf[..., None] * a)
        state_out = state_in * decay + dtf[..., None] * xf[..., None] * b1[:, :, None, :]
        y = (state_out * c1[:, :, None, :]).sum(dim=-1) + xf * df
        if z_r is not None:
            y = y * F.silu(z_r[:, t])
        out_r[:, t].copy_(y.to(out.dtype))
        state_in = state_out
        if dst_2d:
            _write_state_slots(state, dst_slots[:, t], state_out, null_block_id)
    if not dst_2d:
        _write_state_slots(state, dst_slots, state_in, null_block_id)


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
    del is_blackwell
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
            null_block_id,
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
