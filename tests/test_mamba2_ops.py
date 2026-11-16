"""CPU checks for the torch Mamba2 operator fallback.

The target's vendor SSD kernel fails to compile on Ascend; these tests pin the arithmetic the
replacement must preserve (state update, C read-out, D skip, varlen chunk boundaries and decode
slot update) against a scalar per-token recurrence.  They do not need vLLM or a device.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tileinfer.providers.nemotron.mamba2 import (
    causal_conv1d_fn,
    mamba_chunk_scan_combined_varlen,
    selective_state_update,
)


def _scalar_chunk(x, dt, A, B, C, D, dt_bias, state, group_ratio):
    if D.dim() == 1:
        D = D[:, None]
    out = torch.empty_like(x)
    for t in range(x.shape[0]):
        d = torch.nn.functional.softplus(dt[t] + dt_bias)
        state = state * torch.exp(d[:, None, None] * A[:, None, None])
        state = state + d[:, None, None] * x[t][:, :, None] * B[t].repeat_interleave(
            group_ratio, dim=0
        )[:, None, :]
        out[t] = (state * C[t].repeat_interleave(group_ratio, dim=0)[:, None, :]).sum(-1)
        out[t] = out[t] + x[t] * D
    return out, state


def test_mamba2_chunk_scan_matches_scalar_recurrence() -> None:
    torch.manual_seed(7)
    total, heads, hdim, dstate, groups = 13, 4, 3, 5, 2
    x = torch.randn(total, heads, hdim)
    dt = torch.randn(total, heads)
    A = -torch.rand(heads)
    B = torch.randn(total, groups, dstate)
    C = torch.randn(total, groups, dstate)
    D = torch.randn(heads, hdim)
    dt_bias = torch.randn(heads)
    initial = torch.randn(1, heads, hdim, dstate)
    initial_copy = initial.clone()
    out = torch.empty_like(x)

    final = mamba_chunk_scan_combined_varlen(
        x,
        dt,
        A,
        B,
        C,
        chunk_size=4,
        cu_seqlens=torch.tensor([0, total]),
        cu_chunk_seqlens=None,
        last_chunk_indices=None,
        seq_idx=None,
        out=out,
        D=D,
        dt_bias=dt_bias,
        initial_states=initial,
        dt_softplus=True,
    )

    reference, ref_state = _scalar_chunk(x, dt, A, B, C, D, dt_bias, initial_copy[0], heads // groups)
    torch.testing.assert_close(out, reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(final[0], ref_state, rtol=1e-5, atol=1e-6)
    # initial_states is an input owned by the caller; the op must not write into it.
    torch.testing.assert_close(initial, initial_copy)


def test_mamba2_chunk_scan_respects_varlen_boundaries_and_returns_chunk_states() -> None:
    torch.manual_seed(9)
    total, heads, hdim, dstate, groups = 10, 4, 3, 5, 2
    x = torch.randn(total, heads, hdim)
    dt = torch.randn(total, heads)
    A = -torch.rand(heads)
    B = torch.randn(total, groups, dstate)
    C = torch.randn(total, groups, dstate)
    D = torch.randn(heads, hdim)
    dt_bias = torch.randn(heads)
    cu_seqlens = torch.tensor([0, 5, 10])
    cu_chunks = torch.tensor([0, 4, 5, 8, 10])
    seq_idx = torch.tensor([0, 0, 1, 1])
    out = torch.empty_like(x)

    final = mamba_chunk_scan_combined_varlen(
        x,
        dt,
        A,
        B,
        C,
        chunk_size=4,
        cu_seqlens=cu_seqlens,
        cu_chunk_seqlens=cu_chunks,
        last_chunk_indices=torch.tensor([1, 3]),
        seq_idx=seq_idx,
        out=out,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        return_intermediate_states=True,
    )

    reference = torch.empty_like(x)
    chunk_states = []
    for start, end in ((0, 5), (5, 10)):
        state = torch.zeros(heads, hdim, dstate)
        pos = start
        while pos < end:
            stop = min(end, (pos // 4 + 1) * 4)
            part, state = _scalar_chunk(
                x[pos:stop], dt[pos:stop], A, B[pos:stop], C[pos:stop], D, dt_bias, state,
                heads // groups,
            )
            reference[pos:stop] = part
            chunk_states.append(state.clone())
            pos = stop
    torch.testing.assert_close(out, reference, rtol=1e-5, atol=1e-6)
    assert final.shape == (4, heads, hdim, dstate)
    for got, want in zip(final, chunk_states):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def test_mamba2_selective_state_update_matches_one_token_recurrence() -> None:
    torch.manual_seed(11)
    batch, heads, hdim, dstate, groups = 2, 4, 3, 5, 2
    slots = 3
    state = torch.randn(slots, heads, hdim, dstate)
    state_copy = state.clone()
    x = torch.randn(batch, heads, hdim)
    dt = torch.randn(batch, heads, hdim)
    A = -torch.rand(heads)
    B = torch.randn(batch, groups, dstate)
    C = torch.randn(batch, groups, dstate)
    D = torch.randn(heads, hdim)
    dt_bias = torch.randn(heads)
    indices = torch.tensor([[0], [2]])
    out = torch.empty_like(x)

    selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        dt_bias,
        dt_softplus=True,
        state_batch_indices=indices,
        out=out,
    )

    for row in range(batch):
        slot = int(indices[row, 0])
        d = torch.nn.functional.softplus(dt[row] + dt_bias[:, None])
        previous = state_copy[slot]
        expected_state = previous * torch.exp(d[..., None] * A[:, None, None])
        expected_state = expected_state + d[..., None] * x[row][:, :, None] * B[row].repeat_interleave(
            heads // groups, dim=0
        )[:, None, :]
        expected = (expected_state * C[row].repeat_interleave(heads // groups, dim=0)[:, None, :]).sum(
            -1
        ) + x[row] * D
        torch.testing.assert_close(out[row], expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(state[slot], expected_state, rtol=1e-5, atol=1e-6)


def test_mamba2_target_geometry_matches_scalar_recurrence() -> None:
    """Target Mamba2 geometry: 64 heads x 64 head_dim x 128 state x 8 groups, chunk 64.

    This is still a CPU reference check, not target-weight evidence.  It pins the exact
    geometry the Nemotron declaration and the vLLM MambaMixer2 use, including a varlen
    boundary that is not chunk-aligned.
    """
    torch.manual_seed(23)
    total, heads, hdim, dstate, groups = 90, 64, 64, 128, 8
    chunk = 64
    x = torch.randn(total, heads, hdim)
    dt = torch.randn(total, heads)
    A = -torch.rand(heads)
    B = torch.randn(total, groups, dstate)
    C = torch.randn(total, groups, dstate)
    D = torch.randn(heads, hdim)
    dt_bias = torch.randn(heads)
    cu_seqlens = torch.tensor([0, 37, total])
    initial = torch.randn(2, heads, hdim, dstate)
    initial_copy = initial.clone()
    out = torch.empty_like(x)

    final = mamba_chunk_scan_combined_varlen(
        x,
        dt,
        A,
        B,
        C,
        chunk_size=chunk,
        cu_seqlens=cu_seqlens,
        cu_chunk_seqlens=None,
        last_chunk_indices=None,
        seq_idx=None,
        out=out,
        D=D,
        dt_bias=dt_bias,
        initial_states=initial,
        dt_softplus=True,
    )

    reference = torch.empty_like(x)
    ref_states = []
    for seq, (start, end) in enumerate(((0, 37), (37, total))):
        state = initial_copy[seq].clone()
        pos = start
        while pos < end:
            stop = min(end, (pos // chunk + 1) * chunk)
            part, state = _scalar_chunk(
                x[pos:stop],
                dt[pos:stop],
                A,
                B[pos:stop],
                C[pos:stop],
                D,
                dt_bias,
                state,
                heads // groups,
            )
            reference[pos:stop] = part
            pos = stop
        ref_states.append(state)
    torch.testing.assert_close(out, reference, rtol=1e-3, atol=1e-4)
    for got, want in zip(final, ref_states):
        torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-4)


def test_causal_conv1d_prefill_matches_manual_conv_and_updates_state() -> None:
    """The Ascend prefill convolution returns zeros; this pins the torch contract."""
    torch.manual_seed(31)
    dim, width, total = 5, 4, 7
    taps = width - 1
    x = torch.randn(dim, total)
    weight = torch.randn(dim, width)
    bias = torch.randn(dim)
    conv_states = torch.randn(2, dim, taps)
    state_before = conv_states.clone()
    starts = torch.tensor([0, 3, total], dtype=torch.int32)
    cache = torch.tensor([0, 1], dtype=torch.int32)
    has_initial = torch.tensor([True, False])

    out = causal_conv1d_fn(
        x,
        weight,
        bias,
        conv_states,
        starts,
        cache_indices=cache,
        has_initial_state=has_initial,
        activation="silu",
    )

    expected = torch.empty_like(x)
    expected_state = state_before.clone()
    for seq, (start, end) in enumerate(((0, 3), (3, total))):
        line = int(cache[seq])
        head = state_before[line] if bool(has_initial[seq]) else torch.zeros(dim, taps)
        full = torch.cat((head, x[:, start:end]), dim=1)
        y = F.conv1d(full.unsqueeze(0), weight.unsqueeze(1), bias, groups=dim).squeeze(0)
        expected[:, start:end] = y * torch.sigmoid(y)
        expected_state[line] = full[:, -taps:]
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(conv_states, expected_state, rtol=1e-5, atol=1e-6)
    assert out.abs().sum().item() > 0.0


def test_mamba2_selective_state_update_replays_speculative_tokens() -> None:
    """MTP commits multiple accepted tokens; the SSM state must advance through all of them."""
    torch.manual_seed(41)
    heads, hdim, dstate, groups = 2, 2, 3, 1
    slots = 4
    state = torch.randn(slots, heads, hdim, dstate)
    state_before = state.clone()
    lengths = [2, 1]
    total = sum(lengths)
    x = torch.randn(total, heads, hdim)
    dt = torch.randn(total, heads)
    A = -torch.rand(heads)
    B = torch.randn(total, groups, dstate)
    C = torch.randn(total, groups, dstate)
    D = torch.randn(heads, hdim)
    dt_bias = torch.randn(heads)
    src = torch.tensor([0, 2])
    dst = torch.tensor([1, 3])
    cu = torch.tensor([0, sum(lengths[:1]), total], dtype=torch.int32)
    accepted = torch.tensor(lengths, dtype=torch.int32)
    out = torch.empty_like(x)

    selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        dt_bias,
        dt_softplus=True,
        state_batch_indices=src,
        dst_state_batch_indices=dst,
        out=out,
        num_accepted_tokens=accepted,
        cu_seqlens=cu,
    )

    for seq, count in enumerate(lengths):
        current = state_before[int(src[seq])].clone()
        start = int(cu[seq])
        for pos in range(start, start + count):
            d = torch.nn.functional.softplus(dt[pos] + dt_bias)
            current = current * torch.exp(d[:, None, None] * A[:, None, None])
            current = current + d[:, None, None] * x[pos][:, :, None] * B[pos].repeat_interleave(
                heads // groups, dim=0
            )[:, None, :]
            expected = (
                current * C[pos].repeat_interleave(heads // groups, dim=0)[:, None, :]
            ).sum(-1) + x[pos] * D
            torch.testing.assert_close(out[pos], expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(state[int(dst[seq])], current, rtol=1e-5, atol=1e-6)
