"""CPU oracle tests for the torch port of vLLM's ``causal_conv1d`` mamba operators.

The oracle is a hand-written loop over tokens::

    y[t] = act(bias + sum_i weight[:, i] * x[t - (width - 1) + i])   # x[j < 0] == 0
    state_after = last (width - 1) inputs, oldest first

plus the state carried from call to call.  Nothing in this file re-uses
``tileinfer.vllm.ops.fla.causal_conv1d`` to build its own expectation, except the one property that
*is* the contract the engine relies on: T decode calls == 1 prefill call.

Semantics under test are quoted line-by-line from
``reference/vllm-0.23/vllm__model_executor__layers__mamba__ops__causal_conv1d.py`` in the module
docstring of ``tileinfer/vllm/ops/fla/causal_conv1d.py``.
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.vllm.ops.fla.causal_conv1d import (
    NULL_BLOCK_ID,
    PAD_SLOT_ID,
    causal_conv1d_fn,
    causal_conv1d_update,
)

WIDTH = 4  # the GDN case (conv1d kernel size 4 -> 3 taps of state)


# --------------------------------------------------------------------------------------------------
# oracles
# --------------------------------------------------------------------------------------------------


def oracle_conv(x, weight, bias, state, width, use_silu):
    """Token loop over ``y[t] = act(bias + sum_i w[i] * x[t-(width-1)+i])``.

    ``x``: ``[T, dim]``.  ``state``: ``[width-1, dim]`` chronological (row 0 = oldest) or ``None`` for
    "this sequence starts fresh", which is what zero left padding means.  Returns ``(y, new_state)``.
    """
    x = x.to(torch.float64)
    w = weight.to(torch.float64)
    length, dim = x.shape
    taps = width - 1
    if state is None:
        state = torch.zeros(taps, dim, dtype=torch.float64)
    else:
        state = state.to(torch.float64).clone()
    history = [state[j].clone() for j in range(taps)]
    rows = []
    for t in range(length):
        window = history + [x[t]]
        acc = torch.zeros(dim, dtype=torch.float64)
        if bias is not None:
            acc = acc + bias.to(torch.float64)
        for i in range(width):
            acc = acc + w[:, i] * window[i]
        if use_silu:
            acc = acc / (1 + torch.exp(-acc))
        rows.append(acc)
        history = window[1:]
    y = torch.stack(rows) if rows else torch.zeros(0, dim, dtype=torch.float64)
    return y, torch.stack(history)


def oracle_per_sequence(x_tokens, weight, bias, conv_state, lines, lens, has_init, width, use_silu):
    """Run ``oracle_conv`` for every sequence of a packed batch; returns y, state, new_lines."""
    dim = x_tokens.shape[1]
    taps = width - 1
    out = []
    new_state = []
    offset = 0
    for seq, length in enumerate(lens):
        chunk = x_tokens[offset : offset + length]
        offset += length
        line = int(lines[seq])
        init = conv_state[line, :, :taps].transpose(0, 1) if has_init[seq] else None
        y, state = oracle_conv(chunk, weight, bias, init, width, use_silu)
        out.append(y)
        new_state.append(state)
    return torch.cat(out, dim=0), new_state


def mk_state(num_lines, dim, state_len, generator=None):
    """Positive-valued cache so that "untouched" vs "written" is easy to reason about."""
    return torch.rand(num_lines, dim, state_len, generator=generator, dtype=torch.float32) + 0.5


# --------------------------------------------------------------------------------------------------
# causal_conv1d_fn - prefill
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("activation", [None, "silu", "swish"])
@pytest.mark.parametrize("has_init", [False, True])
def test_prefill_matches_the_token_loop_oracle(activation, has_init):
    torch.manual_seed(1)
    dim, length, state_len, num_lines, line = 6, 19, 8, 5, 3
    x = torch.randn(dim, length, dtype=torch.float32)
    weight = torch.randn(dim, WIDTH, dtype=torch.float32)
    bias = torch.randn(dim, dtype=torch.float32)
    conv_states = mk_state(num_lines, dim, state_len)
    padding_before = conv_states[line, :, WIDTH - 1 :].clone()
    init = conv_states[line, :, : WIDTH - 1].transpose(0, 1).clone()

    y = causal_conv1d_fn(
        x,
        weight,
        bias,
        conv_states,
        torch.tensor([0, length], dtype=torch.int32),
        cache_indices=torch.tensor([line], dtype=torch.int32),
        has_initial_state=torch.tensor([has_init]),
        activation=activation,
    )

    ref, ref_state = oracle_conv(
        x.transpose(0, 1),
        weight,
        bias,
        init if has_init else None,
        WIDTH,
        use_silu=activation in ("silu", "swish"),
    )
    assert y.shape == x.shape
    assert y.dtype is x.dtype
    assert not torch.allclose(y, x)  # it really computed something
    torch.testing.assert_close(y.transpose(0, 1), ref.to(torch.float32), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        conv_states[line, :, : WIDTH - 1].transpose(0, 1), ref_state.to(torch.float32), rtol=1e-6, atol=1e-6
    )
    # columns >= width-1 are alignment padding: never read, never written (l.233/277/304 masks)
    assert torch.equal(conv_states[line, :, WIDTH - 1 :], padding_before)


def test_prefill_zero_left_padding_first_tokens_only_see_earlier_tokens():
    torch.manual_seed(2)
    dim, length, line = 3, 5, 2
    x = torch.randn(dim, length, dtype=torch.float32)
    weight = torch.randn(dim, WIDTH, dtype=torch.float32)
    bias = torch.randn(dim, dtype=torch.float32)
    conv_states = mk_state(4, dim, WIDTH - 1)

    y = causal_conv1d_fn(
        x,
        weight,
        bias,
        conv_states,
        torch.tensor([0, length], dtype=torch.int32),
        cache_indices=torch.tensor([line], dtype=torch.int32),
        has_initial_state=torch.tensor([False]),
        activation=None,
    )
    # t = 0 sees only x[0] (tap width-1); no wrap-around from the cache line.
    torch.testing.assert_close(y[:, 0], bias + weight[:, WIDTH - 1] * x[:, 0], rtol=1e-5, atol=1e-5)
    # t = 1 sees x[0], x[1]
    expect1 = bias + weight[:, WIDTH - 2] * x[:, 0] + weight[:, WIDTH - 1] * x[:, 1]
    torch.testing.assert_close(y[:, 1], expect1, rtol=1e-5, atol=1e-5)
    # t = 2 sees x[0], x[1], x[2] through taps w1, w2, w3 (tap w0 would need x[-1] == 0)
    expect2 = bias + weight[:, 1] * x[:, 0] + weight[:, 2] * x[:, 1] + weight[:, 3] * x[:, 2]
    torch.testing.assert_close(y[:, 2], expect2, rtol=1e-5, atol=1e-5)
    # from t >= 3 the full 4-tap filter is used
    expect3 = (
        bias
        + weight[:, 0] * x[:, 0]
        + weight[:, 1] * x[:, 1]
        + weight[:, 2] * x[:, 2]
        + weight[:, 3] * x[:, 3]
    )
    torch.testing.assert_close(y[:, 3], expect3, rtol=1e-5, atol=1e-5)


def test_prefill_varlen_two_sequences_use_their_own_cache_lines():
    torch.manual_seed(3)
    dim, state_len, lens = 4, WIDTH - 1, [3, 7]
    lines = [1, 3]
    total = sum(lens)
    x = torch.randn(dim, total, dtype=torch.float32)
    weight = torch.randn(dim, WIDTH, dtype=torch.float32)
    bias = torch.randn(dim, dtype=torch.float32)
    conv_states = mk_state(5, dim, state_len)
    init0 = conv_states[lines[0], :, : WIDTH - 1].transpose(0, 1).clone()
    init1 = conv_states[lines[1], :, : WIDTH - 1].transpose(0, 1).clone()
    untouched = {0: conv_states[0].clone(), 2: conv_states[2].clone(), 4: conv_states[4].clone()}

    y = causal_conv1d_fn(
        x,
        weight,
        bias,
        conv_states,
        torch.tensor([0, lens[0], total], dtype=torch.int32),
        cache_indices=torch.tensor(lines, dtype=torch.int32),
        has_initial_state=torch.tensor([True, False]),
        activation="silu",
    )

    ref0, st0 = oracle_conv(x[:, : lens[0]].transpose(0, 1), weight, bias, init0, WIDTH, True)
    ref1, st1 = oracle_conv(x[:, lens[0] :].transpose(0, 1), weight, bias, None, WIDTH, True)
    torch.testing.assert_close(y[:, : lens[0]].transpose(0, 1), ref0.to(torch.float32), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(y[:, lens[0] :].transpose(0, 1), ref1.to(torch.float32), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        conv_states[lines[0], :, : WIDTH - 1].transpose(0, 1), st0.to(torch.float32), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        conv_states[lines[1], :, : WIDTH - 1].transpose(0, 1), st1.to(torch.float32), rtol=1e-6, atol=1e-6
    )
    for line, before in untouched.items():
        assert torch.equal(conv_states[line], before), f"cache line {line} must not be touched"


def test_prefill_shorter_than_the_state_window_mixes_history_into_the_state():
    # T = 2 < width-1 = 3: new state is concat(old_state, x)[-3:] = [old2, x0, x1]  (kernel l.237-280)
    torch.manual_seed(4)
    dim, line = 3, 1
    x = torch.randn(dim, 2, dtype=torch.float32)
    weight = torch.randn(dim, WIDTH, dtype=torch.float32)
    bias = torch.zeros(dim, dtype=torch.float32)
    conv_states = mk_state(3, dim, WIDTH - 1)
    old = conv_states[line, :, : WIDTH - 1].transpose(0, 1).clone()

    causal_conv1d_fn(
        x,
        weight,
        bias,
        conv_states,
        torch.tensor([0, 2], dtype=torch.int32),
        cache_indices=torch.tensor([line], dtype=torch.int32),
        has_initial_state=torch.tensor([True]),
        activation=None,
    )
    expect = torch.cat([old, x.transpose(0, 1)], dim=0)[-(WIDTH - 1) :]

    torch.testing.assert_close(conv_states[line, :, : WIDTH - 1].transpose(0, 1), expect)


def test_prefill_empty_sequence_without_initial_state_zeroes_the_cache_line():
    # concat(zeros, nothing)[-(width-1):] == zeros; the kernel's `loaded_x` mask does the same (l.293-298)
    torch.manual_seed(5)
    dim, line = 2, 1
    conv_states = mk_state(3, dim, WIDTH - 1)
    weight = torch.randn(dim, WIDTH, dtype=torch.float32)
    x = torch.randn(dim, 0, dtype=torch.float32)
    y = causal_conv1d_fn(
        x,
        weight,
        None,
        conv_states,
        torch.tensor([0, 0], dtype=torch.int32),
        cache_indices=torch.tensor([line], dtype=torch.int32),
        has_initial_state=torch.tensor([False]),
        activation=None,
    )
    assert y.shape == (dim, 0)
    assert torch.count_nonzero(conv_states[line, :, : WIDTH - 1]) == 0


def test_prefill_state_is_updated_in_place():
    torch.manual_seed(6)
    dim, length, line = 3, 6, 2
    conv_states = mk_state(4, dim, WIDTH - 1)
    pointer = conv_states.data_ptr()
    weight = torch.randn(dim, WIDTH, dtype=torch.float32)
    x = torch.randn(dim, length, dtype=torch.float32)
    causal_conv1d_fn(
        x,
        weight,
        None,
        conv_states,
        torch.tensor([0, length], dtype=torch.int32),
        cache_indices=torch.tensor([line], dtype=torch.int32),
        has_initial_state=torch.tensor([False]),
        activation=None,
    )
    assert conv_states.data_ptr() == pointer  # same storage, mutated in place
    torch.testing.assert_close(
        conv_states[line, :, : WIDTH - 1].transpose(0, 1), x[:, -(WIDTH - 1) :].transpose(0, 1)
    )


def test_prefill_skips_null_block_cache_lines():
    # upstream: `if conv_states_input_coord == null_block_id: return` (l.136-139); NULL_BLOCK_ID == 0
    torch.manual_seed(7)
    dim, length, line = 3, 5, 2
    conv_states = mk_state(4, dim, WIDTH - 1)
    null_before = conv_states[NULL_BLOCK_ID].clone()
    real_before = conv_states[line].clone()
    weight = torch.randn(dim, WIDTH, dtype=torch.float32)
    x = torch.randn(dim, 2 * length, dtype=torch.float32)

    y = causal_conv1d_fn(
        x,
        weight,
        None,
        conv_states,
        torch.tensor([0, length, 2 * length], dtype=torch.int32),
        cache_indices=torch.tensor([NULL_BLOCK_ID, line], dtype=torch.int32),
        has_initial_state=torch.tensor([True, True]),
        activation=None,
    )
    assert torch.count_nonzero(y[:, :length]) == 0  # skipped rows are zeros here (garbage upstream)
    assert torch.equal(conv_states[NULL_BLOCK_ID], null_before)
    assert not torch.equal(conv_states[line], real_before)  # the real sequence was processed

    # null_block_id=None disables the check, i.e. line 0 becomes usable
    conv_states2 = mk_state(4, dim, WIDTH - 1)
    y2 = causal_conv1d_fn(
        x[:, :length],
        weight,
        None,
        conv_states2,
        torch.tensor([0, length], dtype=torch.int32),
        cache_indices=torch.tensor([NULL_BLOCK_ID], dtype=torch.int32),
        has_initial_state=torch.tensor([True]),
        activation=None,
        null_block_id=None,
    )
    assert torch.count_nonzero(y2) != 0


def test_prefill_accepts_and_ignores_metadata_and_pad_slot_id():
    # metadata only carries the Triton launch plan (l.546-550, l.663-687); pad_slot_id only marks
    # unused programs (l.81-82).  Neither may change the numbers.
    torch.manual_seed(8)
    dim, length, line = 3, 4, 1
    weight = torch.randn(dim, WIDTH, dtype=torch.float32)
    x = torch.randn(dim, length, dtype=torch.float32)
    qsl = torch.tensor([0, length], dtype=torch.int32)
    cache = torch.tensor([line], dtype=torch.int32)
    has = torch.tensor([False])

    s1 = mk_state(3, dim, WIDTH - 1)
    s2 = s1.clone()
    y1 = causal_conv1d_fn(x, weight, None, s1, qsl, cache_indices=cache, has_initial_state=has, activation="silu")
    y2 = causal_conv1d_fn(
        x,
        weight,
        None,
        s2,
        qsl,
        cache_indices=cache,
        has_initial_state=has,
        activation="silu",
        metadata=object(),
        pad_slot_id=PAD_SLOT_ID,
    )
    assert torch.equal(y1, y2)
    assert torch.equal(s1, s2)


def test_prefill_returns_a_fresh_tensor_and_leaves_x_alone():
    torch.manual_seed(9)
    dim, length = 3, 4
    conv_states = mk_state(2, dim, WIDTH - 1)
    x = torch.randn(dim, length, dtype=torch.float32)
    x_before = x.clone()
    y = causal_conv1d_fn(
        x,
        torch.randn(dim, WIDTH),
        None,
        conv_states,
        torch.tensor([0, length], dtype=torch.int32),
        cache_indices=torch.tensor([1], dtype=torch.int32),
        has_initial_state=torch.tensor([False]),
        activation=None,
    )
    assert y.data_ptr() != x.data_ptr()
    assert torch.equal(x, x_before)


# --------------------------------------------------------------------------------------------------
# causal_conv1d_update - decode
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("activation", [None, "silu", True, False])
def test_update_single_token_matches_the_oracle(activation):
    torch.manual_seed(10)
    dim, batch, state_len = 5, 3, WIDTH - 1
    conv_state = mk_state(batch + 2, dim, state_len)
    weight = torch.randn(dim, WIDTH, dtype=torch.float32)
    bias = torch.randn(dim, dtype=torch.float32)
    x = torch.randn(batch, dim, dtype=torch.float32)
    indices = torch.tensor([1, 3, 4], dtype=torch.int32)
    init = conv_state[indices.tolist(), :, : WIDTH - 1].transpose(1, 2).clone()

    out = causal_conv1d_update(
        x, conv_state, weight, bias, activation, conv_state_indices=indices
    )
    use_silu = activation is True or activation == "silu"
    assert out.shape == x.shape and out.dtype is x.dtype
    for b in range(batch):
        ref_b, state_b = oracle_conv(x[b : b + 1], weight, bias, init[b], WIDTH, use_silu)
        # out[b] is [dim] (batch axis, one token) and the oracle row is [1, dim]
        torch.testing.assert_close(out[b], ref_b[0].to(torch.float32), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            conv_state[int(indices[b]), :, : WIDTH - 1].transpose(0, 1),
            state_b.to(torch.float32),
            rtol=1e-6,
            atol=1e-6,
        )


def test_update_writes_the_new_state_in_place_and_keeps_padding_columns():
    torch.manual_seed(11)
    dim, line, state_len = 3, 2, 8  # state_len > width-1: trailing columns must survive
    conv_state = mk_state(4, dim, state_len)
    padding_before = conv_state[line, :, WIDTH - 1 :].clone()
    pointer = conv_state.data_ptr()
    old = conv_state[line, :, : WIDTH - 1].clone()
    weight = torch.randn(dim, WIDTH, dtype=torch.float32)
    x = torch.randn(1, dim, dtype=torch.float32)

    causal_conv1d_update(
        x, conv_state, weight, None, None, conv_state_indices=torch.tensor([line], dtype=torch.int32)
    )
    assert conv_state.data_ptr() == pointer
    # shift-left by one and append x: [old1, old2, x]
    expect = torch.cat([old[:, 1:], x.transpose(0, 1)], dim=1)
    assert torch.equal(conv_state[line, :, : WIDTH - 1], expect)
    assert torch.equal(conv_state[line, :, WIDTH - 1 :], padding_before)


def test_update_token_by_token_equals_one_prefill_over_the_whole_sequence():
    """The property the engine relies on: decode step-by-step == prefill in one shot.

    Layout note (this is why the two sides are aligned with a transpose, not a reshape): the two
    upstream entry points take *different* layouts, and the port keeps both.
      * ``causal_conv1d_fn``: ``x_ptr  # (dim, cu_seqlen)`` (l.18), channel-last with the channel axis
        innermost (``is_channel_last = (x.stride(0) == 1) & (x.stride(1) > 1)``, l.563), i.e. the
        result is ``[dim, T]``.
      * ``causal_conv1d_update``: ``[batch, dim]`` / ``[batch, dim, seqlen]`` / ``[num_tokens, dim]``
        (l.1085-1090), i.e. the result is token-major ``[T, dim]`` for a one-token decode step.
    Comparing them therefore requires ``full.transpose(0, 1)``; the values must agree exactly.
    """
    torch.manual_seed(12)
    dim, length, line = 5, 12, 2
    weight = torch.randn(dim, WIDTH, dtype=torch.float32)
    bias = torch.randn(dim, dtype=torch.float32)
    tokens = torch.randn(dim, length, dtype=torch.float32)
    conv_states = mk_state(4, dim, WIDTH - 1)
    prefill_states = conv_states.clone()
    decode_states = conv_states.clone()
    qsl = torch.tensor([0, length], dtype=torch.int32)
    cache = torch.tensor([line], dtype=torch.int32)
    has = torch.tensor([True])

    for activation in ("silu", None):
        full = causal_conv1d_fn(
            tokens,
            weight,
            bias,
            prefill_states,
            qsl,
            cache_indices=cache,
            has_initial_state=has,
            activation=activation,
        )
        assert full.shape == (dim, length)  # channel-first, as the kernel declares (l.18)
        stepwise = []
        for t in range(length):
            # one decode step = x of shape [batch, dim] with batch == 1 (l.1085-1090)
            out = causal_conv1d_update(
                tokens[:, t].unsqueeze(0),
                decode_states,
                weight,
                bias,
                activation,
                conv_state_indices=cache,
            )
            assert out.shape == (1, dim)  # token-major, not channel-first
            stepwise.append(out[0])
        stepwise = torch.stack(stepwise, dim=0)  # [T, dim]
        torch.testing.assert_close(stepwise, full.transpose(0, 1), rtol=0, atol=1e-6)
        # ... and the cache lines agree exactly (no arithmetic, just a sliding window)
        assert torch.equal(decode_states[line], prefill_states[line])

    # the state also matches the oracle, i.e. the walk above really consumed the same history
    _, ref_state = oracle_conv(tokens.transpose(0, 1), weight, bias, None, WIDTH, use_silu=False)
    decoded = decode_states[line, :, : WIDTH - 1].transpose(0, 1)
    torch.testing.assert_close(decoded, ref_state.to(torch.float32), rtol=1e-6, atol=1e-6)


def test_update_carries_the_state_between_calls():
    torch.manual_seed(13)
    dim, line = 4, 1
    weight = torch.randn(dim, WIDTH, dtype=torch.float32)
    conv_state = mk_state(2, dim, WIDTH - 1)
    x = torch.randn(dim, 6, dtype=torch.float32)
    cache = torch.tensor([line], dtype=torch.int32)
    init = conv_state[line, :, : WIDTH - 1].transpose(0, 1).clone()

    outs = [
        causal_conv1d_update(
            x[:, t].unsqueeze(0), conv_state, weight, None, None, conv_state_indices=cache
        ).squeeze(0)
        for t in range(6)
    ]
    ref, ref_state = oracle_conv(x.transpose(0, 1), weight, None, init, WIDTH, use_silu=False)
    for t in range(6):
        torch.testing.assert_close(outs[t], ref[t].to(torch.float32), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        conv_state[line, :, : WIDTH - 1].transpose(0, 1), ref_state.to(torch.float32), rtol=1e-6, atol=1e-6
    )


def test_update_varlen_two_sequences_match_the_oracle():
    torch.manual_seed(14)
    dim, length, lines = 3, 3, [1, 3]
    total = length * len(lines)
    conv_state = mk_state(5, dim, WIDTH - 1)
    weight = torch.randn(dim, WIDTH, dtype=torch.float32)
    bias = torch.randn(dim, dtype=torch.float32)
    x = torch.randn(total, dim, dtype=torch.float32)
    indices = torch.tensor(lines, dtype=torch.int32)
    init = conv_state[lines, :, : WIDTH - 1].transpose(1, 2).clone()

    out = causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        "silu",
        conv_state_indices=indices,
        query_start_loc=torch.tensor([0, length, total], dtype=torch.int32),
        max_query_len=length,
    )
    assert out.shape == x.shape
    for seq in range(len(lines)):
        chunk = x[seq * length : (seq + 1) * length]
        ref, ref_state = oracle_conv(chunk, weight, bias, init[seq], WIDTH, use_silu=True)
        torch.testing.assert_close(out[seq * length : (seq + 1) * length], ref.to(torch.float32), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            conv_state[lines[seq], :, : WIDTH - 1].transpose(0, 1),
            ref_state.to(torch.float32),
            rtol=1e-6,
            atol=1e-6,
        )


def test_update_varlen_shorter_sequence_trims_only_the_state_write():
    """Faithful upstream quirk, pinned so a "cleanup" cannot change numbers silently.

    `state_len = state_len - (seqlen - (query_end - query_start))` (l.823-824) trims the *write* window
    to the actual length, while the read of the prior columns (l.864-879) is never trimmed.  A
    non-spec varlen batch with unequal lengths therefore reads a full `width-1` window but shifts only
    `width-1-(max_len-len)` columns.  GDN only uses the varlen form together with
    `num_accepted_tokens` (l.1344-1357), where the trim exactly compensates the spec-decoding growth.
    """
    torch.manual_seed(15)
    dim, width, max_len, line = 3, WIDTH, 3, 2
    taps = width - 1
    conv_state = mk_state(4, dim, taps)
    weight = torch.randn(dim, width, dtype=torch.float32)
    old = conv_state[line, :, :taps].clone()
    x = torch.randn(1 + max_len, dim, dtype=torch.float32)  # seq0: 1 token, seq1: 3 tokens
    indices = torch.tensor([line, line + 1], dtype=torch.int32)

    out = causal_conv1d_update(
        x,
        conv_state,
        weight,
        None,
        None,
        conv_state_indices=indices,
        query_start_loc=torch.tensor([0, 1, 4], dtype=torch.int32),
        max_query_len=max_len,
    )
    # y of the 1-token sequence uses the full stale window state[0:3] (read is not trimmed)
    ref, _ = oracle_conv(x[:1], weight, None, old.transpose(0, 1), width, use_silu=False)
    torch.testing.assert_close(out[:1], ref.to(torch.float32), rtol=1e-5, atol=1e-5)
    # but only state_len_eff = taps - (max_len - 1) = 1 column is rewritten: x[0] itself
    expect_col0 = x[0]
    torch.testing.assert_close(conv_state[line, :, 0], expect_col0)
    assert torch.equal(conv_state[line, :, 1:taps], old[:, 1:taps])  # stale, by (upstream) design


def test_update_spec_decoding_grows_the_window():
    """`num_accepted_tokens` -> size `width-2+seqlen`, `[history_{2+a} .. history_M, draft..]` (l.837-852).

    Window arithmetic: read at column ``a-1``, then store ``old[a : a+width-2] ++ x`` from column 0.
    """
    torch.manual_seed(16)
    dim, width, seqlen, state_len = 3, WIDTH, 3, 8
    taps = width - 1
    conv_state = mk_state(5, dim, state_len)
    weight = torch.randn(dim, width, dtype=torch.float32)
    bias = torch.randn(dim, dtype=torch.float32)
    accepted = [1, 3]
    indices = torch.tensor([1, 3], dtype=torch.int32)
    x = torch.randn(2 * seqlen, dim, dtype=torch.float32)
    old = conv_state[indices.tolist()].clone()  # [2, dim, state_len]

    out = causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        "silu",
        conv_state_indices=indices,
        num_accepted_tokens=torch.tensor(accepted, dtype=torch.int32),
        query_start_loc=torch.tensor([0, seqlen, 2 * seqlen], dtype=torch.int32),
        max_query_len=seqlen,
    )

    for seq in range(2):
        offset = accepted[seq] - 1
        prior = old[seq, :, offset : offset + taps].transpose(0, 1)  # read window
        ref, _ = oracle_conv(
            x[seq * seqlen : (seq + 1) * seqlen], weight, bias, prior, width, use_silu=True
        )
        torch.testing.assert_close(
            out[seq * seqlen : (seq + 1) * seqlen], ref.to(torch.float32), rtol=1e-4, atol=1e-4
        )
        # new state = last (width-2) still-live history entries ++ all seqlen drafts
        grow = width - 2 + seqlen
        expect = torch.cat(
            [old[seq, :, offset + 1 : offset + 1 + (width - 2)].transpose(0, 1),
             x[seq * seqlen : (seq + 1) * seqlen]],
            dim=0,
        )
        assert expect.shape[0] == grow
        torch.testing.assert_close(conv_state[int(indices[seq]), :, :grow].transpose(0, 1), expect)
        # columns past the grown window are untouched
        assert torch.equal(
            conv_state[int(indices[seq]), :, grow:], old[seq, :, grow:]
        )


def test_update_returns_a_fresh_tensor_and_leaves_x_alone():
    # deviation from upstream (`out = x`, l.1163), documented in the module docstring: the caller only
    # reads the return value, and keeping the input intact is strictly safer.
    torch.manual_seed(17)
    dim = 3
    conv_state = mk_state(2, dim, WIDTH - 1)
    weight = torch.randn(dim, WIDTH, dtype=torch.float32)
    x = torch.randn(2, dim, dtype=torch.float32)
    before = x.clone()
    out = causal_conv1d_update(
        x, conv_state, weight, None, None, conv_state_indices=torch.tensor([1, 1], dtype=torch.int32)
    )
    assert torch.equal(x, before)
    assert out.data_ptr() != x.data_ptr()


def test_update_skips_null_block_cache_lines():
    torch.manual_seed(18)
    dim, line = 3, 2
    conv_state = mk_state(4, dim, WIDTH - 1)
    null_before = conv_state[NULL_BLOCK_ID].clone()
    weight = torch.randn(dim, WIDTH, dtype=torch.float32)
    x = torch.randn(2, dim, dtype=torch.float32)
    out = causal_conv1d_update(
        x,
        conv_state,
        weight,
        None,
        None,
        conv_state_indices=torch.tensor([NULL_BLOCK_ID, line], dtype=torch.int32),
    )
    assert torch.equal(out[0], x[0])  # skipped rows keep the input (upstream aliases out = x)
    assert torch.equal(conv_state[NULL_BLOCK_ID], null_before)
    assert not torch.equal(out[1], x[1])


def test_update_multi_token_3d_input_matches_the_oracle():
    torch.manual_seed(19)
    dim, batch, seqlen = 4, 2, 3
    conv_state = mk_state(4, dim, WIDTH - 1)
    weight = torch.randn(dim, WIDTH, dtype=torch.float32)
    bias = torch.randn(dim, dtype=torch.float32)
    x = torch.randn(batch, dim, seqlen, dtype=torch.float32)
    indices = torch.tensor([1, 2], dtype=torch.int32)
    init = conv_state[[1, 2], :, : WIDTH - 1].transpose(1, 2).clone()

    out = causal_conv1d_update(
        x, conv_state, weight, bias, "silu", conv_state_indices=indices
    )
    assert out.shape == x.shape
    for b in range(batch):
        ref, ref_state = oracle_conv(x[b].transpose(0, 1), weight, bias, init[b], WIDTH, use_silu=True)
        torch.testing.assert_close(out[b].transpose(0, 1), ref.to(torch.float32), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            conv_state[int(indices[b]), :, : WIDTH - 1].transpose(0, 1),
            ref_state.to(torch.float32),
            rtol=1e-6,
            atol=1e-6,
        )


# --------------------------------------------------------------------------------------------------
# dtype, layout, error handling
# --------------------------------------------------------------------------------------------------


def test_dtype_preservation():
    torch.manual_seed(20)
    dim, length, line = 3, 5, 1
    weight = torch.randn(dim, WIDTH, dtype=torch.float32)
    qsl = torch.tensor([0, length], dtype=torch.int32)
    cache = torch.tensor([line], dtype=torch.int32)
    has = torch.tensor([False])

    # fp32 in -> fp32 out; bf16 in with an fp32 cache -> bf16 out computed through the cache dtype
    base_states = mk_state(3, dim, WIDTH - 1)
    x32 = torch.randn(dim, length, dtype=torch.float32)
    x_bf16 = x32.to(torch.bfloat16)
    ref_states = base_states.clone()
    ref = causal_conv1d_fn(x32, weight, None, ref_states, qsl, cache_indices=cache, has_initial_state=has, activation=None)
    assert ref.dtype is torch.float32

    bf16_states = base_states.clone()
    y = causal_conv1d_fn(
        x_bf16, weight, None, bf16_states, qsl, cache_indices=cache, has_initial_state=has, activation=None
    )
    assert y.dtype is torch.bfloat16
    assert bf16_states.dtype is torch.float32  # the cache keeps its dtype
    # compute path: x is widened to the cache dtype (l.544), so the bf16-rounded input is what counts
    ref_through_bf16 = causal_conv1d_fn(
        x_bf16.float(), weight, None, base_states.clone(), qsl, cache_indices=cache, has_initial_state=has, activation=None
    )
    assert torch.equal(y, ref_through_bf16.to(torch.bfloat16))
    # the fp32 cache stores the bf16-rounded tokens (x is cast to the cache dtype before the kernels)
    torch.testing.assert_close(
        bf16_states[line, :, : WIDTH - 1].transpose(0, 1),
        x_bf16.float()[:, -(WIDTH - 1) :].transpose(0, 1),
    )

    # a bf16 cache stays bf16 and rounds the stored state
    bf_states = base_states.to(torch.bfloat16)
    y2 = causal_conv1d_fn(
        x32, weight, None, bf_states, qsl, cache_indices=cache, has_initial_state=has, activation=None
    )
    assert y2.dtype is torch.float32  # back to the input dtype (l.745)
    assert bf_states.dtype is torch.bfloat16
    torch.testing.assert_close(y2, ref, rtol=2e-2, atol=2e-2)

    # update: same contract
    conv_state = mk_state(3, dim, WIDTH - 1)
    conv_state32 = conv_state.clone()
    x1 = torch.randn(1, dim, dtype=torch.bfloat16)
    o1 = causal_conv1d_update(x1, conv_state, weight, None, None, conv_state_indices=torch.tensor([line], dtype=torch.int32))
    assert o1.dtype is torch.bfloat16 and o1.shape == (1, dim)
    o32 = causal_conv1d_update(
        x1.float(), conv_state32, weight, None, None, conv_state_indices=torch.tensor([line], dtype=torch.int32)
    )
    assert o32.dtype is torch.float32
    assert torch.equal(o1, o32.to(torch.bfloat16))
    assert torch.equal(conv_state, conv_state32)  # the fp32 cache stores the bf16-rounded token


def test_non_contiguous_channel_last_and_transposed_cache_work():
    # the GDN caller hands in `mixed_qkv.transpose(0, 1)` and can hand in a `transpose(-1, -2)` cache
    # (l.1311-1315), so neither tensor may be assumed contiguous.
    torch.manual_seed(21)
    dim, length, line = 4, 5, 2
    tokens = torch.randn(length, dim, dtype=torch.float32)
    x = tokens.transpose(0, 1)
    assert x.stride(0) == 1 and x.stride(1) > 1  # channel-last, like upstream's assertion (l.563/620)
    weight = torch.randn(dim, WIDTH, dtype=torch.float32)
    wide = mk_state(4, dim, 2 * (WIDTH - 1))
    conv_states = wide[:, :, : WIDTH - 1]  # a sliced view, like the caller's transpose(-1, -2)
    assert not conv_states.is_contiguous()

    y = causal_conv1d_fn(
        x,
        weight,
        None,
        conv_states,
        torch.tensor([0, length], dtype=torch.int32),
        cache_indices=torch.tensor([line], dtype=torch.int32),
        has_initial_state=torch.tensor([False]),
        activation=None,
    )
    assert y.shape == x.shape
    assert y.stride(0) == 1  # still channel-last, so the caller's transpose(0, 1) stays a view
    ref, _ = oracle_conv(tokens, weight, None, None, WIDTH, use_silu=False)
    torch.testing.assert_close(y.transpose(0, 1), ref.to(torch.float32), rtol=1e-5, atol=1e-5)


def test_rejects_prefix_caching_arguments():
    dim, length = 3, 2
    conv_states = mk_state(2, dim, WIDTH - 1)
    weight = torch.randn(dim, WIDTH, dtype=torch.float32)
    x = torch.randn(dim, length, dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="prefix-caching"):
        causal_conv1d_fn(
            x,
            weight,
            None,
            conv_states,
            torch.tensor([0, length], dtype=torch.int32),
            cache_indices=torch.tensor([1], dtype=torch.int32),
            has_initial_state=torch.tensor([False]),
            block_idx_last_scheduled_token=torch.tensor([0], dtype=torch.int32),
        )
    with pytest.raises(NotImplementedError, match="prefix-caching"):
        causal_conv1d_update(
            torch.randn(1, dim),
            mk_state(2, dim, WIDTH - 1),
            weight,
            initial_state_idx=torch.tensor([0], dtype=torch.int32),
            conv_state_indices=torch.tensor([1], dtype=torch.int32),
        )


def test_rejects_inconsistent_shapes():
    dim, length = 3, 2
    weight = torch.randn(dim, WIDTH, dtype=torch.float32)
    x = torch.randn(dim, length, dtype=torch.float32)
    qsl = torch.tensor([0, length], dtype=torch.int32)
    cache = torch.tensor([1], dtype=torch.int32)
    has = torch.tensor([False])

    with pytest.raises(ValueError, match="state_len"):
        causal_conv1d_fn(x, weight, None, mk_state(2, dim, WIDTH - 2), qsl, cache_indices=cache, has_initial_state=has)
    with pytest.raises(ValueError, match="channels"):
        causal_conv1d_fn(x, weight, None, mk_state(2, dim + 1, WIDTH - 1), qsl, cache_indices=cache, has_initial_state=has)
    with pytest.raises(ValueError, match="width"):
        causal_conv1d_fn(x, torch.randn(dim, 1), None, mk_state(2, dim, WIDTH - 1), qsl, cache_indices=cache, has_initial_state=has)
    with pytest.raises(ValueError, match="must start at 0"):
        causal_conv1d_fn(x, weight, None, mk_state(2, dim, WIDTH - 1), torch.tensor([1, length + 1]), cache_indices=cache, has_initial_state=has)
    with pytest.raises(ValueError, match="state_len"):
        causal_conv1d_update(torch.randn(1, dim), mk_state(2, dim, WIDTH - 2), weight, conv_state_indices=cache)
    with pytest.raises(ValueError, match="max_query_len"):
        causal_conv1d_update(
            torch.randn(length, dim),
            mk_state(2, dim, WIDTH - 1),
            weight,
            conv_state_indices=cache,
            query_start_loc=torch.tensor([0, length], dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="activation"):
        causal_conv1d_update(torch.randn(1, dim), mk_state(2, dim, WIDTH - 1), weight, None, "gelu", conv_state_indices=cache)


def test_width_three_and_five_also_match_the_oracle():
    # the GDN case is width 4, but the same kernel unrolls 2/3/4/5/6 (l.399-409); the generic loop must
    # agree for the neighbouring widths too, including an odd tap count.
    for width in (2, 3, 5, 6):
        torch.manual_seed(30 + width)
        dim, length, line = 3, 7, 1
        conv_states = mk_state(3, dim, width - 1)
        init = conv_states[line, :, : width - 1].transpose(0, 1).clone()
        weight = torch.randn(dim, width, dtype=torch.float32)
        x = torch.randn(dim, length, dtype=torch.float32)
        y = causal_conv1d_fn(
            x,
            weight,
            None,
            conv_states,
            torch.tensor([0, length], dtype=torch.int32),
            cache_indices=torch.tensor([line], dtype=torch.int32),
            has_initial_state=torch.tensor([True]),
            activation=None,
        )
        ref, ref_state = oracle_conv(x.transpose(0, 1), weight, None, init, width, use_silu=False)
        torch.testing.assert_close(y.transpose(0, 1), ref.to(torch.float32), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            conv_states[line, :, : width - 1].transpose(0, 1), ref_state.to(torch.float32), rtol=1e-6, atol=1e-6
        )
