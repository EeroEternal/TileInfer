"""The fused conv kernel must *degrade observably*, never silently (§89.2).

Two bugs hid behind `except Exception: return None`: a bf16-vs-fp32 bias (raised inside the launch)
and a missing `NameError: name 'C'` build failure.  Both made the probe report ``max_abs = 0``, which
reads exactly like "the kernel ran and agreed perfectly" — the most misleading failure mode there is.

These tests pin the two properties that make that impossible to reintroduce:

* an unsupported shape returns ``None`` (so the torch path runs) and does **not** touch the counter;
* a *failed launch* returns ``None`` **and** increments ``_CONV_KERNEL_FALLBACKS``.

No device and no tilelang are needed: the branch under test is pure Python bookkeeping.
"""

from __future__ import annotations

import torch

from tileinfer.vllm.ops.fla import causal_conv1d


def _state(channels=64, lines=4):
    return torch.zeros(lines, channels, 3, dtype=torch.bfloat16)


def _weight(channels=64):
    return torch.zeros(channels, 4, dtype=torch.bfloat16)


def test_unsupported_shape_returns_none_without_counting_a_fallback():
    """A shape the kernel cannot serve is *not* a failure — it must not inflate the counter."""
    before = causal_conv1d._CONV_KERNEL_FALLBACKS
    # width 3 is not the 4-tap kernel's shape
    result = causal_conv1d._try_conv_kernel(
        torch.zeros(1, 64, dtype=torch.bfloat16),
        _state(),
        torch.zeros(64, 3, dtype=torch.bfloat16),
        None,
        0,
        torch.bfloat16,
    )
    assert result is None
    assert causal_conv1d._CONV_KERNEL_FALLBACKS == before


def test_fp32_state_is_rejected_by_the_shape_gate():
    """The kernel is bf16-only; an fp32 state must fall through to the torch path."""
    before = causal_conv1d._CONV_KERNEL_FALLBACKS
    result = causal_conv1d._try_conv_kernel(
        torch.zeros(1, 64, dtype=torch.float32),
        torch.zeros(4, 64, 3, dtype=torch.float32),
        _weight().float(),
        None,
        0,
        torch.float32,
    )
    assert result is None
    assert causal_conv1d._CONV_KERNEL_FALLBACKS == before


def test_channels_not_divisible_by_the_core_count_are_rejected():
    before = causal_conv1d._CONV_KERNEL_FALLBACKS
    result = causal_conv1d._try_conv_kernel(
        torch.zeros(1, 40, dtype=torch.bfloat16),
        _state(channels=40),
        _weight(channels=40),
        None,
        0,
        torch.bfloat16,
    )
    assert result is None, "40 channels cannot be split evenly across the kernel's cores"
    assert causal_conv1d._CONV_KERNEL_FALLBACKS == before


def test_a_failed_launch_is_counted(monkeypatch):
    """When the kernel *is* attempted and raises, the fallback must become visible in the counter."""
    import tileinfer.kernels.gdn.conv_update_ascend950 as kernel_mod

    def boom(**kwargs):
        raise RuntimeError("simulated launch failure")

    monkeypatch.setattr(kernel_mod, "gdn_conv_update_kernel_available", lambda: True)
    monkeypatch.setattr(kernel_mod, "build_conv_update_kernel", boom)
    before = causal_conv1d._CONV_KERNEL_FALLBACKS
    result = causal_conv1d._try_conv_kernel(
        torch.zeros(1, 64, dtype=torch.bfloat16),
        _state(),
        _weight(),
        None,
        0,
        torch.bfloat16,
    )
    assert result is None
    assert causal_conv1d._CONV_KERNEL_FALLBACKS == before + 1, (
        "a swallowed launch failure is indistinguishable from a successful identical result"
    )


def test_non_contiguous_state_or_weight_is_rejected():
    """The kernel gets raw pointers: a view would silently read the wrong addresses.

    This is the failure mode that matters most — not a crash, but *plausible wrong numbers*.  A
    non-contiguous cache line (a slice of a larger pool) or a transposed weight must fall through to
    the torch path instead.
    """
    before = causal_conv1d._CONV_KERNEL_FALLBACKS
    big = torch.zeros(4, 64, 6, dtype=torch.bfloat16)
    sliced = big[:, :, :3]  # contiguous-looking shape, but stride 6 on the last axis
    assert not sliced.is_contiguous()
    assert (
        causal_conv1d._try_conv_kernel(
            torch.zeros(1, 64, dtype=torch.bfloat16), sliced, _weight(), None, 0, torch.bfloat16
        )
        is None
    )

    transposed = _weight().t()  # [W, C], not [C, W]
    assert not transposed.is_contiguous()
    assert (
        causal_conv1d._try_conv_kernel(
            torch.zeros(1, 64, dtype=torch.bfloat16),
            _state(),
            transposed,
            None,
            0,
            torch.bfloat16,
        )
        is None
    )
    assert causal_conv1d._CONV_KERNEL_FALLBACKS == before


def test_multi_sequence_batch_reaches_the_kernel(monkeypatch):
    """``--max-num-seqs 4`` means a 4-sequence decode step must still use the kernel (§91).

    The first wiring required ``x.shape[0] == 1``, which made the kernel unreachable in every real
    serving configuration — a silently dead optimisation.  The real condition is one token *per
    sequence*, so a batch of several sequences must call the kernel once per sequence.
    """
    calls = []

    def fake_kernel(x_seq, conv_state, weight, bias, line, conv_dtype):
        calls.append(int(line))
        return torch.zeros(int(weight.shape[0]), dtype=torch.bfloat16)

    monkeypatch.setenv("TILEINFER_GDN_CONV_KERNEL", "1")
    monkeypatch.delenv("TILEINFER_FLAG_WATCH", raising=False)
    monkeypatch.setattr(causal_conv1d, "_try_conv_kernel", fake_kernel)

    batch, channels = 3, 64
    x = torch.randn(batch, channels, dtype=torch.bfloat16)  # [batch, dim] -> one token each
    state = torch.zeros(8, channels, 3, dtype=torch.bfloat16)
    weight = torch.zeros(channels, 4, dtype=torch.bfloat16)
    idx = torch.tensor([1, 2, 3], dtype=torch.int32)

    causal_conv1d.causal_conv1d_update(
        x, state, weight, None, "silu", conv_state_indices=idx
    )
    assert calls == [1, 2, 3], f"expected one kernel call per sequence, got {calls}"


def test_fp32_bias_conversion_is_cached_per_parameter():
    """Converting the bf16 bias on every layer of every step was a 32 KB copy x30 per step."""
    bias = torch.randn(64, dtype=torch.bfloat16)
    first = causal_conv1d._fp32_bias(bias)
    second = causal_conv1d._fp32_bias(bias)
    assert first is second, "the same parameter must reuse its fp32 conversion"
    assert first.dtype == torch.float32

    other = torch.randn(64, dtype=torch.bfloat16)
    assert causal_conv1d._fp32_bias(other) is not first, "a different parameter must convert again"

    already = torch.randn(64, dtype=torch.float32)
    assert causal_conv1d._fp32_bias(already) is already, "an fp32 bias needs no copy at all"


def test_switch_is_off_by_default(monkeypatch):
    monkeypatch.delenv("TILEINFER_GDN_CONV_KERNEL", raising=False)
    monkeypatch.delenv("TILEINFER_FLAG_WATCH", raising=False)
    assert causal_conv1d._conv_kernel_enabled() is False
    monkeypatch.setenv("TILEINFER_GDN_CONV_KERNEL", "1")
    assert causal_conv1d._conv_kernel_enabled() is True
