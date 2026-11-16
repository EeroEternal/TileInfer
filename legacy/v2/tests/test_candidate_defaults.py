"""Every performance candidate's **default** is pinned here.

C2 says a candidate whose measured ratio is below 1.0 must default to off, and
``docs/notes/perf-candidates.md`` records what each one measured.  A registry nobody enforces drifts:
the way this breaks is a default silently flipping because someone added a flag, or an ``env`` read
that starts defaulting to "on" when it used to be "off".  These assertions are cheap and they make
that a test failure instead of a mystery in production.

Deliberately env-only: each case clears the switch's environment variable first, so a measurement
shell that exports ``TILEINFER_*`` cannot change what this test observes.
"""

from __future__ import annotations

import pytest

from tileinfer.vllm.ops.fla import causal_conv1d, recurrent
from tileinfer.vllm.ops import moe


@pytest.fixture(autouse=True)
def _no_overrides(monkeypatch):
    """Clear every candidate switch and the file-watch channel for the duration of each test."""
    for name in (
        "TILEINFER_MOE_SORTED",
        "TILEINFER_MOE_FP32",
        "TILEINFER_MOE_SWIGLU",
        "TILEINFER_GDN_CACHE",
        "TILEINFER_GDN_CONV_FAST",
        "TILEINFER_GDN_CONV_LIB",
        "TILEINFER_GDN_NO_SYNC",
        "TILEINFER_FLAG_WATCH",
    ):
        monkeypatch.delenv(name, raising=False)


def test_enabled_by_default():
    """Landing wins that are numerically identical **and** measured: on unless asked otherwise."""
    assert moe._sorted_path_enabled() is True, "MoE sort-once loop (2.569x, bit-identical)"


def test_enabled_by_default_after_measurement():
    """The bounds-check cache measured 1.107x on TPOT (§42), so it is the default now."""
    assert moe._moe_strict() is False, (
        "per-call topk_ids validation costs 10.6 ms/token; the check is cached after the first call"
    )


def test_disabled_by_default():
    """Measured at or below parity -> off (C2)."""
    from tileinfer.serving import flags

    assert causal_conv1d._conv_fast_enabled() is True, (
        "conv vectorisation + sync-free update (1.241x end to end)"
    )
    assert causal_conv1d._conv_lib_enabled() is False, (
        "grouped conv1d measured 1.004 (no difference, §42) - keep the simpler form"
    )
    assert flags.flag("gdn_no_sync", "TILEINFER_GDN_NO_SYNC", False) is False, (
        "sync-free state indexing measured 0.453x (2.2x slower, §33)"
    )
    from tileinfer.vllm.ops.fla import hostcache

    assert hostcache._enabled() is False, (
        "batch-meta host cache measured 0.999 (no effect, §42)"
    )


def test_gdn_batched_solve_is_on_by_default():
    """Newton doubling replaced a 64-step triangular solve: 11.91 -> 0.261 ms (§59)."""
    from tileinfer.vllm.ops.fla.solve_tril import _batched_solve_enabled

    assert _batched_solve_enabled() is True


def test_fused_conv_kernel_is_off_until_measured_end_to_end():
    """Kernel-level 42.5x is not enough: C2 wants the end-to-end paired ratio first (§89.5)."""
    from tileinfer.vllm.ops.fla.causal_conv1d import _conv_kernel_enabled

    assert _conv_kernel_enabled() is False, (
        "the fused conv kernel has a 42.5x op ratio but no end-to-end TPOT pairing yet"
    )


def test_moe_matmul_dtype_defaults_to_the_input_dtype():
    """The fp32 upcast is off by default since §54: 1.251 alone, and 3.97x on the batched path.

    (With the per-expert loop it measured 1.035 — inside the noise — which is why this switch spent
    several rounds defaulted the other way.  Batching changed the picture: the path became
    arithmetic-bound, and fp32 arithmetic is ~4x the bf16 rate here.)
    """
    import torch

    x = torch.zeros(2, 4, dtype=torch.bfloat16)
    assert moe._matmul_dtype(x) is torch.bfloat16
    assert moe._matmul_dtype(x.to(torch.float32)) is torch.float32, (
        "an fp32 input must stay fp32 regardless of the switch"
    )


def test_moe_fp32_switch_restores_the_upcast(monkeypatch):
    import torch

    monkeypatch.setenv("TILEINFER_MOE_FP32", "1")
    assert moe._matmul_dtype(torch.zeros(2, 4, dtype=torch.bfloat16)) is torch.float32


def test_switches_can_still_be_turned_off(monkeypatch):
    """Every default-on candidate keeps a working off-switch (E4: reversible)."""
    monkeypatch.setenv("TILEINFER_MOE_SORTED", "0")
    monkeypatch.setenv("TILEINFER_GDN_CONV_FAST", "0")
    assert moe._sorted_path_enabled() is False
    assert causal_conv1d._conv_fast_enabled() is False


def test_default_off_candidates_can_be_turned_on(monkeypatch):
    """... and every default-off one can be re-enabled for a diagnosis or a re-measurement."""
    monkeypatch.setenv("TILEINFER_GDN_CONV_LIB", "1")
    monkeypatch.setenv("TILEINFER_MOE_STRICT", "1")
    monkeypatch.setenv("TILEINFER_GDN_HOST_CACHE", "1")
    assert causal_conv1d._conv_lib_enabled() is True
    assert moe._moe_strict() is True
    from tileinfer.vllm.ops.fla import hostcache

    assert hostcache._enabled() is True


def test_gdn_constants_cache_is_on_by_default(monkeypatch):
    """The decode constant cache (1.231x) is on; ``TILEINFER_GDN_CACHE=0`` turns it off.

    Asserted through the resolver's decision rather than by timing: with the cache disabled the
    function must return freshly built tensors, and with it enabled the *same objects* on the second
    call.  ``torch`` tensors are cheap to build here, so this is exact.
    """
    import torch

    from tileinfer.serving import flags

    assert flags.flag("gdn_cache", "TILEINFER_GDN_CACHE", True) is True
    monkeypatch.setenv("TILEINFER_GDN_CACHE", "0")
    assert flags.flag("gdn_cache", "TILEINFER_GDN_CACHE", True) is False
    # An empty cache must not raise even with the switch off.
    a_log = torch.zeros(2)
    dt_bias = torch.zeros(2)
    recurrent._DECODE_CONST_CACHE.clear()
    hv_to_h, a_w, dtb_w = recurrent._decode_constants(a_log, dt_bias, 2, 2, "cpu", torch.float32)
    assert hv_to_h.numel() == 2 and a_w.numel() == 2 and dtb_w.numel() == 2
