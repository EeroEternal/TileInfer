from __future__ import annotations

from tileinfer.bridge import metrics


def test_graph_replay_is_counted_even_when_vendor_says_not_uniform_decode(tmp_path) -> None:
    metrics.reset()
    metrics.configure(str(tmp_path / "metrics.json"))
    metrics.record_capture_done(2)

    # Ascend can replay a captured graph while reporting uniform_decode=False.  The
    # replay must still advance the counter; this is the exact target-dummy symptom.
    metrics.record_dispatch(False, "FULL", True)
    assert metrics.COUNTERS.decode_calls == 1
    assert metrics.COUNTERS.decode_graph_hits == 1
    assert metrics.COUNTERS.decode_fallbacks == 0

    # A non-uniform NONE is observable, but is not the strict fallback event.
    metrics.record_dispatch(False, "NONE", True)
    assert metrics.COUNTERS.none_after_capture == 1
    assert metrics.COUNTERS.decode_calls == 1
    assert metrics.COUNTERS.decode_fallbacks == 0

    # The strict gate: after capture, a uniform decode with graph allowed chose NONE.
    metrics.record_dispatch(True, "NONE", True)
    assert metrics.COUNTERS.decode_calls == 2
    assert metrics.COUNTERS.decode_fallbacks == 1

    snapshot = metrics.COUNTERS.snapshot()
    assert snapshot["decode_graph_hits"] == 1
    assert snapshot["none_after_capture"] == 2


def test_dispatch_before_capture_is_not_a_fallback(tmp_path) -> None:
    metrics.reset()
    metrics.configure(str(tmp_path / "metrics.json"))

    metrics.record_dispatch(True, "NONE", True)
    assert metrics.COUNTERS.decode_calls == 0
    assert metrics.COUNTERS.decode_fallbacks == 0
    assert metrics.COUNTERS.none_after_capture == 0

    metrics.record_capture_done(1)
    metrics.record_dispatch(True, "PIECEWISE", True)
    assert metrics.COUNTERS.decode_calls == 1
    assert metrics.COUNTERS.decode_graph_hits == 1
    assert metrics.COUNTERS.decode_fallbacks == 0
