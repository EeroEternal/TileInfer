"""CPU-only checks for the C3 metrics acceptance helper."""

from __future__ import annotations

from tools.nemotron.check_c3_metrics import check_metrics


def _good() -> dict:
    return {
        "capture_calls": 1,
        "captured_graphs": 6,
        "capture_done": True,
        "decode_calls": 15,
        "decode_graph_hits": 15,
        "decode_fallbacks": 0,
        "errors": 0,
    }


def test_good_metrics_pass() -> None:
    assert check_metrics(_good()) == []


def test_zero_decode_is_rejected_by_default() -> None:
    data = _good()
    data["decode_calls"] = 0
    data["decode_graph_hits"] = 0
    assert any("decode_calls" in e for e in check_metrics(data))


def test_zero_decode_can_be_allowed_for_capture_smoke() -> None:
    data = _good()
    data["decode_calls"] = 0
    data["decode_graph_hits"] = 0
    assert check_metrics(data, allow_zero_decode=True) == []


def test_hit_mismatch_is_rejected() -> None:
    data = _good()
    data["decode_graph_hits"] = 14
    assert any("decode_graph_hits" in e for e in check_metrics(data))


def test_fallback_is_rejected() -> None:
    data = _good()
    data["decode_fallbacks"] = 1
    assert any("decode_fallbacks" in e for e in check_metrics(data))


def test_missing_capture_done_is_rejected() -> None:
    data = _good()
    data["capture_done"] = False
    assert any("capture_done" in e for e in check_metrics(data))
