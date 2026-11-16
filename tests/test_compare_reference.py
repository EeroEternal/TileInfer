from __future__ import annotations

from benchmarks.compare_reference import compare


def _ref(logprobs: dict[str, float]) -> dict:
    return {
        "text": "hello world",
        "first_token_top5": [{"token": k, "logprob": v} for k, v in logprobs.items()],
    }


def _resp(logprobs: dict[str, float], text: str = "hello world") -> dict:
    return {
        "choices": [
            {
                "text": text,
                "logprobs": {"top_logprobs": [logprobs]},
            }
        ],
        "usage": {"completion_tokens": 2},
    }


def test_reference_match_and_tie_boundary() -> None:
    # A top-k cut tie is allowed when the shared tokens are within tolerance.
    ref = _ref({"a": -1.0, "b": -2.0, "c": -3.0})
    resp = _resp({"a": -1.0, "b": -2.0, "d": -3.0})
    result = compare(ref, resp, atol=0.01)
    assert result["same"] is True
    assert result["top5_strict_equal"] is False
    assert result["top5_close_tie_aware"] is True


def test_reference_text_mismatch_is_not_same() -> None:
    ref = _ref({"a": -1.0})
    result = compare(ref, _resp({"a": -1.0}, text="hello there"), atol=1e-4)
    assert result["same"] is False
    assert result["text_equal"] is False
    assert result["first_text_diff"] is not None


def test_reference_logprob_outside_tolerance_is_not_same() -> None:
    ref = _ref({"a": -1.0})
    result = compare(ref, _resp({"a": -1.2}), atol=0.1)
    assert result["same"] is False
    assert result["max_abs_logprob_diff"] == 0.19999999999999996
