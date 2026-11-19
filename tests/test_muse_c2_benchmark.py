"""Shape and tolerance checks for the frozen MuseGlimmer C2 sample.

The numeric parity itself needs the device (see tools/muse/check_c2.py); this test keeps the
declaration well-formed and prevents a silent "benchmark is still empty" regression.
"""

from __future__ import annotations

from tileinfer.manifests.muse_glimmer import MUSE_GLIMMER


def test_c2_sample_is_present_and_well_formed() -> None:
    sample = MUSE_GLIMMER.benchmark
    assert sample, "MuseGlimmer must carry the C2 first-token sample"
    for key in ("prompt", "max_tokens", "temperature", "top_logprobs", "first_token", "top5"):
        assert key in sample, f"benchmark sample is missing {key!r}"
    assert sample["prompt"] == "The capital of France is"
    assert int(sample["top_logprobs"]) == 5
    assert len(sample["top5"]) == 5
    assert sample["top5"][0][0] == sample["first_token"]


def test_c2_top5_is_ordered_and_has_a_declared_tolerance() -> None:
    values = [float(logprob) for _, logprob in MUSE_GLIMMER.benchmark["top5"]]
    assert values == sorted(values, reverse=True)
    tolerance = MUSE_GLIMMER.benchmark["tolerance"]
    assert float(tolerance["top1_logprob"]) > 0
    assert float(tolerance["top5_max_logprob"]) > 0
    assert tolerance["token_set"] == "exact"
    tp_tolerance = MUSE_GLIMMER.benchmark["tolerance_tp"]
    assert float(tp_tolerance["top1_logprob"]) > float(tolerance["top1_logprob"])
    assert tp_tolerance["token_set"] == "exact"


def test_manifest_still_validates() -> None:
    MUSE_GLIMMER.validate()


def test_c4_gate_is_declared_and_measured_within_thresholds() -> None:
    gate = MUSE_GLIMMER.benchmark["c4_gate"]
    for key in ("single_stream_tpot_ms", "ttft_8k_s", "eight_way_tok_s", "cold_start_s"):
        assert float(gate[key]) > 0, key
    for arm in ("measured_tp1", "measured_tp2"):
        measured = gate[arm]
        assert measured["tpot_ms"] <= gate["single_stream_tpot_ms"]
        assert measured["ttft_8k_s"] <= gate["ttft_8k_s"]
        assert measured["eight_way_tok_s"] >= gate["eight_way_tok_s"]
        assert measured["cold_start_s"] <= gate["cold_start_s"]


def test_tp_support_is_declared_with_the_evidence() -> None:
    assert "tp" in MUSE_GLIMMER.parallel
    assert MUSE_GLIMMER.support["tp"] == ["1", "2"]
    assert MUSE_GLIMMER.support["max_context_verified"] == 131072
