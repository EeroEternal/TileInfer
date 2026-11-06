"""Tests for the device-free A/B report core (``benchmarks/ab_report.py``).

Everything here is CPU-only and imports no vllm/torch: the harness core must be
runnable on any box, so its contract is tested with plain numbers and JSON files.
"""
from __future__ import annotations

import pytest

from benchmarks import ab_report
from benchmarks.ab_report import (
    check_reportable,
    compare_baseline,
    format_table,
    main,
    summarize,
)


def test_summarize_median_and_p90_five_values():
    stats = summarize([1, 2, 3, 4, 5])
    assert stats["count"] == 5
    assert stats["median"] == 3
    # interpolated at rank (n-1)*0.9 = 3.6 -> 4 + 0.6*(5-4)
    assert stats["p90"] == pytest.approx(4.6)
    assert stats["min"] == 1
    assert stats["max"] == 5
    assert stats["mean"] == 3


def test_summarize_single_two_and_ten_values():
    one = summarize([10])
    assert one["median"] == 10 and one["p90"] == 10 and one["mean"] == 10
    assert one["min"] == one["max"] == 10

    two = summarize([1, 3])
    assert two["median"] == 2
    assert two["p90"] == pytest.approx(2.8)  # 1 + 0.9*(3-1)

    ten = summarize(range(1, 11))
    assert ten["median"] == 5.5
    assert ten["p90"] == pytest.approx(9.1)  # rank 8.1 -> 9 + 0.1*(10-9)


def test_summarize_matches_statistics_inclusive_p90():
    import statistics

    values = [3.5, 1.0, 7.25, 2.0, 9.5, 4.75, 6.0, 8.0]
    expected = statistics.quantiles(values, n=10, method="inclusive")[8]
    assert summarize(values)["p90"] == pytest.approx(expected)


def test_summarize_empty_raises():
    with pytest.raises(ValueError):
        summarize([])


def test_compare_within_tolerance_passes():
    observed = {"p1": [1.0, 2.0], "p2": [3.0]}
    baseline = {"p1": [1.00005, 1.99995], "p2": [3.00005]}
    report = compare_baseline(observed, baseline, atol=1e-4)
    assert report["ok"] is True
    assert report["errors"] == []
    assert {r["prompt"] for r in report["results"]} == {"p1", "p2"}
    assert all(r["ok"] for r in report["results"])


def test_compare_over_tolerance_fails_and_lists_prompt():
    observed = {"good": [1.0], "bad": [2.01]}
    baseline = {"good": [1.0], "bad": [2.0]}
    report = compare_baseline(observed, baseline, atol=1e-4)
    assert report["ok"] is False
    by_prompt = {r["prompt"]: r for r in report["results"]}
    assert by_prompt["good"]["ok"] is True
    assert by_prompt["bad"]["ok"] is False
    assert by_prompt["bad"]["max_abs_delta"] == pytest.approx(0.01)


def test_compare_missing_prompt_is_an_error():
    observed = {"p1": [1.0], "only_observed": [2.0]}
    baseline = {"p1": [1.0], "only_baseline": [3.0]}
    report = compare_baseline(observed, baseline, atol=1e-4)
    assert report["ok"] is False
    joined = " | ".join(report["errors"])
    assert "only_observed" in joined and "missing from baseline" in joined
    assert "only_baseline" in joined and "missing from observed" in joined
    # the missing prompts must not show up as passing results
    assert {r["prompt"] for r in report["results"]} == {"p1"}


def test_check_reportable_ready_clean_passes():
    arms = {
        "ti": {"ready": True, "fallbacks": 0, "cold": False},
        "fia": {"ready": True},
    }
    ok, reasons = check_reportable(arms)
    assert ok is True
    assert reasons == []


def test_check_reportable_not_ready_names_arm():
    arms = {"ti": {"ready": False, "fallbacks": 0}}
    ok, reasons = check_reportable(arms)
    assert ok is False
    assert any("ti" in r and "not ready" in r for r in reasons)


def test_check_reportable_missing_ready_is_refused():
    ok, reasons = check_reportable({"ti": {"fallbacks": 0}})
    assert ok is False
    assert any("ti" in r for r in reasons)


def test_check_reportable_fallbacks_refuses():
    ok, reasons = check_reportable({"ti": {"ready": True, "fallbacks": 3}})
    assert ok is False
    assert any("fallbacks" in r and "3" in r and "ti" in r for r in reasons)


def test_check_reportable_cold_refuses():
    ok, reasons = check_reportable({"ti": {"ready": True, "fallbacks": 0, "cold": True}})
    assert ok is False
    assert any("cold" in r and "ti" in r for r in reasons)


def test_format_table_keeps_headers_and_rows():
    out = format_table(
        [["decode d256", 1.25], ["prefill 8k", 12.5]],
        ["scenario", "ttft_ms"],
    )
    lines = out.splitlines()
    assert len(lines) == 4  # header + rule + 2 rows
    assert "scenario" in lines[0] and "ttft_ms" in lines[0]
    assert "decode d256" in lines[2] and "1.25" in lines[2]
    assert "prefill 8k" in lines[3] and "12.5" in lines[3]


def test_main_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    assert "--compare" in capsys.readouterr().out


def test_main_check_not_ready_exits_one(monkeypatch):
    # The DSH sandbox denies writing throwaway files, so stub the loader: this
    # test is about main()'s exit code for a refused report, not about I/O.
    monkeypatch.setattr(
        ab_report, "_load_json", lambda path: {"arms": {"ti": {"ready": False, "fallbacks": 0}}}
    )
    assert main(["--check", "report.json"]) == 1


def test_main_unknown_arg_exits_two():
    with pytest.raises(SystemExit) as excinfo:
        main(["--not-a-mode"])
    assert excinfo.value.code == 2
