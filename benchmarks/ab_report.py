#!/usr/bin/env python
"""Device-free core of the TileInfer vs vllm-ascend A/B harness.

This module is deliberately stdlib-only (no vllm, no torch): it turns raw
per-run numbers into summary statistics, compares two arms prompt by prompt,
and decides whether a report is allowed to be quoted at all.  The device
driving wrapper (``benchmarks/ab_native_vs_ascend.py``) is a later task and
will import these helpers.

CLI::

    python benchmarks/ab_report.py --summarize values.json
    python benchmarks/ab_report.py --compare observed.json baseline.json --atol 1e-4
    python benchmarks/ab_report.py --check report.json

Exit codes: 0 = success, 1 = the comparison/check found a problem,
2 = usage or input error.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence


def _interpolated_percentile(sorted_values: Sequence[float], q: float) -> float:
    """Percentile at ``q`` in [0, 1] over an already-sorted sample.

    Uses position ``(n - 1) * q`` and linear interpolation between the two
    neighbouring order statistics.  This is exactly the convention of
    ``statistics.quantiles(data, n=10, method="inclusive")`` (whose ninth cut
    point is the p90), so hand-computed expectations match the stdlib.
    """
    n = len(sorted_values)
    pos = (n - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac)


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    """Summarize one sample: count/median/p90/min/max/mean.

    ``p90`` is the linearly interpolated percentile at rank ``(n-1) * 0.9``,
    i.e. the same value as
    ``statistics.quantiles(values, n=10, method="inclusive")[8]``.

    Raises ``ValueError`` on an empty sample.
    """
    vals = sorted(float(v) for v in values)
    if not vals:
        raise ValueError("summarize() needs at least one value")
    return {
        "count": len(vals),
        "median": float(statistics.median(vals)),
        "p90": _interpolated_percentile(vals, 0.9),
        "min": vals[0],
        "max": vals[-1],
        "mean": sum(vals) / len(vals),
    }


def _as_float_list(values: object, prompt: object) -> list[float]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"prompt {prompt!r}: expected a list of numbers")
    return [float(v) for v in values]


def compare_baseline(
    observed: Mapping[object, Sequence[float]],
    baseline: Mapping[object, Sequence[float]],
    atol: float,
) -> dict[str, object]:
    """Compare per-prompt samples against a frozen baseline.

    Returns ``{"ok": bool, "results": [...], "errors": [...]}`` where each
    result is ``{"prompt", "max_abs_delta", "ok"}``.  A prompt present on only
    one side is an *error*, never a silent pass; mismatched sample lengths are
    errors too.  ``ok`` is true only when there are no errors and every
    prompt's max absolute delta is within ``atol``.
    """
    if not isinstance(observed, Mapping) or not isinstance(baseline, Mapping):
        return {
            "ok": False,
            "results": [],
            "errors": ["observed and baseline must be mappings of prompt -> list of numbers"],
        }

    errors: list[str] = []
    results: list[dict[str, object]] = []

    observed_keys = set(observed)
    baseline_keys = set(baseline)
    for prompt in sorted(observed_keys - baseline_keys, key=str):
        errors.append(f"prompt {prompt!r} is missing from baseline")
    for prompt in sorted(baseline_keys - observed_keys, key=str):
        errors.append(f"prompt {prompt!r} is missing from observed")

    for prompt in sorted(observed_keys & baseline_keys, key=str):
        try:
            obs = _as_float_list(observed[prompt], prompt)
            base = _as_float_list(baseline[prompt], prompt)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
            continue
        if len(obs) != len(base):
            errors.append(
                f"prompt {prompt!r}: length mismatch observed={len(obs)} baseline={len(base)}"
            )
            continue
        max_abs_delta = max((abs(a - b) for a, b in zip(obs, base)), default=0.0)
        results.append(
            {"prompt": prompt, "max_abs_delta": max_abs_delta, "ok": max_abs_delta <= atol}
        )

    if not results and not errors:
        errors.append("nothing to compare: observed and baseline share no prompts")

    ok = not errors and all(bool(r["ok"]) for r in results)
    return {"ok": ok, "results": results, "errors": errors}


def check_reportable(
    arms: Mapping[str, Mapping[str, object]], require_readiness: bool = True
) -> tuple[bool, list[str]]:
    """Decide whether a multi-arm report may be quoted.

    Refuses when an arm is not ``ready`` (unless ``require_readiness=False``),
    when its ``fallbacks`` counter (if present) is non-zero, or when it was
    measured ``cold``.  Returns ``(ok, reasons)`` with one reason per problem,
    each naming the offending arm.
    """
    if not isinstance(arms, Mapping) or not arms:
        return False, ["no arms in report"]

    reasons: list[str] = []
    for name in sorted(arms, key=str):
        arm = arms[name]
        if not isinstance(arm, Mapping):
            reasons.append(f"arm {name!r}: not an object")
            continue
        if require_readiness and arm.get("ready") is not True:
            reasons.append(f"arm {name!r}: not ready (ready must be true)")
        fallbacks = arm.get("fallbacks")
        if fallbacks:
            reasons.append(f"arm {name!r}: fallbacks={fallbacks!r} (must be 0)")
        if arm.get("cold") is True:
            reasons.append(f"arm {name!r}: cold start (warm up before measuring)")
    return (not reasons), reasons


def format_table(rows: Iterable[Sequence[object]], headers: Sequence[object]) -> str:
    """Render a plain-text, left-aligned table (no rich/tabulate).

    Width is measured in code points, so CJK cells may look narrower than
    their terminal width; output is intentionally width-agnostic.
    """
    head = [str(h) for h in headers]
    body: list[list[str]] = []
    for row in rows:
        cells = [str(c) for c in row]
        if len(cells) != len(head):
            raise ValueError(f"row has {len(cells)} cells, expected {len(head)}")
        body.append(cells)

    widths = [len(h) for h in head]
    for cells in body:
        widths = [max(w, len(c)) for w, c in zip(widths, cells)]

    def render(cells: Sequence[str]) -> str:
        return " | ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()

    lines = [render(head), "-+-".join("-" * w for w in widths)]
    lines.extend(render(cells) for cells in body)
    return "\n".join(lines)


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _load_json(path: str) -> object:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _extract_values(data: object) -> list[float]:
    if isinstance(data, list):
        return [float(v) for v in data]
    if isinstance(data, Mapping):
        if "values" in data:
            values = data["values"]
            if not isinstance(values, list):
                raise ValueError("'values' must be a list of numbers")
            return [float(v) for v in values]
        try:
            return [float(v) for v in data.values()]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "summarize input must be a list, {'values': [...]}, or a mapping of numbers"
            ) from exc
    raise ValueError("summarize input must be a list, {'values': [...]}, or a mapping of numbers")


def _load_arms(path: str) -> Mapping[str, Mapping[str, object]]:
    report = _load_json(path)
    if isinstance(report, Mapping) and isinstance(report.get("arms"), Mapping):
        return report["arms"]  # type: ignore[return-value]
    if isinstance(report, Mapping):
        return report  # type: ignore[return-value]
    raise ValueError("--check expects a JSON object (or one with an 'arms' object)")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ab_report.py",
        description="Device-free summary/compare/refusal tools for the TileInfer A/B harness.",
    )
    parser.add_argument("--summarize", metavar="FILE", help="summarize a JSON list of numbers")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("OBSERVED", "BASELINE"),
        help="compare two JSON objects: prompt -> list of numbers",
    )
    parser.add_argument(
        "--atol", type=float, default=1e-4, help="absolute tolerance for --compare (default 1e-4)"
    )
    parser.add_argument("--check", metavar="REPORT.json", help="refuse unquotable arm reports")
    args = parser.parse_args(argv)

    selected = sum(
        flag is not None for flag in (args.summarize, args.compare, args.check)
    )
    if selected != 1:
        parser.print_usage(sys.stderr)
        print(
            "ab_report.py: error: choose exactly one of --summarize/--compare/--check",
            file=sys.stderr,
        )
        return 2

    try:
        if args.summarize is not None:
            stats = summarize(_extract_values(_load_json(args.summarize)))
            rows = [[name, _fmt(value)] for name, value in stats.items()]
            print(format_table(rows, ["metric", "value"]))
            return 0

        if args.compare is not None:
            observed = _load_json(args.compare[0])
            baseline = _load_json(args.compare[1])
            report = compare_baseline(observed, baseline, args.atol)  # type: ignore[arg-type]
            rows = [
                [r["prompt"], _fmt(r["max_abs_delta"]), "ok" if r["ok"] else "FAIL"]
                for r in report["results"]  # type: ignore[union-attr]
            ]
            if rows:
                print(format_table(rows, ["prompt", "max_abs_delta", "ok"]))
            for reason in report["errors"]:  # type: ignore[union-attr]
                print(f"error: {reason}", file=sys.stderr)
            print("RESULT:", "ok" if report["ok"] else "FAIL")
            return 0 if report["ok"] else 1

        arms = _load_arms(args.check)
        ok, reasons = check_reportable(arms)
        for reason in reasons:
            print(f"refused: {reason}", file=sys.stderr)
        print("reportable:", "yes" if ok else "no")
        return 0 if ok else 1
    except (OSError, ValueError) as exc:
        # json.JSONDecodeError subclasses ValueError, so bad JSON lands here too.
        print(f"ab_report.py: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
