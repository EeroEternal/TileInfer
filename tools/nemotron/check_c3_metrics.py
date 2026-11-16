#!/usr/bin/env python3
"""Check a TileInfer metrics JSON against the C3 acceptance shape.

C3 is not "capture happened": it requires a real decode step to hit a captured graph and
zero fallbacks.  The metrics file is only written every ``dump_every_decode_steps`` steps
(500 by default), so a capture-only snapshot must not be mistaken for a pass.  This tool is
deliberately strict: it rejects ``decode_calls == 0`` unless ``--allow-zero-decode`` is
given for a capture-only smoke test.

Usage::

    python tools/nemotron/check_c3_metrics.py /tmp/m5-nemotron-c3-metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


def _as_int(data: Mapping[str, Any], key: str, errors: list[str]) -> int | None:
    try:
        return int(data[key])
    except (KeyError, TypeError, ValueError):
        errors.append(f"{key}: missing or not an integer")
        return None


def check_metrics(data: Mapping[str, Any], *, allow_zero_decode: bool = False) -> list[str]:
    """Return every C3 acceptance violation; empty means pass."""
    errors: list[str] = []
    capture_calls = _as_int(data, "capture_calls", errors)
    if capture_calls is not None and capture_calls < 1:
        errors.append(f"capture_calls: expected >= 1, got {capture_calls}")
    if not bool(data.get("capture_done", False)):
        errors.append("capture_done: expected true")

    decode_calls = _as_int(data, "decode_calls", errors)
    graph_hits = _as_int(data, "decode_graph_hits", errors)
    fallbacks = _as_int(data, "decode_fallbacks", errors)
    if decode_calls is not None and decode_calls <= 0 and not allow_zero_decode:
        errors.append(f"decode_calls: expected > 0, got {decode_calls}")
    if decode_calls is not None and graph_hits is not None and graph_hits != decode_calls:
        errors.append(f"decode_graph_hits: expected {decode_calls}, got {graph_hits}")
    if fallbacks is not None and fallbacks != 0:
        errors.append(f"decode_fallbacks: expected 0, got {fallbacks}")

    internal_errors = data.get("errors", 0)
    try:
        internal_errors_i = int(internal_errors)
    except (TypeError, ValueError):
        errors.append(f"errors: expected integer, got {internal_errors!r}")
    else:
        if internal_errors_i != 0:
            errors.append(f"errors: expected 0, got {internal_errors_i}")
    return errors


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("metrics", type=Path, help="metrics JSON produced by TileInfer")
    ap.add_argument(
        "--allow-zero-decode",
        action="store_true",
        help="capture-only smoke test: allow decode_calls == 0 (still requires capture_done)",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        data = json.loads(args.metrics.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read {args.metrics}: {exc}", file=sys.stderr)
        return 2
    errors = check_metrics(data, allow_zero_decode=args.allow_zero_decode)
    result = {
        "metrics": str(args.metrics),
        "pid": data.get("pid"),
        "decode_calls": data.get("decode_calls"),
        "decode_graph_hits": data.get("decode_graph_hits"),
        "decode_fallbacks": data.get("decode_fallbacks"),
        "ok": not errors,
        "errors": errors,
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
