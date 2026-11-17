#!/usr/bin/env python3
"""Validate a safetensors checkpoint directory before serving it.

Catches the failure we actually hit: a shard whose first bytes are zero (an unwritten
preallocated hole materialised by a copy), so the 8-byte header length reads as 0 and vLLM
dies later with "invalid JSON in header". Also checks that every offset in every header is
inside the file, and that the index only references shards that exist.

Usage:
    python scripts/verify-safetensors.py /home/models/Muse-Glimmer-30B

Exit codes: 0 all shards valid, 1 at least one problem, 2 usage error.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import Iterable


def _check_one(path: Path) -> list[str]:
    problems: list[str] = []
    size = path.stat().st_size
    with path.open("rb") as fh:
        head = fh.read(8)
        if len(head) < 8:
            return [f"{path.name}: file shorter than the 8-byte header"]
        header_len = struct.unpack("<Q", head)[0]
        if header_len == 0:
            return [
                f"{path.name}: header length is 0 (zeroed prefix / unwritten hole); "
                "this shard must be re-uploaded"
            ]
        if header_len > size - 8:
            return [f"{path.name}: header length {header_len} exceeds file size {size}"]
        raw = fh.read(header_len)
    try:
        header = json.loads(raw)
    except json.JSONDecodeError as exc:
        return [f"{path.name}: header is not valid JSON: {exc}"]
    data_start = 8 + header_len
    end_max = 0
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        offsets = meta.get("data_offsets")
        if not (isinstance(offsets, list) and len(offsets) == 2):
            problems.append(f"{path.name}: tensor {name!r} has no valid data_offsets")
            continue
        begin, end = int(offsets[0]), int(offsets[1])
        if begin < 0 or end < begin:
            problems.append(f"{path.name}: tensor {name!r} has bad offsets {offsets}")
        end_max = max(end_max, end)
    if data_start + end_max > size:
        problems.append(
            f"{path.name}: data ends at {data_start + end_max}, file is {size} "
            f"({data_start + end_max - size} bytes missing)"
        )
    return problems


def _iter_shards(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("*.safetensors"))


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.split("Usage:")[1].strip(), file=sys.stderr)
        return 2
    root = Path(argv[1])
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2
    shards = list(_iter_shards(root))
    if not shards:
        print(f"no *.safetensors under {root}", file=sys.stderr)
        return 1

    problems: list[str] = []
    for shard in shards:
        problems.extend(_check_one(shard))

    index_path = root / "model.safetensors.index.json"
    referenced: set[str] = set()
    if index_path.exists():
        try:
            weight_map = json.loads(index_path.read_text())["weight_map"]
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            problems.append(f"{index_path.name}: cannot read weight_map: {exc}")
        else:
            referenced = set(weight_map.values())
            missing = sorted(referenced - {p.name for p in shards})
            if missing:
                problems.append(f"index references missing shards: {', '.join(missing)}")

    for shard in shards:
        print(f"{'OK  ' if not any(p.startswith(shard.name) for p in problems) else 'FAIL'} "
              f"{shard.name}  {shard.stat().st_size / 1e9:.2f} GB")

    if problems:
        print("\nProblems:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"\nAll {len(shards)} shards valid." + (
        f" Index references {len(referenced)} shards." if referenced else ""
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
