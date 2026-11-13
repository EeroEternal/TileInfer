#!/usr/bin/env python3
"""Validate the headers of the shards that are actually on disk.

Only the safetensors header (a JSON block at the start of the file) is read - never the
tensor payload.  It catches truncated/partial shards, duplicate keys, index/header drift and
offsets that run past EOF, all before a GPU is touched.

Missing shards are reported separately: a header audit of 5/14 shards can be clean while the
checkpoint is still unloadable.
"""

from __future__ import annotations

import argparse
import json
import struct
from collections import Counter
from pathlib import Path


def _read_header(path: Path) -> tuple[dict, int, int]:
    with path.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError("shorter than the 8-byte header length")
        header_len = struct.unpack("<Q", raw)[0]
        if header_len == 0 or header_len > 64 * 1024 * 1024:
            raise ValueError(f"implausible header length {header_len}")
        raw_header = fh.read(header_len)
        if len(raw_header) != header_len:
            raise ValueError("header is truncated")
    return json.loads(raw_header), header_len, 8 + header_len


def audit(model_dir: Path) -> dict:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return {"model_dir": str(model_dir), "index_present": False, "headers_ok": False}
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map") or {}
    shard_keys: dict[str, set[str]] = {}
    for key, shard in weight_map.items():
        shard_keys.setdefault(shard, set()).add(key)

    errors: list[str] = []
    header_errors: list[str] = []
    shards = []
    seen: dict[str, str] = {}
    dtypes: Counter = Counter()
    for path in sorted(model_dir.glob("*.safetensors")):
        record = {"shard": path.name, "ok": False}
        try:
            header, header_len, data_start = _read_header(path)
            entries = {k: v for k, v in header.items() if k != "__metadata__"}
            expected = shard_keys.get(path.name, set())
            actual = set(entries)
            max_end = 0
            for key, entry in entries.items():
                shape = entry.get("shape")
                dtype = str(entry.get("dtype", ""))
                offsets = entry.get("data_offsets")
                if not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
                    header_errors.append(f"{path.name}: malformed header entry {key}")
                    continue
                if offsets[0] > offsets[1]:
                    header_errors.append(f"{path.name}: reversed offsets for {key}")
                max_end = max(max_end, int(offsets[1]))
                dtypes[dtype] += 1
                if key in seen:
                    header_errors.append(f"{path.name}: duplicate key {key} (also in {seen[key]})")
                else:
                    seen[key] = path.name
            expected_size = data_start + max_end
            actual_size = path.stat().st_size
            if actual_size != expected_size:
                header_errors.append(
                    f"{path.name}: size {actual_size} != header-derived {expected_size}"
                )
            if actual != expected:
                header_errors.append(
                    f"{path.name}: header/index drift missing={sorted(expected - actual)[:3]} "
                    f"extra={sorted(actual - expected)[:3]}"
                )
            record.update(
                {
                    "ok": True,
                    "header_bytes": header_len,
                    "tensors": len(entries),
                    "data_bytes": max_end,
                    "file_bytes": actual_size,
                }
            )
        except Exception as exc:
            header_errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
            record["error"] = str(exc)
        shards.append(record)

    local = {s["shard"] for s in shards}
    missing = sorted(s for s in shard_keys if s not in local)
    if missing:
        errors.append(f"{len(missing)} declared shard(s) not on disk")
    if not shards:
        header_errors.append("no .safetensors files on disk")
    errors = list(header_errors) + errors

    return {
        "model_dir": str(model_dir),
        "index_present": True,
        "shards_declared": len(shard_keys),
        "shards_present": len(shards),
        "shards_missing": missing,
        "tensor_headers": sum(s.get("tensors", 0) for s in shards),
        "dtypes": dict(dtypes),
        "shards": shards,
        "headers_ok": not header_errors,
        "complete": not missing and bool(shards),
        "errors": errors,
        "weights_read": False,
        "hardware_used": False,
        "note": "only safetensors headers were read; tensor payload was not opened",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--strict", action="store_true", help="fail if any header check fails")
    args = ap.parse_args()
    payload = audit(Path(args.model))
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if args.strict:
        return 0 if (payload.get("headers_ok") and payload.get("complete")) else 1
    return 0 if payload.get("headers_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
