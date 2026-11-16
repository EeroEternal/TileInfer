#!/usr/bin/env python3
"""Check that a local checkpoint directory is complete enough to load.

Reads only filenames, sizes and the safetensors index; never opens tensor data and never
touches an accelerator.  Incomplete downloads are reported as missing/zero-byte shards so a
serve attempt is not wasted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def check(model_dir: Path) -> dict:
    required = ["config.json"]
    missing = [name for name in required if not (model_dir / name).is_file()]
    index_path = model_dir / "model.safetensors.index.json"
    shards: list[str] = []
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = sorted(set((index.get("weight_map") or {}).values()))
    else:
        shards = sorted(p.name for p in model_dir.glob("*.safetensors"))
    missing += [name for name in shards if not (model_dir / name).is_file()]
    empty = [name for name in shards if (model_dir / name).is_file() and (model_dir / name).stat().st_size == 0]
    tokenizer_candidates = [
        name for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json") if (model_dir / name).is_file()
    ]
    shards_on_disk = sorted(p.name for p in model_dir.glob("*.safetensors"))
    return {
        "model_dir": str(model_dir),
        "config_present": (model_dir / "config.json").is_file(),
        "index_present": index_path.is_file(),
        "shards_declared": len(shards),
        "shards_missing": missing,
        "shards_empty": empty,
        "tokenizer_files": tokenizer_candidates,
        "shards_on_disk": shards_on_disk,
        "ready_to_load": not missing and not empty and bool(tokenizer_candidates),
        # Dummy loading needs config + tokenizer only; it deliberately does not claim the
        # random-weight run is a correctness check.
        "ready_for_dummy": (model_dir / "config.json").is_file() and bool(tokenizer_candidates),
        "hardware_used": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    payload = check(Path(args.model))
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0 if payload["ready_to_load"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
