#!/usr/bin/env python3
"""Build a config-shim directory for checkpoints whose config vLLM 0.23 cannot load.

Only ``config.json`` is replaced; every other entry is a symlink to the original checkpoint,
so no weight/tokenizer bytes are copied and the source directory is never modified.  The
shim's config is produced by ``config_to_manifest.normalize_config``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config_to_manifest import normalize_config


def make_shim(model_dir: Path, shim_dir: Path) -> dict:
    config = model_dir / "config.json"
    if not config.is_file():
        raise ValueError(f"missing {config}")
    shim_dir.mkdir(parents=True, exist_ok=True)
    linked = []
    for child in sorted(model_dir.iterdir()):
        if child.name == "config.json":
            continue
        target = shim_dir / child.name
        if target.exists() or target.is_symlink():
            raise ValueError(f"shim already has {target}; use a fresh directory")
        target.symlink_to(child.resolve(), target_is_directory=child.is_dir())
        linked.append(child.name)
    normalized = normalize_config(json.loads(config.read_text(encoding="utf-8")))
    (shim_dir / "config.json").write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"model_dir": str(model_dir), "shim_dir": str(shim_dir), "linked": linked}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--shim", required=True)
    args = ap.parse_args()
    try:
        result = make_shim(Path(args.model), Path(args.shim))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
