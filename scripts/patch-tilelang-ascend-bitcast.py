#!/usr/bin/env python3
"""Patch a `tilelang` Ascend wheel whose kernels fail to compile with

    numeric_limits.h: error: no member named 'bit_cast' in namespace 'std'

The wheel's `src/tl_templates/ascend/numeric_limits.h` uses `std::bit_cast` without including
`<bit>`, and the Bisheng compiler's standard headers on this CANN release do not pull it in, so every
kernel compile fails.  `__builtin_bit_cast` needs no header and works on the same compiler (this is
what the colleague's working environment already does).

    python scripts/patch-tilelang-ascend-bitcast.py <site-packages>/tilelang
"""
from __future__ import annotations

import pathlib
import re
import sys

PATTERN = re.compile(r"std::bit_cast<([^>]+)>\(")


def patch(root: pathlib.Path) -> int:
    changed = 0
    for path in sorted(root.rglob("*")):
        if path.suffix not in (".h", ".hpp", ".cce", ".cc") or not path.is_file():
            continue
        text = path.read_text(errors="ignore")
        if "std::bit_cast" not in text:
            continue
        path.write_text(PATTERN.sub(r"__builtin_bit_cast(\1, ", text))
        print(f"patched {path}")
        changed += 1
    return changed


if __name__ == "__main__":
    target = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    n = patch(target)
    print(f"{n} file(s) patched" if n else "nothing to patch")
