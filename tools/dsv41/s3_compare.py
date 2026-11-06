#!/usr/bin/env python3
"""S3: compare two ``s3_first_token.py`` artifacts (greedy token + top-5 logits).

    python tools/dsv41/s3_compare.py /tmp/s3_official.json /tmp/s3_ours.json

Exit code 0 when the greedy tokens match and the top-5 sets overlap (the logits themselves are only
comparable within bf16 resolution across different arithmetic, see facts section 22.6).
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--tol", type=float, default=1e-2)
    args = parser.parse_args(argv)

    left = json.load(open(args.left))
    right = json.load(open(args.right))
    print(f"left : impl={left['impl']:8s} prompt_ids={len(left['prompt_ids'])} "
          f"token={left['token']} ({left['token_text']!r}) peak={left['peak_rss_kb']/1024/1024:.0f} GiB")
    print(f"right: impl={right['impl']:8s} prompt_ids={len(right['prompt_ids'])} "
          f"token={right['token']} ({right['token_text']!r}) peak={right['peak_rss_kb']/1024/1024:.0f} GiB")

    same_prompt = left["prompt_ids"] == right["prompt_ids"]
    same_token = left["token"] == right["token"]
    l_top = {int(idx): float(val) for idx, val in left["top5"]}
    r_top = {int(idx): float(val) for idx, val in right["top5"]}
    overlap = sorted(set(l_top) & set(r_top))
    diffs = {i: abs(l_top[i] - r_top[i]) for i in overlap}
    worst = max(diffs.values()) if diffs else float("nan")

    print(f"prompt ids identical: {same_prompt}")
    print(f"greedy token identical: {same_token}  (left {left['token']} / right {right['token']})")
    print(f"top-5 overlap: {len(overlap)}/5 {overlap}")
    print(f"max |logit diff| over the overlap: {worst:.3e} (tol {args.tol:g})")
    for side in (left, right):
        print(f"  {side['impl']:8s} load: {side['load']['summary']}")

    ok = same_prompt and same_token and len(overlap) >= 4 and (worst <= args.tol or worst != worst)
    print("RESULT:", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
