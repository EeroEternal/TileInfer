#!/usr/bin/env python3
"""S1 (reproducible): build the DeepSeek-V4.1-Flash parameter tree and load the HF checkpoint on CPU.

Acceptance for S1 (``docs/notes/dsv41-facts.md`` §18): the 476 GB checkpoint is read in full, every
tensor's name/shape/dtype is checked against the config-derived tree, and the peak RSS is recorded.

    # dry run: headers vs the tree, no allocation (seconds, works on any box)
    python tools/dsv41/s1_load_cpu.py --dry-run

    # the real load (needs ~520 GB RAM)
    python tools/dsv41/s1_load_cpu.py --json-out /tmp/s1_load.json

Besides the two documented transformations (packed fp4 experts; ``wo_a`` folded to bf16), every
parameter must match its shard entry exactly, so a clean report is a per-tensor conformance proof.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tileinfer.models.dsv41 import (  # noqa: E402
    Dsv41Config,
    Dsv41Model,
    load_shards,
    shard_paths,
    verify_headers,
)

GB = 1e9


def _status_kb(field: str) -> int:
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith(field):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="/home/models/DeepSeek-V4.1-Flash")
    parser.add_argument("--config", default="", help="default: <model-dir>/config.json")
    parser.add_argument("--dry-run", action="store_true", help="meta device + headers only")
    parser.add_argument("--no-vision", action="store_true", help="skip the ViT/aligner tensors")
    parser.add_argument("--no-mtp", action="store_true", help="skip the DSpark stages")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--report-every", type=int, default=4000)
    args = parser.parse_args(argv)

    config_path = args.config or os.path.join(args.model_dir, "config.json")
    cfg = Dsv41Config.from_json(config_path)
    print(
        f"config: {config_path}\n"
        f"  layers={cfg.n_layers}+{cfg.n_mtp_layers} hidden={cfg.hidden_size} heads={cfg.num_attention_heads}"
        f"x{cfg.head_dim} experts={cfg.n_routed_experts}/{cfg.num_experts_per_tok}"
        f" engram={list(cfg.engram_layer_ids)} kv_src={list(cfg.kv_source_layers)}"
        f" idx_src={list(cfg.index_source_layers)}"
    )
    problems = cfg.validate()
    if problems:
        print("config warnings:")
        for item in problems:
            print("  -", item)

    paths = shard_paths(args.model_dir)
    print(f"shards: {len(paths)}")

    device = "meta" if args.dry_run else "cpu"
    t0 = time.time()
    model = Dsv41Model(cfg, device=device, with_vision=not args.no_vision, with_mtp=not args.no_mtp)
    n_params = sum(1 for _ in model.parameters())
    print(f"built parameter tree on {device}: {n_params} parameters in {time.time()-t0:.1f}s")
    if not args.dry_run:
        print(f"  model bytes: {model.n_parameters_bytes()/GB:.1f} GB")
        named = dict(model.named_parameters())
        offload = model.host_offload_names()
        offload_bytes = sum(
            named[name].numel() * named[name].element_size() for name in offload
        )
        del named
        print(
            f"  host-offload candidates (S4): {len(offload)} tensors, {offload_bytes/GB:.1f} GB"
        )

    if args.dry_run:
        report = verify_headers(model, paths)
        print("dry-run report:", report.summary())
        for label, items in (
            ("shape_mismatch", report.shape_mismatch[:10]),
            ("dtype_mismatch", report.dtype_mismatch[:10]),
            ("missing", report.missing[:10]),
            ("extra", report.extra[:10]),
        ):
            for item in items:
                print(f"  {label}: {item}")
        ok = report.ok
    else:
        t0 = time.time()
        report = load_shards(
            model,
            paths,
            verify=True,
            progress_every=args.report_every,
            log=lambda msg: print(msg, flush=True),
        )
        elapsed = time.time() - t0
        print(f"load report: {report.summary()}")
        print(f"  elapsed {elapsed/60:.1f} min ({report.bytes_in/GB/max(elapsed,1e-9):.2f} GB/s)")
        for label, items in (
            ("shape_mismatch", report.shape_mismatch[:10]),
            ("dtype_mismatch", report.dtype_mismatch[:10]),
            ("missing", report.missing[:10]),
            ("extra", report.extra[:10]),
        ):
            for item in items:
                print(f"  {label}: {item}")
        peak = _status_kb("VmHWM")
        rss = _status_kb("VmRSS")
        print(f"  peak RSS {peak/1024/1024:.1f} GiB, final RSS {rss/1024/1024:.1f} GiB")
        ok = report.ok and abs(peak - model.n_parameters_bytes() / 1024) < 0.5 * peak
        if args.json_out:
            payload = {
                "config": config_path,
                "params": n_params,
                "model_bytes": model.n_parameters_bytes(),
                "elapsed_s": elapsed,
                "peak_rss_kb": peak,
                "final_rss_kb": rss,
                "host_offload_tensors": len(offload),
                "host_offload_bytes": offload_bytes,
                "report": report.as_dict(),
            }
            with open(args.json_out, "w") as handle:
                json.dump(payload, handle, indent=2)
            print(f"  wrote {args.json_out}")

    print("RESULT:", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
