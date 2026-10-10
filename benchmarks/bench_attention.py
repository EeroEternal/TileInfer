"""Micro-benchmark: TileInfer backends (and CANN's FIA, when it is available).

What we want out of this harness, in the order the project cares about:

1. *Correctness under load* — every configuration is compared against the reference oracle, so a
   fast-but-wrong kernel cannot look good in the table.
2. *Serving-relevant shapes* — batch sizes and KV lengths that vLLM-Ascend actually produces
   (short decode with a big batch, long-context decode with a small batch, and the ugly middle
   where the batch is skewed).
3. *A baseline that is not us* — ``npu_fused_inference_attention_score`` (FIA) when torch_npu
   exposes it, so "we are fast" is measured against the CANN operator, not against torch.

Usage::

    python benchmarks/bench_attention.py --backend tilelang --device npu
    python benchmarks/bench_attention.py --sweep --fia
    python benchmarks/bench_attention.py --quick
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from tileinfer import BatchAttention, RaggedMetadata  # noqa: E402
from tileinfer.testing.cache import build_paged_cache, random_q  # noqa: E402
from tileinfer.testing.reference import reference_attention  # noqa: E402
from tileinfer.utils import device_name, is_npu_available, synchronize  # noqa: E402


# ---------------------------------------------------------------------------------------
# shape sweeps
# ---------------------------------------------------------------------------------------


@dataclass
class Case:
    """One benchmark configuration."""

    batch: int
    kv_len: int
    h_q: int = 32
    h_kv: int = 8
    dim: int = 128
    page_size: int = 128
    qo_len: int = 1

    @property
    def label(self) -> str:
        return (
            f"b{self.batch}/kv{self.kv_len}/{self.h_q}x{self.h_kv}x{self.dim}"
            f"/p{self.page_size}"
        )

    def as_dict(self) -> dict:
        return {
            "batch": self.batch,
            "kv_len": self.kv_len,
            "num_qo_heads": self.h_q,
            "num_kv_heads": self.h_kv,
            "head_dim": self.dim,
            "page_size": self.page_size,
            "qo_len": self.qo_len,
        }

    def flops(self) -> float:
        """2 * (QK^T + PV) for a decode step, in FLOPs."""
        return 4.0 * self.batch * self.qo_len * self.h_q * self.kv_len * self.dim

    def bytes(self, dtype_size: int = 2) -> int:
        """K and V bytes that must be read once: the real limit for decode, not FLOPs.

        Decode with one query row per request is memory bound, so GB/s is the metric to watch;
        GFLOP/s mostly measures how well we *hide* the loads.
        """
        kv = 2.0 * self.batch * self.h_kv * self.kv_len * self.dim * dtype_size
        page_table = 4.0 * self.batch * max(1, self.kv_len // max(self.page_size, 1))
        q = self.batch * self.h_q * self.dim * dtype_size
        return int(kv + page_table + q)


DEFAULT_SWEEP: List[Case] = [
    Case(batch=1, kv_len=4096),
    Case(batch=8, kv_len=4096),
    Case(batch=32, kv_len=2048),
    Case(batch=64, kv_len=1024),
    Case(batch=128, kv_len=512),
    Case(batch=256, kv_len=128),
    Case(batch=8, kv_len=32768),
    Case(batch=1, kv_len=131072),
]

QUICK_SWEEP: List[Case] = [
    Case(batch=4, kv_len=1024),
    Case(batch=16, kv_len=512),
]

#: Decode shapes a serving engine actually produces: short-context/large-batch (chat stepping) and
#: long-context/small-batch (long documents, coding agents).  Sized to fit next to the vLLM
#: instance that already holds most of the HBM on the reference machine.
SERVING_SWEEP: List[Case] = [
    Case(batch=64, kv_len=512),
    Case(batch=32, kv_len=2048),
    Case(batch=16, kv_len=4096),
    Case(batch=8, kv_len=8192),
    Case(batch=4, kv_len=16384),
    Case(batch=1, kv_len=32768),
    Case(batch=1, kv_len=65536),
]

#: The three shapes that reproduced KI-1 before the split kernel's first failure, in order: two
#: unsplit shapes followed by the first split shape.  Kept as a regression harness for that bug.
REPRO_SWEEP: List[Case] = [
    Case(batch=64, kv_len=512),
    Case(batch=32, kv_len=2048),
    Case(batch=16, kv_len=4096),
]

LONG_SWEEP: List[Case] = [
    Case(batch=1, kv_len=32768),
    Case(batch=4, kv_len=32768),
    Case(batch=1, kv_len=131072),
]

SKEWED_SWEEP: List[Case] = []  # filled in by --skew: ragged batches, the real serving case


# ---------------------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------------------


@dataclass
class Result:
    label: str
    backend: str
    latency_ms: float
    throughput_gflops: float
    bandwidth_gbps: Optional[float] = None
    max_abs_err: Optional[float] = None
    note: str = ""
    shape: Dict = field(default_factory=dict)

    def as_row(self) -> str:
        err = "-" if self.max_abs_err is None else f"{self.max_abs_err:.2e}"
        bw = "-" if self.bandwidth_gbps is None else f"{self.bandwidth_gbps:.0f}"
        return (
            f"{self.label:<40} {self.backend:<18} {self.latency_ms:>9.3f} "
            f"{self.throughput_gflops:>9.1f} {bw:>8} {err:>9} {self.note}"
        )


def timeit(fn: Callable[[], torch.Tensor], device: torch.device, warmup: int, iters: int) -> float:
    """Median latency in ms over ``iters`` runs (median, not mean: NPUs have noisy tails)."""
    for _ in range(warmup):
        fn()
    synchronize(device)
    samples: List[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


# ---------------------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------------------


def fia_baseline(case: Case, dtype: torch.dtype, device: torch.device):
    """Wrap CANN's fused-infer-attention operator as a callable, if torch_npu has it."""
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return None, "torch_npu missing"
    fn = getattr(torch.ops.npu, "npu_fused_infer_attention_score", None)
    if fn is None:
        return None, "FIA op not exposed by torch_npu"
    return fn, ""


def make_case_tensors(case: Case, dtype: torch.dtype, device: torch.device, seed: int = 0):
    """Random paged cache + query for one case, with shuffled pages."""
    torch.manual_seed(seed)
    k = torch.randn(case.batch, case.h_kv, case.kv_len, case.dim, dtype=dtype, device=device)
    v = torch.randn(case.batch, case.h_kv, case.kv_len, case.dim, dtype=dtype, device=device)
    k_cache, v_cache, meta = build_paged_cache(
        k, v, page_size=case.page_size, shuffle_pages=True, seed=seed, device=device
    )
    q = random_q(case.batch, case.h_q, case.qo_len, case.dim, dtype=dtype, device=device)
    meta.qo_indptr = torch.arange(0, case.batch + 1, dtype=torch.int32, device=device) * case.qo_len
    return q, k_cache, v_cache, meta


def run_backend(
    name: str,
    case: Case,
    dtype: torch.dtype,
    device: torch.device,
    kwargs: dict,
    check: bool,
    warmup: int,
    iters: int,
) -> Result:
    from tileinfer.attention.backends.base import get_backend

    q, k_cache, v_cache, meta = make_case_tensors(case, dtype, device)
    attn = BatchAttention(backend=name, device=device, dtype=dtype)
    plan = attn.plan(
        kv_indptr=meta.kv_indptr,
        kv_indices=meta.kv_indices,
        kv_last_page_len=meta.kv_last_page_len,
        page_size=case.page_size,
        num_qo_heads=case.h_q,
        num_kv_heads=case.h_kv,
        head_dim=case.dim,
        k_cache=k_cache,
        v_cache=v_cache,
        **kwargs,
    )

    out = attn.run(q, k_cache, v_cache, plan=plan)
    synchronize(device)

    err: Optional[float] = None
    if check:
        # Reference on the CPU: torch's NPU path routes through CANN ops we do not control (and it
        # has wedged the device on large shapes - `aclnnGeTensor`, vector-core exception, after
        # which every later op fails).  The oracle is about numbers, not speed, so it runs where it
        # is boring and predictable.
        host_meta = RaggedMetadata(
            kv_indptr=plan.meta.kv_indptr.cpu(),
            kv_indices=plan.meta.kv_indices.cpu(),
            kv_last_page_len=plan.meta.kv_last_page_len.cpu(),
            page_size=plan.meta.page_size,
            qo_indptr=None if plan.meta.qo_indptr is None else plan.meta.qo_indptr.cpu(),
        )
        expected = reference_attention(
            q.cpu().float(), k_cache.cpu().float(), v_cache.cpu().float(), host_meta, causal=False
        )
        err = (out.cpu().float() - expected.float()).abs().max().item()

    latency = timeit(lambda: attn.run(q, k_cache, v_cache, plan=plan), device, warmup, iters)
    note = f"tiles={plan.schedule.num_tiles}"
    if plan.needs_merge:
        note += " (split)"
    dtype_size = torch.tensor([], dtype=dtype).element_size()
    return Result(
        label=case.label,
        backend=name,
        latency_ms=latency,
        throughput_gflops=case.flops() / (latency * 1e-3) / 1e9,
        bandwidth_gbps=case.bytes(dtype_size) / (latency * 1e-3) / 1e9,
        max_abs_err=err,
        note=note,
        shape=case.as_dict(),
    )


def run_fia(
    case: Case, dtype: torch.dtype, device: torch.device, warmup: int, iters: int
) -> Optional[Result]:
    """Time CANN's FIA on the same data, when the installed release exposes it.

    The op's signature moves between CANN releases, so any failure means "no baseline" rather
    than a bogus number.
    """
    fn, _reason = fia_baseline(case, dtype, device)
    if fn is None:
        return None
    q, k_cache, v_cache, _meta = make_case_tensors(case, dtype, device)
    try:  # pragma: no cover - depends on the installed CANN
        qf = q.permute(0, 2, 1, 3)  # [B, S, H, D]
        kf = k_cache.view(case.batch * -(-case.kv_len // case.page_size), case.page_size, case.h_kv, case.dim)
        fn(qf, kf, v_cache, None, None, None)
        synchronize(device)
    except Exception:
        return None
    latency = timeit(lambda: fn(qf, kf, v_cache, None, None, None), device, warmup, iters)
    return Result(
        label=case.label,
        backend="fia",
        latency_ms=latency,
        throughput_gflops=case.flops() / (latency * 1e-3) / 1e9,
        note="CANN baseline",
        shape=case.as_dict(),
    )


# ---------------------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="tilelang", help="tilelang | reference | auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sweep", action="store_true", help="run the default shape sweep")
    parser.add_argument("--quick", action="store_true", help="two shapes only")
    parser.add_argument(
        "--preset",
        choices=["quick", "serving", "long", "repro"],
        default=None,
        help="curated shape sets; 'serving' is the decode grid we care about",
    )
    parser.add_argument(
        "--dtype", choices=["bf16", "fp16"], default="bf16", help="compute dtype (Ascend 950: bf16)"
    )
    parser.add_argument("--skew", action="store_true", help="ragged, load-balanced batches")
    parser.add_argument("--fia", action="store_true", help="also try the CANN FIA baseline")
    parser.add_argument("--kv-tile-pages", type=int, default=0, help="split-KV tile size in pages")
    parser.add_argument("--no-check", action="store_true", help="skip the correctness check")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--json", default=None, help="write results as JSON to this path")
    args = parser.parse_args(argv)

    device = torch.device("npu", 0) if args.device == "auto" and is_npu_available() else torch.device(
        args.device if args.device != "auto" else "cpu"
    )
    presets = {
        "quick": QUICK_SWEEP,
        "serving": SERVING_SWEEP,
        "long": LONG_SWEEP,
        "repro": REPRO_SWEEP,
    }
    if args.preset:
        cases = presets[args.preset]
    else:
        cases = QUICK_SWEEP if args.quick else (DEFAULT_SWEEP if (args.sweep or not args.skew) else [])
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print(f"TileInfer micro-benchmark | device={device_name(device)} | torch={torch.__version__}")
    print(f"backends available: {BatchAttention.available_backends(available_only=True)}")
    header = (
        f"{'shape':<40} {'backend':<18} {'ms':>9} {'GFLOP/s':>9} {'GB/s':>8} {'maxerr':>9} note"
    )
    print(header)
    print("-" * len(header))

    results: List[Result] = []
    kwargs = {"kv_tile_pages": args.kv_tile_pages}
    for case in cases:
        try:
            results.append(
                run_backend(
                    args.backend,
                    case,
                    dtype,
                    device,
                    kwargs,
                    not args.no_check,
                    args.warmup,
                    args.iters,
                )
            )
        except Exception as exc:  # keep sweeping: one unsupported shape must not kill the run
            print(f"{case.label:<44} {args.backend:<12} {'-':>10} {'-':>10} {'-':>10} skipped: {exc}")
            continue
        print(results[-1].as_row())
        if args.fia:
            fia = run_fia(case, dtype, device, args.warmup, args.iters)
            if fia is not None:
                results.append(fia)
                print(fia.as_row())

    if args.skew:
        print("\nskeewed batches (ragged KV lengths) — where load balancing is supposed to pay off")
        torch.manual_seed(0)
        for label, kv_lens in (
            ("uniform", [2048] * 32),
            ("mild-skew", [4096] * 8 + [512] * 24),
            ("heavy-skew", [16384, 8192, 4096, 2048] + [256] * 28),
        ):
            batch = len(kv_lens)
            case = Case(batch=batch, kv_len=max(kv_lens))
            k = torch.randn(
                batch, case.h_kv, case.kv_len, case.dim, dtype=dtype, device=device
            )
            v = torch.randn_like(k)
            k_cache, v_cache, base_meta = build_paged_cache(
                k, v, page_size=case.page_size, shuffle_pages=True, device=device
            )
            indptr = [0]
            for L in kv_lens:
                indptr.append(indptr[-1] + -(-L // case.page_size))
            meta = RaggedMetadata(
                kv_indptr=torch.tensor(indptr, dtype=torch.int32, device=device),
                kv_indices=base_meta.kv_indices,
                kv_last_page_len=torch.tensor(
                    [case.page_size if L % case.page_size == 0 else L % case.page_size for L in kv_lens],
                    dtype=torch.int32,
                    device=device,
                ),
                page_size=case.page_size,
                qo_indptr=torch.arange(0, batch + 1, dtype=torch.int32, device=device),
            )
            q = random_q(batch, case.h_q, 1, case.dim, dtype=torch.float16, device=device)
            attn = BatchAttention(backend=args.backend, device=device)
            for kv_tile_pages in (0, 8, 64):
                try:
                    plan = attn.plan(
                        kv_indptr=meta.kv_indptr,
                        kv_indices=meta.kv_indices,
                        kv_last_page_len=meta.kv_last_page_len,
                        page_size=case.page_size,
                        num_qo_heads=case.h_q,
                        num_kv_heads=case.h_kv,
                        head_dim=case.dim,
                        kv_tile_pages=kv_tile_pages,
                        k_cache=k_cache,
                        v_cache=v_cache,
                    )
                except NotImplementedError as exc:
                    print(f"{label:<12} kv_tile_pages={kv_tile_pages:<4} n/a: {exc}")
                    continue
                latency = timeit(
                    lambda: attn.run(q, k_cache, v_cache, plan=plan), device, args.warmup, args.iters
                )
                flops = 4.0 * case.qo_len * case.h_q * case.dim * sum(kv_lens)
                print(
                    f"{label:<12} kv_tile_pages={kv_tile_pages:<4} tiles={plan.schedule.num_tiles:<5} "
                    f"balanced={str(plan.schedule.balanced):<5} {latency:>8.3f} ms "
                    f"{flops / (latency * 1e-3) / 1e9:>8.1f} GFLOP/s"
                )

    if args.json:
        with open(args.json, "w") as fh:
            json.dump([r.__dict__ for r in results], fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
