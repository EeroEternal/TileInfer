#!/usr/bin/env python3
"""S4 step 3c-2: one whole MoE layer, paged through ``ExpertArena`` vs resident on the device.

The arena round-trip is already bit-exact per expert (``s4_arena_device.py``).  What this adds is the
*composition*: a real gate picks six experts for a real token, and the layer output is computed twice --
once with every weight already on the device (the "resident" baseline) and once through the paging path
(pack -> one H2D per layer -> slice -> GEMM) -- then compared.  That is the exact substitution
``moe_forward`` has to make for S4, so a mismatch here is a mismatch there.

    python tools/dsv41/s4_layer_moe_stream.py --layer 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tileinfer.models.dsv41 import Dsv41Config, load_shards, shard_paths  # noqa: E402
from tileinfer.models.dsv41.block import gate_forward  # noqa: E402
from tileinfer.models.dsv41.expert_stream import ExpertArena, expert_specs  # noqa: E402
from tileinfer.models.dsv41.modules import Dsv41Block  # noqa: E402
from tools.dsv41 import ref_cpu  # noqa: E402

M = "/home/models/DeepSeek-V4.1-Flash"


def expert_out(x: torch.Tensor, w1, s1, w2, s2, w3, s3, limit: float) -> torch.Tensor:
    """``Expert.forward`` on a batch of tokens: swiglu with the checkpoint's clamp."""
    gate = ref_cpu.linear(x, w1, s1)
    up = ref_cpu.linear(x, w3, s3)
    if limit:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    act = torch.nn.functional.silu(gate.float()).to(x.dtype) * up
    return ref_cpu.linear(act, w2, s2)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=M)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--seqlen", type=int, default=4)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args(argv)

    import torch_npu  # noqa: F401
    from dataclasses import replace

    dev = "npu:0"
    torch.set_default_dtype(torch.bfloat16)
    base = Dsv41Config.from_json(os.path.join(args.model_dir, "config.json"))
    cfg = replace(
        base,
        num_hidden_layers=args.layer + 1,
        num_nextn_predict_layers=0,
        max_seq_len=64,
        compress_ratios=base.compress_ratios[: args.layer + 1],
    )

    class _Layers(torch.nn.Module):
        def __init__(self, blk, layer_id):
            super().__init__()
            self.layers = torch.nn.ModuleDict({str(int(layer_id)): blk})

    block = Dsv41Block(cfg, args.layer, device="cpu")
    rep = load_shards(
        _Layers(block, args.layer),
        shard_paths(args.model_dir),
        verify=True,
        progress_every=100000,
        name_filter=lambda n: n.startswith(f"layers.{args.layer}."),
    )
    print("load:", rep.summary())
    if rep.missing or rep.extra or rep.shape_mismatch or rep.dtype_mismatch:
        raise SystemExit("layer did not load cleanly; refusing to benchmark garbage weights")

    moe = block.ffn
    gen = torch.Generator().manual_seed(args.seed)
    x = torch.randn(1, args.seqlen, cfg.hidden_size, generator=gen).to(torch.bfloat16)

    flat_in = x.reshape(-1, cfg.hidden_size)
    # a real gate decision: the same code the model runs
    weights_all, indices_all = gate_forward(moe.gate, flat_in, None)
    ids = indices_all[0].tolist()
    scores = weights_all[0].tolist()
    print(f"gate picked experts {ids} with scores {[round(s, 4) for s in scores]}")

    limit = float(getattr(cfg, "swiglu_limit", 0.0) or 0.0)
    flat = x.reshape(-1, cfg.hidden_size)
    y_ref = torch.zeros(flat.shape[0], cfg.hidden_size, dtype=torch.bfloat16)
    for eid, sc in zip(ids, scores):
        w = moe.experts[eid]
        y_ref += expert_out(
            flat, w.w1.weight, w.w1.scale, w.w2.weight, w.w2.scale,
            w.w3.weight, w.w3.scale, limit,
        ).float().to(torch.bfloat16) * torch.tensor(sc, dtype=torch.bfloat16)

    # resident baseline: every needed expert already on the device
    t0 = time.perf_counter()
    res = {}
    for eid in ids:
        w = moe.experts[eid]
        res[eid] = {k: getattr(w, k).weight.to(dev) for k in ("w1", "w2", "w3")} | {
            f"{k}_s": getattr(w, k).scale.to(dev) for k in ("w1", "w2", "w3")
        }
    torch.npu.synchronize()
    resident_ms = 1000 * (time.perf_counter() - t0)
    y_res = torch.zeros(flat.shape[0], cfg.hidden_size, dtype=torch.float32)
    xd = flat.to(dev)
    t0 = time.perf_counter()
    for eid, sc in zip(ids, scores):
        r = res[eid]
        y_res += expert_out(
            xd, r["w1"], r["w1_s"], r["w2"], r["w2_s"], r["w3"], r["w3_s"], limit
        ).float().cpu() * sc
    torch.npu.synchronize()
    resident_compute_ms = 1000 * (time.perf_counter() - t0)

    # paged: one arena, one H2D for the whole layer
    arena = ExpertArena(specs=expert_specs(moe.experts[0]), n_slots=max(len(ids), 64), pin=True)
    t0 = time.perf_counter()
    for eid in ids:
        slot, _ = arena.bind(args.layer, eid)
        arena.pack(slot, moe.experts[eid])
        arena.release(args.layer, eid)
    arena.to_device(dev)  # warm-up allocation, not timed
    torch.npu.synchronize()
    arena.to_device(dev)
    torch.npu.synchronize()
    paged_copy_ms = 1000 * (time.perf_counter() - t0)

    y_paged = torch.zeros(flat.shape[0], cfg.hidden_size, dtype=torch.float32)
    t0 = time.perf_counter()
    for eid, sc in zip(ids, scores):
        v = arena.views(arena.cache.slot_of((args.layer, eid)), on_device=True)
        y_paged += expert_out(
            xd, v["w1"], v["w1.scale"], v["w2"], v["w2.scale"], v["w3"], v["w3.scale"], limit
        ).float().cpu() * sc
    torch.npu.synchronize()
    paged_compute_ms = 1000 * (time.perf_counter() - t0)

    # per-expert isolation: is the device math identical to expert_forward() on the same weights?
    from tileinfer.models.dsv41.block import expert_forward

    print("\nper-expert (all tokens, no routing weight):")
    for eid in ids:
        exp = moe.experts[eid]
        want = expert_forward(exp, flat).float()
        v = arena.views(arena.cache.slot_of((args.layer, eid)), on_device=True)
        g = ref_cpu.linear(xd, v["w1"], v["w1.scale"]).float()
        u = ref_cpu.linear(xd, v["w3"], v["w3.scale"]).float()
        lim = exp.swiglu_limit
        if lim > 0:
            u = torch.clamp(u, min=-lim, max=lim)
            g = torch.clamp(g, max=lim)
        o = ref_cpu.linear((torch.nn.functional.silu(g) * u).to(xd.dtype), v["w2"], v["w2.scale"])
        d = (o.float().cpu() - want).abs()
        print(f"  expert {eid:3d}: max|diff|={d.max():.3e} mean={d.mean():.3e}")

    # the production entry point: moe_forward_streamed() from block.py
    from tileinfer.models.dsv41.block import moe_forward_streamed

    x4 = x.unsqueeze(2).repeat(1, 1, 1, 1)  # block layout: [b, s, hc=1, dim] is not used here
    t0 = time.perf_counter()
    y_fn = moe_forward_streamed(
        moe, flat, args.layer, arena, device=dev, swiglu_limit=limit
    ).float()
    torch.npu.synchronize()
    fn_ms = 1000 * (time.perf_counter() - t0)
    # the right reference: moe_forward() itself (it also adds the shared expert, y_ref above does not)
    from tileinfer.models.dsv41.block import moe_forward

    t0 = time.perf_counter()
    y_cpu = moe_forward(moe, flat).float()
    cpu_ms = 1000 * (time.perf_counter() - t0)
    d_fn = (y_fn - y_cpu).abs()
    d_cpu_res = (y_cpu - y_ref.float()).abs()
    print(f"cpu moe_forward vs routed-only ref: max|diff|={d_cpu_res.max():.3e} (shared expert, {cpu_ms:.1f} ms)")
    print(f"moe_forward_streamed vs cpu moe_forward: max|diff|={d_fn.max():.3e} "
          f"mean={d_fn.mean():.3e} ({fn_ms:.1f} ms)")
    y_sh = expert_forward(moe.shared_experts, flat).float()
    print(f"  routed part only: cpu {(y_cpu - y_sh).abs().max():.3e} vs ref "
          f"{(y_ref.float() - y_sh).abs().max():.3e} | streamed {(y_fn - y_sh).abs().max():.3e}")
    print(f"  shared part: cpu {y_sh.abs().max():.3e} streamed {y_sh.abs().max():.3e}")
    print(f"  row-wise |diff| per token: {[round(float(v), 4) for v in d_fn.max(dim=1).values]}")

    d_res = (y_res - y_ref.float()).abs()
    d_paged = (y_paged - y_ref.float()).abs()
    d_cross = (y_paged - y_res).abs()
    print(f"\nresident vs cpu ref : max|diff|={d_res.max():.3e} mean={d_res.mean():.3e}")
    print(f"paged    vs cpu ref : max|diff|={d_paged.max():.3e} mean={d_paged.mean():.3e}")
    print(f"paged    vs resident: max|diff|={d_cross.max():.3e}  bit-exact={torch.equal(y_paged, y_res)}")
    print(f"timing: resident load {resident_ms:.1f} ms, resident compute {resident_compute_ms:.1f} ms, "
          f"paged (pack+2xH2D) {paged_copy_ms:.2f} ms, paged compute {paged_compute_ms:.1f} ms")
    ok = d_cross.max().item() == 0.0
    ok = ok and d_fn.max().item() <= 1.0
    print("RESULT:", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
