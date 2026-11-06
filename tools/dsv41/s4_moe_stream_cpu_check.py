#!/usr/bin/env python3
"""S4: is the ``moe_forward_streamed`` residual logical or device-specific?

``s4_layer_moe_stream.py`` shows paged-vs-resident bit-exact per expert and on the whole layer, yet
``moe_forward_streamed`` differs from ``moe_forward`` by 9.3 with the routed part 26% too small.  Running
the *same function* with ``device="cpu"`` and a CPU arena removes the NPU from the picture entirely:

* matches -> the bug only shows up on device (indexing / dtype of the gather), and the fix is local;
* differs -> it is pure logic in the routing accumulation, debuggable without a card.

    python tools/dsv41/s4_moe_stream_cpu_check.py
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tileinfer.models.dsv41 import Dsv41Config, load_shards, shard_paths  # noqa: E402
from tileinfer.models.dsv41.block import moe_forward, moe_forward_streamed  # noqa: E402
from tileinfer.models.dsv41.expert_stream import ExpertArena, expert_specs  # noqa: E402
from tileinfer.models.dsv41.modules import Dsv41Block  # noqa: E402

M = "/home/models/DeepSeek-V4.1-Flash"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=M)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--seqlen", type=int, default=4)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args(argv)
    from dataclasses import replace

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
        raise SystemExit("layer did not load cleanly")

    moe = block.ffn
    gen = torch.Generator().manual_seed(args.seed)
    x = torch.randn(1, args.seqlen, cfg.hidden_size, generator=gen).to(torch.bfloat16)
    flat = x.reshape(-1, cfg.hidden_size)

    # per-expert contributions *with* the routing weight, from both paths
    from tileinfer.models.dsv41.block import expert_forward, gate_forward

    weights, indices = gate_forward(moe.gate, flat, None)
    print("ids:", indices[0].tolist(), "weights:", [round(w, 5) for w in weights[0].tolist()])
    for e in sorted({int(v) for v in indices.flatten().tolist()}):
        idx, top = torch.where(indices == e)
        ref = expert_forward(moe.experts[e], flat[idx], weights[idx, top, None]).float()
        # my path, on the same rows and the same weight
        exp = moe.experts[e]
        lim = exp.swiglu_limit
        g = exp.w1.weight
        u = exp.w3.weight
        import tools.dsv41.ref_cpu as R
        go = R.linear(flat[idx], exp.w1.weight, exp.w1.scale).float()
        up = R.linear(flat[idx], exp.w3.weight, exp.w3.scale).float()
        if lim > 0:
            up = torch.clamp(up, min=-lim, max=lim)
            go = torch.clamp(go, max=lim)
        mid = torch.nn.functional.silu(go) * up
        mid = mid * weights[idx, top, None].float()
        mine = R.linear(mid.to(flat.dtype), exp.w2.weight, exp.w2.scale).float()
        d = (mine - ref).abs()
        print(f"  expert {e:3d}: rows={idx.tolist()} w={[round(float(v),5) for v in weights[idx, top]]} "
              f"ref|max|={ref.abs().max():.3e} mine|max|={mine.abs().max():.3e} max|diff|={d.max():.3e}")

    arena = ExpertArena(specs=expert_specs(moe.experts[0]), n_slots=64, pin=False)
    # replicate moe_forward_streamed's loop by hand, straight off the arena views
    arena2 = ExpertArena(specs=expert_specs(moe.experts[0]), n_slots=64, pin=False)
    y_manual = torch.zeros(flat.shape, dtype=torch.float32)
    for e in sorted({int(v) for v in indices.flatten().tolist()}):
        idx, top = torch.where(indices == e)
        arena2.bind(args.layer, e)
        arena2.pack(arena2.cache.slot_of((args.layer, e)), moe.experts[e])
        arena2.release(args.layer, e)
    arena2.to_device("cpu")
    for e in sorted({int(v) for v in indices.flatten().tolist()}):
        idx, top = torch.where(indices == e)
        exp = moe.experts[e]
        v = arena2.views(arena2.cache.slot_of((args.layer, e)), on_device=True)
        lim = exp.swiglu_limit
        go = R.linear(flat[idx], v["w1"], v["w1.scale"]).float()
        up = R.linear(flat[idx], v["w3"], v["w3.scale"]).float()
        if lim > 0:
            up = torch.clamp(up, min=-lim, max=lim)
            go = torch.clamp(go, max=lim)
        mid = torch.nn.functional.silu(go) * up
        y_manual[idx] += R.linear(mid.to(flat.dtype), v["w2"], v["w2.scale"]).float() * weights[
            idx, top, None
        ].float()
    y_manual = y_manual + expert_forward(moe.shared_experts, flat).float()
    y_cpu = moe_forward(moe, flat).float()
    print(f"hand-written arena loop vs moe_forward: max|diff|={(y_manual-y_cpu).abs().max():.3e}")
    y_str = moe_forward_streamed(
        moe, flat, args.layer, arena, device="cpu"
    ).float()
    d = (y_str - y_cpu).abs()
    print(f"cpu arena vs moe_forward: max|diff|={d.max():.3e} mean={d.mean():.3e} "
          f"exact={torch.equal(y_str, y_cpu)}")
    print("cache:", arena.snapshot())
    # whole-block integration: swap moe_forward for the streamed one and compare block outputs
    import tileinfer.models.dsv41.block as B
    from tileinfer.models.dsv41.block import ModelState, make_identity_pre_mix

    state = ModelState(cfg=cfg, max_seq_len=64, bsz=1, device="cpu")
    x4 = x.unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
    pre = make_identity_pre_mix(x4, cfg.hc_mult)
    out_ref, pre_ref = B.block_forward(block, x4, 0, pre.clone(), state)
    orig = B.moe_forward
    try:
        B.moe_forward = lambda moe, xx, mask=None: moe_forward_streamed(
            moe, xx, args.layer, arena, device="cpu", image_mask=mask
        )
        out_str, pre_str = B.block_forward(block, x4, 0, pre.clone(), state)
    finally:
        B.moe_forward = orig
    db = (out_str - out_ref).abs()
    print(f"whole block streamed vs reference: max|diff|={db.max():.3e} "
          f"exact={torch.equal(out_str, out_ref)} (pre_mix exact={torch.equal(pre_str, pre_ref)})")

    ok = d.max().item() <= 1e-2 and db.max().item() <= 1e-2
    print("RESULT:", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
