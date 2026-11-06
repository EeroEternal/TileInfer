#!/usr/bin/env python3
"""S4 milestone: the full 40-layer stack on the card, experts paged in from host memory.

Dense weights (~9 GiB) live on the card; the routed experts (268.9 GiB) and the two engram tables
(189.1 GiB) stay in host memory and are paged in per layer.  The acceptance for S4 is that this
reproduces the CPU oracle's first greedy token (``11111`` for "The capital of France is") with HBM
inside the watermark and no crash while the slots turn over.

    python tools/dsv41/s4_full_stack_device.py
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

from tileinfer.models.dsv41 import Dsv41Config, Dsv41Model, load_shards, shard_paths  # noqa: E402

M = "/home/models/DeepSeek-V4.1-Flash"
PROMPT = "The capital of France is"
EXPECTED = 11111


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=M)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--slots", type=int, default=96)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--gen", type=int, default=1, help="extra decode steps (1 = prefill only)")
    parser.add_argument("--batched", action="store_true", help="use the batched decode MoE")
    args = parser.parse_args(argv)

    import torch_npu  # noqa: F401
    from dataclasses import replace

    from tileinfer.models.dsv41.block import (
        ModelState,
        moe_forward_batched,
        moe_forward_streamed,
        transformer_forward,
    )
    from tileinfer.models.dsv41.engram import EngramHash
    from tileinfer.models.dsv41.expert_stream import ExpertArena, expert_specs

    dev = "npu:0"
    torch.set_default_dtype(torch.bfloat16)
    base = Dsv41Config.from_json(os.path.join(args.model_dir, "config.json"))
    cfg = replace(base, num_nextn_predict_layers=0, max_seq_len=args.max_seq_len)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    ids = tok.encode(args.prompt)
    print(f"prompt {args.prompt!r} -> {ids}")

    t0 = time.time()
    model = Dsv41Model(cfg, device="cpu", with_vision=False, with_mtp=False)
    rep = load_shards(model, shard_paths(args.model_dir), verify=True, progress_every=20000, log=print)
    print(f"loaded in {time.time()-t0:.0f}s: {rep.summary()}")
    if rep.missing or rep.extra or rep.shape_mismatch or rep.dtype_mismatch:
        raise SystemExit("weights did not load cleanly")

    free, total = torch.npu.mem_get_info()
    print(f"NPU HBM free={free/2**30:.1f} GiB / {total/2**30:.1f} GiB")

    # everything dense goes to the card; routed experts and engram tables stay on the host
    moved = 0
    kept = 0
    for name, param in model.named_parameters():
        if ".ffn.experts." in name or ".engram.embed." in name:
            kept += param.numel() * param.element_size()
            continue
        param.data = param.data.to(dev)
        moved += param.numel() * param.element_size()
    print(f"on card: {moved/2**30:.2f} GiB dense | on host: {kept/2**30:.2f} GiB experts+engram tables")
    torch.npu.empty_cache()
    free, total = torch.npu.mem_get_info()
    print(f"NPU HBM free after placement={free/2**30:.1f} GiB")

    arena = ExpertArena(specs=expert_specs(model.layers[0].ffn.experts[0]), n_slots=args.slots, pin=True)
    print(f"arena: {args.slots} slots x {arena.slot_bytes/2**20:.2f} MiB = "
          f"{arena.slot_bytes*args.slots/2**30:.2f} GiB")

    state = ModelState(
        cfg=cfg,
        max_seq_len=args.max_seq_len,
        bsz=1,
        device=dev,
        engram_hash=EngramHash(cfg, tokenizer=tok, max_seq_len=args.max_seq_len, device="cpu"),
    )

    def streamed_moe(moe, x, image_mask=None):
        layer_id = getattr(moe, "_layer_id_for_stream", 0)
        if args.batched and x.numel() // x.shape[-1] == 1:  # prefill falls back to the streaming path
            return moe_forward_batched(moe, x, layer_id, arena, device=dev, image_mask=image_mask)
        return moe_forward_streamed(
            moe, x, layer_id, arena, device=dev, image_mask=image_mask
        )

    # tag each MoE with its layer so the arena keys are unique per layer
    for i, block in enumerate(model.layers):
        block.ffn._layer_id_for_stream = i

    captured = {}
    cap_layers = {}
    t0 = time.time()
    with torch.no_grad():
        logits, _ = transformer_forward(
            model, torch.tensor([ids]), state, 0, capture=captured, moe_fn=streamed_moe,
            capture_layers=cap_layers,
        )
    torch.npu.synchronize()
    fwd = time.time() - t0
    top5 = torch.topk(logits.flatten().float(), 5)
    printed = [(int(i), round(float(v), 3)) for i, v in zip(top5.indices.tolist(), top5.values.tolist())]
    token = int(logits.flatten().argmax())
    # reference values for the vLLM bisect (see docs/notes/dsv41-vllm-integration-plan.md §23):
    # compare these against the DSV41_DEBUG line that the adapter prints in compute_logits
    _h = captured.get("h")
    if _h is not None:
        print("hidden[:4] =", [round(float(v), 5) for v in _h.flatten()[:4].tolist()])
    # per-layer reference for the vLLM bisect (plan §23): the first layer that differs is the culprit
    for _lid in sorted(cap_layers)[:3]:
        _v = cap_layers[_lid]
        print(f"layer[{_lid}][:4] =", [round(float(x), 5) for x in _v.flatten()[:4].tolist()])
    print("logits top5:", printed, "argmax:", token)
    print(f"\nforward {fwd:.1f}s")
    print(f"token = {token} ({tok.decode([token])!r})  top5 = {printed}")
    print("arena:", arena.snapshot())
    # the oracle token is only pinned for the default prompt\n    ok = True if args.prompt != PROMPT else token == EXPECTED

    # decode: each step is one token, so this is the TPOT the S5 curves are built from
    out_ids = [token]
    tpots = []
    pos = len(ids)
    for step in range(max(0, args.gen - 1)):
        t1 = time.perf_counter()
        dec_timings = {}
        with torch.no_grad():
            logits, _ = transformer_forward(
                model, torch.tensor([[out_ids[-1]]]), state, pos, moe_fn=streamed_moe,
                timings=dec_timings,
            )
        torch.npu.synchronize()
        dt = time.perf_counter() - t1
        tpots.append(dt)
        nxt = int(logits.flatten().argmax())
        out_ids.append(nxt)
        pos += 1
        free_now, _ = torch.npu.mem_get_info()
        moe_ms = 1000 * dec_timings.get("moe", 0.0)
        print(f"  step {step+1}: moe {moe_ms:.0f} ms of {1000*dt:.0f} ms | token={nxt} ({tok.decode([nxt])!r}) "
              f"{1000*dt:.0f} ms  HBM free={free_now/2**30:.1f} GiB")
    if tpots:
        mean = sum(tpots) / len(tpots)
        print(f"\nTPOT mean {1000*mean:.0f} ms ({1/mean:.1f} tok/s) over {len(tpots)} steps")
    print(f"generated text: {tok.decode(out_ids)!r}")
    print("arena:", arena.snapshot())
    ok = True if args.prompt != PROMPT else token == EXPECTED
    print("RESULT:", "OK" if ok else f"MISMATCH (expected {EXPECTED} for the default prompt)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
