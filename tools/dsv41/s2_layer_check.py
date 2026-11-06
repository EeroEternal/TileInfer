#!/usr/bin/env python3
"""S2 (reproducible): one attention layer, ours vs the *official* ``inference/model.py``.

The official layer is the oracle: this script imports ``inference/model.py`` unchanged and swaps its
six tilelang kernels for the torch/CPU transcription in :mod:`tools.dsv41.ref_cpu`, so the official
``Attention``/``Compressor``/``Indexer`` code itself runs -- the comparison is against the reference's
own math, not against a re-implementation of it.

Both sides get the *same* weights: the oracle's parameters are filled with synthetic values in the
checkpoint's storage dtypes (fp8 + e8m0 block-32 scales, bf16, fp32), then copied by name into our
``tileinfer.models.dsv41`` layer.  Both then see the same input through the same step sequence
(prefill + decodes), and the per-step outputs and the discrete top-k index lists are compared.

    # CPU vs CPU (works on any box, seconds)
    python tools/dsv41/s2_layer_check.py --layers 2 --steps 8,1,1,1

    # a whole stream segment: layer 2 publishes the compressed KV, 3..7 reuse it
    python tools/dsv41/s2_layer_check.py --chain 2,3,4,5,6,7
    python tools/dsv41/s2_layer_check.py --chain 20,21,22,23,24

    # device: ours on NPU, oracle on CPU
    python tools/dsv41/s2_layer_check.py --layers 20 --device npu

Exit code 0 when every compared tensor is within ``--tol`` (default 1e-2, the bf16 acceptance bar)
and every discrete index list is identical.  A standalone (``--layers``) run is only meaningful for
layers that publish their own stream (2, 8, 14, 20); consumers read what an earlier layer wrote, so
run those as a ``--chain``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tileinfer.models.dsv41 import Dsv41Config  # noqa: E402
from tileinfer.models.dsv41.layer import (  # noqa: E402
    LayerRuntime,
    SharedAttentionState,
    attention_forward,
)
from tileinfer.models.dsv41.modules import Dsv41Attention  # noqa: E402
from tools.dsv41 import ref_cpu  # noqa: E402

M = "/home/models/DeepSeek-V4.1-Flash"
E8M0 = torch.float8_e8m0fnu


def official_model(model_dir: str = M):
    """Import the read-only oracle and point its kernel names at the CPU transcription."""
    inf = os.path.join(model_dir, "inference")
    if inf not in sys.path:
        sys.path.insert(0, inf)
    import model as official  # noqa: E402  (official, read-only)

    official.act_quant = ref_cpu.act_quant
    official.fp4_act_quant = ref_cpu.fp4_act_quant
    official.fp8_gemm = ref_cpu.fp8_gemm
    official.fp4_gemm = ref_cpu.fp4_gemm
    official.sparse_attn = ref_cpu.sparse_attn
    official.hc_split_sinkhorn = ref_cpu.hc_split_sinkhorn
    return official


def oracle_args(official, cfg: Dsv41Config, max_seq_len: int, bsz: int):
    """``ModelArgs`` with the released shapes (the oracle's own flat spelling)."""
    return official.ModelArgs(
        max_batch_size=bsz,
        max_seq_len=max_seq_len,
        temperature=0.0,  # greedy: sample() returns argmax, so argmax comparisons are exact
        dtype="fp8",
        expert_dtype="fp4",
        vocab_size=cfg.vocab_size,
        dim=cfg.hidden_size,
        moe_inter_dim=cfg.moe_intermediate_size,
        n_layers=cfg.n_layers,
        n_mtp_layers=cfg.n_mtp_layers,
        n_heads=cfg.num_attention_heads,
        n_routed_experts=cfg.n_routed_experts,
        n_shared_experts=cfg.n_shared_experts,
        n_activated_experts=cfg.num_experts_per_tok,
        score_func=cfg.scoring_func,
        route_scale=cfg.routed_scaling_factor,
        norm_topk_prob=True,
        swiglu_limit=cfg.swiglu_limit,
        q_lora_rank=cfg.q_lora_rank,
        head_dim=cfg.head_dim,
        rope_head_dim=cfg.rope_head_dim,
        norm_eps=cfg.rms_norm_eps,
        o_groups=cfg.o_groups,
        o_lora_rank=cfg.o_lora_rank,
        window_size=cfg.sliding_window,
        compress_ratios=cfg.full_compress_ratios,
        kv_source_layers=cfg.kv_source_layers,
        index_source_layers=cfg.index_source_layers,
        compress_rope_theta=cfg.compress_rope_theta,
        original_seq_len=cfg.original_seq_len,
        rope_theta=cfg.rope_theta,
        rope_factor=cfg.rope_factor,
        beta_fast=cfg.beta_fast,
        beta_slow=cfg.beta_slow,
        index_n_heads=cfg.index_n_heads,
        index_head_dim=cfg.index_head_dim,
        index_topk=cfg.index_topk,
        candidate_source_layer=cfg.candidate_source_layer,
        candidate_topk_blocks=cfg.candidate_topk_blocks,
        candidate_block_size=cfg.candidate_block_size,
        # engram: the oracle needs exactly these to build EngramLayout/NgramHashState
        engram_layer_ids=cfg.engram_layer_ids,
        engram_num_embeddings=cfg.engram_num_embeddings,
        engram_max_ngram_size=cfg.engram_max_ngram_size,
        engram_vocab_size=cfg.engram_vocab_size,
        engram_n_heads=cfg.engram_n_heads,
        engram_head_dim=cfg.engram_head_dim,
        engram_pad_id=cfg.engram_pad_token_id,
        engram_compressed_vocab_size=cfg.engram_compressed_vocab_size,
        hc_mult=cfg.hc_mult,
        hc_sinkhorn_iters=cfg.hc_sinkhorn_iters,
        hc_eps=cfg.hc_eps,
        vision_n_layers=cfg.vision_num_hidden_layers,
        vision_dim=cfg.vision_hidden_size,
        vision_n_heads=cfg.vision_num_attention_heads,
        vision_inter_dim=cfg.vision_intermediate_size,
        vision_patch_size=cfg.vision_patch_size,
        vision_downsample_ratio=cfg.vision_downsample_ratio,
        vision_rope_theta=cfg.vision_rope_theta,
    )


# =============================================================================================
# Synthetic weights, shared by both sides
# =============================================================================================


GENUINE_FP32 = ("attn_sink", "gate.bias", "hc_")


def fill_weights(params: dict, seed: int = 1234, std: float = 0.02) -> None:
    """Fill parameters with synthetic values in the checkpoint's own storage dtypes.

    The *scale layout* decides how an fp8 tensor is quantized, and it is read off the scale itself:
    a linear weight has one e8m0 per 32x32 block (``[out/32, in/32]``), while an engram table row has
    one per 32 columns (``[rows, dim/32]`` -- its first dim matches the weight's).  Packed fp4 expert
    weights have no CPU cast, so they are packed by hand.
    """
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in params.items():
            if name.endswith(".scale"):
                continue  # written together with its weight below
            scale = params.get(name[: -len("weight")] + "scale") if name.endswith("weight") else None
            if name.endswith("weight") and param.dtype == torch.float8_e4m3fn:
                w = torch.randn(param.shape, generator=gen) * std

                if (
                    scale is not None
                    and scale.shape[0] == param.shape[0]
                    and scale.shape[1] == param.shape[1] // 32
                ):
                    # row-block layout (engram table): one e8m0 per 32 columns of the row
                    groups = w.unflatten(-1, (-1, 32))
                    amax = groups.abs().amax(dim=-1).clamp_min(1e-4)
                    s = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
                    q = (groups / s.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
                    param.copy_(q.flatten(-2))
                    scale.copy_(s.to(torch.float8_e8m0fnu))
                else:
                    w8, s = ref_cpu.quantize_weight_fp8_block32(w)
                    if tuple(scale.shape) != tuple(s.shape):
                        raise RuntimeError(
                            f"{name}: scale shape {tuple(s.shape)} != param {tuple(scale.shape)}"
                        )
                    param.copy_(w8)
                    scale.copy_(s)
            elif param.dtype == torch.float4_e2m1fn_x2:
                # packed fp4 expert weight: quantize + pack by hand (no CPU cast to that dtype)
                w = torch.randn(param.shape[0], param.shape[1] * 2, generator=gen) * std
                packed, s = ref_cpu.quantize_weight_fp4_block32(w)
                param.view(torch.uint8).copy_(packed)
                if tuple(scale.shape) != tuple(s.shape):
                    raise RuntimeError(
                        f"{name}: scale shape {tuple(s.shape)} != param {tuple(scale.shape)}"
                    )
                scale.copy_(s)
            elif name.endswith("attn_sink"):
                param.copy_(torch.randn(param.shape, generator=gen) * 0.5)
            elif name.endswith("norm.weight"):
                param.copy_((1.0 + 0.02 * torch.randn(param.shape, generator=gen)).to(param.dtype))
            elif param.dtype in (torch.bfloat16, torch.float32):
                value = torch.randn(param.shape, generator=gen) * std
                if param.dtype == torch.float32 and not any(
                    key in name for key in GENUINE_FP32
                ):
                    # these are fp32 *in the model* but bf16 in the checkpoint (the reference
                    # upcasts at load); generate bf16-representable values so both sides match.
                    value = value.to(torch.bfloat16).float()
                param.copy_(value.to(param.dtype))


def copy_weights(src: dict, dst: dict) -> None:
    missing = [n for n in dst if n not in src]
    extra = [n for n in src if n not in dst]
    if missing or extra:
        raise RuntimeError(f"parameter sets differ: missing={missing[:5]} extra={extra[:5]}")
    with torch.no_grad():
        for name, param in dst.items():
            param.copy_(src[name])


# =============================================================================================
# The run
# =============================================================================================


@dataclass
class Comparison:
    layer_id: int
    step: int
    max_abs: float = 0.0
    mean_abs: float = 0.0
    ref_absmax: float = 0.0
    max_ulp: float = 0.0
    idx_match: bool = True
    idx_total: int = 0


def _max_ulp(ref: torch.Tensor, diff: torch.Tensor) -> float:
    """Worst deviation expressed in bf16 units in the last place (1 ULP == numerically equal)."""
    if ref.numel() == 0:
        return 0.0
    _, exponent = torch.frexp(ref.float().abs())
    ulp = torch.pow(2.0, (exponent - 8).float())  # bf16: 8 significand bits, incl. the implicit one
    ratio = diff / ulp.clamp_min(torch.finfo(torch.float32).tiny)
    return float(ratio.max())


def run_chain(
    official,
    cfg: Dsv41Config,
    layer_ids,
    steps,
    bsz: int = 1,
    max_seq_len: int = 32,
    device: str = "cpu",
    seed: int = 1234,
    verbose: bool = True,
):
    """Run the same layer chain on both sides and compare every output tensor, step by step."""
    torch.set_default_dtype(torch.bfloat16)
    args = oracle_args(official, cfg, max_seq_len, bsz)

    oracle_layers, our_layers, runtimes = [], [], []
    shared = SharedAttentionState()
    for lid in layer_ids:
        oracle_layer = official.Attention(lid, args)
        params = dict(oracle_layer.named_parameters())
        fill_weights(params, seed=seed + 17 * lid)
        ours = Dsv41Attention(cfg, lid, device="cpu")
        copy_weights(params, dict(ours.named_parameters()))
        if device != "cpu":
            ours = ours.to(device)
        rt = LayerRuntime(
            cfg=cfg,
            layer_id=lid,
            shared=shared,
            bsz=bsz,
            max_seq_len=max_seq_len,
            device=device,
        ).build()
        oracle_layers.append(oracle_layer)
        our_layers.append(ours)
        runtimes.append(rt)

    results = []
    total_len = sum(steps)
    gen = torch.Generator().manual_seed(seed)
    x_seq = torch.randn(bsz, total_len, cfg.hidden_size, dtype=torch.bfloat16, generator=gen)
    pos = 0
    for step_index, seqlen in enumerate(steps):
        start_pos = pos
        x_step = x_seq[:, pos : pos + seqlen].contiguous()
        x_ours = x_step.to(device)
        for olayer, mlayer, rt in zip(oracle_layers, our_layers, runtimes):
            ref = olayer.forward(x_step.clone(), start_pos).detach()
            got = attention_forward(mlayer, x_ours.clone(), start_pos, rt).detach().float().cpu()
            diff = (got - ref.float()).abs()
            cmp = Comparison(layer_id=mlayer.layer_id, step=step_index)
            cmp.max_abs = float(diff.max())
            cmp.mean_abs = float(diff.mean())
            cmp.ref_absmax = float(ref.float().abs().max())
            cmp.max_ulp = _max_ulp(ref, diff)
            ref_idx = official.shared_attn.topk_idxs
            our_idx = shared.topk_idxs
            if ref_idx is not None and our_idx is not None:
                ri = ref_idx.detach().cpu().int()
                oi = our_idx.detach().cpu().int()
                cmp.idx_total = int(ri.numel())
                cmp.idx_match = bool(ri.shape == oi.shape and torch.equal(ri, oi))
            results.append(cmp)
            if verbose:
                passed = cmp.max_abs <= 1e-2 or cmp.max_ulp <= 1.0
                flag = "" if passed and cmp.idx_match else "   <-- FAIL"
                print(
                    f"  layer {cmp.layer_id:2d} step {step_index} len {seqlen:2d} "
                    f"max|diff|={cmp.max_abs:.3e} ({cmp.max_ulp:.2f} bf16 ulp) "
                    f"mean={cmp.mean_abs:.3e} (ref |x|max={cmp.ref_absmax:.2f}) "
                    f"idx_match={cmp.idx_match} (idx {cmp.idx_total}){flag}"
                )
        pos += seqlen
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=M)
    parser.add_argument("--layers", default="2", help="comma separated layer ids, run standalone")
    parser.add_argument(
        "--chain", default="", help="comma separated layer ids run as one chain (shares streams)"
    )
    parser.add_argument("--steps", default="8,1,1,1", help="token counts per step; first is prefill")
    parser.add_argument(
        "--max-seq-len", type=int, default=32, help="cache capacity; must cover sum(steps)"
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "npu"])
    parser.add_argument("--tol", type=float, default=1e-2)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args(argv)

    config_path = os.path.join(args.model_dir, "config.json")
    cfg = Dsv41Config.from_json(config_path)
    official = official_model(args.model_dir)

    layer_ids = [int(x) for x in (args.chain or args.layers).split(",") if x.strip()]
    steps = [int(x) for x in args.steps.split(",") if x.strip()]
    device = "cpu" if args.device == "cpu" else "npu"
    if device == "npu":
        import torch_npu  # noqa: F401

        # Ascend lowers fp32 matmuls to HF32 (19-bit) by default; that alone moves results by ~1e-3.
        try:
            torch.npu.matmul.allow_hf32 = False
            print("  torch.npu.matmul.allow_hf32 = False")
        except Exception as exc:  # noqa: BLE001
            print(f"  could not disable HF32 matmul: {exc!r}")

    print(
        f"S2 layer check | device={device} | layers={layer_ids} steps={steps} | "
        f"hidden={cfg.hidden_size} heads={cfg.num_attention_heads}x{cfg.head_dim} "
        f"window={cfg.sliding_window} topk={cfg.index_topk}"
    )
    t0 = time.time()
    results = run_chain(
        official, cfg, layer_ids, steps, device=device, max_seq_len=args.max_seq_len
    )
    elapsed = time.time() - t0

    worst = max(results, key=lambda r: r.max_abs)
    mismatch = [r for r in results if not r.idx_match]
    # bf16 acceptance: either the absolute bar, or a deviation of at most one bf16 unit in the last
    # place -- at |x| ~ 4 that is 1.6e-2, which is 1 ULP and therefore "the same number" in bf16.
    over_tol = [r for r in results if r.max_abs > args.tol and r.max_ulp > 1.0]
    ok = not over_tol and not mismatch
    worst_ulp = max(r.max_ulp for r in results)
    print(
        f"\nworst max|diff| = {worst.max_abs:.3e} ({worst_ulp:.2f} bf16 ulp) at layer "
        f"{worst.layer_id} step {worst.step} (tol {args.tol:g}); "
        f"idx mismatches = {len(mismatch)}; {elapsed:.1f}s"
    )
    print("RESULT:", "OK" if ok else "FAIL")
    if args.json_out:
        payload = {
            "device": device,
            "layers": layer_ids,
            "steps": steps,
            "tol": args.tol,
            "ok": ok,
            "elapsed_s": elapsed,
            "rows": [r.__dict__ for r in results],
        }
        with open(args.json_out, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {args.json_out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
