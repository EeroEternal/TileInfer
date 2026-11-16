"""Is a single decode step bit-reproducible?  No server, no vLLM engine — ~1 minute of device time.

§44: two identical requests to our arm produced different tokens (diverging at token 22), while the
vllm-ascend arm was bit-exact.  §46 then found two uninitialised-buffer bugs in MoE, which did *not*
explain it.  This test removes the serving stack and asks the narrowest question: given the same
inputs, does ``TileInferAttentionImpl.forward`` return the same hidden numbers twice?

  identical  -> the attention path is clean; the non-reproducibility lives in GDN/MoE/sampler
  differing  -> the kernel (or one of its staging buffers) is the cause, and the diff is right here

Also written for a hostile window: it needs one process, one plan, five forwards.
"""

import sys

import torch

import torch_npu  # noqa: F401

DT = torch.bfloat16
NH, NKV, D = 16, 2, 256
BLK = 2048          # engine page
BLOCKS = 32
TOKENS = 37         # ragged-ish: not a tile multiple
RUNS = 5


def main() -> int:
    from tileinfer.serving.buckets import fused_halves
    from tileinfer.vllm.attn.impl import TileInferAttentionImpl
    from tileinfer.vllm.attn.kv_cache import write_kv_cache
    from tileinfer.vllm.attn.metadata import TileInferMetadata

    dev = torch.device("npu", 0)
    torch.manual_seed(0)
    # A prefill-shaped step first (so the KV pool holds a context), then repeated decode steps.
    seq_len, qo_len = 512, 1
    max_pages = (seq_len + BLK - 1) // BLK
    table = torch.zeros(1, max_pages, dtype=torch.int32, device=dev)
    for p in range(max_pages):
        table[0, p] = p
    slots = []
    for i in range(seq_len):
        page, off = divmod(i, BLK)
        slots.append(int(table[0, page]) * BLK + off)
    slots = torch.tensor(slots, dtype=torch.int64, device=dev)

    kv_cache = torch.zeros(2, BLOCKS, BLK, NKV, D, dtype=DT, device=dev).permute(1, 0, 2, 3, 4)
    k_all = torch.randn(seq_len, NKV, D, dtype=DT, device=dev) * 0.2
    v_all = torch.randn(seq_len, NKV, D, dtype=DT, device=dev) * 0.2
    k_half, v_half = fused_halves(kv_cache)
    write_kv_cache(k_all, v_all, k_half, v_half, slots)

    # one decode step at position seq_len
    q = torch.randn(1, NH, D, dtype=DT, device=dev) * 0.2
    k = torch.randn(1, NKV, D, dtype=DT, device=dev) * 0.2
    v = torch.randn(1, NKV, D, dtype=DT, device=dev) * 0.2
    slot = int(table[0, 0]) * BLK + seq_len
    qsl = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    meta = TileInferMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl.cpu(),
        seq_lens=torch.tensor([seq_len + 1], dtype=torch.int32, device=dev),
        block_table_tensor=table,
        slot_mapping=torch.tensor([slot], dtype=torch.int64, device=dev),
        num_reqs=1,
        num_actual_tokens=1,
        max_query_len=1,
        bucket=1,
        block_size=BLK,
    )

    impl = TileInferAttentionImpl(NH, D, D**-0.5, num_kv_heads=NKV)
    outs = []
    for run in range(RUNS):
        out = torch.zeros(1, NH * D, dtype=DT, device=dev)
        impl.forward(None, q, k, v, kv_cache, meta, out)
        torch.npu.synchronize()
        outs.append(out.clone())

    base = outs[0]
    differing = 0
    worst = 0.0
    for i, got in enumerate(outs[1:], start=1):
        if not torch.equal(got, base):
            differing += 1
            worst = max(worst, float((got.float() - base.float()).abs().max()))
    print(f"### decode-step determinism over {RUNS} runs: {differing}/{RUNS - 1} differ, "
          f"worst_abs={worst:.3e}")
    print("### RESULT:", "DETERMINISTIC" if differing == 0 else "NON-DETERMINISTIC")

    # Which op in the step is the likely culprit: the same question for the MoE combine and the
    # conv update, both of which were touched for determinism.
    from tileinfer.vllm.ops.moe import fused_experts

    torch.manual_seed(1)
    x = torch.randn(1, 64, dtype=DT, device=dev) * 0.2
    w13 = torch.randn(4, 32, 64, dtype=DT, device=dev) * 0.05
    w2 = torch.randn(4, 64, 16, dtype=DT, device=dev) * 0.05
    ids = torch.tensor([[0, 1]], dtype=torch.int32, device=dev)
    wts = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=dev)
    first = fused_experts(x, w13, w2, wts, ids, activation="silu")
    moe_bad = 0
    for _ in range(20):
        if not torch.equal(fused_experts(x, w13, w2, wts, ids, activation="silu"), first):
            moe_bad += 1
    print(f"### moe fused_experts: {moe_bad}/20 differ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
