"""Whole-block and whole-stack parity against the official ``inference/model.py`` (CPU).

S3's foundation: the parts S2 did not cover -- Hyper-Connections (hc_mixes / hc_pre / hc_post with
the Sinkhorn comb), MoE routing + fp4 expert GEMM + shared expert, the block's cross-sublayer
``pre_mix`` timing, and the full embed -> 40 blocks -> collapse -> head path.

Same method as ``test_dsv41_layer_parity.py``: the official model runs on CPU with its six tilelang
kernels swapped for ``tools/dsv41/ref_cpu.py``, both sides get the same synthetic weights (in the
checkpoint's dtypes) and the same input, and the outputs must be **bit-exact**.

Skipped when the checkpoint directory (which holds the read-only ``inference/`` sources) is absent.
"""

from __future__ import annotations

import os
import sys

import pytest

MODEL_DIR = "/home/models/DeepSeek-V4.1-Flash"
INFERENCE = os.path.join(MODEL_DIR, "inference")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(INFERENCE),
    reason=f"official inference sources not present at {INFERENCE}",
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def tiny_config(**overrides):
    from tileinfer.models.dsv41 import Dsv41Config

    base = dict(
        vocab_size=128,
        hidden_size=64,
        moe_intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=2,
        head_dim=32,
        qk_rope_head_dim=8,
        q_lora_rank=32,
        o_lora_rank=16,
        o_groups=2,
        sliding_window=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        compress_ratios=(0, 2, 1, 0, 0),
        kv_source_layers=(1, 2),
        index_source_layers=(1, 2),
        candidate_source_layer=2,
        candidate_block_size=2,
        candidate_topk_blocks=4,
        index_n_heads=2,
        index_head_dim=32,
        index_topk=4,
        hc_mult=2,
        engram_layer_ids=(),
        engram_num_embeddings=(),
        num_nextn_predict_layers=0,
        vision_num_hidden_layers=1,
    )
    base.update(overrides)
    return Dsv41Config(**base)


def _setup(seed=99):
    import torch

    from tools.dsv41.s2_layer_check import official_model

    torch.set_default_dtype(torch.bfloat16)
    return official_model(MODEL_DIR)


def _pair_blocks(official, cfg, layer_ids, max_seq_len=32):
    """Official + our blocks for ``layer_ids``, sharing one synthetic weight set per layer."""
    from tileinfer.models.dsv41.modules import Dsv41Block
    from tools.dsv41.s2_layer_check import copy_weights, fill_weights, oracle_args

    args = oracle_args(official, cfg, max_seq_len, 1)
    out = []
    for layer_id in layer_ids:
        ref = official.Block(layer_id, args, None)
        params = dict(ref.named_parameters())
        fill_weights(params, seed=1234 + 17 * layer_id)
        ours = Dsv41Block(cfg, layer_id, device="cpu")
        copy_weights(params, dict(ours.named_parameters()))
        out.append((layer_id, ref, ours))
    return args, out


def test_block_matches_the_reference_bit_for_bit():
    import torch

    from tileinfer.models.dsv41.block import ModelState, block_forward, make_identity_pre_mix

    official = _setup()
    cfg = tiny_config()
    args, pairs = _pair_blocks(official, cfg, [0, 1, 2])
    gen = torch.Generator().manual_seed(5)

    for layer_id, ref, ours in pairs:
        state = ModelState(cfg=cfg, max_seq_len=32, bsz=1, device="cpu")
        x = torch.randn(1, 6, cfg.hc_mult, cfg.hidden_size, generator=gen).to(torch.bfloat16)
        pre_mix = make_identity_pre_mix(x, cfg.hc_mult)
        want, want_pre = ref.forward(x.clone(), 0, pre_mix.clone(), None)
        got, got_pre = block_forward(ours, x.clone(), 0, pre_mix.clone(), state)
        assert torch.equal(got, want), (
            f"layer {layer_id}: max|diff|={(got - want).abs().max().item():.3e} "
            f"mean={(got - want).abs().mean().item():.3e}"
        )
        assert torch.equal(got_pre, want_pre), f"layer {layer_id}: pre_mix differs"


def test_block_decode_step_matches():
    """The second call runs the decode branch (ring buffer, compressed-stream tail)."""
    import torch

    from tileinfer.models.dsv41.block import ModelState, block_forward, make_identity_pre_mix

    official = _setup()
    cfg = tiny_config()
    args, pairs = _pair_blocks(official, cfg, [1])
    layer_id, ref, ours = pairs[0]
    state = ModelState(cfg=cfg, max_seq_len=32, bsz=1, device="cpu")
    gen = torch.Generator().manual_seed(6)
    x = torch.randn(1, 6, cfg.hc_mult, cfg.hidden_size, generator=gen).to(torch.bfloat16)
    pre_mix = make_identity_pre_mix(x, cfg.hc_mult)
    ref.forward(x.clone(), 0, pre_mix.clone(), None)
    block_forward(ours, x.clone(), 0, pre_mix.clone(), state)

    x1 = torch.randn(1, 1, cfg.hc_mult, cfg.hidden_size, generator=gen).to(torch.bfloat16)
    pre1 = make_identity_pre_mix(x1, cfg.hc_mult)
    want, _ = ref.forward(x1.clone(), 6, pre1.clone(), None)
    got, _ = block_forward(ours, x1.clone(), 6, pre1.clone(), state)
    assert torch.equal(got, want), (
        f"decode step: max|diff|={(got - want).abs().max().item():.3e}"
    )


def test_transformer_greedy_token_matches():
    """The whole stack: embed -> blocks -> hc_pre -> norm -> head, greedy argmax identical."""
    import torch

    from tileinfer.models.dsv41.block import ModelState, transformer_forward
    from tileinfer.models.dsv41.modules import Dsv41Model
    from tools.dsv41.s2_layer_check import copy_weights, fill_weights, oracle_args

    official = _setup()
    cfg = tiny_config()
    args = oracle_args(official, cfg, 32, 1)
    ref = official.Transformer(args, None)
    params = dict(ref.named_parameters())
    fill_weights(params, seed=7)
    ours = Dsv41Model(cfg, device="cpu", with_vision=True, with_mtp=False)
    copy_weights(params, dict(ours.named_parameters()))

    state = ModelState(cfg=cfg, max_seq_len=32, bsz=1, device="cpu")
    ids = torch.randint(0, cfg.vocab_size, (1, 5), generator=torch.Generator().manual_seed(11))
    want_ids, want_logits, _ = ref.forward(ids.clone(), 0)
    got_logits, extras = transformer_forward(ours, ids.clone(), state, 0)
    diff = (got_logits - want_logits.float()).abs()
    assert torch.equal(got_logits, want_logits.float()), (
        f"logits: max|diff|={diff.max().item():.3e} mean={diff.mean().item():.3e}"
    )
    assert int(got_logits.argmax()) == int(want_logits.float().argmax())
    assert "main_hiddens" not in extras or extras["main_hiddens"] is not None


def test_transformer_second_step_matches():
    """A decode step after a 5-token prefill: state (caches, compressor tails) must line up."""
    import torch

    from tileinfer.models.dsv41.block import ModelState, transformer_forward
    from tileinfer.models.dsv41.modules import Dsv41Model
    from tools.dsv41.s2_layer_check import copy_weights, fill_weights, oracle_args

    official = _setup()
    cfg = tiny_config()
    args = oracle_args(official, cfg, 32, 1)
    ref = official.Transformer(args, None)
    params = dict(ref.named_parameters())
    fill_weights(params, seed=8)
    ours = Dsv41Model(cfg, device="cpu", with_vision=True, with_mtp=False)
    copy_weights(params, dict(ours.named_parameters()))

    state = ModelState(cfg=cfg, max_seq_len=32, bsz=1, device="cpu")
    gen = torch.Generator().manual_seed(12)
    ids = torch.randint(0, cfg.vocab_size, (1, 5), generator=gen)
    ref.forward(ids.clone(), 0)
    transformer_forward(ours, ids.clone(), state, 0)

    nxt = torch.randint(0, cfg.vocab_size, (1, 1), generator=gen)
    want_ids, want_logits, _ = ref.forward(nxt.clone(), 5)
    got_logits, _ = transformer_forward(ours, nxt.clone(), state, 5)
    diff = (got_logits - want_logits.float()).abs()
    assert torch.equal(got_logits, want_logits.float()), (
        f"step 2 logits: max|diff|={diff.max().item():.3e}"
    )
    assert int(got_logits.argmax()) == int(want_logits.float().argmax())
