"""Engram parity: the n-gram hash ids and one engram layer, against ``inference/engram.py``.

Two discrete/critical things are checked:

1. **hash ids** -- same tokenizer, same config, same positions (prefill + decode split, with and
   without the image-span mask) must produce *bit-identical* ids.  A single wrong prime, multiplier
   or pad id rehashes the whole table, so this is the cheap decisive test.
2. **the engram layer itself** -- the official ``Engram(args, layer_id, layout)`` vs ours, same
   weights (fp8 table + e8m0 row-block scales, wkv fp8), same hash ids: output must be bit-exact.

The tables are made small by shrinking ``engram_vocab_size`` (the prime base): the row count is the
sum of the layer's 24 primes, which both sides derive independently.  The compressed token map (99,092
entries for this tokenizer) is computed once and shared with the oracle through a monkeypatched
``build_compressed_token_map`` -- identical data, one normalization pass.
"""

from __future__ import annotations

import os
import sys

import pytest

MODEL_DIR = "/home/models/DeepSeek-V4.1-Flash"
TOKENIZER_DIR = MODEL_DIR
INFERENCE = os.path.join(MODEL_DIR, "inference")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(INFERENCE),
    reason=f"official inference sources not present at {INFERENCE}",
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_MAP_CACHE: dict = {}


def _compressed_map(tokenizer):
    from tileinfer.models.dsv41.engram import build_compressed_token_map

    if "m" not in _MAP_CACHE:
        _MAP_CACHE["m"] = build_compressed_token_map(tokenizer)
    return _MAP_CACHE["m"]


def _official_modules():
    """Official ``model``/``engram`` with the kernel names swapped (read-only oracle)."""
    from tools.dsv41.s2_layer_check import official_model

    official = official_model(MODEL_DIR)
    inf = os.path.join(MODEL_DIR, "inference")
    if inf not in sys.path:
        sys.path.insert(0, inf)
    import engram as official_engram  # noqa: E402

    # share one normalization pass; the values are the oracle's own
    real_build = official_engram.build_compressed_token_map

    def shared_build(tokenizer):
        return _compressed_map(tokenizer)

    official_engram.build_compressed_token_map = shared_build
    return official, official_engram, real_build


def _cfg(**overrides):
    from tileinfer.models.dsv41 import Dsv41Config

    base = dict(
        vocab_size=128,
        hidden_size=64,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        head_dim=32,
        qk_rope_head_dim=8,
        q_lora_rank=32,
        o_lora_rank=16,
        o_groups=2,
        sliding_window=8,
        n_routed_experts=2,
        num_experts_per_tok=1,
        compress_ratios=(0, 2, 0),
        kv_source_layers=(1,),
        index_source_layers=(1,),
        candidate_source_layer=-1,
        candidate_topk_blocks=2,
        candidate_block_size=2,
        index_n_heads=2,
        index_head_dim=32,
        index_topk=4,
        hc_mult=2,
        engram_layer_ids=(0,),
        engram_num_embeddings=(0,),  # patched below with the real sum of primes
        engram_max_ngram_size=3,
        engram_vocab_size=1000,
        engram_n_heads=2,
        engram_head_dim=64,  # two 32-column groups, so a wrong unflatten cannot hide
        num_nextn_predict_layers=0,
        vision_num_hidden_layers=0,
    )
    base.update(overrides)
    cfg = Dsv41Config(**base)
    from tileinfer.models.dsv41.engram import primes_rows

    rows = primes_rows(cfg)[0]
    return Dsv41Config(**{**base, "engram_num_embeddings": (rows,)})


def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_DIR)


def test_hash_ids_match_the_reference():
    import torch

    official, official_engram, _real = _official_modules()
    from tools.dsv41.s2_layer_check import oracle_args

    cfg = _cfg()
    tok = _tokenizer()
    lookup, vocab = _compressed_map(tok)
    assert vocab == cfg.engram_compressed_vocab_size

    args = oracle_args(official, cfg, 32, 1)
    layout = official_engram.EngramLayout.from_args(args)
    ref = official_engram.NgramHashState(args, layout, tok)

    from tileinfer.models.dsv41.engram import EngramHash

    ours = EngramHash(cfg, lookup=lookup, vocab_size=vocab, max_seq_len=32, device="cpu")

    gen = torch.Generator().manual_seed(3)
    prompt = torch.randint(0, cfg.vocab_size, (1, 6), generator=gen)
    mask = torch.ones(1, 6, dtype=torch.bool)
    mask[:, 2] = False  # emulate an image span in the middle

    for use_mask in (False, True):
        want = ref.forward(prompt.clone(), 0, mask.clone() if use_mask else None)
        got = ours.forward(prompt.clone(), 0, mask.clone() if use_mask else None)
        assert torch.equal(got, want), (
            f"prefill hash ids differ (mask={use_mask}): "
            f"{(got != want).sum().item()} of {got.numel()}"
        )
        nxt = torch.randint(0, cfg.vocab_size, (1, 1), generator=gen)
        want2 = ref.forward(nxt.clone(), 6, None)
        got2 = ours.forward(nxt.clone(), 6, None)
        assert torch.equal(got2, want2), f"decode hash ids differ (mask={use_mask})"
        # re-run the prefill so the cache is in the same state for the next round
        ref.forward(prompt.clone(), 0, mask.clone() if use_mask else None)
        ours.forward(prompt.clone(), 0, mask.clone() if use_mask else None)


def test_engram_layer_matches_the_reference():
    import torch

    official, official_engram, _real = _official_modules()
    from tools.dsv41.s2_layer_check import copy_weights, fill_weights, oracle_args

    from tileinfer.models.dsv41.block import engram_forward
    from tileinfer.models.dsv41.modules import Dsv41Engram

    cfg = _cfg()
    tok = _tokenizer()
    lookup, vocab = _compressed_map(tok)
    args = oracle_args(official, cfg, 32, 1)
    layout = official_engram.EngramLayout.from_args(args)

    torch.set_default_dtype(torch.bfloat16)
    ref = official.Engram(args, 0, layout)  # the layer lives in model.py, not engram.py
    params = dict(ref.named_parameters())
    fill_weights(params, seed=21)
    ours = Dsv41Engram(cfg, 0, device="cpu")
    copy_weights(params, dict(ours.named_parameters()))

    from tileinfer.models.dsv41.engram import EngramHash

    hash_state = EngramHash(cfg, lookup=lookup, vocab_size=vocab, max_seq_len=32, device="cpu")
    ids = torch.randint(0, cfg.vocab_size, (1, 6), generator=torch.Generator().manual_seed(4))
    hashes = hash_state.forward(ids, 0, None)

    gen = torch.Generator().manual_seed(9)
    x = torch.randn(1, 6, cfg.hc_mult, cfg.hidden_size, generator=gen).to(torch.bfloat16)
    want = ref.forward(x.clone(), hashes.clone(), None)
    got = engram_forward(ours, x.clone(), hashes.clone(), None)
    assert torch.equal(got, want), (
        f"engram layer: max|diff|={(got - want).abs().max().item():.3e} "
        f"mean={(got - want).abs().mean().item():.3e}"
    )

    mask = torch.ones(1, 6, dtype=torch.bool)
    mask[:, 1:3] = False
    want_m = ref.forward(x.clone(), hashes.clone(), mask.clone())
    got_m = engram_forward(ours, x.clone(), hashes.clone(), mask.clone())
    assert torch.equal(got_m, want_m), (
        f"masked engram: max|diff|={(got_m - want_m).abs().max().item():.3e}"
    )
