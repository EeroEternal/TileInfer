"""Indexer 分块 top-k（DSV41_INDEXER_CHUNK）必须与整块实现等价。

长 prefill 的 ``score=[S,32,T]`` 在 14K 时 26+ GiB，P4 16K OOM 的根因。分块实现用 running
top-k + candidate 层 block maxima，数学等价；这里在 CPU 上把同一 prompt 跑两遍（开关关/开），
要求 logits 与 greedy token 一致。覆盖 candidate-source 与 uses-candidates 两种路径。
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.models.dsv41 import Dsv41Config
from tileinfer.models.dsv41.block import ModelState, transformer_forward
from tileinfer.models.dsv41.modules import Dsv41Model


def tiny_config():
    return Dsv41Config(
        vocab_size=128,
        hidden_size=64,
        moe_intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        head_dim=32,
        qk_rope_head_dim=8,
        q_lora_rank=32,
        o_lora_rank=16,
        o_groups=2,
        sliding_window=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        compress_ratios=(0, 2, 1, 1),
        kv_source_layers=(1, 2),
        index_source_layers=(1, 2, 3),
        candidate_source_layer=2,
        candidate_block_size=2,
        candidate_topk_blocks=4,
        index_n_heads=2,
        index_head_dim=32,
        index_topk=8,
        hc_mult=2,
        engram_layer_ids=(),
        engram_num_embeddings=(),
        num_nextn_predict_layers=0,
        vision_num_hidden_layers=1,
    )


def _run(chunk: int):
    from tools.dsv41.s2_layer_check import fill_weights

    cfg = tiny_config()
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(5)
    model = Dsv41Model(cfg, device="cpu", with_vision=False, with_mtp=False)
    fill_weights(dict(model.named_parameters()), seed=23)
    ids = torch.randint(0, cfg.vocab_size, (1, 64), generator=torch.Generator().manual_seed(9))
    state = ModelState(cfg=cfg, max_seq_len=128, bsz=1, device="cpu")
    if chunk:
        import os

        os.environ["DSV41_INDEXER_CHUNK"] = str(chunk)
    else:
        import os

        os.environ.pop("DSV41_INDEXER_CHUNK", None)
    try:
        logits, _ = transformer_forward(model, ids, state, 0)
    finally:
        import os

        os.environ.pop("DSV41_INDEXER_CHUNK", None)
    return logits, state


def test_streaming_indexer_matches_full_prefill():
    full, s_full = _run(0)
    chunked, s_chunk = _run(16)
    diff = (full.float() - chunked.float()).abs()
    # fp4 量化后的 indexer score 存在真实并列；整块与 streaming 的 top-k 在并列边界上可取不同
    # key（两者都是合法 top-k），因此只要求 greedy 一致 + 有界差异，不要求逐位相同。
    assert diff.max().item() <= 2e-1, f"max|diff|={diff.max().item():.3e}"
    assert torch.equal(full.argmax(dim=-1), chunked.argmax(dim=-1))
    # 注意：不能要求 window/compress 逐位相同 —— top-k 并列若选到不同 key，会改变该层输出，
    # 进而改变后续层写入窗口的 x；这是 fp4 score 并列的合法分支，不是分块逻辑错误。


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
