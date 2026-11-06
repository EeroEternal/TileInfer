"""True batched decode vs the per-request serial path (CPU, no checkpoint needed).

The P3 target is "4 requests in one 40-layer pass".  The only acceptable proof at the unit level is
that a batched step produces the **same next-token logits** as four independent serial steps, and
that the per-request state after the batched step is still a valid continuation (a serial step from
the updated state must match a purely serial run).  Both are checked here with synthetic weights.

Uses the tiny config from ``test_dsv41_block_parity.py``: layer 1 is a ratio-2 source, layer 2 a
ratio-1 candidate source, layer 0 pure sliding window -- i.e. all three attention shapes.
"""

from __future__ import annotations

import copy

import pytest
import torch

from tileinfer.models.dsv41 import Dsv41Config
from tileinfer.models.dsv41.block import ModelState, transformer_forward
from tileinfer.models.dsv41.batched_decode import transformer_forward_decode
from tileinfer.models.dsv41.modules import Dsv41Model


def tiny_config(**overrides):
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
        compress_ratios=(0, 2, 1),
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


def _model_and_states(cfg, prompts):
    from tools.dsv41.s2_layer_check import fill_weights

    # the stack is bf16 end to end; norms/sinks are created with the default dtype
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(3)
    model = Dsv41Model(cfg, device="cpu", with_vision=False, with_mtp=False)
    fill_weights(dict(model.named_parameters()), seed=17)

    gen = torch.Generator().manual_seed(21)
    states, ids = [], []
    for n in prompts:
        s = ModelState(cfg=cfg, max_seq_len=32, bsz=1, device="cpu")
        cur = torch.randint(0, cfg.vocab_size, (1, n), generator=gen)
        transformer_forward(model, cur, s, 0)
        states.append(s)
        ids.append(cur)
    return model, states, ids


def _step_ids(cfg, count):
    gen = torch.Generator().manual_seed(99)
    return torch.randint(0, cfg.vocab_size, (count, 1), generator=gen)


@pytest.mark.parametrize("prompts", [(5, 3), (6, 4), (1, 7)])
def test_batched_decode_matches_serial(prompts):
    """One batched step must equal the serial step for every request (logits + argmax)."""
    cfg = tiny_config()
    model, states, _ = _model_and_states(cfg, prompts)
    serial_states = copy.deepcopy(states)
    positions = list(prompts)
    nxt = _step_ids(cfg, len(prompts))

    want = []
    for i, s in enumerate(serial_states):
        logits, _ = transformer_forward(model, nxt[i : i + 1].reshape(1, 1), s, positions[i])
        want.append(logits.reshape(-1))
    want = torch.stack(want, dim=0)

    got, hidden = transformer_forward_decode(
        model, nxt.reshape(-1), positions, states
    )
    assert got.shape == (len(prompts), cfg.vocab_size)
    assert hidden.shape == (len(prompts), 1, cfg.hidden_size)

    diff = (got.float() - want.float()).abs()
    assert diff.max().item() <= 2e-3, f"max|diff|={diff.max().item():.3e}"
    assert torch.equal(got.argmax(dim=-1), want.argmax(dim=-1))


def test_batched_state_is_a_valid_serial_continuation():
    """After a batched step, continuing serially must agree with an all-serial run."""
    cfg = tiny_config()
    model, states, _ = _model_and_states(cfg, (5, 3))
    serial_states = copy.deepcopy(states)
    pos = [5, 3]
    tok = _step_ids(cfg, 2).reshape(-1)

    transformer_forward_decode(model, tok, pos, states)

    # one more step, both from "state after the first batched/serial step"
    nxt = _step_ids(cfg, 2).reshape(-1)
    want = []
    for i, s in enumerate(serial_states):
        transformer_forward(model, tok[i : i + 1].reshape(1, 1), s, pos[i])
        logits, _ = transformer_forward(model, nxt[i : i + 1].reshape(1, 1), s, pos[i] + 1)
        want.append(logits.reshape(-1))
    want = torch.stack(want, dim=0)

    got, _ = transformer_forward_decode(
        model, nxt, [p + 1 for p in pos], states
    )
    diff = (got.float() - want.float()).abs()
    assert diff.max().item() <= 2e-3, f"state after batch is not a serial continuation: {diff.max():.3e}"
    assert torch.equal(got.argmax(dim=-1), want.argmax(dim=-1))


def test_batched_matches_serial_for_a_mixed_position_batch():
    """Requests at different absolute positions (2 vs 9) must not share a window/compress slice."""
    cfg = tiny_config()
    model, states, _ = _model_and_states(cfg, (2, 9))
    serial_states = copy.deepcopy(states)
    nxt = _step_ids(cfg, 2).reshape(-1)

    want = []
    for i, s in enumerate(serial_states):
        logits, _ = transformer_forward(model, nxt[i : i + 1].reshape(1, 1), s, (2, 9)[i])
        want.append(logits.reshape(-1))
    want = torch.stack(want, dim=0)

    got, _ = transformer_forward_decode(model, nxt, [2, 9], states)
    assert (got.float() - want.float()).abs().max().item() <= 2e-3
    assert torch.equal(got.argmax(dim=-1), want.argmax(dim=-1))


def test_per_row_attention_parity_mode_matches_serial(monkeypatch):
    """VLLM_DSV41_BATCHED_ATTN=per_row 用单请求 attention 形状，必须与串行逐位一致。"""
    cfg = tiny_config()
    model, states, _ = _model_and_states(cfg, (5, 3))
    serial_states = copy.deepcopy(states)
    nxt = _step_ids(cfg, 2).reshape(-1)
    want = []
    for i, s in enumerate(serial_states):
        logits, _ = transformer_forward(model, nxt[i : i + 1].reshape(1, 1), s, (5, 3)[i])
        want.append(logits.reshape(-1))
    want = torch.stack(want, dim=0)

    monkeypatch.setenv("VLLM_DSV41_BATCHED_ATTN", "per_row")
    got, _ = transformer_forward_decode(model, nxt, [5, 3], states)
    assert (got - want).abs().max().item() <= 1e-6, f"max|diff|={(got - want).abs().max().item():.3e}"
    assert torch.equal(got.argmax(dim=-1), want.argmax(dim=-1))
