"""The vLLM adapter's runner contract (integration step A), tested without vLLM or a checkpoint.

``DeepseekV41ForCausalLM.forward`` ignores vLLM's paged KV and drives the standalone stack, so the
things worth pinning are the *contract* details: absolute positions are honoured, ``positions[0] == 0``
starts a fresh request state, the return value is per-token hidden states, and no weights are needed to
exercise any of it (``transformer_forward`` is stubbed here -- the maths has its own tests; this one
covers the plumbing).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tileinfer.vllm.models.dsv41 import DeepseekV41ForCausalLM


class _StateStub:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _Embed(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, hidden, dtype=torch.bfloat16))


class _ModelStub(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.embed = _Embed(hidden)


def _adapter(hidden: int = 8) -> DeepseekV41ForCausalLM:
    obj = object.__new__(DeepseekV41ForCausalLM)
    torch.nn.Module.__init__(obj)
    obj.dsv41_config = SimpleNamespace(max_seq_len=64)
    obj._runtime_moe = lambda: None  # the real one builds an arena from the checkpoint
    obj._moe_fn = None
    obj.model = _ModelStub(hidden)
    return obj


def _patch(monkeypatch, hidden_states):
    calls = []

    def fake_transformer_forward(model, ids, state, start_pos, capture=None, moe_fn=None,
                                 capture_layers=None, timings=None, skip_logits=False):
        calls.append({"ids": tuple(ids.shape), "start_pos": start_pos})
        capture["h"] = hidden_states(len(calls))
        return None, {}

    monkeypatch.setattr("tileinfer.models.dsv41.block.transformer_forward", fake_transformer_forward)
    monkeypatch.setattr("tileinfer.models.dsv41.block.ModelState", _StateStub)
    return calls


def test_forward_passes_absolute_position_and_returns_per_token_hidden(monkeypatch):
    calls = _patch(
        monkeypatch, lambda i: torch.zeros(1, 3, 8, dtype=torch.bfloat16)
    )
    obj = _adapter()
    out = obj.forward(input_ids=torch.tensor([10, 11, 12]), positions=torch.tensor([0, 1, 2]))
    assert tuple(out.shape) == (3, 8), tuple(out.shape)
    assert calls[0] == {"ids": (1, 3), "start_pos": 0}, calls


def test_decode_keeps_the_same_state_until_a_new_request(monkeypatch):
    calls = _patch(monkeypatch, lambda i: torch.zeros(1, 1, 8, dtype=torch.bfloat16))
    obj = _adapter()
    obj.forward(torch.tensor([1]), torch.tensor([0]))
    first = obj._stream_state
    obj.forward(torch.tensor([2]), torch.tensor([1]))
    assert obj._stream_state is first, "a decode step must not reset the request state"
    obj.forward(torch.tensor([3]), torch.tensor([0]))
    assert obj._stream_state is not first, "positions[0] == 0 must start a fresh state"
    assert [c["start_pos"] for c in calls] == [0, 1, 0]


def test_compute_logits_uses_the_last_token(monkeypatch):
    seen = {}

    def fake_logits_from_hidden(model, h):
        seen["shape"] = tuple(h.shape)
        return torch.zeros(1, 16, dtype=torch.float32)

    monkeypatch.setattr("tileinfer.models.dsv41.block.logits_from_hidden", fake_logits_from_hidden)
    obj = _adapter()
    out = obj.compute_logits(torch.zeros(3, 8, dtype=torch.bfloat16))
    assert tuple(out.shape) == (1, 16), tuple(out.shape)  # [num_tokens, vocab]: vLLM reads size(1)
    # logits_from_hidden本身只取 x[:, -1]，所以这里收到全部 token 是对的（last-token 语义在函数内部）
    assert seen["shape"] == (1, 3, 8), "the adapter forwards the block layout; the head keeps the last token"


def test_finalize_placement_moves_dense_and_keeps_experts_and_engram_on_host():
    """Step C: after loading, dense goes to the card and the 457.78 GiB stay in host memory."""

    class _Slot(torch.nn.Module):
        def __init__(self, elems: int):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(elems, dtype=torch.float32))

    class _Ffn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = torch.nn.ModuleList([_Slot(4)])
            self.gate = _Slot(2)

    class _Engram(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = _Slot(8)
            self.wkv = _Slot(1)

    class _Blk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _Slot(3)
            self.ffn = _Ffn()

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([_Blk()])
            self.engram = _Engram()

    obj = _adapter()
    obj.model = _M()
    report = obj.finalize_placement(device="cpu")
    # dense: attn.weight(3) + ffn.gate.weight(2) + engram.wkv.weight(1) = 6 floats
    assert report["on_device_bytes"] == 6 * 4, report
    # host: experts[0].weight(4) + engram.embed.weight(8) = 12 floats
    assert report["on_host_bytes"] == 12 * 4, report
