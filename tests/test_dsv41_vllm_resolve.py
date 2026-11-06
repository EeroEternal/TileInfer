"""Regression tests for the ``wo_a`` resolution order (the bug behind the wrong served logits).

``_resolve`` used to look the name up in the parameter table *first*.  The tree holds a folded **bf16**
``layers.N.attn.wo_a.weight``, so the lookup always hit and the loader copied the raw fp8 checkpoint
tensor into that bf16 parameter -- the fold branch was dead code.  The measured effect inside
``vllm serve`` was an attention output ~4000x too large and a wrong first token (23684 instead of
11111); the offline two-path comparison showed ``wo_a`` rms 92.82 where it must be 0.022787.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tileinfer.vllm.models.dsv41 import DeepseekV41ForCausalLM


def _adapter():
    obj = object.__new__(DeepseekV41ForCausalLM)
    torch.nn.Module.__init__(obj)
    obj.dsv41_config = SimpleNamespace(max_seq_len=64)
    obj._model_dir = ""
    # exactly the names the real attention module exposes (audited against the checkpoint)
    obj._named_params = {
        "layers.0.attn.wo_a.weight": torch.nn.Parameter(torch.zeros(8, 4, dtype=torch.bfloat16), requires_grad=False),
        "layers.0.attn.wkv.weight": torch.nn.Parameter(torch.zeros(4, 4, dtype=torch.float8_e4m3fn), requires_grad=False),
        "layers.0.attn.wkv.scale": torch.nn.Parameter(torch.zeros(1, 1, dtype=torch.uint8), requires_grad=False),
    }
    return obj


def test_wo_a_weight_is_folded_not_copied():
    """The folded bf16 parameter must not swallow the raw fp8 tensor."""
    obj = _adapter()
    param, kind = obj._resolve("layers.0.attn.wo_a.weight")
    assert param is None, "wo_a.weight must go through the fold, never a direct copy"
    assert kind == "wo_a_weight"


def test_wo_a_scale_is_paired_with_its_weight():
    obj = _adapter()
    param, kind = obj._resolve("layers.0.attn.wo_a.scale")
    assert param is None
    assert kind == "wo_a_scale"


def test_wo_a_of_a_missing_layer_is_skipped():
    """A text-only build has no ``mtp.N.attn.wo_a``; those tensors must be skipped, not KeyError."""
    obj = _adapter()
    param, kind = obj._resolve("mtp.0.attn.wo_a.weight")
    assert (param, kind) == (None, "skip")


def test_ordinary_parameters_still_resolve_directly():
    obj = _adapter()
    param, kind = obj._resolve("layers.0.attn.wkv.weight")
    assert param is not None and kind is None
    param, kind = obj._resolve("layers.0.attn.wkv.scale")
    assert param is not None and kind is None
