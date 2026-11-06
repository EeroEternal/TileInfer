"""The vLLM adapter's registration surface (the runner contract has its own test file).

``test_dsv41_vllm_forward.py`` covers ``forward``/``compute_logits``; this file only pins what the
plugin exposes to vLLM: the architecture name it registers.
"""

from __future__ import annotations

from tileinfer.vllm.models.dsv41 import MODEL_ARCH, DeepseekV41ForCausalLM


def test_architecture_name_is_the_one_we_register():
    assert MODEL_ARCH == "DeepseekV41ForCausalLM"


def test_module_declares_no_packed_modules_mapping():
    """vLLM uses this to fuse QKV/MLP; the released shards are already in our spelling."""
    assert DeepseekV41ForCausalLM.packed_modules_mapping == {}


def test_adapter_exposes_the_documented_entry_points():
    for name in ("forward", "compute_logits", "load_weights", "host_offload_names"):
        assert hasattr(DeepseekV41ForCausalLM, name), name
