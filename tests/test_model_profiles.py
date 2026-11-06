"""Contract tests for the pluggable model-profile layer (``tileinfer.vllm.profiles``).

What this file proves
---------------------
1. The registry resolves Qwen3.6 from its vLLM architecture names, and an unknown architecture
   raises with the known-profile list.
2. The profile's values are what ``platform`` / ``attn.backend`` / ``worker`` *enforce* - asserted
   by calling their helpers, not by re-reading the module constants that only mirror the profile.
3. A second profile registered at test time (a dummy with different head widths, GQA floor, page
   policy and KV layout) drives the platform and the backend without editing either file.  That is
   the extensibility seam the refactor exists for.
4. ``ops.install`` iterates the operator family named by the active profile, so a new family is a
   new module rather than an edit to ``install.py``.

The profile/ops registries are process-global (the engine asks backend hooks without a config), so
the fixture snapshots and restores them; the dummy profile/family never leaks into the other
contract tests.
"""

from __future__ import annotations

import types
from typing import Any, Dict, Tuple

import pytest
import torch

import fake_vllm

# One shared fake vLLM, and the platform-side fake owned by tests/test_platform_contract.py.
VLLM_MODE = fake_vllm.install()
import test_platform_contract as _platform_contract  # noqa: E402,F401  (installs the Platform stub)

from tileinfer.vllm import platform as platform_mod  # noqa: E402
from tileinfer.vllm import profiles  # noqa: E402
from tileinfer.vllm import worker as worker_mod  # noqa: E402
from tileinfer.vllm.attn.backend import TileInferAttentionBackend  # noqa: E402
from tileinfer.vllm.ops import families as families_mod  # noqa: E402
from tileinfer.vllm.ops import install as install_mod  # noqa: E402
from tileinfer.vllm.profiles import (  # noqa: E402
    BlockSizePolicy,
    HybridKind,
    KvShape,
    ModelProfile,
    UnsupportedFeature,
)

AttentionType = __import__("sys").modules["vllm.v1.attention.backend"].AttentionType

QWEN_ARCHS = (
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5ForCausalLM",
    "Qwen3_5MoeForCausalLM",
)


# =============================================================================================
# Fixtures / config stand-ins
# =============================================================================================


@pytest.fixture(autouse=True)
def _restore_registries() -> Any:
    """Every test starts from the shipped registries and leaves them as it found them."""
    profile_names = tuple(profiles.PROFILES)
    family_names = tuple(families_mod.FAMILIES)
    profiles.reset_active_profile()
    install_mod._reset()
    try:
        yield
    finally:
        for name in tuple(families_mod.FAMILIES):
            if name not in family_names:
                families_mod.unregister_family(name)
        for name in tuple(profiles.PROFILES):
            if name not in profile_names:
                profiles.unregister_profile(name)
        profiles.reset_active_profile()
        install_mod._reset()


def _cache_config(**overrides: Any) -> types.SimpleNamespace:
    fields: Dict[str, Any] = {
        "block_size": 2048,
        "cache_dtype": "auto",
        "user_specified_block_size": True,
        "mamba_cache_mode": "align",
        "mamba_block_size": None,
        "mamba_page_size_padded": None,
    }
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def _model_config(**overrides: Any) -> types.SimpleNamespace:
    def get_num_kv_heads(parallel_config: Any = None) -> int:
        return 2

    def get_head_size() -> int:
        return 256

    fields: Dict[str, Any] = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "is_hybrid": True,
        "use_mla": False,
        "enable_sleep_mode": False,
        "enforce_eager": True,
        "dtype": torch.bfloat16,
        "max_model_len": 32768,
        "get_num_kv_heads": get_num_kv_heads,
        "get_head_size": get_head_size,
    }
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def _config(**overrides: Any) -> types.SimpleNamespace:
    config = types.SimpleNamespace(
        speculative_config=None,
        cache_config=_cache_config(),
        model_config=_model_config(),
        parallel_config=types.SimpleNamespace(worker_cls="auto"),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _mamba_config(*mamba_types: str) -> types.SimpleNamespace:
    groups = [
        types.SimpleNamespace(
            kv_cache_spec=types.SimpleNamespace(mamba_type=name),
            layer_names=(f"model.layers.{index}.linear_attn",),
        )
        for index, name in enumerate(mamba_types)
    ]
    return types.SimpleNamespace(kv_cache_groups=tuple(groups))


def _validate(**overrides: Any) -> list:
    kwargs: Dict[str, Any] = {
        "head_size": 256,
        "dtype": torch.bfloat16,
        "kv_cache_dtype": "auto",
        "block_size": 2048,
        "use_mla": False,
        "has_sink": False,
        "use_sparse": False,
        "use_mm_prefix": False,
        "use_per_head_quant_scales": False,
        "device_capability": None,
        "attn_type": AttentionType.DECODER,
    }
    kwargs.update(overrides)
    return TileInferAttentionBackend.validate_configuration(**kwargs)


# =============================================================================================
# 1. the registry
# =============================================================================================


def test_registry_resolves_qwen36_by_arch() -> None:
    qwen = profiles.profile_by_name("qwen3.6")
    for arch in QWEN_ARCHS:
        assert profiles.profile_for(arch) is qwen
        assert profiles.profile_for([arch]) is qwen
    assert profiles.profile_for(list(QWEN_ARCHS)) is qwen
    assert profiles.default_profile() is qwen
    assert "qwen3.6" in profiles.known_profiles()
    assert qwen.archs == QWEN_ARCHS


def test_unknown_arch_raises_with_the_known_profile_list() -> None:
    with pytest.raises(ValueError) as excinfo:
        profiles.profile_for(["NotARealForCausalLM"])
    message = str(excinfo.value)
    assert "NotARealForCausalLM" in message
    assert "known profiles" in message
    assert "qwen3.6" in message
    # The error names the file to add a profile in, so the fix is obvious.
    assert "tileinfer/vllm/profiles" in message


def test_unknown_profile_name_raises_with_the_known_profile_list() -> None:
    with pytest.raises(ValueError) as excinfo:
        profiles.profile_by_name("no-such-profile")
    assert "no-such-profile" in str(excinfo.value)
    assert "qwen3.6" in str(excinfo.value)


# =============================================================================================
# 2. the profile values are what the consumers enforce
# =============================================================================================


def test_platform_enforces_the_profile_page_and_mamba_geometry() -> None:
    qwen = profiles.profile_by_name("qwen3.6")
    platform_mod.set_active_profile(qwen)
    config = _config()
    platform_mod.TileInferPlatform.check_and_update_config(config)
    assert platform_mod.profile_for_config(config) is qwen
    # The hybrid page the platform aligned to is the profile's, and the state page is its
    # geometry function (2 * kv_heads * head_dim * element_size, times the engine page).
    assert config.cache_config.block_size == qwen.block_size_policy.hybrid_align
    assert config.cache_config.mamba_block_size == qwen.block_size_policy.hybrid_align
    assert config.cache_config.mamba_page_size_padded == qwen.hybrid.padded_page_size(
        config.cache_config.block_size, 2, 256, 2
    )

    # A pinned, misaligned hybrid page is refused with the profile's alignment in the message.
    with pytest.raises(ValueError) as excinfo:
        platform_mod.validate_block_size(
            _config(cache_config=_cache_config(block_size=1024)), qwen
        )
    assert str(qwen.block_size_policy.hybrid_align) in str(excinfo.value)
    # A non-multiple of the profile's kernel page is refused too.
    with pytest.raises(ValueError) as excinfo:
        platform_mod.validate_block_size(
            _config(
                model_config=_model_config(is_hybrid=False),
                cache_config=_cache_config(block_size=100),
            ),
            qwen,
        )
    assert str(qwen.block_size_policy.kernel_page_size) in str(excinfo.value)


def test_backend_enforces_the_profile_shape_heads_and_kernel_page() -> None:
    qwen = profiles.profile_by_name("qwen3.6")
    platform_mod.set_active_profile(qwen)

    shape = TileInferAttentionBackend.get_kv_cache_shape(7, 2048, 2, 256)
    assert shape == qwen.kv_shape(7, 2048, 2, 256)
    assert TileInferAttentionBackend.get_kv_cache_stride_order() == qwen.kv_shape.stride_order
    assert TileInferAttentionBackend.get_supported_head_sizes() == list(qwen.head_size)
    sizes = TileInferAttentionBackend.get_supported_kernel_block_sizes()
    assert sizes[0].base == qwen.block_size_policy.kernel_page_size
    # The head width is accepted/rejected by the profile's list, not a module tuple.
    assert _validate(head_size=qwen.head_size[0]) == []
    assert any("head_size" in reason for reason in _validate(head_size=64))
    # The GQA floor is the profile's.
    TileInferAttentionBackend.check_head_layout(
        num_heads=int(qwen.group_min) * 2,
        num_kv_heads=2,
        head_size=qwen.head_size[0],
    )
    with pytest.raises(ValueError) as excinfo:
        TileInferAttentionBackend.check_head_layout(
            num_heads=1, num_kv_heads=1, head_size=qwen.head_size[0]
        )
    assert f"group >= {qwen.group_min}" in str(excinfo.value)


def test_worker_mamba_gate_uses_the_profile_hybrid_kind() -> None:
    qwen = profiles.profile_by_name("qwen3.6")
    platform_mod.set_active_profile(qwen)

    path = worker_mod.mamba_state_path(_mamba_config(*qwen.hybrid.mamba_types))
    assert path["decision"] == "gdn-fla"
    assert set(path["types"]) == set(qwen.hybrid.mamba_types)
    assert path["gdn_layers"] == len(qwen.hybrid.mamba_types)

    # A type the profile routes through vLLM's SSU dispatch is refused at start-up, and the
    # message names the profile's state-update op.
    ssu_type = qwen.hybrid.ssu_mamba_types[0]
    with pytest.raises(RuntimeError) as excinfo:
        worker_mod.require_supported_mamba_state_path(_mamba_config(ssu_type))
    assert ssu_type in str(excinfo.value)
    assert qwen.hybrid.state_update_op in str(excinfo.value)


# =============================================================================================
# 3. a second profile drives the seam (the future-model proof)
# =============================================================================================


def _dummy_kv_shape(num_blocks: int, block_size: int, num_kv_heads: int, head_size: int):
    # Deliberately the *other* fused order: K/V outermost, blocks in dim 1.
    return (2, int(num_blocks), int(block_size), int(num_kv_heads), int(head_size))


def _dummy_profile() -> ModelProfile:
    return ModelProfile(
        name="dummy-profile",
        archs=("DummyForCausalLM",),
        head_size=(64,),
        group_min=4,
        block_size_policy=BlockSizePolicy(kernel_page_size=64, hybrid_align=512),
        kv_shape=KvShape(shape_fn=_dummy_kv_shape, stride_order=(0, 1, 2, 3, 4), block_dim=1),
        hybrid=None,
        op_family="dummy-family",
        unsupported_features=(
            UnsupportedFeature(
                field="use_mla",
                reason="dummy: MLA is not served",
                backend_reason="dummy backend refuses MLA",
            ),
            UnsupportedFeature(
                field="attn_type",
                kind="attn_type",
                reason="dummy: decoder only",
            ),
        ),
        notes="test-only profile proving the seam",
    )


def test_a_dummy_profile_drives_platform_and_backend() -> None:
    dummy = profiles.register_profile(_dummy_profile())
    config = _config(
        model_config=_model_config(
            architectures=["DummyForCausalLM"],
            is_hybrid=False,
            get_num_kv_heads=lambda parallel_config=None: 1,
            get_head_size=lambda: 64,
        ),
        cache_config=_cache_config(block_size=64, user_specified_block_size=True),
    )

    # The platform resolves the dummy by architecture and enforces *its* policy: 64-token pages
    # are valid here, while the Qwen profile would reject them as not a multiple of 128.
    platform_mod.TileInferPlatform.check_and_update_config(config)
    assert profiles.active_profile() is dummy
    assert config.cache_config.block_size == 64
    assert platform_mod.HYBRID_BLOCK_SIZE != 64  # the module alias still describes the boot profile
    assert TileInferAttentionBackend.get_supported_head_sizes() == [64]
    assert TileInferAttentionBackend.get_supported_kernel_block_sizes()[0].base == 64
    assert TileInferAttentionBackend.get_kv_cache_shape(7, 64, 1, 64) == (2, 7, 64, 1, 64)
    assert TileInferAttentionBackend.get_kv_cache_stride_order() == (0, 1, 2, 3, 4)
    assert platform_mod.TileInferPlatform.support_hybrid_kv_cache() is False  # dummy.hybrid is None

    # The dummy's GQA floor (4) is the one enforced, and its head width is the one accepted.
    assert _validate(head_size=64, block_size=64) == []
    TileInferAttentionBackend.check_head_layout(num_heads=8, num_kv_heads=2, head_size=64)
    with pytest.raises(ValueError) as excinfo:
        TileInferAttentionBackend.check_head_layout(num_heads=2, num_kv_heads=1, head_size=64)
    assert "group >= 4" in str(excinfo.value)

    # Its reject list is the one the platform/backend consult: use_mla is rejected with the
    # dummy's own wording, not Qwen3.6's.
    reasons = _validate(head_size=64, block_size=64, use_mla=True)
    assert reasons == ["dummy backend refuses MLA"]


# =============================================================================================
# 4. ops.install follows the profile's op family
# =============================================================================================


class _DummyFamily:
    """A minimal operator family: one op whose vLLM target deliberately does not resolve."""

    @staticmethod
    def required_names() -> Tuple[str, ...]:
        return ("dummy_probe_op",)

    @staticmethod
    def bind() -> list:
        return [
            install_mod.OpBinding(
                op="dummy_probe_op",
                family="gdn",  # an enabled switch family, so the plan reaches target resolution
                targets=("no_such_tileinfer_family.module.dummy_probe_op",),
                impl=lambda *args, **kwargs: None,
                why="test-only binding",
                remove_when="never; this family exists only in a test",
                upstream_ref="tests/test_model_profiles.py",
            )
        ]


def test_ops_family_iteration_follows_the_profile() -> None:
    dummy = profiles.register_profile(_dummy_profile())
    families_mod.register_family("dummy-family", _DummyFamily)
    profiles.set_active_profile(dummy)

    assert [binding.op for binding in install_mod.bindings()] == ["dummy_probe_op"]
    assert install_mod.required_names() == ("dummy_probe_op",)

    report = install_mod.install_ops()
    assert "dummy_probe_op" in report.unbound
    assert "no vLLM target resolves" in report.details["dummy_probe_op"]

    # Restoring the Qwen profile restores the shipped operator table: no edit to install.py.
    qwen = profiles.profile_by_name("qwen3.6")
    profiles.set_active_profile(qwen)
    install_mod._reset()
    assert [binding.op for binding in install_mod.bindings()] == list(
        families_mod.family_for("gdn-moe").required_names()
    )
