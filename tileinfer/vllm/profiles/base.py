"""The pluggable **model profile** layer: every model fact in one typed record.

Why this module exists
----------------------
The vLLM integration started as the Qwen3.6 port, so a handful of model facts (head widths, the
GQA group floor, the 2048-token hybrid page, the fused K/V layout, the GDN op family, the reject
list) lived as constants and ``if`` statements in ``platform.py``, ``attn/backend.py``,
``worker.py`` and ``ops/install.py``.  That is fine while exactly one model is served and wrong the
moment a second one is: the backend, the platform and the worker then each have to be edited, and
nothing says which of their statements were *model* choices and which were *hardware* facts.

This module draws that line.  A :class:`ModelProfile` carries the model-dependent facts; the
hardware/kernel facts (bf16/fp16, the 128-token kernel tile's arithmetic, the vLLM ABC shape) stay
where they are.  Consumers call :func:`active_profile` (or :func:`profile_for` when they have an
architecture list) instead of reading a module constant.

What is deliberately **not** here
---------------------------------
* The kernel page tile (``serving.buckets.KERNEL_PAGE_SIZE``) is a compiled-kernel property, not a
  model choice; the profile *references* it in its block-size policy rather than redefining it.
* The compute / KV-cache dtype sets are backend properties (the paged kernels are bf16/fp16).
* Worker/platform mechanics (``worker_cls``, the ABC reject policy shape, eager gating) are fixed
  for this backend line.

Import policy
-------------
No ``vllm`` and no ``torch_npu`` import: this package must stay importable on a CPU box so the
launcher's ``--plan-only`` and the contract tests can resolve a profile without an engine.  It does
import ``tileinfer.serving.buckets`` (which imports ``torch``) for the kernel page constant.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

__all__ = [
    "BlockSizePolicy",
    "HybridKind",
    "KvShape",
    "ModelProfile",
    "UnsupportedFeature",
    "PROFILES",
    "profile_for",
    "profile_by_name",
    "register_profile",
    "unregister_profile",
    "known_profiles",
    "default_profile",
    "active_profile",
    "set_active_profile",
    "reset_active_profile",
]


# =============================================================================================
# Small value objects
# =============================================================================================


@dataclass(frozen=True)
class BlockSizePolicy:
    """How the engine page relates to the kernel tile, and to a hybrid model's state page.

    ``kernel_page_size``
        Tokens per kernel page.  An engine page must be a whole number of these, because the pool
        is re-viewed as ``[pages, 128, heads, dim]`` without a copy.
    ``hybrid_align``
        For an attention + state (mamba/GDN) hybrid, the engine page must also be a multiple of the
        model's state page.  ``None`` means the model has no such page.
    """

    kernel_page_size: int
    hybrid_align: Optional[int] = None

    def __post_init__(self) -> None:
        if int(self.kernel_page_size) <= 0:
            raise ValueError(
                f"BlockSizePolicy.kernel_page_size must be positive, got {self.kernel_page_size!r}"
            )
        if self.hybrid_align is not None:
            align = int(self.hybrid_align)
            if align <= 0 or align % int(self.kernel_page_size) != 0:
                raise ValueError(
                    "BlockSizePolicy.hybrid_align must be a positive multiple of "
                    f"kernel_page_size={self.kernel_page_size}, got {self.hybrid_align!r}"
                )

    def required_multiple(self, hybrid: bool) -> int:
        """The page multiple the model needs (the hybrid alignment when it has one)."""
        if hybrid and self.hybrid_align is not None:
            return int(self.hybrid_align)
        return int(self.kernel_page_size)

    def is_kernel_multiple(self, block_size: int) -> bool:
        return int(block_size) > 0 and int(block_size) % int(self.kernel_page_size) == 0

    def align_up(self, block_size: int, hybrid: bool) -> int:
        """Round ``block_size`` up to the model's required page multiple."""
        multiple = self.required_multiple(hybrid)
        return -(-int(block_size) // multiple) * multiple


@dataclass(frozen=True)
class KvShape:
    """The fused KV-cache declaration: a callable shape plus its physical stride order.

    ``shape_fn(num_blocks, block_size, num_kv_heads, head_size)`` returns the *logical* shape
    vLLM sees.  ``stride_order`` is what the runner allocates physically (the permutation it then
    permutes back), so the two belong together: a shape whose K/V halves must stay contiguous is
    only meaningful with the matching stride order.

    ``block_dim`` is the index of ``num_blocks`` in the logical shape.  The runner's probe derives
    it from the shape, but the worker's pre-warm also needs it for the legacy-pool fallback, so it
    is declared rather than inferred in two places.
    """

    shape_fn: Callable[[int, int, int, int], Tuple[int, ...]]
    stride_order: Tuple[int, ...] = (1, 0, 2, 3, 4)
    block_dim: int = 0

    def __post_init__(self) -> None:
        if not callable(self.shape_fn):
            raise ValueError("KvShape.shape_fn must be callable")
        order = tuple(int(i) for i in self.stride_order)
        if sorted(order) != list(range(5)):
            raise ValueError(
                f"KvShape.stride_order must be a permutation of 0..4, got {self.stride_order!r}"
            )
        if int(self.block_dim) not in range(5):
            raise ValueError(f"KvShape.block_dim must be in 0..4, got {self.block_dim!r}")

    def __call__(
        self, num_blocks: int, block_size: int, num_kv_heads: int, head_size: int
    ) -> Tuple[int, ...]:
        return self.shape(num_blocks, block_size, num_kv_heads, head_size)

    def shape(
        self, num_blocks: int, block_size: int, num_kv_heads: int, head_size: int
    ) -> Tuple[int, ...]:
        values = self.shape_fn(
            int(num_blocks), int(block_size), int(num_kv_heads), int(head_size)
        )
        shape = tuple(int(n) for n in values)
        if len(shape) != 5:
            raise ValueError(
                f"KvShape.shape_fn must return a 5-D shape, got {shape!r}"
            )
        if shape[int(self.block_dim)] != int(num_blocks):
            raise ValueError(
                "KvShape.block_dim does not address the num_blocks dimension: "
                f"block_dim={self.block_dim}, shape={shape!r}, num_blocks={num_blocks}"
            )
        return shape


@dataclass(frozen=True)
class HybridKind:
    """An attention + state hybrid (Qwen3.6's GDN layers) and its page geometry.

    ``mamba_types``
        ``MambaSpec.mamba_type`` values this profile serves (e.g. ``("GDN_ATTN",)``).
    ``ssu_mamba_types``
        Values that would route the state update through vLLM's Mamba **SSU** dispatch - the one
        mamba piece with no public extension point on this backend.  A model that needs one is
        refused at start-up, loudly.
    ``page_size_padded``
        The geometry function: ``(block_size, num_kv_heads, head_size, element_size) -> bytes`` for
        one padded mamba state page.  It must produce exactly one attention page, which is what
        makes the two page kinds interchangeable to the block manager.
    ``state_update_op``
        The vLLM/op name of the state update this profile's layers use (documented in the
        worker's refusal message).
    """

    name: str
    mamba_types: Tuple[str, ...]
    ssu_mamba_types: Tuple[str, ...]
    page_size_padded: Callable[[int, int, int, int], int]
    state_update_op: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("HybridKind.name must not be empty")
        if not self.mamba_types:
            raise ValueError("HybridKind.mamba_types must not be empty")
        if not callable(self.page_size_padded):
            raise ValueError("HybridKind.page_size_padded must be callable")

    def page_bytes_per_token(self, num_kv_heads: int, head_size: int, element_size: int) -> int:
        """Size of one token's attention page: ``2 (K/V) * kv_heads * head_dim * dtype``."""
        return 2 * int(num_kv_heads) * int(head_size) * int(element_size)

    def padded_page_size(
        self, block_size: int, num_kv_heads: int, head_size: int, element_size: int
    ) -> int:
        """One padded mamba state page, in bytes (the profile's geometry function)."""
        return int(
            self.page_size_padded(
                int(block_size), int(num_kv_heads), int(head_size), int(element_size)
            )
        )


@dataclass(frozen=True)
class UnsupportedFeature:
    """One feature a profile refuses, with the reason its consumer should print.

    ``kind`` is one of:

    ``selector``
        An ``AttentionSelectorConfig`` boolean that identifies a real selector and is rejected when
        true (``use_mla``, ``use_sparse``, ...).
    ``plain_selector``
        A selector boolean whose absence on a duck-typed object is tolerated (engine-mode switches).
    ``attn_type``
        The selector's non-boolean ``attn_type`` field.
    ``config``
        A ``VllmConfig`` / ``CacheConfig`` / ``ModelConfig`` flag rejected in
        :meth:`TileInferPlatform.check_and_update_config`.  ``path``/``owners``/``predicate``/
        ``message``/``why`` describe how to read and refuse it.

    ``reason`` is the selector policy's parenthetical reason (``platform.SELECTOR_REJECT_REASONS``);
    ``backend_reason`` is the sentence ``TileInferAttentionBackend.validate_configuration`` returns
    for the same field (the two messages were always different, and the tests pin both).
    """

    field: str
    reason: str
    kind: str = "selector"
    backend_reason: str = ""
    path: Tuple[str, ...] = ()
    owners: Tuple[str, ...] = ()
    predicate: str = ""
    message: str = ""
    why: str = ""

    def __post_init__(self) -> None:
        if not self.field:
            raise ValueError("UnsupportedFeature.field must not be empty")
        if self.kind not in ("selector", "plain_selector", "attn_type", "config"):
            raise ValueError(f"UnsupportedFeature.kind={self.kind!r} is not one of the four kinds")
        if self.kind == "config":
            if not self.path or len(self.path) != len(self.owners):
                raise ValueError(
                    "a config UnsupportedFeature needs a non-empty path and one owner per path step"
                )
            if self.predicate not in ("is_not_none", "is_truthy", "not_in_cache_dtypes"):
                raise ValueError(
                    f"config feature {self.field!r} has unknown predicate {self.predicate!r}"
                )
            if not self.message:
                raise ValueError(f"config feature {self.field!r} has no message")


# =============================================================================================
# The profile record
# =============================================================================================


@dataclass(frozen=True)
class ModelProfile:
    """One served model family: every fact the vLLM layer would otherwise hard-code.

    ``archs``
        vLLM architecture strings (``Qwen3_5MoeForConditionalGeneration``, ...).  A trailing ``*``
        is accepted as a family prefix, but explicit spellings are preferred: they are what vLLM's
        registry names.
    ``head_size``
        Attention head widths the profile's kernels accept.
    ``group_min``
        Minimum ``num_heads // num_kv_heads`` (the paged kernel's GQA floor).
    ``block_size_policy``
        Kernel-page multiple + hybrid state-page alignment.
    ``kv_shape``
        Fused KV declaration (callable shape + physical stride order + block dim).
    ``hybrid``
        ``None`` for an attention-only model, otherwise the state-page geometry.
    ``op_family``
        The operator family module naming this model's takeovers
        (``tileinfer.vllm.ops.families.<op_family>``; ``gdn-moe`` today).
    ``unsupported_features``
        Everything this profile refuses, with the reason each consumer prints.
    ``mixed_attention_specs``
        True when the model's attention layers mix spec types that vLLM's unified-spec mode
        would flatten (MuseGlimmer's iRoPE: full-attention + sliding-window layers).  The
        platform answers ``support_hybrid_kv_cache() == True`` for such a profile so vLLM builds
        one group per spec and the sliding-window allocator can drop out-of-window blocks —
        the memory optimisation the unified path (``unify_hybrid_kv_cache_specs``) gives up.
        The window still lives in the *compute* (the backend's reference path), not in the
        pool layout; this flag is only about how many groups the engine creates.
    ``notes``
        Free-form provenance / scope notes; never parsed.
    """

    name: str
    archs: Tuple[str, ...]
    head_size: Tuple[int, ...]
    group_min: int
    block_size_policy: BlockSizePolicy
    kv_shape: KvShape
    hybrid: Optional[HybridKind]
    op_family: str
    unsupported_features: Tuple[UnsupportedFeature, ...]
    notes: str = ""
    mixed_attention_specs: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ModelProfile.name must not be empty")
        if not self.archs:
            raise ValueError(f"model profile {self.name!r} declares no architectures")
        if not self.head_size:
            raise ValueError(f"model profile {self.name!r} declares no head sizes")
        if int(self.group_min) < 1:
            raise ValueError(f"model profile {self.name!r} has group_min < 1")
        if not self.op_family:
            raise ValueError(f"model profile {self.name!r} declares no op_family")

    # -- small queries the consumers share ----------------------------------------------------

    def supports_head_size(self, head_size: int) -> bool:
        return int(head_size) in tuple(int(n) for n in self.head_size)

    def features(self, *kinds: str) -> Tuple[UnsupportedFeature, ...]:
        wanted = set(kinds)
        return tuple(feature for feature in self.unsupported_features if feature.kind in wanted)

    def selector_features(self) -> Tuple[UnsupportedFeature, ...]:
        return self.features("selector")

    def plain_selector_features(self) -> Tuple[UnsupportedFeature, ...]:
        return self.features("plain_selector")

    def attn_type_features(self) -> Tuple[UnsupportedFeature, ...]:
        return self.features("attn_type")

    def config_features(self) -> Tuple[UnsupportedFeature, ...]:
        return self.features("config")

    def backend_features(self) -> Tuple[UnsupportedFeature, ...]:
        """Selector fields the attention backend refuses (its ``validate_configuration`` list)."""
        return self.features("selector", "plain_selector")


# =============================================================================================
# Registry
# =============================================================================================

#: ``name -> ModelProfile``, in registration order (the first profile is the default).
PROFILES: Dict[str, ModelProfile] = {}

_LOCK = threading.RLock()
_ACTIVE_NAME: Optional[str] = None


def register_profile(profile: ModelProfile, *, replace: bool = False) -> ModelProfile:
    """Add ``profile`` to the registry; returns it (so a module can ``NAME = register_profile(...)``)."""
    if not isinstance(profile, ModelProfile):
        raise TypeError(f"register_profile expects a ModelProfile, got {type(profile).__name__}")
    with _LOCK:
        if profile.name in PROFILES and not replace:
            existing = PROFILES[profile.name]
            if existing is profile:
                return existing
            raise ValueError(
                f"a model profile named {profile.name!r} is already registered; pass replace=True "
                "to replace it, or unregister_profile() first"
            )
        PROFILES[profile.name] = profile
    return profile


def unregister_profile(name: str) -> ModelProfile:
    """Remove a profile (tests / a plugin reload); returns the removed record."""
    with _LOCK:
        if name not in PROFILES:
            raise KeyError(
                f"no model profile named {name!r}; known profiles: {known_profiles()}"
            )
        profile = PROFILES.pop(name)
        global _ACTIVE_NAME
        if _ACTIVE_NAME == name:
            _ACTIVE_NAME = None
    return profile


def known_profiles() -> Tuple[str, ...]:
    with _LOCK:
        return tuple(PROFILES)


def profile_by_name(name: str) -> ModelProfile:
    with _LOCK:
        profile = PROFILES.get(str(name))
        if profile is not None:
            return profile
        known = ", ".join(PROFILES) or "<none>"
    raise ValueError(f"no model profile named {name!r}; known profiles: {known}")


def default_profile() -> ModelProfile:
    """The first registered profile (Qwen3.6 today).  Raises when the registry is empty."""
    with _LOCK:
        if not PROFILES:
            raise RuntimeError(
                "no model profiles are registered; tileinfer.vllm.profiles.qwen3_6 must be "
                "imported (importing tileinfer.vllm.profiles does that)"
            )
        return next(iter(PROFILES.values()))


def active_profile() -> ModelProfile:
    """The profile in force in this process (the platform sets it from the model's arch)."""
    with _LOCK:
        if _ACTIVE_NAME is not None and _ACTIVE_NAME in PROFILES:
            return PROFILES[_ACTIVE_NAME]
        return default_profile()


def set_active_profile(profile: "ModelProfile | str") -> ModelProfile:
    """Make a registered profile the process-wide active one; returns it.

    The platform calls this from ``check_and_update_config`` after resolving the model's
    architecture.  Backend/worker helpers have no config argument, so this is the seam that lets
    them answer for the model being served instead of for the default profile.
    """
    global _ACTIVE_NAME
    resolved = profile if isinstance(profile, ModelProfile) else profile_by_name(profile)
    with _LOCK:
        if resolved.name not in PROFILES:
            raise ValueError(
                f"model profile {resolved.name!r} is not registered; call register_profile() first"
            )
        if PROFILES[resolved.name] is not resolved:
            raise ValueError(
                f"a different profile named {resolved.name!r} is registered; activate the "
                "registered instance (or register_profile(..., replace=True))"
            )
        _ACTIVE_NAME = resolved.name
    return resolved


def reset_active_profile() -> None:
    """Forget the explicit activation; :func:`active_profile` falls back to the default."""
    global _ACTIVE_NAME
    with _LOCK:
        _ACTIVE_NAME = None


# -- resolution -------------------------------------------------------------------------------


def _iter_archs(archs: Any) -> Tuple[str, ...]:
    if archs is None:
        return ()
    if isinstance(archs, str):
        return (archs.strip(),) if archs.strip() else ()
    return tuple(str(arch).strip() for arch in archs if str(arch).strip())


def _arch_matches(pattern: str, arch: str) -> bool:
    if pattern == arch:
        return True
    if pattern.endswith("*"):
        return arch.startswith(pattern[:-1])
    return False


def _known_list() -> str:
    with _LOCK:
        entries = tuple(PROFILES.values())
    if not entries:
        return "<none>"
    return ", ".join(
        f"{profile.name} (archs={list(profile.archs)})" for profile in entries
    )


def _no_match_error(archs: Tuple[str, ...], hints: Mapping[str, Any], detail: str = "") -> ValueError:
    asked = list(archs) if archs else "<no architecture given>"
    extra = f" {detail}" if detail else ""
    return ValueError(
        f"no TileInfer model profile matches {asked};{extra} known profiles: {_known_list()}. "
        "Register the model with register_profile() in tileinfer/vllm/profiles/ (a new "
        "tileinfer/vllm/profiles/<name>.py that calls register_profile()) or pass the profile "
        "name explicitly. Hints seen: "
        f"{dict(hints) if hints else '{}'}."
    )


def profile_for(
    archs: Any, config_hints: Optional[Mapping[str, Any]] = None
) -> ModelProfile:
    """Resolve a profile by vLLM architecture name, or raise listing every known profile.

    ``config_hints`` may carry ``name``/``profile`` to select a profile explicitly, and
    ``head_size`` / ``group_min`` / ``is_hybrid`` to narrow candidates when an architecture string
    is shared between families.  Hints never *invent* a match: an unknown architecture still
    raises.
    """
    hints: Mapping[str, Any] = dict(config_hints or {})
    explicit = hints.get("profile", hints.get("name"))
    if explicit is not None:
        return profile_by_name(str(explicit))

    wanted = _iter_archs(archs)
    with _LOCK:
        candidates = tuple(PROFILES.values())

    if wanted:
        candidates = tuple(
            profile
            for profile in candidates
            if any(_arch_matches(pattern, arch) for pattern in profile.archs for arch in wanted)
        )
        if not candidates:
            raise _no_match_error(wanted, hints)
        # The architecture is the authoritative key: one exact match is the answer even when a
        # config hint disagrees (the mismatch is then the model's problem, reported by the backend's
        # head/group checks, not by a confusing "profile not found").
        if len(candidates) == 1:
            return candidates[0]

    if hints:
        narrowed = []
        for profile in candidates:
            if "head_size" in hints and not profile.supports_head_size(int(hints["head_size"])):
                continue
            if "group_min" in hints and int(profile.group_min) < int(hints["group_min"]):
                continue
            if "is_hybrid" in hints:
                wants_hybrid = bool(hints["is_hybrid"])
                if (profile.hybrid is not None) is not wants_hybrid:
                    continue
            narrowed.append(profile)
        if not narrowed:
            raise _no_match_error(wanted, hints, "the config hints exclude every candidate;")
        candidates = tuple(narrowed)

    if not candidates:
        raise _no_match_error(wanted, hints)
    if len(candidates) > 1:
        names = ", ".join(profile.name for profile in candidates)
        raise ValueError(
            f"architecture(s) {list(wanted) or '<none>'} are ambiguous between profiles {names}; "
            "pass config_hints={'profile': <name>} or register distinct architecture strings"
        )
    return candidates[0]
