"""``TileInferAttentionBackend`` — the vLLM `AttentionBackend` for the TileInfer kernels.

What this class owns
--------------------
Three things, and nothing else:

* **the KV-cache layout** (:meth:`get_kv_cache_shape` + :meth:`get_kv_cache_stride_order`).  The
  logical shape is ``(num_blocks, 2, block_size, num_kv_heads, head_size)`` - block-first, so that
  vLLM discovers the block in dim 0 (``v1/attention/backend.py:98-116``) and its hybrid
  attention/mamba rewrite, which only fires for ``block_dim == 1``, leaves the pool alone
  (``gpu_model_runner.py:7167-7199``).  The *physical* order is declared as
  ``(1, 0, 2, 3, 4)`` = ``[2][num_blocks][block_size][heads][dim]``, which the runner honours when
  it allocates (``gpu_model_runner.py:7093-7110``); that is what keeps ``kv_cache[:, 0]`` and
  ``kv_cache[:, 1]`` contiguous halves, so ``write_kv_cache_fused`` and ``pool_sub_view`` stay
  zero-copy.  Any other layout (HND, K/V interleaved per block, or the un-fused *2 tensors per
  layer* form) either breaks the write path or forces a per-step transpose/repack.
* **the kernel block-size policy** (:meth:`get_supported_kernel_block_sizes`).  The kernels tile
  ``KERNEL_PAGE_SIZE = 128`` tokens, so the engine page must be a multiple of 128 — expressed as
  ``MultipleOf(128)`` so vLLM's hybrid-block-size logic may pick 2048 (the mamba page) and still
  accept it (``v1/attention/backend.py:183-191``: a ``MultipleOf`` base only requires
  ``block_size % base == 0``).
* **the start-up reject policy** (:meth:`validate_configuration`).  A backend that raises on the
  first serving step instead of at configuration time is undebuggable: the engine has already
  loaded weights, allocated the KV pool and captured graphs.  Every limitation of the TileInfer
  kernel is therefore rejected here, where vLLM is still willing to fall back to another backend.

What it deliberately does *not* do
----------------------------------
``accept_output_buffer`` is part of the port plan (``docs/plan-vllm-native-backend.md`` §W4) but
**does not exist in the local vLLM 0.23 source**: ``grep -n accept_output_buffer`` over
``reference/vllm-0.23/`` finds no match, and the ``AttentionBackend`` ABC
(``v1/attention/backend.py:55-349``) has no such name.  A minimal forward-compatible hook is kept
below because the impl *does* accept an ``output`` buffer (``v1/attention/backend.py:797-809``);
it is documented as not being part of the 0.23 contract rather than being presented as one.

Import policy
-------------
This module is the boundary: it may only be imported inside a vLLM process.  The vLLM import is
guarded so that the failure names the exact upstream module and version, instead of surfacing as
an ``AttributeError`` three frames deep in the engine.
"""

from __future__ import annotations

import torch

try:  # pragma: no cover - exercised only outside a vLLM env (the CPU suite stubs `vllm`)
    from vllm.v1.attention.backend import AttentionBackend, AttentionType, MultipleOf
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend as vllm_register_backend,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "tileinfer.vllm.attn.backend requires vLLM 0.23's v1 attention API "
        "(vllm.v1.attention.backend.AttentionBackend/MultipleOf and "
        "vllm.v1.attention.backends.registry); importing it failed with "
        f"{exc!r}. Install vllm>=0.23,<0.24 - this module is not importable without vLLM."
    ) from exc

from tileinfer.serving.buckets import KERNEL_PAGE_SIZE
from tileinfer.vllm.profiles import active_profile, default_profile

__all__ = [
    "TileInferAttentionBackend",
    "SUPPORTED_HEAD_SIZES",
    "SUPPORTED_DTYPES",
    "SUPPORTED_KV_CACHE_DTYPES",
    "register_with_vllm",
]

#: The boot profile: what the module-level compatibility constants below describe.  The *methods*
#: consult :func:`active_profile` at call time, so a second model family drives them; these aliases
#: exist because older call sites/tests read them as data (the launcher pins them to the profile).
_BOOT_PROFILE = default_profile()

#: Attention head sizes of the boot profile.  ``o_proj`` of Qwen3.6 is ``[4096 -> 2048]`` = 16 x 256,
#: so 256 is the live case; 128 is the GDN head width.  A future model's profile supplies its own
#: list through :meth:`TileInferAttentionBackend.get_supported_head_sizes`.
SUPPORTED_HEAD_SIZES = tuple(_BOOT_PROFILE.head_size)

#: Compute dtypes.  The kernels are bf16/fp16 only (``docs/plan-vllm-native-backend.md`` §0
#: excludes quantisation), and a silent fp32 fallback would be a different numerical contract.
SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)

#: KV-cache dtypes: unquantised only.  ``"auto"`` means "the model dtype" (vLLM's own spelling,
#: see the base class's ``supported_kv_cache_dtypes`` at ``v1/attention/backend.py:59-63``).
SUPPORTED_KV_CACHE_DTYPES = ("auto", "bfloat16", "float16")


class TileInferAttentionBackend(AttentionBackend):
    """Paged attention on the TileInfer kernels (Ascend 950), selected per attention layer."""

    # ------------------------------------------------------------------ class-level policy
    #
    # These are read by the ABC's generic checks (`supports_dtype`, `supports_kv_cache_dtype`,
    # `supports_head_size`), which is why they are class attributes and not properties.

    supported_dtypes = list(SUPPORTED_DTYPES)
    supported_kv_cache_dtypes = list(SUPPORTED_KV_CACHE_DTYPES)

    #: We write this step's K/V inside `forward` (via `kv_cache.write_kv_cache`, on the halves
    #: `kv_cache.fused_halves` returns), which is what the base class's flag means
    #: (`v1/attention/backend.py:65-66`).  Setting it truthfully matters: vLLM skips the separate
    #: `do_kv_cache_update` hook when it is True, and if the flag and the impl disagree the cache
    #: is written twice or not at all.
    forward_includes_kv_cache_update = True

    # ------------------------------------------------------------------ identity

    @staticmethod
    def get_name() -> str:
        """The registry slot, **not** a brand name.

        vLLM does ``AttentionBackendEnum[self.attn_backend.get_name()]`` while building every
        attention layer, so this string must be an existing enum member; ``"TILEINFER"`` is not
        one and raised ``ValueError: Unknown attention backend: 'TILEINFER'`` on the first
        hardware run. We occupy the ``CUSTOM`` slot, which is also the value
        ``--attention-backend CUSTOM`` selects and the slot the plugin registers
        (``AttentionBackendEnum.CUSTOM``).
        """
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        # Imported lazily: `impl` pulls in the serving stack (plan cache, step buffers) and the
        # attention registry.  Keeping it out of module import means a backend *selection* probe
        # does not initialise that whole layer.
        from .impl import TileInferAttentionImpl

        return TileInferAttentionImpl

    @staticmethod
    def get_builder_cls():
        from .metadata import TileInferMetadataBuilder

        return TileInferMetadataBuilder

    # ------------------------------------------------------------------ capability bits
    #
    # Every one of these is an explicit `False` instead of an inherited default: the defaults are
    # the *absence* of a feature, and the port plan's whole reject policy is "say no loudly".  The
    # base implementations being overridden are at `v1/attention/backend.py:204-245`.

    @classmethod
    def is_mla(cls) -> bool:
        return False

    @classmethod
    def is_sparse(cls) -> bool:
        return False

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return False

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @classmethod
    def supports_non_causal(cls) -> bool:
        return False

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return False

    @classmethod
    def supports_kv_connector(cls) -> bool:
        # The ABC defaults to True (`v1/attention/backend.py:243-245`), which is wrong for us: a KV
        # connector moves cache blocks behind the layer's back, while our `forward` writes the
        # fused pool directly and the serving plan is built for one pool capacity per process.
        return False

    # ------------------------------------------------------------------ KV-cache layout

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """``(num_blocks, 2, block_size, num_kv_heads, head_size)``: block-first fused NHD.

        Why block-first, and not the vLLM-native ``(2, num_blocks, ...)``:

        * ``2`` is the K/V factor.  The base class discovers the block dimension with
          ``shape.index(1234567)`` (`v1/attention/backend.py:98-116`); with this shape that is
          dim 0, i.e. ``block_dim == 0``.
        * ``block_dim == 0`` is what makes vLLM's hybrid rewrite a no-op.  In a run with both
          attention and mamba layers, ``_update_hybrid_attention_mamba_layout``
          (``gpu_model_runner.py:7167-7199``) *reinterprets* a ``block_dim == 1`` pool as physically
          ``[blocks][2][page][heads][dim]`` - it keeps ``size=kv_cache.shape`` and only rewrites
          dims 0/1's strides (``:7196-7199``).  That interleaves K and V per block, so a page is
          addressed by three indices and there is no flat ``[pages, 128, heads, dim]`` view of
          either half; the paged kernel, which reads ``KCache[pid, ...]`` and ``VCache[pid, ...]``
          with one page id (``kernels/attention/paged_decode_ascend950.py:148-149``, ``:202-203``),
          could then only be fed by copying the pool.  ``:7190-7191`` skips the rewrite for
          ``block_dim == 0``, so this shape keeps the halves separable.
        * NHD (``[block, token, head, dim]``) is not cosmetic either: the write scatters per-token
          slots and ``pool_sub_view`` re-views ``[blocks, block_size, heads, dim]`` as
          ``[blocks * sub, 128, heads, dim]`` zero-copy.  Both are only valid in NHD.
        * The *physical* order is not this shape: see :meth:`get_kv_cache_stride_order`.

        The shape itself is the active model profile's declaration
        (:attr:`tileinfer.vllm.profiles.ModelProfile.kv_shape`), not a backend constant: another
        model family may fuse K/V differently or put the block elsewhere, and the runner discovers
        both from this call.
        """
        return active_profile().kv_shape(
            int(num_blocks), int(block_size), int(num_kv_heads), int(head_size)
        )

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        """Physical order ``(1, 0, 2, 3, 4)`` = ``[2][blocks][page][heads][dim]``.

        The runner allocates ``raw.view(tuple(shape[i] for i in stride_order))`` and then permutes
        the view back to the logical shape (`gpu_model_runner.py:7093-7110`), so this return value
        alone decides the bytes.  With the logical shape ``(blocks, 2, ...)``:

        * physical ``[2][blocks][...]`` -> ``kv_cache[:, 0]`` / ``kv_cache[:, 1]`` are *contiguous*
          ``[blocks, page, heads, dim]`` tensors (they differ by one storage offset), so the write,
          ``flat_pool``, ``pool_sub_view`` and the kernel's ``[pages, 128, heads, dim]`` argument
          all see one flat array - zero copy, unchanged page ids;
        * physical ``[blocks][2][...]`` (the default, no override) would instead make each half
          strided in the block dimension: exactly the layout the kernel cannot read.

        Note what this is *not*: the physical layout is K/V-outermost, so the per-block page is
        **not** one contiguous ``page_size_bytes`` region.  vLLM's block zeroer
        (``v1/worker/utils.py:131-170``) derives the per-block byte count from
        ``stride(block_dim)``, which is therefore half a page here - see the port's status notes;
        it is a vLLM-side assumption that no attention backend with a stride order satisfies.

        ``include_num_layers_dimension`` is unsupported on purpose: a cross-layer uniform layout
        exists for KV-connector transfers, and this backend rejects KV connectors
        (:meth:`supports_kv_connector` is ``False``), so there is no layout to declare.
        """
        if include_num_layers_dimension:
            raise NotImplementedError(
                "TileInfer declares no cross-layer uniform KV layout: KV connectors are not "
                "supported (TileInferAttentionBackend.supports_kv_connector() is False)"
            )
        return tuple(active_profile().kv_shape.stride_order)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list:
        """Multiples of the kernel page (128), not one fixed size.

        The engine runs block_size 2048 for the Qwen3.6 hybrid model (the mamba state page), and
        2048 = 16 x 128 consecutive sub-pages.  Returning ``MultipleOf(128)`` lets
        ``supports_block_size`` accept 2048 while still rejecting 192 (`v1/attention/backend.py:
        174-191`).  Both numbers come from the active model profile's block-size policy.
        """
        policy = active_profile().block_size_policy
        return [MultipleOf(int(policy.kernel_page_size))]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [int(width) for width in active_profile().head_size]

    # ------------------------------------------------------------------ start-up validation

    @classmethod
    def validate_configuration(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        use_per_head_quant_scales: bool,
        device_capability,
        attn_type: str,
        use_non_causal: bool = False,
        use_batch_invariant: bool = False,
        use_kv_connector: bool = False,
    ) -> list[str]:
        """Return every reason this configuration cannot run; empty means "supported".

        The base implementation (`v1/attention/backend.py:274-341`) already covers the policy
        encoded in the class attributes above.  It is *called* (so a future upstream check is
        inherited) **and** the TileInfer-specific statements are made explicitly, because this is
        the last place where vLLM can still choose a different backend: everything here is a
        start-up rejection, never a first-step failure.

        ``num_heads`` / ``num_kv_heads`` are **not** parameters of this method (they are absent
        from the signature at `v1/attention/backend.py:275-291`), so the GQA group cannot be
        checked here.  That check lives in ``TileInferAttentionImpl.__init__`` — which runs while
        the model is being constructed, i.e. also before the first step
        (``TileInferAttentionBackend.check_head_layout`` is the shared implementation).
        """
        reasons: list[str] = []
        try:
            reasons.extend(
                super().validate_configuration(
                    head_size,
                    dtype,
                    kv_cache_dtype,
                    block_size,
                    use_mla,
                    has_sink,
                    use_sparse,
                    use_mm_prefix,
                    use_per_head_quant_scales,
                    device_capability,
                    attn_type,
                    use_non_causal=use_non_causal,
                    use_batch_invariant=use_batch_invariant,
                    use_kv_connector=use_kv_connector,
                )
            )
        except (TypeError, AttributeError):  # pragma: no cover - non-0.23 base / stubbed base
            # A stub base (CPU contract tests) has no generic validation.  Never swallow it in a
            # real vLLM env: there the attributes exist and a failure is a genuine upstream
            # incompatibility, so only the two "shape changed" errors are tolerated.
            if _is_real_vllm_base():
                raise

        # --- TileInfer-specific policy (each one is a hard reject, not a fallback) -------------
        profile = active_profile()
        head_sizes = [int(width) for width in profile.head_size]
        if head_size not in head_sizes:
            reasons.append(
                f"head_size {head_size} is not one of {head_sizes} "
                "(the TileInfer paged kernels are compiled for these widths only)"
            )
        if dtype not in SUPPORTED_DTYPES:
            reasons.append(
                f"dtype {dtype} is not bf16/fp16 "
                f"(supported: {[str(d) for d in SUPPORTED_DTYPES]})"
            )
        if kv_cache_dtype not in SUPPORTED_KV_CACHE_DTYPES:
            reasons.append(
                f"kv_cache_dtype {kv_cache_dtype!r} is quantised/unknown "
                f"(supported: {list(SUPPORTED_KV_CACHE_DTYPES)})"
            )
        kernel_page = int(profile.block_size_policy.kernel_page_size)
        if block_size is not None and int(block_size) % kernel_page != 0:
            reasons.append(
                f"block_size {block_size} is not a multiple of {kernel_page} "
                "(one engine page must be a whole number of kernel pages)"
            )
        # The feature switches are the profile's reject list: the same fields the platform refuses
        # in get_attn_backend_cls, with the backend's own wording.  Order is the profile's order,
        # which is the order vLLM printed before this list moved out of this file.
        switch_values = {
            "use_mla": use_mla,
            "has_sink": has_sink,
            "use_sparse": use_sparse,
            "use_mm_prefix": use_mm_prefix,
            "use_per_head_quant_scales": use_per_head_quant_scales,
            "use_non_causal": use_non_causal,
            "use_batch_invariant": use_batch_invariant,
            "use_kv_connector": use_kv_connector,
        }
        for feature in profile.backend_features():
            if switch_values.get(feature.field, False):
                reasons.append(feature.backend_reason or feature.reason)
        if attn_type != AttentionType.DECODER:
            reasons.append(
                f"attention type {attn_type!r} is not supported "
                f"(only {AttentionType.DECODER.value!r})"
            )

        # Deduplicate while keeping order: the base may report the same condition with its own
        # wording, and vLLM prints the list verbatim, so repeated lines are just noise.
        return list(dict.fromkeys(reasons))

    # ------------------------------------------------------------------ layer-level check

    @classmethod
    def check_head_layout(
        cls,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
    ) -> None:
        """Reject a layer shape the kernel cannot express; called from the impl's constructor.

        This is where the GQA group is available (``num_heads // num_kv_heads``), which
        ``validate_configuration`` cannot see (see its docstring).  The kernel packs one query
        group per KV head into an M tile of ``padded_rows(group) = max(32, ceil(group/32)*32)``
        rows (``tileinfer/serving/buckets.py:206-219``), so:

        * ``group < 2`` would mean MHA: the kernel's ragged/paged addressing is written for GQA
          and the model does not use it — rejected rather than silently served by a path that was
          never validated;
        * ``num_heads % num_kv_heads != 0`` cannot be expressed as a group at all.
        """
        if num_kv_heads < 1:
            raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")
        if num_heads < num_kv_heads:
            raise ValueError(
                f"num_heads ({num_heads}) < num_kv_heads ({num_kv_heads}); the TileInfer kernel "
                "implements grouped-query attention, not grouped-KV attention"
            )
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) is not a multiple of num_kv_heads ({num_kv_heads}); "
                "the TileInfer kernel needs an integral GQA group"
            )
        group = num_heads // num_kv_heads
        profile = active_profile()
        if group < int(profile.group_min):
            raise ValueError(
                f"GQA group is {group} (num_heads={num_heads}, num_kv_heads={num_kv_heads}); "
                f"the TileInfer paged kernel requires group >= {int(profile.group_min)} "
                "(MHA is not implemented)"
            )
        head_sizes = [int(width) for width in profile.head_size]
        if head_size not in head_sizes:
            raise ValueError(f"head_size {head_size} is not one of {head_sizes}")

    # ------------------------------------------------------------------ output buffers

    @classmethod
    def accept_output_buffer(cls) -> bool:
        """The impl writes into the caller's ``output`` when one is supplied.

        **Not part of the local vLLM 0.23 API.**  ``grep`` over ``reference/vllm-0.23/`` finds no
        ``accept_output_buffer``, and the ``AttentionBackend`` ABC (`v1/attention/backend.py:
        55-349`) never names it.  It is kept because the port plan lists it
        (``docs/plan-vllm-native-backend.md:132``) and because the answer is unconditionally
        true: ``AttentionImpl.forward`` takes ``output: torch.Tensor`` (`v1/attention/backend.py:
        805`) and ``TileInferAttentionImpl.forward`` copies into it.  If a later vLLM version
        queries this hook, the answer is already correct; in 0.23 nothing calls it.
        """
        return True


def _is_real_vllm_base() -> bool:
    """True when the imported ``AttentionBackend`` is vLLM's, not a test stub."""
    return getattr(AttentionBackend, "__module__", "").startswith("vllm.")


def register_with_vllm(backend: AttentionBackendEnum | None = None) -> str:
    """Register this class in vLLM's attention-backend registry; returns the class path.

    The registry API is `v1/attention/backends/registry.py`: ``register_backend`` accepts either a
    decorator (``class_path is None``, lines 252-257) or a direct ``(enum, "module.Class")`` pair
    (lines 259-264), and ``AttentionBackendEnum.CUSTOM`` is the placeholder reserved for
    third-party backends (lines 99-101; ``get_path`` raises until it is registered, lines
    112-117).

    Registration is **not** performed at import time: importing this module must not mutate
    vLLM's process-wide registry, and the platform hook (``get_attn_backend_cls``, W5) can return
    the class path directly.  Call this from ``tileinfer.vllm.plugin.install()`` if the engine
    should be able to select the backend by name (``--attention-backend custom``).
    """
    enum = backend if backend is not None else AttentionBackendEnum.CUSTOM
    path = f"{__name__}.{TileInferAttentionBackend.__qualname__}"
    vllm_register_backend(enum, path)
    return path
