"""``TileInferPlatform`` — the vLLM 0.23 ``Platform`` for Ascend NPUs (TileInfer backend, v0).

The entry-point contract, read out of the local source
-----------------------------------------------------
``vllm/platforms/__init__.py`` builds ``current_platform`` from the ``vllm.platform_plugins``
entry-point group:

* every plugin callable is invoked and a **non-``None`` return value is taken as a class-path
  string** (``:217-222``), appended under the plugin's name (``:222``);
* two active *out-of-tree* plugins are a hard error (``:231-235``), and the winning plugin's
  callable is invoked **a second time** (``:237``) - so :func:`register` must be cheap and have no
  side effects;
* the string is resolved with ``resolve_obj_by_qualname`` and **instantiated with no arguments**
  (``:277-278``), which is why :class:`TileInferPlatform` defines no ``__init__``;
* ``current_platform`` itself is only materialised on first attribute access (``:262-281``),
  precisely so an out-of-tree platform can ``from vllm.platforms import Platform`` (``:265-267``).

Everything else is the ``Platform`` ABC in ``vllm/platforms/interface.py``: **78 methods, 75 of
them public** (two of those are properties).  This module implements **10** and records the other
**65** in :data:`UNIMPLEMENTED_ABC_METHODS` with a reason code, so the remaining work is a number
a test can hold down (``tests/test_platform_contract.py``):

===========================  ======  ==========================================================
reason code                  count   meaning
===========================  ======  ==========================================================
``implemented``                 10   overridden here, with the line that justifies it
``must override``                6   inherited body raises (or a wrong default); NPU cannot use it
``unverified``                   3   the local source does not settle it; inherited for now
``default is fine``             56   the inherited body is correct for a single-card v0
===========================  ======  ==========================================================

:func:`inventory_problems` re-derives the ABC surface from the live base class, so a vLLM point
release that *adds* a method makes the inventory inconsistent and the test fails loudly instead
of the gap sitting in a docstring nobody reads.

What this platform deliberately does **not** do (v0 scope, ``docs/plan-vllm-native-backend.md``
§0): TP/EP/PP/DP and HCCL, quantisation, MLA / sparse / sliding-window / multimodal, speculative
decoding, other models, and the CUDA/ROCm/CPU paths.  Each of those is either rejected loudly at
start-up (:meth:`TileInferPlatform.check_and_update_config`) or left at a default that fails
closed (see the inventory).

Hybrid KV cache (10 full-attention + 30 GDN layers)
---------------------------------------------------
The 10/30 split is vLLM's, not this platform's: the specs come from the model
(``gpu_worker.py:547-548`` delegates to ``gpu_model_runner.py:7440-7470``) and the runner allocates
both kinds (``gpu_model_runner.py:7284-7321``).  The platform contributes exactly three things,
all in this module: the capability bit :meth:`TileInferPlatform.support_hybrid_kv_cache` (the ABC
default ``False``, ``interface.py:945-950``, makes vLLM 0.23 turn the hybrid KV cache manager off
and then reject an attention+mamba spec mix - v0.23.0 ``config/vllm.py:1453-1455`` ->
``v1/core/kv_cache_utils.py:1421-1428``, both outside ``runtime/vllm-api/``), the engine page
(:data:`HYBRID_BLOCK_SIZE`), and the mamba state page size (``mamba_block_size`` /
``mamba_page_size_padded``, :func:`_align_mamba_state_pages`).  No spec class is registered: both
``FullAttentionSpec`` and ``MambaSpec`` are vLLM's own, so
:meth:`register_custom_kv_cache_specs` stays the ABC's ``pass`` (``interface.py:701-706``).  The
state-update dispatch needs no registration either: vLLM's Mamba SSU backend is skipped for GDN
(see the ``HARDWARE:`` paragraph below), and a model that would need it is refused at start-up
rather than served with a Triton kernel.  The items that only hardware can settle are in the
method's ``HARDWARE:`` paragraph and in ``tileinfer.vllm.worker.HARDWARE_CHECKLIST``.

Startup-path assumptions, now read out of the source
----------------------------------------------------
Two things a platform can quietly get wrong are checked here against the vendored vLLM 0.23 files
(``runtime/vllm-api/``), not against folklore.  The worker field was marked ``[unverified]`` in an
earlier revision and the model-architecture path had no call-site check at all; both are settled
now, and the code below implements what the source says.

**Worker selection.**  The mechanism is ``ParallelConfig.worker_cls``, a plain string field with the
sentinel default ``"auto"`` (``vllm__config__parallel.py:248-250``).  It is consumed in one place:
``WorkerWrapperBase.init_worker`` resolves it with ``resolve_obj_by_qualname``
(``vllm__v1__worker__worker_base.py:249-253``) when the executor has already built the wrapper
(``vllm__v1__executor__uniproc_executor.py:48`` - the single-card ``uni`` backend this deployment
uses) - i.e. long after
``VllmConfig.__post_init__`` called ``current_platform.check_and_update_config(self)``
(``vllm__config__vllm.py:1335``).  The in-tree platforms therefore write their class *there*, and
only when the field still holds the sentinel (the CPU platform is the local copy of that branch:
``vllm__platforms__cpu.py:153-154``).  There is **no** ``Platform.get_worker_cls`` in 0.23: the
``Platform`` ABC (``vllm__platforms__interface.py:105-1052``) does not declare one, and there is no
worker plugin group either: ``vllm__plugins____init__.py:14-22`` defines only ``general_plugins``,
``io_processor_plugins``, ``platform_plugins`` and ``stat_logger_plugins`` - none of them selects a
worker.  :func:`_select_worker_class` follows that contract exactly, including refusing a
conflicting explicit value rather than silently ignoring it.

**Model-architecture acceptance.**  An out-of-tree platform is a **veto**, not an allow-list:
``Platform.verify_model_arch`` defaults to ``pass`` ("By default all models are considered
supported", ``vllm__platforms__interface.py:708-718``) and is called immediately before a model
class is imported (``vllm__model_executor__models__registry.py:945`` inside ``_try_load_model_cls``,
``:939-950``).  Acceptance itself is the registry table: ``_VLLM_MODELS`` (``:693-704``) contains
``Qwen3_5ForConditionalGeneration`` / ``Qwen3_5MoeForConditionalGeneration`` mapped to ``qwen3_5``
(``:569-573``), and a bare ``...ForCausalLM`` spelling is normalised onto the registered suffix by
``_normalize_arch`` (``:1134-1158``) through ``model.py:1912-1929`` / ``:1936-1953``.  The model
module does **not** register itself - it only defines the classes
(``vllm__model_executor__models__qwen3_5.py:529``, ``:533``, ``:554``, ``:771``) - so there is no
import-time side effect to depend on.  Conclusion: overriding ``verify_model_arch`` would *break*
every architecture it does not list, while the default ``pass`` accepts the Qwen3.5/3.6 family
through the registry.  This module deliberately keeps the ABC body (see
:data:`UNIMPLEMENTED_ABC_METHODS`).

Attribute choices, each with its evidence
-----------------------------------------
==================================  ==================  ============================================
attribute                           value               why
==================================  ==================  ============================================
``_enum``                           ``PlatformEnum.OOT``  out-of-tree; also what makes
                                                        ``CustomOp`` dispatch reach ``forward_oot``
                                                        (``model_executor/custom_op.py:204-205``)
``device_name`` / ``device_type``   ``"npu"``           ``device_type`` is the ``torch`` attribute
                                                        name used by ``Platform.__getattr__``
                                                        (``interface.py:887-906``) and in
                                                        ``torch.device("npu:0")``
``dispatch_key``                    ``"NPU"``           [unverified] torch_npu registers
                                                        PrivateUse1 under the dispatch key
                                                        ``NPU``; the local tree contains no call
                                                        site, see :data:`UNVERIFIED_NOTES`
``device_control_env_var``          ``ASCEND_RT_VISIBLE_DEVICES``
                                                        the CANN equivalent of
                                                        ``CUDA_VISIBLE_DEVICES``; consumed by
                                                        ``device_id_to_physical_device_id``
                                                        (``interface.py:234-247``), which is *not*
                                                        overridden
``ray_device_key``                  ``""`` (default)    empty means "no ray device support"
                                                        (``interface.py:115-118``); single card,
                                                        no ray
``simple_compile_backend``          ``"eager"``         [unverified] the default is ``inductor``
                                                        (``interface.py:131-136``) and this tree
                                                        has no inductor backend for Ascend;
                                                        ``eager`` is the only defensible answer
                                                        offline
``dist_backend``                    ``""`` (default)    [unverified] should become ``"hccl"``
                                                        when the worker initialises a process
                                                        group; v0 is single-card and the worker
                                                        only passes this value through
==================================  ==================  ============================================

Import policy
-------------
``import tileinfer`` never imports this module (``tileinfer/vllm/__init__.py`` keeps the package
cheap), and the one vLLM import below is guarded: outside a vLLM environment
``import tileinfer.vllm.platform`` raises an :class:`ImportError` that names the missing module
and the expected version, instead of surfacing as an ``AttributeError`` from inside ``vllm``.
``torch_npu`` is imported **inside** the functions that need it (never at module level) so the
module stays importable on a CPU box and so ``tests/test_layer_deps.py``'s rule holds even if
``tileinfer/vllm`` is added to its engine-free list later.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import torch

# ---------------------------------------------------------------------------------------------
# The one vLLM import.  `vllm.platforms` re-exports both names (`:14`, `__all__` at `:298-305`),
# and importing it is safe from inside a platform plugin because the module body has already run
# when the plugin callable is called (that is exactly what `:265-267` documents).
# ---------------------------------------------------------------------------------------------
try:  # pragma: no cover - exercised only outside a vLLM env (the CPU suite injects a fake)
    from vllm.platforms import Platform, PlatformEnum
except ImportError as _exc:  # pragma: no cover - fallback for a `vllm.platforms` that stops
    # re-exporting the interface names (`:14` is the only place they come from).
    try:
        from vllm.platforms.interface import Platform, PlatformEnum
    except ImportError:
        raise ImportError(
            "tileinfer.vllm.platform requires vLLM 0.23's platform API "
            "(vllm.platforms.Platform / PlatformEnum, i.e. vllm/platforms/interface.py); "
            f"importing it failed with {_exc!r}. Install vllm>=0.23,<0.24 - this module is the "
            "vllm.platform_plugins entry point and is not importable without vLLM."
        ) from _exc

from tileinfer.serving.buckets import KERNEL_PAGE_SIZE
from tileinfer.utils import is_npu_available as _is_npu_available
from tileinfer.vllm.profiles import (
    ModelProfile,
    UnsupportedFeature,
    active_profile,
    default_profile,
    profile_for,
    set_active_profile,
)

__all__ = [
    "TileInferPlatform",
    "register",
    # identity / paths
    "DEVICE_TYPE",
    "DEVICE_CONTROL_ENV_VAR",
    "DISPATCH_KEY",
    "ATTN_BACKEND_CLASS_PATH",
    "WORKER_CLASS_PATH",
    "WORKER_CLS_ATTR",
    "WORKER_CLS_AUTO",
    # inventory
    "AbcMethod",
    "IMPLEMENTED_ABC_METHODS",
    "UNIMPLEMENTED_ABC_METHODS",
    "EXTRA_HELPERS",
    "REASON_IMPLEMENTED",
    "REASON_MUST_OVERRIDE",
    "REASON_UNVERIFIED",
    "REASON_DEFAULT_FINE",
    "REASONS",
    "DOCUMENTED_ABC_SURFACE",
    "DOCUMENTED_UNIMPLEMENTED_CAP",
    "inventory_problems",
    "unimplemented_report",
    # selector policy (shared with the tests; both derive from the active profile)
    "SELECTOR_REJECT_SWITCHES",
    "SELECTOR_PLAIN_SWITCHES",
    "SELECTOR_REJECT_REASONS",
    "SELECTOR_ATTN_TYPE",
    "HYBRID_BLOCK_SIZE",
    "SUPPORTED_DTYPES",
    # profile seam (public so a test can assert the enforcement, not the constants)
    "BOOT_PROFILE",
    "profile_for_config",
    "validate_block_size",
    "align_mamba_state_pages",
]

logger = logging.getLogger("tileinfer.vllm.platform")

# =============================================================================================
# Device identity
# =============================================================================================

DEVICE_TYPE = "npu"
DEVICE_NAME = "npu"

#: [unverified] torch_npu registers ``PrivateUse1`` as ``"npu"``; the *dispatch key* spelling is
#: ``"NPU"``.  Nothing in ``runtime/vllm-api/`` reads ``dispatch_key``, so the value cannot be
#: settled from the local source - it is recorded here (and in ``UNVERIFIED_NOTES``) rather than
#: guessed silently.  If the device pass shows a torch warning about an unknown dispatch key,
#: this is the line to fix.
DISPATCH_KEY = "NPU"

#: The CANN device-visibility variable.  The ABC's ``device_id_to_physical_device_id``
#: (``interface.py:234-247``) splits exactly this variable, and it is the documented ray/CANN
#: analogue of ``CUDA_VISIBLE_DEVICES`` (``interface.py:120-124``).
DEVICE_CONTROL_ENV_VAR = "ASCEND_RT_VISIBLE_DEVICES"

#: ``Platform.get_attn_backend_cls`` returns a **class path string** (``interface.py:259-267``),
#: which is what ``AttentionBackendEnum.get_class`` resolves with ``resolve_obj_by_qualname``
#: (``backends/registry.py:122-132``).  Kept as a literal so this module does not import the
#: attention backend (which needs vLLM's v1 attention API) just to name it; the contract test
#: asserts the literal equals ``f"{backend.__name__}.{TileInferAttentionBackend.__qualname__}"``.
ATTN_BACKEND_CLASS_PATH = "tileinfer.vllm.attn.backend.TileInferAttentionBackend"

#: The worker class vLLM must instantiate.  ``worker.py`` implements all four hooks
#: (hardware-unverified); the path is what ``check_and_update_config`` writes into
#: ``ParallelConfig.worker_cls``.  That field is real, not an assumption:
#: ``runtime/vllm-api/vllm__config__parallel.py:248`` declares ``worker_cls: str = "auto"``, and its
#: only consumer resolves the string when the worker wrapper is initialised
#: (``vllm__v1__worker__worker_base.py:249-253``, ``resolve_obj_by_qualname``); a non-string value
#: is refused there outright (``:254-259``).  The in-tree platforms select their own class in
#: ``check_and_update_config`` by replacing the ``"auto"`` sentinel
#: (``vllm__platforms__cpu.py:153-154``; CUDA/XPU/ROCm use the same branch), and vLLM 0.23 has
#: **no** ``Platform.get_worker_cls`` hook at all (gone from ``vllm__platforms__interface.py`` and
#: not replaced by an entry point).  ``vllm.v1.worker.gpu_worker.Worker`` hard-codes
#: ``torch.device(f"cuda:...")`` in ``init_device`` (``gpu_worker.py:285``) and would fail on the
#: first NPU, which is why this platform must win the field.
WORKER_CLASS_PATH = "tileinfer.vllm.worker.NPUWorker"
WORKER_MODULE = "tileinfer.vllm.worker"
WORKER_CLS_ATTR = "worker_cls"
#: The sentinel vLLM's own platforms test before writing their class: ``if
#: parallel_config.worker_cls == "auto"`` (``cpu.py:153``).  ``ParallelConfig`` defaults to it
#: (``parallel.py:248``), so "auto" is the normal case, not a special case.
WORKER_CLS_AUTO = "auto"

#: The boot profile: the model facts this module used to hold as constants now come from here.
#: ``BOOT_PROFILE`` is only what the backwards-compatible module-level aliases below describe; the
#: platform resolves the *served* model's profile in :func:`profile_for_config` at config time.
BOOT_PROFILE: ModelProfile = default_profile()

#: The engine page must be a whole number of kernel pages (``serving/buckets.py:52``), and for a
#: hybrid model it must also leave room for one mamba state page (``interface.py:663-674``:
#: ``block_size`` is raised until ``block_size * attention_page_1_token >= mamba_page_size``).  The
#: profile's ``block_size_policy`` carries both numbers; this alias is the boot profile's hybrid
#: alignment, kept because older probes/tests and the docstrings read it as data.
HYBRID_BLOCK_SIZE = int(BOOT_PROFILE.block_size_policy.required_multiple(hybrid=True))

#: kv-cache dtype spellings v0 accepts.  ``"auto"`` is vLLM's default and resolves to the model
#: dtype (``interface.py:557-560``); everything else is a different, unvalidated code path
#: (``CacheConfig.cache_dtype`` - note the field is *not* called ``kv_cache_dtype``).
SUPPORTED_CACHE_DTYPES = ("auto",)

#: The compute dtypes the TileInfer paged kernels are compiled for.  Same set the attention
#: backend declares (``tileinfer/vllm/attn/backend.py:72-74``), and the first entry is what
#: ``--dtype auto`` resolves to (``interface.py:152-158``: "The first dtype will be used as the
#: default dtype fallback").
SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)

#: Quantisation method spellings that mean "no quantisation".  [unverified] the exact string the
#: engine passes when nothing is quantised is not in the local subset; the override below is
#: therefore tolerant (empty/``None`` counted as "none") and *strict about everything else*.
NO_QUANT_SPELLINGS = ("", "none")

#: ``AttentionSelectorConfig``'s boolean switches, read out of the real source
#: (``runtime/vllm-api/vllm__v1__attention__selector.py:21-34``):
#:
#:     head_size, dtype, kv_cache_dtype, block_size,          # :22-25  (the layer's geometry)
#:     use_mla=False, has_sink=False, use_sparse=False,       # :26-28
#:     use_mm_prefix=False, use_per_head_quant_scales=False,  # :29-30
#:     attn_type=AttentionType.DECODER,                      # :31     (not a bool - see below)
#:     use_non_causal=False, use_batch_invariant=False,       # :32-33
#:     use_kv_connector=False                                 # :34
#:
#: There is **no** ``use_compress`` field (this module used to read one, from an older in-repo
#: vLLM-Ascend shim): "compressed"/MLA-sparse attention is spelled ``use_sparse`` here, and a
#: selector that exposes ``use_compress`` is not a vLLM 0.23 ``AttentionSelectorConfig``.
#:
#: The two tuples split those eight booleans by what a ``True`` means for this platform:
#:
#: * :data:`SELECTOR_REJECT_SWITCHES` - the **attention-shape** switches.  Every one of them is
#:   declared unsupported by :class:`tileinfer.vllm.attn.backend.TileInferAttentionBackend`
#:   (``is_mla``/``is_sparse``/``supports_sink``/``supports_mm_prefix``/
#:   ``supports_per_head_quant_scales``/``supports_non_causal`` all return False,
#:   ``attn/backend.py:125-147``), so a ``True`` means "this layer's attention is not the paged
#:   GQA/MHA the TileInfer kernels implement".  These names are also what identifies a real
#:   ``AttentionSelectorConfig``: a selector that exposes none of them is refused (see
#:   :func:`_selector_switches`).
#: * :data:`SELECTOR_PLAIN_SWITCHES` - the **engine-mode** switches.  They do not change the layer's
#:   maths; they change how the engine drives the backend, and the same backend predicates reject
#:   them (``supports_batch_invariance`` False at ``attn/backend.py:149-151``;
#:   ``supports_kv_connector`` False at ``:153-158`` - our ``forward`` writes the fused pool
#:   directly, and v0 has no KV-transfer/disaggregation path, plan §0).  Absence of one of these on
#:   a duck-typed object is tolerated (warned), because only the switch set above identifies the
#:   config.
#:
#: Both tuples are *reject when True*: the policy is one rule for every field (see
#: :func:`_selector_switches`).  Verified against vLLM 0.23's ``selector.py`` and cross-checked
#: against the ABC's own transcription of the same fields, ``AttentionBackend.validate_configuration``
#: (``runtime/vllm-api/vllm__v1__attention__backend.py:275-341``), whose parameter list is exactly
#: these eight booleans plus ``head_size``/``dtype``/``kv_cache_dtype``/``block_size``/
#: ``device_capability``/``attn_type`` - so no field of the real config is unhandled.
#: The tuples and the reason table are **derived from the boot profile's reject list**; the profile
#: (``tileinfer/vllm/profiles/qwen3_6.py``) is the one place a new model family edits.  They remain
#: module data because the selector policy is shared with the tests and printed in messages.
SELECTOR_REJECT_SWITCHES: Tuple[str, ...] = tuple(
    feature.field for feature in BOOT_PROFILE.selector_features()
)
SELECTOR_PLAIN_SWITCHES: Tuple[str, ...] = tuple(
    feature.field for feature in BOOT_PROFILE.plain_selector_features()
)
SELECTOR_REJECT_REASONS: Dict[str, str] = {
    feature.field: feature.reason
    for feature in BOOT_PROFILE.unsupported_features
    if feature.kind in ("selector", "plain_selector", "attn_type")
}

#: the only ``attn_type`` this platform serves (``AttentionType.DECODER``, selector.py:31).  The
#: value is the enum's *str* form, so ``attn_type`` compares equal whether vLLM passes the
#: ``AttentionType`` member or a bare string (``AttentionType(str, Enum)``, backend.py:32).
SELECTOR_ATTN_TYPE: str = "decoder"


def profile_for_config(vllm_config: Any) -> ModelProfile:
    """Resolve the model profile for a ``VllmConfig`` (the public seam the tests call).

    The architecture names are the authoritative key: ``ModelConfig.architectures`` is what vLLM
    itself loads (``model_config.py:834-835`` reads ``model_arch_config.architectures``).  The
    config hints let ``profile_for`` narrow a shared architecture family; they never turn an
    unknown architecture into a guess.

    A hand-built config (the CPU contract tests, a launcher probe) may carry no architecture at
    all.  That is not an error: the process-wide active profile answers - the same rule the engine
    follows, because there is exactly one model per engine process in v0.
    """
    model_config = getattr(vllm_config, "model_config", None)
    archs = getattr(model_config, "architectures", None) if model_config is not None else None
    hints: Dict[str, Any] = {}
    if model_config is not None:
        if hasattr(model_config, "is_hybrid"):
            hints["is_hybrid"] = bool(getattr(model_config, "is_hybrid"))
        get_head_size = getattr(model_config, "get_head_size", None)
        if callable(get_head_size):
            try:
                hints["head_size"] = int(get_head_size())
            except (TypeError, ValueError):  # a hand-built config with an odd method
                pass
    if archs:
        return profile_for(archs, hints)
    return active_profile()

#: Open questions, plus the ones the local source has since settled (kept as an audit trail: an
#: entry that starts with "settled" is no longer an assumption, and it says which file settled it).
#: Each one is a line a hardware engineer can check in one `grep`/`python -c`; the code below fails
#: closed wherever an answer is required.
UNVERIFIED_NOTES: Dict[str, str] = {
    "dispatch_key": (
        "'NPU' is the torch_npu PrivateUse1 dispatch-key spelling; no local file reads the "
        "attribute, so it cannot be confirmed offline."
    ),
    "simple_compile_backend": (
        "the ABC default is 'inductor' (interface.py:131-136); no Ascend inductor backend exists "
        "in this tree, so 'eager' is chosen - confirm no vLLM code path requires torch.compile "
        "codegen for standalone functions."
    ),
    "selector_fields": (
        "settled - AttentionSelectorConfig's 13 fields are transcribed from "
        "runtime/vllm-api/vllm__v1__attention__selector.py:21-34 into SELECTOR_REJECT_SWITCHES / "
        "SELECTOR_PLAIN_SWITCHES / SELECTOR_ATTN_TYPE. The old use_compress name was not a vLLM "
        "0.23 field (it came from this repo's earlier vLLM-Ascend shim); there is no field of the "
        "real config that this policy ignores."
    ),
    "worker_cls_attr": (
        "settled - ParallelConfig.worker_cls is a real vLLM 0.23 field with the sentinel default "
        "'auto' (runtime/vllm-api/vllm__config__parallel.py:248-250), resolved by "
        "WorkerWrapperBase.init_worker with resolve_obj_by_qualname "
        "(vllm__v1__worker__worker_base.py:249-253) after VllmConfig.__post_init__ has already "
        "called check_and_update_config (vllm__config__vllm.py:1335). The in-tree platforms write "
        "their class in that hook, only when the field is still 'auto' "
        "(vllm__platforms__cpu.py:153-154 is the local copy of that branch); vLLM 0.23 has no "
        "Platform.get_worker_cls hook and no worker entry-point group, so the field is the "
        "mechanism, not one option among several."
    ),
    "dist_backend": (
        "the HCCL process-group backend name ('hccl') is not in the local subset; v0 is "
        "single-card, so the default '' is left alone while the worker passes it straight to "
        "init_worker_distributed_environment (gpu_worker.py:1162-1204)."
    ),
    "graph_pool_handle": (
        "Platform.get_global_graph_pool (interface.py:908-915) calls graph_pool_handle(), which "
        "resolves through __getattr__ (interface.py:887-906) to torch.npu.graph_pool_handle; "
        "whether torch_npu exposes it is not verifiable offline (see tileinfer/vllm/graphs.py)."
    ),
}

# =============================================================================================
# ABC inventory: implemented vs left to the base class
# =============================================================================================

REASON_IMPLEMENTED = "implemented"
REASON_MUST_OVERRIDE = "must override"
REASON_UNVERIFIED = "unverified"
REASON_DEFAULT_FINE = "default is fine"

#: The complete reason vocabulary.  A test asserts every entry uses one of these, so the
#: inventory cannot grow a free-text reason that hides a "we don't know".
REASONS: Tuple[str, ...] = (
    REASON_IMPLEMENTED,
    REASON_MUST_OVERRIDE,
    REASON_UNVERIFIED,
    REASON_DEFAULT_FINE,
)

#: How many public members (78 methods - 3 private = 75; two of the 75 are properties) the ABC in
#: ``runtime/vllm-api/vllm__platforms__interface.py`` declares.  Derived by AST over that file;
#: ``tests/test_platform_contract.py`` re-derives it and fails if the local source no longer
#: matches, i.e. if the runtime subset is refreshed to a vLLM whose ABC has more methods.
DOCUMENTED_ABC_SURFACE = 75

#: How many of them this platform leaves to the base class.  This is the *only* number that may
#: shrink: adding a method to the ABC (or discovering one) must grow it, and the test fails until
#: somebody consciously edits this cap - which is the point of a ratchet.
DOCUMENTED_UNIMPLEMENTED_CAP = 62


class AbcMethod(NamedTuple):
    """One inventory entry: the reason code plus the sentence that justifies it."""

    reason: str
    note: str


# The ABC's plain *attributes* (``dispatch_key``, ``device_control_env_var``,
# ``simple_compile_backend``, ``dist_backend``, ``ray_device_key``, ``supported_quantization``,
# ``ray_noset_device_env_vars``, ``additional_env_vars``, ``_global_graph_pool``) are not part of
# this inventory, because the surface counted here is the ABC's public methods and properties
# (75 of them). They are decided in the class body below and in the module-docstring table. The
# only one with an open question is ``dist_backend`` - see UNVERIFIED_NOTES.

#: ABC members this platform overrides, and why that is justified from the local source.
IMPLEMENTED_ABC_METHODS: Dict[str, AbcMethod] = {
    "supported_dtypes": AbcMethod(
        REASON_IMPLEMENTED,
        "narrowed to (bf16, fp16): interface.py:152-158 defines the property (the first entry is "
        "the 'auto'-dtype fallback) and attn/backend.py:74 says the paged kernels are bf16/fp16 "
        "only; a wider list here would let the engine pick a dtype the backend then rejects.",
    ),
    "get_attn_backend_cls": AbcMethod(
        REASON_IMPLEMENTED,
        "the ABC body returns '' (interface.py:259-267), which resolves to no class at all "
        "(backends/registry.py:103-120 raises on an empty path), so every OOT platform must "
        "override it. See the reject policy in the method.",
    ),
    "check_and_update_config": AbcMethod(
        REASON_IMPLEMENTED,
        "ABC body is 'pass' (interface.py:457-468); this is the documented place to update or "
        "reject the config before weights are loaded ('it can raise an exception if the "
        "configuration is not compatible', :461-464).",
    ),
    "get_device_name": AbcMethod(
        REASON_IMPLEMENTED,
        "ABC raises (interface.py:377-380); torch_npu answers it directly, and "
        "tileinfer.utils.device_name already reads it the same way (utils.py:82-91).",
    ),
    "get_device_total_memory": AbcMethod(
        REASON_IMPLEMENTED,
        "ABC raises (interface.py:387-390); the 'bytes' contract is explicit and torch_npu "
        "answers it, and the vLLM memory snapshot needs the same number "
        "(v1/worker/gpu_worker.py:313 constructs MemorySnapshot(device=...)).",
    ),
    "num_compute_units": AbcMethod(
        REASON_IMPLEMENTED,
        "ABC raises (interface.py:1027-1035) and the runner asks for it during construction "
        "(utils/platform_utils.py:64) - the first hardware run stopped there. Answers the AI-core "
        "count from torch.npu.get_device_properties (Ascend950PR: multi_processor_count=56, "
        "cube_core_num=28, vector_core_num=56), with those names as fallbacks.",
    ),
    "get_current_memory_usage": AbcMethod(
        REASON_IMPLEMENTED,
        "ABC raises (interface.py:764-771), and the device proved it is on the start-up path: "
        "vLLM's DeviceMemoryProfiler (utils/mem_utils.py:55) wraps weight loading, so a raising "
        "body stopped the engine in load_model on Ascend950PR. Answers torch.npu.memory_allocated "
        "(the allocator counter the profiler's before/after delta is about).",
    ),
    "manual_seed_all": AbcMethod(
        REASON_IMPLEMENTED,
        "ABC raises (interface.py:422-425) and vLLM's seeding path calls it (the first hardware "
        "run logged the NotImplementedError and fell back to torch.manual_seed). Seeds both the "
        "CPU generator and torch.npu.manual_seed_all.",
    ),
    "set_device": AbcMethod(
        REASON_IMPLEMENTED,
        "ABC raises (interface.py:415-420); the worker's device init has to call it before any "
        "tensor exists on the device (gpu_worker.py:285-286 does the torch.accelerator "
        "equivalent for CUDA).",
    ),
    "is_pin_memory_available": AbcMethod(
        REASON_IMPLEMENTED,
        "the inherited body says True except on WSL (interface.py:751-762). Pinned host memory "
        "on Ascend is not established by anything in the local tree, and a wrong True corrupts "
        "or aborts a transfer while a False only costs a copy - so this answers False until the "
        "device pass measures torch_npu pin_memory.",
    ),
    "check_if_supports_dtype": AbcMethod(
        REASON_IMPLEMENTED,
        "ABC raises (interface.py:938-943) but the local Worker calls it during init_device "
        "(gpu_worker.py:288), so leaving it raising would abort every NPU start-up. It now "
        "enforces exactly `supported_dtypes`.",
    ),
    "support_hybrid_kv_cache": AbcMethod(
        REASON_IMPLEMENTED,
        "the ABC returns False (interface.py:945-950), and False is not a performance switch: "
        "vLLM 0.23 turns the hybrid KV cache manager off (v0.23.0 config/vllm.py:1453-1455, not "
        "vendored locally) and then rejects the attention+mamba spec mix this model always "
        "produces (v0.23.0 v1/core/kv_cache_utils.py:1642-1643 -> :1421-1428). The runner needs "
        "nothing else from the platform - the specs come from the model (gpu_worker.py:547-548 "
        "-> gpu_model_runner.py:7440-7470) - so the bit plus the page sizes of "
        "_validate_block_size / _align_mamba_state_pages are the whole platform contribution.",
    ),
    "verify_quantization": AbcMethod(
        REASON_IMPLEMENTED,
        "the ABC only rejects when `supported_quantization` is non-empty (interface.py:720-728), "
        "so with the default empty list *every* quantisation would be accepted. v0 has no "
        "quantised path (docs/plan-vllm-native-backend.md §0), so this rejects instead.",
    ),
}

#: ABC members left to the base class.  Grouped by reason code; the note says what would have to
#: change for an override to be needed, and names the line the decision rests on.
UNIMPLEMENTED_ABC_METHODS: Dict[str, AbcMethod] = {
    # --- "must override": the inherited body raises, or its default is actively wrong ---------
    "get_device_uuid": AbcMethod(
        REASON_MUST_OVERRIDE,
        "ABC raises (interface.py:382-385). Needed by ray / NIC-PCIE mapping / KV-transfer "
        "topologies, none of which exist on one card; [unverified] whether the engine asks for "
        "it at start-up - if it does, this is the first crash and torch_npu answers with "
        "get_device_properties().uuid.",
    ),
    "get_punica_wrapper": AbcMethod(
        REASON_MUST_OVERRIDE,
        "ABC raises (interface.py:773-778). LoRA is out of v0 scope (plan §0), so the raise is "
        "the correct signal until a LoRA path exists.",
    ),
    "stateless_init_device_torch_dist_pg": AbcMethod(
        REASON_MUST_OVERRIDE,
        "ABC raises (interface.py:924-936). Only needed for stateless torch.distributed "
        "initialisation (ray / external launcher); v0 is one card with no process group.",
    ),
    "get_global_graph_pool": AbcMethod(
        REASON_MUST_OVERRIDE,
        "the inherited body (interface.py:908-915) calls graph_pool_handle(), which "
        "__getattr__ (:887-906) resolves to torch.npu.graph_pool_handle - [unverified] for "
        "torch_npu. W8 (tileinfer/vllm/graphs.py) verifies it; if it is missing, override here.",
    ),
    # --- "unverified": inherited for now, because the local source does not settle it ---------
    "get_device_capability": AbcMethod(
        REASON_UNVERIFIED,
        "returns None (interface.py:306-312) and every has/is_device_capability then answers "
        "False (:314-375). That is the fail-closed direction, but the NPU capability encoding "
        "(if any) is not in the local tree; revisit with the attention backend's "
        "device_capability parameter (v1/attention/backend.py:287-289).",
    ),
    "import_ir_kernels": AbcMethod(
        REASON_UNVERIFIED,
        "the ABC imports vllm.kernels (interface.py:225-232) and its own docstring says OOT "
        "platforms should override it once they have IR kernels; v0 has no IR op layer, so the "
        "default import is harmless - to be revisited when tileinfer/vllm/ops/ lands.",
    ),
    # --- "default is fine": the inherited body is exactly what a single-card v0 wants --------
    "pass_key": AbcMethod(REASON_DEFAULT_FINE, "inductor PassManager key (interface.py:147-150); "
                          "only used with VLLM_COMPILE, which is not the NPU path."),
    "is_cuda": AbcMethod(REASON_DEFAULT_FINE, "False for _enum=OOT (interface.py:160-161)."),
    "is_rocm": AbcMethod(REASON_DEFAULT_FINE, "False for _enum=OOT (interface.py:163-164)."),
    "is_tpu": AbcMethod(REASON_DEFAULT_FINE, "False for _enum=OOT (interface.py:166-167)."),
    "is_xpu": AbcMethod(REASON_DEFAULT_FINE, "False for _enum=OOT (interface.py:169-170)."),
    "is_cpu": AbcMethod(REASON_DEFAULT_FINE, "False for _enum=OOT (interface.py:172-173)."),
    "uses_host_device_handling": AbcMethod(REASON_DEFAULT_FINE,
                                           "True only for TPU (interface.py:175-177)."),
    "is_zen_cpu": AbcMethod(REASON_DEFAULT_FINE, "False (interface.py:179-180)."),
    "is_out_of_tree": AbcMethod(REASON_DEFAULT_FINE,
                                "True, derived from _enum (interface.py:182-183) - and that is "
                                "what routes CustomOp.dispatch_forward to forward_oot "
                                "(model_executor/custom_op.py:204-205)."),
    "is_unspecified": AbcMethod(REASON_DEFAULT_FINE, "False (interface.py:185-186)."),
    "get_max_output_tokens": AbcMethod(REASON_DEFAULT_FINE, "sys.maxsize (interface.py:188-189)."),
    "is_cuda_alike": AbcMethod(REASON_DEFAULT_FINE,
                              "False for OOT (interface.py:191-193); gpu_worker.py:202-217 uses "
                              "it to skip the CUDA memory-pool context, which is the behaviour we "
                              "want left alone."),
    "is_sleep_mode_available": AbcMethod(REASON_DEFAULT_FINE,
                                         "False outside CUDA/ROCm/XPU (interface.py:195-200); sleep "
                                         "mode is also rejected in check_and_update_config."),
    "is_cumem_allocator_available": AbcMethod(REASON_DEFAULT_FINE,
                                              "probes vllm.device_allocator.cumem and returns "
                                              "False on ImportError (interface.py:202-208)."),
    "get_pass_manager_cls": AbcMethod(REASON_DEFAULT_FINE,
                                     "PostGradPassManager (interface.py:210-216); compile-only."),
    "get_compile_backend": AbcMethod(REASON_DEFAULT_FINE,
                                     "returns simple_compile_backend (interface.py:218-223), which "
                                     "this class sets to 'eager'."),
    "device_id_to_physical_device_id": AbcMethod(REASON_DEFAULT_FINE,
                                                 "splits device_control_env_var and indexes it "
                                                 "(interface.py:234-247); with "
                                                 "ASCEND_RT_VISIBLE_DEVICES set that is exactly the "
                                                 "CANN mapping, so no override."),
    "import_kernels": AbcMethod(REASON_DEFAULT_FINE,
                                "imports vllm._C and only *logs* a failure (interface.py:249-257)."),
    "get_supported_vit_attn_backends": AbcMethod(REASON_DEFAULT_FINE,
                                                 "[TORCH_SDPA] (interface.py:269-273); v0 has no "
                                                 "multimodal tower."),
    "get_vit_attn_backend": AbcMethod(REASON_DEFAULT_FINE,
                                      "asserts the caller's choice against the list above and "
                                      "defaults to TORCH_SDPA (interface.py:275-304)."),
    "has_device_capability": AbcMethod(REASON_DEFAULT_FINE,
                                       "False because get_device_capability is None "
                                       "(interface.py:314-336)."),
    "is_device_capability": AbcMethod(REASON_DEFAULT_FINE,
                                      "False for the same reason (interface.py:338-360)."),
    "is_device_capability_family": AbcMethod(REASON_DEFAULT_FINE,
                                             "False for the same reason (interface.py:362-375)."),
    "get_all_gpu_pci_bus_ids": AbcMethod(REASON_DEFAULT_FINE,
                                         "raises with a message that names the unsupported "
                                         "feature (interface.py:392-403) - the documented answer "
                                         "for a platform without PCI bus ids."),
    "inference_mode": AbcMethod(REASON_DEFAULT_FINE,
                                "torch.inference_mode (interface.py:405-413); torch_npu supports it "
                                "[unverified only in the sense that no local file proves it]."),
    "pre_register_and_update": AbcMethod(REASON_DEFAULT_FINE,
                                         "'pass' (interface.py:427-441); nothing to register before "
                                         "the CLI is parsed in v0."),
    "apply_config_platform_defaults": AbcMethod(REASON_DEFAULT_FINE,
                                                "'pass' (interface.py:443-455); the defaults live in "
                                                "check_and_update_config."),
    "update_block_size_for_backend": AbcMethod(REASON_DEFAULT_FINE,
                                               "the inherited body is the rule this model needs: "
                                               "preferred block size from the backend, then "
                                               "mamba-page alignment (interface.py:490-527, "
                                               "530-699). It is *not* overridden, but it is also "
                                               "not called anywhere in the local subset (its only "
                                               "reference is the definition at :490), and its "
                                               "entry point resolves the backend from layers that "
                                               "do not exist yet when check_and_update_config "
                                               "runs (_find_non_ssm_backend, :470-488). So "
                                               "_validate_block_size / _align_mamba_state_pages "
                                               "apply the same two rules at config time; if vLLM "
                                               "runs the hook later the inputs are identical and "
                                               "the call is a no-op."),
    "register_custom_kv_cache_specs": AbcMethod(REASON_DEFAULT_FINE,
                                                "'pass' (interface.py:701-706). Not needed for the "
                                                "hybrid Qwen3.6 run: both spec classes are vLLM's "
                                                "own (FullAttentionSpec from the attention layer, "
                                                "MambaSpec from MambaBase.get_kv_cache_spec), so "
                                                "the registry check at kv_cache_utils.py:2007 "
                                                "sees only registered types. Revisit only if a "
                                                "TileInfer spec class is shipped."),
    "verify_model_arch": AbcMethod(REASON_DEFAULT_FINE,
                                    "'pass' means 'all models supported' (interface.py:708-718), and "
                                    "that is exactly what an out-of-tree platform needs here. It is "
                                    "called immediately before a model class is imported: "
                                    "registry.py:945 (in _try_load_model_cls, :939-950) invokes "
                                    "current_platform.verify_model_arch(model_arch). Acceptance is "
                                    "the registry table, not this hook: _VLLM_MODELS (:693-704) "
                                    "registers Qwen3_5ForConditionalGeneration and "
                                    "Qwen3_5MoeForConditionalGeneration against the qwen3_5 module "
                                    "(:569-573), and a ...ForCausalLM spelling is normalised onto "
                                    "that suffix by _normalize_arch (:1134-1158) using "
                                    "model.py:1912-1929. qwen3_5.py never self-registers (:529, "
                                    ":533, :554, :771 define the classes only), so there is no "
                                    "import-time registration to rely on. An override here would "
                                    "be an allow-list and would reject every arch it omits - do "
                                    "not add one for the Qwen3.5/3.6 family."),
    "get_cpu_architecture": AbcMethod(REASON_DEFAULT_FINE,
                                       "platform.machine() (interface.py:730-749)."),
    "get_infinity_values": AbcMethod(REASON_DEFAULT_FINE, "(-inf, inf) (interface.py:780-785)."),
    "can_update_inplace": AbcMethod(REASON_DEFAULT_FINE, "True (interface.py:787-792)."),
    "get_lora_vocab_padding_size": AbcMethod(REASON_DEFAULT_FINE,
                                              "256 (interface.py:794-799); unused without LoRA."),
    "get_device_communicator_cls": AbcMethod(REASON_DEFAULT_FINE,
                                             "the base communicator (interface.py:801-806); no "
                                             "collectives on one card."),
    "is_integrated_gpu": AbcMethod(REASON_DEFAULT_FINE, "False (interface.py:808-818)."),
    "supports_mx": AbcMethod(REASON_DEFAULT_FINE,
                             "False (interface.py:820-825); v0 has no quantisation."),
    "supports_fp8": AbcMethod(REASON_DEFAULT_FINE,
                              "False (interface.py:827-832); plan §0 keeps the run bf16."),
    "is_fp8_fnuz": AbcMethod(REASON_DEFAULT_FINE, "False (interface.py:834-846)."),
    "fp8_dtype": AbcMethod(REASON_DEFAULT_FINE,
                           "e4m3fn (interface.py:848-855); unreachable while supports_fp8 is "
                           "False."),
    "use_all_gather": AbcMethod(REASON_DEFAULT_FINE, "True (interface.py:857-862)."),
    "use_custom_allreduce": AbcMethod(REASON_DEFAULT_FINE,
                                       "False (interface.py:864-869); custom allreduce is dropped "
                                       "with TP in plan §0."),
    "opaque_attention_op": AbcMethod(REASON_DEFAULT_FINE,
                                      "False (interface.py:871-877): our attention is a normal "
                                      "AttentionImpl, not one opaque custom op."),
    "validate_request": AbcMethod(REASON_DEFAULT_FINE,
                                   "no-op hook (interface.py:879-885); nothing to reject per "
                                   "request in v0."),
    "get_static_graph_wrapper_cls": AbcMethod(REASON_DEFAULT_FINE,
                                               "AbstractStaticGraphWrapper "
                                               "(interface.py:917-922); unreachable while "
                                               "support_static_graph_mode is False."),
    "support_static_graph_mode": AbcMethod(REASON_DEFAULT_FINE,
                                            "False (interface.py:952-957); v0 has no static-graph "
                                            "path."),
    "support_deep_gemm": AbcMethod(REASON_DEFAULT_FINE, "False (interface.py:959-964)."),
    "use_custom_op_collectives": AbcMethod(REASON_DEFAULT_FINE, "False (interface.py:966-973)."),
    "use_sync_weight_loader": AbcMethod(REASON_DEFAULT_FINE, "False (interface.py:975-980)."),
    "make_synced_weight_loader": AbcMethod(REASON_DEFAULT_FINE,
                                            "returns the loader unchanged while the flag above is "
                                            "False (interface.py:982-996)."),
    "get_nixl_supported_devices": AbcMethod(REASON_DEFAULT_FINE,
                                             "{} (interface.py:998-1004); no NIXL / disaggregation."),
    "get_nixl_memory_type": AbcMethod(REASON_DEFAULT_FINE, "None (interface.py:1006-1011)."),
    "check_max_model_len": AbcMethod(REASON_DEFAULT_FINE,
                                      "identity (interface.py:1013-1018); the 2048-token page is "
                                      "the only platform constraint and it is not a length cap."),
    "set_additional_forward_context": AbcMethod(REASON_DEFAULT_FINE, "{} (interface.py:1020-1025)."),
    "get_default_ir_op_priority": AbcMethod(REASON_DEFAULT_FINE,
                                             "native (interface.py:1037-1045); no IR kernels in "
                                             "v0."),
    "is_arch_support_pdl": AbcMethod(REASON_DEFAULT_FINE, "False (interface.py:1047-1052)."),
}

#: Helpers this class provides that are **not** ABC methods.  They exist because
#: ``Platform.__getattr__`` (``interface.py:887-906``) would otherwise forward them to
#: ``torch.npu``; naming them explicitly means a missing torch_npu gives *our* error message
#: instead of the ABC's "Current platform npu does not have 'x' attribute" warning plus None.
EXTRA_HELPERS: Dict[str, AbcMethod] = {
    "empty_cache": AbcMethod(
        REASON_IMPLEMENTED,
        "not in the ABC: it is reached through __getattr__ (interface.py:887-906) and the local "
        "Worker calls the torch.accelerator spelling (gpu_worker.py:310).",
    ),
    "synchronize": AbcMethod(
        REASON_IMPLEMENTED,
        "not in the ABC: same __getattr__ route; tileinfer.utils.synchronize already prefers "
        "torch.npu.synchronize (utils.py:99-111).",
    ),
}


def _abc_member_names(base: Optional[type] = None) -> List[str]:
    """Public members of the ABC (callables *and* properties), sorted.

    Properties are included because two of the 75 public members are properties
    (``pass_key``, ``supported_dtypes``, ``interface.py:147-158``) and a property is not
    ``callable`` on the class.  Private members are excluded: ``_find_non_ssm_backend`` /
    ``_align_hybrid_block_size`` (``:470-700``) are upstream helpers, and ``__getattr__``
    (``:887``) is the mechanism described above, not a hook.
    """
    import inspect

    base = base if base is not None else Platform
    names: List[str] = []
    for name in dir(base):
        if name.startswith("_"):
            continue
        try:
            raw = inspect.getattr_static(base, name)
        except AttributeError:  # pragma: no cover - dir() and getattr cannot disagree in practice
            continue
        if isinstance(raw, property) or callable(raw):
            names.append(name)
    return sorted(names)


def inventory_problems(base: Optional[type] = None) -> List[str]:
    """Every inconsistency between the inventory and the live ABC (empty list == consistent).

    Checks, in order:

    1. **coverage** - every public ABC member is in exactly one of :data:`IMPLEMENTED_ABC_METHODS`
       / :data:`UNIMPLEMENTED_ABC_METHODS`.  A vLLM release that adds a method fails here, which
       is the ratchet: the new method cannot sit in neither list.
    2. **no double entry** - a name in both lists is a contradiction.
    3. **no stale praise** - every ``implemented`` name is really overridden *below* ``Platform``
       (``name in vars(TileInferPlatform)``), so the inventory cannot claim work that is not
       there.
    4. **no stale TODO** - every unimplemented name is really *not* overridden; implementing a
       method without moving it out of the TODO list fails.
    5. **vocabulary** - every reason is one of :data:`REASONS`.
    6. **no name outside the ABC** - an entry that the ABC does not declare is either a typo or a
       helper that belongs in :data:`EXTRA_HELPERS`.
    """
    problems: List[str] = []
    abc = set(_abc_member_names(base))
    implemented = set(IMPLEMENTED_ABC_METHODS)
    unimplemented = set(UNIMPLEMENTED_ABC_METHODS)
    overridden = set(vars(TileInferPlatform))

    for name in sorted(abc - implemented - unimplemented):
        problems.append(f"{name}: in the ABC but in neither inventory list")
    for name in sorted(implemented & unimplemented):
        problems.append(f"{name}: listed as both implemented and unimplemented")
    for name in sorted(implemented - abc):
        problems.append(f"{name}: claimed implemented but the ABC does not declare it")
    for name in sorted(unimplemented - abc):
        problems.append(f"{name}: listed as unimplemented but the ABC does not declare it")
    for name in sorted(implemented):
        if name not in overridden:
            problems.append(f"{name}: claimed implemented but not overridden on the class")
    for name in sorted(unimplemented & overridden):
        problems.append(f"{name}: overridden on the class but still listed as unimplemented")
    for table, label in (
        (IMPLEMENTED_ABC_METHODS, "IMPLEMENTED_ABC_METHODS"),
        (UNIMPLEMENTED_ABC_METHODS, "UNIMPLEMENTED_ABC_METHODS"),
        (EXTRA_HELPERS, "EXTRA_HELPERS"),
    ):
        for name, entry in table.items():
            if entry.reason not in REASONS:
                problems.append(f"{label}[{name}]: unknown reason {entry.reason!r}")
    for name in sorted(set(EXTRA_HELPERS) & abc):
        problems.append(f"{name}: in EXTRA_HELPERS but the ABC does declare it")
    return problems


def unimplemented_report() -> Dict[str, Dict[str, int]]:
    """The inventory as plain counts, for probes / ``tileinfer doctor``."""
    report: Dict[str, Dict[str, int]] = {"by_reason": {}, "totals": {}}
    for entry in UNIMPLEMENTED_ABC_METHODS.values():
        report["by_reason"][entry.reason] = report["by_reason"].get(entry.reason, 0) + 1
    report["totals"] = {
        "abc_surface": DOCUMENTED_ABC_SURFACE,
        "implemented": len(IMPLEMENTED_ABC_METHODS),
        "unimplemented": len(UNIMPLEMENTED_ABC_METHODS),
        "unimplemented_cap": DOCUMENTED_UNIMPLEMENTED_CAP,
        "extra_helpers": len(EXTRA_HELPERS),
    }
    return report


# =============================================================================================
# Device access helpers (torch_npu imported inside the functions, never at module level)
# =============================================================================================


def _npu(what: str) -> Any:
    """Import ``torch_npu`` and return the ``torch.npu`` module, or raise with the reason."""
    try:
        import torch_npu  # noqa: F401  (the import is what registers the backend)
    except Exception as exc:  # pragma: no cover - needs a CPU-only box to exercise
        raise RuntimeError(
            f"tileinfer.vllm.platform.{what} requires torch_npu, which is not importable "
            f"({exc!r}). The TileInfer platform is Ascend-only; without torch_npu no device "
            "operation can be answered."
        ) from exc
    npu = getattr(torch, "npu", None)
    if npu is None:  # pragma: no cover - torch_npu always registers torch.npu
        raise RuntimeError(
            f"tileinfer.vllm.platform.{what}: torch_npu imported but torch.npu is missing; the "
            "installed torch_npu does not match this torch build."
        )
    return npu


def _npu_available() -> bool:
    """True when a usable NPU stack with at least one visible device exists.

    Mirrors the CUDA plugin's "a device must exist, not just a driver" test
    (``vllm/platforms/__init__.py:74-79``).  Never raises: this runs inside vLLM's plugin loop,
    which swallows exceptions anyway (``:217-224``) - returning ``None`` is the honest answer.
    """
    if not _is_npu_available():
        return False
    try:
        return int(torch.npu.device_count()) > 0
    except Exception:  # pragma: no cover - a broken torch_npu
        return False


# =============================================================================================
# The platform
# =============================================================================================


class TileInferPlatform(Platform):
    """Ascend NPU platform served by the TileInfer attention backend and worker."""

    # --- identity (see the module docstring table for the evidence) ---------------------------
    _enum = PlatformEnum.OOT
    device_name = DEVICE_NAME
    device_type = DEVICE_TYPE
    dispatch_key = DISPATCH_KEY
    device_control_env_var = DEVICE_CONTROL_ENV_VAR

    #: [unverified] see UNVERIFIED_NOTES["simple_compile_backend"].
    simple_compile_backend = "eager"

    #: No quantisation in v0; `verify_quantization` below is the enforcement, this is only what
    #: the engine reads to decide what it may offer.
    supported_quantization: List[str] = []

    # No __init__: vLLM instantiates the platform with no arguments
    # (``vllm/platforms/__init__.py:277-278``).

    # ---------------------------------------------------------------- overridden ABC members --

    @property
    def supported_dtypes(self) -> List[torch.dtype]:
        """bf16/fp16 only: the first entry is the "auto" fallback (``interface.py:152-158``).

        The paged kernels are bf16/fp16 (``tileinfer/vllm/attn/backend.py:72-74``), and an fp32
        entry here would let the engine resolve ``--dtype auto`` to a dtype the attention backend
        then rejects - after the weights are loaded.
        """
        return list(SUPPORTED_DTYPES)

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: Any,
        attn_selector_config: Any,
        num_heads: Optional[int] = None,
    ) -> str:
        """Return the TileInfer backend's class path for plain MHA/GQA layers only.

        ``num_heads`` is accepted (``interface.py:259-265``) and unused: the GQA group is
        ``num_heads // num_kv_heads`` and neither ``num_kv_heads`` nor ``head_size`` is a
        parameter of this hook, so the real check lives in the backend's
        ``check_head_layout``/``validate_configuration`` (``attn/backend.py:202-352``).

        Raises instead of guessing in three cases:

        * the selector asks for an attention feature these kernels do not implement - MLA, sparse,
          sink, multimodal prefix, per-head quant scales, non-causal, batch invariance or a KV
          connector; the message names the field and the backend predicate behind the decision
          (:data:`SELECTOR_REJECT_SWITCHES` / :data:`SELECTOR_PLAIN_SWITCHES`, real field names from
          ``v1/attention/selector.py:21-34``);
        * the selector's ``attn_type`` is not decoder self-attention (:data:`SELECTOR_ATTN_TYPE`);
        * the selector exposes none of the switches this policy depends on, or the user pinned a
          different backend by name - v0 serves exactly one, so either is refused rather than
          resolved to a vLLM backend.
        """
        profile = active_profile()
        switches = _selector_switches(attn_selector_config, profile)
        rejected = sorted(name for name, on in switches.items() if on)
        if rejected:
            reasons = "; ".join(
                f"{name} ({_selector_reason(profile, name)})" for name in rejected
            )
            raise ValueError(
                "TileInfer's attention backend serves plain, causal, decoder MHA/GQA only; this "
                f"layer's AttentionSelectorConfig asks for: {reasons}. Those are the real vLLM "
                "0.23 selector fields (runtime/vllm-api/vllm__v1__attention__selector.py:21-34) "
                "and every one of them is a feature the TileInfer paged kernel does not implement "
                "(docs/plan-vllm-native-backend.md §0). Silently falling back to a vLLM backend is "
                "not possible here: the platform must name one class. Reject the request, or "
                "extend the backend and move the field out of the model profile's "
                "unsupported_features list (SELECTOR_REJECT_SWITCHES / SELECTOR_PLAIN_SWITCHES "
                "are derived from it) in tileinfer/vllm/profiles/."
            )
        _reject_foreign_backend_selection(selected_backend)
        logger.info(
            "TileInfer attention backend selected (%s)", ATTN_BACKEND_CLASS_PATH
        )
        return ATTN_BACKEND_CLASS_PATH

    @classmethod
    def check_and_update_config(cls, vllm_config: Any) -> None:
        """Validate the config for an NPU run, or raise before any weight is loaded.

        The ABC documents this as the place that "can raise an exception if the configuration is
        not compatible with the current platform, or ... update the configuration to make it
        compatible" (``interface.py:457-468``).  Order matters: every rejection is evaluated
        *before* the first update, so a rejected run never leaves a half-modified config behind.

        Rejected (each with the line the field name comes from):

        * speculative decoding - ``vllm_config.speculative_config``
          (``model_executor/models/qwen3_5.py:696-697``, ``v1/attention/backend.py:568-576``);
        * ``cache_config.cache_dtype != "auto"`` - the field is ``cache_dtype``, not
          ``kv_cache_dtype`` (``interface.py:557-560``);
        * sleep mode - ``model_config.enable_sleep_mode`` (``gpu_worker.py:204-213``);
        * MLA - ``model_config.use_mla`` (``interface.py:565``).

        Updated:

        * ``cache_config.block_size`` - aligned up to :data:`HYBRID_BLOCK_SIZE` for a hybrid model
          when the user did not pin it, or rejected when they did (``user_specified_block_size``,
          ``interface.py:511``; ``is_hybrid``, ``interface.py:526``).  When the platform performs
          that alignment it also marks the value as specified, because it *is* the platform's
          decision: vLLM's later ``update_block_size_for_backend`` phase 1 (``interface.py:511``)
          skips a specified size and would otherwise replace 2048 with the kernel minimum (128),
          undoing the mamba-page alignment below.
        * ``cache_config.mamba_block_size`` / ``cache_config.mamba_page_size_padded`` - for a
          hybrid model, see :func:`_align_mamba_state_pages`, which pads one mamba state page to
          exactly one attention page (``interface.py:676-699``).
        * ``parallel_config.worker_cls`` - :data:`WORKER_CLASS_PATH`, but **only when the field is
          still the ``"auto"`` sentinel** that ``ParallelConfig`` defaults to
          (``runtime/vllm-api/vllm__config__parallel.py:248``); that is what every in-tree platform
          does (``vllm__platforms__cpu.py:153-154`` is the local copy).  An
          explicit, conflicting value is refused instead of overwritten (:func:`_select_worker_class`)
          because it would otherwise build vLLM's CUDA worker (``gpu_worker.py:285``) and fail on
          the first NPU - the exact silent-unused failure this hook exists to prevent.
        * ``model_config.enforce_eager`` - forced True while
          ``tileinfer.vllm.graphs.capture_path_ready()`` is False (W8 not landed), with a warning.
        """
        from tileinfer.vllm import graphs

        # Resolve the served model *first*, and make it the process-wide profile: the attention
        # backend and the worker ask for the model facts without carrying a VllmConfig.
        profile = profile_for_config(vllm_config)
        set_active_profile(profile)
        is_hybrid = bool(getattr(vllm_config.model_config, "is_hybrid", False))
        if is_hybrid and profile.hybrid is None:
            raise ValueError(
                f"model_config.is_hybrid is True but model profile {profile.name!r} declares no "
                "hybrid state-page geometry. Fix the profile (HybridKind) or the model config - "
                "the platform cannot size a mamba state page without it."
            )

        _reject_unsupported_features(vllm_config, profile)
        block_size = validate_block_size(vllm_config, profile)
        if is_hybrid:
            align_mamba_state_pages(vllm_config, block_size, profile)
        worker_path = _select_worker_class(vllm_config)
        _disable_cuda_only_compilation_passes(vllm_config)
        _enable_aclgraph_if_ready(vllm_config, graphs)
        eager = _force_eager_until_capture_lands(vllm_config, graphs)

        logger.info(
            "TileInfer platform: profile=%s device=%s block_size=%d hybrid=%s worker=%s "
            "enforce_eager=%s",
            profile.name,
            DEVICE_TYPE,
            block_size,
            is_hybrid,
            worker_path,
            eager,
        )

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """``torch.npu.get_device_name`` (the ABC raises: ``interface.py:377-380``)."""
        npu = _npu("get_device_name")
        try:
            return str(npu.get_device_name(device_id))
        except AttributeError as exc:  # pragma: no cover - needs a torch_npu variant
            raise RuntimeError(
                "torch_npu does not expose torch.npu.get_device_name(); "
                "torch.npu.get_device_properties(device_id).name is the fallback to implement "
                "here (see UNVERIFIED_NOTES in tileinfer/vllm/platform.py)."
            ) from exc

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """Total device memory in bytes (the ABC raises: ``interface.py:387-390``).

        Two torch_npu spellings exist and neither is in the local tree, so both are tried in
        order and a clear error is raised if neither works - returning a plausible number from a
        failed probe would corrupt the memory profiling that sizes the KV pool
        (``gpu_worker.py:312-318``).
        """
        npu = _npu("get_device_total_memory")
        properties = getattr(npu, "get_device_properties", None)
        if callable(properties):
            try:
                props = properties(device_id)
            except (TypeError, RuntimeError):  # a signature variant: fall through to mem_get_info
                props = None
            total = getattr(props, "total_memory", None)
            if total:
                return int(total)
        mem_get_info = getattr(npu, "mem_get_info", None)
        if callable(mem_get_info):
            try:
                info = mem_get_info(device_id)
            except (TypeError, RuntimeError):
                info = None
            if isinstance(info, tuple) and len(info) == 2:
                return int(info[1])
        raise RuntimeError(
            "torch_npu answers neither get_device_properties(device_id).total_memory nor "
            "mem_get_info(device_id)[1]; tileinfer.vllm.platform.get_device_total_memory "
            "cannot report the device size, and a guess here would mis-size the KV pool."
        )

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """Select the current device (the ABC raises: ``interface.py:415-420``)."""
        _npu("set_device").set_device(device)

    @classmethod
    def num_compute_units(cls, device_id: int = 0) -> int:
        """AI-core count, which is what vLLM wants ``num_compute_units`` for.

        Required at start-up on hardware: the model runner asks
        ``current_platform.num_compute_units()`` (``utils/platform_utils.py:64``) and the ABC raises
        (``interface.py:1027-1035``), which stopped the engine during construction.  Ascend950PR
        reports ``multi_processor_count=56``, ``cube_core_num=28`` and ``vector_core_num=56``
        (measured with torch_npu 2.10.0.post4); the first attribute is the closest CUDA-style
        "SM count", and the rest are fallbacks so a rename does not break start-up.
        """
        npu = _npu("num_compute_units")
        try:
            props = npu.get_device_properties(device_id)
        except Exception:  # noqa: BLE001 - a signature variant
            return 1
        for attr in ("multi_processor_count", "cube_core_num", "vector_core_num", "num_aicore"):
            value = getattr(props, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        return 1

    @classmethod
    def get_current_memory_usage(cls, device: Any = None) -> float:
        """Bytes currently allocated on the device by torch (``interface.py:764-771``).

        Verified necessary on hardware: vLLM's ``DeviceMemoryProfiler`` (``utils/mem_utils.py:55``)
        calls this around weight loading, so leaving the ABC's ``raise`` stopped the engine during
        ``load_model`` with a bare ``NotImplementedError`` (observed on Ascend950PR with
        torch_npu 2.10.0.post4).  The ABC documents "memory usage in bytes", which is
        ``torch.npu.memory_allocated`` - the torch allocator's own counter, not the driver's HBM
        figure, so the profiler's before/after delta measures what vLLM itself allocated.
        """
        npu = _npu("get_current_memory_usage")
        index = getattr(device, "index", None)
        if index is None and isinstance(device, int):
            index = device
        try:
            return float(npu.memory_allocated(index))
        except TypeError:  # a torch_npu variant that takes no index
            return float(npu.memory_allocated())

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        """Seed every device RNG (the ABC raises: ``interface.py:422-425``)."""
        import torch as _torch

        _torch.manual_seed(seed)
        _npu("manual_seed_all").manual_seed_all(seed)

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        """Answer False: pinned host memory on Ascend is not established by the local source.

        The inherited body returns True except on WSL (``interface.py:751-762``).  A wrong True
        costs correctness (a transfer from pageable memory is silently slower; from a *wrongly*
        pinned buffer it aborts), a wrong False costs one copy, so a platform that cannot prove
        the capability answers False.  Flip this - and only this line - once the device pass
        shows ``tensor.pin_memory()`` works under torch_npu and the async-output path is faster
        with it.
        """
        return False

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype) -> None:
        """Reject a compute dtype the kernels cannot run (ABC raises: ``interface.py:938-943``).

        Called from the worker's device init (``gpu_worker.py:288``), so it must answer rather
        than raise-by-default.  It enforces exactly :meth:`supported_dtypes`, i.e. the same set
        the attention backend declares (``attn/backend.py:72-74``).
        """
        supported = list(SUPPORTED_DTYPES)
        if dtype not in supported:
            raise ValueError(
                f"dtype {dtype} is not supported on the TileInfer Ascend platform "
                f"(supported: {[str(d) for d in supported]}). The paged kernels are bf16/fp16 "
                "only; pass --dtype bfloat16 or --dtype float16."
            )

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        """True for a hybrid profile: vLLM owns the attention/mamba split; this platform stops blocking it.

        What the source settles, in order:

        * the ABC default is ``False`` (``interface.py:945-950``) and that is not a performance
          switch.  vLLM 0.23 sets ``scheduler_config.disable_hybrid_kv_cache_manager = True`` when
          this answers False (v0.23.0 ``config/vllm.py:1453-1455`` - not vendored locally), and
          ``get_kv_cache_groups`` then tries to collapse every layer onto one spec type (v0.23.0
          ``v1/core/kv_cache_utils.py:1642-1643``) and raises for an attention+mamba mix
          (``:1421-1428``).  Qwen3.6 always produces that mix, so the default fails at start-up
          rather than costing performance.
        * the split is the model's job: ``Worker.get_kv_cache_spec`` delegates to
          ``GPUModelRunner.get_kv_cache_spec`` (``gpu_worker.py:547-548`` ->
          ``gpu_model_runner.py:7440-7470``), which asks each layer for its spec - the full
          attention layers answer ``FullAttentionSpec`` and the GDN layers answer ``MambaSpec``
          through ``MambaBase.get_kv_cache_spec``.  The platform never enumerates layers, so the
          10+30 split is nowhere hard-coded here.
        * the runner side is device-neutral already: ``initialize_kv_cache``
          (``gpu_model_runner.py:7284-7321``) calls ``initialize_mamba_ssu_backend`` (``:7301``),
          ``prepare_kernel_block_sizes`` (``:7309-7311``) and ``initialize_kv_cache_tensors``
          (``:7319``, defined ``:7201``), whose ``MambaSpec`` branch builds the state tensors
          (``:7135-7158``) and whose hybrid branch (``:7162-7163``) is a **no-op** for this
          backend: see below.
        * no spec hook is needed: both spec classes are vLLM's own, so
          ``register_custom_kv_cache_specs`` stays the ABC's ``pass`` (``interface.py:701-706``).

        What the platform must add is only the page geometry, and it is in this module:
        :func:`_validate_block_size` fixes the engine page (:data:`HYBRID_BLOCK_SIZE`) and
        :func:`_align_mamba_state_pages` pads one mamba state page to exactly that attention page.

        HARDWARE: the GDN state-update port and the KV-zeroer outcome.  Everything the attention
        pool needs is settled by the declarations in ``tileinfer/vllm/attn/backend.py``.

        1. ``initialize_mamba_ssu_backend`` (``gpu_model_runner.py:7301``) is **not** a device gap
           for this model.  Its body returns unless a group is ``MambaSpec(mamba_type=MAMBA1|
           MAMBA2)`` (v0.23.0 ``.../mamba/ops/ssu_dispatch.py:202-208``); every GDN layer reports
           ``GDN_ATTN`` (``.../mamba/gdn/base.py:49-51``), so no SSU backend is instantiated and
           Qwen3.6's decode state update is ``fused_recurrent_gated_delta_rule_packed_decode``
           (``qwen_gdn_linear_attn.py:1684``), which the already-built torch port is bound to.
           The SSU backend itself is **not extensible** in 0.23: ``MambaConfig.backend`` accepts
           only ``triton``/``flashinfer`` (``vllm/config/mamba.py:25-29``) and
           ``ssu_dispatch._BACKEND_REGISTRY`` is a private dict with no registration function
           (``ssu_dispatch.py:185-188``) - unlike the attention registry's public
           ``register_backend(..., is_mamba=True)``.  A model that *does* need MAMBA1/MAMBA2 is
           therefore refused at start-up by
           :func:`tileinfer.vllm.worker.require_supported_mamba_state_path`, which names the
           missing NPU backend and the two options; it never reaches a Triton import mid-step.
           The minimal upstream change to serve such a model would be one ``MambaBackendEnum``
           member plus a class in ``_BACKEND_REGISTRY`` (or a public
           ``register_mamba_ssu_backend``), then ``--mamba-backend <name>`` - not something this
           repository can do from a config value.
           The remaining hardware question is the GDN port's state write, not the SSU choice - see
           :meth:`tileinfer.vllm.worker.NPUWorker.initialize_from_config` HARDWARE item 3.
        2. the attention pool is **block-first and K/V-outermost**, so it is a contiguous
           allocation only in that order: ``get_kv_cache_shape`` returns ``(num_blocks, 2,
           block_size, heads, dim)`` and ``get_kv_cache_stride_order`` returns ``(1, 0, 2, 3, 4)``
           (``attn/backend.py:167-231``).  ``get_kv_cache_block_dim`` therefore answers 0 and
           ``_update_hybrid_attention_mamba_layout`` (``gpu_model_runner.py:7167-7199``) skips the
           pool (``:7189-7191``) instead of re-striding it to a physically per-block
           ``[blocks][2][...]`` layout; ``kv_cache[:, 0]`` / ``kv_cache[:, 1]`` are the two
           contiguous halves ``fused_halves`` and ``pool_sub_view`` need
           (``serving/buckets.py:150-305``).  The first ``forward`` is expected to reach the
           TileInfer write/plan path - the old contiguous-pool raise belonged to the
           pre-block-first layout and is gone.  What is still a device-run observation is the
           ``KVBlockZeroer``: ``needs_kv_cache_zeroing`` is ``has_mamba_layers`` (vLLM
           ``v1/kv_cache_interface.py:873-879``), but the zeroer sizes a block from
           ``stride(block_dim)``, which is half a page under this order, so
           ``TileInferModelRunner._init_kv_zero_meta`` declines it with a logged reason
           (``worker.py``).  The device run must show no garbage on a recycled attention page -
           that is this skip's residual cost, not a boot-time expectation.

           The 128-token sub-page view itself is legal: 2048 % 128 == 0, so the runner keeps the
           whole 2048-token page as the kernel block (v0.23.0 ``v1/worker/utils.py:
           select_common_block_size`` case 1) and the split into 16 sub-pages stays inside the
           TileInfer kernels.

        The bit follows the **active model profile**: a profile with ``hybrid=None`` is
        attention-only, and turning the hybrid KV manager on for it would only make vLLM group a
        single spec type through the hybrid code path.  For Qwen3.6 ``hybrid`` is set, so the
        answer stays True.
        """
        return active_profile().hybrid is not None

    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        """Reject every quantisation method (the ABC's default would accept all of them).

        ``interface.py:720-728`` only rejects when ``supported_quantization`` is non-empty, and
        this platform's list is empty (no quantised path in v0, plan §0) - so without this
        override ``--quantization fp8`` would be accepted at start-up and fail inside a kernel.
        The "no quantisation" spellings are not in the local subset, so empty/``None`` are
        tolerated (see :data:`NO_QUANT_SPELLINGS`).
        """
        name = "" if quant is None else str(quant).strip().lower()
        if name in NO_QUANT_SPELLINGS:
            return
        raise ValueError(
            f"quantisation {quant!r} is not supported on the TileInfer Ascend platform: v0 runs "
            "bf16/fp16 only (docs/plan-vllm-native-backend.md §0 - W8A8/W4A8/FP8-KV are out of "
            "scope). Remove --quantization, or use unquantised weights."
        )

    # ------------------------------------------------------- non-ABC helpers (see EXTRA_HELPERS) --

    @staticmethod
    def empty_cache() -> None:
        """``torch.npu.empty_cache`` (reached through ``__getattr__`` otherwise: ``:887-906``)."""
        _npu("empty_cache").empty_cache()

    @staticmethod
    def synchronize(device: Optional[torch.device] = None) -> None:
        """``torch.npu.synchronize`` (same ``__getattr__`` route as above)."""
        npu = _npu("synchronize")
        if device is None:
            npu.synchronize()
        else:
            npu.synchronize(device)


# =============================================================================================
# Config checks (module functions so the tests can call them without a VllmConfig)
# =============================================================================================


def _require_field(obj: Any, owner: str, name: str, why: str) -> Any:
    """Read ``obj.name`` or raise, because a missing field must never pass a rejection check."""
    if not hasattr(obj, name):
        raise RuntimeError(
            f"{owner} has no attribute {name!r}, so this check cannot be evaluated: {why}. "
            "Refusing to continue rather than assuming the feature is absent "
            "(the local vLLM subset in runtime/vllm-api/ may predate this field - see "
            "UNVERIFIED_NOTES in tileinfer/vllm/platform.py)."
        )
    return getattr(obj, name)


def _read_feature_value(vllm_config: Any, feature: UnsupportedFeature) -> Any:
    """Read a config feature's value through its declared path, requiring every step to exist."""
    value: Any = vllm_config
    for name, owner in zip(feature.path, feature.owners):
        value = _require_field(value, owner, name, feature.why)
    return value


def _feature_rejected(feature: UnsupportedFeature, value: Any) -> bool:
    """Apply one config feature's predicate; an unknown predicate is a bug, not a "no"."""
    if feature.predicate == "is_not_none":
        return value is not None
    if feature.predicate == "is_truthy":
        return bool(value)
    if feature.predicate == "not_in_cache_dtypes":
        return str(value) not in SUPPORTED_CACHE_DTYPES
    raise RuntimeError(
        f"model profile feature {feature.field!r} uses unknown predicate "
        f"{feature.predicate!r}; refusing to silently accept the feature"
    )


def _reject_unsupported_features(vllm_config: Any, profile: Optional[ModelProfile] = None) -> None:
    """Reject every config-level feature the model profile does not serve, with its evidence.

    The list (speculative decoding, a non-``auto`` kv-cache dtype, sleep mode, MLA) is data in the
    profile; this function is only the reader.  A missing config field still raises through
    :func:`_require_field`, so a field the local vLLM source names can never silently pass.
    """
    profile = profile or active_profile()
    for feature in profile.config_features():
        value = _read_feature_value(vllm_config, feature)
        if _feature_rejected(feature, value):
            raise ValueError(
                feature.message.format(value=value, supported=list(SUPPORTED_CACHE_DTYPES))
            )


def validate_block_size(vllm_config: Any, profile: Optional[ModelProfile] = None) -> int:
    """Enforce the profile's kernel-page / mamba-page alignment (public seam for the tests).

    The rules themselves are the model profile's ``block_size_policy``: an engine page must be a
    whole number of kernel pages, and a hybrid model's page must be a multiple of its state page.
    """
    profile = profile or active_profile()
    policy = profile.block_size_policy
    kernel_page = int(policy.kernel_page_size)
    cache_config = _require_field(vllm_config, "VllmConfig", "cache_config", "block size")
    model_config = _require_field(vllm_config, "VllmConfig", "model_config", "hybrid flag")
    block_size = int(
        _require_field(
            cache_config,
            "CacheConfig",
            "block_size",
            "the engine page size must be validated against the kernel page "
            "(interface.py:511-522 reads and writes it)",
        )
    )
    user_specified = bool(
        _require_field(
            cache_config,
            "CacheConfig",
            "user_specified_block_size",
            "without it we cannot tell 'align it for me' from 'the operator asked for this value' "
            "(interface.py:511 is the local use)",
        )
    )
    is_hybrid = bool(
        _require_field(
            model_config,
            "ModelConfig",
            "is_hybrid",
            "the mamba state-page alignment only applies to a hybrid model (interface.py:526)",
        )
    )

    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    if is_hybrid:
        align = int(policy.required_multiple(hybrid=True))
        if block_size % align != 0:
            if user_specified:
                raise ValueError(
                    f"block_size={block_size} is not a multiple of {align}: this "
                    "model is hybrid (attention + mamba/GDN layers) and the engine page must "
                    "match the mamba state page (docs/plan-vllm-native-backend.md §1: 'engine "
                    f"block_size {align}'), otherwise one KV page and one mamba state page are "
                    "not interchangeable and vLLM's block manager pads/copies behind our back. "
                    f"Drop --block-size, or pass a multiple of {align}."
                )
            aligned = policy.align_up(block_size, hybrid=True)
            logger.info(
                "TileInfer: aligning block_size %d -> %d for hybrid attention/mamba pages",
                block_size,
                aligned,
            )
            cache_config.block_size = aligned
            block_size = aligned
        # For a hybrid model the page is the platform's decision, not a default.  Mark it as
        # specified so vLLM's later `update_block_size_for_backend` phase 1 (``interface.py:511``)
        # does not lower it to the kernel minimum and leave the mamba page padded for a page
        # the engine no longer uses.
        cache_config.user_specified_block_size = True

    if block_size % kernel_page != 0:
        raise ValueError(
            f"block_size={block_size} is not a multiple of the TileInfer kernel page "
            f"({kernel_page}, tileinfer/serving/buckets.py:52): the pool is re-viewed as "
            f"{kernel_page}-token kernel pages without a copy "
            "(tileinfer/serving/buckets.py:pool_sub_view, "
            "tileinfer/vllm/attn/backend.py:185-194), which requires the engine page to be a "
            "whole number of kernel pages."
        )
    return block_size


def align_mamba_state_pages(
    vllm_config: Any, block_size: int, profile: Optional[ModelProfile] = None
) -> Optional[int]:
    """Pad one mamba state page to exactly one attention page (``interface.py:676-699``).

    The geometry is the model profile's ``hybrid.page_size_padded``; this function only reads the
    config, applies the two ``mamba_cache_mode`` layouts, and writes the two fields vLLM's
    ``MambaSpec`` is built from.

    The two fields written here are what the model's ``MambaSpec`` is built from, and the model -
    not the platform - builds it: ``MambaBase.get_kv_cache_spec`` reads
    ``cache_config.mamba_block_size`` and ``cache_config.mamba_page_size_padded`` (v0.23.0
    ``model_executor/layers/mamba/abstract.py:44-60``; the local ``interface.py:632-637`` reads
    the block size and ``:676-690`` is where the upstream helper writes both).  Equal page sizes
    are the only configuration that survives grouping: when the state page is smaller,
    ``unify_kv_cache_spec_page_size`` rescales the layer's ``block_size`` to reach the maximum
    (v0.23.0 ``v1/core/kv_cache_utils.py:1029-1066``), but a padded ``MambaSpec`` page does not
    depend on ``block_size`` at all (v0.23.0 ``v1/kv_cache_interface.py:616-625``), so the resize
    cannot reconcile the two and trips its own assertion.  That is why the state page is padded
    rather than left smaller.

    Why this is done here rather than by the ABC hook ``update_block_size_for_backend``: that
    hook resolves the attention backend from the *constructed* layers
    (``_find_non_ssm_backend``, ``interface.py:470-488``), and this runs while the config is
    still being built (``check_and_update_config`` is invoked as soon as ``VllmConfig`` exists),
    so the hook would return early.  The rules are the same two the hook applies
    (``interface.py:511-527``): fix the engine page, then align the mamba page to it.

    ``mamba_cache_mode`` is read because the two modes size the state page differently:
    ``"align"`` keeps one state per engine page (``interface.py:676-677``), ``"none"`` keeps one
    per request and vLLM has already set ``mamba_block_size`` to ``max_model_len`` (v0.23.0
    ``model_executor/models/config.py:387-394``, not vendored locally).  ``"all"`` is refused
    because the Qwen3.5/3.6 model rejects it itself
    (``model_executor/models/qwen3_5.py:459-463``).

    Returns the padded page size, or ``None`` when the config does not carry the fields - a
    hand-built config such as the CPU contract tests' ``SimpleNamespace``.  Every real vLLM 0.23
    ``CacheConfig`` has them, so that is not a production branch.
    """
    profile = profile or active_profile()
    hybrid = profile.hybrid
    if hybrid is None:
        logger.debug("TileInfer: model profile %r declares no hybrid state page", profile.name)
        return None
    cache_config = getattr(vllm_config, "cache_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    if cache_config is None or model_config is None:
        return None
    mode = getattr(cache_config, "mamba_cache_mode", None)
    if mode is None:
        logger.debug(
            "TileInfer: cache_config has no mamba_cache_mode (hand-built config, not a vLLM 0.23 "
            "CacheConfig), so the mamba state page is left as-is"
        )
        return None
    if mode not in ("align", "none"):
        raise ValueError(
            f"mamba_cache_mode={mode!r} is not supported on the TileInfer Ascend platform. "
            f"'all' caches a state per token, which {profile.name} refuses itself "
            "(model_executor/models/qwen3_5.py:459-463), and the runner's state-update paths are "
            "written for the 'align' layout (gpu_model_runner.py:1005-1008, :1515, :4159). Use "
            "--mamba-cache-mode=align."
        )
    if mode == "align":
        if getattr(cache_config, "mamba_block_size", None) != block_size:
            # The engine page and the state block are the same unit in align mode
            # (interface.py:676-677).
            cache_config.mamba_block_size = block_size
    elif getattr(cache_config, "mamba_block_size", None) is None:
        max_model_len = getattr(model_config, "max_model_len", None)
        if isinstance(max_model_len, int) and max_model_len > 0:
            # "none" mode: one state per request, addressed in max_model_len-sized units
            # (v0.23.0 model_executor/models/config.py:393-394).
            cache_config.mamba_block_size = max_model_len

    get_num_kv_heads = getattr(model_config, "get_num_kv_heads", None)
    get_head_size = getattr(model_config, "get_head_size", None)
    dtype = getattr(model_config, "dtype", None)
    if not callable(get_num_kv_heads) or not callable(get_head_size) or not isinstance(
        dtype, torch.dtype
    ):
        logger.debug(
            "TileInfer: ModelConfig cannot report (num_kv_heads, head_size, dtype); the mamba "
            "page size is left to vLLM (hand-built config, not a vLLM 0.23 ModelConfig)"
        )
        return None
    # One attention page per token is exactly FullAttentionSpec(block_size=1).page_size_bytes:
    # 2 (K and V) * num_kv_heads * head_size * dtype size (the profile's geometry function).
    num_kv_heads = int(get_num_kv_heads(getattr(vllm_config, "parallel_config", None)))
    head_size = int(get_head_size())
    element_size = torch.tensor([], dtype=dtype).element_size()
    page_bytes_per_token = hybrid.page_bytes_per_token(num_kv_heads, head_size, element_size)
    padded = hybrid.padded_page_size(block_size, num_kv_heads, head_size, element_size)
    cache_config.mamba_page_size_padded = padded
    logger.info(
        "TileInfer hybrid KV: block_size=%d mamba_block_size=%s mamba_page_size_padded=%d "
        "(attention page = 2*%d kv_heads*%d head_dim = %d B/token)",
        block_size,
        getattr(cache_config, "mamba_block_size", None),
        padded,
        num_kv_heads,
        head_size,
        page_bytes_per_token,
    )
    return padded


# Backwards-compatible private spellings (the module docstrings and older probes cite them).
_validate_block_size = validate_block_size
_align_mamba_state_pages = align_mamba_state_pages


def _select_worker_class(vllm_config: Any) -> str:
    """Select :data:`WORKER_CLASS_PATH` through ``parallel_config.worker_cls``, or refuse to start.

    How the source settles it (``runtime/vllm-api/``):

    * the field exists and its default is the sentinel ``"auto"``
      (``vllm__config__parallel.py:248-250``);
    * it is consumed exactly once, when the worker wrapper is initialised:
      ``WorkerWrapperBase.init_worker`` does ``resolve_obj_by_qualname(parallel_config.worker_cls)``
      (``vllm__v1__worker__worker_base.py:249-253``).  The wrapper is constructed by the executor
      (``vllm__v1__executor__uniproc_executor.py:48``, the single-card ``uni`` backend), i.e. after
      ``VllmConfig.__post_init__`` has run ``current_platform.check_and_update_config(self)``
      (``vllm__config__vllm.py:1335``) - so a write here is seen;
    * a non-string value is refused by the wrapper (``worker_base.py:254-259``), which is why this
      function only ever writes the qualified-name string;
    * the in-tree platforms replace the sentinel in this same hook
      (``vllm__platforms__cpu.py:153-154`` is the local copy).  There is no
      ``Platform.get_worker_cls`` in the 0.23 ABC (``vllm__platforms__interface.py:105-1052``) and
      no worker plugin group (``vllm__plugins____init__.py:14-22`` lists general / io_processor /
      platform / stat_logger, none of which selects a worker), so the config field is the whole
      mechanism.

    Three cases:

    * ``"auto"`` (the real default), or ``None``/``""`` on a hand-built config - write
      :data:`WORKER_CLASS_PATH` and return it;
    * already :data:`WORKER_CLASS_PATH` - keep it (idempotent: vLLM or a launcher may have set it);
    * anything else - raise.  Silently overwriting would hide an operator's explicit
      ``--worker-cls``; silently accepting it would instantiate a worker that cannot run here
      (``gpu_worker.py:285`` builds ``cuda:<rank>``).  A refused start-up is the only outcome that
      names the conflict instead of failing later inside device init.
    """
    if importlib.util.find_spec(WORKER_MODULE) is None:  # pragma: no cover - the file exists here
        raise NotImplementedError(
            f"the worker module {WORKER_MODULE!r} is missing, so check_and_update_config cannot "
            f"select {WORKER_CLASS_PATH}. vLLM would then instantiate its own CUDA worker "
            "(vllm/v1/worker/gpu_worker.py:285 hard-codes torch.device('cuda:...')) and fail on "
            "the first NPU. Restore tileinfer/vllm/worker.py before starting the engine."
        )
    parallel_config = _require_field(
        vllm_config, "VllmConfig", "parallel_config", "the worker class is selected there"
    )
    if not hasattr(parallel_config, WORKER_CLS_ATTR):
        raise NotImplementedError(
            f"ParallelConfig has no attribute {WORKER_CLS_ATTR!r}. The field is real in vLLM 0.23 "
            "(runtime/vllm-api/vllm__config__parallel.py:248: worker_cls: str = \"auto\"), so a "
            "config without it is not a vLLM 0.23 ParallelConfig - or the field was renamed in the "
            "installed release. Refusing to start: without it vLLM builds its CUDA worker and "
            "fails at device init on the NPU. Re-fetch vllm/config/parallel.py into "
            "runtime/vllm-api/ and update WORKER_CLS_ATTR in tileinfer/vllm/platform.py."
        )
    current = getattr(parallel_config, WORKER_CLS_ATTR)
    if current is None or current == "" or current == WORKER_CLS_AUTO:
        # The sentinel path: this is the same branch cpu.py:153 takes.
        parallel_config.worker_cls = WORKER_CLASS_PATH
        return WORKER_CLASS_PATH
    if isinstance(current, str) and current == WORKER_CLASS_PATH:
        return WORKER_CLASS_PATH
    raise ValueError(
        f"parallel_config.worker_cls={current!r} conflicts with the TileInfer Ascend platform, "
        f"which can only run {WORKER_CLASS_PATH}. vLLM resolves this string into the worker class "
        "(runtime/vllm-api/vllm__v1__worker__worker_base.py:249-253), and the class it names here "
        "would be built on a device that is not an NPU (gpu_worker.py:285 hard-codes cuda:<rank>). "
        "Drop --worker-cls (the default is the 'auto' sentinel, "
        "runtime/vllm-api/vllm__config__parallel.py:248) or pass "
        f"--worker-cls {WORKER_CLASS_PATH} explicitly."
    )


def _disable_cuda_only_compilation_passes(vllm_config: Any) -> None:
    """Turn off vLLM's fusion passes that only exist on CUDA/XPU (an upstream bug, not a choice).

    ``vllm/compilation/passes/pass_manager.py`` imports these classes **only** under
    ``current_platform.is_cuda_alike()`` (lines 33-44) or ``is_xpu()`` (lines 46-50), but then
    instantiates them guarded only by the *config flag*:

    ```
    if self.pass_config.fuse_norm_quant:
        ...
        self.passes += [RMSNormQuantFusionPass(config)]      # pass_manager.py:164-165
    if self.pass_config.fuse_act_quant:
        self.passes += [ActivationQuantFusionPass(config)]   # pass_manager.py:169
    ```

    On any other accelerator that is a ``NameError`` at engine start.  Measured: dropping
    ``--enforce-eager`` (which disables the whole pass manager) turned "graph capture unreachable"
    into ``NameError: name 'RMSNormQuantFusionPass' is not defined`` (§67) — i.e. the compilation
    path could never even initialise on this platform.

    Every one of these is a *quantisation* fusion for fp8/int8 models, and this backend refuses
    quantised attention outright (``TileInferAttentionImpl.forward`` raises on
    ``output_scale``/``output_block_scale``), so switching them off cannot change a number here.
    """
    compilation = getattr(vllm_config, "compilation_config", None)
    passes = getattr(compilation, "pass_config", None)
    if passes is None:
        return
    disabled = []
    for name in (
        "fuse_norm_quant",
        "fuse_act_quant",
        "fuse_attn_quant",
        "fuse_mla_attn_quant",
        "fuse_mla_dual_rms_norm",
        "enable_qk_norm_rope_fusion",
        "fuse_rope_kvcache",
    ):
        if not hasattr(passes, name):
            continue
        try:
            if bool(getattr(passes, name)):
                setattr(passes, name, False)
                disabled.append(name)
        except Exception as exc:  # noqa: BLE001 - a frozen config must not abort start-up
            logger.warning("TileInfer: could not disable pass_config.%s (%r)", name, exc)
    if disabled:
        logger.warning(
            "TileInfer: disabled CUDA-only compilation pass(es) %s. vLLM imports their classes "
            "only for is_cuda_alike()/is_xpu() but instantiates them from the config flag alone "
            "(pass_manager.py:33-50 vs :164-169), so they are a NameError on NPU; all of them are "
            "fp8/int8 quantisation fusions, which this backend rejects anyway.",
            ", ".join(disabled),
        )


def _enable_aclgraph_if_ready(vllm_config: Any, graphs: Any) -> None:
    """Ask for decode graph capture when the bridge is in place.  Opt-in via ``TILEINFER_ACLGRAPH=1``.

    Needed because vLLM resolves ``cudagraph_mode`` to ``NONE`` before we see the config: measured
    on this stack, passing ``--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`` still
    produced *"Skipping CUDA graph capture. To turn on CUDA graph capture, ensure `cudagraph_mode`
    was not manually set to `NONE`"* (§68) — the flag does not survive the non-CUDA platform's
    post-init, so it has to be re-applied here, after resolution.
    """
    if os.environ.get("TILEINFER_ACLGRAPH", "").strip().lower() not in ("1", "true", "yes"):
        return
    ready, reason = graphs.capture_path_ready()
    if not ready:
        logger.warning("TileInfer: TILEINFER_ACLGRAPH=1 but the bridge is not ready (%s)", reason)
        return
    compilation = getattr(vllm_config, "compilation_config", None)
    if compilation is None:
        return
    try:
        from vllm.config import CUDAGraphMode  # noqa: PLC0415

        compilation.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        # ``TILEINFER_COMPILE=0`` keeps torch.compile off while *still* asking for decode graphs.
        # That is the configuration §68 pointed at: piecewise compilation gave TPOT 1.064x but cost
        # TTFT 0.63x at 128 tokens (every prefill step re-dispatches through the compiled pieces),
        # and graph capture does not need the compiler at all - vLLM captures the eager module.
        if os.environ.get("TILEINFER_COMPILE", "1").strip().lower() in ("0", "false", "no"):
            try:
                from vllm.config import CompilationMode  # noqa: PLC0415

                compilation.mode = CompilationMode.NONE
            except Exception as exc:  # noqa: BLE001
                logger.warning("TileInfer: could not set compilation.mode (%r)", exc)
        # vLLM derives both from the config and asserts on the derived value
        # ("Maximum cudagraph capture size must be set when cudagraphs are enabled",
        # cudagraph_dispatcher.py:80), so set the capture grid explicitly rather than hoping a
        # default materialises on a non-CUDA platform.
        sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        if hasattr(compilation, "cudagraph_capture_sizes"):
            compilation.cudagraph_capture_sizes = sizes
        if hasattr(compilation, "max_cudagraph_capture_size"):
            compilation.max_cudagraph_capture_size = sizes[-1]
        logger.warning(
            "TileInfer: TILEINFER_ACLGRAPH=1 -> cudagraph_mode=%s capture_sizes=%s (decode graphs "
            "on the wired torch.cuda -> torch.npu alias bridge)",
            compilation.cudagraph_mode,
            sizes,
        )
    except Exception as exc:  # noqa: BLE001 - a frozen config must not abort start-up
        logger.warning("TileInfer: could not set cudagraph_mode (%r)", exc)


def _force_eager_until_capture_lands(vllm_config: Any, graphs: Any) -> Optional[bool]:
    """Force ``enforce_eager`` while the graph-capture path is not verified (W8).

    ``graphs.capture_path_ready()`` answers whether ``vllm.compilation.cuda_graph`` can actually
    capture on this device (see that module's docstring: it needs two CUDA call sites bridged).
    While the answer is False, asking vLLM to capture would crash inside the model runner - a
    place where the message is unreadable - so the platform asks for eager execution instead and
    says so once, loudly.  Returns the resulting ``enforce_eager`` value (``None`` if the field
    could not be written, which is logged, not swallowed).
    """
    model_config = _require_field(vllm_config, "VllmConfig", "model_config", "eager flag")
    ready, reason = graphs.capture_path_ready()
    if ready:
        return getattr(model_config, "enforce_eager", None)
    if bool(getattr(model_config, "enforce_eager", False)):
        return True
    try:
        model_config.enforce_eager = True
    except Exception as exc:  # noqa: BLE001 - a frozen config must not abort start-up
        logger.warning(
            "TileInfer: could not force enforce_eager=True (%r); the engine may attempt CUDA "
            "graph capture, which is not implemented for NPU yet (%s)",
            exc,
            reason,
        )
        return None
    logger.warning(
        "TileInfer: forcing enforce_eager=True. Graph capture is not verified on this device "
        "(%s), and vllm.compilation.cuda_graph holds two hard-coded torch.cuda call sites "
        "(cuda_graph.py:283, :313). Delete this once W8 lands - see tileinfer/vllm/graphs.py.",
        reason,
    )
    return True


def _selector_reason(profile: ModelProfile, name: str) -> str:
    """The profile's parenthetical reason for one selector field (or a safe fallback)."""
    for feature in profile.unsupported_features:
        if feature.field == name and feature.kind in ("selector", "plain_selector", "attn_type"):
            return feature.reason
    return SELECTOR_REJECT_REASONS.get(name, "not supported")


def _selector_switches(
    attn_selector_config: Any, profile: Optional[ModelProfile] = None
) -> Dict[str, bool]:
    """Read every boolean switch of a real ``AttentionSelectorConfig``; refuse to classify an unknown.

    The field names are vLLM 0.23's (``runtime/vllm-api/vllm__v1__attention__selector.py:21-34``),
    and the policy is one rule for all of them: **a switch that is True means the layer asks for an
    attention feature this platform does not implement**, so it is rejected.  The two tuples differ
    only in what a *missing* name means (the profile's ``selector`` fields identify a real selector;
    its ``plain_selector`` fields may be absent on a duck-typed object, and then it is warned about
    rather than silently assumed False).

    A selector object that exposes **none** of the reject switches is not classified: an object we
    cannot read is not evidence that the layer is plain.  Everything else the real config carries is
    handled elsewhere, so no field is silently dropped:

    * ``attn_type`` - checked here (:data:`SELECTOR_ATTN_TYPE`), because a non-decoder layer has a
      different mask/bias contract entirely;
    * ``head_size``, ``dtype`` - enforced by the backend's ``check_head_layout`` /
      ``validate_configuration`` and by :attr:`supported_dtypes`;
    * ``kv_cache_dtype``, ``block_size`` - enforced by :meth:`check_and_update_config` through
      ``CacheConfig.cache_dtype`` (``interface.py:557-560``) and :func:`validate_block_size`.

    Returns the switches as ``{name: bool}``; :meth:`get_attn_backend_cls` rejects on any ``True``.
    """
    if attn_selector_config is None:
        raise ValueError(
            "get_attn_backend_cls received attn_selector_config=None, so the layer's selector "
            "cannot be classified as plain MHA/GQA. Refusing to claim the backend slot."
        )
    profile = profile or active_profile()
    reject_names = tuple(feature.field for feature in profile.selector_features())
    plain_names = tuple(feature.field for feature in profile.plain_selector_features())
    names = reject_names + plain_names
    present = {
        name: bool(getattr(attn_selector_config, name))
        for name in names
        if hasattr(attn_selector_config, name)
    }
    if not any(name in present for name in reject_names):
        raise NotImplementedError(
            "AttentionSelectorConfig exposes none of "
            f"{list(reject_names)}, so the feature policy cannot be evaluated and the "
            "TileInfer platform refuses to claim the attention backend slot for "
            f"{type(attn_selector_config).__name__}. Those are the real vLLM 0.23 field names "
            "(runtime/vllm-api/vllm__v1__attention__selector.py:21-34); an object without them is "
            "not an AttentionSelectorConfig, and classifying it as 'plain' would be a guess. Check "
            "SELECTOR_REJECT_SWITCHES (derived from the model profile's unsupported_features) in "
            "tileinfer/vllm/platform.py against the selector.py of the vLLM actually installed."
        )
    missing = [name for name in names if name not in present]
    if missing:
        logger.warning(
            "TileInfer: AttentionSelectorConfig does not expose %s; those switches are skipped "
            "rather than read as False, and this warning is the record that the policy could not "
            "see them (a real vLLM 0.23 config has all of %s, selector.py:21-34). Update "
            "SELECTOR_REJECT_SWITCHES / SELECTOR_PLAIN_SWITCHES if the installed vLLM renamed them.",
            missing,
            list(names),
        )
    switches = {name: bool(getattr(attn_selector_config, name, False)) for name in present}
    # attn_type is not a boolean, so it is handled as its own rejection below rather than as a
    # switch: a non-decoder layer needs a different mask/bias contract, not a flag flip.
    if not _is_decoder_attention(attn_selector_config):
        switches["attn_type"] = True
    return switches


def _is_decoder_attention(attn_selector_config: Any) -> bool:
    """True when the selector's ``attn_type`` is (or defaults to) decoder self-attention.

    ``attn_type: str = AttentionType.DECODER`` is the real field (``selector.py:31``) and
    ``AttentionType`` is a ``str`` Enum (``v1/attention/backend.py:32``), so comparing its string
    form works for both the enum member and a bare string.  An object that does not expose the field
    at all is warned about by :func:`_selector_switches` and treated as decoder, which is the
    field's own default - the reject path only fires on a selector that *says* it is something else.
    """
    attn_type = getattr(attn_selector_config, "attn_type", None)
    if attn_type is None:
        return True
    name = str(getattr(attn_type, "value", attn_type)).strip().lower()
    return name == SELECTOR_ATTN_TYPE


def _reject_foreign_backend_selection(selected_backend: Any) -> None:
    """Allow ``None`` (auto) and TileInfer's own registration; reject a pinned foreign backend.

    What the real source settles (``runtime/vllm-api/vllm__v1__attention__backends__registry.py``):

    * the selector passes ``backend=vllm_config.attention_config.backend``
      (``v1/attention/selector.py:106-110``), i.e. an ``AttentionBackendEnum`` member or ``None``;
    * ``AttentionBackendEnum.CUSTOM`` is the out-of-tree placeholder and its value is ``None``
      (``registry.py:99-101``: "set to None to avoid alias with other backend, whose value is an
      empty string") - so the CUSTOM test must be an **identity** comparison against the member, not
      a falsiness/``None`` test on the value;
    * ``AttentionBackendEnum.TORCH_SDPA`` is the member whose value is the **empty string**
      (``registry.py:61``, "this tag is only used for ViT"), so ``""`` is *not* a synonym for
      "custom": accepting it would silently claim a ViT-only tag for a decoder layer.

    A bare string is still tolerated for ``custom``/``tileinfer`` because a CLI may hand the hook the
    text form; any other value - pinned name, foreign enum member, empty string - is refused.  That
    tolerance is a convenience, not a claim about vLLM: the enum identity check above is the real
    contract.
    """
    if selected_backend is None:
        return
    if isinstance(selected_backend, str):
        name = selected_backend.strip().lower()
        if name in ("custom", "tileinfer"):
            return
        if name == "":
            raise ValueError(
                "an empty attention-backend string is AttentionBackendEnum.TORCH_SDPA "
                "(backends/registry.py:61 - 'this tag is only used for ViT'), not 'auto' and not "
                f"TileInfer's backend. v0 provides exactly one: {ATTN_BACKEND_CLASS_PATH}. Pass "
                "--attention-backend custom or leave it unset."
            )
        raise ValueError(
            f"--attention-backend {selected_backend!r} is not served by the TileInfer Ascend "
            "platform; v0 provides exactly one attention backend "
            f"({ATTN_BACKEND_CLASS_PATH}). Use 'custom' or leave it unset."
        )
    try:  # pragma: no cover - needs a real vLLM
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
    except ImportError:  # pragma: no cover - the fake/absent vLLM
        raise RuntimeError(
            "cannot validate the requested attention backend: "
            "vllm.v1.attention.backends.registry.AttentionBackendEnum is not importable. "
            f"Only {ATTN_BACKEND_CLASS_PATH} can run on this platform."
        )
    if selected_backend is AttentionBackendEnum.CUSTOM:
        return
    raise ValueError(
        f"attention backend {selected_backend!r} is not served by the TileInfer Ascend platform; "
        f"v0 provides exactly one ({ATTN_BACKEND_CLASS_PATH}). Pass --attention-backend custom "
        "or leave it unset (AttentionBackendEnum.CUSTOM is the out-of-tree placeholder, "
        "backends/registry.py:99-101)."
    )


# =============================================================================================
# vllm.platform_plugins entry point
# =============================================================================================


def register() -> Optional[str]:
    """``vllm.platform_plugins`` entry point: the platform class path, or ``None``.

    Return contract, straight out of the local source: the value must be a **string** class path
    (``vllm/platforms/__init__.py:221`` ``if platform_cls_qualname is not None``), and it is
    resolved and instantiated with no arguments (``:277-278``).  Returning ``None`` means "not
    applicable" and lets the built-in plugins (CUDA/ROCm/CPU/...) be considered instead
    (``:244-250``).

    Called twice by vLLM on the winning plugin (``:221`` then ``:237``), so it must stay cheap:
    one import probe plus one device check, no side effects, no caching needed.
    """
    if not _npu_available():
        logger.debug("TileInfer platform plugin: no NPU visible, not activating")
        return None
    import os

    if str(os.environ.get("VLLM_DSV41", "")).strip().lower() in ("1", "true", "yes", "on"):
        # DeepSeek-V4.1-Flash: fp8-block-32 dense + fp4 experts, which the v0 gate rejects.  The
        # subclass overrides only verify_quantization, so with the flag unset this function behaves
        # exactly as before and the Qwen3.6 arm is untouched (C2).
        logger.info("TileInfer platform plugin: VLLM_DSV41=1 -> Dsv41Platform (fp8/fp4 admitted)")
        return "tileinfer.vllm.platform_dsv41.Dsv41Platform"
    return "tileinfer.vllm.platform.TileInferPlatform"
