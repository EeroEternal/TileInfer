"""``NPUWorker`` — the Ascend worker: the four hooks vLLM 0.23's GPU worker cannot serve.

Status
------
Implemented against the local vLLM 0.23 sources.  ``gpu_model_runner.py`` is now in
``runtime/vllm-api/`` (334,736 B), so the signatures below are *read*, not inferred:

* ``runtime/vllm-api/vllm__v1__worker__gpu_worker.py`` (the ``Worker`` we subclass, 1,204 lines);
* ``runtime/vllm-api/vllm__v1__worker__gpu_model_runner.py`` (``GPUModelRunner``, 7,561 lines:
  class ``:418``, ``_init_device_properties`` ``:1109-1112``, ``_dummy_run`` ``:5599``,
  ``profile_run`` ``:6168``, ``profile_cudagraph_memory`` ``:6376``, ``capture_model`` ``:6525``,
  ``initialize_kv_cache`` ``:7284``).

Every hook below has a working default derived from the GPU worker's own body, and a docstring
that separates three things: what the inherited body does (with file:line), what this override
does, and which part only hardware can confirm (marked ``HARDWARE:`` in prose, ``# HARDWARE:``
in code) together with the *first* command to run when it fails.

The four hooks, and what the inherited body would do wrong
----------------------------------------------------------
``init_device`` (``gpu_worker.py:249-345``)
    branches on ``device_config.device_type == "cuda"`` (``:251``), builds
    ``torch.device(f"cuda:{local_rank}")`` (``:285``), and the ``else`` raises
    ``RuntimeError("Not support device type")`` (``:319-320``).

``determine_available_memory`` (``:371-524``)
    profiles through ``torch.accelerator.memory_stats`` (``:412``) and accounts CUDA-graph memory
    with ``profile_cudagraph_memory`` (``:416-425``); both are CUDA quantities.

``initialize_from_config`` (``:562-589``)
    wraps the allocation in the CuMem memory-pool context (``:577`` -> ``:202-223``), which for an
    out-of-tree platform falls through to ``get_mem_allocator_instance()`` and raises - the
    XPU/CPU branches at ``:209-216`` are the precedent for ``nullcontext``.

``compile_or_warm_up_model`` (``:591-750``)
    calls ``kernel_warmup`` (``:626``, a Triton/CUDA path), calls
    ``self.model_runner.capture_model()`` whenever the model is not eager (``:630``), and starts
    the Triton JIT monitor (``:736-740``).

What *is* reused verbatim, and why that is safe
-----------------------------------------------
The hybrid KV-cache question ("do we hand-build 10 attention + 30 mamba specs?") is answered by
the source: **no**.  ``Worker.get_kv_cache_spec`` (``gpu_worker.py:547-548``) delegates to
``self.model_runner.get_kv_cache_spec()`` (``gpu_model_runner.py:7440``), i.e. the specs come from
the model, and ``GPUModelRunner.initialize_kv_cache`` (``:7284-7341``) consumes whatever
``KVCacheConfig`` it is handed:

* ``initialize_attn_backend`` (``:7300``, defined ``:6677``) builds the attention backends and
  metadata builders through ``current_platform.get_attn_backend_cls`` -> the TileInfer backend;
* ``initialize_mamba_ssu_backend`` (``:7301``) is the mamba-state half of the hybrid path;
* ``initialize_kv_cache_tensors`` (``:7319``, defined ``:7201``) allocates with
  ``torch.zeros(..., device=self.device)`` (``:7014``) - device-agnostic;
* ``_reshape_kv_cache_tensors`` (``:7040``) walks attention **and** mamba groups
  (``has_attn, has_mamba`` at ``:7057``).

So the GPU worker's KV path is device-agnostic (modulo the memory-pool context, which is
overridden here as ``nullcontext``), and this module does **not** build or reorder specs.  There is
also no ``NpuPlatformKVCache`` in this repository yet (``grep -r NpuPlatformKVCache`` is empty), so
there is no TileInfer pool class to route the attention pool through: the pool is allocated by the
runner just as on CUDA, and the TileInfer *write* path (``tileinfer/vllm/attn/kv_cache.py``) takes
the resulting fused NHD tensor.  If a ``NpuPlatformKVCache`` is added later, the seam is
``initialize_kv_cache_tensors`` (``:7201``) / ``_allocate_kv_cache_tensors`` (``:6999``), not this
worker.

The model runner: ``TileInferModelRunner`` is needed, for one incompatible default plus one
-----------------------------------------------------------------------------------------------
``GPUModelRunner.__init__`` calls ``self._init_device_properties()`` (``:711``), whose body is
``self.num_sms = num_compute_units(self.device.index)`` (``:1109-1112``).  ``num_compute_units``
comes from ``vllm/utils/platform_utils.py`` (imported ``:118``), and the platform-side
implementation is ``Platform.num_compute_units`` - which
``TileInferPlatform`` deliberately leaves raising (``UNIMPLEMENTED_ABC_METHODS
["num_compute_units"]``, reason ``unverified``: "ABC raises (interface.py:1027-1035)").
Constructing the GPU runner unmodified would therefore abort during ``__init__``, before a single
weight is loaded.

The upstream source marks that seam as intended for exactly this: ``:1108`` reads
``# Note: used for model runner override.``  :class:`TileInferModelRunner` also declines
``_init_kv_zero_meta``: ``KVBlockZeroer`` sizes a block from ``stride(block_dim)``, which is half
a page under this backend's K/V-outermost stride order, so it would zero K and never V (the
method's docstring has the line refs and the cost).  Everything else in the runner either goes
through ``self.device`` / ``current_platform`` or is switched off in this deployment:

* ``_sync_device`` -> ``torch.accelerator.synchronize()`` (``:1115-1116``);
* ``initialize_kv_cache`` as listed above;
* the CUDA-graph paths are gated off: ``capture_model`` (``:6525``) is never reached while
  :func:`tileinfer.vllm.graphs.capture_path_ready` is False, and it also hard-codes
  ``torch.cuda.mem_get_info()`` (``:6547``, ``:6565``) *in addition to* the two wrapper call sites
  that ``graphs.py`` analyses - see ``TileInferModelRunner``'s docstring.

Import policy
-------------
This module imports vLLM's ``Worker`` and ``GPUModelRunner`` (and therefore vLLM, and therefore the
class it subclasses).  Outside a vLLM environment the import raises an :class:`ImportError` naming
both missing modules and the expected version.  ``torch_npu`` is imported **inside** functions
(``_npu_module`` and its callers) so this module imports on a CPU box, and
``tests/test_worker_contract.py`` pins that with an AST check.

Anything this module cannot settle from the local sources is written down three times: in
:data:`HARDWARE_CHECKLIST` (data, printable by ``tileinfer doctor`` and asserted by tests), in the
hook's docstring ("HARDWARE:" + "First diagnostic command"), and where it is decided in the code
(``# HARDWARE:``).
"""

from __future__ import annotations

import gc
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch

# ---------------------------------------------------------------------------------------------
# The two vLLM imports this module builds on.  Both files are transcribed into
# `runtime/vllm-api/`; the error below names them because a partial vLLM install (a `vllm` that
# ships the worker but not the runner) would otherwise surface as an AttributeError much later.
# ---------------------------------------------------------------------------------------------
try:  # pragma: no cover - exercised only outside a vLLM env (the CPU suite injects a fake)
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "tileinfer.vllm.worker requires vLLM 0.23's V1 worker and model runner "
        "(vllm.v1.worker.gpu_worker.Worker and vllm.v1.worker.gpu_model_runner.GPUModelRunner); "
        f"importing them failed with {exc!r}. Install vllm>=0.23,<0.24 - this module subclasses "
        "both (see runtime/vllm-api/vllm__v1__worker__gpu_worker.py and "
        "runtime/vllm-api/vllm__v1__worker__gpu_model_runner.py)."
    ) from exc

from tileinfer.vllm import graphs
from tileinfer.vllm.profiles import active_profile, default_profile

__all__ = [
    "NPUWorker",
    "TileInferModelRunner",
    "NpuMemorySnapshot",
    "WORKER_CLASS_PATH",
    "HOOKS",
    "HARDWARE_CHECKLIST",
    "NPU_DEVICE_TYPES",
    "NPU_CORE_COUNT_ATTRS",
    "CAPTURE_COUNTER_KEYS",
    "PREWARM_COUNTER_KEYS",
    "PREWARM_PLAN_KEYS",
]

#: The path :data:`tileinfer.vllm.platform.WORKER_CLASS_PATH` must equal (a contract test pins the
#: two together so the platform cannot point at a class that was renamed here).
WORKER_CLASS_PATH = "tileinfer.vllm.worker.NPUWorker"

#: The four hooks the engine relies on, in call order: ``init_device`` (worker start) ->
#: ``determine_available_memory`` (after weights) -> ``initialize_from_config`` (KV pool) ->
#: ``compile_or_warm_up_model`` (warm-up, then serving).  Kept as data so the tests can parametrise
#: over them and ``tileinfer doctor`` can print the same list the docstring talks about.
HOOKS: Tuple[str, ...] = (
    "init_device",
    "determine_available_memory",
    "initialize_from_config",
    "compile_or_warm_up_model",
)

#: Device-type spellings this worker serves: ``npu`` once ``torch_npu`` has renamed
#: ``PrivateUse1``, and the un-renamed ``privateuseone`` alias for the same backend.
NPU_DEVICE_TYPES: Tuple[str, ...] = ("npu", "privateuseone")

#: Attributes that could carry the AI-core count behind ``GPUModelRunner._init_device_properties``
#: (``gpu_model_runner.py:1112``).  None of the spellings is in the local tree, so
#: :meth:`TileInferModelRunner._init_device_properties` tries them in order and *says so* when it
#: has to fall back.
NPU_CORE_COUNT_ATTRS: Tuple[str, ...] = ("num_aicore", "num_ai_core", "aicore_num", "cube_core_num")

#: Counter keys of :meth:`NPUWorker.capture_model`, kept as data so a probe/test can assert on the
#: same names the logs use.
CAPTURE_COUNTER_KEYS: Tuple[str, ...] = ("attempts", "captured", "skipped")

#: Counter keys of :meth:`NPUWorker._prewarm_decode_plans`, kept as data for the same reason.
#:
#: ``attempts``
#:     calls to the pre-warm (one per ``compile_or_warm_up_model``);
#: ``layers``
#:     TileInfer attention layers whose bucket state was prepared (a 40-layer hybrid model has 10);
#: ``buckets``
#:     size of the bucket set derived from the config;
#: ``plans``
#:     decode plans *compiled* by this worker, cumulative (a second call rebuilds nothing);
#: ``failed``
#:     buckets or layers that raised (counted and logged; one bad bucket must not stop the boot).
PREWARM_COUNTER_KEYS: Tuple[str, ...] = ("attempts", "layers", "buckets", "plans", "failed")

#: The non-counter entries :meth:`NPUWorker._prewarm_decode_plans` records beside the counters: the
#: derived plan itself, so a probe can compare what the worker pre-warmed with what
#: ``scripts/serve_qwen36.py --plan-only`` printed before start-up.
PREWARM_PLAN_KEYS: Tuple[str, ...] = (
    "skipped_reason",
    "bucket_set",
    "fitted_bucket_set",
    "width",
    "engine_width",
    "block_size",
    "max_num_reqs",
    "pool_cap",
)

#: What the hardware pass still has to establish.  Kept as data (not prose) so ``tileinfer doctor``
#: and the tests can print it; each entry names the file:line the decision rests on and the command
#: or check that settles it.
HARDWARE_CHECKLIST: Tuple[str, ...] = (
    "torch.npu.set_device must accept a torch.device (torch_npu also exposes the int-index form): "
    "python -c \"import torch, torch_npu; torch.npu.set_device(torch.device('npu:0'))\" - "
    "TileInferPlatform.set_device (platform.py:993-995) is what init_device calls",
    "the process-group backend: current_platform.dist_backend is '' (UNVERIFIED_NOTES['dist_backend']) "
    "and init_worker_distributed_environment (gpu_worker.py:299, defined :1162-1204) passes it to "
    "torch.distributed, so HCCL must be named there (or the group skipped) on one card",
    "the device-memory API spellings behind NpuMemorySnapshot: torch.npu.mem_get_info vs "
    "torch.npu.memory.mem_get_info, and torch.npu.memory_stats(device)['allocated_bytes.all.peak'] "
    "vs torch.npu.max_memory_allocated(device) (gpu_worker.py:412-414 uses the CUDA spelling)",
    "graph capture is still gated off: graphs.capture_path_ready() must be True *and* the runner's "
    "own CUDA call sites (gpu_model_runner.py:6547, :6565) must be bridged - the wrapper's two sites "
    "(cuda_graph.py:283, :313) are not the whole story",
    "the AI-core count for GPUModelRunner._init_device_properties: torch.npu.get_device_properties(0) "
    "must carry one of NPU_CORE_COUNT_ATTRS, else num_sms stays the warned-about sentinel 1",
    "the hybrid KV pool allocation: support_hybrid_kv_cache() now answers True (platform.py) and "
    "the platform pads the mamba state page to the attention page, so initialize_kv_cache "
    "(gpu_model_runner.py:7284) must allocate the 10 NHD attention layers + 30 mamba state layers "
    "for block_size 2048 (docs/plan-vllm-native-backend.md §5 / W7); the judge is the count "
    "(10 + 30 = 40 layers across vLLM's page-size groups), not the log wording",
    "the mamba state-update path, which is NOT the SSU backend for this model: vLLM 0.23 calls "
    "initialize_mamba_ssu_backend (gpu_model_runner.py:7301) but its body returns unless a group "
    "is mamba_type MAMBA1/MAMBA2 (ssu_dispatch.py:202-208), and every GDN layer reports GDN_ATTN "
    "(mamba/gdn/base.py:49-51), so Qwen3.6 never instantiates TritonSSUBackend. The decode state "
    "update is fused_recurrent_gated_delta_rule_packed_decode (qwen_gdn_linear_attn.py:1684; "
    "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE defaults to 1), bound to our torch port by "
    "tileinfer/vllm/ops/install.py. NPUWorker.initialize_from_config raises at start-up if a "
    "model ever needs MAMBA1/MAMBA2 (there is no NPU SSU backend and no registration hook: "
    "MambaBackendEnum is triton/flashinfer only, ssu_dispatch.py:185-188). What only a device "
    "run can show: our port's state write into the real paged ssm_state tensor, the conv_state "
    "transpose and the packed-decode contiguity contract at NPU dtypes/layouts",
    "the KVBlockZeroer outcome: the hybrid attention layout is settled, not a risk - "
    "get_kv_cache_shape is (num_blocks, 2, block_size, heads, dim) with "
    "get_kv_cache_stride_order() == (1, 0, 2, 3, 4), so block_dim is 0, "
    "_update_hybrid_attention_mamba_layout skips the pool "
    "(gpu_model_runner.py:7189-7191), kv_cache[:, 0] / kv_cache[:, 1] are the contiguous halves, "
    "and the first forward is expected to reach the write/plan path (the old contiguous-pool "
    "raise is gone). What remains is the zeroer: this runner declines it "
    "(TileInferModelRunner._init_kv_zero_meta) because per-block bytes come from "
    "stride(block_dim) = half a page (v1/worker/utils.py:149-150), so a device run must show no "
    "garbage on a recycled attention page (the residual cost of the skip is documented there)",
)

logger = logging.getLogger("tileinfer.vllm.worker")


# =============================================================================================
# The mamba state-update path: decided from vLLM 0.23's source, not from the flag name
# =============================================================================================
#
# Two different things in vLLM are called "mamba backend"; only the second one is about the
# state update this file has to police:
#
# * the **mamba attention backend** (``MambaAttentionBackendEnum``, metadata builder / state
#   shape) is extensible: ``register_backend(..., is_mamba=True)`` is public
#   (``v1/attention/backends/registry.py:214-266``).  This platform does not register one; the
#   in-tree GDN one is metadata-only.
# * the **Mamba SSU backend** (``selective_state_update``) is not extensible: the config value is
#   ``MambaConfig.backend`` (``triton``/``flashinfer`` only, ``vllm/config/mamba.py:25-29``) and
#   the registry is the module-private dict ``ssu_dispatch._BACKEND_REGISTRY``
#   (``.../mamba/ops/ssu_dispatch.py:185-188``) with no ``register_*`` function.
#
# And for this deployment's model it is never reached: ``initialize_mamba_ssu_backend`` returns
# unless a KV-cache group is ``MambaSpec(mamba_type=MAMBA1|MAMBA2)`` (``ssu_dispatch.py:202-208``),
# while every GDN layer answers ``GDN_ATTN`` (``.../mamba/gdn/base.py:49-51``) and that value is
# copied into the spec by ``MambaBase.get_kv_cache_spec`` (``.../mamba/abstract.py:44-60``).
# The GDN state update is the FLA packed-decode op, which ``tileinfer.vllm.ops.install`` binds to
# our torch port - so "our backend selected" for Qwen3.6 means that binding, not an SSU class.

#: ``MambaSpec.mamba_type`` values of the boot profile's hybrid kind.  The *policy* below reads
#: :func:`tileinfer.vllm.profiles.active_profile` at call time, so a second model family can accept
#: a different mamba type; these aliases describe the default profile for probes/tests.
_BOOT_PROFILE = default_profile()
_SSU_TYPES = (
    tuple(_BOOT_PROFILE.hybrid.ssu_mamba_types) if _BOOT_PROFILE.hybrid is not None else ()
)
_MAMBA_TYPES = (
    tuple(_BOOT_PROFILE.hybrid.mamba_types) if _BOOT_PROFILE.hybrid is not None else ()
)

#: ``MambaSpec.mamba_type`` values whose state update runs through vLLM's Mamba **SSU**
#: (selective state update) dispatch.  Exactly the set ``initialize_mamba_ssu_backend`` tests
#: before it instantiates anything (``ssu_dispatch.py:202-208``).
SSU_MAMBA_TYPES: Tuple[str, ...] = _SSU_TYPES

#: ``MambaSpec.mamba_type`` of the GDN layers this deployment serves.  Deliberately *not* in
#: :data:`SSU_MAMBA_TYPES`: vLLM skips the SSU backend for it, and its decode state update is
#: ``fused_recurrent_gated_delta_rule_packed_decode`` (bound by ``tileinfer.vllm.ops.install``).
GDN_MAMBA_TYPES: Tuple[str, ...] = _MAMBA_TYPES


def mamba_state_path(kv_cache_config: Any) -> Dict[str, Any]:
    """Classify the config's mamba groups by ``MambaSpec.mamba_type`` - a pure function.

    It reads only ``kv_cache_groups[*].kv_cache_spec.mamba_type`` (and the group's layer names),
    so it runs against the real ``KVCacheConfig`` and against the CPU fakes without importing
    vLLM or touching a device.  Attention groups (no ``mamba_type``) are ignored; a spec whose
    ``mamba_type`` is a plain string is normalised to upper case, so the enum and the fake compare
    the same way.

    Returns ``{"types": {name: layer_count}, "ssu_layers": int, "gdn_layers": int,
    "decision": str}`` where ``decision`` is:

    * ``"ssu"`` - at least one group is MAMBA1/MAMBA2.  vLLM would run
      ``TritonSSUBackend``/``FlashInferSSUBackend``; this platform has neither, so
      :func:`require_supported_mamba_state_path` refuses to start.
    * ``"gdn-fla"`` - mamba groups, none of them SSU: the GDN hybrid. vLLM's SSU backend is a
      no-op and the state update is our ported packed-decode op.
    * ``"none"`` - attention-only.
    """
    layers: Dict[str, int] = {}
    for group in getattr(kv_cache_config, "kv_cache_groups", ()) or ():
        spec = getattr(group, "kv_cache_spec", None)
        mamba_type = getattr(spec, "mamba_type", None)
        if mamba_type is None:
            continue
        name = str(getattr(mamba_type, "name", mamba_type)).upper()
        layers[name] = layers.get(name, 0) + len(tuple(getattr(group, "layer_names", ()) or ()))
    # The classification is the active model profile's hybrid kind: a profile with ``hybrid=None``
    # serves no mamba state path, and a future GDN-like family supplies its own accepted/SSU types.
    hybrid = active_profile().hybrid
    ssu_accepted = tuple(hybrid.ssu_mamba_types) if hybrid is not None else ()
    gdn_accepted = tuple(hybrid.mamba_types) if hybrid is not None else ()
    ssu_types = tuple(sorted(name for name in layers if name in ssu_accepted))
    gdn_types = tuple(sorted(name for name in layers if name in gdn_accepted))
    ssu_layers = sum(layers[name] for name in ssu_types)
    gdn_layers = sum(layers[name] for name in gdn_types)
    if ssu_types:
        decision = "ssu"
    elif gdn_types:
        decision = "gdn-fla"
    else:
        decision = "none"
    return {
        "types": layers,
        "ssu_layers": ssu_layers,
        "gdn_layers": gdn_layers,
        "decision": decision,
    }


def require_supported_mamba_state_path(kv_cache_config: Any) -> Dict[str, Any]:
    """Return :func:`mamba_state_path`, or refuse to start with the missing backend named.

    This is the "loud at start-up" half of the decision above.  ``initialize_kv_cache`` calls
    ``initialize_mamba_ssu_backend`` (``gpu_model_runner.py:7301``) before any step runs; for a
    MAMBA1/MAMBA2 model that call would import vLLM's Triton kernel (or FlashInfer) and fail
    inside the engine with a traceback that names a kernel, not the missing platform backend.
    Raising here - from the worker, with the exact config object - names the backend, the two
    supported values, why neither runs on Ascend, and the two real options.

    GDN hybrids pass (``decision == "gdn-fla"``); their SSU backend is a no-op, and refusing them
    would block the only model this repository targets.
    """
    path = mamba_state_path(kv_cache_config)
    if path["decision"] != "ssu":
        return path
    profile = active_profile()
    hybrid = profile.hybrid
    needs = ", ".join(
        name
        for name in path["types"]
        if hybrid is not None and name in tuple(hybrid.ssu_mamba_types)
    )
    gdn_types = "/".join(tuple(hybrid.mamba_types)) if hybrid is not None else "GDN_ATTN"
    state_update_op = (
        hybrid.state_update_op
        if hybrid is not None and hybrid.state_update_op
        else "fused_recurrent_gated_delta_rule_packed_decode"
    )
    raise RuntimeError(
        "TileInfer cannot start this model: "
        f"{path['ssu_layers']} mamba layer(s) use vLLM's Mamba SSU dispatch ({needs}), and "
        "vLLM 0.23 has no NPU backend for it. The only supported values are 'triton' and "
        "'flashinfer' (vllm/config/mamba.py:25-29), and both are CUDA/Triton kernels "
        "(vllm/model_executor/layers/mamba/ops/ssu_dispatch.py:57-182). There is no registration "
        "hook: the registry is the private dict _BACKEND_REGISTRY (ssu_dispatch.py:185-188), and "
        "--mamba-backend strings resolve through MambaBackendEnum[...] (engine/arg_utils.py:"
        "2125-2128), so no config value can name a TileInfer class. Two options: "
        f"(1) run a GDN/linear-attention hybrid ({profile.name}) - its MambaSpec.mamba_type is "
        f"{gdn_types} (mamba/gdn/base.py:49-51), initialize_mamba_ssu_backend skips it "
        f"(ssu_dispatch.py:202-208), and the state update is "
        f"{state_update_op}, already ported and bound by "
        "tileinfer.vllm.ops.install; "
        "(2) add an NPU SSU backend upstream - a new MambaBackendEnum member plus an "
        "implementation in ssu_dispatch._BACKEND_REGISTRY (or a public "
        "register_mamba_ssu_backend), then --mamba-backend <name> - which is a vLLM change this "
        "platform cannot select. TileInfer does not mutate the private registry or extend the "
        "enum."
    )


# =============================================================================================
# Device / memory helpers.  torch_npu is imported in _npu_module only, never at module level.
# =============================================================================================


def _npu_module(what: str) -> Any:
    """Import ``torch_npu`` and return the ``torch.npu`` module, or raise with the reason.

    ``torch_npu`` is the layer-dependency guard's one forbidden module at import time
    (``tests/test_layer_deps.py:40``), so it is imported here - inside a function - and every
    device access goes through the returned module object.  The import is what *registers* the
    backend; the module object itself is ``torch.npu``.
    """
    try:
        import torch_npu  # noqa: F401  (the import is what registers the backend)
    except Exception as exc:  # pragma: no cover - needs a CPU-only box to exercise
        raise RuntimeError(
            f"tileinfer.vllm.worker.{what} requires torch_npu, which is not importable ({exc!r}). "
            "The TileInfer worker is Ascend-only: every one of its hooks reads or writes device "
            "memory, and answering without a device would mean inventing a KV-pool size. Install "
            "a torch_npu that matches this torch build (docs/hardware-checklist.md §1), then run: "
            "python -c \"import torch, torch_npu; print(torch.__version__, torch_npu.__version__, "
            "torch.npu.device_count())\""
        ) from exc
    npu = getattr(torch, "npu", None)
    if npu is None:  # pragma: no cover - torch_npu always registers torch.npu
        raise RuntimeError(
            f"tileinfer.vllm.worker.{what}: torch_npu imported but torch.npu is missing; the "
            "installed torch_npu does not match this torch build."
        )
    return npu


def _check_npu_device(device: Any, what: str) -> str:
    """Return the device type, or raise if it is not an NPU.

    This is the guard behind "the memory helper rejects a non-npu device": the accounting below
    turns raw byte counts into a KV-pool size, and a CUDA/CPU device would silently answer for a
    different allocator.  ``privateuseone`` is accepted because that is the un-renamed spelling of
    the same backend (``torch_npu`` renames ``PrivateUse1`` to ``npu``).
    """
    kind = getattr(device, "type", None)
    if kind not in NPU_DEVICE_TYPES:
        raise ValueError(
            f"tileinfer.vllm.worker.{what} only accounts memory on an Ascend NPU; got "
            f"device={device!r} (device type {kind!r}). Call it with torch.device('npu:<rank>') - "
            f"or the 'privateuseone' alias of the same backend, which is what "
            f"NPU_DEVICE_TYPES={NPU_DEVICE_TYPES!r} accepts. A CPU/CUDA device would produce a "
            "number for a different allocator, and a wrong KV-pool size fails later, inside the "
            "block manager."
        )
    return str(kind)


def _npu_device(local_rank: int) -> torch.device:
    """``torch.device("npu:<local_rank>")``, or the ``privateuseone`` alias when the rename is absent.

    ``gpu_worker.py:285`` builds ``torch.device(f"cuda:{self.local_rank}")``; the NPU spelling is
    the same construction, but ``torch.device("npu", 0)`` only parses once ``torch_npu`` has
    renamed the ``PrivateUse1`` backend.  Trying both spellings is the difference between a clear
    error naming the device type and a bare ``RuntimeError: Expected one of cpu, cuda, ...``.
    """
    attempts = []
    for type_name in NPU_DEVICE_TYPES:
        try:
            return torch.device(type_name, int(local_rank))
        except (RuntimeError, TypeError) as exc:
            attempts.append(f"{type_name!r}: {exc}")
    raise RuntimeError(
        "tileinfer.vllm.worker.init_device cannot build a device for local_rank="
        f"{local_rank}: neither the 'npu' nor the 'privateuseone' spelling is registered on this "
        "torch build (" + "; ".join(attempts) + "). torch_npu renames PrivateUse1 to 'npu' on "
        "import; if this fires, torch_npu imported but did not register - check the version pair "
        "(docs/hardware-checklist.md §1)."
    )


def _set_npu_device(npu: Any, device: torch.device, local_rank: int) -> str:
    """Make ``device`` current; return the spelling that worked.

    ``gpu_worker.py:286`` uses ``torch.accelerator.set_device_index(self.device)``.  On NPU the
    direct spelling is ``torch.npu.set_device`` and the platform already wraps it
    (:meth:`TileInferPlatform.set_device`, ``platform.py:993-995``), so that is what is called - but
    the index form is the other known spelling and is retried once, because a ``TypeError`` here
    would otherwise abort start-up over a signature difference.

    HARDWARE: that ``torch.npu.set_device`` accepts a ``torch.device`` (the platform's wrapper
    passes one).  First diagnostic:
    ``python -c "import torch, torch_npu; torch.npu.set_device(torch.device('npu:0')); print(torch.npu.current_device())"``
    """
    setter = getattr(npu, "set_device", None)
    if not callable(setter):  # pragma: no cover - needs a torch_npu variant without set_device
        raise RuntimeError(
            "torch.npu.set_device is missing from this torch_npu build; the worker cannot select "
            "device " + str(device) + ". Check the version pair (docs/hardware-checklist.md §1)."
        )
    try:
        setter(device)
        return "torch.npu.set_device(torch.device)"
    except (TypeError, ValueError) as exc:
        logger.debug(
            "torch.npu.set_device(%r) failed (%r); retrying with the int index", device, exc
        )
        setter(int(local_rank))
        return "torch.npu.set_device(int)"


@dataclass(frozen=True)
class NpuMemorySnapshot:
    """The three fields of ``vllm.utils.mem_utils.MemorySnapshot`` the accounting reads.

    ``vllm/utils/mem_utils.py`` is **not** in the local subset, so this is the shape reconstructed
    from the only thing the worker does with it: ``MemorySnapshot(device=...)`` at
    ``gpu_worker.py:313`` and the reads at ``:390`` (``free_memory``), ``:429`` (``torch_peak``),
    ``:467`` / ``:486`` (``free_memory`` / ``total_memory``).  vLLM's class measures with
    ``torch.cuda.mem_get_info`` / ``torch.cuda.memory_stats``; this one measures the same three
    quantities with the torch_npu APIs, which is why it exists instead of being imported.

    HARDWARE: that ``torch.npu`` populates both statistics (see ``_npu_mem_get_info`` /
    ``_npu_peak_allocated``, each of which names both candidate spellings).  First diagnostic:
    ``python -c "import torch, torch_npu; print(torch.npu.mem_get_info(0)); print(torch.npu.memory_stats(0))"``

    Field semantics, identical to the CUDA class: ``total_memory`` is the device size,
    ``free_memory`` is what the *device* has free right now (so other processes count against it),
    and ``torch_peak`` is the allocator's high-water mark for *this* process, in bytes.
    """

    total_memory: int
    free_memory: int
    torch_peak: int
    #: The device the numbers describe.  ``vllm.v1.worker.utils.request_memory`` prints it in its
    #: diagnostics (``f"Free memory on device {init_snapshot.device_}"``) and raises when it is
    #: missing -- which is exactly how the engine refused to start once memory was tight enough for
    #: that helper to be reached ("'NpuMemorySnapshot' object has no attribute 'device_'").  It is a
    #: required field on purpose: defaulting it would print a device that is not the measured one.
    device_: torch.device


def _as_device(device: Any) -> torch.device:
    """``device`` as a ``torch.device``, tolerating a spelling this torch build does not know.

    ``torch.device("npu:0")`` raises on a CPU-only build (no ``torch_npu`` registered), and the
    boot rehearsal runs exactly there.  The value is only ever printed by
    ``vllm.v1.worker.utils.request_memory``, so keeping the caller's own object is both safe and
    more honest than substituting a device that was not measured.
    """
    if isinstance(device, torch.device):
        return device
    try:
        return torch.device(device)
    except Exception:  # noqa: BLE001 - unknown backend spelling on a CPU-only build
        return torch.device("cpu")


def _npu_mem_get_info(npu: Any, device: torch.device) -> Tuple[int, int]:
    """``(free, total)`` in bytes, trying both torch_npu spellings of ``mem_get_info``.

    CUDA's function is ``torch/cuda/memory.py:847``, re-exported as ``torch.cuda.mem_get_info``
    (``torch/cuda/__init__.py:2152``); torch_npu mirrors the top-level name at least.  Both
    spellings are tried, and if neither exists the error names **both**, because the alternative
    is a silent zero that mis-sizes the KV pool.
    """
    candidates = (
        ("torch.npu.mem_get_info", getattr(npu, "mem_get_info", None)),
        (
            "torch.npu.memory.mem_get_info",
            getattr(getattr(npu, "memory", None), "mem_get_info", None),
        ),
    )
    for spelling, fn in candidates:
        if not callable(fn):
            continue
        try:
            info = fn(device)
        except TypeError:  # the index form: mem_get_info(device_index)
            info = fn(getattr(device, "index", 0) or 0)
        if not (isinstance(info, tuple) and len(info) == 2):
            raise RuntimeError(
                f"{spelling}({device!r}) returned {info!r}, not a (free, total) pair of ints. "
                "The worker cannot account device memory from that; check the torch_npu build."
            )
        return int(info[0]), int(info[1])
    raise RuntimeError(
        "torch_npu exposes neither torch.npu.mem_get_info nor torch.npu.memory.mem_get_info, so "
        "the worker cannot measure the NPU's free memory. Both candidate spellings were tried; "
        "CUDA's function lives at torch/cuda/memory.py:847 and is re-exported at "
        "torch/cuda/__init__.py:2152, which is the shape torch_npu mirrors. Find this device's "
        "spelling (or use tileinfer.vllm.platform.get_device_total_memory for the total) and "
        "teach _npu_mem_get_info about it - do not return 0, a zero here mis-sizes the KV pool."
    )


def _npu_peak_allocated(npu: Any, device: torch.device) -> int:
    """Peak *allocated* bytes of this process: ``memory_stats`` first, ``max_memory_allocated`` second.

    ``gpu_worker.py:412-414`` reads ``torch.accelerator.memory_stats(self.device).get(
    "allocated_bytes.all.peak", 0)`` - the CUDA allocator's high-water mark.  torch_npu may expose
    the whole statistics dict or only the dedicated accessor; the two are the same quantity on
    CUDA (``max_memory_allocated`` is defined in ``torch/cuda/memory.py:542`` over the same
    counter), so either spelling is acceptable *and only those two*.  A missing one raises naming
    both, rather than answering 0 - which would under-report peak activation and over-size the KV
    pool.
    """
    stats = getattr(npu, "memory_stats", None)
    if callable(stats):
        result = stats(device) or {}
        peak = result.get("allocated_bytes.all.peak")
        if peak is not None:
            return int(peak)
    max_allocated = getattr(npu, "max_memory_allocated", None)
    if callable(max_allocated):
        return int(max_allocated(device))
    raise RuntimeError(
        "torch_npu exposes neither torch.npu.memory_stats(device)['allocated_bytes.all.peak'] nor "
        "torch.npu.max_memory_allocated(device). gpu_worker.py:412-414 uses the first (through "
        "torch.accelerator.memory_stats) and torch/cuda/memory.py:542 defines the second; the "
        "worker needs one of them to measure peak activation. Teach _npu_peak_allocated about this "
        "device's spelling - answering 0 would over-size the KV pool."
    )


def _npu_memory_snapshot(device: Any, *, npu: Any = None) -> NpuMemorySnapshot:
    """Take an :class:`NpuMemorySnapshot` of ``device`` (the NPU equivalent of ``:313``).

    ``npu`` may be passed in so a caller that already probed ``torch.npu`` does not import
    ``torch_npu`` twice; tests pass a stub.  A non-NPU device is rejected **before** any device
    call (see :func:`_check_npu_device`).
    """
    _check_npu_device(device, "_npu_memory_snapshot")
    npu = npu if npu is not None else _npu_module("_npu_memory_snapshot")
    free_memory, total_memory = _npu_mem_get_info(npu, device)
    return NpuMemorySnapshot(
        total_memory=int(total_memory),
        free_memory=int(free_memory),
        torch_peak=int(_npu_peak_allocated(npu, device)),
        device_=_as_device(device),
    )


def _requested_memory(snapshot: NpuMemorySnapshot, cache_config: Any) -> int:
    """``request_memory(snapshot, cache_config)`` - vLLM's own helper, over our snapshot.

    ``gpu_worker.py:76`` imports ``request_memory`` from ``vllm/v1/worker/utils.py`` (not in the
    local subset) and calls it at ``:314``.  It is device-agnostic arithmetic over
    ``snapshot.total_memory`` / ``snapshot.free_memory`` / ``cache_config.gpu_memory_utilization``,
    which is exactly why :class:`NpuMemorySnapshot` reproduces those field names: vLLM's helper is
    reused, not re-implemented.  If a vLLM point release makes it read a field this class does not
    carry, the ``AttributeError`` is re-raised with both sides named - a silently different
    arithmetic here would be a second, invisible memory model.
    """
    from vllm.v1.worker.utils import request_memory

    try:
        return int(request_memory(snapshot, cache_config))
    except AttributeError as exc:
        raise RuntimeError(
            "vllm.v1.worker.utils.request_memory (gpu_worker.py:76, called at :314) rejected the "
            f"TileInfer NPU memory snapshot: {exc!r}. It reads MemorySnapshot fields; "
            f"NpuMemorySnapshot carries {sorted(NpuMemorySnapshot.__dataclass_fields__)}. Add the "
            "missing field to NpuMemorySnapshot (measured with torch.npu) rather than dropping the "
            "call - the requested-memory arithmetic must stay vLLM's, not a copy of it."
        ) from exc


def _format_gib(num_bytes: int) -> str:
    """``vllm.utils.mem_utils.format_gib``'s job, locally (that module is not in the subset).

    Only used for log lines; keeping it here means the worker does not import one more vLLM
    module just to print a number.
    """
    return f"{int(num_bytes) / (1 << 30):.2f}GiB"


# =============================================================================================
# The model runner: one override, because the GPU runner's device-properties seam is CUDA-only
# =============================================================================================


class TileInferModelRunner(GPUModelRunner):
    """``GPUModelRunner`` with the device-properties seam replaced - the only real override.

    **Why a subclass is needed at all** (the question the task asks): the GPU runner is *almost*
    device-agnostic, because it is parameterised by ``device`` and goes through
    ``current_platform``.  The exception is on the construction path:

    ``GPUModelRunner.__init__`` -> ``self._init_device_properties()`` (``:711``) ->
    ``self.num_sms = num_compute_units(self.device.index)`` (``:1109-1112``), where
    ``num_compute_units`` is ``vllm.utils.platform_utils.num_compute_units`` (imported ``:118``),
    i.e. the ``Platform`` ABC's method.  :class:`tileinfer.vllm.platform.TileInferPlatform` leaves
    that method raising on purpose (``UNIMPLEMENTED_ABC_METHODS``, reason ``unverified``: the ABC
    body raises and the NPU AI-core count is not in the local tree).  Unmodified, the runner would
    therefore abort inside ``__init__`` - before ``load_model``, with a traceback that names a
    platform method rather than the missing number.

    The upstream file marks this exact method as the extension point: ``:1108`` says
    ``# Note: used for model runner override.``  So this subclass exists for that method, and for
    the KV-block-zeroer decline below (vLLM's zeroer assumes a layout this backend does not have;
    :meth:`_init_kv_zero_meta` explains why running it would corrupt rather than initialise).

    Everything else was read and left alone (``gpu_model_runner.py`` line numbers):

    * ``initialize_kv_cache`` ``:7284-7341`` - attention backends through
      ``current_platform.get_attn_backend_cls``, tensors through
      ``torch.zeros(..., device=self.device)`` (``:7014``).  Its ``initialize_mamba_ssu_backend``
      call (``:7301``) is a no-op for this model (GDN specs are not MAMBA1/MAMBA2, see
      :func:`require_supported_mamba_state_path`), not an unhandled device seam;
    * ``profile_run`` ``:6168-6241`` - ``_dummy_run`` + ``_sync_device`` (``torch.accelerator``);
    * ``_dummy_run`` ``:5599`` - takes the device from ``self.device``;
    * ``capture_model`` ``:6525`` - **not** device-agnostic (``torch.cuda.mem_get_info`` at
      ``:6547`` and ``:6565``), which is why :meth:`NPUWorker.capture_model` gates it off rather
      than relying on the wrapper's two bridged call sites alone;
    * ``torch.cuda.Stream()`` at ``:696`` and ``:1121`` - reached only when async scheduling is
      enabled (``self.use_async_scheduling = self.scheduler_config.async_scheduling``, ``:504``).
      ``:696`` is inline in ``__init__``, so *no* override can replace it: if
      ``--async-scheduling`` is on for this deployment, that line is the first failure and it must
      be turned off in the config (or ``torch.cuda`` aliased by torch_npu), not patched here.

    HARDWARE: the AI-core count.  ``NPU_CORE_COUNT_ATTRS`` lists the candidate attribute names on
    ``torch.npu.get_device_properties``; none is in the local tree.  ``num_sms`` has exactly one
    consumer - ``use_cascade_attention(..., num_sms=self.num_sms)``
    (``gpu_model_runner.py:2629``), a CUDA flash-attention heuristic that the TileInfer metadata
    builder answers ``False`` to - so a wrong value costs nothing today and a missing one must not
    abort start-up.  First diagnostic:
    ``python -c "import torch, torch_npu; print(torch.npu.get_device_properties(0))"``
    """

    def _init_device_properties(self) -> None:
        """Set ``self.num_sms`` from the device, without the platform ABC's raising method.

        ``:1109-1112`` reads ``num_compute_units(self.device.index)``; this reads the same
        information from ``torch.npu.get_device_properties``.  If none of
        :data:`NPU_CORE_COUNT_ATTRS` is present, ``num_sms`` is set to the sentinel ``1`` *and a
        warning names the attributes that were tried* - a silent ``1`` would look like a measured
        value in a log, which is exactly what this file is trying not to do.
        """
        _check_npu_device(self.device, "TileInferModelRunner._init_device_properties")
        npu = _npu_module("TileInferModelRunner._init_device_properties")
        index = getattr(self.device, "index", 0)
        index = 0 if index is None else int(index)
        properties = None
        get_properties = getattr(npu, "get_device_properties", None)
        if callable(get_properties):
            try:
                properties = get_properties(index)
            except (TypeError, RuntimeError) as exc:  # pragma: no cover - torch_npu variant
                logger.debug("torch.npu.get_device_properties(%d) failed: %r", index, exc)
        for attr in NPU_CORE_COUNT_ATTRS:
            value = getattr(properties, attr, None)
            if isinstance(value, int) and value > 0:
                self.num_sms = int(value)
                logger.info(
                    "TileInfer: num_sms=%d (torch.npu.get_device_properties.%s)", value, attr
                )
                return
        self.num_sms = 1
        logger.warning(
            "TileInfer: torch.npu.get_device_properties(%d) carries none of %s, so num_sms is the "
            "sentinel 1 and not a measurement. Its only consumer is use_cascade_attention "
            "(gpu_model_runner.py:2629), which the TileInfer attention builder answers False to - "
            "but if a log ever shows cascade attention being considered, this value is the cause. "
            "First diagnostic: python -c \"import torch, torch_npu; "
            "print(torch.npu.get_device_properties(0))\"",
            index,
            list(NPU_CORE_COUNT_ATTRS),
        )

    def _init_kv_zero_meta(self) -> None:
        """Decline vLLM's ``KVBlockZeroer``: on this pool it would zero the wrong bytes.

        What the engine does
        --------------------
        ``KVCacheConfig.needs_kv_cache_zeroing`` is just ``has_mamba_layers`` (vLLM
        ``v1/kv_cache_interface.py:873-879``), so every hybrid run - Qwen3.6 included - asks for
        it.  ``gpu_worker.initialize_from_config`` builds the bookkeeping right after the pool is
        allocated (``gpu_worker.py:583-589``), and ``GPUModelRunner._update_states`` calls it each
        step for freshly allocated attention blocks (``gpu_model_runner.py:1151-1154`` ->
        ``_zero_block_ids`` ``:1103-1106``).

        Why it cannot run on this K/V layout
        ------------------------------------
        ``KVBlockZeroer`` takes the per-block byte count from ``kv.stride(block_dim) *
        element_size`` (``v1/worker/utils.py:149-150``) and assumes block ``b`` begins at
        ``b * page_size_el`` of one contiguous segment (``:161-170``; kernel offset ``:74``).
        ``block_dim`` comes from the backend's shape probe (``:131-136`` ->
        ``v1/attention/backend.py:98-116``), i.e. 0 for the block-first shape this backend
        declares.  Under the physical order ``get_kv_cache_stride_order() == (1, 0, 2, 3, 4)``
        that stride is ``page * heads * dim`` - exactly **half a page** (``attn/backend.py:
        200-231``) - and the physical buffer is the two flat halves ``[K blocks][V blocks]``.
        The zeroer therefore clears the K half of the requested block and *never touches V*: no
        assertion, no error, just a silently different buffer from what vLLM believes it wrote.
        Option (a) - giving the zeroer a stride arrangement it can address - would mean
        interleaving K and V per block, the one physical order ``pool_sub_view`` and the paged
        kernel cannot read (``serving/buckets.py:262-305``, ``attn/backend.py:184-191``).

        What is lost by declining it
        ----------------------------
        The zeroer exists so a recycled attention block cannot leak stale data/NaN into attention
        or SSM (``gpu_model_runner.py:1151-1152``).  Mamba state is never zeroed by this class -
        it skips every non-``FullAttentionSpec`` group (``v1/worker/utils.py:108``,
        ``:125-126``) - so this is not the mamba-state-init path and disabling it cannot leave a
        GDN state block uninitialised.  For attention, ``TileInferAttentionImpl.forward`` writes
        every scheduled token slot before the kernel runs (``attn/impl.py:432-436``) and the
        paged kernel dereferences only the pages in ``kv_indptr``/``kv_indices``, so a freshly
        allocated block is not read before it is written.  The residual cost is defensive only:
        if a future path reads an unwritten attention page it will see stale data instead of
        zeros.  That is a device-run observation - ``HARDWARE_CHECKLIST`` keeps it open.

        Not constructing ``self._kv_block_zeroer`` is what makes the inherited
        ``_zero_block_ids`` a no-op (``gpu_model_runner.py:1105``), so no per-step call reaches a
        zeroer built for a layout this pool does not have.
        """
        logger.warning(
            "TileInfer: declining vLLM's KVBlockZeroer (needs_kv_cache_zeroing is True for a "
            "hybrid model). Its per-block size is stride(block_dim)*element_size "
            "(v1/worker/utils.py:149-150), which is half a page under this backend's K/V-outermost "
            "stride order (attn/backend.py:200-231): it would zero the K half of each new block "
            "and never the V half. TileInfer writes every scheduled slot before attention reads "
            "it and the paged kernel only dereferences kv_indptr/kv_indices, so fresh blocks are "
            "not read un-written; mamba state is never zeroed by this class anyway "
            "(v1/worker/utils.py:125-126). Residual cost: a future path that reads an unwritten "
            "attention page sees stale data instead of zeros - verify on the first device run "
            "(HARDWARE_CHECKLIST)."
        )


# =============================================================================================
# The worker
# =============================================================================================


class NPUWorker(Worker):
    """The Ascend NPU worker: the four hooks of ``vllm.v1.worker.gpu_worker.Worker``, on torch_npu.

    Instantiate order (``vllm/v1/worker/worker_base.py`` is not in the local subset; this is the
    order the four bodies assume): ``init_device`` -> ``load_model`` -> ``determine_available_memory``
    -> ``initialize_from_config`` -> ``compile_or_warm_up_model`` -> ``execute_model``.  Each hook
    below is therefore allowed to rely only on what the previous one set.

    Three of the four hooks reuse the GPU worker's body almost verbatim, because reading
    ``gpu_model_runner.py`` showed the *runner* is the device-agnostic half (``torch.zeros(...,
    device=self.device)``, ``current_platform``, ``torch.accelerator``).  What changes is the device
    selection, the memory accounting, and the two hooks whose bodies are CUDA-shaped end to end
    (``init_device``'s ``elif``, ``compile_or_warm_up_model``'s Triton warm-up + unconditional
    capture).
    """

    # ------------------------------------------------------------------ device init -----------

    def init_device(self) -> None:
        """Select the NPU, initialise the process group, snapshot memory, build the model runner.

        **What the GPU worker does** (``gpu_worker.py:249-345``): branches on
        ``self.device_config.device_type == "cuda"`` (``:251``) and raises
        ``RuntimeError("Not support device type")`` otherwise (``:319-320``); applies the
        DP-local-rank arithmetic (``:255-283``); builds ``torch.device(f"cuda:{local_rank}")`` and
        calls ``torch.accelerator.set_device_index`` (``:285-286``); checks the dtype against the
        platform (``:288``); initialises the distributed environment **before** the memory
        snapshot so NCCL buffers are already allocated (``:290-300``); seeds (``:306``); takes
        ``MemorySnapshot`` and computes ``requested_memory`` (``:309-318``); initialises the
        workspace manager (``:322-324``); constructs the runner (``:327-341``, ``GPUModelRunner``
        or the V2 one).

        **What this override does**: the same sequence with ``torch.npu`` in place of
        ``torch.accelerator``/``torch.cuda``, ``npu:<rank>`` in place of ``cuda:<rank>``,
        :class:`NpuMemorySnapshot` in place of ``MemorySnapshot``, and
        :class:`TileInferModelRunner` in place of ``GPUModelRunner``.  The DP-local-rank
        arithmetic and the workspace manager call are copied unchanged: they only touch
        ``parallel_config`` and the *current* device, both of which are device-neutral.

        HARDWARE: four things.

        1. ``torch.npu.set_device`` must accept a ``torch.device`` (the index form is retried
           once).  First diagnostic:
           ``python -c "import torch, torch_npu; torch.npu.set_device(torch.device('npu:0')); print(torch.npu.current_device())"``
        2. ``current_platform.dist_backend`` is ``""`` (``tileinfer/vllm/platform.py``
           ``UNVERIFIED_NOTES['dist_backend']``) while the GPU worker passes that value straight to
           ``init_distributed_environment`` (``:1186-1193``).  On one card the group may not be
           needed at all, and if it is, the backend name is ``hccl``.  First diagnostic:
           ``python -c "import torch.distributed as d; print(d.is_available(), d.is_hccl_available())"``
        3. the memory snapshot's two API spellings - see :class:`NpuMemorySnapshot`.
        4. ``vllm.utils.torch_utils.set_random_seed`` (``:306``) may call
           ``Platform.manual_seed_all``, which this platform leaves raising
           (``UNIMPLEMENTED_ABC_METHODS['manual_seed_all']``); :meth:`_set_random_seed` falls back
           to ``torch.manual_seed`` and warns when that happens.  First diagnostic:
           ``python -c "from vllm.utils.torch_utils import set_random_seed; set_random_seed(0)"``
        """
        # torch_npu first: every step below needs the device, and this is the one error a CPU box
        # can produce, so it must be the error the user sees.
        npu = _npu_module("init_device")

        declared = getattr(self.device_config, "device_type", None)
        if declared not in NPU_DEVICE_TYPES:
            raise RuntimeError(
                f"NPUWorker.init_device was handed device_config.device_type={declared!r}; the "
                "inherited body accepts only 'cuda' (gpu_worker.py:251) and raises otherwise "
                f"(``:319-320``), and this override serves {list(NPU_DEVICE_TYPES)}. The value "
                "comes from TileInferPlatform.device_type ('npu', tileinfer/vllm/platform.py:819); "
                "a different one means the TileInfer platform was not the selected platform - "
                "check `python -c \"from vllm.platforms import current_platform; "
                "print(type(current_platform).__name__, current_platform.device_type)\"`."
            )

        # --- DP-local-rank arithmetic, copied from gpu_worker.py:255-283 ---------------------
        parallel_config = self.parallel_config
        if (
            parallel_config.distributed_executor_backend not in ("ray", "external_launcher")
            and parallel_config.data_parallel_backend != "ray"
            and parallel_config.nnodes_within_dp == 1
        ):
            dp_local_rank = parallel_config.data_parallel_rank_local
            if dp_local_rank is None:
                dp_local_rank = parallel_config.data_parallel_index
            tp_pp_world_size = (
                parallel_config.pipeline_parallel_size * parallel_config.tensor_parallel_size
            )
            # DP_LOCAL_RANK * TP_PP_WORLD_SIZE + TP_LOCAL_RANK
            self.local_rank += dp_local_rank * tp_pp_world_size
            # gpu_worker.py:273-283 asks torch.accelerator; the NPU spelling is torch.npu.
            visible_device_count = int(npu.device_count())
            assert self.local_rank < visible_device_count, (
                f"DP adjusted local rank {self.local_rank} is out of bounds for "
                f"{visible_device_count} visible NPU device(s) (gpu_worker.py:273-275)."
            )
            assert parallel_config.local_world_size <= visible_device_count, (
                f"local_world_size ({parallel_config.local_world_size}) must be less than or equal "
                f"to the number of visible NPU devices ({visible_device_count}) "
                "(gpu_worker.py:279-283)."
            )

        self.device = _npu_device(self.local_rank)
        spelling = _set_npu_device(npu, self.device, self.local_rank)
        logger.info("TileInfer worker device=%s (%s)", self.device, spelling)

        from vllm.platforms import current_platform

        current_platform.check_if_supports_dtype(self.model_config.dtype)

        # Distributed environment BEFORE the snapshot, so collective buffers are already counted
        # (gpu_worker.py:290-300).
        # HARDWARE: the backend name passed below is current_platform.dist_backend, which is ''
        # (UNVERIFIED_NOTES['dist_backend']); HCCL must be named there, or the group skipped.
        from vllm.v1.worker.gpu_worker import init_worker_distributed_environment

        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )

        self._set_random_seed(self.model_config.seed)

        # Now take the snapshot, after the process group (gpu_worker.py:309-318).
        gc.collect()
        npu.empty_cache()  # gpu_worker.py:310 uses torch.accelerator.empty_cache
        self.init_snapshot = _npu_memory_snapshot(self.device, npu=npu)
        self.requested_memory = _requested_memory(self.init_snapshot, self.cache_config)
        logger.info(
            "worker init memory snapshot: free=%s total=%s torch_peak=%s; requested=%s",
            _format_gib(self.init_snapshot.free_memory),
            _format_gib(self.init_snapshot.total_memory),
            _format_gib(self.init_snapshot.torch_peak),
            _format_gib(self.requested_memory),
        )

        # Initialize workspace manager (gpu_worker.py:322-324); device-neutral in principle.
        from vllm.v1.worker.workspace import init_workspace_manager

        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1
        init_workspace_manager(self.device, num_ubatches)

        # The model runner (gpu_worker.py:327-341).  V2 is a different runner file
        # (vllm/v1/worker/gpu/model_runner.py) that is not in the local subset and has no
        # TileInfer subclass, so it is refused rather than silently replaced.
        if self.use_v2_model_runner:
            raise NotImplementedError(
                "vllm_config.use_v2_model_runner is set, but TileInfer only implements the V1 "
                "runner (vllm.v1.worker.gpu_model_runner.GPUModelRunner, gpu_worker.py:337-341); "
                "the V2 runner lives in vllm/v1/worker/gpu/model_runner.py, which is not in "
                "runtime/vllm-api/ and has no subclass here. Run without --use-v2-model-runner, "
                "or add a TileInfer V2 runner and construct it here."
            )
        self.model_runner = TileInferModelRunner(self.vllm_config, self.device)

        if self.rank == 0:
            # Usage telemetry, exactly as gpu_worker.py:343-345 (best-effort by vLLM's design).
            from vllm.v1.utils import report_usage_stats

            report_usage_stats(self.vllm_config)

    # ---------------------------------------------------------------- small helpers -----------

    def _set_random_seed(self, seed: Any) -> None:
        """``set_random_seed`` (``gpu_worker.py:306`` / ``:732``), with the platform gap logged.

        ``vllm/utils/torch_utils.py`` is not in the local subset, so whether it reaches
        ``Platform.manual_seed_all`` cannot be read here.  ``TileInferPlatform`` leaves
        ``manual_seed_all`` raising (``UNIMPLEMENTED_ABC_METHODS``, reason ``must override``), and
        seeding must not be the reason an NPU run dies - so the ``NotImplementedError`` is turned
        into one warning plus ``torch.manual_seed``, which is the part that is known to work.

        HARDWARE: whether the platform hook is what vLLM calls, and whether
        ``torch.npu.manual_seed_all`` is needed for reproducibility.  First diagnostic:
        ``python -c "import torch, torch_npu; torch.npu.manual_seed_all(0); print('ok')"``
        """
        from vllm.utils.torch_utils import set_random_seed

        try:
            set_random_seed(seed)
        except NotImplementedError as exc:
            logger.warning(
                "TileInfer: vLLM's set_random_seed raised NotImplementedError (%s) - this is "
                "TileInferPlatform.manual_seed_all, which platform.py deliberately leaves "
                "unimplemented (interface.py:422-425). Falling back to torch.manual_seed(%r): the "
                "Python-side RNG is seeded, a device-side seed may not be. Implement "
                "manual_seed_all with torch.npu.manual_seed_all to close this.",
                exc,
                seed,
            )
            torch.manual_seed(int(seed))

    def _capture_stats(self) -> dict:
        """The capture counters (created lazily so ``object.__new__`` instances work in tests).

        Counters rather than silence: ``compile_or_warm_up_model`` is reached on every start, and a
        capture that did not happen must be visible in a log line *and* in a number a probe can
        read - otherwise "eager, and by design" looks identical to "capture silently failed".
        """
        stats = self.__dict__.get("capture_stats")
        if stats is None:
            stats = {"attempts": 0, "captured": 0, "skipped": 0, "last_skip_reason": ""}
            self.capture_stats = stats
        return stats

    # ---------------------------------------------------------------- decode-plan pre-warm -----

    def _prewarm_stats(self) -> dict:
        """The pre-warm counters plus the derived plan (created lazily; keys are pinned as data).

        Same shape as :meth:`_capture_stats`: a pre-warm that did not happen must be visible in a
        *number* a probe can read, not only in a log line (otherwise "0 plans, by design" and
        "0 plans, silently skipped" look identical).
        """
        stats = self.__dict__.get("prewarm_stats")
        if stats is None:
            stats = {key: 0 for key in PREWARM_COUNTER_KEYS}
            stats["skipped_reason"] = ""
            stats["bucket_set"] = ()
            stats["fitted_bucket_set"] = ()
            for key in ("width", "engine_width", "block_size", "max_num_reqs", "pool_cap"):
                stats[key] = 0
            self.prewarm_stats = stats
        return stats

    def _prewarm_skip(self, stats: dict, reason: str) -> int:
        """Record and log why the pre-warm did not run; returns 0 so the caller can ``return``."""
        stats["skipped_reason"] = str(reason)
        logger.warning(
            "TileInfer: decode-plan pre-warm skipped: %s. Serving still works, but the first "
            "decode step of each bucket would compile its plan lazily - which must not happen "
            "inside a captured graph, so graph capture has to stay off until this is fixed.",
            reason,
        )
        return 0

    def _prewarm_plan_spec(self) -> dict:
        """Derive the decode bucket set and the plan width from the engine's own config.

        Two independent halves, and both are needed:

        * the **batch bound** comes from ``scheduler_config.max_num_seqs`` *and*
          ``scheduler_config.max_num_batched_tokens``: a decode step schedules one token per
          request, so the token budget bounds the reachable request count too
          (``min`` of the two - the same bound the sampler warm-up in
          ``compile_or_warm_up_model`` uses).  ``serving/planner.buckets_up_to`` turns that count
          into the bucket set.
        * the **page-table width** comes from ``model_config.max_model_len`` and
          ``cache_config.block_size``: one request owns ``ceil(max_model_len / block_size)``
          engine pages, each of which is ``sub_pages(block_size)`` 128-token kernel pages
          (``serving/planner.plan_width``).

        Raises (never swallows) when a field is missing or nonsensical; the caller turns that into
        a logged skip, because start-up must not die over an optional warm-up.
        """
        from tileinfer.serving.buckets import sub_pages
        from tileinfer.serving.planner import buckets_up_to, plan_width

        scheduler_config = getattr(self, "scheduler_config", None)
        cache_config = getattr(self, "cache_config", None)
        model_config = getattr(self, "model_config", None)
        missing = [
            name
            for name, value in (
                ("scheduler_config", scheduler_config),
                ("cache_config", cache_config),
                ("model_config", model_config),
            )
            if value is None
        ]
        if missing:
            raise RuntimeError(
                f"the worker has no {', '.join(missing)}; the decode bucket set cannot be derived"
            )

        max_model_len = int(getattr(model_config, "max_model_len", 0) or 0)
        if max_model_len <= 0:
            raise ValueError(
                f"model_config.max_model_len is {max_model_len!r}; the block-table width (and so "
                "every plan's staging buffers) cannot be sized without it"
            )
        block_size = int(getattr(cache_config, "block_size", 0) or 0)
        if block_size <= 0:
            raise ValueError(
                f"cache_config.block_size is {block_size!r}; the engine page size is what the "
                "kernel pages are derived from"
            )
        max_num_seqs = int(getattr(scheduler_config, "max_num_seqs", 0) or 0)
        if max_num_seqs <= 0:
            raise ValueError(f"scheduler_config.max_num_seqs is {max_num_seqs!r}")
        token_budget = int(getattr(scheduler_config, "max_num_batched_tokens", 0) or 0)
        if token_budget <= 0:  # an unset budget cannot be tighter than max_num_seqs
            token_budget = max_num_seqs
        max_num_reqs = max(1, min(max_num_seqs, token_budget))

        sub = sub_pages(block_size)  # raises for a page that is not k * 128 tokens
        width = plan_width(max_model_len, block_size)
        return {
            "buckets": buckets_up_to(max_num_reqs),
            "width": width,
            "engine_width": width // sub,
            "block_size": block_size,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": token_budget,
            "max_num_reqs": max_num_reqs,
        }

    def _tileinfer_attention_layers(self) -> list:
        """``(name, module, impl)`` for every TileInfer attention layer the runner holds.

        The engine builds one backend + impl per attention layer and binds the layer's fused NHD
        pool onto the module (``GPUModelRunner.initialize_kv_cache_tensors`` ->
        ``bind_kv_cache``, ``gpu_model_runner.py:7248-7253``), which is why
        ``compilation_config.static_forward_context`` is the place both are reachable from.
        The 30 GDN/mamba layers are skipped: they have no ``impl`` of ours (their state is the
        model's own, ``docs/plan-vllm-native-backend.md`` §5).
        """
        runner = getattr(self, "model_runner", None)
        compilation_config = getattr(runner, "compilation_config", None)
        if compilation_config is None:
            compilation_config = getattr(self, "compilation_config", None)
        if compilation_config is None:
            compilation_config = getattr(
                getattr(self, "vllm_config", None), "compilation_config", None
            )
        context = getattr(compilation_config, "static_forward_context", None)
        if not context:
            return []

        from tileinfer.vllm.attn.impl import TileInferAttentionImpl

        layers = []
        for name, module in dict(context).items():
            impl = getattr(module, "impl", None)
            if isinstance(impl, TileInferAttentionImpl):
                layers.append((str(name), module, impl))
        return layers

    def _prewarm_layer_pool(self, module: Any, spec: dict, sub: int) -> dict:
        """The kernel-page views of one layer's KV pool - or a shape-only stand-in for them.

        **Ordering constraint.**  This hook only runs in ``compile_or_warm_up_model``, i.e. *after*
        ``initialize_from_config``: the engine has allocated the pool and bound each attention
        layer's fused ``[blocks, 2, block_size, kv_heads, head_size]`` tensor onto the layer
        (``gpu_model_runner.py:7248-7253``; the block-first shape comes from
        ``TileInferAttentionBackend.get_kv_cache_shape`` and the physical ``[2][blocks]`` order from
        ``get_kv_cache_stride_order``).  Using that tensor is what makes the prepared state survive
        the first real step: ``TileInferAttentionImpl._state_for`` rejects a pool whose
        ``data_ptr`` changed between steps (``attn/impl.py:641-652``).

        **Both pool orders are recognised, never guessed.**  ``serving.buckets.fused_halves``
        splits the fused tensor by its declared block axis; for this backend that is dim 0, so the
        K half is ``pool[:, 0]`` (a contiguous ``[blocks, page, heads, dim]`` array because the
        physical layout is K/V-outermost).  The legacy ``(2, blocks, ...)`` form - which a vLLM
        point release could still hand us - is split at ``pool[0]`` instead.  When both dims are 2
        the shape cannot tell the orders apart, so the backend's own ``get_kv_cache_block_dim``
        answers (guessing would silently swap K and V; ``fused_halves`` explains why it refuses).

        **Shape-only path.**  A layer with no bound pool (a unit test, or a vLLM point release that
        renames ``layer.kv_cache``) gets a placeholder of the declared page count instead.  The
        placeholder is never read: the planner only asks the cache tensors for ``shape[0]`` when it
        sizes the kernel (``attention/backends/tilelang_ascend950.py:109-123``), and
        ``serving/buckets.declared_pool_cap`` rounds that to a power of two.  Because the plan key
        contains that *declared* capacity (not the tensor), the plan built from the placeholder is
        the same plan the real pool hits later.  The caller drops the synthetic state afterwards
        (``_prewarm_attention_layer``), so the first real step binds the layer's own pool instead of
        tripping the ``data_ptr`` check; the plan cache hit is what was pre-warmed.
        """
        from tileinfer.serving.buckets import fused_halves, pool_sub_view

        pool = getattr(module, "kv_cache", None)
        if isinstance(pool, torch.Tensor) and pool.dim() == 5:
            # Both orders keep the engine page size at dim 2; only the K/V axis moves.
            block_size = int(pool.shape[2])
            if block_size != int(spec["block_size"]):
                raise ValueError(
                    f"the layer's KV pool has a {block_size}-token page but "
                    f"cache_config.block_size is {spec['block_size']}; the plan would be sized for "
                    "a page the engine does not use"
                )
            block_dim = None
            if int(pool.shape[0]) == 2 and int(pool.shape[1]) == 2:
                # Two blocks: indistinguishable from the shape, and a wrong pick swaps K and V.
                # The declared layout (not a second shape probe) answers.
                block_dim = int(active_profile().kv_shape.block_dim)
            k_half, v_half = fused_halves(pool, block_dim)
            k_pool = pool_sub_view(k_half, sub)
            v_pool = pool_sub_view(v_half, sub)
            return {
                "k_pool": k_pool,
                "v_pool": v_pool,
                "pages": int(k_pool.shape[0]),
                "dtype": pool.dtype,
                "device": pool.device,
                "synthetic": False,
            }

        cache_config = getattr(self, "cache_config", None)
        num_blocks = int(getattr(cache_config, "num_gpu_blocks", 0) or 0)
        if num_blocks <= 0:
            raise ValueError(
                "no KV pool is bound to this layer and cache_config.num_gpu_blocks is unset, so "
                "the declared page capacity is unknown"
            )
        device = getattr(self, "device", None)
        dtype = getattr(getattr(self, "model_config", None), "dtype", None)
        if device is None or not isinstance(dtype, torch.dtype):
            raise ValueError(
                f"the shape-only path needs a device and a torch dtype (got device={device!r}, "
                f"dtype={dtype!r})"
            )
        pages = num_blocks * sub
        # One element per page: only shape[0] is read (see this method's docstring).
        k_pool = torch.empty((pages, 1, 1, 1), dtype=dtype, device=device)
        v_pool = torch.empty((pages, 1, 1, 1), dtype=dtype, device=device)
        return {
            "k_pool": k_pool,
            "v_pool": v_pool,
            "pages": pages,
            "dtype": dtype,
            "device": device,
            "synthetic": True,
        }

    def _prewarm_attention_layer(
        self, name: str, module: Any, impl: Any, spec: dict
    ) -> Tuple[int, int, int, Tuple[int, ...]]:
        """Prewarm one layer's reachable buckets.

        Returns ``(newly built plans, failures, pool cap, buckets that fit)``.

        The work is driven entirely through the impl's own path - ``_state_for`` to allocate the
        per-bucket scratch (which capture must never do) and ``prewarm`` to build the plans in
        ``attn.impl.PLAN_CACHE`` with ``_plan_key``/``_build_plan`` (``attn/impl.py:499-543,
        715-769``).  There is no second plan-key implementation here.
        """
        from tileinfer.serving.buckets import declared_pool_cap, sub_pages
        from tileinfer.vllm.attn.impl import PLAN_CACHE
        from tileinfer.vllm.attn.metadata import TileInferMetadata

        sub = sub_pages(int(spec["block_size"]))
        pool = self._prewarm_layer_pool(module, spec, sub)
        cap = declared_pool_cap(int(pool["pages"]))
        width = int(spec["width"])
        # `_state_for` needs one trash slot past every staging destination
        # (bucket * width < cap - 1; attn/impl.py:662-674).  Trimming here keeps the pre-warm from
        # logging a failure for a bucket the first step would reject anyway.
        fit = (cap - 2) // width if width > 0 else 0
        usable = tuple(bucket for bucket in spec["buckets"] if bucket <= fit)
        too_large = tuple(bucket for bucket in spec["buckets"] if bucket > fit)
        if too_large:
            logger.warning(
                "TileInfer: decode bucket(s) %s of layer %s do not fit its KV pool (%d kernel "
                "pages, declared cap %d, table width %d); the impl's staging table requires "
                "bucket*width < cap - 1 (attn/impl.py:_state_for). They are not pre-warmed - a "
                "batch that reaches one would raise at its first step. Lower --max-model-len / "
                "--max-num-seqs, or give the model more KV cache.",
                list(too_large),
                name,
                int(pool["pages"]),
                cap,
                width,
            )
        if not usable:
            return 0, 0, cap, ()

        device = pool["device"]
        dtype = pool["dtype"]
        k_pool = pool["k_pool"]
        # Shape-only for the query staging: `_state_for` reads dtype/device, never the rows.
        q3 = torch.empty((0, int(impl.num_heads), int(impl.head_size)), dtype=dtype, device=device)
        engine_width = int(spec["engine_width"])
        failures = 0
        for bucket in usable:
            state = impl._states.get(bucket)
            if state is not None:
                # Idempotent when it is the same pool, and self-healing when it is not: vLLM's
                # profiling pass builds a throwaway pool and then clears it
                # (``_cleanup_profiling_kv_cache``, gpu_model_runner.py:6305-6324), so a state can
                # outlive the pool it was built from.  A stale state would either raise in
                # ``_state_for`` or key a plan the real step never looks up; drop it and rebuild.
                if (
                    state.k_pool.data_ptr() == k_pool.data_ptr()
                    and int(state.width) == int(spec["width"])
                ):
                    continue
                logger.info(
                    "TileInfer: bucket %d of layer %s holds a state from a different KV pool "
                    "(width %d, %d pages; this step needs width %d, %d pages) - dropping it and "
                    "rebuilding from the pool bound now.",
                    bucket,
                    name,
                    int(state.width),
                    int(state.k_pool.shape[0]),
                    int(spec["width"]),
                    int(k_pool.shape[0]),
                )
                impl._states.pop(bucket, None)
            meta = TileInferMetadata(
                query_start_loc=torch.arange(bucket + 1, dtype=torch.int32, device=device),
                query_start_loc_cpu=torch.arange(bucket + 1, dtype=torch.int32),
                seq_lens=torch.ones(bucket, dtype=torch.int32, device=device),
                # `_state_for` derives the kernel-table width from this shape, and every row must
                # look like one 1-token request so the synthetic schedule is the decode one.
                block_table_tensor=torch.zeros(
                    (bucket, engine_width), dtype=torch.int32, device=device
                ),
                slot_mapping=torch.arange(bucket, dtype=torch.int64, device=device),
                num_reqs=bucket,
                num_actual_tokens=bucket,
                max_query_len=1,
                max_seq_len=1,
                bucket=int(bucket),
                block_size=int(spec["block_size"]),
            )
            try:
                impl._state_for(meta, pool["k_pool"], pool["v_pool"], int(spec["block_size"]), q3)
            except Exception as exc:  # noqa: BLE001 - one bucket must not stop the others
                failures += 1
                logger.warning(
                    "TileInfer: cannot prepare decode bucket %d for layer %s (%r). Its plan will "
                    "compile lazily on the first step, so capture must stay off; the rest of the "
                    "pre-warm continues.",
                    bucket,
                    name,
                    exc,
                )

        before = PLAN_CACHE.failure_count()
        built = int(impl.prewarm(usable))
        failures += max(0, PLAN_CACHE.failure_count() - before)

        if pool["synthetic"]:
            for bucket in usable:
                impl._states.pop(bucket, None)
            logger.info(
                "TileInfer: layer %s had no bound KV pool, so its bucket state(s) were prepared "
                "from the declared shape only and dropped again; the plan(s) stay in the cache "
                "(the key uses the declared capacity), and the first real step binds the layer's "
                "own pool without a data_ptr mismatch.",
                name,
            )
        return built, failures, cap, usable

    def _prewarm_decode_plans(self) -> int:
        """Build every reachable decode plan **before** graph capture; returns newly built.

        W8's contract (``docs/plan-vllm-native-backend.md`` §6) is that a captured region may only
        *look plans up*: no compile, no allocation, no host sync.  This method is the synchronous
        half of that contract and runs at the one point where all of it is legal - after
        ``initialize_from_config`` (KV pool allocated and bound to the layers) and before
        ``capture_model()``.

        What it derives, and from where
        -------------------------------
        * buckets: ``min(scheduler_config.max_num_seqs, max_num_batched_tokens)`` through
          ``serving/planner.buckets_up_to`` (a decode step is one token per request);
        * the block-table width: ``serving/planner.plan_width(max_model_len, block_size)``;
        * the pool capacity: the layer's own pool (``declared_pool_cap`` rounds it to a power of
          two, so the engine's run-to-run pool drift does not invalidate the plan).

        Properties this method guarantees
        ---------------------------------
        * **Idempotent.**  Existing states are reused and ``PLAN_CACHE`` skips cached plans, so a
          second call builds nothing (``prewarm_stats["plans"]`` is cumulative).
        * **Never fatal.**  A missing device (no ``torch_npu``/TileLang, or a CPU box), a missing
          pool, a config field that cannot be derived, or one bucket that fails to compile is a
          logged skip/counter, not an exception: a warm-up must not turn a bootable engine into a
          traceback.  Per-bucket failures are counted in ``prewarm_stats["failed"]`` and logged
          with their traceback by ``impl.prewarm`` - nothing is swallowed silently.
        * **The impl's path, not a copy.**  The plan is built by
          ``TileInferAttentionImpl._build_plan`` under ``PLAN_CACHE`` with the impl's own
          ``_plan_key`` (``attn/impl.py``); this module only decides *which* buckets and supplies
          the pool geometry.  ``impl.py`` has no separate "shape-only prewarm" entry point - its
          plan key is derived from a ``_BucketState``'s geometry - so the shape-only mode is
          realised by feeding ``_state_for`` tensors that carry only the declared page capacity and
          then dropping the synthetic state (``_prewarm_layer_pool`` documents why that is the same
          plan later).
        """
        stats = self._prewarm_stats()
        stats["attempts"] += 1
        try:
            spec = self._prewarm_plan_spec()
        except Exception as exc:  # noqa: BLE001 - a warm-up must not abort the boot
            return self._prewarm_skip(
                stats, f"the decode bucket set cannot be derived from the server config ({exc!r})"
            )
        stats["buckets"] = len(spec["buckets"])
        stats["bucket_set"] = tuple(spec["buckets"])
        stats["width"] = int(spec["width"])
        stats["engine_width"] = int(spec["engine_width"])
        stats["block_size"] = int(spec["block_size"])
        stats["max_num_reqs"] = int(spec["max_num_reqs"])

        try:
            layers = self._tileinfer_attention_layers()
        except Exception as exc:  # noqa: BLE001
            return self._prewarm_skip(stats, f"the attention layers cannot be discovered ({exc!r})")
        if not layers:
            return self._prewarm_skip(
                stats,
                "no TileInfer attention layer is reachable through "
                "model_runner.compilation_config.static_forward_context (the model is not loaded, "
                "or it has no GQA/MHA layer served by this backend)",
            )

        first_impl = layers[0][2]
        for method in ("_state_for", "prewarm", "_decode_attn"):
            if not callable(getattr(first_impl, method, None)):
                return self._prewarm_skip(
                    stats,
                    f"the attention impl no longer exposes {method}(); the pre-warm drives the "
                    "impl's own state/plan path (tileinfer/vllm/attn/impl.py) and refuses to "
                    "build a second one",
                )
        try:
            backend = first_impl._decode_attn()  # noqa: SLF001 - the impl's own availability probe
        except Exception as exc:  # noqa: BLE001
            return self._prewarm_skip(stats, f"the decode backend probe raised {exc!r}")
        if backend is None:
            return self._prewarm_skip(
                stats,
                "the TileInfer decode kernel backend is unavailable on this device (no Ascend "
                "device, or no TileLang toolchain), so there is nothing to compile against",
            )

        cache_config = getattr(self, "cache_config", None)
        pool_ready = bool(getattr(cache_config, "num_gpu_blocks", 0)) or any(
            isinstance(getattr(module, "kv_cache", None), torch.Tensor)
            for _name, module, _impl in layers
        )
        if not pool_ready:
            return self._prewarm_skip(
                stats,
                "the KV pool has not been allocated yet (cache_config.num_gpu_blocks is unset and "
                "no layer has a bound pool), so the declared page capacity - and with it every "
                "plan - cannot be sized; this pre-warm must run after initialize_from_config",
            )

        newly_built = 0
        fitted: Optional[set] = None
        for name, module, impl in layers:
            try:
                built, failures, cap, usable = self._prewarm_attention_layer(
                    name, module, impl, spec
                )
            except Exception:  # noqa: BLE001 - one layer must not stop the remaining nine
                stats["failed"] += 1
                logger.exception(
                    "TileInfer: decode-plan pre-warm failed for layer %s; continuing with the "
                    "remaining layer(s). This layer's plans will compile lazily on its first step.",
                    name,
                )
                continue
            stats["layers"] += 1
            stats["plans"] += built
            stats["failed"] += failures
            stats["pool_cap"] = max(int(stats["pool_cap"]), int(cap))
            newly_built += built
            fitted = set(usable) if fitted is None else fitted & set(usable)

        from tileinfer.vllm.attn.impl import PLAN_CACHE

        cache = PLAN_CACHE.snapshot()
        fitted_set = tuple(sorted(fitted or ()))
        stats["fitted_bucket_set"] = fitted_set
        if newly_built:
            stats["skipped_reason"] = ""
            logger.info(
                "TileInfer: pre-warmed %d decode plan(s) for bucket(s) %s (derived %s; width=%d "
                "kernel pages, engine_width=%d, block_size=%d) across %d attention layer(s); "
                "plan cache now: %d ready, %d failed. Graph capture runs next and may only look "
                "plans up.",
                newly_built,
                list(fitted_set) or "<none>",
                list(spec["buckets"]),
                int(spec["width"]),
                int(spec["engine_width"]),
                int(spec["block_size"]),
                stats["layers"],
                cache.get("ready_plans", 0),
                cache.get("failed", 0),
            )
        else:
            logger.info(
                "TileInfer: decode plans for bucket(s) %s are already warm (%d in the plan cache, "
                "%d failed); nothing was rebuilt, so this call is idempotent.",
                list(fitted_set) or "<none>",
                cache.get("ready_plans", 0),
                cache.get("failed", 0),
            )
        if int(stats["plans"]) == 0:
            return self._prewarm_skip(
                stats,
                "no decode plan could be built (see the per-layer / per-bucket warnings above); "
                "the serving step will compile lazily and must not be captured",
            )
        stats["skipped_reason"] = ""
        return newly_built

    # ------------------------------------------------------------------ memory profiling ------

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Return the bytes that may be used for the KV cache, measured with torch_npu.

        **What the GPU worker does** (``gpu_worker.py:371-524``): short-circuits to
        ``cache_config.kv_cache_memory_bytes`` after a profile run (``:384-402``); otherwise runs
        ``memory_profiling`` around ``model_runner.profile_run()`` (``:406-410``), reads the peak
        allocated bytes from ``torch.accelerator.memory_stats`` (``:412-414``), optionally adds
        ``profile_cudagraph_memory()`` (``:416-425``), recomputes
        ``torch_peak_increase``/``non_kv_cache_memory`` (``:428-435``), asserts that the initial
        free memory did not *grow* during profiling (``:452-460``), and returns
        ``requested_memory - non_kv_cache_memory - cudagraph_estimate`` (``:461-465``).

        **What this override does**: the same shape, with :class:`NpuMemorySnapshot` for the
        snapshot and :func:`_npu_peak_allocated` for the peak.  ``memory_profiling`` is not in the
        local subset, so its two derived quantities are recomputed here from the definitions the
        worker's own uses imply (``:429`` ``torch_peak``, ``:445-446``):

        * ``torch_peak_increase = peak_after - before.torch_peak``;
        * ``non_torch_increase = (before.free - before.torch_peak) - (after.free - after.torch_peak)``
          - free memory that disappeared *without* the torch allocator accounting for it
          (contexts, collectives, other processes);
        * ``non_kv_cache_memory = non_torch_increase + torch_peak_increase + weights_memory``,
          which is ``:431-435`` verbatim.

        HARDWARE: three quantities, each with its own trap.

        1. ``torch.npu.mem_get_info`` / ``torch.npu.memory.mem_get_info`` - which spelling exists
           (the helper names both).  First diagnostic:
           ``python -c "import torch, torch_npu; print(torch.npu.mem_get_info(0))"``
        2. ``allocated_bytes.all.peak`` vs ``max_memory_allocated``: the arithmetic subtracts a
           *process* high-water mark from a *device* free count, so the two must be the same
           quantity.  First diagnostic:
           ``python -c "import torch, torch_npu; print(torch.npu.memory_stats(0)['allocated_bytes.all.peak'], torch.npu.max_memory_allocated(0))"``
        3. the CUDA-graph term is **0 here** because capture is gated off
           (:meth:`_cudagraph_memory_estimate`), i.e. this is the ROCm situation the comment at
           ``:437-438`` describes: the KV pool is sized as if no graph memory were needed.  When
           W8 lands, that estimate must be measured, or the first graph capture will OOM.
        """
        kv_cache_memory_bytes = getattr(self.cache_config, "kv_cache_memory_bytes", None)
        if kv_cache_memory_bytes:
            # Still profile (it compiles the model for max_num_batched_tokens), exactly as
            # gpu_worker.py:384-402, and do not touch the memory APIs at all.
            self.model_runner.profile_run()
            logger.info(
                "Initial free memory %s; reserved %s for the KV cache as specified by "
                "cache_config.kv_cache_memory_bytes - memory profiling skipped. This ignores "
                "gpu_memory_utilization (gpu_worker.py:389-401).",
                _format_gib(self.init_snapshot.free_memory),
                _format_gib(int(kv_cache_memory_bytes)),
            )
            return int(kv_cache_memory_bytes)

        npu = _npu_module("determine_available_memory")
        init_snapshot = getattr(self, "init_snapshot", None)
        if init_snapshot is None:  # pragma: no cover - init_device always sets it
            init_snapshot = _npu_memory_snapshot(self.device, npu=npu)
            self.init_snapshot = init_snapshot
        requested_memory = getattr(self, "requested_memory", None)
        if requested_memory is None:  # pragma: no cover - init_device always sets it
            requested_memory = _requested_memory(init_snapshot, self.cache_config)
            self.requested_memory = requested_memory

        # Execute a forward pass with dummy inputs to profile the memory usage of the model
        # (gpu_worker.py:404-410).
        before_profile = _npu_memory_snapshot(self.device, npu=npu)
        self.model_runner.profile_run()
        # gpu_worker.py:412-414 reads this *inside* memory_profiling, before empty_cache.
        profile_torch_peak = _npu_peak_allocated(npu, self.device)
        after_profile = _npu_memory_snapshot(self.device, npu=npu)

        torch_peak_increase = int(profile_torch_peak) - int(before_profile.torch_peak)
        non_torch_increase = (
            before_profile.free_memory
            - before_profile.torch_peak
            - (after_profile.free_memory - after_profile.torch_peak)
        )
        weights_memory = int(getattr(self.model_runner, "model_memory_usage", 0) or 0)
        non_kv_cache_memory = non_torch_increase + torch_peak_increase + weights_memory

        cudagraph_memory_estimate = self._cudagraph_memory_estimate()

        free_device_memory = after_profile.free_memory
        # NOTE(woosuk) at gpu_worker.py:450-460: this assumes other processes did not change their
        # usage during profiling.  The assertion is as correct on NPU as on CUDA.
        assert init_snapshot.free_memory >= free_device_memory, (
            "Error in memory profiling. "
            f"Initial free memory {_format_gib(init_snapshot.free_memory)}, current free memory "
            f"{_format_gib(free_device_memory)}. This happens when other processes sharing the "
            "same device release device memory while vLLM is profiling during initialization. "
            "To fix this, ensure consistent NPU memory allocation or isolate vLLM in its own "
            "container (gpu_worker.py:450-460)."
        )

        self.non_torch_memory = non_torch_increase
        self.peak_activation_memory = torch_peak_increase
        self.cudagraph_memory_estimate = cudagraph_memory_estimate
        self.available_kv_cache_memory_bytes = (
            requested_memory - non_kv_cache_memory - cudagraph_memory_estimate
        )

        logger.info(
            "Initial free memory: %s; requested memory: %s (util %.4f); free after profiling: %s",
            _format_gib(init_snapshot.free_memory),
            _format_gib(requested_memory),
            float(getattr(self.cache_config, "gpu_memory_utilization", 0.0)),
            _format_gib(free_device_memory),
        )
        logger.info(
            "Memory accounting: weights %s, peak activation %s, non-torch %s, cudagraph %s -> "
            "available KV cache %s",
            _format_gib(weights_memory),
            _format_gib(torch_peak_increase),
            _format_gib(non_torch_increase),
            _format_gib(cudagraph_memory_estimate),
            _format_gib(self.available_kv_cache_memory_bytes),
        )
        if self.available_kv_cache_memory_bytes <= 0:
            logger.warning(
                "TileInfer: available KV cache memory is %s (<= 0). The pool will be empty and "
                "the block manager will refuse to schedule; lower --gpu-memory-utilization "
                "expectations or raise it, and check the accounting above against `npu-smi info`.",
                _format_gib(self.available_kv_cache_memory_bytes),
            )
        return int(self.available_kv_cache_memory_bytes)

    def _cudagraph_memory_estimate(self) -> int:
        """Graph-pool bytes to reserve: 0 while capture is gated off (``:416-425``).

        The GPU worker asks ``profile_cudagraph_memory()`` only on CUDA (``:420-425``) and treats
        0 as "ROCm, no-op" (``:437-438``).  On NPU it is 0 for a different, *stated* reason: graph
        capture is not ready (:func:`tileinfer.vllm.graphs.capture_path_ready`), so no graph pool
        will be allocated and reserving for it would only shrink the KV cache.
        """
        ready, reason = graphs.capture_path_ready()
        if not ready:
            logger.info(
                "TileInfer: not reserving CUDA-graph memory in the KV-pool accounting (%s) "
                "(gpu_worker.py:416-425 is the CUDA path).",
                reason,
            )
            return 0
        profile = getattr(self.model_runner, "profile_cudagraph_memory", None)
        if not callable(profile):  # pragma: no cover - the runner always has it (gpu_model_runner)
            return 0
        # HARDWARE: profile_cudagraph_memory itself hard-codes torch.cuda.mem_get_info
        # (gpu_model_runner.py:6442, :6456, :6478, :6481) and current_platform.graph_pool_handle,
        # so "capture_path_ready()" is necessary but not sufficient for this line to work.
        estimate = int(profile())
        logger.info("TileInfer: estimated CUDA-graph memory %s", _format_gib(estimate))
        return estimate

    # ------------------------------------------------------------------ KV cache --------------

    def _maybe_get_memory_pool_context(self, tag: str) -> Any:
        """Always ``nullcontext()``: there is no device memory-pool allocator for NPU.

        **Why an override outside the four hooks**: the inherited body
        (``gpu_worker.py:202-223``) returns ``nullcontext()`` for ``is_cuda_alike()`` without the
        CuMem allocator (``:203-207``), for XPU without sleep mode (``:209-213``) and for CPU
        (``:215-216``) - and for anything else it reaches ``get_mem_allocator_instance()``
        (``:218``).  An out-of-tree platform is none of those three (``Platform.is_cuda_alike`` is
        False for ``PlatformEnum.OOT``, ``interface.py:191-193``), and TileInfer has no allocator
        registered, so the inherited body raises - in ``initialize_from_config`` (``:577``) *and*
        in ``load_model`` (``:351``), before any KV cache exists.  Nullcontext is the XPU/CPU
        precedent, i.e. the behaviour those branches were put there to express.

        HARDWARE: whether Ascend has a CuMem-equivalent pool that would be worth using.  If a
        torch_npu allocator pool appears (for sleep mode or fragmentation), implement it here and
        remember that ``load_model`` uses the same hook with ``tag="weights"``, where
        ``gpu_worker.py:219-222`` asserts single-instance usage.  First diagnostic:
        ``python -c "import vllm.device_allocator as a; print(getattr(a, '_MEM_ALLOCATOR', None))"``
        """
        logger.debug(
            "TileInfer: no device memory-pool allocator for tag %r; using nullcontext "
            "(gpu_worker.py:202-223, XPU/CPU precedent at :209-216)",
            tag,
        )
        return nullcontext()

    def initialize_from_config(self, kv_cache_config: Any) -> None:
        """Allocate the pool the ``KVCacheConfig`` describes - unchanged from the GPU path.

        **What the GPU worker does** (``gpu_worker.py:562-589``): stores
        ``cache_config.num_gpu_blocks`` (``:568``), initialises the KV connector *before* the
        allocation because ``initialize_kv_cache`` injects connector-unrelated groups (``:570-575``),
        allocates inside the memory-pool context (``:577-578``), then optionally initialises the
        routed-experts capturer (``:580-581``) and the KV-zero metadata outside the pool (``:583-589``).

        **What this override does**: the same body.  The task asks whether the GPU path is
        device-agnostic - for this hook it is, with one exception that is fixed elsewhere in this
        file: the pool context (:meth:`_maybe_get_memory_pool_context`).  Everything downstream of
        ``model_runner.initialize_kv_cache`` is read and device-neutral
        (``gpu_model_runner.py:7284-7341``): attention backends via ``current_platform``
        (``:6677``), tensors via ``torch.zeros(..., device=self.device)`` (``:7014``), and the
        reshape walks both attention and mamba groups (``:7040-7057``).  The mamba half -
        ``initialize_mamba_ssu_backend`` (``:7301``) - is not a device seam for this model: GDN
        specs are not MAMBA1/MAMBA2, so the function returns before instantiating a backend
        (:func:`require_supported_mamba_state_path`, asserted here before the call).

        **The specs are not hand-built here, by design.**  ``Worker.get_kv_cache_spec``
        (``gpu_worker.py:547-548``) asks the model (``gpu_model_runner.py:7440``), so the 10 NHD
        attention layers and 30 mamba-state layers arrive as the model's own
        ``KVCacheConfig.kv_cache_groups``; this hook only forwards it.  There is no
        ``NpuPlatformKVCache`` class in this repository (``grep -rn NpuPlatformKVCache tileinfer``
        finds nothing), so there is no TileInfer pool to route the attention pool through: if one
        is added, the seam is ``initialize_kv_cache_tensors`` (``gpu_model_runner.py:7201``) /
        ``_allocate_kv_cache_tensors`` (``:6999``) - not this worker.

        HARDWARE: the hybrid split, the state-update port and the zeroing outcome.

        1. the runner must allocate all 40 groups for block_size 2048 (``docs/plan-vllm-native-backend.md``
           §5, W7) and ``npu-smi info`` must show the growth.  First diagnostic:
           ``python -c "import torch, torch_npu, vllm; print(torch.npu.memory_allocated(0))"``
           before/after, or ``npu-smi info -t memory -i 0`` while start-up blocks here.
        2. ``kv_cache_config.needs_kv_cache_zeroing`` (``:586-589``): the guard is
           ``hasattr(self.model_runner, "_init_kv_zero_meta")``, and this deployment's runner
           *has* the method but declines it with a logged reason
           (``TileInferModelRunner._init_kv_zero_meta``): vLLM's zeroer derives a block's size
           from ``stride(block_dim)``, which is half a page under this backend's K/V-outermost
           stride order, so it would clear K and never V.  Skipping is safe for every read this
           stack performs (write-before-read, ``kv_indptr``-bounded kernels) and mamba state is
           never zeroed by that class anyway.  What still needs a device run: no garbage on a
           recycled attention page (a path that reads before writing would surface it there).
        3. the mamba state update itself: for GDN it is our torch port of
           ``fused_recurrent_gated_delta_rule_packed_decode`` (``ops/fla/recurrent.py``), not
           vLLM's SSU backend.  A device run must show the port writing the new state into the
           real paged ``ssm_state`` tensor in place and reading the ``conv_state`` layout that
           ``causal_conv1d_update`` wrote, at bf16/fp16.  The CPU oracles pin the math, not those
           layouts; that is the remaining hardware-only risk of this path.
        """
        # Decide the mamba state-update path before initialize_kv_cache reaches vLLM's
        # initialize_mamba_ssu_backend (gpu_model_runner.py:7301).  GDN hybrids pass and record
        # "gdn-fla"; a MAMBA1/MAMBA2 model raises here with the missing backend named, instead of
        # failing one step later inside vLLM's Triton kernel import.
        self.mamba_state_path = require_supported_mamba_state_path(kv_cache_config)
        if self.mamba_state_path["decision"] == "gdn-fla":
            logger.info(
                "TileInfer: mamba state path - %d GDN layer(s) in %d mamba spec type(s) (%s); "
                "vLLM's SSU backend is a no-op for mamba_type=GDN_ATTN, and the decode state "
                "update is our torch port of fused_recurrent_gated_delta_rule_packed_decode "
                "(tileinfer.vllm.ops.install)",
                self.mamba_state_path["gdn_layers"],
                len(self.mamba_state_path["types"]),
                ", ".join(f"{name}={count}" for name, count in sorted(
                    self.mamba_state_path["types"].items())),
            )
        else:
            logger.debug(
                "TileInfer: mamba state path - %r (no mamba group needs the SSU backend)",
                self.mamba_state_path["decision"],
            )

        # Available to the warm-up stage after profiling (gpu_worker.py:566-568).  The field is
        # CUDA-flavoured but device-neutral; it is what vLLM reads back.
        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks

        # Init the KV connector before initialize_kv_cache (gpu_worker.py:570-575): the latter
        # injects groups unrelated to the connector.
        from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized

        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)

        with self._maybe_get_memory_pool_context(tag="kv_cache"):
            self.model_runner.initialize_kv_cache(kv_cache_config)

        if getattr(self.model_config, "enable_return_routed_experts", False):
            self.model_runner.init_routed_experts_capturer()

        # Bookkeeping tensors stay outside the pool (gpu_worker.py:583-589).  The call is kept
        # (the guard is the inherited hasattr), but this runner's override logs and declines:
        # KVBlockZeroer's stride math cannot address a K/V-outermost pool (see
        # TileInferModelRunner._init_kv_zero_meta).
        if getattr(kv_cache_config, "needs_kv_cache_zeroing", False) and hasattr(
            self.model_runner, "_init_kv_zero_meta"
        ):
            self.model_runner._init_kv_zero_meta()

        logger.info(
            "TileInfer: KV pool initialised by the runner - %d block(s), %d group(s) (%s)",
            int(kv_cache_config.num_blocks),
            len(getattr(kv_cache_config, "kv_cache_groups", ()) or ()),
            "hybrid" if getattr(self.model_config, "is_hybrid", False) else "attention-only",
        )

    # ------------------------------------------------------------------ warm-up / capture -----

    def compile_or_warm_up_model(self) -> Any:
        """Warm the model up, skip the CUDA-only warm-ups, then capture only if the graph path is ready.

        **What the GPU worker does** (``gpu_worker.py:591-750``): computes the compile sizes only
        in ``CompilationMode.VLLM_COMPILE`` (``:595-616``), runs ``_dummy_run`` for each
        (``:619-621``), removes dummy LoRAs (``:622``), runs ``kernel_warmup(self)``
        (``:626``, the Triton tuning path), calls ``model_runner.capture_model()`` when not eager
        (``:628-630``), prints the graph-memory estimate comparison (``:632-646``) and the KV-size
        suggestion (``:648-703``), warms the sampler / pools ``_dummy_run`` **after** capture so
        ``empty_cache`` cannot clear the buffers (``:708-728``), reseeds (``:732``), activates the
        Triton JIT monitor (``:736-740``) and freezes the GC heap (``:742-745``), returning
        ``CompilationTimes`` (``:747-750``).

        **What this override does**: the same ordering with three CUDA-only steps replaced by
        explicit no-ops:

        * ``kernel_warmup(self)`` (``:626``) is **not** called - it is Triton tuning for CUDA
          kernels, and this deployment runs the TileInfer paged kernels, whose per-bucket prewarm
          is ``tileinfer/serving/planner.py`` (``docs/plan-vllm-native-backend.md:226-229``).
          The warm-up sizes below still drive ``_dummy_run``, so the model, the attention backend
          and the sampler are exercised; only the Triton autotune is skipped, with a log line;
        * **the decode plans are pre-warmed here** (:meth:`_prewarm_decode_plans`), after the
          eager ``_dummy_run`` loop and **before** :meth:`capture_model`.  That is the only window
          where both halves of W8's contract can be satisfied: the KV pool exists (the runner
          bound it in ``initialize_from_config``) and no graph has been captured yet.  The bucket
          set is derived from ``max_num_seqs`` / ``max_num_batched_tokens`` / ``max_model_len`` /
          ``cache_config.block_size`` through ``serving/planner`` and the plans are built through
          the impl's *own* plan cache (``attn/impl.py``), so a captured step can only look them
          up.  The call is idempotent and a missing device is a logged skip;
        * ``capture_model()`` goes through :meth:`capture_model` instead of the runner's directly,
          so a skipped capture is counted and explained (the platform already forces
          ``enforce_eager``, and capture must not silently not-happen);
        * the Triton JIT monitor (``:736-740``) is skipped - it watches ``triton_utils``
          compilations, which cannot happen on the TileInfer path.

        The sampler / logits pre-allocation (``:708-728``) keeps its position *after* capture on
        purpose: ``empty_cache`` inside capture would free those buffers.

        HARDWARE: whether the TileInfer plan prewarm has run for every bucket before any capture
        (that is W8's contract, ``docs/plan-vllm-native-backend.md:160``), and whether
        ``_dummy_run``'s activation peak fits the pool measured above.  First diagnostic:
        ``python -c "from tileinfer.vllm import graphs; print(graphs.capture_path_ready())"``
        """
        from vllm.config import CUDAGraphMode
        from vllm.config.compilation import CompilationMode
        from vllm.distributed.parallel_state import get_pp_group

        compilation_config = getattr(self, "compilation_config", None) or (
            self.vllm_config.compilation_config
        )

        # --- the compile/warm-up sizes (gpu_worker.py:593-616), copied -------------------------
        warmup_sizes: list[int] = []
        if compilation_config.mode == CompilationMode.VLLM_COMPILE:
            compile_sizes = compilation_config.compile_sizes
            warmup_sizes = list(compile_sizes) if compile_sizes is not None else []
            cg_capture_sizes: list[int] = []
            if compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
                cg_sizes = compilation_config.cudagraph_capture_sizes
                cg_capture_sizes = [] if cg_sizes is None else list(cg_sizes)
                warmup_sizes = [x for x in warmup_sizes if x not in cg_capture_sizes]
            compile_ranges = compilation_config.get_compile_ranges()
            # For each compile_range, if none of the sizes is in it, add the end of the range so
            # that something inside the range is compiled (gpu_worker.py:608-616).
            all_sizes = set(cg_capture_sizes)
            all_sizes.update(x for x in warmup_sizes if isinstance(x, int))
            for compile_range in compile_ranges:
                if not any(x in compile_range for x in all_sizes):
                    warmup_sizes.append(compile_range.end)

        # We skip EPLB here since we don't want to record dummy metrics (gpu_worker.py:618).
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size, skip_eplb=True, remove_lora=False)
        self.model_runner.maybe_remove_all_loras(self.model_runner.lora_config)

        # Decode-plan pre-warm, before any capture attempt.  Ordering is the contract: the pool was
        # allocated and bound in initialize_from_config, and a captured region may only *look up*
        # plans, so every reachable bucket is compiled here, synchronously.  Idempotent, and a
        # missing device is a logged skip (see _prewarm_decode_plans).
        self._prewarm_decode_plans()

        # HARDWARE: kernel_warmup(self) (gpu_worker.py:626) is NOT called.  It is
        # `vllm.model_executor.warmup.kernel_warmup`, whose body tunes CUDA/Triton kernels; the
        # TileInfer paged kernels are prewarmed per bucket by tileinfer/serving/planner.py instead.
        # If a kernel-level warm-up turns out to be needed for the Ascend build, it belongs here,
        # before capture - the same position, a different warm-up function.
        logger.info(
            "TileInfer: skipping kernel_warmup() (gpu_worker.py:626) - it tunes CUDA/Triton "
            "kernels; per-bucket prewarm is tileinfer/serving/planner.py "
            "(docs/plan-vllm-native-backend.md:226-229)"
        )

        # The capture gate lives inside capture_model() so that a skip is counted and logged even
        # when the entry point is reached (gpu_worker.py:628-630 would simply not call it).
        graph_memory_bytes = self.capture_model()

        if (
            hasattr(self, "cudagraph_memory_estimate")
            and self.cudagraph_memory_estimate > 0
        ):  # pragma: no cover - only when capture is ready
            logger.info(
                "CUDA graph pool memory: %s (actual), %s (estimated)",
                _format_gib(graph_memory_bytes),
                _format_gib(self.cudagraph_memory_estimate),
            )

        # `kv_cache_memory_bytes` lives on CacheConfig in some 0.23 builds and is absent in
        # others (the first Ascend950PR run raised AttributeError here).  Read it defensively so a
        # missing field only skips the informational suggestion below.
        _kv_cache_memory = getattr(compilation_config, "kv_cache_memory_bytes", None)
        if _kv_cache_memory is None and not hasattr(compilation_config, "kv_cache_memory_bytes"):
            _kv_cache_memory = getattr(
                getattr(self.vllm_config, "cache_config", None), "kv_cache_memory_bytes", None
            )
        if _kv_cache_memory is None and hasattr(
            self, "peak_activation_memory"
        ):
            # The --kv-cache-memory suggestion (gpu_worker.py:648-703).  Kept because it is the
            # number a bring-up engineer needs when the pool is too small or too large.
            redundancy_buffer_memory = 150 * (1 << 20)
            non_kv_cache_memory = (
                int(getattr(self.model_runner, "model_memory_usage", 0) or 0)
                + self.peak_activation_memory
                + self.non_torch_memory
                + graph_memory_bytes
            )
            kv_cache_memory_bytes_to_gpu_limit = (
                self.init_snapshot.free_memory - non_kv_cache_memory - redundancy_buffer_memory
            )
            kv_cache_memory_bytes_to_requested_limit = (
                int(self.requested_memory) - non_kv_cache_memory - redundancy_buffer_memory
            )
            logger.info(
                "Free memory on device (%s/%s) at startup; desired utilization %.4f "
                "(%s requested). Actual usage: weights %s, peak activation %s, non-torch %s, "
                "CUDAGraph %s. For --kv-cache-memory: %d to fit the requested memory, %d to use "
                "the whole device (gpu_worker.py:681-701).",
                _format_gib(self.init_snapshot.free_memory),
                _format_gib(self.init_snapshot.total_memory),
                float(getattr(self.cache_config, "gpu_memory_utilization", 0.0)),
                _format_gib(self.requested_memory),
                _format_gib(int(getattr(self.model_runner, "model_memory_usage", 0) or 0)),
                _format_gib(self.peak_activation_memory),
                _format_gib(self.non_torch_memory),
                _format_gib(graph_memory_bytes),
                kv_cache_memory_bytes_to_requested_limit,
                kv_cache_memory_bytes_to_gpu_limit,
            )

        # Sampler / pooled-output warm-up, AFTER capture on purpose (gpu_worker.py:708-728): the
        # buffers must not be cleared by the empty_cache inside graph capture.
        if get_pp_group().is_last_rank:
            max_num_reqs = min(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens,
            )
            hidden_states, last_hidden_states = self.model_runner._dummy_run(
                num_tokens=max_num_reqs,
                skip_eplb=True,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            if self.model_runner.is_pooling_model:
                self.model_runner._dummy_pooler_run(hidden_states)
            else:
                self.model_runner._dummy_sampler_run(hidden_states=last_hidden_states)

        # Reset the seed so initialization/profiling cannot affect the RNG (gpu_worker.py:730-732).
        self._set_random_seed(self.model_config.seed)

        # HARDWARE: the Triton JIT monitor (gpu_worker.py:736-740) is NOT activated - it watches
        # triton_utils compilations, and the TileInfer path compiles no Triton kernels.  If a
        # latency spike ever needs the same instrument, `tileinfer/vllm/ops` has its own counters.

        from vllm.utils.gc_utils import freeze_gc_heap, maybe_attach_gc_debug_callback

        freeze_gc_heap()  # device-neutral (gpu_worker.py:742-745)
        maybe_attach_gc_debug_callback()

        from vllm.v1.worker.worker_base import CompilationTimes

        return CompilationTimes(
            language_model=compilation_config.compilation_time,
            encoder=compilation_config.encoder_compilation_time,
        )

    def capture_model(self) -> int:
        """Capture the graphs - or, while the path is not ready, count and explain the skip.

        This is **not** one of the four base-worker hooks: the GPU worker calls
        ``self.model_runner.capture_model()`` directly (``gpu_worker.py:630``).  It is a method
        here so the gate has exactly one home and can be tested without a device.

        The gate is :func:`tileinfer.vllm.graphs.capture_path_ready`, the module that owns the
        verdict: its two CUDA call sites (``cuda_graph.py:283``, ``:313``) need a bridge, and until
        that exists capture would run on the CUDA backend.  ``TileInferPlatform`` therefore forces
        ``model_config.enforce_eager = True`` (``platform.py:1392-1426``) - but *that* is not the
        gate used here: a skipped capture still has to be visible, whereas
        ``gpu_worker.py:628-630`` would simply not call it and print nothing.

        Returns
        -------
        int
            Bytes of graph pool consumed: ``0`` when skipped (the same value the runner returns
            when ``cudagraph_mode`` is ``NONE``, ``gpu_model_runner.py:6526-6531``).

        HARDWARE: when ``capture_path_ready()`` becomes True, the delegation below is still not
        sufficient on its own - ``GPUModelRunner.capture_model`` hard-codes
        ``torch.cuda.mem_get_info()`` at ``gpu_model_runner.py:6547`` and ``:6565``, and
        ``profile_cudagraph_memory`` does the same at ``:6442/:6456/:6478/:6481``.  Both must be
        bridged (or the wrapper replaced), then this method can delegate.  First diagnostic:
        ``python -c "from tileinfer.vllm import graphs; print(graphs.capture_path_ready())"``
        """
        stats = self._capture_stats()
        stats["attempts"] += 1
        ready, reason = graphs.capture_path_ready()
        if not ready:
            stats["skipped"] += 1
            stats["last_skip_reason"] = reason
            eager = bool(getattr(getattr(self, "model_config", None), "enforce_eager", False))
            logger.warning(
                "TileInfer: skipping graph capture (skip #%d of %d attempt(s)); enforce_eager=%s. "
                "Reason: %s. Serving eagerly is the intended state until W8 lands "
                "(tileinfer/vllm/graphs.py, cuda_graph.py:283/:313); the counters are in "
                "worker.capture_stats and the decision in this log line, so an eager run cannot "
                "be mistaken for a capture that silently failed.",
                stats["skipped"],
                stats["attempts"],
                eager,
                reason,
            )
            return 0

        logger.info(
            "TileInfer: graph capture path reports ready (%s); delegating to the runner's "
            "capture_model()",
            reason,
        )
        captured_bytes = int(self.model_runner.capture_model())
        stats["captured"] += 1
        return captured_bytes
