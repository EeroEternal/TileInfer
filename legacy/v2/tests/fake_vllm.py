"""The **one** fake ``vllm`` the CPU contract tests share, with shapes that never depend on order.

Why this module exists
----------------------
``tests/test_worker_contract.py`` and ``tests/test_attn_contract.py`` both import TileInfer modules
that speak vLLM 0.23's API (``tileinfer/vllm/worker.py`` subclasses two vLLM classes;
``tileinfer/vllm/attn/*`` imports the attention backend ABC and the registry).  CI has neither vLLM
nor an NPU, so the *names* those modules import are injected into ``sys.modules`` before the import.

Each file used to carry its own private fake, and the two fakes had **different shapes**.  That made
the suite order-dependent: whichever contract test imported first decided what ``vllm`` looked like
for the whole pytest process, and the second file either found a ``vllm`` missing the names it needs
(``ModuleNotFoundError: No module named 'vllm.v1.attention'`` during *collection* - not a test
failure, the whole session is interrupted) or, worse, silently ran its assertions against a thinner
fake than it was written for.  A suite whose meaning depends on ``pytest a.py b.py`` vs
``pytest b.py a.py`` is not a suite.

So the fake lives here, once, and it provides the **union** of everything the contract tests need:

=============================================  ===============================================
module                                         names (and the ``file:line`` they stand for)
=============================================  ===============================================
``vllm.v1.attention.backend``                  ``AttentionType`` (:32), ``MultipleOf`` (:48),
                                               ``AttentionBackend`` (:55), ``AttentionMetadata``
                                               (:352), ``AttentionCGSupport`` (:506),
                                               ``AttentionMetadataBuilder`` (:523),
                                               ``AttentionImpl`` (:770),
                                               ``CommonAttentionMetadata`` (:359)
``vllm.v1.attention.backends.registry``        ``AttentionBackendEnum`` (:34),
                                               ``register_backend`` (:252-264),
                                               ``registry_calls`` (recorder)
``vllm.v1.worker.gpu_worker``                  ``Worker`` (gpu_worker.py:117)
``vllm.v1.worker.gpu_model_runner``            ``GPUModelRunner`` (gpu_model_runner.py:418),
                                               ``init_device_properties_calls`` (proof the seam
                                               under test really runs)
``vllm.v1.worker.utils``                       ``request_memory`` (gpu_worker.py:76),
                                               ``request_memory_calls`` (recorder)
``vllm.distributed.kv_transfer``               ``ensure_kv_transfer_initialized`` (:570-575),
                                               ``kv_transfer_calls`` (recorder)
=============================================  ===============================================

Why a helper module and not a ``conftest.py`` fixture
----------------------------------------------------
The names have to exist **at import time** of the test module: the
``from tileinfer.vllm.attn.backend import ...`` statement at module scope is what needs them, and
it runs before any fixture could be requested.  So each contract test keeps its existing
``_install_fake_vllm()`` call site, whose body is now one line delegating to :func:`install`.

Rules this fake follows (they are why it is safe to share)
----------------------------------------------------------
* **Cooperating, never clobbering.**  Every module and attribute goes through :func:`ensure` /
  :func:`setdefault`, so a fake installed earlier (by ``tests/test_platform_contract.py`` or by
  this module in another test file) is *augmented* and its module objects are kept: already-imported
  TileInfer modules hold references to objects inside them, and replacing a module would leave those
  references pointing at a stale copy.
* **Real vLLM wins.**  If ``vllm`` is importable (or already imported from a real installation),
  this module installs nothing and says so; the contract tests then assert against the real 0.23
  API.
* **A fake, labelled as one.**  Each class reproduces only what the tests depend on, and says so in
  its docstring.  The fake base classes are deliberately *thinner* than vLLM's (e.g.
  ``AttentionBackend.validate_configuration`` returns ``[]``): the TileInfer overrides must state
  their own rejects, and a stub that already validated would make those tests vacuous.
* **Shapes are documented, not implied.**  ``__all__`` and the mode strings are part of this
  module's contract; ``tests/test_attn_contract.py::test_mode_is_documented`` pins the modes.
"""

from __future__ import annotations

import dataclasses
import enum
import importlib.util
import sys
import types
from typing import Any, List

import torch

__all__ = [
    "install",
    "ensure",
    "ensure_chain",
    "setdefault",
    "MODES",
]

#: Values :func:`install` can return.  ``fake``: this module created ``vllm``; ``augmented``: a
#: fake ``vllm`` was already in ``sys.modules`` and was completed in place; ``real``: a real vLLM
#: is installed/already imported, nothing was injected.
MODES = ("fake", "augmented", "real")


# ---------------------------------------------------------------------------------------
# Cooperation helpers (also used by the test files for their own test-local fakes)
# ---------------------------------------------------------------------------------------


def ensure(name: str, package: bool = False) -> types.ModuleType:
    """``sys.modules[name]``, creating an empty module (with a package marker) when absent."""
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    if package and not hasattr(module, "__path__"):
        module.__path__ = []  # type: ignore[attr-defined]
    return module


def setdefault(module: types.ModuleType, name: str, value: Any) -> Any:
    """Set an attribute only when the module does not already have one; return what is there.

    This is the whole cooperation rule: an attribute another fake (or a real vLLM) already
    provided is never replaced, so the *first* fake to define a name keeps it and every later fake
    only fills the gaps.  Returns the effective value, which is useful in assertions.
    """
    if not hasattr(module, name):
        setattr(module, name, value)
    return getattr(module, name)


def ensure_chain(name: str) -> types.ModuleType:
    """``sys.modules[name]`` with every parent package created, wired and importable."""
    parts = name.split(".")
    for depth in range(1, len(parts)):
        ensure(".".join(parts[:depth]), package=True)
    module = ensure(name)
    if len(parts) > 1:
        setattr(sys.modules[".".join(parts[:-1])], parts[-1], module)
    return module


# ---------------------------------------------------------------------------------------
# The attention API the three tileinfer.vllm.attn modules import
# (reference/vllm-0.23/vllm__v1__attention__backend.py / ...__registry.py)
# ---------------------------------------------------------------------------------------


class AttentionType(str, enum.Enum):
    """``vllm/v1/attention/backend.py:32-45``."""

    DECODER = "decoder"
    ENCODER = "encoder"
    ENCODER_ONLY = "encoder_only"
    ENCODER_DECODER = "encoder_decoder"


class MultipleOf:
    """``backend.py:48-52`` - a block-size predicate, kept as the object the tests inspect."""

    def __init__(self, base: int):
        self.base = base

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"MultipleOf({self.base})"


class AttentionBackend:
    """``backend.py:55-349``, reduced to what the TileInfer backend's contract reads.

    ``validate_configuration`` returns ``[]`` on purpose: the TileInfer override must state its own
    rejects explicitly (a stub that silently "passed" would make the reject tests vacuous).
    """

    supported_dtypes = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes = ["auto", "float16", "bfloat16"]
    forward_includes_kv_cache_update = True

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [MultipleOf(1)]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ):
        """``backend.py:88-96``: abstract; the TileInfer backend states its own."""
        raise NotImplementedError

    @classmethod
    def get_kv_cache_block_dim(
        cls,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> int:
        """``backend.py:98-116`` verbatim: plant a sentinel and see which dim index it lands in.

        Kept faithful on purpose: this is the probe ``_update_hybrid_attention_mamba_layout``
        uses to decide whether to rewrite the pool (``gpu_model_runner.py:7183-7192``), so a test
        that pins it against the stub is meaningful against the real ABC too.
        """
        _S = 1234567
        shape = cls.get_kv_cache_shape(
            _S,
            block_size,
            num_kv_heads,
            head_size,
            cache_dtype_str=cache_dtype_str,
        )
        return shape.index(_S)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ):
        """``backend.py:118-147``: the default is "physical layout == logical layout"."""
        raise NotImplementedError

    @classmethod
    def is_mla(cls) -> bool:
        return False

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """``:269-273``; the real ABC answers from ``supported_attn_types``."""
        return attn_type == AttentionType.DECODER

    @classmethod
    def validate_configuration(cls, *args, **kwargs):
        return []


class AttentionMetadata:
    """``backend.py:352-353`` - a marker base class only."""


class AttentionCGSupport(enum.Enum):
    """``backend.py:506-520`` (the numeric order is the *level* order)."""

    ALWAYS = 3
    UNIFORM_BATCH = 2
    UNIFORM_SINGLE_TOKEN_DECODE = 1
    NEVER = 0


class AttentionMetadataBuilder:
    """``backend.py:523-670``, reduced to the ClassVar lookup the tests assert on."""

    _cudagraph_support = AttentionCGSupport.NEVER
    reorder_batch_threshold = None
    supports_update_block_table = False

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        return cls._cudagraph_support


class AttentionImpl:
    """``backend.py:770-850``.  Deliberately a plain class: the real base's ``__new__`` needs
    ``vllm.distributed.parallel_state``, which this fake does not pretend to have."""


@dataclasses.dataclass
class CommonAttentionMetadata:
    """``backend.py:359-422`` - the fields the TileInfer builder reads; the rest are defaulted."""

    query_start_loc: object
    query_start_loc_cpu: object
    seq_lens: object
    num_reqs: int
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    block_table_tensor: object
    slot_mapping: object
    causal: bool = True
    is_prefilling: object = None
    seq_lens_cpu_upper_bound: object = None
    positions: object = None


class AttentionBackendEnum(enum.Enum):
    """``backends/registry.py:34-101`` - two members are enough to prove the API shape.

    ``CUSTOM`` starts as ``None`` (the "must be registered before use" slot); the recorder below
    replaces it through ``register_backend``.
    """

    FLASH_ATTN = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    CUSTOM = None


#: Registered ``(enum_member, class_path)`` pairs, in call order.  ``tests/test_attn_contract.py``
#: asserts that importing the backend module adds nothing here and that ``register_with_vllm()``
#: adds exactly one entry.
REGISTRY_CALLS: List[Any] = []


def _register_backend(backend, class_path=None, is_mamba=False):
    """``registry.py:252-264`` - both call forms, recording instead of mutating vLLM's tables."""
    if class_path is None:  # decorator form, `:252-257`

        def decorator(cls):
            REGISTRY_CALLS.append((backend, f"{cls.__module__}.{cls.__qualname__}"))
            return cls

        return decorator
    REGISTRY_CALLS.append((backend, class_path))  # direct form, `:259-264`
    return lambda x: x


def _install_attention() -> None:
    """Create/augment the attention modules with the names the table in the docstring lists."""
    backend_mod = ensure("vllm.v1.attention.backend")
    for name, value in (
        ("AttentionType", AttentionType),
        ("MultipleOf", MultipleOf),
        ("AttentionBackend", AttentionBackend),
        ("AttentionMetadata", AttentionMetadata),
        ("AttentionCGSupport", AttentionCGSupport),
        ("AttentionMetadataBuilder", AttentionMetadataBuilder),
        ("AttentionImpl", AttentionImpl),
        ("CommonAttentionMetadata", CommonAttentionMetadata),
    ):
        setdefault(backend_mod, name, value)

    registry_mod = ensure("vllm.v1.attention.backends.registry")
    setdefault(registry_mod, "AttentionBackendEnum", AttentionBackendEnum)
    setdefault(registry_mod, "register_backend", _register_backend)
    setdefault(registry_mod, "registry_calls", REGISTRY_CALLS)


# ---------------------------------------------------------------------------------------
# The worker API ``tileinfer.vllm.worker`` imports
# (reference/vllm-0.23/vllm__v1__worker__gpu_worker.py / ...__gpu_model_runner.py)
# ---------------------------------------------------------------------------------------


class Worker:
    """``gpu_worker.py:117-1159``, reduced to "a class exists to subclass".

    The four hook bodies are *not* copied here: the signature test compares against the real file
    under ``reference/vllm-0.23/`` (parsed with ``ast``), which is stronger than a hand copy.
    """


#: Recorders for the two vLLM helpers below: ``tests/test_worker_contract.py`` asserts that the
#: worker calls vLLM's own helper rather than a local copy, so the call has to be observable.
REQUEST_MEMORY_CALLS: List[Any] = []
KV_TRANSFER_CALLS: List[Any] = []


class GPUModelRunner:
    """``gpu_model_runner.py:418-425``, reduced to the construction seam under test.

    ``__init__`` calls ``_init_device_properties`` (``:711``) exactly like the real one, and that
    method raises exactly as the real path does once it reaches the platform ABC's
    ``num_compute_units`` (``:1109-1112`` -> ``vllm.utils.platform_utils`` ->
    ``Platform.num_compute_units``, which ``interface.py:1027-1035`` makes
    ``raise NotImplementedError`` and ``TileInferPlatform`` deliberately leaves raising).
    """

    #: Counted so a test can prove the seam really runs during construction.
    init_device_properties_calls = 0

    def __init__(self, vllm_config: Any, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device
        self.model_memory_usage = 0
        self._init_device_properties()

    def _init_device_properties(self) -> None:
        GPUModelRunner.init_device_properties_calls += 1
        raise NotImplementedError(
            "num_compute_units is not implemented for the current platform."
        )


def _request_memory(snapshot: Any, cache_config: Any) -> int:
    """vLLM's helper: ``total * utilization``, clamped by the free memory (device-agnostic)."""
    REQUEST_MEMORY_CALLS.append((snapshot, cache_config))
    requested = int(snapshot.total_memory * cache_config.gpu_memory_utilization)
    return min(requested, int(snapshot.free_memory))


def _ensure_kv_transfer_initialized(vllm_config: Any, kv_cache_config: Any) -> None:
    """``gpu_worker.py:570-575``: a no-op when no connector is configured, which is the v0 case."""
    KV_TRANSFER_CALLS.append((vllm_config, kv_cache_config))


def _install_worker() -> None:
    """Create/augment the worker modules with the names ``tileinfer.vllm.worker`` imports."""
    gpu_worker_mod = ensure("vllm.v1.worker.gpu_worker")
    setdefault(gpu_worker_mod, "Worker", Worker)

    runner_mod = ensure("vllm.v1.worker.gpu_model_runner")
    setdefault(runner_mod, "GPUModelRunner", GPUModelRunner)

    worker_utils = ensure("vllm.v1.worker.utils")
    setdefault(worker_utils, "request_memory", _request_memory)
    setdefault(worker_utils, "request_memory_calls", REQUEST_MEMORY_CALLS)

    kv_transfer = ensure("vllm.distributed.kv_transfer")
    setdefault(kv_transfer, "ensure_kv_transfer_initialized", _ensure_kv_transfer_initialized)
    setdefault(kv_transfer, "kv_transfer_calls", KV_TRANSFER_CALLS)


# ---------------------------------------------------------------------------------------
# Wiring and the one entry point
# ---------------------------------------------------------------------------------------


def _wire(root: types.ModuleType) -> None:
    """Attach every child module to its parent package (``from a.b import c`` needs the chain)."""
    v1 = ensure("vllm.v1", package=True)
    attention_pkg = ensure("vllm.v1.attention", package=True)
    backends_pkg = ensure("vllm.v1.attention.backends", package=True)
    worker_pkg = ensure("vllm.v1.worker", package=True)
    distributed_pkg = ensure("vllm.distributed", package=True)

    setdefault(v1, "attention", attention_pkg)
    setdefault(v1, "worker", worker_pkg)
    setdefault(attention_pkg, "backend", ensure("vllm.v1.attention.backend"))
    setdefault(attention_pkg, "backends", backends_pkg)
    setdefault(backends_pkg, "registry", ensure("vllm.v1.attention.backends.registry"))
    setdefault(worker_pkg, "gpu_worker", ensure("vllm.v1.worker.gpu_worker"))
    setdefault(worker_pkg, "gpu_model_runner", ensure("vllm.v1.worker.gpu_model_runner"))
    setdefault(worker_pkg, "utils", ensure("vllm.v1.worker.utils"))
    setdefault(distributed_pkg, "kv_transfer", ensure("vllm.distributed.kv_transfer"))
    setdefault(root, "v1", v1)
    setdefault(root, "distributed", distributed_pkg)


def install() -> str:
    """Point ``sys.modules['vllm']`` at the shared fake; return the mode from :data:`MODES`.

    Idempotent and cooperating: a second call (a second contract test in the same pytest process)
    only fills in names a previous fake did not define, which is what makes the contract tests
    order-independent.  Call this **before** importing the TileInfer module under test; that is the
    only moment the names have to exist.
    """
    existing = sys.modules.get("vllm")
    if existing is not None and getattr(existing, "__file__", None):
        return "real"
    if existing is None:
        try:
            spec = importlib.util.find_spec("vllm")
        except (ImportError, ValueError):  # a broken install must not masquerade as "present"
            spec = None
        if spec is not None:
            return "real"

    mode = "augmented" if existing is not None else "fake"
    root = ensure("vllm", package=True)
    _install_attention()
    _install_worker()
    _wire(root)
    return mode
