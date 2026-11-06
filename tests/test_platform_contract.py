"""Contract tests for ``TileInferPlatform`` and the graph-capture decision — **without vLLM**.

Why a fake `vllm` package
-------------------------
``tileinfer/vllm/platform.py`` and ``tileinfer/vllm/worker.py`` are the two modules that must
speak vLLM's platform API, and they must speak *exactly* the 0.23 dialect transcribed in
``docs/notes/vllm-api-0.23.md`` and ``docs/notes/vllm-attention-contract.md``.  CI has neither
vLLM nor a device, so the names they import are injected into ``sys.modules`` before the modules
are imported.  Two fakes can meet in that namespace: the **shared** one in ``tests/fake_vllm.py``
(used by ``tests/test_attn_contract.py`` and ``tests/test_worker_contract.py``) and the one this
file installs below (``_ensure`` / ``_setdefault``), which adds the *platform* names.  They
cooperate by construction instead of depending on collection order: each only fills in names the
other left out and never replaces a module already in ``sys.modules``, because an already-imported
TileInfer module holds references to the objects inside it.

The fake is deliberately minimal and documented as a fake.  It provides the *names* plus the
behaviours these tests depend on:

* ``Platform`` — a base whose raising methods really raise, so "left to the base class" is a
  testable statement and not a comment (``get_device_name`` / ``get_device_uuid`` /
  ``set_device`` / ``check_if_supports_dtype`` raise; ``verify_quantization`` mirrors the ABC's
  "accept everything while ``supported_quantization`` is empty" trap);
* ``PlatformEnum`` with ``OOT``/``UNSPECIFIED``, because the device identity is ``_enum = OOT``;
* the three v1-attention names ``tileinfer.vllm.attn.backend`` imports, so this file can assert
  that :data:`ATTN_BACKEND_CLASS_PATH` really is *that* class and not a string that drifted;
* ``vllm.config.CUDAGraphMode`` and ``vllm.compilation.cuda_graph.CUDAGraphWrapper`` so the
  delegating branch of ``graphs.wrap`` can be exercised on a CPU box (with the bridge faked, and
  the fake wrapper recording how it was called).

What is asserted here is the part that goes wrong silently in production:

* the ``vllm.platform_plugins`` return contract (a *string* class path, ``None`` without a device,
  stable across the two calls vLLM makes);
* device identity, including ``ASCEND_RT_VISIBLE_DEVICES``;
* ``get_attn_backend_cls`` returns the TileInfer backend for plain MHA/GQA and raises for every
  attention feature the backend declares unsupported (the eight real ``AttentionSelectorConfig``
  switches, plus a non-decoder ``attn_type``), and for a pinned foreign backend;
* ``check_and_update_config`` rejects each unsupported feature *and* updates the three things it
  must (block size, worker class, eager flag);
* the ABC inventory: complete against the local ``interface.py`` (so a vLLM version bump that
  adds a method fails here), consistent with the class (no stale praise, no stale TODO), and
  capped so it can only shrink;
* the graph verdict is the one the docstring argues, its cited line numbers really contain the
  CUDA calls, and ``wrap`` refuses until the bridge exists.

If a real vLLM is installed, the fake is skipped and the real modules are imported; the
assertions are about TileInfer's contract, not about the stub.
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import importlib
import importlib.util
import inspect
import os
import pathlib
import re
import subprocess
import sys
import types
from typing import Any, Dict, List, Tuple

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: The local vLLM 0.23 source the whole file cites.  Read (never imported) so the line numbers in
#: the platform's docstrings stay checkable.
INTERFACE_SOURCE = ROOT / "runtime" / "vllm-api" / "vllm__platforms__interface.py"
CUDA_GRAPH_SOURCE = ROOT / "runtime" / "vllm-api" / "vllm__compilation__cuda_graph.py"

#: The worker-selection mechanism, read (never imported): ``ParallelConfig`` declares the field,
#: the worker wrapper consumes it, and the two in-tree platforms show the sentinel branch the
#: platform has to follow.
PARALLEL_CONFIG_SOURCE = ROOT / "runtime" / "vllm-api" / "vllm__config__parallel.py"
WORKER_BASE_SOURCE = ROOT / "runtime" / "vllm-api" / "vllm__v1__worker__worker_base.py"
CPU_PLATFORM_SOURCE = ROOT / "runtime" / "vllm-api" / "vllm__platforms__cpu.py"
PLUGINS_SOURCE = ROOT / "runtime" / "vllm-api" / "vllm__plugins____init__.py"

#: Model-architecture acceptance, read (never imported): the registry table and the hook call site.
REGISTRY_SOURCE = ROOT / "runtime" / "vllm-api" / "vllm__model_executor__models__registry.py"
MODEL_CONFIG_SOURCE = ROOT / "runtime" / "vllm-api" / "vllm__config__model.py"
QWEN35_SOURCE = ROOT / "runtime" / "vllm-api" / "vllm__model_executor__models__qwen3_5.py"

#: ``AttentionSelectorConfig`` - the field names the platform's reject policy is built on.  Also
#: read (never imported), because the names live in a vLLM module the CPU suite cannot import.
SELECTOR_SOURCE = ROOT / "runtime" / "vllm-api" / "vllm__v1__attention__selector.py"


# ---------------------------------------------------------------------------------------
# Minimal fake `vllm`
# ---------------------------------------------------------------------------------------


def _ensure(name: str, package: bool = False) -> types.ModuleType:
    """``sys.modules[name]``, creating an empty module (with a package marker) when absent."""
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    if package and not hasattr(module, "__path__"):
        module.__path__ = []  # type: ignore[attr-defined]
    return module


def _setdefault(module: types.ModuleType, name: str, value: Any) -> None:
    """Set an attribute only when the module does not already have one.

    The shared fake in ``tests/fake_vllm.py`` may already have created this module (a contract
    test that ran earlier in the same pytest process); that module object is *shared*, never
    replaced, because already-imported TileInfer modules hold references to the objects inside it.
    """
    if not hasattr(module, name):
        setattr(module, name, value)


def _install_fake_vllm() -> str:
    """Inject the smallest ``vllm`` these modules import; return which one was used."""
    existing = sys.modules.get("vllm")
    if existing is not None and getattr(existing, "__file__", None):
        return "real"
    mode = "augmented" if existing is not None else "fake"
    if existing is None:
        try:
            spec = importlib.util.find_spec("vllm")
        except (ImportError, ValueError):  # a broken install must not masquerade as "present"
            spec = None
        if spec is not None:
            return "real"

    root = _ensure("vllm", package=True)

    # --- vllm.platforms / vllm.platforms.interface (vllm__platforms__interface.py) ------------
    class PlatformEnum(enum.Enum):
        """`:38-47`."""

        CUDA = enum.auto()
        ROCM = enum.auto()
        TPU = enum.auto()
        XPU = enum.auto()
        CPU = enum.auto()
        OOT = enum.auto()
        UNSPECIFIED = enum.auto()

    class Platform:
        """`:105-1052`, reduced to the members these tests exercise.

        Every member here also exists in the real ABC, so ``inventory_problems()`` can be run
        against this base without inventing names.  The raising bodies are the point: they make
        "inherited, therefore raises" observable.
        """

        _enum = PlatformEnum.UNSPECIFIED
        device_name = "unspecified"
        device_type = ""
        dispatch_key = "CPU"
        device_control_env_var = "VLLM_DEVICE_CONTROL_ENV_VAR_PLACEHOLDER"
        simple_compile_backend = "inductor"
        supported_quantization: List[str] = []

        @property
        def pass_key(self) -> str:  # `:147-150`
            return "post_grad_custom_post_pass"

        @property
        def supported_dtypes(self) -> List[torch.dtype]:  # `:152-158`
            return [torch.bfloat16, torch.float16, torch.float32]

        def is_cuda(self) -> bool:  # `:160-161`
            return self._enum is PlatformEnum.CUDA

        def is_cuda_alike(self) -> bool:  # `:191-193`
            return self._enum in (PlatformEnum.CUDA, PlatformEnum.ROCM)

        def is_out_of_tree(self) -> bool:  # `:182-183`
            return self._enum is PlatformEnum.OOT

        def is_unspecified(self) -> bool:  # `:185-186`
            return self._enum is PlatformEnum.UNSPECIFIED

        def is_sleep_mode_available(self) -> bool:  # `:195-200`
            return self._enum in (PlatformEnum.CUDA, PlatformEnum.ROCM, PlatformEnum.XPU)

        @classmethod
        def import_kernels(cls) -> None:  # `:249-257`
            return None

        @classmethod
        def device_id_to_physical_device_id(cls, device_id: int):  # `:234-247`
            return device_id

        @classmethod
        def get_attn_backend_cls(cls, selected_backend, attn_selector_config, num_heads=None):
            return ""  # `:259-267`

        @classmethod
        def get_vit_attn_backend(cls, head_size, dtype, backend=None):  # `:275-304`
            return None

        @classmethod
        def get_device_capability(cls, device_id: int = 0):  # `:306-312`
            return None

        @classmethod
        def get_device_name(cls, device_id: int = 0) -> str:  # `:377-380`
            raise NotImplementedError

        @classmethod
        def get_device_uuid(cls, device_id: int = 0) -> str:  # `:382-385`
            raise NotImplementedError

        @classmethod
        def get_device_total_memory(cls, device_id: int = 0) -> int:  # `:387-390`
            raise NotImplementedError

        @classmethod
        def get_all_gpu_pci_bus_ids(cls) -> Dict[int, str]:  # `:392-403`
            raise NotImplementedError("unsupported on this platform")

        @classmethod
        def inference_mode(cls):  # `:405-413`
            return torch.inference_mode(mode=True)

        @classmethod
        def set_device(cls, device: torch.device) -> None:  # `:415-420`
            raise NotImplementedError

        @classmethod
        def manual_seed_all(cls, seed: int) -> None:  # `:422-425`
            raise NotImplementedError

        @classmethod
        def check_and_update_config(cls, vllm_config) -> None:  # `:457-468`
            return None

        @classmethod
        def verify_model_arch(cls, model_arch: str) -> None:  # `:708-718`
            # The real body is `pass`: "By default all models are considered supported."
            return None

        @classmethod
        def verify_quantization(cls, quant: str) -> None:  # `:720-728`
            if cls.supported_quantization and quant not in cls.supported_quantization:
                raise ValueError(f"{quant} quantization is not supported")

        @classmethod
        def is_pin_memory_available(cls) -> bool:  # `:751-762`
            return True

        @classmethod
        def get_current_memory_usage(cls, device=None) -> float:  # `:764-771`
            raise NotImplementedError

        @classmethod
        def get_punica_wrapper(cls) -> str:  # `:773-778`
            raise NotImplementedError

        @classmethod
        def get_static_graph_wrapper_cls(cls) -> str:  # `:917-922`
            return "vllm.compilation.base_static_graph.AbstractStaticGraphWrapper"

        @classmethod
        def stateless_init_device_torch_dist_pg(cls, *args, **kwargs):  # `:924-936`
            raise NotImplementedError

        @classmethod
        def check_if_supports_dtype(cls, dtype: torch.dtype):  # `:938-943`
            raise NotImplementedError

        @classmethod
        def support_hybrid_kv_cache(cls) -> bool:  # `:945-950`
            return False

        @classmethod
        def use_custom_allreduce(cls) -> bool:  # `:864-869`
            return False

        @classmethod
        def get_device_communicator_cls(cls) -> str:  # `:801-806`
            return (
                "vllm.distributed.device_communicators.base_device_communicator."
                "DeviceCommunicatorBase"
            )

        @classmethod
        def validate_request(cls, processed_inputs, params) -> None:  # `:879-885`
            return None

        def get_global_graph_pool(self):  # `:908-915`
            return None

        @classmethod
        def num_compute_units(cls, device_id: int = 0) -> int:  # `:1027-1035`
            raise NotImplementedError("num_compute_units is not implemented")

        @classmethod
        def get_default_ir_op_priority(cls, vllm_config):  # `:1037-1045`
            return None

    platforms_pkg = _ensure("vllm.platforms", package=True)
    interface_mod = _ensure("vllm.platforms.interface")
    for name, value in (("Platform", Platform), ("PlatformEnum", PlatformEnum)):
        _setdefault(platforms_pkg, name, value)
        _setdefault(interface_mod, name, value)

    # --- vllm.v1.attention.* -----------------------------------------------------------------
    #
    # `tileinfer.vllm.attn.backend` needs AttentionBackend/AttentionType/MultipleOf; the metadata
    # builder needs the other four.  They are all here so that running this module *before*
    # `tests/test_attn_contract.py` still leaves that file a complete fake (the reverse order is
    # the common one: the shared fake in `tests/fake_vllm.py` is then reused and only the
    # platform-side names are added).
    class AttentionType(str, enum.Enum):
        DECODER = "decoder"
        ENCODER = "encoder"
        ENCODER_ONLY = "encoder_only"
        ENCODER_DECODER = "encoder_decoder"

    class MultipleOf:
        def __init__(self, base: int):
            self.base = base

    class AttentionBackend:
        supported_dtypes = [torch.float16, torch.bfloat16]
        supported_kv_cache_dtypes = ["auto", "float16", "bfloat16"]
        forward_includes_kv_cache_update = True

        @staticmethod
        def get_supported_kernel_block_sizes():
            return [MultipleOf(1)]

        @classmethod
        def get_kv_cache_block_dim(
            cls,
            block_size: int,
            num_kv_heads: int,
            head_size: int,
            cache_dtype_str: str = "auto",
        ) -> int:
            """``backend.py:98-116`` verbatim: plant a sentinel and read back its dim index.

            Required by ``tests/test_attn_contract.py`` when *this* module ran first: its two
            hybrid-layout tests call the probe on ``TileInferAttentionBackend``.  The comment below
            promises the fake is complete either way, which is only true with this method here.
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

        @classmethod
        def is_mla(cls) -> bool:
            return False

        @classmethod
        def validate_configuration(cls, *args, **kwargs):
            return []

    class AttentionMetadata:
        """``backend.py:352-353``."""

    class AttentionCGSupport(enum.Enum):
        """``backend.py:506-520`` (the order matters: it is used as a level)."""

        ALWAYS = 3
        UNIFORM_BATCH = 2
        UNIFORM_SINGLE_TOKEN_DECODE = 1
        NEVER = 0

    class AttentionMetadataBuilder:
        """``backend.py:523-670``, reduced to the ClassVar lookup."""

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
        """``backend.py:770-850``: a plain class (the real base's ``__new__`` needs vLLM's
        distributed state, which no fake should pretend to have)."""

    @dataclasses.dataclass
    class CommonAttentionMetadata:
        """``backend.py:359-422``: the fields the TileInfer builder reads."""

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
        FLASH_ATTN = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
        CUSTOM = None

    registry_calls: List[Any] = []

    def register_backend(backend, class_path=None, is_mamba=False):
        if class_path is None:
            def decorator(cls):
                registry_calls.append((backend, f"{cls.__module__}.{cls.__qualname__}"))
                return cls

            return decorator
        registry_calls.append((backend, class_path))
        return lambda x: x

    v1 = _ensure("vllm.v1", package=True)
    attention_pkg = _ensure("vllm.v1.attention", package=True)
    attention_backend_mod = _ensure("vllm.v1.attention.backend")
    backends_pkg = _ensure("vllm.v1.attention.backends", package=True)
    registry_mod = _ensure("vllm.v1.attention.backends.registry")
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
        _setdefault(attention_backend_mod, name, value)
    _setdefault(registry_mod, "AttentionBackendEnum", AttentionBackendEnum)
    _setdefault(registry_mod, "register_backend", register_backend)
    _setdefault(registry_mod, "registry_calls", registry_calls)

    # --- vllm.v1.worker.gpu_worker.Worker (vllm__v1__worker__gpu_worker.py:117) ---------------
    class Worker:
        """`:117-1159`, reduced to "a class exists to subclass"."""

    class GPUModelRunner:
        """``gpu_model_runner.py:418-425``, reduced to the construction seam that needs an override.

        ``tileinfer/vllm/worker.py`` subclasses it (``TileInferModelRunner``, one override), so the
        name has to exist here for the worker module to import at all.  The body reproduced is the
        one the override exists for: ``__init__`` -> ``_init_device_properties`` (``:711``) ->
        ``num_compute_units`` (``:1109-1112``), which reaches ``Platform.num_compute_units`` and
        raises (``interface.py:1027-1035``; ``TileInferPlatform`` leaves it unimplemented).
        ``tests/test_worker_contract.py`` asserts the subclass closes that seam.
        """

        def __init__(self, vllm_config: Any, device: torch.device):
            self.vllm_config = vllm_config
            self.device = device
            self.model_memory_usage = 0
            self._init_device_properties()

        def _init_device_properties(self) -> None:
            raise NotImplementedError(
                "num_compute_units is not implemented for the current platform."
            )

    worker_pkg = _ensure("vllm.v1.worker", package=True)
    gpu_worker_mod = _ensure("vllm.v1.worker.gpu_worker")
    gpu_model_runner_mod = _ensure("vllm.v1.worker.gpu_model_runner")
    _setdefault(gpu_worker_mod, "Worker", Worker)
    _setdefault(gpu_model_runner_mod, "GPUModelRunner", GPUModelRunner)

    # --- vllm.config / vllm.compilation.cuda_graph (only what graphs.wrap imports) ------------
    class CUDAGraphMode(enum.Enum):
        """``vllm.config``; the local readers are ``gpu_worker.py:603`` and ``cuda_graph.py:17``."""

        NONE = 0
        PIECEWISE = 1
        FULL = 2

    config_mod = _ensure("vllm.config")
    _setdefault(config_mod, "CUDAGraphMode", CUDAGraphMode)

    class CUDAGraphWrapper:
        """``cuda_graph.py:145-361``: records how it was constructed, and nothing else."""

        instances: List["CUDAGraphWrapper"] = []

        def __init__(self, runnable, vllm_config, runtime_mode, cudagraph_options=None):
            self.runnable = runnable
            self.vllm_config = vllm_config
            self.runtime_mode = runtime_mode
            self.cudagraph_options = cudagraph_options
            CUDAGraphWrapper.instances.append(self)

    compilation_pkg = _ensure("vllm.compilation", package=True)
    cuda_graph_mod = _ensure("vllm.compilation.cuda_graph")
    _setdefault(cuda_graph_mod, "CUDAGraphWrapper", CUDAGraphWrapper)

    # --- attribute chain, so `from vllm.x.y import Z` works -----------------------------------
    # Assigned even when the module already existed: these are the *same* objects, and the chain
    # must point at whatever module object is registered under each name.
    root.platforms = platforms_pkg
    root.v1 = v1
    root.config = config_mod
    root.compilation = compilation_pkg
    platforms_pkg.interface = interface_mod
    v1.attention = attention_pkg
    v1.worker = worker_pkg
    attention_pkg.backend = attention_backend_mod
    attention_pkg.backends = backends_pkg
    backends_pkg.registry = registry_mod
    worker_pkg.gpu_worker = gpu_worker_mod
    worker_pkg.gpu_model_runner = gpu_model_runner_mod
    compilation_pkg.cuda_graph = cuda_graph_mod
    return mode


VLLM_MODE = _install_fake_vllm()

# Imported *after* the fake is in sys.modules: these modules are guarded and would otherwise raise
# their "not importable without vLLM" error — which is exactly what the subprocess test below
# exercises on purpose.
from tileinfer.vllm import graphs  # noqa: E402
from tileinfer.vllm import platform as platform_mod  # noqa: E402
from tileinfer.vllm import plugin as plugin_mod  # noqa: E402
from tileinfer.vllm import worker as worker_mod  # noqa: E402
from tileinfer.vllm.attn.backend import TileInferAttentionBackend  # noqa: E402
from tileinfer.vllm.platform import (  # noqa: E402
    ATTN_BACKEND_CLASS_PATH,
    DOCUMENTED_ABC_SURFACE,
    DOCUMENTED_UNIMPLEMENTED_CAP,
    IMPLEMENTED_ABC_METHODS,
    REASON_DEFAULT_FINE,
    REASON_IMPLEMENTED,
    REASON_MUST_OVERRIDE,
    REASON_UNVERIFIED,
    REASONS,
    SELECTOR_ATTN_TYPE,
    SELECTOR_PLAIN_SWITCHES,
    SELECTOR_REJECT_REASONS,
    SELECTOR_REJECT_SWITCHES,
    SUPPORTED_DTYPES,
    TileInferPlatform,
    UNIMPLEMENTED_ABC_METHODS,
    WORKER_CLASS_PATH,
    WORKER_CLS_ATTR,
    WORKER_CLS_AUTO,
    inventory_problems,
    register,
    unimplemented_report,
)

vllm_platforms = sys.modules["vllm.platforms"]
vllm_registry = sys.modules["vllm.v1.attention.backends.registry"]
vllm_config_mod = sys.modules["vllm.config"]
vllm_cuda_graph = sys.modules["vllm.compilation.cuda_graph"]
PlatformEnum = vllm_platforms.PlatformEnum
AttentionBackendEnum = vllm_registry.AttentionBackendEnum
CUDAGraphMode = vllm_config_mod.CUDAGraphMode

#: Marks the assertions that can only be made against a stub (the real classes are the same shape,
#: so those tests skip rather than pretend when a real vLLM is installed).
USING_FAKE = VLLM_MODE != "real"


# ---------------------------------------------------------------------------------------
# helpers: the ABC surface from the local source, and small config stand-ins
# ---------------------------------------------------------------------------------------
#
# Why the config stand-ins are `SimpleNamespace` and not dataclasses: a dataclass default is a
# *class* attribute, so `delattr(instance, name)` would fall back to it and `hasattr` would still
# be True — which would make `test_check_and_update_config_refuses_to_skip_a_missing_field`
# vacuous.  With a plain namespace, deleting a field really removes it.


def _abc_source_members(path: pathlib.Path = INTERFACE_SOURCE) -> List[str]:
    """Public member names of ``class Platform`` in the local interface.py (via AST)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    platform_cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Platform"
    )
    return sorted(
        node.name
        for node in platform_cls.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    )


def _class_with_members(names: List[str]) -> type:
    """A throwaway class declaring ``names``, so ``inventory_problems`` can be run on any set."""
    return type("Platform", (object,), {name: (lambda self: None) for name in names})


def _source_line(path: pathlib.Path, needle: str, exact: bool = False) -> int:
    """1-based line number of the first line containing (or, ``exact``, equal to) ``needle``.

    The platform's docstrings cite ``file:line`` for every claim about vLLM.  This turns those
    citations into assertions, so refreshing ``runtime/vllm-api/`` to another point release fails
    the contract test instead of silently invalidating the reasoning.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, 1):
        candidate = line.strip() if exact else line
        if (candidate == needle) if exact else (needle in candidate):
            return number
    raise AssertionError(
        f"{path.name}: no line {'equal to' if exact else 'containing'} {needle!r}; "
        "the docstring citing it is stale"
    )


def _selector_config_fields(path: pathlib.Path = SELECTOR_SOURCE) -> List[str]:
    """``AttentionSelectorConfig``'s field names, in declaration order, via AST (never imported)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    config = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AttentionSelectorConfig"
    )
    return [
        node.target.id
        for node in config.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]


class _Selector:
    """Stand-in for ``AttentionSelectorConfig``, field for field.

    The thirteen names *and* their defaults are transcribed from the real source
    (``runtime/vllm-api/vllm__v1__attention__selector.py:21-34``), so this fake cannot drift from
    vLLM without ``test_selector_policy_covers_every_real_field`` failing.  The four geometry fields
    (``head_size``/``dtype``/``kv_cache_dtype``/``block_size``) are carried even though the policy
    does not read them: a fake without them would hide a policy that quietly started depending on
    one, and the platform's docstring says they are enforced elsewhere (the backend's head-layout
    check and ``check_and_update_config``).
    """

    _FIELDS = {
        "head_size": 128,
        "dtype": torch.bfloat16,
        "kv_cache_dtype": "auto",
        "block_size": 2048,
        "use_mla": False,
        "has_sink": False,
        "use_sparse": False,
        "use_mm_prefix": False,
        "use_per_head_quant_scales": False,
        "attn_type": "decoder",
        "use_non_causal": False,
        "use_batch_invariant": False,
        "use_kv_connector": False,
    }

    def __init__(self, **overrides):
        unknown = set(overrides) - set(self._FIELDS)
        assert not unknown, f"AttentionSelectorConfig has no field(s) {sorted(unknown)}"
        fields = dict(self._FIELDS)
        fields.update(overrides)
        for name, value in fields.items():
            setattr(self, name, value)


def _cache_config(**overrides) -> types.SimpleNamespace:
    """``CacheConfig`` stand-in carrying the fields the platform reads.

    ``mamba_cache_mode`` defaults to ``"align"`` because that is what vLLM 0.23 turns ``"none"``
    into for a hybrid model with prefix caching enabled (v0.23.0
    ``model_executor/models/config.py:350-386``), and Qwen3.5/3.6 refuses ``"all"``
    (``model_executor/models/qwen3_5.py:459-463``).
    """
    fields = {
        "block_size": 2048,
        "cache_dtype": "auto",
        "user_specified_block_size": False,
        "mamba_cache_mode": "align",
        "mamba_block_size": None,
        "mamba_page_size_padded": None,
    }
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def _model_config(**overrides) -> types.SimpleNamespace:
    """``ModelConfig`` stand-in with Qwen3.6's kv geometry and the flags the platform reads.

    ``get_num_kv_heads``/``get_head_size``/``dtype`` are the three inputs of the attention page
    size; the upstream hybrid helper calls the same two methods (``interface.py:568``,
    ``:612``), which is why the platform may rely on them here.
    """

    def get_num_kv_heads(parallel_config=None) -> int:  # noqa: ARG001 - mirrors ModelConfig
        return 2

    def get_head_size() -> int:
        return 256

    fields = {
        "is_hybrid": True,
        "use_mla": False,
        "enable_sleep_mode": False,
        "enforce_eager": False,
        "dtype": torch.bfloat16,
        "get_num_kv_heads": get_num_kv_heads,
        "get_head_size": get_head_size,
    }
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def _parallel_config(**overrides) -> types.SimpleNamespace:
    """``ParallelConfig`` stand-in: the field defaults to the real ``"auto"`` sentinel.

    ``parallel.py:248`` declares ``worker_cls: str = "auto"``, so the helper must start there - a
    helper that pre-set the CUDA worker would test a config vLLM never produces by default.
    """
    fields = {"worker_cls": WORKER_CLS_AUTO}
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def _compilation_config(**overrides) -> types.SimpleNamespace:
    fields = {"cudagraph_mode": CUDAGraphMode.NONE}
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def _config(**overrides) -> types.SimpleNamespace:
    """A vLLM config with individual sections replaced by keyword (``cache_config=...``)."""
    config = types.SimpleNamespace(
        cache_config=_cache_config(),
        model_config=_model_config(),
        parallel_config=_parallel_config(),
        compilation_config=_compilation_config(),
        speculative_config=None,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


# =======================================================================================
# 1. the vllm.platform_plugins entry point
# =======================================================================================


def test_register_returns_the_class_path_when_an_npu_is_visible(monkeypatch):
    """``vllm/platforms/__init__.py:217-222`` takes a *string* class path from the plugin."""
    monkeypatch.setattr(platform_mod, "_npu_available", lambda: True)
    path = register()
    assert path == "tileinfer.vllm.platform.TileInferPlatform"
    # vLLM resolves the string and instantiates it with no arguments (`:277-278`).
    resolved = getattr(importlib.import_module("tileinfer.vllm.platform"), "TileInferPlatform")
    assert TileInferPlatform is resolved
    TileInferPlatform()  # must not need arguments


def test_register_is_stable_across_the_two_calls_vllm_makes(monkeypatch):
    """The winning plugin is invoked twice (`:221`, then `:237`); the answer must not change."""
    monkeypatch.setattr(platform_mod, "_npu_available", lambda: True)
    assert register() == register()


def test_register_returns_none_without_an_npu():
    """``None`` means "not applicable" so the built-in plugins are still considered (`:244-250`)."""
    if platform_mod._npu_available():  # a real device box
        pytest.skip("this machine has an NPU; the None branch cannot be observed here")
    assert register() is None


def test_platform_import_fails_clearly_outside_vllm():
    """The guarded import must name vLLM and the version, not leak a vllm-internal traceback."""
    if VLLM_MODE == "real":
        pytest.skip("a real vLLM is installed, so the import succeeds by design")
    code = "import tileinfer.vllm.platform"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=180,
        env=dict(os.environ),  # PYTHONPATH must stay: it is where this box's torch lives
    )
    assert result.returncode != 0
    assert "tileinfer.vllm.platform requires vLLM 0.23" in result.stderr
    assert "Install vllm>=0.23,<0.24" in result.stderr


# =======================================================================================
# 2. device identity
# =======================================================================================


def test_device_identity():
    assert TileInferPlatform._enum is PlatformEnum.OOT
    assert TileInferPlatform.device_name == "npu"
    assert TileInferPlatform.device_type == "npu"
    assert TileInferPlatform.dispatch_key == "NPU"
    assert TileInferPlatform.device_control_env_var == "ASCEND_RT_VISIBLE_DEVICES"
    # Derived from _enum, and what routes CustomOp.dispatch_forward to forward_oot.
    assert TileInferPlatform().is_out_of_tree() is True
    assert TileInferPlatform().is_cuda() is False
    assert TileInferPlatform().is_cuda_alike() is False


def test_supported_dtypes_match_the_attention_backend():
    """The platform's dtype list and the backend's must not drift: one decides, the other rejects."""
    assert tuple(SUPPORTED_DTYPES) == tuple(TileInferPlatform().supported_dtypes)
    assert tuple(SUPPORTED_DTYPES) == (torch.bfloat16, torch.float16)
    assert set(SUPPORTED_DTYPES) == set(TileInferAttentionBackend.supported_dtypes)


# =======================================================================================
# 3. attention backend selection
# =======================================================================================


def test_attn_backend_cls_returns_that_backend_for_plain_mha():
    path = TileInferPlatform.get_attn_backend_cls(None, _Selector(), num_heads=16)
    assert path == ATTN_BACKEND_CLASS_PATH
    # The literal must be the class this repository actually ships, not a stale string.
    assert path == f"{TileInferAttentionBackend.__module__}.{TileInferAttentionBackend.__qualname__}"


def test_attn_backend_cls_returns_that_backend_for_plain_gqa():
    """"Plain MHA/GQA" includes GQA: the group is checked by the backend, not by this hook."""
    assert TileInferPlatform.get_attn_backend_cls(None, _Selector()) == ATTN_BACKEND_CLASS_PATH


@pytest.mark.parametrize("switch", SELECTOR_REJECT_SWITCHES + SELECTOR_PLAIN_SWITCHES)
def test_attn_backend_cls_rejects_each_special_selector(switch):
    selector = _Selector(**{switch: True})
    with pytest.raises(ValueError) as excinfo:
        TileInferPlatform.get_attn_backend_cls(None, selector)
    assert switch in str(excinfo.value)


@pytest.mark.parametrize("attn_type", ["encoder", "encoder_only", "encoder_decoder"])
def test_attn_backend_cls_rejects_a_non_decoder_attn_type(attn_type):
    """``attn_type`` is a real selector field (``selector.py:31``) and not a boolean.

    ``AttentionType`` is a ``str`` Enum (``v1/attention/backend.py:32-45``) with four members; only
    ``decoder`` is served.  Passing the value as a bare string and as the enum member must both be
    rejected, because vLLM itself can hand the hook either (the field is annotated ``str``).
    """
    if USING_FAKE:
        attn_enum = sys.modules["vllm.v1.attention.backend"].AttentionType
        selector = _Selector(attn_type=attn_enum(attn_type))
    else:  # pragma: no cover - needs a real vLLM
        selector = _Selector(attn_type=attn_type)
    with pytest.raises(ValueError) as excinfo:
        TileInferPlatform.get_attn_backend_cls(None, selector)
    assert "attn_type" in str(excinfo.value)
    # ... and the bare-string spelling of the same value is rejected too.
    with pytest.raises(ValueError, match="attn_type"):
        TileInferPlatform.get_attn_backend_cls(None, _Selector(attn_type=str(attn_type)))


def test_selector_policy_covers_every_real_field():
    """The tuples must be vLLM 0.23's field names — no invented ones, no silent field.

    ``AttentionSelectorConfig`` is read (AST, never imported) from the same local subset the
    platform's docstrings cite.  A vLLM point release that renames or adds a field fails here, which
    is the only way this policy can stay honest across a version bump.
    """
    if not SELECTOR_SOURCE.is_file():
        pytest.skip(f"{SELECTOR_SOURCE.name} is not in this checkout")
    fields = _selector_config_fields()
    # The four geometry fields plus attn_type are handled outside the switch tuples (see
    # _selector_switches' docstring), and the eight booleans are exactly the two tuples.
    geometry = {"head_size", "dtype", "kv_cache_dtype", "block_size"}
    assert set(fields) == geometry | {"attn_type"} | set(
        SELECTOR_REJECT_SWITCHES + SELECTOR_PLAIN_SWITCHES
    ), "the selector policy and the real AttentionSelectorConfig disagree"
    # The specific name this repo used to read and vLLM never had.
    assert "use_compress" not in fields
    assert "use_compress" not in SELECTOR_REJECT_SWITCHES + SELECTOR_PLAIN_SWITCHES
    # Every switch has a stated reason (the rejection message prints it).
    for switch in SELECTOR_REJECT_SWITCHES + SELECTOR_PLAIN_SWITCHES:
        assert switch in SELECTOR_REJECT_REASONS and SELECTOR_REJECT_REASONS[switch]
    # The only attn_type served is the field's own default (selector.py:31).
    assert SELECTOR_ATTN_TYPE == "decoder"
    assert _Selector().attn_type == SELECTOR_ATTN_TYPE


def test_attn_backend_cls_rejects_a_none_selector():
    with pytest.raises(ValueError, match="attn_selector_config=None"):
        TileInferPlatform.get_attn_backend_cls(None, None)


def test_attn_backend_cls_refuses_to_classify_an_unknown_selector_object():
    """An unreadable selector is not evidence that the layer is plain — refuse, and say why."""
    with pytest.raises(NotImplementedError) as excinfo:
        TileInferPlatform.get_attn_backend_cls(None, object())
    message = str(excinfo.value)
    assert "selector.py" in message  # names the file to fetch
    assert "SELECTOR_REJECT_SWITCHES" in message  # names the constant to update


def test_attn_backend_cls_warns_when_a_switch_is_missing(caplog):
    """A partially-readable selector classifies, but says which switches it could not see."""

    class _Partial:
        use_mla = False

    with caplog.at_level("WARNING", logger="tileinfer.vllm.platform"):
        path = TileInferPlatform.get_attn_backend_cls(None, _Partial())
    assert path == ATTN_BACKEND_CLASS_PATH
    assert "use_sparse" in caplog.text and "use_batch_invariant" in caplog.text


def test_attn_backend_cls_accepts_custom_and_rejects_a_foreign_backend():
    if not USING_FAKE:
        pytest.skip("the enum identity check needs the stub registry")
    assert (
        TileInferPlatform.get_attn_backend_cls(AttentionBackendEnum.CUSTOM, _Selector())
        == ATTN_BACKEND_CLASS_PATH
    )
    with pytest.raises(ValueError, match="v0 provides exactly one"):
        TileInferPlatform.get_attn_backend_cls(AttentionBackendEnum.FLASH_ATTN, _Selector())
    with pytest.raises(ValueError, match="v0 provides exactly one"):
        TileInferPlatform.get_attn_backend_cls("flash_attn", _Selector())


def test_attn_backend_cls_rejects_the_empty_backend_string():
    """``""`` is ``AttentionBackendEnum.TORCH_SDPA`` (ViT only), **not** a synonym for custom.

    ``backends/registry.py:61`` gives that member the empty string precisely so it cannot alias
    another backend, so accepting ``""`` here would claim a ViT-only tag for a decoder layer.
    """
    with pytest.raises(ValueError, match="TORCH_SDPA"):
        TileInferPlatform.get_attn_backend_cls("", _Selector())
    with pytest.raises(ValueError, match="TORCH_SDPA"):
        TileInferPlatform.get_attn_backend_cls("   ", _Selector())


def test_custom_backend_enum_compares_by_identity_not_falsiness():
    """``AttentionBackendEnum.CUSTOM = None`` (registry.py:99-101): the *value* is falsy.

    A ``if not selected_backend`` test would therefore treat CUSTOM as "auto" and a foreign enum
    member as truthy for the wrong reason; the identity check is what keeps the two apart.
    """
    if not USING_FAKE:
        pytest.skip("the enum value shape is asserted against the stub registry")
    assert AttentionBackendEnum.CUSTOM.value is None
    assert TileInferPlatform.get_attn_backend_cls(
        AttentionBackendEnum.CUSTOM, _Selector()
    ) == ATTN_BACKEND_CLASS_PATH
    assert not bool(AttentionBackendEnum.CUSTOM.value)  # the trap the identity check avoids


# =======================================================================================
# 4. check_and_update_config: rejections
# =======================================================================================


def test_check_and_update_config_rejects_speculative_decoding():
    config = _config(speculative_config=types.SimpleNamespace(num_speculative_tokens=1))
    with pytest.raises(ValueError, match="speculative decoding"):
        TileInferPlatform.check_and_update_config(config)


@pytest.mark.parametrize("cache_dtype", ["fp8", "bfloat16", "float16"])
def test_check_and_update_config_rejects_a_non_auto_cache_dtype(cache_dtype):
    config = _config(cache_config=_cache_config(cache_dtype=cache_dtype))
    with pytest.raises(ValueError, match="kv-cache dtype"):
        TileInferPlatform.check_and_update_config(config)


def test_check_and_update_config_rejects_sleep_mode():
    config = _config(model_config=_model_config(enable_sleep_mode=True))
    with pytest.raises(ValueError, match="sleep mode"):
        TileInferPlatform.check_and_update_config(config)


def test_check_and_update_config_rejects_mla():
    config = _config(model_config=_model_config(use_mla=True))
    with pytest.raises(ValueError, match="MLA"):
        TileInferPlatform.check_and_update_config(config)


@pytest.mark.parametrize(
    "section,name,owner",
    [
        ("speculative_config", "speculative_config", "VllmConfig"),
        ("cache_config", "cache_dtype", "CacheConfig"),
        ("model_config", "enable_sleep_mode", "ModelConfig"),
        ("model_config", "use_mla", "ModelConfig"),
    ],
)
def test_check_and_update_config_refuses_to_skip_a_missing_field(section, name, owner):
    """A field the local source names but the config lacks must not silently pass the check."""
    config = _config()
    delattr(config if section == "speculative_config" else getattr(config, section), name)
    with pytest.raises(RuntimeError) as excinfo:
        TileInferPlatform.check_and_update_config(config)
    message = str(excinfo.value)
    assert name in message and owner in message
    assert "cannot be evaluated" in message


# =======================================================================================
# 5. check_and_update_config: updates
# =======================================================================================


def test_check_and_update_config_replaces_the_auto_worker_sentinel():
    """The mechanism is ``parallel_config.worker_cls`` (+ the ``"auto"`` sentinel), not a hook."""
    config = _config()
    assert config.parallel_config.worker_cls == WORKER_CLS_AUTO == "auto"
    TileInferPlatform.check_and_update_config(config)
    assert config.parallel_config.worker_cls == WORKER_CLASS_PATH
    assert WORKER_CLASS_PATH == "tileinfer.vllm.worker.NPUWorker"
    # The two modules must agree on the path, or the platform points at a renamed class.
    assert WORKER_CLASS_PATH == worker_mod.WORKER_CLASS_PATH
    module_path, class_name = WORKER_CLASS_PATH.rsplit(".", 1)
    assert hasattr(importlib.import_module(module_path), class_name)


def test_worker_cls_is_the_field_vllm_reads_and_the_in_tree_platforms_write():
    """Pin every line the platform's docstrings cite, so a source refresh fails here first.

    Four facts, in the order vLLM uses them: the field exists with the sentinel default; the worker
    wrapper resolves it with ``resolve_obj_by_qualname``; the two in-tree platforms replace the
    sentinel in ``check_and_update_config``; and there is no ``Platform.get_worker_cls`` hook.
    """
    assert _source_line(PARALLEL_CONFIG_SOURCE, 'worker_cls: str = "auto"') == 248
    assert _source_line(WORKER_BASE_SOURCE, "if isinstance(parallel_config.worker_cls, str):") == 250
    assert _source_line(WORKER_BASE_SOURCE, "parallel_config.worker_cls", exact=True) == 252
    assert _source_line(CPU_PLATFORM_SOURCE, 'if parallel_config.worker_cls == "auto":') == 153
    assert (
        _source_line(
            CPU_PLATFORM_SOURCE, 'parallel_config.worker_cls = "vllm.v1.worker.cpu_worker.CPUWorker"'
        )
        == 154
    )

    assert WORKER_CLS_ATTR == "worker_cls"
    assert WORKER_CLS_AUTO == "auto"
    interface = INTERFACE_SOURCE.read_text(encoding="utf-8")
    assert "def get_worker_cls" not in interface, (
        "vLLM 0.23 has no Platform.get_worker_cls hook; if a refresh adds one, the platform's "
        "docstring and this test must say which mechanism wins"
    )
    plugins = PLUGINS_SOURCE.read_text(encoding="utf-8")
    assert "worker_plugins" not in plugins, (
        "a worker plugin group would be a second mechanism; the vendored 0.23 plugins module "
        "defines only general/io_processor/platform/stat_logger groups"
    )


def test_check_and_update_config_keeps_an_already_correct_worker_class():
    """Idempotent: a vLLM/launcher that already wrote our class must not be treated as a conflict."""
    config = _config(parallel_config=_parallel_config(worker_cls=WORKER_CLASS_PATH))
    TileInferPlatform.check_and_update_config(config)
    assert config.parallel_config.worker_cls == WORKER_CLASS_PATH


def test_check_and_update_config_refuses_a_conflicting_worker_class():
    """An explicit foreign worker is a conflict, not something to overwrite behind the user's back."""
    config = _config(
        parallel_config=_parallel_config(worker_cls="vllm.v1.worker.gpu_worker.Worker")
    )
    with pytest.raises(ValueError) as excinfo:
        TileInferPlatform.check_and_update_config(config)
    message = str(excinfo.value)
    assert "vllm.v1.worker.gpu_worker.Worker" in message
    assert WORKER_CLASS_PATH in message
    assert "--worker-cls" in message
    assert "vllm__v1__worker__worker_base.py:249-253" in message


def test_check_and_update_config_refuses_without_the_worker_cls_attribute():
    """The field is verified; a config without it is not a 0.23 ParallelConfig, so fail loudly."""
    config = _config(parallel_config=types.SimpleNamespace())
    with pytest.raises(NotImplementedError) as excinfo:
        TileInferPlatform.check_and_update_config(config)
    message = str(excinfo.value)
    assert "worker_cls" in message
    assert "vllm__config__parallel.py:248" in message


def test_check_and_update_config_aligns_an_unpinned_block_size_to_the_mamba_page():
    config = _config(cache_config=_cache_config(block_size=16, user_specified_block_size=False))
    TileInferPlatform.check_and_update_config(config)
    assert config.cache_config.block_size == 2048


def test_check_and_update_config_rejects_a_user_pinned_misaligned_block_size():
    config = _config(cache_config=_cache_config(block_size=1024, user_specified_block_size=True))
    with pytest.raises(ValueError, match="2048"):
        TileInferPlatform.check_and_update_config(config)


def test_check_and_update_config_rejects_a_block_size_that_is_not_a_kernel_page_multiple():
    """Non-hybrid model: only the 128-token kernel page applies (buckets.py:52)."""
    config = _config(
        model_config=_model_config(is_hybrid=False),
        cache_config=_cache_config(block_size=100, user_specified_block_size=True),
    )
    with pytest.raises(ValueError, match="128"):
        TileInferPlatform.check_and_update_config(config)


@pytest.mark.parametrize("block_size", [2048, 4096, 128 * 16])
def test_check_and_update_config_accepts_aligned_block_sizes(block_size):
    config = _config(cache_config=_cache_config(block_size=block_size))
    TileInferPlatform.check_and_update_config(config)
    assert config.cache_config.block_size == block_size


def test_check_and_update_config_pads_the_mamba_state_page_to_the_attention_page():
    """A hybrid run needs the state page and the attention page to be the *same* size.

    ``interface.py:679-699`` is the upstream rule (pad, never shrink); a merely "divisible" state
    page cannot be reconciled later, because a padded ``MambaSpec`` page ignores ``block_size``
    (v0.23.0 ``v1/core/kv_cache_utils.py:1029-1066``), so the platform applies the rule here -
    nothing in the local source calls ``update_block_size_for_backend``.
    """
    config = _config()
    TileInferPlatform.check_and_update_config(config)
    assert config.cache_config.block_size == 2048
    assert config.cache_config.mamba_block_size == 2048
    # Qwen3.6: 2 (K/V) * 2 kv_heads * 256 head_dim * 2 bytes (bf16) = 2048 B per token.
    assert config.cache_config.mamba_page_size_padded == 2048 * 2048 == 4 * 1024 * 1024
    # Even a config that already carries 2048 is marked, so the kernel-minimum rule of
    # update_block_size_for_backend phase 1 cannot reinterpret it.
    assert config.cache_config.user_specified_block_size is True


def test_check_and_update_config_fixes_the_hybrid_page_against_vllms_later_hook():
    """An unpinned 16-token default is aligned *and* marked as specified.

    ``update_block_size_for_backend`` phase 1 replaces an unspecified block size with the
    backend's preferred (kernel) minimum (``interface.py:511-522``) - 128 here, not 2048 - so an
    unpinned page must be marked or that later call would undo the hybrid page.
    """
    config = _config(cache_config=_cache_config(block_size=16, user_specified_block_size=False))
    TileInferPlatform.check_and_update_config(config)
    assert config.cache_config.block_size == 2048
    assert config.cache_config.user_specified_block_size is True


def test_check_and_update_config_keeps_a_pinned_page_and_sizes_the_mamba_page_from_it():
    config = _config(cache_config=_cache_config(block_size=4096, user_specified_block_size=True))
    TileInferPlatform.check_and_update_config(config)
    assert config.cache_config.block_size == 4096
    assert config.cache_config.mamba_block_size == 4096
    assert config.cache_config.mamba_page_size_padded == 4096 * 2048


def test_check_and_update_config_does_not_touch_the_mamba_unit_in_none_mode():
    """``"none"`` keeps one state per request, so its unit is ``max_model_len``, not the page."""
    config = _config(
        cache_config=_cache_config(mamba_cache_mode="none", mamba_block_size=32768)
    )
    TileInferPlatform.check_and_update_config(config)
    assert config.cache_config.mamba_block_size == 32768
    # The physical page is still padded to the attention page (upstream does this in both modes).
    assert config.cache_config.mamba_page_size_padded == 2048 * 2048


@pytest.mark.parametrize("mode", ["all", "bogus"])
def test_check_and_update_config_rejects_an_unsupported_mamba_cache_mode(mode):
    config = _config(cache_config=_cache_config(mamba_cache_mode=mode))
    with pytest.raises(ValueError, match="mamba-cache-mode"):
        TileInferPlatform.check_and_update_config(config)


def test_check_and_update_config_leaves_a_non_hybrid_model_without_mamba_pages():
    config = _config(
        model_config=_model_config(is_hybrid=False),
        cache_config=_cache_config(block_size=128, user_specified_block_size=True),
    )
    TileInferPlatform.check_and_update_config(config)
    assert config.cache_config.mamba_block_size is None
    assert config.cache_config.mamba_page_size_padded is None


def test_check_and_update_config_forces_eager_while_capture_is_unverified(caplog):
    """No bridge on this device → vLLM must not attempt capture, and the reason is logged."""
    config = _config()
    with caplog.at_level("WARNING", logger="tileinfer.vllm.platform"):
        TileInferPlatform.check_and_update_config(config)
    if graphs.capture_path_ready()[0]:
        assert config.model_config.enforce_eager is False
        return
    assert config.model_config.enforce_eager is True
    assert "enforce_eager" in caplog.text


# =======================================================================================
# 6. the ABC inventory (the ratchet)
# =======================================================================================


def test_inventory_is_complete_against_the_local_abc_source():
    """Every public member of the real ABC must be accounted for.

    The base class is rebuilt from ``runtime/vllm-api/vllm__platforms__interface.py`` rather than
    from the stub, so this fails when the **local vLLM source** grows a method (e.g. when the
    runtime subset is refreshed for 0.24) - not merely when the fake does.
    """
    members = _abc_source_members()
    assert inventory_problems(base=_class_with_members(members)) == []


def test_inventory_matches_the_stub_base_when_one_is_in_use():
    if not USING_FAKE:
        pytest.skip("a real vLLM is installed; the real base is checked by the test above")
    # The stub declares a subset of the ABC, so "listed but the base does not declare it" entries
    # are expected here; every other category must be empty.
    problems = inventory_problems()
    assert [p for p in problems if "does not declare it" not in p] == []


def test_inventory_catches_a_newly_added_abc_method():
    members = _abc_source_members() + ["a_brand_new_vllm_hook"]
    problems = inventory_problems(base=_class_with_members(members))
    assert any("a_brand_new_vllm_hook" in problem for problem in problems)


def test_inventory_catches_a_claim_the_class_does_not_back_up(monkeypatch):
    """A name claimed 'implemented' but not overridden on the class must be reported."""
    monkeypatch.setitem(
        IMPLEMENTED_ABC_METHODS,
        "get_device_uuid",
        platform_mod.AbcMethod(REASON_IMPLEMENTED, "pretend this was implemented"),
    )
    problems = inventory_problems(base=_class_with_members(_abc_source_members()))
    assert any("get_device_uuid" in p and "not overridden" in p for p in problems)


def test_inventory_catches_a_todo_entry_that_was_actually_implemented(monkeypatch):
    """The inverse: implementing a method without moving it out of the TODO list must fail."""
    monkeypatch.setitem(
        UNIMPLEMENTED_ABC_METHODS,
        "set_device",
        platform_mod.AbcMethod(REASON_UNVERIFIED, "pretend this was not implemented"),
    )
    problems = inventory_problems(base=_class_with_members(_abc_source_members()))
    assert any("set_device" in p and "still listed as unimplemented" in p for p in problems)


def test_documented_counts_are_the_inventory_counts():
    report = unimplemented_report()
    assert report["totals"] == {
        "abc_surface": 75,
        "implemented": 10,
        "unimplemented": 65,
        "unimplemented_cap": 65,
        "extra_helpers": 2,
    }
    assert report["by_reason"] == {
        REASON_MUST_OVERRIDE: 6,
        REASON_UNVERIFIED: 3,
        REASON_DEFAULT_FINE: 56,
    }
    assert sum(report["by_reason"].values()) == report["totals"]["unimplemented"]
    assert report["totals"]["implemented"] + report["totals"]["unimplemented"] == (
        report["totals"]["abc_surface"]
    )


def test_unimplemented_count_cannot_grow_without_editing_the_cap():
    """The ratchet: 65 today, and the documented cap is the only thing that may move."""
    assert len(UNIMPLEMENTED_ABC_METHODS) <= DOCUMENTED_UNIMPLEMENTED_CAP
    assert len(UNIMPLEMENTED_ABC_METHODS) == DOCUMENTED_UNIMPLEMENTED_CAP, (
        "the TODO inventory changed size; if a method was implemented, lower "
        "DOCUMENTED_UNIMPLEMENTED_CAP deliberately, and if one was discovered, raise it - but a "
        "silent change is what this test exists to stop"
    )
    assert DOCUMENTED_ABC_SURFACE == len(_abc_source_members())


def test_every_reason_is_one_of_the_four_codes():
    for name, entry in {**IMPLEMENTED_ABC_METHODS, **UNIMPLEMENTED_ABC_METHODS}.items():
        assert entry.reason in REASONS, name
        assert entry.note, f"{name} has no note"


def test_implemented_methods_really_override_and_todo_ones_really_do_not():
    overridden = set(vars(TileInferPlatform))
    assert set(IMPLEMENTED_ABC_METHODS) <= overridden
    assert set(UNIMPLEMENTED_ABC_METHODS) & overridden == set()
    assert REASON_IMPLEMENTED == "implemented"


@pytest.mark.parametrize(
    "call",
    [
        lambda: TileInferPlatform.get_device_uuid(),
        lambda: TileInferPlatform.manual_seed_all(0),
        lambda: TileInferPlatform.get_current_memory_usage(),
        lambda: TileInferPlatform.get_punica_wrapper(),
        lambda: TileInferPlatform.num_compute_units(),
    ],
)
def test_the_inherited_methods_still_raise(call):
    """The reviewed "must override" bodies are still the base ones (documented, not silent)."""
    with pytest.raises(NotImplementedError):
        call()


def test_hybrid_kv_cache_is_supported():
    """The ABC default False makes vLLM 0.23 reject this model's spec mix at start-up.

    ``interface.py:945-950`` is the default; with it vLLM turns the hybrid KV cache manager off
    (v0.23.0 ``config/vllm.py:1453-1455``) and then refuses to group attention and mamba specs
    together (v0.23.0 ``v1/core/kv_cache_utils.py:1421-1428``).  False would therefore leave the
    30 GDN layers without state tensors *and* abort the run; the platform's whole contribution is
    this bit plus the page sizes.
    """
    assert TileInferPlatform.support_hybrid_kv_cache() is True
    assert "support_hybrid_kv_cache" in vars(TileInferPlatform)
    assert "support_hybrid_kv_cache" in IMPLEMENTED_ABC_METHODS
    assert "support_hybrid_kv_cache" not in UNIMPLEMENTED_ABC_METHODS
    assert IMPLEMENTED_ABC_METHODS["support_hybrid_kv_cache"].reason == REASON_IMPLEMENTED


def test_hybrid_documentation_drops_the_old_skeleton_rationale():
    """What is left is a hardware risk, not a capability gap dressed up as one."""
    source = pathlib.Path(platform_mod.__file__).read_text(encoding="utf-8")
    doc = inspect.getdoc(TileInferPlatform.support_hybrid_kv_cache) or ""
    for stale in (
        "stays default-off",
        "has not been observed on hardware",
        "worker.py is a skeleton",
    ):
        assert stale not in source, f"the stale hybrid rationale is still in platform.py: {stale!r}"
    assert "HARDWARE" in doc, "the remaining risk must be marked as a hardware item"
    assert "initialize_mamba_ssu_backend" in doc  # the SSU call that is a no-op for GDN
    # The old doc claimed SSU would be the first device failure. For GDN it is never reached,
    # and the platforms now refuses MAMBA1/MAMBA2 at start-up instead; the decision names the
    # mechanism it rests on (the private registry and the absent registration hook).
    for needle in (
        "GDN_ATTN",
        "private dict",
        "register_backend(..., is_mamba=True)",
        "require_supported_mamba_state_path",
        "fused_recurrent_gated_delta_rule_packed_decode",
    ):
        assert needle in doc, f"the mamba-state decision must be documented: {needle!r}"
    assert "expected to stop here first" not in doc
    # The old write-path gap ("KV pool must be contiguous") is closed by the block-first shape and
    # the declared stride order; what remains is the zeroer's outcome on a device run.
    assert "get_kv_cache_stride_order" in doc
    assert "(1, 0, 2, 3, 4)" in doc
    assert "KVBlockZeroer" in doc
    assert "KV pool must be contiguous" not in doc
    assert "interface.py:945-950" in doc  # the ABC default this override replaces
    assert "gpu_model_runner.py:7301" in doc  # the mamba half of initialize_kv_cache


def test_hybrid_kv_cache_does_not_enable_anything_else():
    """The bit is the only new capability: no custom KV spec class, no mamba backend registered."""
    assert "register_custom_kv_cache_specs" in UNIMPLEMENTED_ABC_METHODS
    assert not hasattr(platform_mod, "MambaSpec")
    assert "MambaAttentionBackend" not in vars(platform_mod)


def test_verify_model_arch_is_the_abc_pass_and_not_an_allow_list():
    """An OOT platform vetoes architectures; it does not have to whitelist them.

    ``interface.py:708-718`` is a ``pass`` ("By default all models are considered supported"), and
    ``registry.py:945`` calls it right before a model class is imported.  The Qwen3.5/3.6 family is
    accepted by the registry table, so overriding this hook would *break* every arch it omits.
    """
    assert "verify_model_arch" not in vars(TileInferPlatform), (
        "the ABC body is already correct; an allow-list here would reject architectures vLLM "
        "registers itself"
    )
    assert "verify_model_arch" in UNIMPLEMENTED_ABC_METHODS
    assert UNIMPLEMENTED_ABC_METHODS["verify_model_arch"].reason == REASON_DEFAULT_FINE
    # The inherited body does not raise for the two arch spellings this deployment can load.
    for arch in (
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5ForCausalLM",
    ):
        assert TileInferPlatform.verify_model_arch(arch) is None


def test_the_registry_accepts_the_qwen35_family_without_a_platform_hook():
    """Cite the acceptance path: call site, registry entries, suffix normalisation, no self-register."""
    assert _source_line(REGISTRY_SOURCE, "current_platform.verify_model_arch(model_arch)") == 945
    assert _source_line(REGISTRY_SOURCE, "def _try_load_model_cls(") == 939
    assert _source_line(REGISTRY_SOURCE, "def _normalize_arch(") == 1134
    assert _source_line(REGISTRY_SOURCE, "_VLLM_MODELS = {") == 693
    # The two ConditionalGeneration spellings are registered against the qwen3_5 module.
    assert _source_line(REGISTRY_SOURCE, '"Qwen3_5ForConditionalGeneration"') == 569
    assert _source_line(REGISTRY_SOURCE, '"Qwen3_5MoeForConditionalGeneration"') == 570
    registry = REGISTRY_SOURCE.read_text(encoding="utf-8")
    assert '"Qwen3_5ForConditionalGeneration": ("qwen3_5", "Qwen3_5ForConditionalGeneration")' in (
        registry
    )
    moe_entry = registry.split('"Qwen3_5MoeForConditionalGeneration"', 1)[1].split(")", 1)[0]
    assert '"qwen3_5"' in moe_entry and '"Qwen3_5MoeForConditionalGeneration"' in moe_entry

    # A bare ...ForCausalLM spelling is normalised onto the registered suffix: the two suffixes are
    # both architecture defaults, and `_normalize_arch` rewrites one into the other.
    assert _source_line(MODEL_CONFIG_SOURCE, '("ForCausalLM", ("generate", "none")),') == 1913
    assert _source_line(
        MODEL_CONFIG_SOURCE, '("ForConditionalGeneration", ("generate", "none")),'
    ) == 1914

    # qwen3_5.py does not register itself - registration is the central lazy table, so there is no
    # import-time side effect a platform could depend on (and no try/except to work around).
    qwen = QWEN35_SOURCE.read_text(encoding="utf-8")
    assert "register_model" not in qwen
    for cls in (
        "Qwen3_5ForCausalLM",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    ):
        assert f"class {cls}" in qwen, f"qwen3_5.py no longer defines {cls}"


def test_platform_helpers_fail_closed_without_torch_npu():
    if platform_mod._npu_available():
        pytest.skip("this machine has an NPU; the failure branches cannot be observed here")
    with pytest.raises(RuntimeError, match="torch_npu"):
        TileInferPlatform.get_device_name()
    with pytest.raises(RuntimeError, match="torch_npu"):
        TileInferPlatform.get_device_total_memory()
    with pytest.raises(RuntimeError, match="torch_npu"):
        TileInferPlatform.set_device(torch.device("cpu"))
    with pytest.raises(RuntimeError, match="torch_npu"):
        TileInferPlatform.empty_cache()
    with pytest.raises(RuntimeError, match="torch_npu"):
        TileInferPlatform.synchronize()


def test_pin_memory_and_dtype_answers():
    assert TileInferPlatform.is_pin_memory_available() is False
    TileInferPlatform.check_if_supports_dtype(torch.bfloat16)
    TileInferPlatform.check_if_supports_dtype(torch.float16)
    with pytest.raises(ValueError, match="bf16/fp16"):
        TileInferPlatform.check_if_supports_dtype(torch.float32)


@pytest.mark.parametrize("quant", ["fp8", "compressed-tensors", "awq", "gptq"])
def test_verify_quantization_rejects_everything(quant):
    with pytest.raises(ValueError, match="quantisation"):
        TileInferPlatform.verify_quantization(quant)


@pytest.mark.parametrize("quant", ["", "none", "None"])
def test_verify_quantization_accepts_the_no_quantisation_spellings(quant):
    TileInferPlatform.verify_quantization(quant)


# =======================================================================================
# 7. the graph-capture decision
# =======================================================================================


#: The local vLLM sources the alias inventory is derived from (read, never imported).
VLLM_API_DIR = ROOT / "runtime" / "vllm-api"

#: ``torch.cuda.<name>`` as it appears in those sources.  Deliberately a regex over the text, not
#: an import: a new upstream call site must show up in the table/coverage test, not at capture.
TORCH_CUDA_CALL = re.compile(r"torch\.cuda\.([A-Za-z_][A-Za-z0-9_]*)")

#: One distinct object per counterpart, so an identity check can tell torch.npu's object from a
#: look-alike (the whole point of ``capture_path_ready``).
_NPU_COUNTERPARTS: Dict[str, Any] = {
    "CUDAGraph": type("NPUCUDAGraph", (), {}),
    "graph": lambda *args, **kwargs: None,
    "mem_get_info": lambda device=None: (1, 1),
    "Stream": type("NPUStream", (), {}),
    "current_stream": lambda device=None: None,
    "stream": lambda stream: None,
    "set_stream": lambda stream: None,
    "Event": type("NPUEvent", (), {}),
    "OutOfMemoryError": type("NPUOutOfMemoryError", (Exception,), {}),
}


def _fake_npu_module(*, drop: Tuple[str, ...] = (), **overrides: Any) -> types.SimpleNamespace:
    """A ``torch.npu`` stand-in carrying exactly the alias table's counterparts, minus ``drop``."""
    api = {name: value for name, value in _NPU_COUNTERPARTS.items() if name not in drop}
    api.update(overrides)
    if "graph_pool_handle" not in drop:
        api.setdefault("graph_pool_handle", lambda: None)
    if "is_current_stream_capturing" not in drop:
        api.setdefault("is_current_stream_capturing", lambda: True)
    return types.SimpleNamespace(**api)


def _bridge_required(monkeypatch, module: Any) -> None:
    """Alias exactly the required set, as a working torch_npu would."""
    for alias in graphs.required_aliases():
        monkeypatch.setattr(torch.cuda, alias.name, getattr(module, alias.target), raising=False)


@pytest.fixture
def alias_guard():
    """Undo whatever a test's ``install()`` wrote, even when the assertion fails mid-way."""
    yield
    graphs.uninstall()


def _local_torch_cuda_sites() -> Dict[str, List[str]]:
    """``{name: ["file.py:line", ...]}`` for every ``torch.cuda.<name>`` in the local subset."""
    sites: Dict[str, List[str]] = {}
    for path in sorted(VLLM_API_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(text, start=1):
            for name in TORCH_CUDA_CALL.findall(line):
                sites.setdefault(name, []).append(f"{path.name}:{number}")
    return sites


def test_torch_cuda_alias_table_accounts_for_every_name_in_the_local_sources():
    """The table is evidence-driven: every `torch.cuda.<name>` is mapped or explicitly unmapped.

    The scan is a regex over ``runtime/vllm-api/*.py`` rather than a hard-coded list, so a new
    upstream call name (a new CUDA API the runner starts using) fails here instead of failing on
    hardware inside a capture.
    """
    sites = _local_torch_cuda_sites()
    assert sites, f"the scan found no torch.cuda. calls under {VLLM_API_DIR}"

    mapped = {alias.name: alias for alias in graphs.TORCH_CUDA_ALIASES}
    unmapped = dict(graphs.UNMAPPED_TORCH_CUDA_NAMES)

    assert set(sites) == set(mapped) | set(unmapped), (
        "the local vLLM sources use torch.cuda names the graphs.py table does not account for: "
        f"unmapped={sorted(set(sites) - set(mapped) - set(unmapped))}, "
        f"stale={sorted((set(mapped) | set(unmapped)) - set(sites))}. Add each to "
        "TORCH_CUDA_ALIASES (with a torch.npu counterpart) or to UNMAPPED_TORCH_CUDA_NAMES "
        "(with the reason there is nothing to alias)"
    )
    assert not (set(mapped) & set(unmapped)), "a name cannot be both mapped and unmapped"

    # Every entry's cited file:line must really contain that call - a stale audit trail would
    # make the table look evidence-based when it is not.
    for alias in mapped.values():
        assert alias.source in sites[alias.name], (
            f"{alias.cuda_path} cites {alias.source}, which does not contain "
            f"torch.cuda.{alias.name} (actual sites: {sites[alias.name]})"
        )
        assert alias.note, f"{alias.cuda_path} has no note explaining why it is needed"

    # Each unmapped name carries a reason (the inventory is a decision, not a to-do).
    for name, reason in unmapped.items():
        assert reason.strip(), f"{name} is unmapped without a reason"

    # The 13 unique names the scan finds today, pinned so an upstream refresh is visible.
    assert set(sites) == {
        "CUDAGraph",
        "graph",
        "mem_get_info",
        "Stream",
        "current_stream",
        "stream",
        "Event",
        "OutOfMemoryError",
        "set_stream",
        "is_available",
        "get_device_capability",
        "get_device_name",
        "get_device_properties",
    }


def test_the_fetched_set_stream_and_get_device_name_are_classified(monkeypatch):
    """Pin the two names the 0.23 fetch added: one best-effort alias, one unmapped call.

    Both are tempting to "fix" by widening the scan, which is the failure mode this test exists
    for.  ``set_stream`` *is* executed on our path - it is read at module scope of
    ``vllm.utils.torch_utils``, which worker init imports - so it belongs in the alias table,
    but its only call site is CUDA-only, so it must not gate capture.  ``get_device_name`` sits in
    the FLA ops' module-level ``is_nvidia`` probe, which short-circuits on NPU, so there is
    nothing to alias.

    The cited lines are pinned exactly.  A future fetch that moves them, adds a call site, or
    routes ``get_device_name`` through a branch that actually evaluates fails here first.
    """
    sites = _local_torch_cuda_sites()
    mapped = {alias.name: alias for alias in graphs.TORCH_CUDA_ALIASES}
    unmapped = dict(graphs.UNMAPPED_TORCH_CUDA_NAMES)

    # set_stream: read at import (`prev_set_stream = torch.cuda.set_stream`), overwritten by the
    # same module's shim (:679), called only under `is_rocm() or is_cuda()` (:712); :692 and :711
    # are the shim's docstring/comment.
    assert sites["set_stream"] == [
        "vllm__utils__torch_utils.py:669",
        "vllm__utils__torch_utils.py:679",
        "vllm__utils__torch_utils.py:692",
        "vllm__utils__torch_utils.py:711",
        "vllm__utils__torch_utils.py:712",
    ]
    stream_alias = mapped["set_stream"]
    assert stream_alias.required is False, (
        "set_stream must stay best-effort: no NPU path calls it, and requiring it would add a "
        "torch.npu.set_stream gate to capture_path_ready()"
    )
    assert stream_alias.source == "vllm__utils__torch_utils.py:669"
    assert "torch.npu.set_stream" in stream_alias.note, "the note must record why it is optional"

    # get_device_name: one hit, behind `is_nvidia`, in a module qwen3_5.py does import.
    assert sites["get_device_name"] == ["vllm__model_executor__layers__fla__ops__utils.py:152"]
    reason = unmapped["get_device_name"]
    assert "is_nvidia" in reason and "short-circuits" in reason
    assert "qwen3_5.py" in reason, "the reason must say the module is imported, not absent"

    # The graph verdict did not move: bridging the four required counterparts is enough even when
    # the device has no set_stream at all (so this name can never flip capture_path_ready()).
    module = _fake_npu_module(drop=("set_stream",))
    monkeypatch.setattr(graphs, "_accelerator", lambda: ("npu", module))
    _bridge_required(monkeypatch, module)
    ready, detail = graphs.capture_path_ready()
    assert ready is True, detail


def test_required_aliases_are_the_capture_and_runner_memory_stream_set():
    """The split is explicit: four required (capture + memory/stream), five best-effort."""
    required = {alias.name for alias in graphs.required_aliases()}
    optional = {alias.name for alias in graphs.optional_aliases()}
    assert required == {"CUDAGraph", "graph", "mem_get_info", "Stream"}
    assert optional == {"current_stream", "stream", "Event", "OutOfMemoryError", "set_stream"}
    assert required | optional == {alias.name for alias in graphs.TORCH_CUDA_ALIASES}
    for alias in graphs.required_aliases():
        assert alias.source.startswith("vllm__")
    # The capture pair is required; so is the memory probe and the stream constructor.
    capture_sources = {
        alias.name for alias in graphs.required_aliases() if "vllm__compilation" in alias.source
    }
    assert capture_sources == {"CUDAGraph", "graph"}


def test_capture_path_ready_checks_every_required_alias_by_identity(monkeypatch):
    """Bridging the graph pair alone is no longer enough: mem_get_info/Stream are checked too."""
    module = _fake_npu_module()
    monkeypatch.setattr(graphs, "_accelerator", lambda: ("npu", module))
    monkeypatch.setattr(torch.cuda, "CUDAGraph", module.CUDAGraph, raising=False)
    monkeypatch.setattr(torch.cuda, "graph", module.graph, raising=False)
    ready, reason = graphs.capture_path_ready()
    assert ready is False
    assert "mem_get_info" in reason and "Stream" in reason

    _bridge_required(monkeypatch, module)
    assert graphs.capture_path_ready()[0] is True

    # A look-alike is not a bridge: identity is the contract (duck-typing would accept this).
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda device=None: (1, 1), raising=False
    )
    ready, reason = graphs.capture_path_ready()
    assert ready is False
    assert "torch.cuda.mem_get_info" in reason


def test_capture_path_is_not_ready_without_a_bridge(monkeypatch):
    """Unbridged NPU primitives: exactly the situation the verdict calls 'patchable, not yet'."""
    kind, _ = graphs._accelerator()
    if kind:
        pytest.skip(f"this machine has a {kind} device; the fake-device branch is not needed")
    fake_npu = _fake_npu_module()
    monkeypatch.setattr(graphs, "_accelerator", lambda: ("npu", fake_npu))
    ready, reason = graphs.capture_path_ready()
    assert ready is False
    assert "bridged" in reason
    assert "mem_get_info" in reason


def test_capture_path_is_ready_when_the_bridge_aliases_the_primitives(monkeypatch):
    kind, _ = graphs._accelerator()
    if kind:
        pytest.skip(f"this machine has a {kind} device; the fake-device branch is not needed")
    fake_npu = _fake_npu_module()
    monkeypatch.setattr(graphs, "_accelerator", lambda: ("npu", fake_npu))
    _bridge_required(monkeypatch, fake_npu)
    ready, reason = graphs.capture_path_ready()
    assert ready is True and "bridged" in reason


def test_capture_path_rejects_missing_graph_pool_handle(monkeypatch):
    kind, _ = graphs._accelerator()
    if kind:
        pytest.skip(f"this machine has a {kind} device; the fake-device branch is not needed")
    fake_npu = _fake_npu_module(drop=("graph_pool_handle",))
    monkeypatch.setattr(graphs, "_accelerator", lambda: ("npu", fake_npu))
    _bridge_required(monkeypatch, fake_npu)
    ready, reason = graphs.capture_path_ready()
    assert ready is False
    assert "graph_pool_handle" in reason


def test_install_is_all_or_nothing_for_the_required_set(alias_guard):
    """One missing counterpart and *none* of the required aliases is written."""
    original = torch.cuda.CUDAGraph
    module = _fake_npu_module(drop=("mem_get_info",))
    installed, detail = graphs.install(npu=module)
    assert installed is False
    assert "mem_get_info" in detail
    assert torch.cuda.CUDAGraph is original, "a rejected install must not leave CUDAGraph bridged"
    assert graphs.uninstall() == ()


def test_install_rolls_back_when_a_required_write_fails(alias_guard, monkeypatch):
    """A read-only attribute mid-way through the required set restores what was already written."""
    class _WriteRefusingCuda:
        CUDAGraph = object()
        graph = object()
        mem_get_info = object()
        Stream = object()

        def __setattr__(self, name: str, value: Any) -> None:
            if name == "Stream":
                raise RuntimeError("torch.cuda.Stream is read-only here")
            object.__setattr__(self, name, value)

    fake_cuda = _WriteRefusingCuda()
    before = {name: getattr(fake_cuda, name) for name in ("CUDAGraph", "graph", "mem_get_info")}
    monkeypatch.setattr(torch, "cuda", fake_cuda)
    installed, detail = graphs.install(npu=_fake_npu_module())
    assert installed is False
    assert "Stream" in detail
    for name, value in before.items():
        assert getattr(fake_cuda, name) is value, f"torch.cuda.{name} was not rolled back"
    assert graphs.uninstall() == ()


def test_install_is_best_effort_for_optional_aliases_and_names_the_skips(
    alias_guard, monkeypatch, caplog
):
    """A missing optional counterpart is logged by name and does not stop the required set."""
    module = _fake_npu_module(drop=("Event",))
    original_event = torch.cuda.Event
    with caplog.at_level("WARNING", logger="tileinfer.vllm.graphs"):
        installed, detail = graphs.install(npu=module)
    assert installed is True
    assert "torch.cuda.Event" in caplog.text, "the skipped optional alias must be named in a log"
    assert "torch.cuda.Event" in detail and "best-effort skipped" in detail
    for alias in graphs.required_aliases():
        assert getattr(torch.cuda, alias.name) is getattr(module, alias.target)
    assert torch.cuda.current_stream is module.current_stream  # optional that *did* install
    assert torch.cuda.Event is original_event, "a skipped optional alias must not be touched"
    # Optional absence must not fail the verdict: capture_path_ready only checks the required set.
    monkeypatch.setattr(graphs, "_accelerator", lambda: ("npu", module))
    assert graphs.capture_path_ready()[0] is True


def test_uninstall_restores_every_alias_install_touched(alias_guard, monkeypatch):
    """After a full install, uninstall puts back every original object - mapped and optional."""
    originals = {alias.name: getattr(torch.cuda, alias.name) for alias in graphs.TORCH_CUDA_ALIASES}
    module = _fake_npu_module()
    installed, _ = graphs.install(npu=module)
    assert installed is True
    assert set(graphs.active_aliases()) == {alias.cuda_path for alias in graphs.TORCH_CUDA_ALIASES}
    for alias in graphs.TORCH_CUDA_ALIASES:
        assert getattr(torch.cuda, alias.name) is getattr(module, alias.target)

    restored = graphs.uninstall()
    assert set(restored) == {f"torch.cuda.{alias.name}" for alias in graphs.TORCH_CUDA_ALIASES}
    assert graphs.active_aliases() == ()
    for alias in graphs.TORCH_CUDA_ALIASES:
        assert getattr(torch.cuda, alias.name) is originals[alias.name]
    assert graphs.uninstall() == ()  # a second uninstall is a no-op, not a revert of originals


def test_uninstall_does_not_revert_a_bridge_it_did_not_write(alias_guard, monkeypatch):
    """A pre-existing torch_npu/plugin alias is not ours: install skips it, uninstall keeps it."""
    module = _fake_npu_module()
    monkeypatch.setattr(torch.cuda, "CUDAGraph", module.CUDAGraph, raising=False)
    installed, _ = graphs.install(npu=module)
    assert installed is True
    assert "torch.cuda.CUDAGraph" not in graphs.active_aliases()
    restored = graphs.uninstall()
    assert "torch.cuda.CUDAGraph" not in restored
    assert torch.cuda.CUDAGraph is module.CUDAGraph


# ---------------------------------------------------------------------------------------
# 7b. the plugin's graph bridge: it delegates to graphs.py, it does not own an alias table
# ---------------------------------------------------------------------------------------
#
# The bug this section pins: ``plugin._install_graph_bridge`` used to alias only
# ``torch.cuda.CUDAGraph``/``graph`` itself (``_CUDA_GRAPH_ATTRS``), while
# ``graphs.capture_path_ready()`` verifies **four** required aliases by identity.  On a real NPU box
# the plugin therefore logged "installed, but capture_path_ready() is still False" on its *second*
# ask, the platform kept forcing ``enforce_eager=True``, and graph mode silently never engaged.
# The fix is delegation - one alias table, owned by ``graphs.py`` (``graphs.install()`` /
# ``graphs.uninstall()``) - so these tests assert the delegation itself plus the end-to-end verdict
# the old code could not reach.


def test_plugin_graph_bridge_delegates_to_graphs_install(monkeypatch):
    """``_install_graph_bridge`` must call ``graphs.install()`` and nothing of its own."""
    calls: List[str] = []

    def fake_install(npu=None):
        calls.append("graphs.install")
        return True, "aliased CUDAGraph, graph, mem_get_info, Stream"

    answers = [(False, "not bridged yet"), (True, "NPU: the required aliases are bridged")]
    monkeypatch.setattr(graphs, "install", fake_install)
    monkeypatch.setattr(
        graphs, "capture_path_ready", lambda: answers.pop(0) if answers else (True, "ready")
    )
    monkeypatch.setattr(graphs, "active_aliases", lambda: ("torch.cuda.CUDAGraph",))
    monkeypatch.delenv(plugin_mod.ENV_GRAPH_BRIDGE, raising=False)

    info = plugin_mod._install_graph_bridge()

    assert calls == ["graphs.install"], "the plugin must delegate the alias writes to graphs.py"
    assert info["installed"] is True
    assert info["ready"] is True and "bridged" in info["reason"]
    # The reported state comes from graphs.py's own bookkeeping, no local copy.
    assert info["aliases"] == ["torch.cuda.CUDAGraph"]


def test_plugin_graph_bridge_reaches_the_four_alias_verdict(alias_guard, monkeypatch):
    """The regression itself: after the bridge, the *second* ``capture_path_ready()`` is True.

    The old implementation aliased the graph pair only, so this assertion failed with
    ``ready is False`` and "``torch.cuda.mem_get_info``... are not bridged" as the reason.  The
    fake device and the real ``graphs.install()`` are used, so the identity check is genuine.
    """
    kind, _ = graphs._accelerator()
    if kind:
        pytest.skip(f"this machine has a {kind} device; the fake-device branch is not needed")
    module = _fake_npu_module()
    # ``graphs.install()`` reads ``torch.npu`` (the real one is absent on this CPU box), while the
    # verdict reads ``graphs._accelerator``: both are pointed at the same fake device object.
    monkeypatch.setattr(graphs, "_accelerator", lambda: ("npu", module))
    monkeypatch.setattr(torch, "npu", module, raising=False)
    monkeypatch.delenv(plugin_mod.ENV_GRAPH_BRIDGE, raising=False)
    original_cuda_graph = torch.cuda.CUDAGraph

    info = plugin_mod._install_graph_bridge()

    assert info["installed"] is True, info["detail"]
    assert info["ready"] is True, info["reason"]
    assert torch.cuda.CUDAGraph is not original_cuda_graph  # the alias really was written
    required = {f"torch.cuda.{alias.name}" for alias in graphs.required_aliases()}
    assert required <= set(info["aliases"]), "every required alias must be reported as installed"


def test_plugin_graph_bridge_honours_the_env_opt_out(monkeypatch):
    """``TILEINFER_VLLM_GRAPH_BRIDGE=0`` keeps ``graphs.install()`` from being called at all."""
    monkeypatch.setenv(plugin_mod.ENV_GRAPH_BRIDGE, "0")
    monkeypatch.setattr(graphs, "capture_path_ready", lambda: (False, "no bridge yet"))

    def refuse(*args, **kwargs):
        raise AssertionError(
            "graphs.install() was called despite TILEINFER_VLLM_GRAPH_BRIDGE=0"
        )

    monkeypatch.setattr(graphs, "install", refuse)
    info = plugin_mod._install_graph_bridge()
    assert info["installed"] is False
    assert plugin_mod.ENV_GRAPH_BRIDGE in info["detail"]
    assert "no bridge yet" in info["detail"]  # the reason is carried into the opt-out message


def test_plugin_uninstall_delegates_to_graphs_uninstall(monkeypatch):
    """``uninstall()`` must revert through ``graphs.uninstall()``, not a local saved-attribute dict."""
    calls: List[str] = []

    def fake_uninstall():
        calls.append("graphs.uninstall")
        return ("torch.cuda.CUDAGraph", "torch.cuda.Stream")

    monkeypatch.setattr(graphs, "uninstall", fake_uninstall)
    monkeypatch.setitem(plugin_mod._STATE, "backend", {})  # no CUSTOM slot to clear in this test

    assert plugin_mod.uninstall() is True
    assert calls == ["graphs.uninstall"]
    assert plugin_mod._STATE["installed"] is False


def test_plugin_no_longer_owns_an_alias_table():
    """The redundant local machinery must be gone, so the two lists cannot drift again."""
    for stale in ("_CUDA_GRAPH_ATTRS", "_NPU_PRIMITIVES", "_MISSING", "_bridge_graph_names"):
        assert not hasattr(plugin_mod, stale), (
            f"plugin.{stale} duplicates tileinfer.vllm.graphs and must be deleted - two alias "
            "tables is what made the bridge install but capture_path_ready() stay False"
        )
    assert "bridge_saved" not in plugin_mod._STATE
    # The table exists exactly once, in graphs.py, and it is the one capture_path_ready() checks.
    assert graphs.required_aliases()
    assert set(graphs.active_aliases()) <= {
        f"torch.cuda.{alias.name}" for alias in graphs.TORCH_CUDA_ALIASES
    }


def test_plugin_status_reports_the_bridge_from_graphs(monkeypatch):
    """``status()`` keeps the same field names, now sourced from graphs.py instead of ``_STATE``."""
    monkeypatch.setattr(graphs, "capture_path_ready", lambda: (False, "no accelerator"))
    monkeypatch.setattr(graphs, "active_aliases", lambda: ("torch.cuda.CUDAGraph",))
    monkeypatch.setitem(plugin_mod._STATE, "graph", {})

    graph_status = plugin_mod.status()["graphs"]

    assert graph_status["branch"] == graphs.GRAPH_BRANCH
    assert graph_status["capture_path_ready"] is False
    assert graph_status["reason"] == "no accelerator"
    assert graph_status["bridge_installed"] is True  # graphs.active_aliases(), not a local dict
    assert graph_status["active_aliases"] == ["torch.cuda.CUDAGraph"]
    assert graph_status["unverified_symbols"]


def test_graph_branch_is_patchable_and_its_cited_lines_match_the_local_source():
    """The verdict rests on line numbers; if they move, the verdict must be re-derived."""
    assert graphs.GRAPH_BRANCH == graphs.BRANCH_PATCHABLE
    assert graphs.GRAPH_BRANCH != graphs.BRANCH_DEVICE_AGNOSTIC
    lines = CUDA_GRAPH_SOURCE.read_text(encoding="utf-8").splitlines()

    # Every `torch.cuda.` occurrence in the file: one annotation plus the two runtime calls.
    occurrences = [
        number
        for number, line in enumerate(lines, start=1)
        if "torch.cuda." in line and not line.strip().startswith("#")
    ]
    assert occurrences == [130, 283, 313], (
        f"the CUDA surface of cuda_graph.py moved ({occurrences}); the branch verdict in "
        "tileinfer/vllm/graphs.py must be re-read against the new source"
    )
    # The two *runtime* call sites are the patch surface; the annotation only needs the name to
    # exist (a CPU torch build defines it: torch/cuda/graphs.py:76-84,296).
    assert [line_no for line_no, _ in graphs.CUDA_CALL_SITES] == [283, 313]
    assert "| None" in lines[130 - 1]  # an annotation, not a call
    for line_no, expected in graphs.CUDA_CALL_SITES:
        assert expected.strip() in lines[line_no - 1]


def test_unverified_symbols_are_declared():
    """The five outside symbols are part of the decision record, not folklore."""
    assert len(graphs.UNVERIFIED_SYMBOLS) == 5
    joined = " ".join(symbol for symbol, _ in graphs.UNVERIFIED_SYMBOLS)
    for expected in (
        "current_stream",
        "set_graph_pool_id",
        "validate_cudagraph_capturing_enabled",
        "get_offloader",
        "graph_pool_handle",
    ):
        assert expected in joined


def test_capture_contract_accepts_a_prewarmed_clean_capture():
    state = graphs.CaptureState(plan_ready=True)
    ok, reason = graphs.capture_contract_ok(state)
    assert ok is True and reason == "ok"
    graphs.assert_capture_contract(state)  # must not raise


@pytest.mark.parametrize(
    "kwargs,code",
    [
        ({"plan_ready": False}, "plan_not_ready"),
        ({"plan_ready": True, "plan_builds": 1}, "plan_builds"),
        ({"plan_ready": True, "state_builds": 2}, "state_builds"),
        ({"plan_ready": True, "compile_calls": 1}, "compiled"),
        ({"plan_ready": True, "allocated_bytes": 4096}, "allocated"),
        ({"plan_ready": True, "sync_calls": 1}, "synchronized"),
        ({"plan_ready": True, "fallbacks": 1}, "fallback"),
    ],
)
def test_capture_contract_rejects_each_violation(kwargs, code):
    ok, reason = graphs.capture_contract_ok(graphs.CaptureState(**kwargs))
    assert ok is False
    assert reason.startswith(code)
    assert code in graphs.CAPTURE_VIOLATION_CODES
    with pytest.raises(RuntimeError, match=code):
        graphs.assert_capture_contract(graphs.CaptureState(**kwargs))


def test_capture_contract_duck_types_and_requires_plan_ready():
    class _Duck:
        plan_ready = True
        fallbacks = 0

    assert graphs.capture_contract_ok(_Duck())[0] is True

    class _NoPlan:
        pass

    with pytest.raises(TypeError, match="plan_ready"):
        graphs.capture_contract_ok(_NoPlan())


def test_every_violation_code_is_reachable():
    """A code with no check behind it would make a probe silently pass."""
    assert len(graphs.CAPTURE_VIOLATION_CODES) == 7


def test_is_capturing_is_false_without_a_device():
    kind, _ = graphs._accelerator()
    if kind:
        pytest.skip(f"this machine has a {kind} device; the no-device branch cannot be observed")
    assert graphs.is_capturing() is False


def test_wrap_refuses_with_the_reason_and_the_next_step():
    if graphs.capture_path_ready()[0]:
        pytest.skip("graph capture is available on this machine")
    with pytest.raises(NotImplementedError) as excinfo:
        graphs.wrap(model=object(), vllm_config=_config())
    message = str(excinfo.value)
    assert "CUDAGraphWrapper" in message
    assert "patchable" in message
    assert "tileinfer/vllm/graphs.py" in message


def test_wrap_delegates_when_the_bridge_exists(monkeypatch):
    fake_npu = _fake_npu_module()
    monkeypatch.setattr(graphs, "_accelerator", lambda: ("npu", fake_npu))
    _bridge_required(monkeypatch, fake_npu)
    config = _config()
    model = object()
    wrapper = graphs.wrap(model, config, CUDAGraphMode.PIECEWISE)
    assert isinstance(wrapper, vllm_cuda_graph.CUDAGraphWrapper)
    assert wrapper.runnable is model
    assert wrapper.runtime_mode is CUDAGraphMode.PIECEWISE
    assert wrapper.vllm_config is config


def test_wrap_reads_the_mode_from_the_config_and_rejects_none(monkeypatch):
    if not USING_FAKE:
        pytest.skip("the stub CUDAGraphWrapper is needed to observe the delegation")
    fake_npu = _fake_npu_module()
    monkeypatch.setattr(graphs, "_accelerator", lambda: ("npu", fake_npu))
    _bridge_required(monkeypatch, fake_npu)

    config = _config(compilation_config=_compilation_config(cudagraph_mode=CUDAGraphMode.FULL))
    wrapper = graphs.wrap(object(), config)
    assert wrapper.runtime_mode is CUDAGraphMode.FULL

    with pytest.raises(ValueError, match="no capture is requested"):
        graphs.wrap(object(), _config())  # default mode is NONE


# =======================================================================================
# 8. the worker (its own contract file is tests/test_worker_contract.py)
# =======================================================================================


def test_worker_is_a_vllm_worker_with_the_four_hooks():
    assert issubclass(worker_mod.NPUWorker, sys.modules["vllm.v1.worker.gpu_worker"].Worker)
    for hook in (
        "init_device",
        "determine_available_memory",
        "initialize_from_config",
        "compile_or_warm_up_model",
    ):
        assert hook in vars(worker_mod.NPUWorker), f"{hook} is not overridden"


@pytest.mark.parametrize(
    "hook,args",
    [
        ("init_device", ()),
        ("determine_available_memory", ()),
        ("initialize_from_config", (object(),)),
        ("compile_or_warm_up_model", ()),
    ],
)
def test_worker_hooks_are_no_longer_the_raising_skeleton(hook, args):
    """The four hooks used to raise ``NotImplementedError``; they are implemented now.

    The behavioural assertions live in ``tests/test_worker_contract.py`` (which owns the fake's
    ``GPUModelRunner`` and the runner-side seams); what this file keeps is the platform-side
    invariant: the hooks exist on the class the platform names, and none of them is still the
    skeleton's "not implemented" raise.
    """
    source = inspect.getsource(getattr(worker_mod.NPUWorker, hook))
    assert f"TileInfer NPUWorker.{hook} is not implemented" not in source
    assert callable(getattr(worker_mod.NPUWorker, hook))


def test_worker_checklist_names_the_open_hardware_items():
    """The checklist used to lead with "fetch gpu_model_runner.py"; that file is in the repo now."""
    joined = " ".join(worker_mod.HARDWARE_CHECKLIST)
    assert "gpu_model_runner.py" in joined  # still cited, as the source of the seam
    assert "not in the local subset" not in joined
    assert "capture_path_ready" in joined
    assert "mem_get_info" in joined
    # The hybrid item names the device-side gap and the settled layout, not the old boot blocker.
    assert "support_hybrid_kv_cache() now answers True" in joined
    assert "initialize_mamba_ssu_backend" in joined
    assert "KVBlockZeroer" in joined
    assert "get_kv_cache_stride_order" in joined
    assert "KV pool must be contiguous" not in joined
    assert "stays default-off" not in joined
