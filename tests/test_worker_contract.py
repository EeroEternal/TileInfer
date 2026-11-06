"""Contract tests for ``tileinfer.vllm.worker.NPUWorker``, **without vLLM and without an NPU**.

Why a fake ``vllm``
-------------------
``tileinfer/vllm/worker.py`` subclasses two real vLLM 0.23 classes
(``vllm.v1.worker.gpu_worker.Worker`` and ``vllm.v1.worker.gpu_model_runner.GPUModelRunner``), so
importing it needs those *names* to exist.  CI has neither vLLM nor a device, so the names are
injected into ``sys.modules`` before the import.  The fake itself lives in ``tests/fake_vllm.py``,
**shared** with ``tests/test_attn_contract.py``: one fake, one shape, so the order these two files
run in (or which one imports first) cannot change what ``vllm`` looks like.  Private per-file fakes
made this suite order-dependent — whichever contract test ran first owned ``vllm``, and the other
one then faced a module missing the names it imports (a *collection* error that interrupts the
whole session) or silently ran against a thinner stub.  ``fake_vllm.install()`` *augments* whatever
fake is already there, never replaces it, because already-imported TileInfer modules hold
references into those module objects.  This file still never defines ``vllm.platforms.Platform``
(the platform contract test owns that).

What is asserted (and what is not)
----------------------------------
Asserted, all on a CPU box:

* the class derives from the right base and its path is the one the platform hands to vLLM;
* each of the four hooks is overridden in ``NPUWorker.__dict__`` and its parameter names agree
  with the *local* vLLM source (parsed with ``ast``, not retyped), so a renamed keyword fails here;
* every hook docstring states its ``HARDWARE:`` assumptions and the first diagnostic command - the
  task's honesty rule, made a test;
* ``torch_npu`` is never imported at module level, and *is* imported inside the device helpers;
* the memory helpers reject a non-NPU device and, when a torch_npu spelling is absent, the error
  names **both** candidates;
* the KV hook stores ``num_gpu_blocks``, initialises the pool through the runner, uses a
  ``nullcontext`` memory-pool hook, and forwards the model's own specs untouched;
* ``capture_model`` is a logged, counted no-op while the graph path is not ready, and delegates
  when it is;
* ``compile_or_warm_up_model`` skips ``kernel_warmup`` and the Triton monitor, keeps the sampler
  warm-up after the capture gate, and returns ``CompilationTimes``.

Not asserted: anything needing an Ascend device - the real ``torch.npu`` numbers, the ``npu:``
device spelling, HCCL, and an actual graph capture.  Those are the module's ``HARDWARE:`` items,
listed in ``HARDWARE_CHECKLIST``.
"""

from __future__ import annotations

import ast
import contextlib
import enum
import importlib.util
import inspect
import logging
import pathlib
import sys
import types
from typing import Any, Dict, List

import pytest
import torch

import fake_vllm

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER_SOURCE = ROOT / "tileinfer" / "vllm" / "worker.py"
PLATFORM_SOURCE = ROOT / "tileinfer" / "vllm" / "platform.py"
GPU_WORKER_SOURCE = ROOT / "runtime" / "vllm-api" / "vllm__v1__worker__gpu_worker.py"

#: The four hooks the engine relies on, in call order (``worker.py:HOOKS``).
HOOKS = (
    "init_device",
    "determine_available_memory",
    "initialize_from_config",
    "compile_or_warm_up_model",
)


# ---------------------------------------------------------------------------------------
# The shared fake `vllm` (owned by tests/fake_vllm.py, shared with test_attn_contract.py)
# ---------------------------------------------------------------------------------------
#
# One fake, one shape: the worker API this file imports and the attention API its sibling contract
# test imports both live in `tests/fake_vllm.py`.  It is a helper module rather than a fixture
# because the names must exist at import time of this file (the `from tileinfer.vllm import worker`
# below), before any fixture could be requested.  `install()` returns "fake" / "augmented" /
# "real" and never replaces a `vllm` another fake already installed - see that module's docstring
# for why that (and not a per-file copy) is what makes the contract tests order-independent.
_ensure_chain = fake_vllm.ensure_chain
_setdefault = fake_vllm.setdefault


def _install_fake_vllm() -> str:
    """Install the shared fake (``tests/fake_vllm.py``); return ``fake``/``augmented``/``real``."""
    return fake_vllm.install()


VLLM_MODE = _install_fake_vllm()

# Imported *after* the fake is in sys.modules (these modules raise their "not importable without
# vLLM" error otherwise - which is exactly what the guarded import exists for).
from tileinfer.vllm import graphs  # noqa: E402
from tileinfer.vllm import worker as worker_mod  # noqa: E402

gpu_worker_mod = sys.modules["vllm.v1.worker.gpu_worker"]
gpu_runner_mod = sys.modules["vllm.v1.worker.gpu_model_runner"]
worker_utils_mod = sys.modules["vllm.v1.worker.utils"]
request_memory_calls = worker_utils_mod.request_memory_calls
kv_transfer_calls = sys.modules["vllm.distributed.kv_transfer"].kv_transfer_calls

#: The ``privateuseone`` alias of the same NPU backend: constructible on this CPU torch build, so
#: the device guards can be exercised without inventing a device type.
NPU_DEVICE = torch.device("privateuseone", 0)


# ---------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------


def _fake_npu(**overrides: Any) -> types.SimpleNamespace:
    """A ``torch.npu`` stand-in carrying the APIs the worker's helpers use."""
    api: Dict[str, Any] = {
        "device_count": lambda: 1,
        "set_device": lambda device: None,
        "empty_cache": lambda: None,
        "mem_get_info": lambda device=None: (0, 0),
        "memory_stats": lambda device=None: {},
        "max_memory_allocated": lambda device=None: 0,
        "get_device_properties": lambda index=0: types.SimpleNamespace(),
    }
    api.update(overrides)
    return types.SimpleNamespace(**api)


def _snapshot(total: int, free: int, peak: int) -> Any:
    return worker_mod.NpuMemorySnapshot(total_memory=total, free_memory=free, torch_peak=peak)


class _FakeRunner:
    """Every runner method the four hooks touch, recording each call."""

    def __init__(self, *, model_memory_usage: int = 0) -> None:
        self.calls: List[Any] = []
        self.model_memory_usage = model_memory_usage
        self.is_pooling_model = False
        self.lora_config = object()

    # -- KV path ---------------------------------------------------------------------------
    def initialize_kv_cache(self, kv_cache_config: Any, is_profiling: bool = False) -> None:
        self.calls.append(("initialize_kv_cache", kv_cache_config))

    def _init_kv_zero_meta(self) -> None:
        self.calls.append(("_init_kv_zero_meta",))

    def init_routed_experts_capturer(self) -> None:
        self.calls.append(("init_routed_experts_capturer",))

    # -- memory / warm-up path -------------------------------------------------------------
    def profile_run(self) -> None:
        self.calls.append(("profile_run",))

    def _dummy_run(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(("_dummy_run", args, kwargs))
        return (object(), object())

    def _dummy_sampler_run(self, hidden_states: Any = None) -> None:
        self.calls.append(("_dummy_sampler_run",))

    def _dummy_pooler_run(self, hidden_states: Any = None) -> None:
        self.calls.append(("_dummy_pooler_run",))

    def maybe_remove_all_loras(self, lora_config: Any) -> None:
        self.calls.append(("maybe_remove_all_loras", lora_config))

    def capture_model(self) -> int:
        self.calls.append(("capture_model",))
        return 4096

    def profile_cudagraph_memory(self) -> int:
        self.calls.append(("profile_cudagraph_memory",))
        return 8192

    def called(self, name: str) -> bool:
        return any(isinstance(call, tuple) and call[:1] == (name,) for call in self.calls)

    def count(self, name: str) -> int:
        return sum(1 for call in self.calls if isinstance(call, tuple) and call[:1] == (name,))


def _worker_with_runner(runner: _FakeRunner, **attrs: Any) -> Any:
    worker = object.__new__(worker_mod.NPUWorker)  # no __init__: the base one needs a VllmConfig
    worker.device = NPU_DEVICE
    worker.model_runner = runner
    for name, value in attrs.items():
        setattr(worker, name, value)
    return worker


# ---------------------------------------------------------------------------------------
# Source readers: the local vLLM file, not a transcription
# ---------------------------------------------------------------------------------------


def _source_method_params(path: pathlib.Path, class_name: str, method: str) -> List[str]:
    """Parameter names of ``class_name.method`` in the local source, via ``ast``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    fn = next(
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method
    )
    args = fn.args
    names = [a.arg for a in args.posonlyargs] + [a.arg for a in args.args]
    if args.vararg:
        names.append(args.vararg.arg)
    names += [a.arg for a in args.kwonlyargs]
    if args.kwarg:
        names.append(args.kwarg.arg)
    return names


def _module_scope_imports(path: pathlib.Path) -> List[str]:
    """Imported names at module scope, descending into ``try``/``if`` but not into functions.

    The vLLM imports in ``tileinfer/vllm/worker.py`` sit inside a ``try:`` (the guarded-import
    pattern), so a plain ``tree.body`` scan would miss the very imports this file pins.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: List[str] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(child, ast.Import):
                found.extend(alias.name for alias in child.names)
            elif isinstance(child, ast.ImportFrom):
                base = child.module or ""
                found.append(base)
                for alias in child.names:
                    found.append(f"{base}.{alias.name}" if base else alias.name)
            else:
                visit(child)

    visit(tree)
    return found


def _function_scope_imports(path: pathlib.Path) -> List[str]:
    """Imported names inside function/method bodies (the allowed place for ``torch_npu``)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Import):
                found.extend(alias.name for alias in inner.names)
            elif isinstance(inner, ast.ImportFrom):
                found.append(inner.module or "")
    return found


def _torch_npu_importable() -> bool:
    try:
        return importlib.util.find_spec("torch_npu") is not None
    except (ImportError, ValueError):  # a broken install must not masquerade as "present"
        return False


# =======================================================================================
# 1. the class and its hooks
# =======================================================================================


def _platform_constant(name: str) -> Any:
    """Read a module-level constant out of ``tileinfer/vllm/platform.py`` without importing it.

    ``tileinfer.vllm.platform`` needs ``vllm.platforms.Platform`` (the ABC) and its contract test
    owns that fake; reading the assignment with ``ast`` keeps this file independent of import order
    while still pinning the two paths together.
    """
    tree = ast.parse(PLATFORM_SOURCE.read_text(encoding="utf-8"), filename=str(PLATFORM_SOURCE))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not a module-level constant of tileinfer/vllm/platform.py")


def test_worker_derives_from_vllm_worker():
    assert issubclass(worker_mod.NPUWorker, gpu_worker_mod.Worker)
    assert worker_mod.NPUWorker is not gpu_worker_mod.Worker


def test_worker_class_path_is_the_one_the_platform_hands_to_vllm():
    assert worker_mod.WORKER_CLASS_PATH == "tileinfer.vllm.worker.NPUWorker"
    assert worker_mod.WORKER_CLASS_PATH == _platform_constant("WORKER_CLASS_PATH")
    expected = f"{worker_mod.NPUWorker.__module__}.{worker_mod.NPUWorker.__qualname__}"
    assert worker_mod.WORKER_CLASS_PATH == expected


def test_the_platform_hands_the_worker_over_through_parallel_config():
    """The handover field is verified, not assumed (vLLM 0.23 ``parallel.py:248``).

    ``WorkerWrapperBase.init_worker`` resolves ``parallel_config.worker_cls`` with
    ``resolve_obj_by_qualname`` (``vllm__v1__worker__worker_base.py:249-253``), so the class path
    above only reaches the engine through that string field.  The sentinel is vLLM's ``"auto"``;
    the platform replaces it in ``check_and_update_config`` (the in-tree platforms do the same,
    ``vllm__platforms__cpu.py:153-154``).
    """
    assert _platform_constant("WORKER_CLS_ATTR") == "worker_cls"
    assert _platform_constant("WORKER_CLS_AUTO") == "auto"


def test_hooks_are_overridden_on_the_worker_itself():
    for hook in HOOKS:
        assert hook in vars(worker_mod.NPUWorker), f"{hook} is not overridden by NPUWorker"
    assert tuple(worker_mod.HOOKS) == HOOKS


@pytest.mark.parametrize("hook", HOOKS)
def test_hook_signature_matches_the_local_vllm_source(hook):
    """Parameter names come from ``runtime/vllm-api/...gpu_worker.py`` - a rename fails here."""
    expected = _source_method_params(GPU_WORKER_SOURCE, "Worker", hook)
    actual = list(inspect.signature(getattr(worker_mod.NPUWorker, hook)).parameters)
    assert actual == expected, (
        f"NPUWorker.{hook}{tuple(actual)} does not match the GPU worker's {tuple(expected)} "
        "(gpu_worker.py); vLLM calls these hooks from the base class and a renamed parameter is "
        "an immediate TypeError at engine start."
    )


@pytest.mark.parametrize("hook", HOOKS)
def test_hook_docstring_states_hardware_assumptions_and_first_diagnostic(hook):
    """The task's honesty rule, as a test: every hook says what needs hardware and how to check."""
    doc = inspect.getdoc(getattr(worker_mod.NPUWorker, hook)) or ""
    assert "HARDWARE" in doc, f"{hook} does not mark its hardware assumptions"
    assert "First diagnostic" in doc, f"{hook} does not name its first diagnostic command"
    assert "gpu_worker.py:" in doc, f"{hook} does not cite the GPU worker's line numbers"
    assert "python -c" in doc, f"{hook} does not give a runnable first diagnostic command"


def test_hooks_are_not_the_old_raising_skeletons():
    for hook in HOOKS:
        source = inspect.getsource(getattr(worker_mod.NPUWorker, hook))
        assert f"TileInfer NPUWorker.{hook} is not implemented" not in source, (
            f"{hook} still carries the skeleton's 'is not implemented' raise"
        )


def test_hardware_checklist_no_longer_names_the_missing_runner():
    joined = " ".join(worker_mod.HARDWARE_CHECKLIST)
    assert "not in the local subset" not in joined
    assert "capture_path_ready" in joined
    assert "mem_get_info" in joined
    assert "hccl" in joined.lower()
    assert "initialize_kv_cache" in joined


# =======================================================================================
# 2. import policy: torch_npu only inside functions
# =======================================================================================


def test_worker_never_imports_torch_npu_at_module_level():
    offenders = [
        name
        for name in _module_scope_imports(WORKER_SOURCE)
        if name == "torch_npu" or name.startswith("torch_npu.")
    ]
    assert not offenders, f"tileinfer/vllm/worker.py imports torch_npu at module level: {offenders}"


def test_worker_does_import_torch_npu_inside_its_device_helpers():
    """The ban is on *module-level* imports; the helpers must really reach torch_npu."""
    assert "torch_npu" in _function_scope_imports(WORKER_SOURCE), (
        "no function in tileinfer/vllm/worker.py imports torch_npu; the device path would be dead"
    )


def test_worker_module_level_vllm_surface_is_only_the_two_classes():
    vllm_names = sorted(
        {
            name
            for name in _module_scope_imports(WORKER_SOURCE)
            if name == "vllm" or name.startswith("vllm.")
        }
    )
    assert vllm_names == [
        "vllm.v1.worker.gpu_model_runner",
        "vllm.v1.worker.gpu_model_runner.GPUModelRunner",
        "vllm.v1.worker.gpu_worker",
        "vllm.v1.worker.gpu_worker.Worker",
    ], (
        "the module-level vLLM surface grew; every extra name is one more thing the CPU fake must "
        f"reproduce and one more import at engine start-up (got {vllm_names})"
    )


# =======================================================================================
# 3. the memory helpers
# =======================================================================================


def test_memory_snapshot_rejects_a_non_npu_device():
    with pytest.raises(ValueError) as excinfo:
        worker_mod._npu_memory_snapshot(torch.device("cpu"))
    message = str(excinfo.value)
    assert "npu" in message
    assert "cpu" in message
    assert "KV" in message


def test_memory_snapshot_accepts_the_privateuse1_alias(monkeypatch):
    monkeypatch.setattr(worker_mod, "_npu_module", lambda what: _fake_npu())
    snapshot = worker_mod._npu_memory_snapshot(NPU_DEVICE)
    assert snapshot == worker_mod.NpuMemorySnapshot(
        total_memory=0, free_memory=0, torch_peak=0
    )


def test_mem_get_info_error_names_both_candidate_spellings(monkeypatch):
    npu = _fake_npu()
    del npu.mem_get_info  # neither the top-level nor the module spelling exists
    monkeypatch.setattr(worker_mod, "_npu_module", lambda what: npu)
    with pytest.raises(RuntimeError) as excinfo:
        worker_mod._npu_memory_snapshot(NPU_DEVICE)
    message = str(excinfo.value)
    assert "torch.npu.mem_get_info" in message
    assert "torch.npu.memory.mem_get_info" in message


def test_mem_get_info_falls_back_to_the_module_spelling(monkeypatch):
    npu = _fake_npu()
    del npu.mem_get_info
    npu.memory = types.SimpleNamespace(mem_get_info=lambda device=None: (123, 456))
    monkeypatch.setattr(worker_mod, "_npu_module", lambda what: npu)
    snapshot = worker_mod._npu_memory_snapshot(NPU_DEVICE)
    assert (snapshot.free_memory, snapshot.total_memory) == (123, 456)


def test_peak_error_names_both_candidate_spellings():
    npu = _fake_npu()
    del npu.memory_stats
    del npu.max_memory_allocated
    with pytest.raises(RuntimeError) as excinfo:
        worker_mod._npu_peak_allocated(npu, NPU_DEVICE)
    message = str(excinfo.value)
    assert "allocated_bytes.all.peak" in message
    assert "max_memory_allocated" in message


def test_peak_is_read_from_memory_stats_first_and_the_accessor_second():
    npu_stats = _fake_npu(
        memory_stats=lambda device=None: {"allocated_bytes.all.peak": 111},
        max_memory_allocated=lambda device=None: 222,
    )
    assert worker_mod._npu_peak_allocated(npu_stats, NPU_DEVICE) == 111

    npu_accessor = _fake_npu(
        memory_stats=lambda device=None: {},  # dict without the key: fall through
        max_memory_allocated=lambda device=None: 222,
    )
    assert worker_mod._npu_peak_allocated(npu_accessor, NPU_DEVICE) == 222


def test_requested_memory_is_vllms_helper_not_a_local_copy():
    request_memory_calls.clear()
    snapshot = _snapshot(total=1000, free=400, peak=0)
    cache_config = types.SimpleNamespace(gpu_memory_utilization=0.9)
    assert worker_mod._requested_memory(snapshot, cache_config) == 400  # clamped by free memory
    assert request_memory_calls and request_memory_calls[-1][0] is snapshot


def test_requested_memory_explains_a_rejected_snapshot(monkeypatch):
    def _picky(snapshot: Any, cache_config: Any) -> int:
        return snapshot.reserved_memory  # a field a future MemorySnapshot may grow; ours lacks it

    monkeypatch.setattr(worker_utils_mod, "request_memory", _picky)
    with pytest.raises(RuntimeError) as excinfo:
        worker_mod._requested_memory(_snapshot(1, 1, 0), types.SimpleNamespace())
    message = str(excinfo.value)
    assert "NpuMemorySnapshot" in message
    assert "total_memory" in message


# =======================================================================================
# 4. determine_available_memory
# =======================================================================================


def _determine_worker(runner: _FakeRunner, **attrs: Any) -> Any:
    defaults: Dict[str, Any] = {
        "cache_config": types.SimpleNamespace(
            kv_cache_memory_bytes=None, gpu_memory_utilization=0.5
        ),
        "init_snapshot": _snapshot(total=100_000, free=10_000, peak=0),
    }
    defaults.update(attrs)
    return _worker_with_runner(runner, **defaults)


def test_determine_available_memory_honours_kv_cache_memory_bytes(monkeypatch):
    runner = _FakeRunner()
    worker = _determine_worker(
        runner,
        cache_config=types.SimpleNamespace(kv_cache_memory_bytes=1234, gpu_memory_utilization=0.9),
        init_snapshot=_snapshot(total=100, free=50, peak=0),
    )

    def _must_not_touch_the_device(what: str) -> Any:  # pragma: no cover - the point of the test
        raise AssertionError("the kv_cache_memory_bytes path must not probe the device")

    monkeypatch.setattr(worker_mod, "_npu_module", _must_not_touch_the_device)
    assert worker.determine_available_memory() == 1234
    assert runner.called("profile_run")


def test_determine_available_memory_accounting(monkeypatch):
    runner = _FakeRunner(model_memory_usage=2000)
    worker = _determine_worker(runner)
    worker.requested_memory = 50_000

    frees = [10_000, 6_000]  # before profiling, after profiling
    peaks = [1_000, 2_500, 2_500]  # before snapshot, peak inside profiling, after snapshot
    npu = _fake_npu(
        mem_get_info=lambda device=None: (frees.pop(0), 100_000),
        max_memory_allocated=lambda device=None: peaks.pop(0),
    )
    monkeypatch.setattr(worker_mod, "_npu_module", lambda what: npu)
    monkeypatch.setattr(worker_mod.graphs, "capture_path_ready", lambda: (False, "not bridged"))

    available = worker.determine_available_memory()

    # torch_peak_increase = 2500 - 1000 = 1500
    # non_torch          = (10000 - 1000) - (6000 - 2500) = 5500
    # non_kv             = 5500 + 1500 + 2000 weights = 9000
    # available          = 50000 requested - 9000 - 0 (no graph pool) = 41000
    assert available == 41_000
    assert worker.peak_activation_memory == 1_500
    assert worker.non_torch_memory == 5_500
    assert worker.cudagraph_memory_estimate == 0
    assert worker.available_kv_cache_memory_bytes == 41_000
    assert runner.called("profile_run")


def test_determine_available_memory_rejects_free_memory_that_grew(monkeypatch):
    runner = _FakeRunner(model_memory_usage=0)
    worker = _determine_worker(runner, init_snapshot=_snapshot(total=100_000, free=10_000, peak=0))
    worker.requested_memory = 10_000
    frees = [10_000, 12_000]  # another process released memory while we profiled
    monkeypatch.setattr(
        worker_mod, "_npu_module", lambda what: _fake_npu(
            mem_get_info=lambda device=None: (frees.pop(0), 100_000)
        )
    )
    with pytest.raises(AssertionError) as excinfo:
        worker.determine_available_memory()
    assert "Error in memory profiling" in str(excinfo.value)


def test_determine_available_memory_refuses_a_non_npu_device(monkeypatch):
    worker = _determine_worker(_FakeRunner(), device=torch.device("cpu"))
    monkeypatch.setattr(worker_mod, "_npu_module", lambda what: _fake_npu())
    with pytest.raises(ValueError, match="Ascend NPU"):
        worker.determine_available_memory()


# =======================================================================================
# 5. initialize_from_config
# =======================================================================================


def test_initialize_from_config_sets_blocks_and_forwards_the_models_specs():
    runner = _FakeRunner()
    kv_cache_config = types.SimpleNamespace(
        num_blocks=42, kv_cache_groups=("a", "b"), needs_kv_cache_zeroing=False
    )
    cache_config = types.SimpleNamespace(num_gpu_blocks=None)
    worker = _worker_with_runner(
        runner,
        cache_config=cache_config,
        model_config=types.SimpleNamespace(enable_return_routed_experts=False, is_hybrid=True),
        vllm_config=types.SimpleNamespace(),
    )

    worker.initialize_from_config(kv_cache_config)

    assert cache_config.num_gpu_blocks == 42
    # the KV connector is initialised before the allocation (gpu_worker.py:570-575)
    assert kv_transfer_calls[-1] == (worker.vllm_config, kv_cache_config)
    # the specs are forwarded untouched: no hand-built group list anywhere in the worker
    assert ("initialize_kv_cache", kv_cache_config) in runner.calls
    assert not runner.called("init_routed_experts_capturer")
    assert not runner.called("_init_kv_zero_meta")


def test_initialize_from_config_zeroes_metadata_only_when_the_config_asks():
    runner = _FakeRunner()
    kv_cache_config = types.SimpleNamespace(
        num_blocks=1, kv_cache_groups=(), needs_kv_cache_zeroing=True
    )
    worker = _worker_with_runner(
        runner,
        cache_config=types.SimpleNamespace(num_gpu_blocks=None),
        model_config=types.SimpleNamespace(enable_return_routed_experts=False, is_hybrid=False),
        vllm_config=types.SimpleNamespace(),
    )
    worker.initialize_from_config(kv_cache_config)
    assert runner.called("_init_kv_zero_meta")


def test_the_runner_declines_the_kv_block_zeroer_with_a_logged_reason(caplog):
    """R3: vLLM's ``KVBlockZeroer`` sizes a block from ``stride(block_dim)``, half a page here.

    Constructing it would zero the K half of each new block and never the V half (physical
    ``[2][blocks][...]``), with no error.  The runner therefore overrides ``_init_kv_zero_meta``
    to log the decision and leave ``_kv_block_zeroer`` unset - which is what makes the inherited
    ``_zero_block_ids`` a no-op.
    """
    assert "_init_kv_zero_meta" in vars(worker_mod.TileInferModelRunner)
    runner = types.SimpleNamespace()
    with caplog.at_level(logging.WARNING, logger="tileinfer.vllm.worker"):
        worker_mod.TileInferModelRunner._init_kv_zero_meta(runner)
    assert not hasattr(runner, "_kv_block_zeroer")

    doc = inspect.getdoc(worker_mod.TileInferModelRunner._init_kv_zero_meta) or ""
    for needle in ("stride(block_dim)", "half a page", "mamba", "HARDWARE_CHECKLIST"):
        assert needle in doc, f"the skip decision must document {needle!r}"
    message = " ".join(record.getMessage() for record in caplog.records)
    assert "declining" in message and "KVBlockZeroer" in message
    assert "half a page" in message


def test_memory_pool_context_is_a_nullcontext():
    """The inherited body would reach ``get_mem_allocator_instance()`` (gpu_worker.py:218)."""
    worker = object.__new__(worker_mod.NPUWorker)
    for tag in ("weights", "kv_cache"):
        ctx = worker._maybe_get_memory_pool_context(tag)
        assert type(ctx) is contextlib.nullcontext
        with ctx:  # usable as a context manager
            pass
    assert "nullcontext" in (inspect.getdoc(worker._maybe_get_memory_pool_context) or "")


# =======================================================================================
# 6. capture_model / compile_or_warm_up_model
# =======================================================================================


def test_capture_model_is_a_logged_noop_while_capture_is_not_ready(monkeypatch, caplog):
    runner = _FakeRunner()
    worker = _worker_with_runner(runner, model_config=types.SimpleNamespace(enforce_eager=True))
    monkeypatch.setattr(
        worker_mod.graphs, "capture_path_ready", lambda: (False, "torch.npu.CUDAGraph is missing")
    )

    with caplog.at_level(logging.WARNING, logger="tileinfer.vllm.worker"):
        assert worker.capture_model() == 0

    # no-op: the runner was never asked to capture
    assert not runner.called("capture_model")
    assert not runner.called("profile_cudagraph_memory")
    # logged, with the reason and the eager flag
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "skipping graph capture" in message
        and "torch.npu.CUDAGraph is missing" in message
        and "enforce_eager=True" in message
        for message in messages
    ), messages
    # counted
    assert set(worker_mod.CAPTURE_COUNTER_KEYS) <= set(worker.capture_stats)
    assert worker.capture_stats["attempts"] == 1
    assert worker.capture_stats["skipped"] == 1
    assert worker.capture_stats["captured"] == 0
    assert worker.capture_stats["last_skip_reason"] == "torch.npu.CUDAGraph is missing"

    # the counter accumulates across calls
    assert worker.capture_model() == 0
    assert (worker.capture_stats["attempts"], worker.capture_stats["skipped"]) == (2, 2)


def test_capture_model_delegates_when_the_capture_path_is_ready(monkeypatch):
    runner = _FakeRunner()
    worker = _worker_with_runner(runner, model_config=types.SimpleNamespace(enforce_eager=False))
    monkeypatch.setattr(worker_mod.graphs, "capture_path_ready", lambda: (True, "bridged"))

    assert worker.capture_model() == 4096
    assert runner.called("capture_model")
    assert worker.capture_stats["captured"] == 1
    assert worker.capture_stats["skipped"] == 0


class _CompilationMode(enum.Enum):
    NONE = 0
    VLLM_COMPILE = 1


class _CUDAGraphMode(enum.Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


def _compilation_config(*, mode: Any, compile_sizes: Any = None) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        mode=mode,
        compile_sizes=compile_sizes,
        cudagraph_mode=_CUDAGraphMode.NONE,
        cudagraph_capture_sizes=None,
        get_compile_ranges=lambda: [],
        kv_cache_memory_bytes=None,
        compilation_time=1.5,
        encoder_compilation_time=2.5,
    )


def _install_warmup_fakes() -> List[str]:
    """The four vLLM modules ``compile_or_warm_up_model`` touches, plus a freeze recorder."""
    _setdefault(_ensure_chain("vllm.config"), "CUDAGraphMode", _CUDAGraphMode)
    _setdefault(_ensure_chain("vllm.config.compilation"), "CompilationMode", _CompilationMode)
    _setdefault(
        _ensure_chain("vllm.distributed.parallel_state"),
        "get_pp_group",
        lambda: types.SimpleNamespace(is_last_rank=lambda: True),
    )
    _setdefault(_ensure_chain("vllm.utils.torch_utils"), "set_random_seed", lambda seed: None)
    frozen: List[str] = []
    gc_mod = _ensure_chain("vllm.utils.gc_utils")
    _setdefault(gc_mod, "freeze_gc_heap", lambda: frozen.append("freeze"))
    _setdefault(gc_mod, "maybe_attach_gc_debug_callback", lambda: frozen.append("debug"))

    class CompilationTimes:
        """``worker_base.CompilationTimes`` (gpu_worker.py:747-750)."""

        def __init__(self, language_model: Any = None, encoder: Any = None) -> None:
            self.language_model = language_model
            self.encoder = encoder

    _setdefault(_ensure_chain("vllm.v1.worker.worker_base"), "CompilationTimes", CompilationTimes)
    return frozen


def test_compile_or_warm_up_model_skips_the_cuda_only_warmups(monkeypatch, caplog):
    """The engine's entry point: no Triton warm-up, no capture while gated, sampler warmed last."""
    frozen = _install_warmup_fakes()
    runner = _FakeRunner(model_memory_usage=100)
    vllm_config = types.SimpleNamespace(
        compilation_config=_compilation_config(mode=_CompilationMode.NONE)
    )
    worker = _worker_with_runner(
        runner,
        vllm_config=vllm_config,
        compilation_config=vllm_config.compilation_config,
        scheduler_config=types.SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=8),
        model_config=types.SimpleNamespace(seed=0, enforce_eager=True),
        cache_config=types.SimpleNamespace(gpu_memory_utilization=0.9),
    )
    worker.init_snapshot = _snapshot(total=100, free=50, peak=0)
    worker.requested_memory = 40
    worker.peak_activation_memory = 0
    worker.non_torch_memory = 0
    monkeypatch.setattr(worker_mod.graphs, "capture_path_ready", lambda: (False, "not bridged"))

    with caplog.at_level(logging.INFO, logger="tileinfer.vllm.worker"):
        times = worker.compile_or_warm_up_model()

    types_mod = sys.modules["vllm.v1.worker.worker_base"]
    assert isinstance(times, types_mod.CompilationTimes)
    assert (times.language_model, times.encoder) == (1.5, 2.5)

    # no compile sizes -> only the sampler warm-up runs, and it runs after the capture gate
    dummy_calls = [call for call in runner.calls if call[:1] == ("_dummy_run",)]
    assert len(dummy_calls) == 1
    assert dummy_calls[0][2].get("num_tokens") == 4  # min(max_num_seqs, max_num_batched_tokens)
    assert dummy_calls[0][2].get("skip_eplb") is True
    assert runner.called("_dummy_sampler_run")
    assert runner.called("maybe_remove_all_loras")
    assert not runner.called("_dummy_pooler_run")

    # the capture gate was reached and counted; the runner never captured
    assert worker.capture_stats["skipped"] == 1
    assert not runner.called("capture_model")

    # kernel_warmup / the Triton JIT monitor are not activated; the GC heap is frozen
    assert "vllm.triton_utils.jit_monitor" not in sys.modules
    assert frozen == ["freeze", "debug"]
    assert any("skipping kernel_warmup" in record.getMessage() for record in caplog.records)


# =======================================================================================
# 7. init_device (only what a CPU box can assert)
# =======================================================================================


def test_init_device_requires_torch_npu_and_says_so():
    if _torch_npu_importable():  # pragma: no cover - only on an Ascend box
        pytest.skip("torch_npu is importable here; the missing-torch_npu path cannot run")
    worker = object.__new__(worker_mod.NPUWorker)  # nothing is set: the probe must come first
    with pytest.raises(RuntimeError) as excinfo:
        worker.init_device()
    message = str(excinfo.value)
    assert "torch_npu" in message
    assert "init_device" in message
    assert "docs/hardware-checklist.md" in message


def test_init_device_rejects_a_non_npu_device_type(monkeypatch):
    monkeypatch.setattr(worker_mod, "_npu_module", lambda what: _fake_npu())
    worker = object.__new__(worker_mod.NPUWorker)
    worker.device_config = types.SimpleNamespace(device_type="cuda")
    with pytest.raises(RuntimeError) as excinfo:
        worker.init_device()
    message = str(excinfo.value)
    assert "'cuda'" in message
    assert "TileInferPlatform.device_type" in message


def test_npu_device_spelling_falls_back_to_privateuse1():
    """``torch.device('npu:0')`` only parses once torch_npu renamed the backend."""
    assert worker_mod._npu_device(0).type in worker_mod.NPU_DEVICE_TYPES


def test_set_npu_device_retries_the_index_form():
    calls: List[Any] = []

    def setter(device: Any) -> None:
        calls.append(device)
        if isinstance(device, torch.device):
            raise TypeError("expected an int")

    spelling = worker_mod._set_npu_device(_fake_npu(set_device=setter), NPU_DEVICE, 3)
    assert calls == [NPU_DEVICE, 3]
    assert "int" in spelling


# =======================================================================================
# 8. the model runner subclass
# =======================================================================================


def test_model_runner_subclass_is_needed_for_the_device_properties_seam(monkeypatch):
    """The GPU runner raises inside ``__init__``; the subclass must not."""
    assert issubclass(worker_mod.TileInferModelRunner, gpu_runner_mod.GPUModelRunner)

    base_seam = gpu_runner_mod.GPUModelRunner.__dict__.get("_init_device_properties")
    assert base_seam is not None, (
        "the fake GPUModelRunner must reproduce the seam (gpu_model_runner.py:1107-1112); "
        "whichever contract test installed the fake first owns this body"
    )
    with pytest.raises(NotImplementedError, match="num_compute_units"):
        base_seam(types.SimpleNamespace(device=NPU_DEVICE))

    npu = _fake_npu(get_device_properties=lambda index=0: types.SimpleNamespace(num_aicore=40))
    monkeypatch.setattr(worker_mod, "_npu_module", lambda what: npu)
    # no base __init__: the seam under test is the only thing being exercised
    runner = object.__new__(worker_mod.TileInferModelRunner)
    runner.device = NPU_DEVICE
    runner._init_device_properties()
    assert runner.num_sms == 40


def test_model_runner_warns_and_uses_the_sentinel_when_no_core_count_exists(monkeypatch, caplog):
    npu = _fake_npu(get_device_properties=lambda index=0: types.SimpleNamespace())
    monkeypatch.setattr(worker_mod, "_npu_module", lambda what: npu)
    runner = object.__new__(worker_mod.TileInferModelRunner)
    runner.device = NPU_DEVICE
    with caplog.at_level(logging.WARNING, logger="tileinfer.vllm.worker"):
        runner._init_device_properties()
    assert runner.num_sms == 1
    joined = " ".join(record.getMessage() for record in caplog.records)
    assert "sentinel 1" in joined
    for attr in worker_mod.NPU_CORE_COUNT_ATTRS:
        assert attr in joined


def test_model_runner_rejects_a_non_npu_device(monkeypatch):
    monkeypatch.setattr(worker_mod, "_npu_module", lambda what: _fake_npu())
    runner = object.__new__(worker_mod.TileInferModelRunner)
    runner.device = torch.device("cpu")
    with pytest.raises(ValueError, match="Ascend NPU"):
        runner._init_device_properties()


def test_model_runner_overrides_only_the_documented_seams():
    """Two overrides: ``:1108``'s invited seam and the KV-zeroer decline (same layout decision)."""
    own = {name for name in vars(worker_mod.TileInferModelRunner) if not name.startswith("__")}
    assert own == {"_init_device_properties", "_init_kv_zero_meta"}, own
