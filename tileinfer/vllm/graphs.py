r"""The graph-capture decision, and a thin wrapper around vLLM's ``CUDAGraphWrapper``.

The verdict, first
------------------
**Branch 2: "patchable" — with a bounded, enumerable patch surface (2 call sites) and five
outside symbols still unverified.**  It is *not* device-agnostic, and it does *not* have to be
rewritten yet.

Evidence (every line number is a line of ``reference/vllm-0.23/vllm__compilation__cuda_graph.py``,
361 lines, transcribed from ``vllm/compilation/cuda_graph.py`` @ v0.23.0):

==========================  ==========================================================  ======
line(s)                     what it does                                                branch
==========================  ==========================================================  ======
``:283``                    ``cudagraph = torch.cuda.CUDAGraph()``                      **CUDA**
``:313-317``                ``with torch.cuda.graph(cudagraph, pool=self.graph_pool,     **CUDA**
                            stream=current_stream()):``
``:130``                    ``cudagraph: torch.cuda.CUDAGraph | None = None``           name only
                            (a dataclass field annotation; no ``from __future__ import   (see
                            annotations`` in the file, so it *is* evaluated at class       below)
                            creation - it needs the **name** to exist, which a CPU torch
                            build also provides: ``torch/cuda/graphs.py:76-84,296``)
``:200``, ``:303-306``      ``current_platform.get_global_graph_pool()`` /              device-
                            ``set_graph_pool_id(...)``                                 agnostic*
``:233-254``                dispatch on ``get_forward_context()``                       device-
                            (``batch_descriptor`` + ``cudagraph_runtime_mode``)         agnostic
``:277``                    ``validate_cudagraph_capturing_enabled()``                  unknown
``:279-282``, ``:346-355``  input-address bookkeeping in DEBUG                          device-
                                                                                        agnostic
``:286-301``                optional ``gc.collect`` / ``torch.accelerator.empty_cache``  device-
                            suppression (patch *targets* are strings, not calls)         agnostic
``:310``, ``:324``,         offloader stream sync/join                                  unknown
``:359``
``:332``, ``:336``, ``:339``  ``weak_ref_tensors``, ``compilation_counter``               device-
                                                                                        agnostic
==========================  ==========================================================  ======

So exactly **two runtime statements in that file** touch CUDA.  Everything else in the capture
path is either device-agnostic (forward-context dispatch, address bookkeeping, weak refs,
counters) or resolved through the platform (``get_global_graph_pool`` →
``Platform.get_global_graph_pool``, which calls ``graph_pool_handle()`` and
``Platform.__getattr__`` forwards that to ``torch.<device_type>`` —
``vllm__platforms__interface.py:887-915``).  There is **no seam** to subclass: the two CUDA
statements live inline in ``CUDAGraphWrapper.__call__``, so "patchable" means either

(a) a **compat bridge** that aliases the two names (``torch.cuda.CUDAGraph = torch.npu.CUDAGraph``
    and ``torch.cuda.graph = torch.npu.graph``) before capture, so the file runs unmodified; or
(b) copy ``:233-361`` into our own class and swap those two lines.

(a) is what this module judges and what :func:`wrap` gates on; (b) is the fallback and is the
"must write our own" branch.

The alias table (why "the two names" is not the whole bridge)
-------------------------------------------------------------
Delegation does not stop at ``cuda_graph.py``: the GPU model runner's memory profiler and stream
setup call seven more CUDA names.  :data:`TORCH_CUDA_ALIASES` is the named, data-driven table of
every ``torch.cuda.<name>`` the capture/memory/stream path touches, each entry carrying its
``torch.npu`` counterpart, a ``required`` flag, and the local ``file:line`` that needs it:

* **required** (all-or-nothing, exactly the old semantics): ``CUDAGraph`` / ``graph`` for
  capture, and ``mem_get_info`` / ``Stream`` for the runner's memory profile and async copy
  streams.  :func:`install` writes all of them or none (if a counterpart is missing or a write
  fails, everything it wrote in that call is rolled back), and :func:`capture_path_ready`
  verifies every required alias **by identity**.
* **best-effort**: ``current_stream`` / ``stream`` / ``Event`` / ``OutOfMemoryError`` /
  ``set_stream``.  They sit on runner paths that can stay off (async output copy, prefill OOM
  handling) or, for ``set_stream``, are only *read at import*: ``vllm.utils.torch_utils`` (worker
  init imports it) binds ``prev_set_stream = torch.cuda.set_stream`` at module scope, then
  replaces it with its own shim, and its one call site is inside ``is_rocm() or is_cuda()``.  So
  :func:`install` aliases what it can and emits one warning naming what it could not.

Four names in the local sources are deliberately **not** mapped: ``is_available`` (a device
predicate, not a capture/memory call - aliasing it would make CUDA-shaped availability checks
answer for NPU and hide platform detection), ``get_device_properties`` (a docstring; the
initialiser reads ``torch.accelerator``), and ``get_device_name`` / ``get_device_capability``,
which the fetched FLA ops probe at module scope behind ``is_nvidia`` - ``qwen3_5.py`` does import
that module, but the guard short-circuits on NPU, so those two calls never evaluate.  The reasons
live in :data:`UNMAPPED_TORCH_CUDA_NAMES`; ``tests/test_platform_contract.py`` diffs the table
against a regex scan of ``reference/vllm-0.23/*.py``, so a new upstream call name cannot appear
silently.

\* device-agnostic **if** the forwarded symbols exist.  That is the part this module cannot
settle offline, and it is why :func:`capture_path_ready` exists instead of a plain comment.
The five symbols to check on the device (all outside ``reference/vllm-0.23/``):

1. ``vllm.utils.torch_utils.current_stream`` (imported ``:27``, used ``:316``) — must return the
   **NPU** stream; if it is ``torch.cuda.current_stream``, capture would run on the wrong stream.
   The stream shim it builds at import (``:669-:679``) is the table's optional ``set_stream``.
2. ``vllm.distributed.device_communicators.pynccl_allocator.set_graph_pool_id`` (``:18``,
   ``:304``, ``:306``) — NCCL-specific in name; must be a no-op (or NPU-aware) when pynccl is
   absent.
3. ``vllm.compilation.monitor.validate_cudagraph_capturing_enabled`` (``:16``, ``:277``) — must
   not assume CUDA stream state.
4. ``vllm.model_executor.offloader.get_offloader`` (``:25``, ``:310``, ``:324``, ``:359``) —
   ``sync_prev_onload`` / ``join_after_forward`` must exist and be device-agnostic.
5. ``torch.npu.graph`` / ``torch.npu.CUDAGraph`` / ``torch.npu.graph_pool_handle`` — the bridge
   targets; if torch_npu spells them differently, (a) is impossible and (b) is the only path.

If **any** of 1-4 turns out to be CUDA-only, the verdict flips to branch 3 ("must write our
own") and the work is ``:233-361`` re-expressed on ``torch.accelerator.Graph`` (present in the
torch bundled here: ``torch/accelerator/graphs.py:9`` — [unverified] for the torch
version the NPU deployment pins, which is not in this tree).

What the local sources do settle (the runner is in the subset now): the *caller*.
``gpu_model_runner.py:5249`` constructs ``CUDAGraphWrapper(self.model, ...)`` itself, and
``gpu_worker.py:630`` calls ``self.model_runner.capture_model()``.  Two consequences:

* the bridge must be installed **before** capture (at plugin install time); :func:`wrap` is then
  only a convenience for our own runner;
* once the runner builds the wrapper itself, its own ``torch.cuda.mem_get_info`` /
  ``torch.cuda.Stream`` calls (:6547, :696) run in the same process - which is why the alias table
  above is broader than the two graph names.

Capture contract
----------------
The plan's W8 acceptance is "``capture_calls > 0 && fallbacks == 0``, no Python work during
replay" (``docs/plan-vllm-native-backend.md:160``), and §6 of the same document spells out what a
*step* may do inside a capture: plan already built, no compile, no allocation, no sync
(``docs/plan-vllm-native-backend.md:226-229``).  :func:`capture_contract_ok` turns that into one
function with one reason code per violation, so a capture-time regression is a counter and a
string, not a NaN.

This module imports **no** vLLM at module scope: the graph decision and the contract must be
answerable on a CPU box (and inside the platform's ``check_and_update_config``, which asks
:func:`capture_path_ready` before deciding to force eager execution).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch

__all__ = [
    "GRAPH_BRANCH",
    "BRANCH_DEVICE_AGNOSTIC",
    "BRANCH_PATCHABLE",
    "BRANCH_WRITE_OUR_OWN",
    "CUDA_CALL_SITES",
    "UNVERIFIED_SYMBOLS",
    "CudaAlias",
    "TORCH_CUDA_ALIASES",
    "UNMAPPED_TORCH_CUDA_NAMES",
    "required_aliases",
    "optional_aliases",
    "missing_aliases",
    "active_aliases",
    "install",
    "uninstall",
    "is_capturing",
    "CaptureState",
    "CAPTURE_VIOLATION_CODES",
    "capture_contract_ok",
    "assert_capture_contract",
    "capture_path_ready",
    "wrap",
]

logger = logging.getLogger("tileinfer.vllm.graphs")

BRANCH_DEVICE_AGNOSTIC = "device-agnostic"
BRANCH_PATCHABLE = "patchable"
BRANCH_WRITE_OUR_OWN = "write our own"

#: The verdict, as a value a test can pin (see the module docstring for the evidence table).
GRAPH_BRANCH = BRANCH_PATCHABLE

#: The two statements that make the wrapper CUDA-only, with their line numbers.  A test asserts
#: this tuple does not grow silently, because every entry is one more thing a bridge has to fix.
#: (``:313-317`` is a single ``with`` statement; the text below is its first line.)
CUDA_CALL_SITES: Tuple[Tuple[int, str], ...] = (
    (283, "cudagraph = torch.cuda.CUDAGraph()"),
    (313, "with torch.cuda.graph("),
)

#: Symbols the bridge decision depends on that are *not* in ``reference/vllm-0.23/``.  Each entry is
#: ``(symbol, why it matters)``.  `capture_path_ready` can only check the torch-side ones; the
#: vLLM-side ones are listed here so the hardware pass has a checklist instead of a hunch.
UNVERIFIED_SYMBOLS: Tuple[Tuple[str, str], ...] = (
    (
        "vllm.utils.torch_utils.current_stream",
        "imported at cuda_graph.py:27 and passed as the capture stream at :316; must be the NPU "
        "stream, not torch.cuda.current_stream",
    ),
    (
        "vllm.distributed.device_communicators.pynccl_allocator.set_graph_pool_id",
        "called at cuda_graph.py:304/:306; NCCL-specific name, must be a no-op without pynccl",
    ),
    (
        "vllm.compilation.monitor.validate_cudagraph_capturing_enabled",
        "called at cuda_graph.py:277 just before capture; must not inspect CUDA stream state",
    ),
    (
        "vllm.model_executor.offloader.get_offloader",
        "cuda_graph.py:310/:324/:359 call sync_prev_onload/join_after_forward; must be "
        "device-agnostic",
    ),
    (
        "torch.npu.CUDAGraph / torch.npu.graph / torch.npu.graph_pool_handle",
        "the bridge targets; graph_pool_handle is what Platform.get_global_graph_pool reaches "
        "through __getattr__ (platforms/interface.py:887-915)",
    ),
)


# =============================================================================================
# The alias table: every torch.cuda.<name> the capture/memory path touches
# =============================================================================================


@dataclass(frozen=True)
class CudaAlias:
    """One ``torch.cuda.<name>`` bridge entry.

    ``name`` is the attribute on ``torch.cuda`` that vLLM's code reads; ``target`` is its
    counterpart on ``torch.npu`` (today always the same spelling, recorded explicitly so a
    device that names it differently is a data edit, not a code edit).  ``required`` entries are
    all-or-nothing (a missing one disables capture, and :func:`install` writes none of them);
    optional entries are best-effort.  ``source`` is the local ``file:line`` that needs the name,
    so the table can be audited against the sources it was derived from.
    """

    name: str
    target: str
    required: bool
    source: str
    note: str = ""

    @property
    def cuda_path(self) -> str:
        return f"torch.cuda.{self.name}"

    @property
    def npu_path(self) -> str:
        return f"torch.npu.{self.target}"

    def describe(self) -> str:
        kind = "required" if self.required else "best-effort"
        return f"{self.cuda_path} -> {self.npu_path} [{kind}, {self.source}]"


#: Every ``torch.cuda.<name>`` that the capture/memory/stream path touches, in install order.
#: Derived by scanning ``reference/vllm-0.23/*.py`` for ``torch.cuda.<name>`` (13 unique names:
#: 9 mapped here, 4 in :data:`UNMAPPED_TORCH_CUDA_NAMES`).  The contract test re-runs that scan,
#: so a new upstream call name fails the suite instead of failing at capture time.
TORCH_CUDA_ALIASES: Tuple[CudaAlias, ...] = (
    CudaAlias(
        "CUDAGraph",
        "NPUGraph",
        True,
        "vllm__compilation__cuda_graph.py:283",
        "capture: `cudagraph = torch.cuda.CUDAGraph()`; the field annotation at :130 also "
        "evaluates the name at class creation.  The NPU counterpart is called **NPUGraph** on "
        "torch_npu (verified on Ascend950PR with torch_npu 2.10.0.post4: torch.npu.NPUGraph "
        "exists, torch.npu.CUDAGraph does not), so the target name is not the identity here",
    ),
    CudaAlias(
        "graph",
        "graph",
        True,
        "vllm__compilation__cuda_graph.py:313",
        "capture: `with torch.cuda.graph(cudagraph, pool=..., stream=current_stream())`",
    ),
    CudaAlias(
        "mem_get_info",
        "mem_get_info",
        True,
        "vllm__v1__worker__gpu_model_runner.py:6547",
        "memory: profile_run start/end free memory (:6547, :6565), profile_cudagraph_memory "
        "(:6442, :6456, :6478, :6481); gpu_worker.py:166/:177 (sleep) call it too",
    ),
    CudaAlias(
        "Stream",
        "Stream",
        True,
        "vllm__v1__worker__gpu_model_runner.py:696",
        "stream: `torch.cuda.Stream()` only with --async-scheduling; the async output-copy "
        "streams construct it too (:849, :874, :883, :1121; annotations :246, :370, :691, "
        ":840, :863, :867, :1118)",
    ),
    CudaAlias(
        "current_stream",
        "current_stream",
        False,
        "vllm__v1__worker__gpu_model_runner.py:264",
        "stream: the async output copy saves/restores the current stream (:264, :382, :4717, "
        ":4748)",
    ),
    CudaAlias(
        "stream",
        "stream",
        False,
        "vllm__v1__worker__gpu_model_runner.py:265",
        "stream: `with torch.cuda.stream(...)` around the async copies (:265, :383, :4719, "
        ":4751) and around the fused-MoE shared-expert aux stream (shared_experts.py:141).  The "
        "aux stream itself only exists when current_platform.is_cuda_alike() "
        "(torch_utils.py:744), so that overlap path stays off on NPU",
    ),
    CudaAlias(
        "Event",
        "Event",
        False,
        "vllm__v1__worker__gpu_model_runner.py:848",
        "stream: the draft-token copy event (:839 annotation, :848 constructor)",
    ),
    CudaAlias(
        "OutOfMemoryError",
        "OutOfMemoryError",
        False,
        "vllm__v1__worker__gpu_model_runner.py:5176",
        "memory: `except torch.cuda.OutOfMemoryError` around the prefill path",
    ),
    CudaAlias(
        "set_stream",
        "set_stream",
        False,
        "vllm__utils__torch_utils.py:669",
        "stream: worker init imports vllm.utils.torch_utils (gpu_worker.py:61, "
        "cuda_graph.py:27 - also platforms/__init__.py:12), whose module scope reads "
        "`prev_set_stream = torch.cuda.set_stream` (:669) and then overwrites it with its own "
        "shim (:679); the only call site (:712) is inside `is_rocm() or is_cuda()`, which an "
        "NPU run never enters.  Best-effort, not required: the read needs no bridge (torch "
        "defines set_stream unconditionally) and a missing torch.npu.set_stream must not gate "
        "capture readiness",
    ),
)

#: Names in the local sources that are deliberately *not* aliased, and why.  Kept as data (not as
#: prose in a docstring) so the coverage test can prove the inventory is exhaustive.
UNMAPPED_TORCH_CUDA_NAMES: Tuple[Tuple[str, str], ...] = (
    (
        "is_available",
        "vllm__platforms__interface.py:192 and vllm__ir__op.py:277 (docstring examples) plus "
        "vllm__v1__worker__gpu_worker.py:277 (DP visible-device-count check): a device predicate, "
        "not a capture/memory call. Aliasing it would make CUDA-shaped availability checks answer "
        "for NPU and hide platform detection; torch_npu already routes the real predicate.",
    ),
    (
        "get_device_capability",
        "vllm__platforms__interface.py:311 (a docstring: 'Stateless version of "
        "torch.cuda.get_device_capability'; the method returns None and never calls torch.cuda) "
        "and vllm__model_executor__layers__fla__ops__utils.py:153, where it sits in the same "
        "module-level `is_nvidia and (...)` probe as get_device_name below: the call is behind the "
        "short-circuit and never evaluates on NPU",
    ),
    (
        "get_device_name",
        "vllm__model_executor__layers__fla__ops__utils.py:152 only, inside the module-level "
        "`is_nvidia_hopper = is_nvidia and ('NVIDIA H' in torch.cuda.get_device_name(0) or ...)` "
        "probe. The module *is* imported on the Qwen3.6 path (qwen3_5.py:43 -> "
        "qwen_gdn_linear_attn.py:32-33 -> fla/ops/utils.py), but on NPU "
        "`current_platform.is_cuda_alike()` is False, so `device` comes from the Triton backend "
        "and `is_nvidia` is False - the `and` short-circuits before the call. Aliasing it would "
        "only make a CUDA vendor probe answer for NPU",
    ),
    (
        "get_device_properties",
        "vllm__v1__worker__gpu_model_runner.py:1110 only in a docstring; the initialiser reads "
        "torch.accelerator, not torch.cuda",
    ),
)

#: ``torch.cuda`` attributes written by :func:`install`, mapped to the value they had before the
#: first write.  :data:`_MISSING` records "the attribute did not exist", so :func:`uninstall` can
#: delete it rather than invent one.
_ALIAS_SAVED: Dict[str, Any] = {}
_MISSING = object()


def required_aliases() -> Tuple[CudaAlias, ...]:
    """The all-or-nothing set: capture needs two, the runner memory/stream path needs two."""
    return tuple(alias for alias in TORCH_CUDA_ALIASES if alias.required)


def optional_aliases() -> Tuple[CudaAlias, ...]:
    """The best-effort set; a missing counterpart is logged, not fatal."""
    return tuple(alias for alias in TORCH_CUDA_ALIASES if not alias.required)


def missing_aliases(npu: Any = None, *, required_only: bool = False) -> Tuple[str, ...]:
    """``torch.npu.<name>`` paths this table needs but the module does not expose.

    Answers the "does the device have anything to bridge onto?" half; identity is checked by
    :func:`capture_path_ready` and by :func:`install`.  ``npu`` defaults to ``torch.npu``.
    """
    module = getattr(torch, "npu", None) if npu is None else npu
    names: List[str] = []
    for alias in TORCH_CUDA_ALIASES:
        if required_only and not alias.required:
            continue
        if module is None or getattr(module, alias.target, _MISSING) is _MISSING:
            names.append(alias.npu_path)
    return tuple(names)


def _cuda_module() -> Any:
    return getattr(torch, "cuda", None)


def active_aliases() -> Tuple[str, ...]:
    """The ``torch.cuda.<name>`` attributes :func:`install` wrote and has not restored yet."""
    return tuple(f"torch.cuda.{name}" for name in _ALIAS_SAVED)


def _counterpart(npu: Any, alias: CudaAlias) -> Any:
    """``torch.npu.<target>``, or :data:`_MISSING` (never raises for a broken module)."""
    try:
        return getattr(npu, alias.target)
    except Exception:  # noqa: BLE001 - a module with a raising __getattr__ is "missing"
        return _MISSING


def _save_original(name: str) -> None:
    if name not in _ALIAS_SAVED:
        _ALIAS_SAVED[name] = getattr(_cuda_module(), name, _MISSING)


def _restore_alias(name: str) -> None:
    """Put one ``torch.cuda`` attribute back exactly as :func:`install` found it."""
    original = _ALIAS_SAVED[name]
    cuda = _cuda_module()
    if original is _MISSING:
        if hasattr(cuda, name):
            delattr(cuda, name)
    else:
        setattr(cuda, name, original)
    _ALIAS_SAVED.pop(name, None)


def install(npu: Any = None) -> Tuple[bool, str]:
    """Bridge the table onto ``torch.cuda``; all-or-nothing for required, best-effort optional.

    Returns ``(installed, detail)``:

    * ``False`` when ``torch.npu`` / ``torch.cuda`` is unavailable, when a **required** counterpart
      is missing, or when writing one required alias fails.  In every one of those cases nothing
      this call wrote is left behind - a half-bridged ``torch.cuda`` (capture stream bridged, its
      memory probe not) is exactly the state that makes a failure look like a kernel bug.
    * ``True`` when the required set is bridged.  Optional aliases that cannot be written are
      named in one warning and in ``detail``; they never flip the result, because the runner paths
      that use them can stay off.

    Names that are already identical to their counterpart are not touched (so this never claims,
    or later reverts, a bridge installed by torch_npu or by ``tileinfer.vllm.plugin``).
    """
    cuda = _cuda_module()
    if cuda is None:
        return (
            False,
            "torch.cuda is not present in this build, so there is no CUDA-named surface to "
            "bridge onto torch.npu",
        )
    module = getattr(torch, "npu", None) if npu is None else npu
    if module is None:
        return (
            False,
            "torch.npu is not importable, so there is nothing to alias torch.cuda onto "
            "(this is the CPU/GPU box case, not an NPU failure)",
        )

    missing_required = missing_aliases(module, required_only=True)
    if missing_required:
        return (
            False,
            "the required aliases were not installed (all-or-nothing): torch.npu exposes no "
            f"counterpart for {', '.join(missing_required)}. Capture and the runner's memory "
            "profile would half-work, which is the failure mode this table exists to prevent",
        )

    applied: List[str] = []
    writing = "<none>"
    try:
        for alias in required_aliases():
            counterpart = _counterpart(module, alias)
            if counterpart is _MISSING:  # pragma: no cover - prechecked above
                raise AttributeError(f"{alias.npu_path} is missing")
            if getattr(cuda, alias.name, _MISSING) is counterpart:
                continue  # already bridged by torch_npu/plugin; not ours to revert
            writing = alias.cuda_path
            _save_original(alias.name)
            try:
                setattr(cuda, alias.name, counterpart)
            except Exception:
                _ALIAS_SAVED.pop(alias.name, None)  # the write never happened
                raise
            applied.append(alias.name)
    except Exception as exc:  # noqa: BLE001 - any required failure means none of them
        for name in applied:
            try:
                _restore_alias(name)
            except Exception:  # pragma: no cover - a module that refuses both directions
                logger.warning("TileInfer graph bridge: could not roll torch.cuda.%s back", name)
        if applied:
            rollback = ", ".join(f"torch.cuda.{name}" for name in applied) + " were rolled back"
        else:
            rollback = "nothing had been written"
        return (
            False,
            "the required aliases were not installed (all-or-nothing): writing "
            f"{writing} failed with {exc!r}; {rollback}. A partially bridged torch.cuda would "
            "capture on the wrong stream or profile the wrong memory",
        )

    skipped: List[str] = []
    for alias in optional_aliases():
        try:
            counterpart = _counterpart(module, alias)
            if counterpart is _MISSING:
                raise AttributeError(f"{alias.npu_path} is missing")
            if getattr(cuda, alias.name, _MISSING) is counterpart:
                continue
            _save_original(alias.name)
            try:
                setattr(cuda, alias.name, counterpart)
            except Exception:
                _ALIAS_SAVED.pop(alias.name, None)
                raise
        except Exception as exc:  # noqa: BLE001 - best-effort by contract
            skipped.append(f"torch.cuda.{alias.name}")
            logger.debug(
                "TileInfer graph bridge: %s is best-effort and could not be aliased (%r)",
                alias.cuda_path,
                exc,
            )
    if skipped:
        logger.warning(
            "TileInfer graph bridge: could not alias the best-effort name(s) %s; the runner "
            "path that uses them (%s) still calls torch.cuda. Capture is unaffected; keep that "
            "path off until the counterpart exists",
            ", ".join(skipped),
            ", ".join(
                f"{alias.source}" for alias in optional_aliases() if alias.cuda_path in skipped
            ),
        )

    if not applied:
        detail = "the required torch.cuda aliases were already bridged; nothing to change"
    else:
        detail = "aliased " + ", ".join(f"torch.cuda.{name}" for name in applied)
    if skipped:
        detail += "; best-effort skipped: " + ", ".join(skipped)
    return (True, detail)


def uninstall() -> Tuple[str, ...]:
    """Restore every ``torch.cuda`` attribute :func:`install` wrote; return the names restored.

    Only names this module actually wrote are reverted (``_ALIAS_SAVED``), so a pre-existing
    torch_npu/plugin bridge survives.  A restore that fails is logged and left recorded for a
    retry rather than silently dropped.
    """
    restored: List[str] = []
    if not _ALIAS_SAVED:
        logger.debug("TileInfer graph bridge: no aliases to restore")
        return ()
    cuda = _cuda_module()
    for name in list(_ALIAS_SAVED):
        try:
            if cuda is None:  # pragma: no cover - torch always has torch.cuda
                raise RuntimeError("torch.cuda disappeared")
            _restore_alias(name)
            restored.append(f"torch.cuda.{name}")
        except Exception as exc:  # noqa: BLE001 - reported, retried next call
            logger.warning(
                "TileInfer graph bridge: could not restore torch.cuda.%s (%r); it stays recorded",
                name,
                exc,
            )
    logger.info(
        "TileInfer graph bridge: restored %s",
        ", ".join(restored) if restored else "nothing",
    )
    return tuple(restored)


def _bridged(alias: CudaAlias, npu: Any) -> bool:
    """True iff ``torch.cuda.<name>`` *is* the npu counterpart - identity, not similarity."""
    counterpart = _counterpart(npu, alias)
    if counterpart is _MISSING:
        return False
    return getattr(_cuda_module(), alias.name, _MISSING) is counterpart


# =============================================================================================
# Which device would capture, and is it capturing now
# =============================================================================================


def _accelerator() -> Tuple[str, Any]:
    """``("npu" | "cuda" | "", module)`` — the device whose graph primitives would be used.

    NPU first: this backend exists for that device, and on an Ascend box ``torch.cuda`` is
    present as a module (with dummy types: ``torch/cuda/graphs.py:76-84``) but
    unusable.  Never raises: an unavailable device is an answer, not an error.
    """
    npu = getattr(torch, "npu", None)
    if npu is not None:
        try:
            if npu.is_available():
                return "npu", npu
        except Exception:  # pragma: no cover - a broken torch_npu
            pass
    try:
        if torch.cuda.is_available():
            return "cuda", torch.cuda
    except Exception:  # pragma: no cover - a CPU-only torch build
        pass
    return "", None


def is_capturing() -> bool:
    """True iff the current stream is inside a graph capture right now.

    This is the predicate every "must not allocate / compile / sync" assertion in the step path
    should use.  It is deliberately a *device probe*, not a config flag: ``enforce_eager`` /
    ``cudagraph_mode`` say what the engine *intends*, while capture can also be triggered by the
    profile run or by a manual ``torch.npu.graph`` block.

    * CUDA: ``torch.cuda.is_current_stream_capturing()`` — a thin wrapper over
      ``_cuda_isCurrentStreamCapturing`` (``torch/cuda/graphs.py:87-92``).
    * NPU: ``torch.npu.is_current_stream_capturing()`` — [unverified] that torch_npu exposes the
      same name; if it does not, this raises instead of answering False, because a False
      returned by a missing probe would silently re-enable allocation inside a capture.
    * No accelerator: False, and that is a *proof*, not a guess — nothing can be capturing when
      there is no device to capture on.
    """
    kind, module = _accelerator()
    if not kind:
        return False
    probe = getattr(module, "is_current_stream_capturing", None)
    if not callable(probe):
        raise NotImplementedError(
            f"{kind} exposes no is_current_stream_capturing(); tileinfer.vllm.graphs."
            "is_capturing() cannot answer, and answering False would let the step path allocate "
            "inside a captured region. Find the equivalent probe on this device (CUDA's is "
            "torch/cuda/graphs.py:87-92) and teach this function about it."
        )
    return bool(probe())


# =============================================================================================
# The capture contract: plan prewarmed / no compile / no alloc / no sync
# =============================================================================================

#: One code per way a captured step can be illegal.  The first three reuse the counter names the
#: attention impl already bumps (``tileinfer/vllm/attn/impl.py:113-149``: ``plan_not_ready``,
#: ``plan_builds``, ``state_builds``), so a probe can join the two sources on the string.
CAPTURE_VIOLATION_CODES: Tuple[str, ...] = (
    "plan_not_ready",  # the serving plan was not prewarmed for this bucket
    "plan_builds",  # a plan was compiled inside the capture
    "state_builds",  # per-bucket state was allocated inside the capture
    "compiled",  # a JIT/compile happened inside the capture
    "allocated",  # device memory was allocated inside the capture
    "synchronized",  # a device/host sync happened inside the capture
    "fallback",  # a degraded path was taken inside the capture
)


@dataclass(frozen=True)
class CaptureState:
    """What a captured step did, as counts.  Constructed by the caller, one per capture.

    The defaults describe a *perfect* capture (plan ready, nothing built, allocated, compiled or
    synced), so a caller that only knows some of the fields still gets a truthful answer for the
    rest.  Only :attr:`plan_ready` is mandatory: an unknown plan state is exactly the "forgot to
    prewarm" bug the contract exists to catch, so :func:`capture_contract_ok` refuses to guess it.
    """

    plan_ready: bool
    plan_builds: int = 0
    state_builds: int = 0
    compile_calls: int = 0
    allocated_bytes: int = 0
    sync_calls: int = 0
    fallbacks: int = 0
    capture_calls: int = 0


def capture_contract_ok(state: Any) -> Tuple[bool, str]:
    """``(ok, reason)`` — the "no compile / no alloc / no sync / plan prewarmed" assertions.

    ``state`` may be a :class:`CaptureState` or any object with the same attributes (the plan
    cache and the counters live in different modules; a probe joins them by duck-typing).  Fields
    other than ``plan_ready`` default to 0, i.e. "nothing illegal happened", and only
    ``plan_ready`` is required — if it is missing, that is a programming error and a
    ``TypeError``, not a silent pass.

    The checks run in the order of the plan's own list
    (``docs/plan-vllm-native-backend.md:226-229``) and each failure returns its code as the first
    word of the reason, so logs and counters agree.
    """
    if not hasattr(state, "plan_ready"):
        raise TypeError(
            "capture_contract_ok() needs an object with a 'plan_ready' attribute "
            f"(tileinfer.vllm.graphs.CaptureState is the reference shape); got "
            f"{type(state).__name__}. Refusing to report a capture as legal without knowing "
            "whether the plan was prewarmed."
        )

    plan_ready = bool(getattr(state, "plan_ready"))
    plan_builds = int(getattr(state, "plan_builds", 0) or 0)
    state_builds = int(getattr(state, "state_builds", 0) or 0)
    compile_calls = int(getattr(state, "compile_calls", 0) or 0)
    allocated_bytes = int(getattr(state, "allocated_bytes", 0) or 0)
    sync_calls = int(getattr(state, "sync_calls", 0) or 0)
    fallbacks = int(getattr(state, "fallbacks", 0) or 0)

    if not plan_ready:
        return (
            False,
            "plan_not_ready: the serving plan for this shape was not prewarmed before capture, "
            "so the captured region would compile (docs/plan-vllm-native-backend.md:226-228)",
        )
    if plan_builds:
        return (
            False,
            f"plan_builds: {plan_builds} plan(s) were built inside the captured region; every "
            "bucket must be warm before capture (serving/planner.py prewarm)",
        )
    if state_builds:
        return (
            False,
            f"state_builds: {state_builds} per-bucket state object(s) were allocated inside the "
            "captured region (tileinfer/vllm/attn/impl.py:626-703)",
        )
    if compile_calls:
        return (
            False,
            f"compiled: {compile_calls} JIT/compile call(s) inside the captured region",
        )
    if allocated_bytes:
        return (
            False,
            f"allocated: {allocated_bytes} byte(s) of device memory allocated inside the "
            "captured region",
        )
    if sync_calls:
        return (
            False,
            f"synchronized: {sync_calls} device/host sync(s) inside the captured region",
        )
    if fallbacks:
        return (
            False,
            f"fallback: {fallbacks} degraded path(s) inside the captured region; W8 requires "
            "fallbacks == 0 (docs/plan-vllm-native-backend.md:160)",
        )
    return (True, "ok")


def assert_capture_contract(state: Any) -> None:
    """Raise ``RuntimeError`` when :func:`capture_contract_ok` says the capture was illegal.

    Call this at the *end* of a capture (or from the W8 probe).  It is an assertion, not a
    fallback: a capture that took a degraded path produces silently wrong numerics, so the only
    acceptable response is to stop.
    """
    ok, reason = capture_contract_ok(state)
    if not ok:
        raise RuntimeError(
            "captured graph violates the TileInfer capture contract: "
            f"{reason}. The graph must be recaptured after prewarming, or the step must be run "
            "eagerly (tileinfer/vllm/graphs.py, docs/plan-vllm-native-backend.md §6)."
        )


# =============================================================================================
# Can vLLM's wrapper be used as-is?
# =============================================================================================


def capture_path_ready() -> Tuple[bool, str]:
    """``(ready, reason)`` — can vLLM capture *and* profile on this device as it stands?

    ``ready=True`` only in one of two situations, both verified at the object level rather than
    declared:

    * the accelerator is CUDA — the sources were written for it (``cuda_graph.py:283``,
      ``:313``; ``gpu_model_runner.py:6547``; ``gpu_worker.py:166``);
    * the accelerator is NPU **and** every **required** alias in :data:`TORCH_CUDA_ALIASES` is
      already bridged onto torch_npu's counterpart (identity, not "looks similar"), *and*
      ``graph_pool_handle`` exists — without it ``CUDAGraphWrapper.__init__`` (``:200``) raises
      inside ``Platform.get_global_graph_pool`` (``platforms/interface.py:908-915``).

    The required set covers the runner as well as capture: a bridge that aliases only
    ``CUDAGraph``/``graph`` would still call ``torch.cuda.mem_get_info`` inside
    ``profile_cudagraph_memory`` and construct a CUDA ``Stream`` under ``--async-scheduling``.
    Optional aliases are *not* part of the verdict (they are best-effort by contract); their
    absence is logged by :func:`install`, not hidden here.

    Everything else is a reason string naming what is missing, because the only two honest
    answers here are "delegate" and "do not delegate" — a half-bridged capture would produce a
    graph of the wrong stream or fail inside the runner.
    """
    kind, module = _accelerator()
    if not kind:
        return (
            False,
            "no accelerator is available (neither torch.npu nor torch.cuda reports one), so "
            "there is nothing to capture on",
        )

    torch_cuda = getattr(torch, "cuda", None)
    cuda_graph_cls = getattr(torch_cuda, "CUDAGraph", None)
    cuda_graph_cm = getattr(torch_cuda, "graph", None)

    if kind == "cuda":
        if cuda_graph_cls is None or cuda_graph_cm is None:  # pragma: no cover - broken torch
            return (
                False,
                "torch.cuda reports an available device but exposes no CUDAGraph/graph; "
                "vllm.compilation.cuda_graph cannot run",
            )
        return (True, "CUDA: vllm.compilation.cuda_graph was written for this backend")

    # kind == "npu"
    missing_required = missing_aliases(module, required_only=True)
    if missing_required:
        return (
            False,
            "torch.npu exposes no counterpart for the required alias(es) "
            f"{', '.join(missing_required)}, so the CUDA call sites in "
            "vllm/compilation/cuda_graph.py (:283 torch.cuda.CUDAGraph, :313 torch.cuda.graph) "
            "and in the model runner's memory profile (gpu_model_runner.py:6547, "
            "gpu_worker.py:166) have nothing to be bridged onto; branch 3 (write our own "
            "wrapper) is then the only option",
        )

    unbridged = [alias.cuda_path for alias in required_aliases() if not _bridged(alias, module)]
    if unbridged:
        return (
            False,
            f"{', '.join(unbridged)} are not bridged onto torch.npu's required graph/memory/"
            "stream primitives (identity is required, a look-alike is not enough). "
            "vllm/compilation/cuda_graph.py:283/:313 would capture on the CUDA backend, and "
            "gpu_model_runner.py:6547 / gpu_worker.py:166 would report CUDA memory. Install the "
            "compat bridge (tileinfer.vllm.graphs.install, branch 2) before capturing, or write "
            "our own wrapper (branch 3)",
        )
    if not callable(getattr(module, "graph_pool_handle", None)):
        return (
            False,
            "torch.npu.graph_pool_handle is missing, but CUDAGraphWrapper.__init__ "
            "(cuda_graph.py:200) calls current_platform.get_global_graph_pool(), which reaches "
            "graph_pool_handle() through Platform.__getattr__ (platforms/interface.py:887-915). "
            "Override TileInferPlatform.get_global_graph_pool before enabling capture",
        )
    bridged_names = ", ".join(alias.cuda_path for alias in required_aliases())
    return (
        True,
        f"NPU: the required aliases ({bridged_names}) are bridged onto torch.npu, so the "
        "wrapper and the runner's memory profile can run",
    )


def wrap(
    model: Any,
    vllm_config: Any,
    runtime_mode: Any = None,
    *,
    cudagraph_options: Any = None,
) -> Any:
    """Wrap ``model`` in vLLM's ``CUDAGraphWrapper`` if the branch allows it, else raise.

    The branch allows it when :func:`capture_path_ready` says the device's graph primitives are
    usable by that file as it stands.  On this CPU dev box (and on a bare NPU box with no bridge)
    it does not, so this raises :class:`NotImplementedError` with the reason and the next step -
    the point of the module being to make that decision explicit rather than to guess.

    ``runtime_mode`` is vLLM's ``CUDAGraphMode`` (``cuda_graph.py:182``, imported from
    ``vllm.config``).  When omitted it is read from
    ``vllm_config.compilation_config.cudagraph_mode`` (the field the local Worker reads:
    ``gpu_worker.py:603``); ``CUDAGraphWrapper`` asserts it is not ``NONE``
    (``cuda_graph.py:194-196``), and so does this function - wrapping in eager mode would build a
    wrapper that never captures, which looks identical to a working graph path in a profile.
    """
    ready, reason = capture_path_ready()
    if not ready:
        raise NotImplementedError(
            "cannot delegate graph capture to vllm.compilation.cuda_graph.CUDAGraphWrapper: "
            f"{reason}. The local file has exactly {len(CUDA_CALL_SITES)} CUDA call sites "
            f"({', '.join(f':{line}' for line, _ in CUDA_CALL_SITES)}) on the capture path, so "
            "the branch is 'patchable', not 'device-agnostic' - the bridge or our own wrapper "
            "has to exist first. See tileinfer/vllm/graphs.py's docstring for the five symbols "
            "that still need verifying on hardware."
        )

    from vllm.compilation.cuda_graph import CUDAGraphWrapper

    if runtime_mode is None:
        runtime_mode = _runtime_mode_from_config(vllm_config)
    _reject_none_runtime_mode(runtime_mode)
    logger.info("TileInfer: delegating graph capture to vllm CUDAGraphWrapper (%s)", reason)
    return CUDAGraphWrapper(model, vllm_config, runtime_mode, cudagraph_options)


def _runtime_mode_from_config(vllm_config: Any) -> Any:
    """Read ``compilation_config.cudagraph_mode`` (the field ``gpu_worker.py:603`` reads)."""
    from vllm.config import CUDAGraphMode

    compilation_config = getattr(vllm_config, "compilation_config", None)
    if compilation_config is None:
        raise NotImplementedError(
            "vllm_config has no compilation_config, so CUDAGraphMode cannot be determined; pass "
            "runtime_mode explicitly (tileinfer.vllm.graphs.wrap(..., runtime_mode=<mode>))."
        )
    mode = getattr(compilation_config, "cudagraph_mode", None)
    if mode is None:
        raise NotImplementedError(
            "compilation_config has no cudagraph_mode attribute (the field the local Worker "
            "reads at vllm/v1/worker/gpu_worker.py:603), so the capture mode must be passed "
            "explicitly; current CUDAGraphMode is {}".format(list(CUDAGraphMode))
        )
    _reject_none_runtime_mode(mode)
    return mode


def _reject_none_runtime_mode(runtime_mode: Any) -> None:
    """Refuse to build a wrapper that can never capture (``cuda_graph.py:194-196``)."""
    name = getattr(runtime_mode, "name", None)
    if name == "NONE" or runtime_mode is None:
        raise ValueError(
            f"CUDAGraphMode is {runtime_mode!r}: no capture is requested, so wrapping the model "
            "would build a wrapper that never captures (CUDAGraphWrapper asserts the mode is not "
            "NONE, cuda_graph.py:194-196). Do not wrap in eager mode - the caller gates on "
            "model_config.enforce_eager (gpu_worker.py:628-630)."
        )
