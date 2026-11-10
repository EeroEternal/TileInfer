"""Controlled registry for every change we make to vLLM's own behaviour.

Why a registry instead of scattered monkeypatches
-------------------------------------------------
A backend that owns the platform *will* have to adjust a handful of upstream assumptions (vLLM
calls ``torch.cuda.*`` directly in a few places, and it decides graph capture and memory profiling
without asking the platform).  The failure mode of ad-hoc patching is well documented in this
repository's own history: a patch that silently stops applying after an upgrade, or one that
applies in the API server but not in the EngineCore.  Both look like "the engine is slow" or
"the engine hangs", not like "a patch is missing".

So every patch is a :class:`Patch` record with the three facts a future maintainer needs:

* **why** it exists (what breaks without it),
* **since** which vLLM version it is required,
* **remove_when** - the condition under which it must be deleted (usually "when upstream uses
  ``current_platform`` here"), so the list shrinks instead of growing.

``install_all`` is failure-tolerant by design: a target that no longer exists is *skipped and
logged*, not raised.  A vLLM point release that renames a private function must not turn a
serving process into a crash before it has served a single request - the patch is missing, the
symptom is visible in the log and in ``tileinfer doctor``, and the backend degrades.

Nothing in this module imports vLLM: targets are resolved lazily, at install time.
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Patch",
    "InstallReport",
    "register",
    "install_all",
    "describe",
    "resolve",
    "set_attr",
    "wrap_attr",
    "PATCHES",
]

logger = logging.getLogger("tileinfer.vllm.compat")


def resolve(target: str) -> Tuple[Any, str]:
    """Resolve ``"pkg.mod.Class.attr"`` to ``(owner, "attr")`` without importing more than needed.

    The longest importable module prefix wins, so a nested class attribute and a module-level
    function both work.
    """
    parts = target.split(".")
    if len(parts) < 2:
        raise ValueError(f"patch target {target!r} must be at least 'module.attr'")
    for split in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split])
        try:
            owner: Any = importlib.import_module(module_name)
        except ImportError:
            continue
        for attribute in parts[split:-1]:
            owner = getattr(owner, attribute)
        name = parts[-1]
        if not hasattr(owner, name):
            # Raising here is what makes `install_all` treat "target renamed upstream" as a skip.
            raise AttributeError(f"{target!r}: {type(owner).__name__} has no attribute {name!r}")
        return owner, name
    raise ImportError(f"no importable module prefix in patch target {target!r}")


def set_attr(target: str, value: Any) -> None:
    """Replace ``target`` with ``value``."""
    owner, name = resolve(target)
    setattr(owner, name, value)


def wrap_attr(target: str, factory: Callable[[Any], Any]) -> None:
    """Replace ``target`` with ``factory(original)`` (a wrapper, not a from-scratch replacement)."""
    owner, name = resolve(target)
    original = getattr(owner, name)
    setattr(owner, name, factory(original))


@dataclass(frozen=True)
class Patch:
    """One recorded change to upstream behaviour."""

    target: str
    why: str
    apply: Callable[[], None] = field(compare=False, repr=False)
    since: str = "0.23"
    remove_when: str = ""
    upstream_ref: str = ""

    def as_dict(self) -> Dict[str, str]:
        return {
            "target": self.target,
            "why": self.why,
            "since": self.since,
            "remove_when": self.remove_when,
            "upstream_ref": self.upstream_ref,
        }


@dataclass
class InstallReport:
    """What happened to each patch in this process."""

    applied: List[str] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)
    failed: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed

    def summary(self) -> str:
        return (
            f"patches: {len(self.applied)} applied, {len(self.skipped)} skipped "
            f"(target gone), {len(self.failed)} failed"
        )


#: The registry.  Populated at import time by the modules that own each patch, so
#: ``tileinfer.vllm.plugin.install()`` only has to call :func:`install_all`.
PATCHES: List[Patch] = []


def register(patch: Patch) -> Patch:
    for existing in PATCHES:
        if existing.target == patch.target:
            raise ValueError(f"duplicate patch for {patch.target}")
    PATCHES.append(patch)
    return patch


def install_all(*, strict: bool = False) -> InstallReport:
    """Apply every registered patch, tolerating a missing target.

    ``strict=True`` is for tests and for ``tileinfer doctor``: it reports the skips as failures,
    because in a known-good environment a skipped patch means the vLLM version moved under us.
    """
    report = InstallReport()
    for patch in PATCHES:
        try:
            # Resolve first: a target that vanished upstream is a *skip*, which is what makes a
            # vLLM point release degrade instead of crash.  The patch body itself failing is a
            # genuine bug and is reported as such.
            resolve(patch.target)
        except (ImportError, AttributeError) as exc:
            report.skipped.append((patch.target, repr(exc)))
            logger.warning(
                "TileInfer patch skipped (%s): %s; the vLLM version may have moved", patch.target, exc
            )
            continue
        try:
            patch.apply()
        except (ImportError, AttributeError) as exc:
            report.skipped.append((patch.target, repr(exc)))
            logger.warning(
                "TileInfer patch skipped (%s): %s; the vLLM version may have moved", patch.target, exc
            )
        except Exception as exc:  # noqa: BLE001 - a patch must never stop the engine booting
            report.failed.append((patch.target, repr(exc)))
            logger.exception("TileInfer patch failed: %s", patch.target)
        else:
            report.applied.append(patch.target)
    if strict and report.skipped:
        report.failed.extend(report.skipped)
        report.skipped.clear()
    logger.info("TileInfer %s", report.summary())
    return report


def describe() -> List[Dict[str, str]]:
    """The registry as plain data (``tileinfer doctor`` and the probe JSON print this)."""
    return [patch.as_dict() for patch in PATCHES]


def clear() -> None:
    """Test helper: forget every registered patch."""
    PATCHES.clear()


# ---------------------------------------------------------------------------------------------
# Patches discovered on hardware (registered here so `install_all()` picks them up).
# ---------------------------------------------------------------------------------------------


def _probe_leaf_modules(runner: Any, phases: Any, make: Any) -> list:
    """Time every **leaf** module of the model, grouped by a layer-index-free name.

    Why leaves: §18 could attribute a 512-token prefill's 2.4 s only down to ``model_forward``
    (1862 ms) minus the instrumented attention/gdn/moe (~100 ms) — 1.75 s with no owner.  That
    remainder is dense projections, norms, rotary, the conv and the LM head, i.e. things this
    plugin does not implement but whose *number of launches* it can still count.

    Grouping by class path with the layer index stripped (``layers.self_attn.qkv_proj``) is what
    makes the table short: 40 layers report as one row each per module kind, and the average per
    call is the number the plan needs (a per-layer breakdown would be 240 rows of the same thing).
    Leaves only, on purpose: a timed parent includes its timed children, which is exactly the
    double-counting §8 of the plan already warns about.

    Off unless ``TILEINFER_PHASE_TIMING`` asks for device events, and the wrapper is a no-op
    object attribute otherwise — probes must never change what is being measured by much.
    """
    import re  # noqa: PLC0415

    model = getattr(runner, "model", None)
    if model is None or not hasattr(model, "named_modules"):
        return []
    budget = int(os.environ.get("TILEINFER_PHASE_LEAF_LIMIT", "600") or 600)
    wrapped: list = []
    seen = set()
    for name, module in model.named_modules():
        if not name or list(module.children()):
            continue
        if len(wrapped) >= budget:
            break
        forward = getattr(module, "forward", None)
        if forward is None or getattr(forward, "__tileinfer_phase__", False):
            continue
        label = "mod:" + re.sub(r"\.\d+\.", ".", name).removeprefix("model.")
        try:
            # Every *instance* is wrapped; the label is what aggregates them, so a row's average is
            # "one layer's worth of this module" and ``dev/step`` is the whole model's total for
            # that kind.  Wrapping only the first instance (an earlier revision did) reported one
            # layer as if it were the sum, which cannot be reconciled with ``model_forward``.
            setattr(module, "forward", make(forward, label))
        except Exception:  # noqa: BLE001 - an immutable module is not worth failing the run for
            continue
        seen.add(label)
        wrapped.append(label)
    return sorted(seen)


def _apply_phase_probes_on_load() -> None:
    """Wrap the model and sampler forwards with ``phases.timed`` once the weights are loaded.

    M0/W8 (``docs/performance.md``): the phase table showed ~59% of a decode step happening
    outside the ops TileInfer takes over, so that remainder has to be split too - and the only
    place where the model object exists and is stable is right after ``load_model``.  Nothing
    happens unless ``TILEINFER_PHASE_TIMING=device`` (or ``both``).
    """
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # noqa: PLC0415

    original = GPUModelRunner.load_model

    def load_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        try:
            from tileinfer.serving import phases  # noqa: PLC0415

            if not phases.device_enabled():
                return result
            installed = []
            for attribute, label in (("model", "model_forward"), ("sampler", "sampler")):
                obj = getattr(self, attribute, None)
                forward = getattr(obj, "forward", None) if obj is not None else None
                if forward is None or getattr(forward, "__tileinfer_phase__", False):
                    continue

                def make(orig: Any, phase_name: str) -> Any:
                    def wrapped(*a: Any, **k: Any) -> Any:
                        with phases.timed(phase_name):
                            return orig(*a, **k)

                    wrapped.__tileinfer_phase__ = True  # type: ignore[attr-defined]
                    return wrapped

                setattr(obj, "forward", make(forward, label))
                installed.append(label)
            installed.extend(_probe_leaf_modules(self, phases, make))
            if installed:
                logger.warning("TileInfer: phase probes installed on %s", ", ".join(installed))
        except Exception:  # noqa: BLE001 - probes must never break serving
            logger.warning("TileInfer: phase probes could not be installed", exc_info=True)
        return result

    load_model.__tileinfer_original__ = original  # type: ignore[attr-defined]
    GPUModelRunner.load_model = load_model


register(
    Patch(
        target="vllm.v1.worker.gpu_model_runner.GPUModelRunner.load_model",
        why="phase timing (M0/W8): the model and sampler objects only exist after load_model, and "
        "the un-instrumented 59% of a decode step lives inside them",
        apply=_apply_phase_probes_on_load,
        since="diagnostic only",
        remove_when="the phase probe moves into a first-class hook (or the optimisation lands)",
        upstream_ref="vllm/v1/worker/gpu_model_runner.py",
    )
)


def _apply_sampler_no_triton() -> None:
    """Force vLLM's top-k/top-p sampling onto its torch implementation.

    Observed on the first Ascend950PR run: ``topk_topp_sampler.apply_top_k_top_p`` takes the Triton
    branch whenever ``HAS_TRITON`` is true (``:341`` and ``:345`` with ``logits.shape[0] >= 8``),
    and Triton-on-Ascend's ``bishengir-compile`` aborts (MLIRCompilationError, then an LLVM stack
    dump) on that kernel - the engine died inside vLLM's own sampler warmup.  The module binds
    ``HAS_TRITON`` into its own namespace (``:13``) and re-reads it per call, so flipping the
    module attribute selects the equivalent torch path (same maths, slower) without touching the
    sampler's logic.
    """
    from vllm.v1.sample.ops import topk_topp_sampler  # noqa: PLC0415

    topk_topp_sampler.HAS_TRITON = False


register(
    Patch(
        target="vllm.v1.sample.ops.topk_topp_sampler.HAS_TRITON",
        why="the Triton top-k/top-p kernel does not compile on this stack (bishengir-compile "
        "aborts); the torch implementation is equivalent",
        apply=_apply_sampler_no_triton,
        since="vllm 0.23 + triton-ascend",
        remove_when="the Triton sampler kernel compiles on Ascend950",
        upstream_ref="vllm/v1/sample/ops/topk_topp_sampler.py:341,:345",
    )
)


def _apply_accelerator_empty_cache() -> None:
    """Route ``torch.accelerator.empty_cache()`` to ``torch.npu``.

    Observed on the first Ascend950PR run: vLLM's GDN warmup calls
    ``torch.accelerator.empty_cache()`` (``qwen_gdn_linear_attn.py:1195``), whose body is
    ``torch._C._accelerator_isAllocatorInitialized()``, and that assert fires with torch 2.10 +
    torch_npu 2.10 because the NPU allocator is not registered as a ``DeviceAllocator`` there:
    ``RuntimeError: Allocator for npu is not a DeviceAllocator``.  The call's *intent* is "release
    cached blocks", which ``torch.npu.empty_cache()`` does directly, so the shim keeps the
    behaviour instead of skipping the call.
    """
    import torch  # noqa: PLC0415

    accelerator = getattr(torch, "accelerator", None)
    if accelerator is None:  # pragma: no cover - torch < 2.5
        return
    original = accelerator.empty_cache

    def empty_cache(*args: Any, **kwargs: Any) -> Any:
        try:
            import torch_npu  # noqa: F401, PLC0415

            return torch.npu.empty_cache()
        except Exception:  # noqa: BLE001 - no NPU here: keep the stock behaviour
            return original(*args, **kwargs)

    empty_cache.__tileinfer_original__ = original  # type: ignore[attr-defined]
    accelerator.empty_cache = empty_cache


register(
    Patch(
        target="torch.accelerator.empty_cache",
        why="torch.accelerator asserts the NPU allocator is a DeviceAllocator; vLLM's GDN warmup "
        "calls it unconditionally and the engine dies before the first step",
        apply=_apply_accelerator_empty_cache,
        since="torch 2.10 + torch_npu 2.10",
        remove_when="torch_npu registers its allocator with torch.accelerator",
        upstream_ref="qwen_gdn_linear_attn.py:1195",
    )
)
