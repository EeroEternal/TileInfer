"""Wire the already-built torch operators into vLLM's own dispatch, as :class:`Patch` records.

What this module owns, and what moved out
-----------------------------------------
This module owns the **mechanism**: one :class:`OpBinding` per takeover, the plan against the live
module tree, the switch-aware wrapper, the ``Patch`` registration, and the report.  It no longer
owns the **table** of vLLM names: that is model-family data and lives in
``tileinfer/vllm/ops/families/`` (today :mod:`.families.gdn_moe`).  ``install_ops()`` resolves the
active :class:`~tileinfer.vllm.profiles.ModelProfile`'s ``op_family`` through
:func:`tileinfer.vllm.ops.families.family_for` and iterates that family's ``bind()`` records, so a
new model family is a new file, not an edit here.

The three takeover shapes, decided from the local vLLM source
--------------------------------------------------------------
1. **``CustomOp`` subclasses** are dispatched by ``dispatch_forward``
   (``custom_op.py:174-207``), whose out-of-tree branch calls ``forward_oot`` (:204-205).  The GDN
   chunk op re-pins ``self._forward_method`` in ``__init__`` (``qwen_gdn_linear_attn.py:290-311``)
   to ``forward_native`` on a non-CUDA platform, so the takeover replaces **both** methods - the
   pinned bound method and the ladder path.  It must therefore be installed before the layers are
   constructed; that is plugin step 4.
2. **Everything else is a module attribute.**  The GDN family imports its ops *by value* at module
   import (``qwen_gdn_linear_attn.py:24-49``), so both the definition module and the consumer
   namespace are rebound, as separate ``Patch`` records.  MoE attaches at vLLM's Python wrappers
   (``_custom_ops.py:2346-2363`` / ``:2494-2509`` and ``.../fused_moe/fused_moe.py:1474-1515``).
3. **Class-level ``register_oot``** (``kind="custom_op_oot"``) - for the one extension point vLLM
   *documents* rather than dispatches to: the ``CustomOp.register_oot`` subclass registration
   (``custom_op.py:322-353``, whose own example is ``@UnquantizedFusedMoEMethod.register_oot``).
   ``CustomOp.__new__`` (:109-128) substitutes the registered class at **instantiation** time, so
   the target is a class, the implementation is a *factory* ``impl(base_cls) -> subclass``, and the
   patch body calls ``base_cls.register_oot(subclass)``.  Two consequences, both intentional:

   * it is a **one-shot, process-wide registration**, not a per-call wrapper: there is no
     ``_family_on`` guard *inside* a call because there is nothing to fall back to (falling back is
     the stock crash - no kernel is ever built for OOT).  The switch is therefore consulted once, in
     ``apply()``: a family switched off at install time registers nothing, exactly like the other
     two kinds, and a family switched off later keeps the class it already registered (vLLM has no
     ``unregister_oot``);
   * the patched attribute is **not** replaced - the in-tree class is left byte-identical, only
     ``_tileinfer_oot_*`` markers are added - so a test can assert that the stock class still
     behaves as stock after the install.

Fallback semantics
------------------
Each takeover consults :func:`tileinfer.vllm.ops.switch.ops_enabled` **per call** and forwards to
the original vLLM object when its family is switched off.  A malformed ``TILEINFER_OPS`` never
raises from inside a forward: :func:`_family_on` logs it once and uses the switch defaults
(``gdn``/``moe`` on).

Candidates per operator
-----------------------
A family may attach a :class:`~tileinfer.vllm.ops.candidates.CandidateSet` to a binding, in which
case ``impl`` stays the **baseline** and what gets bound is the set's thin dispatcher.  That is the
slot for a second implementation (a TileLang kernel later) selected per call by a shape/dtype
predicate and gated by a recorded measurement; ``TILEINFER_OPS_CANDIDATES=off`` disarms every
non-baseline candidate, and with nothing eligible the call is the baseline's, byte for byte.  A
family that registers no set is wired exactly as before - this module does not require the seam.

Import policy: no ``vllm`` and no ``torch_npu`` at module level; the family table imports the torch
implementations lazily, so ``tileinfer doctor`` and the launcher can import this module on a CPU
box without an engine.
"""

from __future__ import annotations

import functools
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from tileinfer.serving import phases as _phases

#: Read once: the instrumentation is a one-line `if` per op call when this is False.
_PHASE_TIMING = _phases.enabled()

#: When set, each takeover is timed under its own phase name (``op:gdn:<function>``) instead of the
#: family-wide ``op:gdn``.  Off by default because the family row is what existing tables and
#: dashboards read; this is the switch for "which of the family's operators is the 36 ms?" -- the
#: question §22 could not answer for GDN.
_PHASE_OPS_DETAIL = os.environ.get("TILEINFER_PHASE_OPS", "").strip().lower() in (
    "detail",
    "1",
    "true",
    "yes",
)

#: When set, every takeover counts its calls and prints a per-target table every
#: :data:`_OP_COUNT_EVERY` calls.  This is the cheap instrument for §62's open question -- the
#: remaining prefill cost looks like per-layer dispatch, so the first number to get is "how many
#: takeovers run per layer", which no timing table answers reliably (the phase table's columns
#: include one-off compile time, §62).
_OP_COUNT = os.environ.get("TILEINFER_OP_COUNT", "").strip().lower() not in ("", "0", "false", "no")
_OP_COUNT_EVERY = int(os.environ.get("TILEINFER_OP_COUNT_EVERY", "2000") or 2000)
_OP_CALLS: dict = {}

from tileinfer.vllm.compat import patches
from tileinfer.vllm.ops.families import family_for
from tileinfer.vllm.ops.switch import ENV_OPS, OPS_ALL, OPS_DEFAULT_ON, ops_enabled

if TYPE_CHECKING:  # a type only: importing it at runtime would be circular-free but pointless
    from tileinfer.vllm.ops.candidates import CandidateSet

__all__ = [
    "install_ops",
    "OpsReport",
    "OpBinding",
    "bindings",
    "required_names",
    "candidate_report",
]

logger = logging.getLogger("tileinfer.vllm.ops.install")

#: The vLLM series these targets were transcribed from (``plugin.SUPPORTED_VLLM_RANGE``).
SINCE = "0.23"


# =============================================================================================
# The data model: one binding, one report
# =============================================================================================


@dataclass(frozen=True)
class OpBinding:
    """One operator's takeover plan.

    ``targets`` are *alternative* names for the same entry point (a vLLM rename shows up as "the
    first name stopped resolving"); ``aliases`` are *additional* names that must be rebound for the
    takeover to be effective (the consumer-module import described above).
    """

    op: str
    family: str
    targets: Tuple[str, ...]
    why: str
    remove_when: str
    upstream_ref: str = ""
    aliases: Tuple[str, ...] = ()
    impl: Optional[Callable[..., Any]] = field(default=None, repr=False)
    #: ``"attr"`` (rebind a module attribute), ``"custom_op"`` (inject ``forward_native``/
    #: ``forward_oot`` on an in-tree ``CustomOp`` subclass) or ``"custom_op_oot"`` (register a
    #: subclass through ``CustomOp.register_oot``; ``impl`` is then a factory
    #: ``impl(base_cls) -> subclass``, see :func:`_custom_op_oot_takeover`).
    kind: str = "attr"
    abi_ok: bool = True
    abi_note: str = ""
    #: The per-op candidate seam (``tileinfer.vllm.ops.candidates``) when the family registers one:
    #: ``impl`` stays the **baseline** implementation, so every report, doctor line and test that
    #: names the implementation keeps naming it; what gets bound is ``candidates.dispatched()``.
    candidates: Optional["CandidateSet"] = field(default=None, repr=False, compare=False)


@dataclass
class OpsReport:
    """What happened to each operator in this process - the plain-data form of the install."""

    bound: List[str] = field(default_factory=list)
    unbound: List[str] = field(default_factory=list)
    disabled: List[str] = field(default_factory=list)
    targets: Dict[str, str] = field(default_factory=dict)
    details: Dict[str, str] = field(default_factory=dict)
    unbound_aliases: List[str] = field(default_factory=list)
    families: Dict[str, bool] = field(default_factory=dict)
    patches: Optional[patches.InstallReport] = None

    @property
    def ok(self) -> bool:
        return not self.unbound

    def summary(self) -> str:
        families = " ".join(
            f"{name}={'on' if on else 'off'}" for name, on in sorted(self.families.items())
        )
        return (
            f"ops: {len(self.bound)} bound, {len(self.unbound)} unbound "
            f"(no vLLM name / no implementation), {len(self.disabled)} off by switch"
            + (f" | {families}" if families else "")
        )


# =============================================================================================
# The switch, tolerant at call time
# =============================================================================================

_WARNED: set = set()


def _family_on(family: str) -> bool:
    """Per-call family switch; never raises."""
    try:
        return ops_enabled(family)
    except ValueError as exc:  # a malformed TILEINFER_OPS: log once, use the defaults
        if family not in _WARNED:
            _WARNED.add(family)
            logger.warning(
                "TileInfer op family %r: %s; keeping the default (%s). Fix %s and restart.",
                family,
                exc,
                "on" if family in OPS_DEFAULT_ON else "off",
                ENV_OPS,
            )
        return family in OPS_DEFAULT_ON


# =============================================================================================
# The three takeover shapes
# =============================================================================================


def _attribute_takeover(
    target: str, impl: Callable[..., Any], family: str
) -> Callable[[], None]:
    """A patch body that rebinds ``target`` to a switch-aware wrapper around ``impl``.

    Idempotent: a target that already carries our marker is left alone, so re-running the install
    (plugin reload, ``_reset()`` in tests) cannot stack wrappers.
    """

    def apply() -> None:
        owner, name = patches.resolve(target)
        original = getattr(owner, name)
        if getattr(original, "__tileinfer_takeover__", False):
            return
        setattr(owner, name, _wrap(impl, original, family, target))

    return apply


def _count_call(family: str, target: str) -> None:
    """Count one takeover call; every ``_OP_COUNT_EVERY`` calls print the table and reset it.

    Deliberately dumb (dict + integer, no locks beyond the GIL): it must be safe to leave compiled in
    and it must not perturb whatever it is measuring.  The window between prints is what makes it
    readable as "calls per N steps" without needing to know the step count.
    """
    key = f"{family}:{target.rsplit('.', 1)[-1]}"
    entry = _OP_CALLS.get(key)
    _OP_CALLS[key] = 1 if entry is None else entry + 1
    total = sum(_OP_CALLS.values())
    if total < _OP_COUNT_EVERY:
        return
    top = sorted(_OP_CALLS.items(), key=lambda kv: -kv[1])[:12]
    print(
        f"[tileinfer/op-count] {total} calls: "
        + ", ".join(f"{name}={count}" for name, count in top),
        flush=True,
    )
    _OP_CALLS.clear()


def _wrap(
    impl: Callable[..., Any], original: Callable[..., Any], family: str, target: str
) -> Callable[..., Any]:
    """The switch-aware replacement itself (kept out of ``apply`` so it can be introspected)."""

    def takeover(*args: Any, **kwargs: Any) -> Any:
        # Call counting: cheap (one dict increment) and answers "how many of *our* ops run per
        # layer?" -- the question §62 left open, since the per-layer dispatch cost is what remains
        # of the prefill budget (~297 ms over ~1200 ops).  Off unless TILEINFER_OP_COUNT is set.
        if _OP_COUNT:
            _count_call(family, target)
        # Phase timing (docs/plan-performance.md M0/W0b).  Off unless TILEINFER_PHASE_TIMING is set;
        # `timed` records the host clock and, in device mode, an NPU event pair for the same block.
        if _PHASE_TIMING:
            label = (
                f"op:{family}:{target.rsplit('.', 1)[-1]}" if _PHASE_OPS_DETAIL else f"op:{family}"
            )
            with _phases.timed(label):
                if _family_on(family):
                    return impl(*args, **kwargs)
                return original(*args, **kwargs)
        if _family_on(family):
            return impl(*args, **kwargs)
        return original(*args, **kwargs)

    if hasattr(original, "__name__"):
        try:
            takeover = functools.wraps(original)(takeover)
        except (AttributeError, TypeError):  # a builtin/C object: introspection is optional
            pass
    takeover.__tileinfer_takeover__ = True  # type: ignore[attr-defined]
    takeover.__tileinfer_original__ = original  # type: ignore[attr-defined]
    takeover.__tileinfer_impl__ = _baseline_of(impl)  # type: ignore[attr-defined]
    # Present only when the family registered a candidate set; ``None`` means one implementation.
    takeover.__tileinfer_candidates__ = getattr(impl, "__tileinfer_candidates__", None)
    takeover.__tileinfer_target__ = target  # type: ignore[attr-defined]
    return takeover


def _baseline_of(impl: Callable[..., Any]) -> Callable[..., Any]:
    """The implementation behind a candidate dispatcher.

    ``__tileinfer_impl__`` has always named the torch implementation (``tests/test_ops_install.py``
    and ``tests/test_boot_rehearsal.py`` pin that identity), so the candidate seam must not change
    what introspection reports: the dispatcher says which impl it falls back to, and this unwraps
    it.
    """
    return getattr(impl, "__tileinfer_baseline__", impl)


def _custom_op_takeover(
    target: str, impl: Callable[..., Any], family: str
) -> Callable[[], None]:
    """A patch body that injects ``forward_native``/``forward_oot`` onto a ``CustomOp`` subclass.

    The signature is upstream's, verbatim (``qwen_gdn_linear_attn.py:345-359``): ``scale`` is not a
    parameter there, and ``chunk_indices``/``chunk_offsets`` are launcher metadata the TileInfer
    chain re-derives from ``cu_seqlens``, so they are accepted and ignored.
    """

    def apply() -> None:
        owner, name = patches.resolve(target)
        cls = getattr(owner, name)
        if getattr(cls, "_tileinfer_custom_op_takeover", False):
            return
        original = getattr(cls, "forward_native", None)

        def forward_native(
            self: Any,
            q: Any,
            k: Any,
            v: Any,
            g: Any,
            beta: Any,
            initial_state: Any,
            output_final_state: bool,
            cu_seqlens: Any = None,
            chunk_indices: Any = None,
            chunk_offsets: Any = None,
            use_qk_l2norm_in_kernel: bool = True,
            core_attn_out: Any = None,
        ) -> Any:
            if not _family_on(family):
                if original is None:  # pragma: no cover - only if upstream deleted the method
                    raise RuntimeError(
                        "TileInfer: the GDN chunk takeover is switched off "
                        f"({ENV_OPS} does not list {family!r}) but {target} has no "
                        "forward_native to fall back to"
                    )
                return original(
                    self,
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    initial_state=initial_state,
                    output_final_state=output_final_state,
                    cu_seqlens=cu_seqlens,
                    chunk_indices=chunk_indices,
                    chunk_offsets=chunk_offsets,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    core_attn_out=core_attn_out,
                )
            o, final_state = impl(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            if core_attn_out is not None:
                # Same write upstream's forward_cuda does (qwen_gdn_linear_attn.py:339-343); the
                # production prefill call site does not pass the buffer (:1518-1530).
                flat = o.reshape(-1)
                core_attn_out.reshape(-1)[: flat.numel()].copy_(flat)
            return o, final_state

        forward_native.__tileinfer_takeover__ = True  # type: ignore[attr-defined]
        forward_native.__tileinfer_impl__ = _baseline_of(impl)  # type: ignore[attr-defined]
        forward_native.__tileinfer_candidates__ = getattr(  # type: ignore[attr-defined]
            impl, "__tileinfer_candidates__", None
        )
        forward_native.__tileinfer_target__ = target  # type: ignore[attr-defined]
        setattr(cls, "forward_native", forward_native)
        # The ladder path (custom_op.py:204-205), kept identical so a base-ladder dispatch and the
        # re-pinned `self._forward_method` cannot disagree.
        setattr(cls, "forward_oot", forward_native)
        cls._tileinfer_custom_op_takeover = True

    return apply


def _custom_op_oot_takeover(
    target: str, impl: Callable[[type], type], family: str
) -> Callable[[], None]:
    """A patch body that registers an OOT subclass through ``CustomOp.register_oot``.

    The mechanism, not an invention: vLLM's own ``CustomOp.__new__`` (``custom_op.py:109-128``)
    substitutes the class found in ``op_registry_oot[cls.__name__]`` for **every** instantiation of
    the in-tree class, and ``CustomOp.register_oot`` (``:331-353``) is the documented way to put it
    there (its example is literally ``@UnquantizedFusedMoEMethod.register_oot``, ``:326-329``).
    ``layer.py:359`` constructs ``UnquantizedFusedMoEMethod(self.moe_config)``, so registering here
    changes what the model loader builds without touching a single call site.

    ``impl`` is a **factory**, not a callable: it receives the resolved in-tree class and returns
    the subclass (``tileinfer.vllm.ops.fused_moe_oot.make_oot_method``).  That is what lets the
    implementation module stay importable on a box without vLLM - the base class is data, not an
    import.

    Two differences from the per-call kinds, both documented in the module docstring: the family
    switch is read **once, here** (a class registration cannot be undone, so it is also the only
    place it *can* be read), and the in-tree class is not modified - only ``_tileinfer_oot_*``
    markers are added, so "was this installed?" stays answerable and the stock class still behaves
    as stock.
    """

    def apply() -> None:
        owner, name = patches.resolve(target)
        base = getattr(owner, name)
        if getattr(base, "_tileinfer_oot_takeover", False):
            return  # already registered in this process (plugin reload, `_reset()` in tests)
        if not _family_on(family):
            # Same contract as the other two kinds: a family that is off leaves upstream untouched.
            return
        register_oot = getattr(base, "register_oot", None)
        if register_oot is None:
            raise AttributeError(
                f"TileInfer: {target} has no register_oot; the class-level OOT extension point "
                "(custom_op.py:331-353) moved, so this takeover cannot be expressed by rebinding "
                "an attribute. Falling back to vLLM's own class."
            )
        subclass = impl(base)
        register_oot(subclass)  # custom_op.py:331-353 (asserts a duplicate registration)
        base._tileinfer_oot_takeover = True
        base._tileinfer_oot_cls = subclass
        base._tileinfer_oot_family = family
        logger.info(
            "TileInfer: registered %s as the out-of-tree %s (custom_op.py:331-353)",
            getattr(subclass, "__name__", subclass),
            name,
        )

    return apply


# =============================================================================================
# The binding table: resolved from the active model profile's op family
# =============================================================================================

#: ``family name -> its bindings``; one entry per family the active profile uses.
_BINDINGS: Dict[str, List[OpBinding]] = {}


def _profile_family_names(profile: Any = None) -> Tuple[str, ...]:
    """The operator families the active (or given) model profile asks for.

    ``ModelProfile.op_family`` is a dash-separated name (``gdn-moe``); a comma/plus spelling is
    accepted so a future profile can compose more than one family without a new registry concept.
    The profile import is local: it pulls ``tileinfer.serving`` (torch), and this module must stay
    importable before an engine exists.
    """
    if profile is None:
        from tileinfer.vllm.profiles import active_profile

        profile = active_profile()
    raw = str(getattr(profile, "op_family", "") or "").strip()
    names = tuple(part.strip() for part in raw.replace("+", ",").split(",") if part.strip())
    if not names:
        raise ValueError(
            f"model profile {getattr(profile, 'name', profile)!r} declares no op_family; the "
            "operator install cannot decide which takeovers to apply"
        )
    return names


def _families(profile: Any = None) -> List[Tuple[str, Any, List[OpBinding]]]:
    """``[(family name, family module, bindings)]`` for the active profile (cached per family).

    The family module owns the table (``required_names()`` + ``bind()``); this function only
    resolves it and checks that the table still contains every name the family promises.
    """
    resolved: List[Tuple[str, Any, List[OpBinding]]] = []
    for name in _profile_family_names(profile):
        module = family_for(name)
        cached = _BINDINGS.get(name)
        if cached is None:
            cached = list(module.bind())
            declared = {binding.op for binding in cached}
            missing = set(module.required_names()) - declared
            if missing:
                logger.warning(
                    "TileInfer op family %r promises %s but its bind() table does not contain "
                    "them; the family module and its required_names() disagree",
                    name,
                    sorted(missing),
                )
            _BINDINGS[name] = cached
        resolved.append((name, module, cached))
    return resolved


def bindings() -> List[OpBinding]:
    """The takeover table for the active model profile (built once per family, impls lazy)."""
    return [binding for _name, _module, entries in _families() for binding in entries]


def required_names() -> Tuple[str, ...]:
    """Every op name the active profile's families promise, in family/table order."""
    names: Tuple[str, ...] = ()
    for _name, module, _entries in _families():
        names += tuple(module.required_names())
    return names


def candidate_report() -> Dict[str, Dict[str, Any]]:
    """``{op: CandidateSet.describe()}`` for the active profile - what the doctor/probe prints.

    Empty when a family registers no candidate sets (the seam is optional).  Never raises for a
    missing family: it goes through the same lazy resolution as :func:`bindings`, which is the
    contract every probe here has.
    """
    report: Dict[str, Dict[str, Any]] = {}
    for binding in bindings():
        if binding.candidates is not None:
            report[binding.op] = binding.candidates.describe()
    return report


# =============================================================================================
# install_ops()
# =============================================================================================

_LOCK = threading.RLock()

_STATE: Dict[str, Any] = {
    "installed": False,
    "report": None,
    "registered": [],  # patch targets this module registered (so _reset() can unregister them)
}


def _resolvable(target: str) -> Tuple[bool, str]:
    """``(resolves, reason)`` - a probe, so planning never raises on a missing name."""
    try:
        patches.resolve(target)
    except Exception as exc:  # noqa: BLE001 - any import-time failure means "cannot bind"
        return False, repr(exc)
    return True, ""


def _installed_impl(binding: OpBinding) -> Optional[Callable[..., Any]]:
    """The callable a patch installs for ``binding``.

    With a candidate set the family's ``impl`` is the *baseline*; what gets bound is the set's thin
    dispatcher, so the ABI stays the implementation's own while the set may still gain
    implementations (``tileinfer.vllm.ops.candidates``).  Without a set this is the impl itself -
    the pre-candidate wiring, unchanged.  A ``custom_op_oot`` binding always lands here unchanged:
    its ``impl`` is a class factory (``impl(base_cls) -> subclass``), not a per-call function, so
    there is no call shape for a candidate predicate to dispatch on.
    """
    if binding.candidates is None:
        return binding.impl
    return binding.candidates.dispatched()


def _register(binding: OpBinding, target: str, why: str, registered: List[str]) -> None:
    """Register one ``Patch`` for ``target`` unless an identical one is already registered."""
    for existing in patches.PATCHES:
        if existing.target == target:
            if target not in registered:
                registered.append(target)
            return
    impl = _installed_impl(binding)
    if binding.kind == "custom_op":
        apply = _custom_op_takeover(target, impl, binding.family)
    elif binding.kind == "custom_op_oot":
        apply = _custom_op_oot_takeover(target, impl, binding.family)
    else:
        apply = _attribute_takeover(target, impl, binding.family)
    patch = patches.Patch(
        target=target,
        why=why,
        apply=apply,
        since=SINCE,
        remove_when=binding.remove_when,
        upstream_ref=binding.upstream_ref,
    )
    try:
        patches.register(patch)
    except Exception as exc:  # noqa: BLE001 - a duplicate is a planning bug, not a boot failure
        logger.warning(
            "TileInfer op %s: could not register a patch for %s (%r); continuing",
            binding.op,
            target,
            exc,
        )
        return
    registered.append(target)


def _plan(binding: OpBinding, report: OpsReport, registered: List[str]) -> str:
    """Decide one binding's fate and register its patches; return its status string."""
    if not report.families.get(binding.family, False):
        report.details[binding.op] = (
            f"switched off: {ENV_OPS} does not enable the {binding.family!r} family"
        )
        return "disabled"
    if not binding.abi_ok:
        report.details[binding.op] = (
            f"unbound: {binding.abi_note} (vLLM name: {binding.targets[0]})"
        )
        return "unbound"
    if binding.impl is None:
        report.details[binding.op] = (
            "unbound: no torch implementation in tileinfer.vllm.ops; the layer calls "
            f"{binding.targets[0] if binding.targets else '(unknown name)'} and there is no "
            "fallback for it on Ascend"
        )
        return "unbound"

    primary: Optional[str] = None
    tried: List[str] = []
    for target in binding.targets:
        ok, reason = _resolvable(target)
        if ok:
            primary = target
            break
        tried.append(f"{target} ({reason})")
    if primary is None:
        report.details[binding.op] = "unbound: no vLLM target resolves; tried " + "; ".join(tried)
        return "unbound"

    _register(binding, primary, binding.why, registered)
    bound_aliases: List[str] = []
    for alias in binding.aliases:
        ok, reason = _resolvable(alias)
        if not ok:
            report.unbound_aliases.append(alias)
            logger.warning(
                "TileInfer op %s: bound %s, but the consumer name %s does not resolve (%s); on "
                "that path vLLM's own implementation would still be called",
                binding.op,
                primary,
                alias,
                reason,
            )
            continue
        _register(
            binding,
            alias,
            f"{binding.op} is imported by value into this namespace (chunk.py:17, "
            "qwen_gdn_linear_attn.py:24-49), so rebinding the definition module alone is "
            f"import-order dependent; {binding.why}",
            registered,
        )
        bound_aliases.append(alias)
    report.targets[binding.op] = primary
    report.details[binding.op] = (
        f"bound at {primary}"
        + (f" (+{len(bound_aliases)} consumer alias)" if bound_aliases else "")
    )
    return "planned"


def install_ops() -> OpsReport:
    """Wire every family the active model profile names into vLLM's dispatch; never raises.

    Idempotent: the second call in a process returns the first call's report unchanged.  The
    install is *planned* against the live module tree before anything is applied, so a vLLM point
    release that moved a symbol shows up as an ``unbound`` line naming the exact target.
    """
    with _LOCK:
        cached = _STATE.get("report")
        if _STATE.get("installed") and isinstance(cached, OpsReport):
            logger.debug("TileInfer ops: already installed in this process; no-op")
            return cached

        report = OpsReport()
        registered: List[str] = []
        try:
            report.families = {family: _family_on(family) for family in OPS_ALL}
            try:
                families = _families()
            except Exception as exc:  # noqa: BLE001 - an unknown family must not stop the boot
                logger.warning(
                    "TileInfer ops: cannot resolve the model profile's operator family (%r); "
                    "no takeover is planned. Check ModelProfile.op_family and "
                    "tileinfer/vllm/ops/families/.",
                    exc,
                )
                families = []

            status: Dict[str, str] = {}
            for _name, _module, entries in families:
                for binding in entries:
                    try:
                        status[binding.op] = _plan(binding, report, registered)
                    except Exception as exc:  # noqa: BLE001 - one bad entry must not stop the rest
                        report.details[binding.op] = f"unbound: planning raised {exc!r}"
                        status[binding.op] = "unbound"

            try:
                report.patches = patches.install_all()
            except Exception as exc:  # noqa: BLE001 - install_all is already tolerant; belt+braces
                logger.exception("TileInfer ops: patches.install_all() raised (%r)", exc)
            applied = set(report.patches.applied) if report.patches is not None else set()

            for _name, _module, entries in families:
                for binding in entries:
                    outcome = status.get(binding.op, "unbound")
                    if outcome == "disabled":
                        report.disabled.append(binding.op)
                    elif outcome == "planned" and report.targets.get(binding.op) in applied:
                        report.bound.append(binding.op)
                    else:
                        if outcome == "planned":
                            report.details[binding.op] = (
                                f"unbound: the patch for {report.targets.get(binding.op)} was "
                                "registered but not applied (target vanished between planning "
                                "and apply)"
                            )
                        report.unbound.append(binding.op)

            for op in report.unbound:
                logger.warning("TileInfer op unbound: %s - %s", op, report.details.get(op, ""))
            for op in report.disabled:
                logger.info("TileInfer op disabled: %s - %s", op, report.details.get(op, ""))
            logger.info(
                "TileInfer %s | registered %d patch(es) | %s",
                report.summary(),
                len(registered),
                "bound: " + ", ".join(report.bound) if report.bound else "nothing bound",
            )
        except Exception:  # noqa: BLE001 - the plugin must still boot with ops unwired
            logger.exception(
                "TileInfer ops: install_ops() failed; continuing with vLLM's own operators "
                "(the model will not run on Ascend, and `tileinfer doctor` shows this report)"
            )
        _STATE.update(installed=True, report=report, registered=registered)
        return report


def _reset() -> None:
    """Test helper: forget that :func:`install_ops` ran and unregister its patches.

    Already-applied takeovers stay applied (the registry has no undo hook, by design - see
    ``compat/patches.py``); the wrappers are idempotent, so a fresh install leaves them alone.  The
    per-family binding cache is cleared too, so a test can activate another profile and observe a
    different family table.
    """
    with _LOCK:
        for target in list(_STATE.get("registered") or []):
            for patch in list(patches.PATCHES):
                if patch.target == target:
                    patches.PATCHES.remove(patch)
        _BINDINGS.clear()
        _STATE.update(installed=False, report=None, registered=[])
        _WARNED.clear()
