"""``tileinfer.vllm.plugin`` — the ``vllm.general_plugins`` entry point: one ``install()`` per process.

Where this sits
---------------
``pyproject.toml`` declares exactly two entry points into this package, and they are different
mechanisms with different call sites::

    [project.entry-points."vllm.platform_plugins"]
    tileinfer = "tileinfer.vllm.platform:register"      # -> ``current_platform`` (one class path)
    [project.entry-points."vllm.general_plugins"]
    tileinfer = "tileinfer.vllm.plugin:install"         # -> this module (runs in EVERY process)

``platform:register`` answers "which platform is this box?" and is therefore called while vLLM is
still choosing a platform.  This module is the *other* half: it runs after a platform exists, in
every process of the server (API server, EngineCore, each worker), and it owns everything that is
not expressible as a ``Platform`` override — the patch registry, the attention-backend registry
slot and the graph bridge.

Install sequence (and why this order)
-------------------------------------
1. **Patch registry** (:func:`tileinfer.vllm.compat.patches.install_all`).  First, because every
   later step may rely on a patched assumption, and because a skipped patch has to be visible in
   the log *before* the engine starts doing work.  ``install_all`` is failure-tolerant by design:
   a target that no longer exists upstream is reported as a skip, not raised.
2. **Attention-backend registration** — see the decision below.
3. **Graph bridge** — only when :func:`tileinfer.vllm.graphs.capture_path_ready` says the capture
   path is (or can be made) usable; see that decision below too.
4. **Operator wiring** (:func:`tileinfer.vllm.ops.install.install_ops`) — last, because it attaches
   to vLLM's *model* modules (``CustomOp`` subclasses and the module globals of the GDN layer),
   which only exist once the platform has been chosen and the model modules are importable.  It is
   the step that makes the model runnable at all on Ascend: without the GDN and MoE takeovers there
   is no non-CUDA implementation of those operators (``docs/plan-vllm-native-backend.md`` §4).
   ``TILEINFER_OPS`` selects the families (default: ``gdn,moe``; ``dense`` off) — see
   ``tileinfer/vllm/ops/switch.py``.
4. **Model registration** (:func:`_install_models`) — the architectures this package supplies and
   vLLM 0.23.0 does not ship (``DeepseekV41ForCausalLM``).  Registered *by class path*, so the
   96k-tensor parameter tree is imported only when such a checkpoint is actually resolved.

Then one INFO banner with all five outcomes, so ``grep "TileInfer vLLM plugin" serve.log`` gives
the whole install state of that process.

Decision 1: is the attention backend still registered in vLLM's registry?  **Yes, under
``AttentionBackendEnum.CUSTOM``** — and it is *not* what makes the backend get used.
The platform hook returns the class path directly
--------------------------------------------------
``Platform.get_attn_backend_cls`` returns a **class path string**
(``vllm/platforms/interface.py:259-267``) and the caller resolves it with
``resolve_obj_by_qualname`` — it never goes through ``AttentionBackendEnum.get_path()``.  So for
the normal path (platform chooses, engine instantiates) a registry entry would be dead weight.
What it *does* buy is the by-name path: ``registry.py:99-101`` reserves ``CUSTOM`` for "<
third-party backends - must be registered before use", and ``get_path`` raises until then
(``registry.py:103-120``).  ``--attention-backend custom`` is exactly the spelling
``TileInferPlatform.get_attn_backend_cls`` accepts (see ``_reject_foreign_backend_selection``), and
an operator who passes it must not be told "Backend CUSTOM must be registered before use" by
upstream.  So we register:

* it is one ``dict`` write (``registry.py:259-264``), idempotent by construction;
* it does not affect the platform path at all (that path is a plain string);
* it makes ``AttentionBackendEnum.CUSTOM.get_class()`` a working probe, which is how the
  registration is *verified* here (identity check) instead of assumed;
* ``TILEINFER_VLLM_REGISTER_BACKEND=0`` opts out - the only reason to need that is another
  third-party backend competing for the same single ``CUSTOM`` slot, which is a real (if unlikely)
  collision because ``CUSTOM`` is a process-wide singleton.

Mamba/GDN layers are **not** registered: 30 of the 40 target-model layers run upstream's
``GDNAttentionBackend``; v0 has no mamba backend of its own (``docs/plan-vllm-native-backend.md``
§5), and registering a second ``CUSTOM`` slot for mamba would claim work that does not exist.

Decision 2: the graph bridge is conditional, and it proves itself
-----------------------------------------------------------------
``tileinfer/vllm/graphs.py`` establishes the verdict: vLLM's ``CUDAGraphWrapper`` is "patchable"
(branch 2) — two CUDA call sites (``cuda_graph.py:283`` ``torch.cuda.CUDAGraph``,
``:313`` ``torch.cuda.graph``) that a bridge can alias onto ``torch.npu``'s primitives, plus the
model runner's memory/stream calls (``mem_get_info``, ``Stream``, …), plus ``graph_pool_handle``
reached through ``Platform.__getattr__``.  That module **owns the alias table**
(:data:`tileinfer.vllm.graphs.TORCH_CUDA_ALIASES`: four *required* aliases, five best-effort) and
the all-or-nothing write/rollback in :func:`tileinfer.vllm.graphs.install`; this module only
*calls* it.  There is deliberately **no second alias table here**: two tables is exactly how the
"bridge installed, but ``capture_path_ready()`` is still False" bug appears — the local pair
(``CUDAGraph``/``graph``) gets aliased while the oracle, which verifies the *required four* by
identity, keeps answering ``False`` and the platform keeps forcing eager.  So
:func:`graphs.capture_path_ready` answers ``(ready, reason)`` and is only ``True`` when the
primitives are *identical objects*, never when they merely "look similar": this module asks it,
delegates only if it says not-ready, and then asks it **again**.  If the second answer is still
``False``, nothing is enabled, the reason is logged, and
``TileInferPlatform.check_and_update_config`` keeps forcing ``enforce_eager=True``
(``platform.py:_force_eager_until_capture_lands``) — capture never silently runs half-bridged.

The bridge is *installed* from here (step 3) rather than from ``compat/patches.py`` on purpose: the
registry patches *vLLM* symbols and applies unconditionally, while this one patches ``torch.cuda``
(aliasing it away on a box that has no CUDA device) and must be gated on a device probe.
``uninstall()`` delegates to :func:`tileinfer.vllm.graphs.uninstall`; registry patches have no undo
hook at all.

[unverified] Whether vLLM loads general plugins *before* the frontend process resolves
``current_platform`` and runs ``check_and_update_config`` is not recorded in
``reference/vllm-0.23/`` (the loader, ``vllm/plugins/__init__.py``, is not in the local subset; the
legacy adapter's note in the frozen snapshot under ``legacy/`` says general plugins load in
``EngineCore.__init__``).  Consequence, deliberately benign: if the bridge lands *after* config
validation, the platform still asked for eager mode — correct and slow, not wrong-and-fast.  The
first run on hardware therefore has to check that the *banner* line says the bridge is ready before
trusting any ``capture_calls > 0`` measurement (``docs/hardware-checklist.md`` step 10).

Failure policy: never raise out of ``install()``
------------------------------------------------
``vllm.general_plugins`` is loaded in every process of a server.  A plugin that raises there turns
a working engine into a process that dies before it serves, with a traceback from an optional
component.  So every step is individually guarded (and ``install_all`` is already tolerant); the
whole body has a last-resort ``except Exception`` that logs and returns ``False``.  A *missing*
optional piece (no torch_npu, no vLLM attention registry, a vLLM point release that moved a
symbol) is a logged skip, not a failure.

Idempotence and the CPU box
---------------------------
``install()`` is guarded by a module flag plus a lock: entry points can be loaded more than once
(two ``tileinfer`` distributions on ``sys.path``, a re-entrant import) and the second call must not
apply patches twice or re-alias ``torch.cuda``.  Outside a vLLM environment ``install()`` returns
``False`` and imports nothing vLLM-side, so ``import tileinfer.vllm.plugin`` (and ``import
tileinfer``) stays clean on a CPU box — which is what lets the doctor/status path be exercised in
CI.  Nothing below imports ``vllm`` or ``torch_npu`` at module level.

Environment knobs (all optional)
--------------------------------
======================================  =======  ===============================================
variable                                default  meaning
======================================  =======  ===============================================
``TILEINFER_VLLM_STRICT``               ``0``    ``patches.install_all(strict=True)``: a skipped
                                                 patch counts as a failure. For
                                                 ``tileinfer doctor`` / CI, never for serving.
``TILEINFER_VLLM_REGISTER_BACKEND``     ``1``    register ``AttentionBackendEnum.CUSTOM``.
``TILEINFER_VLLM_GRAPH_BRIDGE``         ``1``    delegate the ``torch.cuda`` -> ``torch.npu``
                                                 alias table to :func:`graphs.install` before
                                                 capture.
``TILEINFER_OPS``                       ``gdn,moe``  operator families ``install_ops()``
                                                 wires (``gdn``, ``moe``, ``dense``; unset
                                                 means gdn+moe, see ``ops/switch.py``).
======================================  =======  ===============================================
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

from tileinfer.vllm import graphs
from tileinfer.vllm.compat import patches

__all__ = [
    "install",
    "uninstall",
    "status",
    "PLUGIN_ENTRY_POINT",
    "PLATFORM_ENTRY_POINT",
    "PLATFORM_CLASS_PATH",
    "BACKEND_ENUM_SLOT",
    "SUPPORTED_VLLM_RANGE",
    "ENTRY_POINTS",
    "ENV_STRICT",
    "ENV_REGISTER_BACKEND",
    "ENV_GRAPH_BRIDGE",
    "ENV_OPS",
]

logger = logging.getLogger("tileinfer.vllm.plugin")

#: This module's own entry point, as ``pyproject.toml`` spells it (kept as data so ``status()``
#: can show what an operator's installed metadata *should* contain).
PLUGIN_ENTRY_POINT = "tileinfer.vllm.plugin:install"

#: The platform half, declared by the same ``pyproject.toml`` block (see ``platform.register``).
PLATFORM_ENTRY_POINT = "tileinfer.vllm.platform:register"
PLATFORM_CLASS_PATH = "tileinfer.vllm.platform.TileInferPlatform"

#: The registry slot the backend is registered under (``registry.py:99-101``).
BACKEND_ENUM_SLOT = "AttentionBackendEnum.CUSTOM"

#: The vLLM range this package is written against.  A version outside it only produces a warning:
#: the real gate is the contract suite, and failing the boot on a string comparison would be worse
#: than serving with a skipped patch.
SUPPORTED_VLLM_RANGE = ">=0.23,<0.24"

#: The two entry points, as a mapping a probe can compare against ``importlib.metadata``.
ENTRY_POINTS: Dict[str, Dict[str, str]] = {
    "vllm.platform_plugins": {"tileinfer": PLATFORM_ENTRY_POINT},
    "vllm.general_plugins": {"tileinfer": PLUGIN_ENTRY_POINT},
}

ENV_STRICT = "TILEINFER_VLLM_STRICT"
ENV_REGISTER_BACKEND = "TILEINFER_VLLM_REGISTER_BACKEND"
ENV_GRAPH_BRIDGE = "TILEINFER_VLLM_GRAPH_BRIDGE"
#: The operator-family switch is spelled by ``tileinfer.vllm.ops.switch`` (``ENV_OPS``); it is
#: re-exported here only so ``status()``/the banner can name it without importing that module.
ENV_OPS = "TILEINFER_OPS"

#: Re-entrant: ``install()`` may be reached from a loader that itself imports something which
#: re-enters the plugin (a patch that triggers an entry-point reload), and a plain ``Lock`` would
#: deadlock inside the interpreter rather than log.
_LOCK = threading.RLock()

_STATE: Dict[str, Any] = {
    "installed": False,
    "reason": "install() has not run in this process",
    "strict": False,
    "report": None,  # Optional[patches.InstallReport]
    "backend": {},  # the dict _install_backend() returns
    "graph": {},  # the dict _install_graph_bridge() returns
    "ops": {},  # the dict _install_ops() returns
    "models": {},  # the dict _install_models() returns
}


# =============================================================================================
# Small helpers: env flags and the vLLM probe (neither imports vLLM)
# =============================================================================================


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean env var; an unset/empty var means ``default``.

    Explicit spellings only: ``"0"`` / ``"false"`` / ``"no"`` / ``"off"`` are the only off values.
    (The legacy adapter learned this the hard way — a truthiness test made ``TILEINFER_VLLM=0``
    still enable the plugin, see the frozen snapshot under ``legacy/``.)
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _version_in_range(version: str) -> bool:
    """Best-effort "is this 0.23.x?"; a wrong answer only changes a log line, never behaviour."""
    if not version:
        return False
    parts = version.split("+")[0].split(".")
    if len(parts) < 2:
        return False
    try:
        return (int(parts[0]), int(parts[1])) == (0, 23)
    except ValueError:
        return False


def _vllm_probe() -> Dict[str, Any]:
    """``{available, version, supported, detail}`` — is this a vLLM environment at all?

    Two probes, cheapest first: an already-imported ``vllm`` module is the strongest evidence there
    is (and the only evidence that survives a frozen build, where ``__spec__`` is ``None`` and
    ``find_spec`` therefore raises); otherwise ``find_spec``, which does not import anything and so
    keeps a CPU box clean.  A broken install raises from the import machinery in two ways
    (``ImportError`` for a parent package, ``ValueError`` for a spec-less module); both are caught,
    because the only thing this function may do is classify — never raise.
    """
    info: Dict[str, Any] = {"available": False, "version": "", "supported": False, "detail": ""}
    if sys.modules.get("vllm") is None:
        try:
            spec = importlib.util.find_spec("vllm")
        except (ImportError, ValueError) as exc:
            info["detail"] = f"vLLM is not importable here ({exc!r})"
            return info
        if spec is None:
            info["detail"] = "vLLM is not installed (no 'vllm' module on sys.path)"
            return info
    info["available"] = True
    try:
        from importlib.metadata import version as _distribution_version

        info["version"] = _distribution_version("vllm")
    except Exception:  # noqa: BLE001 - a source checkout has no distribution metadata
        info["version"] = ""
    info["supported"] = _version_in_range(info["version"])
    if info["version"]:
        info["detail"] = f"vLLM {info['version']}"
    else:
        info["detail"] = "vLLM (already imported; distribution version unknown)"
    return info


# =============================================================================================
# Step 1: the patch registry
# =============================================================================================


def _install_patches(strict: bool) -> Optional[patches.InstallReport]:
    """Apply :data:`tileinfer.vllm.compat.patches.PATCHES`; ``None`` when the registry itself broke.

    ``install_all`` already tolerates a vanished target and logs its own summary line.  The extra
    wrap here is for the one failure it cannot report: ``install_all`` raising before/around the
    loop (a patch list mutated concurrently, an ``__iter__`` bug).  That must not stop the boot.
    """
    try:
        report = patches.install_all(strict=strict)
    except Exception:  # noqa: BLE001 - the plugin must not stop the engine from booting
        logger.exception(
            "TileInfer patch registry: install_all() raised; continuing without patches "
            "(the engine boots, the patches are missing, and `tileinfer doctor` will show them)"
        )
        return None
    for target, error in report.failed:
        logger.warning("TileInfer patch failed: %s (%s)", target, error)
    for target, error in report.skipped:
        logger.info("TileInfer patch skipped (target gone upstream?): %s (%s)", target, error)
    return report


# =============================================================================================
# Step 2: the attention-backend registry slot (decision in the module docstring)
# =============================================================================================


def _install_backend() -> Dict[str, Any]:
    """Register the attention backend under ``CUSTOM`` and *verify* the slot resolves to it.

    Not registering is not an error: the platform returns the class path directly, so the serving
    path works without this.  The verification is an identity check on
    ``AttentionBackendEnum.CUSTOM.get_class()`` because a wrong registration is worse than none —
    it would silently hand the engine a different class.
    """
    info: Dict[str, Any] = {
        "slot": BACKEND_ENUM_SLOT,
        "class_path": None,
        "registered": False,
        "detail": "",
    }
    if not _env_flag(ENV_REGISTER_BACKEND, True):
        info["detail"] = (
            f"not registered ({ENV_REGISTER_BACKEND}=0); the platform hook returns the class path "
            "directly, so serving is unaffected - only '--attention-backend custom' would fail"
        )
        return info

    try:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
    except Exception as exc:  # noqa: BLE001 - a vLLM point release may move the registry
        info["detail"] = f"not registered: the vLLM attention registry is not importable ({exc!r})"
        logger.warning(
            "TileInfer attention backend not registered: vllm.v1.attention.backends.registry is "
            "not importable (%r). The platform hook still returns the class path, so serving is "
            "unaffected; only '--attention-backend custom' would fail.",
            exc,
        )
        return info

    try:
        from tileinfer.vllm.attn.backend import register_with_vllm

        path = register_with_vllm()
    except Exception as exc:  # noqa: BLE001
        info["detail"] = f"registration failed: {exc!r}"
        logger.warning(
            "TileInfer attention backend registration failed (%r); continuing - the platform "
            "hook returns the class path directly, so serving is unaffected",
            exc,
        )
        return info

    info["class_path"] = path
    try:
        resolved = AttentionBackendEnum.CUSTOM.get_class()
    except Exception as exc:  # noqa: BLE001
        info["detail"] = f"registered {path}, but CUSTOM does not resolve ({exc!r})"
        logger.warning("TileInfer attention backend: %s", info["detail"])
        return info
    resolved_path = f"{getattr(resolved, '__module__', '')}.{getattr(resolved, '__qualname__', '')}"
    if resolved_path != path:
        info["detail"] = f"registered {path}, but CUSTOM resolves to {resolved_path}"
        logger.warning(
            "TileInfer attention backend: %s - another plugin may have taken the CUSTOM slot "
            "(it is a process-wide singleton); the platform hook is unaffected",
            info["detail"],
        )
        return info

    info["registered"] = True
    info["detail"] = f"registered {path} in {BACKEND_ENUM_SLOT}"
    return info


# =============================================================================================
# Step 3: the graph bridge (conditional, and it proves itself)
# =============================================================================================


def _install_graph_bridge() -> Dict[str, Any]:
    """Ask :mod:`tileinfer.vllm.graphs` first, delegate to it if not-ready, then ask again.

    The alias table, the all-or-nothing write and the rollback all live in
    :func:`tileinfer.vllm.graphs.install` (four required aliases + five best-effort ones, see
    :data:`tileinfer.vllm.graphs.TORCH_CUDA_ALIASES`).  Nothing here duplicates that list: the
    oracle and the writer are the same module, so "the bridge ran but the verdict is still False"
    can only mean what it says (a required counterpart is missing, or something rewrote
    ``torch.cuda`` afterwards) instead of meaning "the two alias lists disagree".

    Returns the dict ``status()["graphs"]`` is built from.  Never raises.
    """
    info: Dict[str, Any] = {
        "ready": False,
        "reason": "",
        "installed": False,
        "aliases": [],
        "detail": "",
    }
    try:
        ready, reason = graphs.capture_path_ready()
    except Exception as exc:  # noqa: BLE001 - a probe must not abort the install
        info["detail"] = f"capture_path_ready() raised {exc!r}; no bridge attempted"
        logger.warning("TileInfer graph bridge: %s", info["detail"])
        return info

    info["ready"] = bool(ready)
    info["reason"] = reason
    if ready:
        info["detail"] = f"nothing to do: the capture path already works ({reason})"
        return info
    if not _env_flag(ENV_GRAPH_BRIDGE, True):
        info["detail"] = (
            f"not installed ({ENV_GRAPH_BRIDGE}=0); the platform will force eager execution "
            f"({reason})"
        )
        logger.info("TileInfer graph bridge: %s", info["detail"])
        return info

    try:
        installed, detail = graphs.install()
    except Exception as exc:  # noqa: BLE001 - graphs.install() promises not to raise; belt+braces
        installed, detail = False, f"graphs.install() raised {exc!r}"
    info["installed"] = bool(installed)
    info["detail"] = detail
    try:
        info["aliases"] = list(graphs.active_aliases())
    except Exception:  # noqa: BLE001 - reporting only; the verdict below is what matters
        info["aliases"] = []

    # Re-ask the *same* oracle that gated the bridge: it proves the identity, it does not trust us.
    try:
        info["ready"], info["reason"] = graphs.capture_path_ready()
    except Exception as exc:  # noqa: BLE001
        info["ready"], info["reason"] = False, f"capture_path_ready() raised {exc!r} after bridging"

    if info["installed"] and info["ready"]:
        logger.info("TileInfer graph bridge ready: %s", info["reason"])
    elif info["installed"]:
        logger.warning(
            "TileInfer graph bridge: %s, but capture_path_ready() is still False (%s); the "
            "platform will keep forcing eager execution - do not trust capture_calls yet",
            detail,
            info["reason"],
        )
    else:
        logger.info(
            "TileInfer graph bridge not installed: %s. The platform forces eager execution, which "
            "is correct but slower (docs/hardware-checklist.md step 10).",
            detail,
        )
    return info


# =============================================================================================
# Step 4: the operator wiring (CustomOp methods + the GDN layer's module globals)
# =============================================================================================


def _install_ops() -> Dict[str, Any]:
    """Hand the target model's operators to :mod:`tileinfer.vllm.ops.install`.

    The import is local on purpose: ``tileinfer.vllm.ops`` pulls the operator modules (and torch),
    and ``import tileinfer.vllm.plugin`` has to stay clean on a CPU box.  ``install_ops()`` is
    itself failure-tolerant (it plans against the live module tree, logs every name it could not
    bind and returns a report), so the only thing this wrapper adds is one more guard: a broken
    operator module must not stop the engine from booting.
    """
    info: Dict[str, Any] = {
        "bound": [],
        "unbound": [],
        "disabled": [],
        "detail": "install_ops() did not run",
    }
    try:
        from tileinfer.vllm.ops.install import install_ops

        report = install_ops()
    except Exception as exc:  # noqa: BLE001 - a general plugin must not raise
        info["detail"] = f"install_ops() raised {exc!r}; vLLM's own operators are still in place"
        logger.warning("TileInfer ops: %s", info["detail"])
        return info
    info.update(
        bound=list(report.bound),
        unbound=list(report.unbound),
        disabled=list(report.disabled),
        detail=report.summary(),
    )
    for op in report.unbound:
        logger.warning(
            "TileInfer ops: %s is NOT taken over (%s); the engine will call vLLM's own "
            "implementation for it",
            op,
            report.details.get(op, "no reason recorded"),
        )
    return info


# =============================================================================================
# Step 5: the model registry (out-of-tree architectures)
# =============================================================================================


def _install_models() -> Dict[str, Any]:
    """Register the architectures this package implements and vLLM 0.23 does not ship.

    Only ``DeepseekV41ForCausalLM`` today: vLLM 0.23.0 has ``DeepseekV4ForCausalLM`` but not v4.1,
    whose engram tables / DSpark stages / two-level candidate indexer / vision tower are absent
    upstream at this version (``docs/notes/dsv41-facts.md`` §18).  Registration is by class path
    string, so importing the model module (and its 96k-tensor parameter tree) waits until a
    checkpoint with that architecture is actually resolved.

    Failure-tolerant like every other step: a missing vLLM registry or a duplicate registration is a
    logged detail, never a boot failure.
    """
    info: Dict[str, Any] = {"registered": [], "detail": "model registration did not run"}
    details: List[str] = []
    # vLLM reads config.json through transformers.AutoConfig before it looks at architectures, and the
    # checkpoint's `model_type: deepseek_v41` is unknown to released transformers -- so teach AutoConfig
    # the name first, otherwise `vllm` cannot even parse the model directory.
    try:
        from tileinfer.vllm.config import register_config

        details.append(f"config {register_config()['config']}")
    except Exception as exc:  # noqa: BLE001 - a general plugin must not raise
        details.append(f"config registration raised {exc!r}")
        logger.warning("TileInfer config registration: %s", details[-1])
    actors = (
        ("tileinfer.vllm.models.dsv41", "register_models"),
    )
    for module_name, entry in actors:
        try:
            module = __import__(module_name, fromlist=[entry])
            result = getattr(module, entry)()
        except Exception as exc:  # noqa: BLE001 - a general plugin must not raise
            details.append(f"{module_name}.{entry}() raised {exc!r}")
            logger.warning("TileInfer model registration: %s", details[-1])
            continue
        if result.get("registered"):
            info["registered"].append(result["arch"])
        details.append(str(result.get("detail", "")))
    info["detail"] = "; ".join(d for d in details if d) or "nothing to register"
    return info


# =============================================================================================
# The entry point
# =============================================================================================


def install(*, strict: Optional[bool] = None) -> bool:
    """``vllm.general_plugins`` entry point: patch, register, bridge — once per process.

    Called with no arguments by vLLM (``load_plugins_by_group`` invokes each entry point and
    ignores the return value), so every parameter is optional and keyword-only.

    Returns ``True`` when this process is (or was already) patched and wired, ``False`` when this is
    not a vLLM environment or when something failed.  It never raises: a broken optional component
    must not stop the engine from booting (the whole reason ``patches.install_all`` is tolerant).

    ``strict`` defaults to ``TILEINFER_VLLM_STRICT``; ``strict=True`` turns a skipped patch into a
    reported failure (for ``tileinfer doctor`` / CI), and changes nothing else — the boot still
    continues.
    """
    with _LOCK:
        if _STATE["installed"]:
            logger.debug("TileInfer vLLM plugin: already installed in this process; no-op")
            return True
        try:
            probe = _vllm_probe()
            if not probe["available"]:
                _STATE["reason"] = probe["detail"]
                logger.debug(
                    "TileInfer vLLM plugin: %s; nothing installed (import tileinfer stays clean)",
                    probe["detail"],
                )
                return False
            if probe["version"] and not probe["supported"]:
                logger.warning(
                    "TileInfer vLLM plugin: %s is outside the range vllm%s this package is written "
                    "against; the patch registry will report what it cannot find instead of "
                    "failing the boot",
                    probe["detail"],
                    SUPPORTED_VLLM_RANGE,
                )
            elif not probe["version"]:
                logger.debug(
                    "TileInfer vLLM plugin: %s; the vllm%s range check was skipped",
                    probe["detail"],
                    SUPPORTED_VLLM_RANGE,
                )

            strict = _env_flag(ENV_STRICT, False) if strict is None else bool(strict)
            report = _install_patches(strict)
            backend = _install_backend()
            graph = _install_graph_bridge()
            ops = _install_ops()
            models = _install_models()
            _STATE.update(
                installed=True,
                reason=probe["detail"],
                strict=strict,
                report=report,
                backend=backend,
                graph=graph,
                ops=ops,
                models=models,
            )
            logger.info(
                "TileInfer vLLM plugin installed | %s | strict=%s | %s | backend: %s | graph: %s "
                "| %s | models: %s",
                probe["detail"],
                strict,
                report.summary() if report is not None else "patch registry unavailable",
                backend["detail"],
                graph["detail"],
                ops["detail"],
                models["detail"],
            )
            return True
        except Exception:  # noqa: BLE001 - the last resort: never raise out of a general plugin
            logger.exception(
                "TileInfer vLLM plugin: install() failed; continuing without it - the engine must "
                "still boot. Run `python -c \"from tileinfer.vllm.plugin import status;"
                "print(status())\"` for what did install."
            )
            _STATE["installed"] = False
            _STATE["reason"] = "install() raised; see the log"
            return False


def uninstall() -> bool:
    """Revert what *can* be reverted: the graph aliases and the ``CUSTOM`` slot.

    The aliases belong to :mod:`tileinfer.vllm.graphs`, so reverting them is a delegation to
    :func:`tileinfer.vllm.graphs.uninstall` (which restores exactly the attributes *it* wrote and
    leaves a pre-existing torch_npu bridge alone) — there is no second copy of the "what did we
    write" bookkeeping here.

    Patches applied through :mod:`tileinfer.vllm.compat.patches` cannot be undone — the registry
    has no restore hook by design (a patch is an attribute write on vLLM's own modules, and
    pretending to undo it would be a lie the next patch hides behind).  A process restart is the
    only clean revert for those.  This function exists for the probes and for tests that need the
    CPU-side aliases gone; it is not part of any serving path.
    """
    with _LOCK:
        reverted: List[str] = []
        try:
            reverted.extend(graphs.uninstall())
        except Exception as exc:  # noqa: BLE001 - best effort, a broken alias is not fatal
            logger.warning("TileInfer graph bridge could not be reverted: %r", exc)

        if _STATE.get("backend", {}).get("registered"):
            try:
                from vllm.v1.attention.backends.registry import AttentionBackendEnum

                AttentionBackendEnum.CUSTOM.clear_override()
                reverted.append(f"{BACKEND_ENUM_SLOT} (clear_override)")
            except Exception as exc:  # noqa: BLE001
                logger.debug("TileInfer: could not clear %s (%r)", BACKEND_ENUM_SLOT, exc)

        _STATE.update(
            installed=False,
            reason="uninstalled in this process",
            backend={},
            graph={},
            ops={},
            models={},
        )
        logger.info(
            "TileInfer vLLM plugin uninstalled: reverted %s; patches applied earlier stay applied "
            "until this process exits",
            reverted or "nothing",
        )
        return True


# =============================================================================================
# status(): what the doctor command prints
# =============================================================================================


def _backend_class_path() -> Tuple[Optional[str], str]:
    """The attention backend's class path, read from the platform's constant when importable.

    Deliberately *not* duplicated as a literal here: ``tests/test_attn_contract.py`` pins
    ``platform.ATTN_BACKEND_CLASS_PATH`` to the real class, and a second copy in this module would
    be a second thing to drift.  Outside a vLLM environment the platform is not importable, so the
    answer is ``(None, reason)`` rather than a plausible string.
    """
    try:
        from tileinfer.vllm.platform import ATTN_BACKEND_CLASS_PATH
    except Exception as exc:  # noqa: BLE001 - expected on a CPU box
        return (None, f"tileinfer.vllm.platform is not importable ({exc!r})")
    return (ATTN_BACKEND_CLASS_PATH, "from tileinfer.vllm.platform.ATTN_BACKEND_CLASS_PATH")


def status() -> Dict[str, Any]:
    """The install state as plain data: the patch report, graph readiness/aliases, the backend path.

    Safe to call in any process, installed or not, with or without vLLM: it never installs anything
    and never raises.  Meant for ``tileinfer doctor`` and for the bring-up checklist, where the
    question is always "did the plugin actually do anything in *this* process?".  The graph
    section's ``bridge_installed`` / ``active_aliases`` are read from
    :mod:`tileinfer.vllm.graphs`' own bookkeeping (plus this process's last bridge verdict), so they
    cannot claim an alias that module did not write.
    """
    probe = _vllm_probe()
    try:
        ready, reason = graphs.capture_path_ready()
    except Exception as exc:  # noqa: BLE001 - status must answer even on a broken device stack
        ready, reason = False, f"capture_path_ready() raised {exc!r}"
    try:
        # The aliases this module actually wrote (graphs.py's own bookkeeping, not a second copy).
        active_aliases = [str(name) for name in graphs.active_aliases()]
    except Exception:  # noqa: BLE001 - a broken torch.cuda is a reason, not a crash
        active_aliases = []
    graph = dict(_STATE.get("graph") or {})

    report = _STATE["report"]
    backend = dict(_STATE.get("backend") or {})
    if not backend:
        path, detail = _backend_class_path()
        backend = {
            "slot": BACKEND_ENUM_SLOT,
            "class_path": path,
            "registered": False,
            "detail": f"the plugin did not run in this process; {detail}",
        }

    return {
        "plugin": {
            "installed": bool(_STATE["installed"]),
            "reason": _STATE["reason"],
            "strict": bool(_STATE["strict"]),
            "entry_points": {group: dict(names) for group, names in ENTRY_POINTS.items()},
        },
        "vllm": probe,
        "patches": {
            "summary": report.summary() if report is not None else "not installed in this process",
            "applied": list(report.applied) if report is not None else [],
            "skipped": (
                [{"target": target, "why": why} for target, why in report.skipped]
                if report is not None
                else []
            ),
            "failed": (
                [{"target": target, "why": why} for target, why in report.failed]
                if report is not None
                else []
            ),
            "registry": patches.describe(),
        },
        "graphs": {
            "branch": graphs.GRAPH_BRANCH,
            "capture_path_ready": bool(ready),
            "reason": reason,
            # True when a bridge is live in this process: the aliases graphs.install() wrote, or the
            # verdict of the last _install_graph_bridge() call in this process.
            "bridge_installed": bool(graph.get("installed")) or bool(active_aliases),
            "bridge_detail": graph.get("detail", ""),
            "active_aliases": active_aliases,
            "unverified_symbols": [name for name, _why in graphs.UNVERIFIED_SYMBOLS],
        },
        "backend": backend,
        "ops": dict(_STATE.get("ops") or {}),
        "models": dict(_STATE.get("models") or {}),
        "platform": {"entry_point": PLATFORM_ENTRY_POINT, "class_path": PLATFORM_CLASS_PATH},
    }
