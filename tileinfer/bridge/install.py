"""The ``vllm.general_plugins`` entry point: one idempotent ``install()`` per process.

This is the **only** entry point the package registers.  There is deliberately no platform
plugin: ``vllm-ascend`` owns that slot, and a second claimant makes the process refuse to
start before any of our code can explain itself (``docs/integration.md`` §2).

Order of operations, and why it is fixed:

1. **Read the plan.**  No plan, no injection.  The vendor path then runs exactly as it would
   without us installed, which is a supported outcome, not an error.
2. **Version gate.**  The plan records the vLLM / vllm-ascend / Python versions it was
   computed against.  A mismatch means "do not inject": the usual symptom of injecting across
   versions is wrong numbers, not a traceback.
3. **Configure counters.**  Before any table is swapped, so a failed seam is still visible.
4. **Install the seams** named by the plan, each independently and idempotently.
5. **Probe what actually took effect** and log one line.  We never infer "it must have
   worked" from a table write (``docs/integration.md`` §7).

Red lines honoured here: no model config is read (the entry point is argument-free), no
kernel is compiled, and no exception leaves this function.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Any, Dict, Optional

from ..contracts.plan import Plan
from . import metrics, seams

logger = logging.getLogger("tileinfer.bridge")
# vLLM configures handlers for its own loggers but not necessarily the root level; without
# this, the one-line install record is filtered out before it reaches the server log.
logger.setLevel(logging.INFO)

ENTRY_POINT = "vllm.general_plugins:tileinfer"
PLAN_ENV = "TILEINFER_PLAN"
DISABLE_ENV = "TILEINFER_V3"
STRICT_ENV = "TILEINFER_BRIDGE_STRICT"
MANIFEST_ENV = "TILEINFER_MANIFEST"
AUDIT_ENV = "TILEINFER_AUDIT"
AUDIT_DEFAULT_PATH = "/tmp/tileinfer-ops.json"
_AUDIT_OFF = {"", "0", "false", "off", "no"}
_AUDIT_DEFAULT = {"1", "true", "on", "yes"}


def _audit_path() -> Optional[str]:
    """Where the op audit writes, or ``None`` when it is off (the default)."""
    raw = os.environ.get(AUDIT_ENV, "").strip()
    if raw.lower() in _AUDIT_OFF:
        return None
    return AUDIT_DEFAULT_PATH if raw.lower() in _AUDIT_DEFAULT else raw

_LOCK = threading.Lock()
_INSTALLED = False
_STATUS: Dict[str, Any] = {
    "installed": False,
    "skipped": None,
    "plan_id": None,
    "plan_source": None,
    "seams": [],
    "versions": {},
}


def _python_key(version: str) -> str:
    parts = version.split(".")
    return ".".join(parts[:2])


def _versions() -> Dict[str, str]:
    import importlib.metadata

    def _v(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except Exception:
            return "missing"

    return {
        "vllm": _v("vllm"),
        "vllm_ascend": _v("vllm_ascend"),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch": _v("torch"),
    }


def _load_plan() -> tuple[Optional[Plan], Optional[str]]:
    plan_path = os.environ.get(PLAN_ENV)
    if plan_path:
        if not os.path.exists(plan_path):
            return None, f"{PLAN_ENV}={plan_path} does not exist"
        return Plan.load(plan_path), plan_path

    # No plan file: build the vendor-first plan from the declaration registry.  This keeps
    # the "installed but forgot the deploy step" path working, and records that it happened.
    from ..manifests import find_manifest
    from ..resolve.decide import decide
    from ..resolve.probe import probe

    manifest = find_manifest(os.environ.get(MANIFEST_ENV, "qwen3_6"))
    plan, _report = decide(manifest, probe(), policy="vendor_first")
    return plan, "auto"


def _gate(plan: Plan, live: Dict[str, str]) -> Optional[str]:
    planned = plan.versions
    if not os.environ.get("TILEINFER_BRIDGE_IGNORE_VERSIONS"):
        for key, planned_value, live_value in (
            ("vllm", planned.vllm, live["vllm"]),
            ("vllm_ascend", planned.vllm_ascend, live["vllm_ascend"]),
        ):
            if planned_value != live_value:
                return f"version gate: plan {key}={planned_value}, live={live_value}"
        if _python_key(planned.python) != _python_key(live["python"]):
            return f"version gate: plan python={planned.python}, live={live['python']}"
    return None


def install() -> bool:
    """Apply the plan.  Safe to call many times and from many processes."""
    global _INSTALLED
    if os.environ.get(DISABLE_ENV, "1").strip().lower() in {"0", "false", "off", "no"}:
        _STATUS.update(installed=False, skipped=f"{DISABLE_ENV} disables the bridge")
        logger.info("TileInfer v3 bridge: disabled by %s", DISABLE_ENV)
        return False
    with _LOCK:
        if _INSTALLED:
            return True
        try:
            live = _versions()
            _STATUS["versions"] = live
            plan, source = _load_plan()
            _STATUS["plan_source"] = source
            if plan is None:
                _STATUS["skipped"] = source
                logger.warning("TileInfer v3 bridge: no plan (%s); vendor path unchanged", source)
                return False
            _STATUS["plan_id"] = plan.plan_id
            gate = _gate(plan, live)
            if gate:
                _STATUS["skipped"] = gate
                logger.warning("TileInfer v3 bridge: %s; not injecting", gate)
                return False

            metrics.COUNTERS.plan_id = plan.plan_id
            instrumentation = plan.instrumentation or {}
            metrics_path = os.environ.get(
                "TILEINFER_METRICS_PATH", instrumentation.get("metrics_path")
            )
            metrics.configure(
                metrics_path,
                int(instrumentation.get("dump_every_decode_steps", 500)),
            )

            report = seams.install_all(plan)
            _STATUS["seams"] = report["seams"]

            # Diagnostic, off unless asked for: which implementation each op instance got.
            audit_path = _audit_path()
            if audit_path:
                try:
                    _STATUS["audit"] = seams.install_op_audit(audit_path)
                except Exception as exc:  # never let a diagnostic break the install
                    _STATUS["audit"] = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
            # Critical seams (config gate, graph hook, attention selector) must all be in
            # place; a partial install is reported as failure so `doctor` and the caller see
            # it, even though the hooks that did install remain harmless.
            _STATUS["installed"] = bool(report["ok"])
            _INSTALLED = bool(report["ok"])

            ok = [s["name"] for s in report["seams"] if s["ok"]]
            bad = [s for s in report["seams"] if not s["ok"]]
            skipped = ",".join(f"{s['name']}:{s['detail']}" for s in bad) or "none"
            message = (
                f"TileInfer v3 bridge: {plan.summary()} | seams={','.join(ok) or 'none'} "
                f"| skipped={skipped} | metrics={metrics_path}"
            )
            if audit_path:
                message += f" | audit={audit_path}"
            logger.info(message)
            # A general plugin may run before vLLM installs its log handlers (the API server
            # path does), so the logger record can be dropped.  One stderr line is the
            # observability contract; it cannot raise and does not touch the request path.
            print(f"[tileinfer] {message}", file=sys.stderr, flush=True)
            return bool(report["ok"])
        except Exception as exc:  # entry points must never raise: upstream catches imports only
            _STATUS["skipped"] = f"{type(exc).__name__}: {exc}"
            logger.exception("TileInfer v3 bridge failed to install; vendor path continues")
            if os.environ.get(STRICT_ENV):
                raise
            return False


def uninstall() -> None:
    """Best-effort: clear our hooks so the vendor path is restored in tests."""
    global _INSTALLED
    try:
        from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

        for attr in ("_tileinfer_graph_hooked",):
            if hasattr(CudagraphDispatcher, attr):
                delattr(CudagraphDispatcher, attr)
    except Exception:
        pass
    try:
        from vllm_ascend.platform import NPUPlatform

        if hasattr(NPUPlatform, "_tileinfer_attn_hooked"):
            delattr(NPUPlatform, "_tileinfer_attn_hooked")
    except Exception:
        pass
    _INSTALLED = False
    _STATUS["installed"] = False
    _STATUS["skipped"] = "uninstalled"


def status() -> Dict[str, Any]:
    """Machine-readable install status, for ``doctor`` and the acceptance evidence."""
    data = dict(_STATUS)
    data["metrics"] = metrics.COUNTERS.snapshot()
    data["metrics_path"] = metrics.path()
    return data


__all__ = ["install", "uninstall", "status", "ENTRY_POINT", "PLAN_ENV", "DISABLE_ENV"]
