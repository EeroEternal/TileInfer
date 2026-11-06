"""``tileinfer doctor`` — the bring-up checklist of ``docs/hardware-checklist.md`` as one command.

Why this exists
---------------
``docs/hardware-checklist.md`` steps 1-5 are the prerequisites for *anything* else: a CANN shell,
``torch``/``torch_npu``, a visible device, the TileLang Ascend wheel and its cache, a vLLM in the
supported range, the general plugin installed, the alias table the graph verdict reads, and the
operator takeovers.  Those answers exist scattered across six commands, a JSON dump and a pytest
run; a person bringing the box up for the first time has to know all of them before they can tell
"environment" from "kernel".  ``tileinfer doctor`` asks them in order, in one process, and prints
``PASS`` / ``FAIL`` / ``SKIP`` with the fix hint from the checklist's own "failure triage" tables.

The check list (and the checklist step each one mirrors)
--------------------------------------------------------
=================================  ====  =======================================================
check                              step  what it proves
=================================  ====  =======================================================
``python``                          1    interpreter >= 3.9 (``requires-python``)
``torch``                           1    ``import torch`` and its version
``torch_npu``                       1    ``import torch_npu`` and that it registered ``torch.npu``
``cann_env``                        1    a CANN toolkit root is exported by ``set_env.sh``
``device``                          1    ``torch.npu.device_count()`` > 0 and ``get_device_name(0)``
``tilelang``                        1    the **Ascend** TileLang wheel, not the plain PyPI one
``tilelang_cache``                  1    ``TILELANG_CACHE_DIR`` is set and writable
``vllm_version``                    2    ``import vllm`` with a version in
                                         ``plugin.SUPPORTED_VLLM_RANGE``
``plugin.status``                   2    the general plugin installs and reports all four steps
``ops.binding``                     2    ``tileinfer/vllm/ops/install.py``'s binding report plus
                                         each op's candidate gate status (only an *unexpected*
                                         miss fails: ``chunk_local_cumsum`` is declined on
                                         purpose, ``abi_ok=False``)
``runner_source``                   4    ``runtime/vllm-api/...gpu_model_runner.py`` with the four
                                         hooks the worker needs
``graphs.capture_path_ready``       5    the graph verdict (soft: not-ready is the documented
                                         ``enforce_eager=True`` fallback, i.e. correct-but-slow)
=================================  ====  =======================================================

Step 3 ("the CPU suite is green") is deliberately **not** run: doctor is a one-process diagnostic,
and launching pytest from it would take minutes and bury the result.  The tail line of the report
names the step to start at, so step 3 is still reached by hand (the checklist prints the command).

Exit code and severity
----------------------
``main()`` returns 0 when no **hard** check failed and 1 otherwise (2 for a usage error).  A check
is hard when it names something the engine cannot work around on that device.  The one soft check
is ``graphs.capture_path_ready``: when it says not-ready the platform forces eager execution on
purpose (``platform.py:_force_eager_until_capture_lands``), so that is a performance regression with
a documented reason, not a broken environment.

Invocation
----------
``tileinfer``, ``tileinfer doctor``, ``tileinfer check`` and ``tileinfer status`` are the *same*
command: the subcommand word is optional and only names what the console script already is, so the
natural spelling must not be a usage error.  ``--json`` and ``--help`` combine with it
(``tileinfer doctor --json``); any other word is still a usage error (exit 2).

Nothing here may raise
----------------------
Every check runs through :func:`_run_check`, which owns the ``try``/``except``: a check that raises
is printed as a ``FAIL`` naming the exception, never as a traceback.  ``main()`` has a last-resort
guard around the whole report for the same reason.  The printed text is ASCII-only so an ``npu-smi``
-era console code page cannot turn the report into a ``UnicodeEncodeError``.

What is *not* imported at module scope
--------------------------------------
No ``torch``, no ``vllm``, no ``torch_npu``: an installed console script imports this module before
it knows whether the box has any of them, and a ``ModuleNotFoundError`` at import time would print a
traceback instead of a checklist.  Every import lives inside the check that needs it.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import platform
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "PASS",
    "FAIL",
    "SKIP",
    "Check",
    "CheckSpec",
    "CHECKS",
    "STEP_TITLES",
    "SOFT_CHECKS",
    "SUBCOMMANDS",
    "run_checks",
    "format_report",
    "tail_line",
    "main",
]

logger = logging.getLogger("tileinfer.vllm.doctor")

#: The three printed outcomes.  Strings (not an ``Enum``) so a report line is readable as data.
PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

#: What each checklist step being mirrored is about, as the section header of the report.
STEP_TITLES: Dict[int, str] = {
    1: "environment: CANN, torch/torch_npu, the device, the TileLang cache",
    2: "packages and the general plugin: vLLM version, plugin.status(), the operator wiring",
    3: "the CPU suite (not run from doctor: `python -m pytest tests -q`)",
    4: "the local model-runner subset (runtime/vllm-api/...gpu_model_runner.py)",
    5: "platform selection and the graph verdict (capture_path_ready)",
}

_HEADER = "TileInfer doctor - Ascend bring-up checklist (docs/hardware-checklist.md steps 1-5)"

#: The words a person types after the console script: ``tileinfer doctor`` is the natural one,
#: ``check`` and ``status`` are the two they reach for next.  They are aliases for the command the
#: entry point already is (there is no second action), so ``main()`` accepts at most one of them and
#: otherwise ignores it - a diagnostic must not fail on the spelling of its own name.
SUBCOMMANDS: Tuple[str, ...] = ("doctor", "check", "status")

#: A check function's return value: ``(status, detail, hint)``.  Raising is also allowed; the
#: wrapper turns that into a printed ``FAIL``.
CheckFn = Callable[[], Tuple[str, str, str]]


@dataclass(frozen=True)
class CheckSpec:
    """One check: its stable ``name``, the checklist ``step`` it mirrors, and how severe it is.

    ``hard`` is the exit-code bit: a ``FAIL`` on a hard check means "the engine cannot run here",
    while a ``FAIL`` on a soft check is a documented degradation.
    """

    name: str
    step: int
    hard: bool
    fn: CheckFn


@dataclass(frozen=True)
class Check:
    """One printed line: the spec plus the outcome the check reported (or the wrapper caught)."""

    name: str
    step: int
    hard: bool
    status: str
    detail: str
    hint: str = ""

    @property
    def ok(self) -> bool:
        return self.status == PASS

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "step": self.step,
            "hard": self.hard,
            "status": self.status,
            "detail": self.detail,
            "hint": self.hint,
        }


# =============================================================================================
# Small shared helpers (no heavy import: each one is called from inside a check)
# =============================================================================================


def _ok(detail: str, hint: str = "") -> Tuple[str, str, str]:
    return (PASS, detail, hint)


def _fail(detail: str, hint: str) -> Tuple[str, str, str]:
    return (FAIL, detail, hint)


def _skip(detail: str, hint: str) -> Tuple[str, str, str]:
    return (SKIP, detail, hint)


def _plugin():
    """``tileinfer.vllm.plugin``, imported on demand (it imports ``graphs``, which imports torch)."""
    from tileinfer.vllm import plugin

    return plugin


def _vllm_probe() -> Dict[str, Any]:
    """``plugin._vllm_probe()``: ``{available, version, supported, detail}``, never importing vLLM."""
    return _plugin()._vllm_probe()


def _torch_npu_unavailable() -> Optional[str]:
    """``None`` when ``torch`` + ``torch_npu`` + ``torch.npu`` all work, else the reason to SKIP."""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return f"torch is not importable here ({exc!r})"
    try:
        import torch_npu  # noqa: F401  (the import is what registers torch.npu)
    except Exception as exc:  # noqa: BLE001
        return f"torch_npu is not importable here ({exc!r})"
    if getattr(torch, "npu", None) is None:
        return "torch_npu imported but it did not register torch.npu (a torch/torch_npu mismatch)"
    return None


def _repo_root() -> pathlib.Path:
    """The checkout this module is imported from (``<repo>/tileinfer/vllm/doctor.py``)."""
    return pathlib.Path(__file__).resolve().parents[2]


# =============================================================================================
# Checklist step 1: environment
# =============================================================================================


def _check_python() -> Tuple[str, str, str]:
    version = platform.python_version()
    if sys.version_info < (3, 9):
        return _fail(
            f"Python {version} is older than 3.9",
            "install Python >= 3.9 (pyproject.toml requires-python), then recreate the venv",
        )
    return _ok(f"Python {version}")


def _check_torch() -> Tuple[str, str, str]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return _fail(
            f"import torch failed: {exc!r}",
            "install torch (the CPU wheel is enough for the CPU suite; the device box needs the "
            "torch build torch_npu was compiled against) - docs/hardware-checklist.md step 1",
        )
    return _ok(f"torch {getattr(torch, '__version__', '<unknown version>')}")


def _check_torch_npu() -> Tuple[str, str, str]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return _skip(
            f"skipped: torch is not importable ({exc!r})",
            "fix the torch check first (docs/hardware-checklist.md step 1)",
        )
    try:
        import torch_npu
    except Exception as exc:  # noqa: BLE001
        return _fail(
            f"import torch_npu failed: {exc!r}",
            "source /usr/local/Ascend/ascend-toolkit/set_env.sh, then install the torch_npu build "
            "from the CANN compatibility table for this torch (docs/hardware-checklist.md step 1)",
        )
    version = getattr(torch_npu, "__version__", "<unknown version>")
    if getattr(torch, "npu", None) is None:
        return _fail(
            f"torch_npu {version} imported but torch.npu is missing",
            "this torch_npu build does not match the installed torch: reinstall the pair from the "
            "CANN compatibility table (docs/hardware-checklist.md step 1)",
        )
    return _ok(f"torch_npu {version} (torch {getattr(torch, '__version__', '?')}, torch.npu present)")


#: Variables ``/usr/local/Ascend/ascend-toolkit/set_env.sh`` exports, any one of which proves the
#: CANN shell was sourced (``ASCEND_HOME_PATH`` on 8.x+, ``ASCEND_TOOLKIT_HOME``/``ASCEND_OPP_PATH``
#: on the versions the checklist covers).
_CANN_ENV_VARS: Tuple[str, ...] = ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH")


def _check_cann_env() -> Tuple[str, str, str]:
    found = {name: os.environ[name] for name in _CANN_ENV_VARS if os.environ.get(name)}
    if not found:
        return _fail(
            "none of " + ", ".join(_CANN_ENV_VARS) + " is set (CANN has not been sourced)",
            "run `source /usr/local/Ascend/ascend-toolkit/set_env.sh` in this shell (add it to "
            "~/.bashrc), then `npu-smi info` and `tileinfer doctor` again - "
            "docs/hardware-checklist.md step 1.1",
        )
    detail = ", ".join(f"{name}={value}" for name, value in found.items())
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    detail += f" | ASCEND_RT_VISIBLE_DEVICES={visible if visible else '<unset>'}"
    return _ok(detail)


def _check_device() -> Tuple[str, str, str]:
    reason = _torch_npu_unavailable()
    if reason is not None:
        return _skip(
            f"skipped: {reason}",
            "fix the torch_npu check first (docs/hardware-checklist.md step 1)",
        )
    import torch

    npu = torch.npu
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    try:
        available = bool(npu.is_available())
        count = int(npu.device_count())
    except Exception as exc:  # noqa: BLE001
        return _fail(
            f"torch.npu.is_available()/device_count() raised {exc!r}",
            "record the traceback: it names the torch_npu API that is missing "
            "(docs/hardware-checklist.md step 1, failure triage)",
        )
    if not available or count <= 0:
        return _fail(
            f"torch.npu.is_available()={available}, torch.npu.device_count()={count} "
            f"(ASCEND_RT_VISIBLE_DEVICES={visible if visible else '<unset>'})",
            "`npu-smi info` (driver/CANN mismatch, the card is busy, or "
            "ASCEND_RT_VISIBLE_DEVICES is set to an empty string) - "
            "docs/hardware-checklist.md step 1.2",
        )
    try:
        name = str(npu.get_device_name(0))
    except Exception as exc:  # noqa: BLE001
        return _fail(
            f"device_count()={count} but get_device_name(0) raised {exc!r}",
            "record the traceback: TileInferPlatform.get_device_name calls the same API, so this "
            "is one of docs/hardware-checklist.md step 1's `UNVERIFIED_NOTES` questions",
        )
    detail = f"{count} device(s), torch.npu.get_device_name(0)={name}"
    try:
        total = int(npu.get_device_properties(0).total_memory)
        detail += f", total_memory={total}"
    except Exception as exc:  # noqa: BLE001 - informational only; the checklist's fallback applies
        detail += f", total_memory unavailable ({exc!r})"
    return _ok(detail)


def _check_tilelang() -> Tuple[str, str, str]:
    try:
        import tilelang
    except Exception as exc:  # noqa: BLE001
        return _fail(
            f"import tilelang failed: {exc!r}",
            "install the TileLang Ascend wheel (a GitHub release of tile-ai/tilelang-ascend; it is "
            "not on PyPI) - the CPU suite does not need it, the device does: "
            "docs/hardware-checklist.md step 1.3",
        )
    version = getattr(tilelang, "__version__", "<unknown version>")
    try:
        from tileinfer.utils import have_tilelang

        ascend_backend = bool(have_tilelang())
    except Exception:  # noqa: BLE001 - the Ascend probe is a detail, not a crash
        ascend_backend = False
    if not ascend_backend:
        return _fail(
            f"tilelang {version} is importable but has no Ascend backend "
            "(PassConfigKey.TL_ASCEND_AUTO_SYNC is missing)",
            "the plain PyPI tilelang wheel cannot compile Ascend kernels: install the "
            "tilelang-ascend wheel for CANN 9.3.0 (docs/hardware-checklist.md step 1.3)",
        )
    return _ok(f"tilelang {version} (Ascend backend present)")


def _check_tilelang_cache() -> Tuple[str, str, str]:
    raw = (os.environ.get("TILELANG_CACHE_DIR") or "").strip()
    if not raw:
        return _fail(
            "TILELANG_CACHE_DIR is unset",
            "export TILELANG_CACHE_DIR=<local writable dir> and use the same value for "
            "scripts/prewarm-tileinfer-kernels.py and the server, otherwise the serve process "
            "compiles inline (70-100 s, holding the GIL) - docs/hardware-checklist.md step 1.3",
        )
    path = pathlib.Path(raw)
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".tileinfer-doctor-write"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except Exception as exc:  # noqa: BLE001
        return _fail(
            f"TILELANG_CACHE_DIR={raw} is not writable ({exc!r})",
            "point TILELANG_CACHE_DIR at a writable local disk (a network mount makes every "
            "kernel compile on first use) - docs/hardware-checklist.md step 1.3",
        )
    return _ok(f"TILELANG_CACHE_DIR={raw} (created and writable)")


# =============================================================================================
# Checklist step 2: packages and the general plugin
# =============================================================================================


def _check_vllm_version() -> Tuple[str, str, str]:
    try:
        plugin = _plugin()
        probe = plugin._vllm_probe()  # noqa: SLF001 - the same probe status() uses
        supported_range = plugin.SUPPORTED_VLLM_RANGE
    except Exception as exc:  # noqa: BLE001
        return _skip(
            f"skipped: tileinfer.vllm.plugin is not importable ({exc!r})",
            "fix the torch/torch_npu checks first (docs/hardware-checklist.md step 1)",
        )
    if not probe.get("available"):
        return _fail(
            str(probe.get("detail") or "vLLM is not importable"),
            f"install the vLLM line this backend is written against: "
            f"`pip install 'vllm{supported_range}'` (or `pip install -e .[vllm]`) in the venv "
            "that serves - docs/hardware-checklist.md step 2",
        )
    version = str(probe.get("version") or "")
    if not version:
        return _skip(
            str(probe.get("detail") or "vLLM is importable but its distribution version is unknown"),
            f"a source checkout carries no distribution metadata: `pip show vllm` and confirm "
            f"vllm{supported_range} - docs/hardware-checklist.md step 2",
        )
    if not probe.get("supported"):
        return _fail(
            f"vLLM {version} is outside vllm{supported_range}",
            "install the 0.23 line: the platform ABC and the v1 attention contract in this "
            "package are transcribed from it (docs/notes/vllm-api-0.23.md) - "
            "docs/hardware-checklist.md step 2",
        )
    return _ok(f"vLLM {version} (within vllm{supported_range})")


def _check_plugin_status() -> Tuple[str, str, str]:
    try:
        plugin = _plugin()
        probe = plugin._vllm_probe()  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001
        return _skip(
            f"skipped: tileinfer.vllm.plugin is not importable ({exc!r})",
            "fix the torch/torch_npu checks first (docs/hardware-checklist.md step 1)",
        )
    if not probe.get("available"):
        return _skip(
            "skipped: vLLM is not importable, so the general plugin has no engine to install into",
            "fix the vLLM check first (docs/hardware-checklist.md step 2)",
        )
    # The same call vLLM's loader makes, explicitly - otherwise status() only answers "the plugin
    # did not run in this process" (docs/hardware-checklist.md step 10.1).
    installed = bool(plugin.install())
    status = plugin.status()
    plugin_info = status.get("plugin") or {}
    patches = status.get("patches") or {}
    backend = status.get("backend") or {}
    if not installed or not plugin_info.get("installed"):
        return _fail(
            f"plugin.install() returned {installed}: {plugin_info.get('reason', 'no reason')}",
            "grep the serve log for 'TileInfer vLLM plugin'; if the line is absent the entry "
            "point is not installed (`pip install -e .`) - docs/hardware-checklist.md step 2",
        )
    failed = list(patches.get("failed") or [])
    if failed:
        names = ", ".join(str(entry.get("target", entry)) for entry in failed)
        return _fail(
            f"the plugin installed but {len(failed)} patch(es) failed: {names}",
            "a failing patch is a real breakage (a skipped one is not): read the traceback in the "
            "log and the entry in plugin.status()['patches']['failed']",
        )
    return _ok(
        f"installed ({plugin_info.get('reason', '')}); "
        f"{patches.get('summary', 'patch registry unavailable')}; backend: "
        f"{backend.get('detail', 'not reported')}"
    )


def _candidate_segment() -> str:
    """The per-op candidate gate status as one printable tail segment (``""`` when there is none).

    Why this is not behind the vLLM gate
    ------------------------------------
    The candidate table is TileInfer's own data (``tileinfer/vllm/ops/candidates.py`` plus the active
    profile's operator family), not an engine fact: :meth:`CandidateSet.describe` reads records, so
    it answers "which implementation would a device run choose, and what is the recorded reason it is
    off?" on a box with no vLLM - which is exactly where a reader asks it.  So the segment is printed
    on every path of the ``ops.binding`` check, including the skips.

    Deliberately additive: a missing or unimportable candidate table yields ``""`` and never changes
    the check's status, because the verdict (and the exit code) belong to the binding report.
    """
    try:
        from tileinfer.vllm.ops.install import candidate_report

        report = candidate_report()
    except Exception:  # noqa: BLE001 - the extra detail must never decide the verdict
        return ""
    parts = []
    for op in sorted(report):
        rows = (report[op] or {}).get("candidates") or []
        if rows:  # a baseline-only set has nothing to choose, so it is not reported
            described = ", ".join(f"{row['name']}={row['status']}" for row in rows)
            parts.append(f"{op}[{described}]")
    return " | candidates: " + "; ".join(parts) if parts else ""


def _check_ops_binding() -> Tuple[str, str, str]:
    candidates = _candidate_segment()
    try:
        plugin = _plugin()
        probe = plugin._vllm_probe()  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001
        return _skip(
            f"skipped: tileinfer.vllm.plugin is not importable ({exc!r}){candidates}",
            "fix the torch/torch_npu checks first (docs/hardware-checklist.md step 1)",
        )
    if not probe.get("available"):
        return _skip(
            "skipped: vLLM is not importable, so there are no vLLM operators to take over"
            + candidates,
            "fix the vLLM check first (docs/hardware-checklist.md step 2)",
        )
    try:
        from tileinfer.vllm.ops.install import bindings, install_ops
    except Exception as exc:  # noqa: BLE001
        return _skip(
            f"skipped: tileinfer.vllm.ops.install is not importable ({exc!r}){candidates}",
            "fix the torch check first (docs/hardware-checklist.md step 1)",
        )
    report = install_ops()  # idempotent and never raises by contract
    # ``chunk_local_cumsum`` is reported *unbound on purpose* (abi_ok=False: our port is not
    # upstream's drop-in).  Only an unbound op we expected to bind is a failure.
    declined = {binding.op for binding in bindings() if not binding.abi_ok}
    unexpected = [op for op in report.unbound if op not in declined]
    detail = report.summary()
    if report.bound:
        detail += " | bound: " + ", ".join(report.bound)
    if declined:
        detail += " | declined on purpose: " + ", ".join(sorted(declined))
    if unexpected:
        return _fail(
            detail + " | unexpected unbound: " + ", ".join(unexpected) + candidates,
            "a vLLM rename or a missing port: each reason is in "
            "tileinfer.vllm.ops.install.install_ops().details[<op>]; "
            "`python -m pytest tests/test_ops_install.py -q` pins the full list - "
            "docs/hardware-checklist.md step 9",
        )
    return _ok(detail + candidates)


# =============================================================================================
# Checklist step 4: the local model-runner subset
# =============================================================================================

#: The four definitions ``docs/hardware-checklist.md`` step 4.2 greps for: they are the callers of
#: the worker's four hooks, so a runner without them is not the 0.23 file the hooks were written
#: against.
_RUNNER_FUNCS: Tuple[str, ...] = (
    "capture_model",
    "profile_run",
    "initialize_kv_cache",
    "_dummy_run",
)


def _check_runner_source() -> Tuple[str, str, str]:
    api_dir = _repo_root() / "runtime" / "vllm-api"
    if not api_dir.is_dir():
        return _skip(
            f"skipped: {api_dir} is not present (this is not a source checkout)",
            "step 4 applies to a checkout: copy the file per docs/hardware-checklist.md step 4.1",
        )
    path = api_dir / "vllm__v1__worker__gpu_model_runner.py"
    if not path.is_file():
        return _fail(
            f"{path.name} is missing from runtime/vllm-api/",
            "copy it from the installed vLLM wheel (step 4.1 needs no network) - "
            "docs/hardware-checklist.md step 4",
        )
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return _fail(
            f"could not read {path}: {exc!r}",
            "check the file permissions; it is only read, never imported",
        )
    missing = [name for name in _RUNNER_FUNCS if f"def {name}" not in text]
    if missing:
        return _fail(
            f"{path.name} does not define {', '.join(missing)}",
            "the copied file is not v0.23.x (the hooks are transcribed from that version): re-copy "
            "it from v0.23.0 - docs/hardware-checklist.md step 4.2",
        )
    return _ok(f"{path.name} defines all four hooks ({', '.join(_RUNNER_FUNCS)})")


# =============================================================================================
# Checklist step 5: the graph verdict (soft)
# =============================================================================================


def _check_capture_path() -> Tuple[str, str, str]:
    try:
        from tileinfer.vllm import graphs
    except Exception as exc:  # noqa: BLE001
        return _skip(
            f"skipped: tileinfer.vllm.graphs is not importable ({exc!r})",
            "fix the torch check first (docs/hardware-checklist.md step 1)",
        )
    ready, reason = graphs.capture_path_ready()
    if ready:
        return _ok(str(reason))
    return _fail(
        str(reason),
        "not a hard failure: with capture not ready the platform forces enforce_eager=True "
        "(correct but slower). Follow the reason, and check TILEINFER_VLLM_GRAPH_BRIDGE is not 0 "
        "- docs/hardware-checklist.md step 10.1",
    )


# =============================================================================================
# The registry
# =============================================================================================

#: Every check, in print order.  A test replaces this tuple to simulate failures; ``CHECKS`` (the
#: names, for tests and docs) is derived from it below.
_CHECKS: Tuple[CheckSpec, ...] = (
    CheckSpec("python", 1, True, _check_python),
    CheckSpec("torch", 1, True, _check_torch),
    CheckSpec("torch_npu", 1, True, _check_torch_npu),
    CheckSpec("cann_env", 1, True, _check_cann_env),
    CheckSpec("device", 1, True, _check_device),
    CheckSpec("tilelang", 1, True, _check_tilelang),
    CheckSpec("tilelang_cache", 1, True, _check_tilelang_cache),
    CheckSpec("vllm_version", 2, True, _check_vllm_version),
    CheckSpec("plugin.status", 2, True, _check_plugin_status),
    CheckSpec("ops.binding", 2, True, _check_ops_binding),
    CheckSpec("runner_source", 4, True, _check_runner_source),
    CheckSpec("graphs.capture_path_ready", 5, False, _check_capture_path),
)

#: The check names, in order (what a report printed by a healthy doctor contains).
CHECKS: Tuple[str, ...] = tuple(spec.name for spec in _CHECKS)

#: The checks whose failure is a documented degradation rather than a broken environment.
SOFT_CHECKS: Tuple[str, ...] = tuple(spec.name for spec in _CHECKS if not spec.hard)


# =============================================================================================
# Running and printing
# =============================================================================================


def _run_check(spec: CheckSpec) -> Check:
    """Run one check with the *only* ``try``/``except`` a check is allowed to rely on.

    A check that raises, returns the wrong shape or an unknown status becomes a printed ``FAIL``
    that names the exception and points at doctor itself — the report must always be printable,
    because the alternative is a traceback on a box whose whole problem may be a broken import.
    """
    try:
        result = spec.fn()
        if not (isinstance(result, tuple) and len(result) == 3):
            raise TypeError(
                f"check returned {result!r}; expected a (status, detail, hint) triple"
            )
        status, detail, hint = (str(part) for part in result)
        if status not in (PASS, FAIL, SKIP):
            raise ValueError(f"check returned status {status!r}; expected PASS, FAIL or SKIP")
    except Exception as exc:  # noqa: BLE001 - a broken check is a printed FAIL, never a traceback
        status = FAIL
        detail = f"the {spec.name} check raised {exc!r}"
        hint = (
            "this is a bug in `tileinfer doctor` (or in an import it depends on), not in the "
            "environment: run `python -c \"import tileinfer.vllm.doctor\"` and report the traceback"
        )
    if status != PASS and not hint:
        hint = "no fix hint recorded for this check (a doctor bug: every failure names its fix)"
    return Check(
        name=spec.name,
        step=spec.step,
        hard=spec.hard,
        status=status,
        detail=detail,
        hint=hint,
    )


def run_checks(specs: Optional[Sequence[CheckSpec]] = None) -> List[Check]:
    """Run every check (or ``specs``, for a test) and return the results in registry order."""
    return [_run_check(spec) for spec in (_CHECKS if specs is None else specs)]


def _exit_code(checks: Sequence[Check]) -> int:
    """1 iff a *hard* check failed; a soft ``FAIL`` (the eager fallback) keeps the exit code 0."""
    return 1 if any(check.status == FAIL and check.hard for check in checks) else 0


def tail_line(checks: Sequence[Check]) -> str:
    """The last line: which checklist step to start at, and why.

    Ordered by the checklist's own dependency chain (a step-1 failure makes every later answer
    meaningless), so it names the *first* step with a hard failure — the same rule the document's
    "first failure triage" table follows.
    """
    failed_hard = [check for check in checks if check.status == FAIL and check.hard]
    if failed_hard:
        first_step = min(check.step for check in failed_hard)
        names = ", ".join(check.name for check in failed_hard if check.step == first_step)
        return (
            f"Start at docs/hardware-checklist.md step {first_step}: "
            f"{len(failed_hard)} hard check(s) failed, first at step {first_step} ({names})."
        )
    failed_soft = [check for check in checks if check.status == FAIL]
    if failed_soft:
        names = ", ".join(check.name for check in failed_soft)
        return (
            "Start at docs/hardware-checklist.md step 5: every hard check passed, and the open "
            f"item(s) ({names}) only cost the graph path - the platform keeps enforce_eager=True. "
            "Continue at step 6 (weights + KV pool) meanwhile."
        )
    return (
        "Start at docs/hardware-checklist.md step 3: every hard check passed. Run the CPU suite "
        "(`python -m pytest tests -q -p no:cacheprovider`), then continue at step 6 "
        "(weights + KV pool)."
    )


def format_report(checks: Sequence[Check]) -> str:
    """The printable checklist: one section per mirrored step, then the counts and the tail line."""
    lines: List[str] = [_HEADER, ""]
    for step in sorted({check.step for check in checks}):
        lines.append(f"step {step}: {STEP_TITLES.get(step, '')}")
        for check in (check for check in checks if check.step == step):
            severity = "" if check.hard else " (soft)"
            lines.append(f"  [{check.status}] {check.name:<26} {check.detail}{severity}")
            if check.hint:
                lines.append(f"         fix: {check.hint}")
        lines.append("")

    passed = sum(1 for check in checks if check.status == PASS)
    failed = [check for check in checks if check.status == FAIL]
    skipped = sum(1 for check in checks if check.status == SKIP)
    hard = sum(1 for check in failed if check.hard)
    lines.append(
        f"{len(checks)} check(s): {passed} PASS, {len(failed)} FAIL ({hard} hard, "
        f"{len(failed) - hard} soft), {skipped} SKIP"
    )
    lines.append("")
    lines.append(tail_line(checks))
    return "\n".join(lines)


def _emit(text: str) -> None:
    """Print without ever raising on a console code page that cannot render a detail string."""
    try:
        print(text)
    except UnicodeEncodeError:  # pragma: no cover - depends on the host console, not on the code
        print(text.encode("ascii", "replace").decode("ascii"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``[project.scripts] tileinfer``: print the checklist, return 0/1 (2 for a usage error).

    ``--json`` prints the same results as data (for CI logs / a bug report); ``--help`` prints this
    module's docstring.  :data:`SUBCOMMANDS` may be given once, anywhere among the flags - the word
    says *which* command is meant, and there is only one, so ``main(['doctor'])`` and ``main([])``
    are the same run.  Anything else is a usage error.  Never raises: a diagnostic that dies while
    diagnosing has told the user nothing.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    as_json = False
    subcommand: Optional[str] = None
    for arg in args:
        if arg in ("-h", "--help"):
            _emit((__doc__ or "tileinfer doctor").strip())
            return 0
        if arg == "--json":
            as_json = True
            continue
        if arg in SUBCOMMANDS:
            if subcommand is not None:
                _emit(
                    f"tileinfer doctor: the subcommand was given twice ({subcommand!r} then "
                    f"{arg!r}); it is optional and comes at most once"
                )
                return 2
            subcommand = arg  # accepted and otherwise ignored: it names this same command
            continue
        _emit(
            f"tileinfer doctor: unknown argument {arg!r} "
            f"(supported: {', '.join(SUBCOMMANDS)}, --json, --help)"
        )
        return 2

    try:
        checks = run_checks()
        code = _exit_code(checks)
        if as_json:
            _emit(
                json.dumps(
                    {
                        "checks": [check.as_dict() for check in checks],
                        "exit_code": code,
                        "next": tail_line(checks),
                    },
                    indent=2,
                )
            )
        else:
            _emit(format_report(checks))
        return code
    except Exception as exc:  # noqa: BLE001 - the last resort: a checklist, never a traceback
        _emit(f"tileinfer doctor: the checklist itself failed ({exc!r}); nothing was verified")
        return 1


if __name__ == "__main__":  # `python -m tileinfer.vllm.doctor`
    raise SystemExit(main())
