"""``tileinfer doctor`` - the one command that answers "is this checkout deployable?".

It is deliberately read-only with respect to the device: no model load, no KV allocation, no
graph capture.  It checks the things that are cheap to get wrong and expensive to discover
during a run:

1. the version triangle (Python / vLLM / vllm-ascend) and the platform's device type;
2. the plan: parseable, version-gated, servable, within the access-point budget;
3. the layered import rule: only ``tileinfer/bridge`` may import ``vllm_ascend``;
4. a dry install of the bridge in this throwaway process, and the seams it reports;
5. the last metrics snapshot, if a server has run on this box.

Exit code is 0 only when no check failed.  ``--json`` is for CI and for pasting evidence
into ``docs/notes/``.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import pathlib
import platform
import sys
from typing import Any, Dict, List, Optional, Tuple

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"

#: HCCL's default rankTable/rootinfo path.  If it exists, HCCL treats it as an input file;
#: an empty *directory* there makes TP>1 fail with ``RootInfoDetect failed`` before any
#: network exchange.  Checked by ``doctor`` so the failure is explained before a run.
HCCL_ROOTINFO = "/etc/hccl_rootinfo.json"


def hccl_rootinfo_problem(path: Optional[str] = None) -> str:
    """Return a human-readable problem with the HCCL rootinfo path, or ``""``.

    Pure enough to unit-test: it only inspects the given path.  A missing path is the
    healthy case (HCCL falls back to its own rootinfo exchange).
    """
    target = path or os.environ.get("TILEINFER_HCCL_ROOTINFO", HCCL_ROOTINFO)
    if not os.path.exists(target):
        return ""
    if os.path.isdir(target):
        return (
            f"{target} is a directory; HCCL reads it as a rankTable file and TP>1 fails "
            "with RootInfoDetect. Remove/fix it on the host (do not encode this in code)."
        )
    if not os.path.isfile(target):
        return f"{target} is not a regular file; HCCL cannot use it as a rankTable file."
    try:
        if os.path.getsize(target) == 0:
            return f"{target} is an empty file; HCCL cannot parse it as a rankTable."
        with open(target, encoding="utf-8") as fh:
            json.load(fh)
    except Exception as exc:
        return f"{target} is not valid JSON ({type(exc).__name__}: {exc})."
    return ""


class Report:
    def __init__(self) -> None:
        self.checks: List[Dict[str, str]] = []

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append({"check": name, "status": status, "detail": detail})

    @property
    def failed(self) -> bool:
        return any(c["status"] == FAIL for c in self.checks)

    def as_dict(self) -> Dict[str, Any]:
        return {"ok": not self.failed, "checks": self.checks}


def _scan_vendor_imports(root: pathlib.Path) -> List[Tuple[str, List[str]]]:
    found: List[Tuple[str, List[str]]] = []
    for pattern in ("tileinfer/**/*.py",):
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError:
                continue
            modules: List[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules.extend(a.name for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules.append(node.module)
            hits = sorted({m for m in modules if "vllm_ascend" in m})
            if hits:
                found.append((relative, hits))
    return found


def _check_versions(report: Report) -> None:
    import importlib.metadata

    def version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except Exception:
            return "missing"

    report.add("python", PASS if sys.version_info >= (3, 9) else FAIL, platform.python_version())
    for dist, want in (("vllm", "0.23."), ("vllm_ascend", "0.23.")):
        value = version(dist)
        if value == "missing":
            report.add(dist, FAIL, "not installed (pip install -e '.[vllm]' in the v3 venv)")
        elif not value.startswith(want):
            report.add(dist, FAIL, f"{value} (the bridge targets {want}x)")
        else:
            report.add(dist, PASS, value)
    try:
        from ..resolve.probe import probe as probe_environment

        device = probe_environment().device_type or "unknown"
        report.add("device_type", PASS if device in {"A2", "A3", "A5"} else FAIL, str(device))
    except Exception as exc:
        report.add("device_type", FAIL, f"cannot probe the device type: {exc}")


def _check_plan(report: Report) -> Optional[Any]:
    from ..contracts.plan import Plan
    from ..manifests import find_manifest
    from ..resolve.decide import decide
    from ..resolve.probe import probe

    path = os.environ.get("TILEINFER_PLAN")
    try:
        if path:
            plan = Plan.load(path)
            source = path
        else:
            plan, _ = decide(find_manifest(os.environ.get("TILEINFER_MANIFEST", "qwen3_6")), probe())
            source = "auto (TILEINFER_PLAN unset)"
        report.add("plan", PASS, f"{plan.summary()} [{source}]")
        return plan
    except Exception as exc:
        report.add("plan", FAIL, f"{type(exc).__name__}: {exc}")
        return None


def _check_support(report: Report, plan: Optional[Any]) -> None:
    if plan is None:
        report.add("model_support", SKIP, "no plan")
        return
    support = dict(getattr(plan, "support", {}) or {})
    if not support:
        report.add("model_support", SKIP, "declaration has no support metadata")
        return
    text = str(support.get("text", "unknown"))
    status = PASS if text in {"verified", "partial"} else FAIL
    detail = ", ".join(f"{key}={value}" for key, value in sorted(support.items()))
    report.add("model_support", status, detail)


def _check_parallel(report: Report, plan: Optional[Any]) -> None:
    """Gate the plan's parallel axes against the live environment and its selections.

    A plan built for single-card must not be injected into a sharded run: a single-only
    provider that receives locally sharded tensors produces wrong numbers, not an exception.
    """
    if plan is None:
        report.add("parallel", SKIP, "no plan")
        return
    try:
        from ..bridge.runtime_gate import check_plan
        from ..contracts.capabilities import parallel_supports
        from ..resolve.probe import probe
    except Exception as exc:  # pragma: no cover - defensive
        report.add("parallel", FAIL, f"cannot import parallel helpers: {exc}")
        return

    env = probe()
    live_axes = env.parallel_axes()
    plan_axes = tuple(getattr(plan, "parallel", ("single",)))
    reason = check_plan(plan, {}, parallel_axes=live_axes, check_versions=False)
    if reason:
        report.add("parallel", FAIL, reason)
        return

    bad = []

    # Route-O models ship their own class, so their declared axes matter even when every
    # capability row is vendor-served (MuseGlimmer is exactly this case).
    try:
        from ..manifests import find_manifest

        manifest = find_manifest(plan.model.manifest)
    except Exception:
        manifest = None
    if manifest is not None and getattr(manifest, "model_impl", ""):
        if not parallel_supports(manifest.parallel, live_axes):
            bad.append(
                f"model {manifest.manifest_id} declares parallel={list(manifest.parallel)}"
            )
    if bad:
        report.add(
            "parallel",
            FAIL,
            f"parallel={list(live_axes)} is not covered by: {bad}",
        )
    else:
        report.add("parallel", PASS, f"plan={list(plan_axes)} live={list(live_axes)}")


def _check_hccl_rootinfo(report: Report) -> None:
    """Before a sharded run, verify HCCL's rootinfo path is not a directory/empty file."""
    try:
        from ..resolve.probe import probe

        env = probe()
    except Exception as exc:
        report.add("hccl_rootinfo", FAIL, f"cannot probe the environment: {exc}")
        return
    if all(
        size == 1
        for size in (env.tp_size, env.ep_size, env.dp_size, env.pp_size)
    ):
        report.add("hccl_rootinfo", SKIP, "single-card run; HCCL rootinfo not required")
        return
    problem = hccl_rootinfo_problem()
    if problem:
        report.add("hccl_rootinfo", FAIL, problem)
    else:
        report.add(
            "hccl_rootinfo",
            PASS,
            f"{os.environ.get('TILEINFER_HCCL_ROOTINFO', HCCL_ROOTINFO)} is absent or valid",
        )


def _check_imports(report: Report) -> None:
    root = pathlib.Path(__file__).resolve().parents[2]
    found = _scan_vendor_imports(root)
    offenders = [(p, m) for p, m in found if not p.startswith("tileinfer/bridge/")]
    if offenders:
        report.add("layer_guard", FAIL, f"vllm_ascend imported outside bridge/: {offenders}")
    else:
        report.add("layer_guard", PASS, f"{len(found)} bridge module(s) touch the vendor package")


def _check_bridge(report: Report, plan: Optional[Any]) -> None:
    if plan is None:
        report.add("bridge", SKIP, "no plan to install")
        return
    try:
        import tempfile

        # Dry install: point the counters at a throwaway file so the doctor's own process
        # cannot masquerade as a server that captured no graphs.
        os.environ["TILEINFER_METRICS_PATH"] = tempfile.mktemp(prefix="tileinfer-doctor-")
        from ..bridge.install import install, status

        install()
        data = status()
        if not data.get("installed"):
            report.add("bridge", FAIL, f"not installed: {data.get('skipped')}")
            return
        seams = {s["name"]: s for s in data.get("seams", [])}
        required = ("graph_hook", "attn_select")
        bad = [name for name in required if not seams.get(name, {}).get("ok")]
        if bad:
            report.add("bridge", FAIL, f"seams not installed: {bad}")
        else:
            report.add("bridge", PASS, f"plan={data.get('plan_id')} seams={sorted(seams)}")
    except Exception as exc:
        report.add("bridge", FAIL, f"{type(exc).__name__}: {exc}")


def _check_metrics(report: Report, path: Optional[str] = None) -> None:
    path = path or os.environ.get("TILEINFER_METRICS_PATH", "/tmp/tileinfer-v3-metrics.json")
    if not os.path.exists(path):
        report.add("graph_metrics", SKIP, f"no snapshot at {path} (no server has run yet)")
        return
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        captured = int(data.get("captured_graphs", 0))
        fallbacks = int(data.get("decode_fallbacks", 0))
        decode = int(data.get("decode_calls", 0))
        if captured <= 0:
            report.add("graph_metrics", FAIL, f"captured_graphs={captured} in {path}")
        elif fallbacks > 0:
            report.add("graph_metrics", FAIL, f"{fallbacks}/{decode} decode steps fell back")
        else:
            report.add(
                "graph_metrics",
                PASS,
                f"captured={captured} decode={decode} fallbacks=0 ({path})",
            )
    except Exception as exc:
        report.add("graph_metrics", FAIL, f"{type(exc).__name__}: {exc}")


def _collect(report: Report, metrics_path: str) -> None:
    _check_versions(report)
    plan = _check_plan(report)
    _check_support(report, plan)
    _check_parallel(report, plan)
    _check_hccl_rootinfo(report)
    _check_imports(report)
    _check_bridge(report, plan)
    _check_metrics(report, metrics_path)


def run(json_output: bool = False) -> int:
    report = Report()
    metrics_path = os.environ.get("TILEINFER_METRICS_PATH", "/tmp/tileinfer-v3-metrics.json")
    if json_output:
        # vLLM installs a stdout log handler at import time; run the checks with stdout
        # redirected so `--json` is parseable, then print the document on the real stdout.
        import contextlib

        with contextlib.redirect_stdout(sys.stderr):
            _collect(report, metrics_path)
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        _collect(report, metrics_path)
        width = max(len(c["check"]) for c in report.checks)
        for check in report.checks:
            print(f"{check['status']:<4} {check['check']:<{width}}  {check['detail']}")
        print("OK" if not report.failed else "FAILED")
    return 0 if not report.failed else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="tileinfer doctor", description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)
    return run(json_output=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
