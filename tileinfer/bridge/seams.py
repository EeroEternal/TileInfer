"""The six access-point mechanisms, implemented as *seams* over the vendor platform.

Rules this module follows, from ``docs/integration.md``:

* **No source edits.**  Everything here is an entry-point call, a table write or a method
  wrapper applied at process start.
* **No platform slot.**  The platform plugin stays vllm-ascend's; we only wrap its decision
  methods after it has been chosen.
* **Idempotent.**  Each seam marks the patched object, so the second process or the second
  call is a no-op.
* **Never raises out.**  A seam that cannot install returns ``False`` with a reason; the
  installer logs it and the vendor path continues unchanged.
* **Not the hot path.**  Only the graph counters run per step, and they increment integers.

The six mechanisms and where each lives:

==================  =====================================================
``plugin_entry``    ``pyproject.toml`` -> :func:`tileinfer.bridge.install.install`
``impl_table``      :func:`install_impl_table` (CustomOp / dispatch table)
``model_table``     :func:`install_model_table` (ModelRegistry)
``attn_select``     :func:`install_attention_selector` (platform hook wrapper)
``method_wrap``     :func:`install_method_wrappers` (ordinary functions)
``graph_hook``      :func:`install_graph_hook` (dispatcher + capture counters)
==================  =====================================================
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from . import metrics


class SeamResult:
    """Outcome of one seam: installed or not, and why."""

    __slots__ = ("name", "ok", "detail")

    def __init__(self, name: str, ok: bool, detail: str = "") -> None:
        self.name = name
        self.ok = ok
        self.detail = detail

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SeamResult({self.name!r}, ok={self.ok}, detail={self.detail!r})"


def install_graph_hook() -> SeamResult:
    """Count captures and steady-state decode fallbacks without changing dispatch."""
    try:
        from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
    except Exception as exc:  # pragma: no cover - vLLM always present when called
        return SeamResult("graph_hook", False, f"vllm dispatcher unavailable: {exc}")

    if getattr(CudagraphDispatcher, "_tileinfer_graph_hooked", False):
        return SeamResult("graph_hook", True, "already installed")

    original_init = CudagraphDispatcher.initialize_cudagraph_keys
    original_dispatch = CudagraphDispatcher.dispatch

    def initialize_cudagraph_keys(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_init(self, *args, **kwargs)
        try:
            keys = {
                getattr(mode, "name", str(mode)): len(values)
                for mode, values in getattr(self, "cudagraph_keys", {}).items()
            }
            mode_name = getattr(getattr(self, "cudagraph_mode", None), "name", "")
            metrics.record_capture(keys, mode_name)
        except Exception:
            metrics.COUNTERS.errors += 1
        return result

    def dispatch(self: Any, num_tokens: int, uniform_decode: bool = False, *args: Any, **kwargs: Any) -> Any:
        mode, descriptor = original_dispatch(self, num_tokens, uniform_decode, *args, **kwargs)
        try:
            valid_modes = kwargs.get("valid_modes")
            if valid_modes is None and len(args) >= 3:
                valid_modes = args[2]
            graph_allowed = True
            if valid_modes is not None:
                graph_allowed = any(
                    getattr(item, "name", "") in {"FULL", "PIECEWISE"} for item in valid_modes
                )
            metrics.record_dispatch(
                bool(uniform_decode), getattr(mode, "name", str(mode)), graph_allowed
            )
        except Exception:
            metrics.COUNTERS.errors += 1
        return mode, descriptor

    CudagraphDispatcher.initialize_cudagraph_keys = initialize_cudagraph_keys
    CudagraphDispatcher.dispatch = dispatch

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        if not getattr(GPUModelRunner, "_tileinfer_capture_hooked", False):
            original_capture = GPUModelRunner.capture_model

            def capture_model(self: Any, *args: Any, **kwargs: Any) -> Any:
                result = original_capture(self, *args, **kwargs)
                graphs = 0
                try:
                    descs = self.cudagraph_dispatcher.get_capture_descs()
                    graphs = sum(len(group) for _, group in descs)
                except Exception:
                    metrics.COUNTERS.errors += 1
                metrics.record_capture_done(graphs)
                return result

            GPUModelRunner.capture_model = capture_model
            GPUModelRunner._tileinfer_capture_hooked = True
    except Exception as exc:
        return SeamResult("graph_hook", False, f"capture wrapper unavailable: {exc}")

    CudagraphDispatcher._tileinfer_graph_hooked = True
    return SeamResult("graph_hook", True, "dispatch + capture counters installed")


def install_attention_selector(plan: Any) -> SeamResult:
    """Wrap the vendor platform's backend hook; return the planned class path.

    With a ``vendor`` selection this returns exactly what the vendor returned - the seam is
    installed and observable, but numerics are untouched.  A non-vendor selection is only
    returned when the plan names one explicitly.
    """
    try:
        from vllm_ascend.platform import NPUPlatform
    except Exception as exc:
        return SeamResult("attn_select", False, f"vllm-ascend platform unavailable: {exc}")

    if getattr(NPUPlatform, "_tileinfer_attn_hooked", False):
        return SeamResult("attn_select", True, "already installed")

    original = NPUPlatform.get_attn_backend_cls.__func__

    def choose(capability: str, vendor_path: str) -> str:
        selection = plan.selection(capability) if plan is not None else None
        if selection is None or selection.source == "vendor":
            return vendor_path
        return selection.impl or vendor_path

    def get_attn_backend_cls(cls: Any, selected_backend: Any, attn_selector_config: Any, num_heads: Optional[int] = None) -> str:
        vendor_path = original(cls, selected_backend, attn_selector_config, num_heads)
        chosen = choose("attention.full", vendor_path)
        metrics.record_attention_backend(chosen)
        return chosen

    NPUPlatform.get_attn_backend_cls = classmethod(get_attn_backend_cls)
    NPUPlatform._tileinfer_attn_hooked = True
    metrics.record_attention_backend("vendor:deferred")
    return SeamResult("attn_select", True, "platform hook wrapped (vendor path unchanged)")


def install_config_gate(plan: Any) -> SeamResult:
    """Abort startup if the live checkpoint is not the model the plan was built for.

    The plan is computed from a checkpoint's ``config.json`` at deploy time, but the serving
    command names a path independently.  A mismatch here is the one case where the bridge must
    fail *closed*: injecting a plan for a different architecture would silently select
    implementations for the wrong layer shapes.  The check happens before the model is loaded
    (``check_and_update_config`` runs before weight loading), so the failure is cheap.
    """
    if plan is None:
        return SeamResult("config_gate", False, "no plan")
    try:
        from vllm_ascend.platform import NPUPlatform
    except Exception as exc:
        return SeamResult("config_gate", False, f"vllm-ascend platform unavailable: {exc}")
    if getattr(NPUPlatform, "_tileinfer_config_gate", False):
        return SeamResult("config_gate", True, "already installed")

    original = NPUPlatform.check_and_update_config.__func__

    def check_and_update_config(cls: Any, vllm_config: Any) -> None:
        model_config = getattr(vllm_config, "model_config", None)
        archs = list(getattr(model_config, "architectures", None) or [])
        if archs and plan.model.arch not in archs:
            raise RuntimeError(
                "TileInfer plan/model mismatch: plan is for "
                f"{plan.model.arch!r} ({plan.model.name}), live checkpoint declares {archs!r}; "
                "rebuild the plan with `tileinfer plan build --model <path>`"
            )
        return original(cls, vllm_config)

    NPUPlatform.check_and_update_config = classmethod(check_and_update_config)
    NPUPlatform._tileinfer_config_gate = True
    return SeamResult("config_gate", True, f"arch={plan.model.arch}")


def install_impl_table(plan: Any) -> SeamResult:
    """Apply the plan's non-vendor operator selections through vLLM's CustomOp table.

    A vendor selection is a no-op: the vendor already installed its own implementations and
    re-registering the same class would be the "same-name assertion" trap from
    ``docs/integration.md`` §6.  An empty table is the expected steady state on Qwen3.6.
    """
    if plan is None:
        return SeamResult("impl_table", False, "no plan")
    non_vendor = [s for s in plan.selections if s.source != "vendor"]
    if not non_vendor:
        return SeamResult("impl_table", True, "no non-vendor selections (vendor table kept)")
    try:
        from vllm.model_executor.custom_op import CustomOp  # noqa: F401
    except Exception as exc:
        return SeamResult("impl_table", False, f"CustomOp unavailable: {exc}")
    # Providers are added here once they expose a vendor-compatible class form.  Refusing to
    # guess is deliberate: a wrong entry here is a silent numerical bug.
    return SeamResult(
        "impl_table",
        False,
        f"{len(non_vendor)} non-vendor selections have no registered class form yet",
    )


def install_model_table(plan: Any) -> SeamResult:
    """Reserve the model-table seam; Qwen3.6 is a vendor-served model, so it is empty."""
    return SeamResult("model_table", True, "no model replacement requested")


def install_method_wrappers(plan: Any) -> SeamResult:
    """Ordinary-method seam.  Currently only the dispatcher, which is the graph hook."""
    return SeamResult("method_wrap", True, "covered by graph_hook; no extra wrappers")


def install_all(plan: Any) -> Dict[str, Any]:
    """Install the seams named by the plan.  Returns a report; never raises."""
    report: Dict[str, Any] = {"seams": [], "ok": True}
    results = [
        install_config_gate(plan),
        install_graph_hook(),
        install_attention_selector(plan),
        install_impl_table(plan),
        install_model_table(plan),
        install_method_wrappers(plan),
    ]
    report["seams"] = [r.as_dict() for r in results]
    report["ok"] = all(
        r.ok for r in results if r.name in {"config_gate", "graph_hook", "attn_select"}
    )
    return report
