"""The ordered actor pipeline for a route-O model.

Before this module, ``install()`` ran the config adapter, ``install_model_table`` ran
``register_config``/``register_models``, and the parser was a third code path, each with its
own error policy.  A model adapter is now one ordered, idempotent action:

    1. config adapter
    2. model.register_config
    3. model.register_models
    4. parser.register

Every step returns the common ``InstallResult``; any failure fails the whole adapter and the
caller fails closed.  ``install_model_table`` and the plugin entry point both use this same
pipeline, so a direct seam test and a real boot cannot disagree.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config_adapter import install_config_adapter
from .results import InstallResult

_INSTALLED: Dict[Tuple[str, str, str, str], InstallResult] = {}


def _merge(target: InstallResult, step: InstallResult) -> None:
    target.installed.extend(step.installed)
    target.skipped.extend(step.skipped)
    target.failed.update(step.failed)
    if step.detail:
        target.detail = (
            f"{target.detail}; {step.detail}" if target.detail else step.detail
        )
    if not step.ok:
        target.ok = False


def _run_named(name: str, spec: str, actor: str = "register") -> InstallResult:
    module_name, _, chosen = spec.partition(":")
    chosen = chosen or actor
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        return InstallResult(
            name=name,
            ok=False,
            detail=f"cannot import {module_name!r}: {type(exc).__name__}: {exc}",
        )
    fn = getattr(module, chosen, None)
    if not callable(fn):
        return InstallResult(
            name=name,
            ok=False,
            detail=f"{module_name!r} has no callable {chosen!r}",
        )
    try:
        value = fn()
    except Exception as exc:  # noqa: BLE001
        return InstallResult(
            name=name,
            ok=False,
            detail=f"{spec}.{chosen} raised {type(exc).__name__}: {exc}",
        )
    result = InstallResult.from_mapping(name, value, detail=f"{module_name}.{chosen}")
    if not result.ok:
        result.detail = f"{name} reported failure: {result.detail or value!r}"
    return result


@dataclass(frozen=True)
class ModelAdapters:
    config: str = ""
    model: str = ""
    parser: str = ""
    parser_impl: str = ""

    @classmethod
    def from_plan(cls, plan: Any) -> "ModelAdapters":
        return cls(
            config=str(getattr(plan, "config_impl", "") or ""),
            model=str(getattr(plan, "model_impl", "") or ""),
            parser=str(getattr(plan, "reasoning_parser", "") or ""),
            parser_impl=str(getattr(plan, "reasoning_parser_impl", "") or ""),
        )

    @property
    def requested(self) -> bool:
        return bool(self.config or self.model or (self.parser and self.parser_impl))

    def install(self) -> InstallResult:
        result = InstallResult(name="model_adapters")
        if not self.requested:
            result.detail = "no model replacement requested"
            return result
        key = (self.config, self.model, self.parser, self.parser_impl)
        cached = _INSTALLED.get(key)
        if cached is not None:
            return InstallResult(
                name=cached.name,
                ok=cached.ok,
                installed=cached.installed,
                skipped=cached.skipped,
                failed=cached.failed,
                detail="already installed",
            )

        if self.config:
            _merge(result, install_config_adapter(self.config))
            if not result.ok:
                _INSTALLED[key] = result
                return result

        if self.model:
            try:
                module = importlib.import_module(self.model)
            except Exception as exc:  # noqa: BLE001
                result.ok = False
                result.detail = f"cannot import model_impl {self.model!r}: {exc!r}"
                _INSTALLED[key] = result
                return result
            for actor in ("register_config", "register_models"):
                fn = getattr(module, actor, None)
                if fn is None:
                    continue
                step = _run_named(f"{self.model}.{actor}", self.model, actor)
                _merge(result, step)
                if not result.ok:
                    _INSTALLED[key] = result
                    return result

        if self.parser and self.parser_impl:
            step = self._register_parser()
            _merge(result, step)
            if not result.ok:
                _INSTALLED[key] = result
                return result

        result.detail = result.detail or "model adapters installed"
        _INSTALLED[key] = result
        return result

    def _register_parser(self) -> InstallResult:
        try:
            from vllm.reasoning import ReasoningParserManager

            module_name, _, class_name = self.parser_impl.partition(":")
            parser_cls = getattr(importlib.import_module(module_name), class_name)
            ReasoningParserManager.register_module(self.parser, module=parser_cls)
        except Exception as exc:  # noqa: BLE001
            return InstallResult(
                name="parser_register",
                ok=False,
                detail=f"parser registration failed: {exc!r}",
            )
        return InstallResult(
            name="parser_register",
            ok=True,
            installed=[self.parser],
            detail=f"parser {self.parser!r} registered",
        )


def install_model_adapters(plan: Any) -> InstallResult:
    return ModelAdapters.from_plan(plan).install()


def reset_model_adapters() -> None:
    _INSTALLED.clear()


__all__ = [
    "ModelAdapters",
    "install_model_adapters",
    "reset_model_adapters",
]
