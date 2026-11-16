"""CPU tests for the patch registry (no vLLM needed - it uses a fake module)."""

from __future__ import annotations

import sys
import types

import pytest

from tileinfer.vllm.compat import patches as P


@pytest.fixture(autouse=True)
def fake_module():
    module = types.ModuleType("tileinfer_fake_vllm")

    def original(*args, **kwargs):
        return ("original", args, kwargs)

    class Holder:
        value = 1

    module.fn = original
    module.Holder = Holder
    # Exists so a patch whose *body* raises is distinguishable from one whose target is gone.
    module.boom = lambda: None
    sys.modules["tileinfer_fake_vllm"] = module
    P.clear()
    try:
        yield module
    finally:
        sys.modules.pop("tileinfer_fake_vllm", None)
        P.clear()


def test_resolve_function_and_class_attribute():
    owner, name = P.resolve("tileinfer_fake_vllm.fn")
    assert owner is sys.modules["tileinfer_fake_vllm"] and name == "fn"
    owner, name = P.resolve("tileinfer_fake_vllm.Holder.value")
    assert name == "value" and isinstance(owner, type)


def test_resolve_rejects_bad_targets():
    with pytest.raises(ValueError):
        P.resolve("nocolon")
    with pytest.raises(ImportError):
        P.resolve("module_that_does_not_exist_anywhere.attr")


def test_wrap_attr_keeps_the_original_reachable():
    P.wrap_attr("tileinfer_fake_vllm.fn", lambda original: (lambda *a, **k: ("wrapped", original(*a, **k))))
    assert sys.modules["tileinfer_fake_vllm"].fn(1, x=2) == (
        "wrapped",
        ("original", (1,), {"x": 2}),
    )


def test_install_all_reports_applied_skipped_and_failed():
    applied = []

    P.register(
        P.Patch(
            target="tileinfer_fake_vllm.fn",
            why="test",
            apply=lambda: (P.set_attr("tileinfer_fake_vllm.fn", "patched"), applied.append(True)),
        )
    )
    P.register(
        P.Patch(target="tileinfer_fake_vllm.ghost", why="gone upstream", apply=lambda: None)
    )
    P.register(
        P.Patch(
            target="tileinfer_fake_vllm.boom",
            why="raises",
            apply=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
    )
    report = P.install_all()
    assert sys.modules["tileinfer_fake_vllm"].fn == "patched"
    assert applied == [True]
    assert [t for t, _ in report.skipped] == ["tileinfer_fake_vllm.ghost"]
    assert [t for t, _ in report.failed] == ["tileinfer_fake_vllm.boom"]
    assert not report.ok
    assert "1 applied" in report.summary()


def test_strict_mode_turns_skips_into_failures():
    P.register(P.Patch(target="tileinfer_fake_vllm.ghost", why="gone", apply=lambda: None))
    report = P.install_all(strict=True)
    assert report.skipped == []
    assert [t for t, _ in report.failed] == ["tileinfer_fake_vllm.ghost"]


def test_duplicate_target_is_rejected():
    P.register(P.Patch(target="tileinfer_fake_vllm.fn", why="a", apply=lambda: None))
    with pytest.raises(ValueError, match="duplicate patch"):
        P.register(P.Patch(target="tileinfer_fake_vllm.fn", why="b", apply=lambda: None))


def test_describe_is_plain_data():
    P.register(
        P.Patch(
            target="tileinfer_fake_vllm.fn",
            why="because",
            apply=lambda: None,
            since="0.23",
            remove_when="upstream uses current_platform",
            upstream_ref="#1234",
        )
    )
    described = P.describe()
    assert described[0]["target"] == "tileinfer_fake_vllm.fn"
    assert described[0]["remove_when"] == "upstream uses current_platform"
    assert described[0]["upstream_ref"] == "#1234"
