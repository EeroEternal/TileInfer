"""The backend registry is initialised from several threads at once in real engines.

vLLM builds one backend object per attention layer, and the first decode step of every layer asks the
registry for a backend at the same moment.  The import of the bundled backends used to set its
"loaded" flag *before* importing, so a thread that arrived while another was still importing saw an
incomplete registry and raised ``unknown backend 'tilelang-ascend950'; known: ['reference']`` - which
is what happened inside a vLLM EngineCore (KI-2 in docs/known-issues.md).
"""

import threading
import time

import pytest

from tileinfer.attention.backends import base


@pytest.fixture
def fresh_registry(monkeypatch):
    """Start from an unloaded registry, and restore the real one afterwards."""
    saved_registry = dict(base._REGISTRY)
    saved_errors = dict(base._IMPORT_ERRORS)
    saved_loaded = base._BACKENDS_LOADED
    base._REGISTRY.clear()
    base._IMPORT_ERRORS.clear()
    monkeypatch.setattr(base, "_BACKENDS_LOADED", False)
    try:
        yield
    finally:
        base._REGISTRY.clear()
        base._REGISTRY.update(saved_registry)
        base._IMPORT_ERRORS.clear()
        base._IMPORT_ERRORS.update(saved_errors)
        base._BACKENDS_LOADED = saved_loaded


def test_concurrent_first_lookup_never_sees_a_partial_registry(fresh_registry, monkeypatch):
    """Eight threads race to be the first to use a backend; none may see a half-imported registry."""

    class Dummy(base.AttentionBackend):
        name = "tilelang-ascend950"

        def run(self, *args, **kwargs):  # pragma: no cover - the test never runs a step
            raise NotImplementedError

    def fake_load() -> None:
        time.sleep(0.05)
        base.register_backend(Dummy)

    monkeypatch.setattr(base, "_load_builtin_backends", fake_load)

    errors = []
    barrier = threading.Barrier(8)

    def worker() -> None:
        barrier.wait()  # all eight arrive at the registry together
        try:
            base.get_backend("tilelang-ascend950")
        except Exception as exc:  # noqa: BLE001 - the point of the test
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"threads saw an incomplete registry: {errors}"
    assert "tilelang-ascend950" in base.list_backends()


def test_import_failure_is_reported_with_its_reason(fresh_registry, monkeypatch):
    """`unknown backend` should say *why* a bundled backend is missing."""

    def failing_load() -> None:
        base._IMPORT_ERRORS["tilelang-ascend950"] = ImportError("no torch_npu")

    monkeypatch.setattr(base, "_load_builtin_backends", failing_load)

    with pytest.raises(ValueError) as excinfo:
        base.get_backend("tilelang-ascend950")

    message = str(excinfo.value)
    assert "unknown backend" in message
    assert "no torch_npu" in message
