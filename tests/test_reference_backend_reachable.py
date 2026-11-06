"""The reference backend must be *reachable*, not just registered.

Found on the device (§44.6): with ``TILEINFER_ATTN_BACKEND=reference`` every request died with
``NameError: name 'RaggedMetadata' is not defined`` from ``ReferenceBackend.run``.  The module used
the name in the branch that rebuilds metadata from the step's ragged triple but never imported it.

That branch is not exotic: it is taken whenever a caller passes ``kv_indptr``/``kv_indices``/
``kv_last_page_len`` explicitly, which is exactly what the serving path does -- and the reference
backend is the **degraded** path the engine falls back to when a kernel plan is not ready or failed.
A missing import there turns "degrade to slow but correct" into "crash the engine".

These assertions are deliberately about the module's namespace rather than about behaviour: the defect
was a name that the code relies on and the import block did not provide, and a behavioural test only
catches it if it happens to execute that exact branch.
"""

from __future__ import annotations

import inspect

import torch  # noqa: F401  (imported so the module graph matches a real run)

from tileinfer.attention.backends import reference as reference_backend
from tileinfer.metadata import RaggedMetadata


def test_reference_backend_imports_the_metadata_type_it_constructs():
    assert reference_backend.RaggedMetadata is RaggedMetadata, (
        "ReferenceBackend.run builds a RaggedMetadata in its explicit-ragged branch; the name must "
        "be imported, or every degraded decode step raises NameError (measured on hardware)"
    )


def test_every_global_the_run_body_loads_resolves():
    """Catch the same class of defect for any *other* global in ``run``.

    Uses the bytecode's ``LOAD_GLOBAL`` operands rather than ``co_names``: the latter also contains
    attribute names (``meta.kv_indptr`` contributes both ``meta`` and ``kv_indptr``), which are not
    module globals at all.
    """
    import builtins
    import dis

    code = reference_backend.ReferenceBackend.run.__code__
    module_globals = vars(reference_backend)
    loaded = {
        instr.argval if isinstance(instr.argval, str) else instr.argval[0]
        for instr in dis.get_instructions(reference_backend.ReferenceBackend.run)
        if instr.opname == "LOAD_GLOBAL"
    }
    unresolved = [name for name in sorted(loaded) if name not in module_globals and not hasattr(builtins, name)]
    assert not unresolved, f"{unresolved} are loaded by ReferenceBackend.run but not defined"
    assert "RaggedMetadata" in loaded, (
        "the explicit-ragged branch must load RaggedMetadata; if this assertion fails the branch "
        "was refactored away and this test needs to follow it"
    )


def test_run_signature_still_accepts_the_ragged_triple():
    """The serving path passes these by name; a rename would break it silently."""
    params = inspect.signature(reference_backend.ReferenceBackend.run).parameters
    for name in ("q", "k_cache", "v_cache", "out", "kv_indptr", "kv_indices", "kv_last_page_len"):
        assert name in params, f"ReferenceBackend.run lost its {name!r} parameter"
