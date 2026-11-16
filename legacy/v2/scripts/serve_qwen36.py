"""Compatibility alias: ``scripts/serve_tileinfer.py`` pinned to the Qwen3.6 profile.

The launcher arithmetic lives in ``serve_tileinfer.py`` and nowhere else.  This file exists so the
documented ``serve_qwen36.py`` command keeps working (and so imports/tests that load this module
still see ``main``, ``_build_plan``, ``_model_shape``, ...): it executes the generic launcher's
source **in this module's namespace**, then sets the one model-specific bit - the profile name.

That is deliberately not a wrapper class or a second ``main``: a wrapper would shadow the generic
module's globals, and a monkeypatch of ``serve_qwen36._vllm_gate`` (the tests do exactly that) would
not reach the function doing the work.  Executing the source here keeps one implementation and one
namespace, so every helper is the same object under both names.
"""

from __future__ import annotations

import pathlib

#: The profile this entry point serves.  The generic launcher resolves by architecture when this is
#: unset; pinning it here makes the command's intent explicit (and an unknown architecture in the
#: model config still fails in the profile layer, not silently here).
PROFILE_NAME = "qwen3.6"

_GENERIC = pathlib.Path(__file__).with_name("serve_tileinfer.py")
if not _GENERIC.is_file():  # pragma: no cover - the file is part of the checkout
    raise ImportError(
        f"serve_qwen36.py is an alias for {_GENERIC.name}, which is missing; restore the generic "
        "launcher (scripts/serve_tileinfer.py) - this file intentionally contains no arithmetic."
    )

_GLOBALS = globals()
_THIS_NAME = _GLOBALS["__name__"]
# Suppress the generic file's own ``__main__`` guard while executing it: the alias decides when to
# run (below), after PROFILE_NAME is in place.
_GLOBALS["__name__"] = "serve_tileinfer_alias"
try:
    exec(compile(_GENERIC.read_text(encoding="utf-8"), str(_GENERIC), "exec"), _GLOBALS)
finally:
    _GLOBALS["__name__"] = _THIS_NAME

if __name__ == "__main__":
    raise SystemExit(main())
