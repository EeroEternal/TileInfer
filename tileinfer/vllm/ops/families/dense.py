"""The ``dense`` operator family: **no takeovers** — a dense attention-only model.

MuseGlimmer's text decoder is a standard dense transformer (GQA attention, SwiGLU MLP, RMSNorm
norms).  Every layer op it uses has a working torch path on NPU through vLLM's own layer
implementations, so this family promises an empty takeover table: ``install_ops`` binds nothing
and the model runs on the stock layer code, exactly like the pre-candidate baseline.

It exists so a :class:`~tileinfer.vllm.profiles.base.ModelProfile` can name a real family
(``op_family="dense"``) — ``install.py`` refuses an empty/unknown ``op_family`` — while keeping
the "adding a model family is a new file in this package" contract: ``required_names()`` and
``bind()`` as data, no edit to the installer.
"""

from __future__ import annotations

from typing import Any, List, Tuple

__all__ = ["NAME", "BASELINE_NAME", "required_names", "bind"]

#: Registry name; a ``ModelProfile.op_family`` carries exactly this string.
NAME = "dense"

#: The baseline candidate's name inside every set (unused here: the table is empty).
BASELINE_NAME = "torch"


def required_names() -> Tuple[str, ...]:
    """No takeovers: the dense path runs on vLLM's stock layer implementations."""
    return ()


def bind() -> List[Any]:
    return []
