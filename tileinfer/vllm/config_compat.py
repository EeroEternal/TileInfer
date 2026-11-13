"""Config-class aliases that vLLM 0.23 ships but its AutoConfig map does not name.

This module is imported by the general-plugin path before ``ModelConfig`` is built.  It only
registers a class that vLLM itself ships; it does not implement a model, and it never raises.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def register_nemotron_h_alias() -> Dict[str, Any]:
    """Make ``model_type="nemotron_h"`` resolvable by Transformers.

    vLLM 0.23.0 has ``NemotronHConfig`` and ``NemotronHForCausalLM`` but no ``nemotron_h``
    entry in ``_CONFIG_REGISTRY``.  With Transformers 4.57 the first ``ModelConfig`` then fails
    before any model code runs.  Registering the shipped class is a compatibility alias, not an
    upstream edit.
    """
    info: Dict[str, Any] = {"name": "nemotron_h", "registered": False, "detail": ""}
    try:
        from transformers import AutoConfig
        from vllm.transformers_utils.configs.nemotron_h import NemotronHConfig
    except Exception as exc:  # noqa: BLE001 - CPU/non-Nemotron environments are fine
        info["detail"] = f"not registered: imports unavailable ({exc!r})"
        return info

    try:
        NemotronHConfig.model_type = "nemotron_h"
        AutoConfig.register("nemotron_h", NemotronHConfig, exist_ok=True)
        info["registered"] = True
        info["detail"] = "AutoConfig['nemotron_h'] -> vLLM NemotronHConfig"
        logger.info("TileInfer: registered AutoConfig alias nemotron_h -> NemotronHConfig")
    except Exception as exc:  # noqa: BLE001 - registration must not abort boot
        info["detail"] = f"not registered: {exc!r}"
        logger.warning("TileInfer: could not register nemotron_h config alias: %r", exc)
    return info
