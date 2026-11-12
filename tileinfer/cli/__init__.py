"""L0 - the deploy-time entry point (see :mod:`tileinfer.cli.plan`).

Deliberately does not import the submodule at package import time: ``python -m
tileinfer.cli.plan`` would otherwise find the module already in ``sys.modules`` and warn.
"""

from __future__ import annotations

__all__: list = []
