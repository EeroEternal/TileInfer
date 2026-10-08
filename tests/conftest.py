"""Shared pytest fixtures.

Everything here is CPU-friendly: the suite must pass on a laptop with no NPU so that the planning
logic and the reference implementation stay testable in CI, and so that a kernel bug can be
separated from a scheduling bug.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture
def cpu() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def seed() -> int:
    torch.manual_seed(0)
    return 0


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "npu: test requires an Ascend NPU")
    config.addinivalue_line("markers", "tilelang: test requires the TileLang Ascend toolchain")
