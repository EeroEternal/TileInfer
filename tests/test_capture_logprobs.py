"""The C3 capture tool's device-free half: token-sequence comparison.

``benchmarks/capture_logprobs.py`` has two jobs.  Capturing logprobs needs a server; deciding whether
two captures are *comparable at all* does not, and that decision is the one that protects C3: if the
two arms emitted different tokens, every logprob differs for the trivial reason that the model chose a
different word, and a reader would mistake that for a numerical regression.

These tests pin that check.  The file paths are workspace-local because the sandbox forbids writing
under pytest's ``tmp_path``.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import uuid
from pathlib import Path

import pytest


def _load_module():
    path = Path("benchmarks/capture_logprobs.py").resolve()
    spec = importlib.util.spec_from_file_location("capture_logprobs", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def workdir():
    root = Path(".work") / f"cap-{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _write(folder: Path, name: str, data) -> str:
    path = folder / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_identical_sequences_pass(workdir):
    mod = _load_module()
    a = _write(workdir, "a.tokens.json", {"p": ["a", "b", "c"]})
    b = _write(workdir, "b.tokens.json", {"p": ["a", "b", "c"]})
    assert mod.compare_tokens(a, b) == 0


def test_divergent_sequences_fail_and_name_the_token(workdir):
    mod = _load_module()
    a = _write(workdir, "a.tokens.json", {"p": ["a", "b", "c"]})
    b = _write(workdir, "b.tokens.json", {"p": ["a", "Z", "c"]})
    assert mod.compare_tokens(a, b) == 1


def test_prompt_on_only_one_side_fails(workdir):
    mod = _load_module()
    a = _write(workdir, "a.tokens.json", {"p": ["a"], "q": ["b"]})
    b = _write(workdir, "b.tokens.json", {"p": ["a"]})
    assert mod.compare_tokens(a, b) == 1


def test_length_difference_is_reported(workdir):
    mod = _load_module()
    a = _write(workdir, "a.tokens.json", {"p": ["a", "b"]})
    b = _write(workdir, "b.tokens.json", {"p": ["a"]})
    assert mod.compare_tokens(a, b) == 1


def test_prompts_are_the_stable_contract():
    """A baseline is only comparable to a capture over the *same* prompts."""
    mod = _load_module()
    assert len(mod.PROMPTS) >= 3
    assert len(set(mod.PROMPTS)) == len(mod.PROMPTS), "duplicate prompts would double-count"
    assert all(isinstance(p, str) and p for p in mod.PROMPTS)
    assert mod.MAX_TOKENS > 0 and mod.LOGPROB_TOP > 0
