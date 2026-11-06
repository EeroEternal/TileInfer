"""`MmapWeightSource` 的离线单测（用临时 safetensors 文件，不需要 NPU）。"""

from __future__ import annotations

import os

import pytest
import torch
from safetensors.torch import save_file

from tileinfer.vllm.models.mmap_weights import MmapWeightSource


@pytest.fixture()
def ckpt(tmp_path):
    d = tmp_path / "ckpt"
    d.mkdir()
    save_file(
        {"a.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
         "b.scale": torch.ones(4, dtype=torch.bfloat16)},
        str(d / "model-00001.safetensors"),
    )
    save_file({"c.weight": torch.full((2, 2), 7.0)}, str(d / "model-00002.safetensors"))
    return str(d)


def test_reads_across_shards(ckpt):
    src = MmapWeightSource(ckpt)
    try:
        assert src.num_shards == 2
        assert "a.weight" in src and "c.weight" in src
        assert src.get("a.weight").tolist() == [[0, 1, 2], [3, 4, 5]]
        assert src.get("c.weight").tolist() == [[7.0, 7.0], [7.0, 7.0]]
    finally:
        src.close()


def test_dtypes_preserved(ckpt):
    src = MmapWeightSource(ckpt)
    try:
        assert src.get("a.weight").dtype == torch.float32
        assert src.get("b.scale").dtype == torch.bfloat16
    finally:
        src.close()


def test_tensors_stay_valid_while_source_alive(ckpt):
    """关键性质：取出的张量在 source 存活期间一直可用（句柄保活）。"""
    src = MmapWeightSource(ckpt)
    t = src.get("a.weight")
    try:
        # 再取其它张量、制造 GC 压力后，先前取到的仍有效
        for _ in range(10):
            src.get("c.weight")
        assert t.sum().item() == 15.0
    finally:
        src.close()


def test_close_then_get_raises(ckpt):
    src = MmapWeightSource(ckpt)
    src.close()
    with pytest.raises(RuntimeError):
        src.get("a.weight")


def test_unknown_name(ckpt):
    src = MmapWeightSource(ckpt)
    try:
        with pytest.raises(KeyError):
            src.get("nope")
        assert src.get_optional("nope") is None
    finally:
        src.close()


def test_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        MmapWeightSource(str(tmp_path / "does-not-exist"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
