"""批处理散写原语 `write_ring_batched` 的单测（§24 的 A 步，不需要 NPU）。

核心性质：一次散写必须与"逐请求各写自己的行"**逐位一致**。
"""

from __future__ import annotations

import torch

from tileinfer.models.dsv41.layer import write_ring_batched


def test_matches_per_request_writes():
    win, dim = 6, 3
    r = 4
    batched = torch.zeros(r, win, dim)
    slots = [0, 1, 2, 3]
    positions = [0, 5, 7, 12]
    kv = torch.arange(r * dim, dtype=torch.float32).reshape(r, dim)

    write_ring_batched(batched, slots, positions, kv, win)

    loop = torch.zeros(r, win, dim)
    for i in range(r):
        loop[i, positions[i] % win] = kv[i]
    assert torch.equal(batched, loop)


def test_slots_need_not_be_contiguous():
    """槽位号可以乱序/不连续（真实场景里由 StateSlots 分配）。"""
    win, dim = 4, 2
    batched = torch.zeros(6, win, dim)
    slots = [5, 2]
    positions = [3, 9]
    kv = torch.tensor([[1.0, 1.0], [2.0, 2.0]])

    write_ring_batched(batched, slots, positions, kv, win)

    assert batched[5, 3].tolist() == [1.0, 1.0]
    assert batched[2, 1].tolist() == [2.0, 2.0]
    # 其它行/位置保持为零
    assert batched.sum().item() == 2.0 + 4.0


def test_wrap_around_overwrites_old_content():
    """写满一圈后，旧内容应被覆盖（无回滚语义）。"""
    win, dim = 3, 1
    batched = torch.zeros(1, win, dim)
    for pos in range(6):
        write_ring_batched(batched, [0], [pos], torch.tensor([[float(pos)]]), win)
    # 最后写入的是 pos=5 -> 槽位 2
    assert batched[0, :, 0].tolist() == [3.0, 4.0, 5.0]


def test_dtype_follows_buffer():
    win, dim = 4, 2
    batched = torch.zeros(2, win, dim, dtype=torch.bfloat16)
    kv = torch.ones(2, dim, dtype=torch.float32)
    write_ring_batched(batched, [0, 1], [0, 0], kv, win)
    assert batched.dtype == torch.bfloat16
    assert batched[0, 0].tolist() == [1.0, 1.0]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
