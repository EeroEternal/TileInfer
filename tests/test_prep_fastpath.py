"""prep_fastpath(H70)契约测试:门控、bt_commit 跳过、snap 镜像、slotmap CPU 公式。

背景:H70 第一批三手法(PR#19033 同构,E281/E282 取证),opt-in
``TILEINFER_PREP_FASTPATH=1``(默认关=零行为变化)。本测试 CPU 可跑:
  1) 门控:env 非 "1" → install 返回 False 零副作用;
  2) bt_commit 跳过:内容不变零拷贝,内容/行数变化触发拷贝,clear 天然失效;
  3) snap 镜像:镜像复用+覆盖前 event 门(形态),返回值透传;
  4) slotmap CPU 公式:与 vendor triton kernel 的 CP==1 分支逐项同构
     (slot = block_table[r, pos//bs]*bs + pos%bs;pad 区 [n, max_n) = PAD_ID),
     oracle 用独立 numpy 实现对拍;门控不满足回退原实现。
设备面(在役 A/B)由 h70fast 探针链覆盖,不在 CPU 测试面内。

运行::

    .venv/bin/python -m pytest tests/test_prep_fastpath.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

try:
    from tileinfer.integrations import prep_fastpath as pf

    _HAVE_MOD = True
except Exception:  # pragma: no cover
    _HAVE_MOD = False

pytestmark = pytest.mark.skipif(not _HAVE_MOD, reason="prep_fastpath module unavailable")


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.delenv("TILEINFER_PREP_FASTPATH", raising=False)
    monkeypatch.setattr(pf, "_INSTALLED", False)
    yield


class _FakeBuffer:
    """CpuGpuBuffer 形态替身:np 视图 + 拷贝计数。"""

    def __init__(self, rows: int, cols: int, device: str = "cpu"):
        cpu = torch.zeros(rows, cols, dtype=torch.int32)
        self.cpu = cpu
        self.np = cpu.numpy()
        self.gpu = torch.zeros(rows, cols, dtype=torch.int32, device=device)
        self.copies = 0

    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        self.copies += 1
        if n is None:
            self.gpu.copy_(self.cpu)
            return self.gpu
        self.gpu[:n].copy_(self.cpu[:n])
        return self.gpu[:n]


class _FakeBT:
    """vendor BlockTable 形态替身(仅 commit_block_table 所需面)。"""

    def __init__(self, rows: int = 4, cols: int = 8):
        self.block_table = _FakeBuffer(rows, cols)
        self.commits = 0

    def commit_block_table(self, num_reqs: int) -> None:
        self.commits += 1
        bt = self.block_table
        bt.gpu[:num_reqs].copy_(bt.cpu[:num_reqs])


def test_gate_rejects_everything_but_literal_one(monkeypatch):
    for value in (None, "", "0", "true", "2"):
        if value is None:
            monkeypatch.delenv("TILEINFER_PREP_FASTPATH", raising=False)
        else:
            monkeypatch.setenv("TILEINFER_PREP_FASTPATH", value)
        assert pf.fastpath_enabled() is False
    monkeypatch.setenv("TILEINFER_PREP_FASTPATH", "1")
    assert pf.fastpath_enabled() is True


def test_bt_commit_skip_semantics():
    """内容不变跳过(零拷贝);内容变化/行数变化重拷并刷新快照。"""
    fake_cls = type("BT", (), {"commit_block_table": _FakeBT.commit_block_table})
    seg = pf._patch_bt_commit(fake_cls)
    assert seg == "bt_commit_skip"
    bt = _FakeBT()
    bt.block_table.np[0, :3] = [7, 8, 9]
    fake_cls.commit_block_table(bt, 2)
    assert bt.commits == 1
    fake_cls.commit_block_table(bt, 2)  # 内容不变 → 跳过
    assert bt.commits == 1
    bt.block_table.np[1, 0] = 5  # 行数内内容变 → 重拷
    fake_cls.commit_block_table(bt, 2)
    assert bt.commits == 2
    fake_cls.commit_block_table(bt, 3)  # 行数变 → 重拷
    assert bt.commits == 3
    fake_cls.commit_block_table(bt, 3)  # 新快照下不变 → 跳过
    assert bt.commits == 3
    # gpu 内容与 cpu 一致(跳过时保持上次提交内容)
    assert torch.equal(bt.block_table.gpu[:2].cpu(), bt.block_table.cpu[:2])


def test_snap_copy_mirror_reuses_and_returns():
    class _MRV:
        pass

    def _orig_copy(buffer):
        buffer.gpu.copy_(buffer.cpu)
        return buffer.gpu

    mrv = _MRV()
    mrv.copy_snapshot_to_gpu = _orig_copy
    seg = pf._patch_snap_copy(mrv)
    assert seg == "snap_copy_mirror"
    buf = _FakeBuffer(2, 4)
    buf.cpu[0, 0] = 3
    out = mrv.copy_snapshot_to_gpu(buf)
    assert torch.equal(out.cpu(), buf.cpu)
    out2 = mrv.copy_snapshot_to_gpu(buf)  # 复用镜像,无异常,内容一致
    assert torch.equal(out2.cpu(), buf.cpu)
    buf.cpu[1, 1] = 9
    out3 = mrv.copy_snapshot_to_gpu(buf)
    assert torch.equal(out3.cpu(), buf.cpu)


def _oracle_slotmap(bt_np, block_size, num_reqs, sched, num_computed, query_pos):
    """vendor triton kernel(CP==1 分支)的独立 oracle 实现。"""
    slots = []
    start = 0
    for r in range(num_reqs):
        for i in range(start, start + sched[r]):
            pos = int(num_computed[r]) + int(query_pos[i])
            blk = pos // block_size
            slots.append(int(bt_np[r, blk]) * block_size + pos % block_size)
        start += sched[r]
    return np.array(slots, dtype=np.int64)


class _FakeSlotSelf:
    """compute_slot_mapping 的 self 形态替身(slotmap 公式所需面)。"""

    def __init__(self, rows, cols, block_size=128, max_n=64):
        self.block_table = _FakeBuffer(rows, cols)
        self.slot_mapping = _FakeBuffer(max_n, 1)
        self.slot_mapping.np = self.slot_mapping.np.reshape(-1)
        self.slot_mapping.cpu = self.slot_mapping.cpu.reshape(-1)
        self.slot_mapping.gpu = self.slot_mapping.gpu.reshape(-1)
        self.block_size = block_size
        self.max_num_batched_tokens = max_n
        self.max_num_reqs = rows
        self.pcp_world_size = 1
        self.dcp_world_size = 1
        self.kernel_sizes = None
        self.fallbacks = 0

    def _orig_compute(self, num_reqs, qsl, positions, *a, **k):
        self.fallbacks += 1
        return "orig"


def _fake_prepare(self, *a, **k):
    return "prep"


class _FakeRunner:
    def __init__(self, num_reqs, sched, num_computed, rows, cols, block_size=128):
        self.sched = np.array(sched, dtype=np.int32)
        self.num_scheduled_tokens = type(
            "B", (), {"np": self.sched}
        )()
        self.num_computed = np.array(num_computed, dtype=np.int32)
        self.input_batch = type("IB", (), {"num_computed_tokens_cpu": self.num_computed})()
        qp = np.concatenate([np.arange(s, dtype=np.int32) for s in sched])
        self.query_pos = type("B", (), {"np": qp})()
        # 填充块表:行 r 的块号 = 100*r + 块序号(可辨识)
        for r in range(rows):
            for b in range(cols):
                pass
        self.rows, self.cols, self.block_size = rows, cols, block_size


def test_slotmap_cpu_fast_oracle_decode_and_multirow():
    """decode b1 与多行 batch:CPU 公式 == oracle;pad 区懒刷新语义。"""
    from tileinfer.integrations import prep_fastpath as pf_mod

    PAD = -1
    rows, cols, bs = 4, 8, 128
    fake_cls = type("BT", (), {"compute_slot_mapping": _FakeSlotSelf._orig_compute})
    seg = pf_mod._patch_bt_slotmap(fake_cls, type("M", (), {"NPUModelRunner": type("R", (), {"_prepare_inputs": _fake_prepare})}))
    assert seg == "bt_slotmap_cpu"

    for num_reqs, sched in ((1, [1]), (3, [1, 1, 1]), (2, [2, 3])):
        self = _FakeSlotSelf(rows, cols, bs)
        for r in range(rows):
            self.block_table.np[r, :] = np.arange(100 * r, 100 * r + cols, dtype=np.int32)
        runner = _FakeRunner(num_reqs, sched, [128 * 2 + 5 * r for r in range(num_reqs)], rows, cols, bs)
        qp_total = int(sum(sched))
        runner.query_pos.np = np.concatenate(
            [np.arange(s, dtype=np.int32) for s in sched]
        ).astype(np.int32)
        pf_mod._tls.runner = runner
        try:
            fake_cls.compute_slot_mapping(self, num_reqs, None, None)
        finally:
            pf_mod._tls.runner = None
        expect = _oracle_slotmap(
            self.block_table.np, bs, num_reqs, sched, runner.num_computed, runner.query_pos.np
        )
        got = self.slot_mapping.np[:qp_total]
        assert np.array_equal(got, expect), (num_reqs, got, expect)
        assert self.slot_mapping.copies == 1
        assert int(self.slot_mapping.gpu[qp_total : self.max_num_batched_tokens].min()) == PAD
        assert self.fallbacks == 0


def test_slotmap_gate_fallback_without_runner():
    """无 runner(threadlocal 空)→ 回退原实现。"""
    from tileinfer.integrations import prep_fastpath as pf_mod

    fake_cls = type("BT", (), {"compute_slot_mapping": _FakeSlotSelf._orig_compute})
    pf_mod._patch_bt_slotmap(fake_cls, type("M", (), {"NPUModelRunner": type("R", (), {"_prepare_inputs": _fake_prepare})}))
    self = _FakeSlotSelf(2, 8)
    pf_mod._tls.runner = None
    ret = fake_cls.compute_slot_mapping(self, 1, None, None)
    assert ret == "orig" and self.fallbacks == 1


def test_slotmap_gate_fallback_on_pcp():
    """pcp>1 → 回退原实现。"""
    from tileinfer.integrations import prep_fastpath as pf_mod

    fake_cls = type("BT", (), {"compute_slot_mapping": _FakeSlotSelf._orig_compute})
    pf_mod._patch_bt_slotmap(fake_cls, type("M", (), {"NPUModelRunner": type("R", (), {"_prepare_inputs": _fake_prepare})}))
    self = _FakeSlotSelf(2, 8)
    self.pcp_world_size = 2
    pf_mod._tls.runner = object()
    ret = fake_cls.compute_slot_mapping(self, 1, None, None)
    assert ret == "orig" and self.fallbacks == 1


def test_slotmap_exception_falls_back(monkeypatch):
    """fast path 内部异常 → 回退原实现(不得上抛破坏在役)。"""
    from tileinfer.integrations import prep_fastpath as pf_mod

    fake_cls = type("BT", (), {"compute_slot_mapping": _FakeSlotSelf._orig_compute})
    pf_mod._patch_bt_slotmap(fake_cls, type("M", (), {"NPUModelRunner": type("R", (), {"_prepare_inputs": _fake_prepare})}))
    self = _FakeSlotSelf(2, 8)
    runner = _FakeRunner(1, [1], [0], 2, 8)
    pf_mod._tls.runner = runner
    # 让公式内抛异常:np 视图换成坏的
    runner.num_scheduled_tokens.np = None
    try:
        ret = fake_cls.compute_slot_mapping(self, 1, None, None)
    finally:
        pf_mod._tls.runner = None
    assert ret == "orig" and self.fallbacks == 1
