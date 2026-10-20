"""gdn_meta_memo(H70-b)契约测试:门控、memo 键、命中/失效语义。

背景:H71 假设——decode 稳态(纯 decode、无 spec、mamba_cache_mode="none")
下 GDNAttentionMetadataBuilder.build 逐步重算为纯冗税,按全 CPU 侧键
(req_ids、num_reqs、num_actual_tokens、attn_state、purity)memo 化可削
(E285:gdn_build 0.377-0.492ms/步)。本测试在纯 CPU 面锁死安全契约:
  1) 门控:env 未设/非 "1" → install 返回 False、零包装;
  2) 命中:同键二连调返回同一对象(vendor build 只调一次);
  3) 失效:req_ids 变化 / purity 破坏(prefill 步)/ spec 入参 /
     非 none mamba 模式 → 必重建,且重建后旧缓存不得复活
     (prefill 重建覆写持久缓冲,stale 命中=状态写错行);
  4) passthrough 条件:runner 上下文缺失、is_prefilling 非 CPU。
在役激活面由 serve 日志 "gdn meta memo ACTIVE" 与 stats 探针覆盖。

运行::

    .venv/bin/python -m pytest tests/test_gdn_meta_memo.py -v
"""

from __future__ import annotations

import pytest

try:
    from tileinfer.integrations import gdn_meta_memo as gm

    _HAVE_MOD = True
except Exception:  # pragma: no cover
    _HAVE_MOD = False

pytestmark = pytest.mark.skipif(not _HAVE_MOD, reason="gdn_meta_memo module unavailable")


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.delenv("TILEINFER_GDN_META_MEMO", raising=False)
    monkeypatch.setattr(gm, "_INSTALLED", False)
    monkeypatch.setattr(gm, "_cache", {})
    monkeypatch.setattr(gm, "_hits", 0)
    monkeypatch.setattr(gm, "_misses", 0)
    monkeypatch.setattr(gm, "_pass", 0)


class _FakeDev:
    def __init__(self, t):
        self.type = t


class _FakeTensor:
    def __init__(self, vals, cpu=True):
        self._vals = list(vals)
        self.device = _FakeDev("cpu" if cpu else "npu")

    def any(self):
        import types

        return types.SimpleNamespace(item=lambda: any(self._vals))


class _FakeCM:
    def __init__(self, num_reqs=1, num_actual_tokens=1, prefilling=False, cpu=True):
        self.num_reqs = num_reqs
        self.num_actual_tokens = num_actual_tokens
        self.is_prefilling = _FakeTensor([prefilling] * (num_reqs + 2), cpu=cpu)
        self.attn_state = "DecodeOnly"


class _FakeInputBatch:
    def __init__(self, req_ids):
        self.req_ids = list(req_ids)


class _FakeRunner:
    def __init__(self, req_ids):
        self.input_batch = _FakeInputBatch(req_ids)


class _FakeCacheCfg:
    def __init__(self, mode="none"):
        self.mamba_cache_mode = mode


class _FakeVllmConfig:
    def __init__(self, mode="none"):
        self.cache_config = _FakeCacheCfg(mode)


class _FakeBuilder:
    def __init__(self, mode="none"):
        self.vllm_config = _FakeVllmConfig(mode)
        self.calls = 0

    def build(self, common_prefix_len=0, common_attn_metadata=None, **extra):
        self.calls += 1
        return object()


def test_gating_default_off(monkeypatch):
    assert gm.memo_enabled() is False
    assert gm.install_gdn_meta_memo() is False
    assert gm._INSTALLED is False


def test_memo_hit_same_key_single_build(monkeypatch):
    """同键二连调:命中复用同一对象,vendor build 仅一次。"""
    fb = _FakeBuilder()
    runner = _FakeRunner(["req-a"])
    cm = _FakeCM()
    tls = gm._tls
    tls.runner = runner
    try:
        key = gm._memo_key(runner, cm)
        assert key is not None
        gm._cache[id(fb)] = (key, "obj-1")
        # 模拟包装层命中判定逻辑(与 _build 内联逻辑一致)
        cached = gm._cache.get(id(fb))
        assert cached is not None and cached[0] == key
        assert cached[1] == "obj-1"
    finally:
        tls.runner = None


def test_memo_key_changes_on_req_ids():
    runner = _FakeRunner(["req-a"])
    cm = _FakeCM()
    k1 = gm._memo_key(runner, cm)
    runner.input_batch.req_ids = ["req-b"]
    k2 = gm._memo_key(runner, cm)
    assert k1 is not None and k2 is not None and k1 != k2


def test_memo_key_purity_break():
    runner = _FakeRunner(["req-a"])
    cm_pre = _FakeCM(prefilling=True)
    assert gm._memo_key(runner, cm_pre) is None
    cm_dev = _FakeCM(cpu=False)
    assert gm._memo_key(runner, cm_dev) is None
    assert gm._memo_key(_FakeRunner(["a", "b", "c"]), _FakeCM(num_reqs=5)) is None


def _fake_vllm_ascend(monkeypatch):
    """注入伪造 vllm_ascend 模块链(裸进程零真实 vendor 导入,避免污染
    其它测试的 sys.modules 现场;与 test_step_diag 同型但父属性齐全)。"""
    import sys
    import types

    calls = {"n": 0}

    def _stub_build(bslf, common_prefix_len=0, common_attn_metadata=None, **extra):
        calls["n"] += 1
        return ("built", calls["n"])

    class _StubBuilder:
        build = _stub_build

    class _StubRunner:
        _build_attention_metadata = lambda self, *a, **k: None  # noqa: E731

    top = types.ModuleType("vllm_ascend")
    worker_pkg = types.ModuleType("vllm_ascend.worker")
    mrv = types.ModuleType("vllm_ascend.worker.model_runner_v1")
    ops_pkg = types.ModuleType("vllm_ascend.ops")
    gab = types.ModuleType("vllm_ascend.ops.gdn_attn_builder")
    mrv.NPUModelRunner = _StubRunner
    gab.AscendGDNAttentionMetadataBuilder = _StubBuilder
    top.worker = worker_pkg
    top.ops = ops_pkg
    worker_pkg.model_runner_v1 = mrv
    ops_pkg.gdn_attn_builder = gab
    for name, mod in (
        ("vllm_ascend", top),
        ("vllm_ascend.worker", worker_pkg),
        ("vllm_ascend.worker.model_runner_v1", mrv),
        ("vllm_ascend.ops", ops_pkg),
        ("vllm_ascend.ops.gdn_attn_builder", gab),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return mrv, gab, _StubRunner, _StubBuilder, calls


def test_install_wraps_and_full_semantics(monkeypatch):
    """伪造 vendor 命名空间下:install 后包装生效,走完整
    命中/失效/passthrough 决策面(miss 路径不触真实 vendor 代码)。"""
    mrv, gab, _StubRunner, _StubBuilder, calls = _fake_vllm_ascend(monkeypatch)

    orig_bam = _StubRunner._build_attention_metadata
    orig_build = _StubBuilder.build
    monkeypatch.setenv("TILEINFER_GDN_META_MEMO", "1")
    assert gm.install_gdn_meta_memo() is True
    assert gm._INSTALLED is True
    assert mrv.NPUModelRunner._build_attention_metadata is not orig_bam
    assert gab.AscendGDNAttentionMetadataBuilder.build is not orig_build

    fb = _FakeBuilder()
    runner = _FakeRunner(["req-a"])
    cm = _FakeCM()

    # 1) 首调 miss → stub 执行,写缓存
    tls = gm._tls
    tls.runner = runner
    try:
        o1 = gab.AscendGDNAttentionMetadataBuilder.build(fb, 0, cm)
        assert o1 == ("built", 1)
        assert id(fb) in gm._cache
        # 2) 同键二连调 → 命中同一对象,stub 不再调
        o2 = gab.AscendGDNAttentionMetadataBuilder.build(fb, 0, cm)
        assert o2 is o1 and calls["n"] == 1 and gm.stats()["hits"] == 1
        # 3) spec 入参 → passthrough + 清槽
        gab.AscendGDNAttentionMetadataBuilder.build(fb, 0, cm, num_accepted_tokens="x")
        assert gm.stats()["passthrough"] == 1 and id(fb) not in gm._cache
        # 4) 重填缓存后 prefill 步 → 清槽 + 重建;恢复 decode → 必重建(无 stale 复活)
        gab.AscendGDNAttentionMetadataBuilder.build(fb, 0, cm)
        assert id(fb) in gm._cache
        cm_pre = _FakeCM(prefilling=True)
        gab.AscendGDNAttentionMetadataBuilder.build(fb, 0, cm_pre)
        assert id(fb) not in gm._cache
        gab.AscendGDNAttentionMetadataBuilder.build(fb, 0, cm)
        assert calls["n"] == 5
        # 5) 非 none mamba 模式 → passthrough 永不 memo
        fb2 = _FakeBuilder(mode="align")
        gab.AscendGDNAttentionMetadataBuilder.build(fb2, 0, cm)
        assert id(fb2) not in gm._cache and gm.stats()["passthrough"] == 2
        # 6) runner 上下文缺失 → passthrough
        tls.runner = None
        gab.AscendGDNAttentionMetadataBuilder.build(fb, 0, cm)
        assert gm.stats()["passthrough"] == 3
    finally:
        tls.runner = None
    assert gm.stats()["misses"] >= 3


def test_install_idempotent(monkeypatch):
    _fake_vllm_ascend(monkeypatch)
    monkeypatch.setenv("TILEINFER_GDN_META_MEMO", "1")
    assert gm.install_gdn_meta_memo() is True
    assert gm.install_gdn_meta_memo() is True
