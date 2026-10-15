"""H21:decode split-KV per-bucket ktp 策略(CPU 契约,零设备依赖)。

背景(E61/D54/D57):split-KV 并发税是波次量子化的——schedule 发射
``bucket × ceil(width/ktp)`` 个 tile,空 tile(页循环被内核钳制为 0)仍占满 AIV 波槽
(b8×ktp16=128 tile ≈ 4.6 波 vs plain 1 波 = 实测 4.54x;≤28 tile 时同一 split 机制零税,
E61 矩阵 C 宽度 16 plan ≡ plain)。全局单值 ktp 因此小批饿场/大批收税二选一;
H21 把 ktp 变成 per-bucket 策略(env ``TILEINFER_KV_TILE_PAGES_POLICY="1:16,8:128"``),
未列出的 bucket 回退全局 ``TILEINFER_KV_TILE_PAGES``。

锁定契约:
  1. 策略解析(合法/空白/非法项);
  2. ktp_for(bucket):策略命中 → 策略值;未命中 → 全局值;
  3. plan key 按「bucket 自身 ktp」区分(不同 ktp 的 bucket 不得共享 plan);
  4. ctor 副作用:策略含 split(ktp>0)⇒ TILEINFER_ALLOW_SPLIT_KV 以 setdefault 语义
     自动置 1(显式值不被覆盖);策略不含 split ⇒ 环境零改动;
  5. prewarm 按 bucket 用各自 ktp 构键(_build_plan 录制法,零设备);
  6. 在役探针视图 _policy_view(global + 策略表)随 ctor 刷新。

运行::

    .venv/bin/python -m pytest tests/test_vllm_ascend_ktp_policy.py -v
"""

from __future__ import annotations

import pytest

try:
    from tileinfer.integrations import vllm_ascend as mod
    from tileinfer.integrations.vllm_ascend import TileInferDecodeAttention

    _HAVE_MOD = True
except Exception:  # pragma: no cover - CPU 机器无 vllm_ascend 依赖时跳过
    _HAVE_MOD = False


pytestmark = pytest.mark.skipif(not _HAVE_MOD, reason="vllm_ascend integration unavailable")


# ---------------------------------------------------------------- 1. 解析


def test_parse_policy_valid():
    assert mod._parse_ktp_policy("1:16,8:128") == {1: 16, 8: 128}
    assert mod._parse_ktp_policy(" 2 : 32 , 4:64 ") == {2: 32, 4: 64}
    # 重复键后者覆盖(dict 语义),ktp=0 是合法条目(该 bucket 显式 plain)
    assert mod._parse_ktp_policy("8:0") == {8: 0}
    assert mod._parse_ktp_policy("") == {}


def test_parse_policy_invalid():
    for bad in ("1-16", "8:", ":16", "x:16", "8:y", "8:16:", "1:16;2:32"):
        with pytest.raises(ValueError):
            mod._parse_ktp_policy(bad)


# ---------------------------------------------------------------- 2/3. ktp_for 与 plan key


def _impl(policy_env=None, global_ktp=0):
    import os

    old = os.environ.pop("TILEINFER_KV_TILE_PAGES_POLICY", None)
    try:
        if policy_env is not None:
            os.environ["TILEINFER_KV_TILE_PAGES_POLICY"] = policy_env
        return TileInferDecodeAttention(
            num_heads=32,
            num_kv_heads=8,
            head_size=256,
            scale=0.1,
            device="cpu",
            kv_tile_pages=global_ktp,
        )
    finally:
        if old is None:
            os.environ.pop("TILEINFER_KV_TILE_PAGES_POLICY", None)
        else:
            os.environ["TILEINFER_KV_TILE_PAGES_POLICY"] = old


def test_ktp_for_policy_hit_and_fallback():
    impl = _impl("1:16,8:128", global_ktp=32)
    assert impl.ktp_for(1) == 16
    assert impl.ktp_for(8) == 128
    # 未列出的 bucket 回退全局值
    assert impl.ktp_for(2) == 32
    assert impl.ktp_for(32) == 32


def test_ktp_for_no_policy_equals_global():
    impl = _impl(None, global_ktp=16)
    for bucket in (1, 2, 4, 8, 16, 32):
        assert impl.ktp_for(bucket) == 16


def test_plan_key_separates_buckets_by_their_own_ktp():
    impl = _impl("1:16,8:128", global_ktp=32)
    k1 = impl._plan_key(1, 64, 128, 1024)
    k8 = impl._plan_key(8, 64, 128, 1024)
    k2 = impl._plan_key(2, 64, 128, 1024)
    k2_alt = impl._plan_key(2, 64, 128, 1024)
    # 同 bucket 同参数 → 同键(缓存命中不受影响)
    assert k2 == k2_alt
    # 不同 bucket 不同 ktp → 键不同(不共享 plan)
    assert len({k1, k8, k2}) == 3
    # 键中显式含该 bucket 的 ktp
    assert int(impl.ktp_for(1)) in k1 and int(impl.ktp_for(8)) in k8


# ---------------------------------------------------------------- 4. ctor 副作用(门语义)


def test_policy_with_split_autogates_split_kv(monkeypatch):
    monkeypatch.delenv("TILEINFER_ALLOW_SPLIT_KV", raising=False)
    _impl("1:16", global_ktp=0)
    import os

    assert os.environ.get("TILEINFER_ALLOW_SPLIT_KV") == "1"


def test_policy_split_keeps_explicit_gate_value(monkeypatch):
    # setdefault 语义:用户显式设过的值不被覆盖
    monkeypatch.setenv("TILEINFER_ALLOW_SPLIT_KV", "0")
    _impl("1:16", global_ktp=0)
    import os

    assert os.environ.get("TILEINFER_ALLOW_SPLIT_KV") == "0"


def test_plain_policy_leaves_gate_untouched(monkeypatch):
    monkeypatch.delenv("TILEINFER_ALLOW_SPLIT_KV", raising=False)
    _impl("8:0", global_ktp=0)
    import os

    assert os.environ.get("TILEINFER_ALLOW_SPLIT_KV") is None


# ---------------------------------------------------------------- 5. prewarm 按 bucket 构键


def test_prewarm_builds_per_bucket_ktp_keys(monkeypatch):
    impl = _impl("1:16,8:128,32:0", global_ktp=0)
    seen: dict = {}

    def fake_build(self, key, bucket, width, block_size, key_cache):
        seen[bucket] = (key, self.ktp_for(bucket))

    monkeypatch.setattr(TileInferDecodeAttention, "_build_plan", fake_build)
    # decode prewarm 末尾会连带 P3 prefill 阶梯预热(真实 tilelang 编译,CPU 机器禁触)
    monkeypatch.setattr(
        TileInferDecodeAttention, "prewarm_prefill", lambda self, bs, kc: -1, raising=True
    )
    monkeypatch.setattr(
        TileInferDecodeAttention, "_plans", {}
    )  # 隔离类级缓存,保证全量构建
    monkeypatch.setattr(TileInferDecodeAttention, "_failed", set())
    monkeypatch.setattr(TileInferDecodeAttention, "_building", set())
    key_cache = __import__("torch").zeros(4, 128, 2, 256)
    impl.prewarm(width=64, block_size=128, key_cache=key_cache)
    # max_num_reqs 未传 → 建到最大桶;这里只断言策略桶的 ktp 编码进键与构建参数
    assert seen[1][1] == 16 and int(impl.ktp_for(1)) in seen[1][0]
    assert seen[8][1] == 128 and int(impl.ktp_for(8)) in seen[8][0]
    assert seen[32][1] == 0
    # 非策略桶回退全局 ktp=0,键与策略桶互不相同
    assert seen[2][1] == 0
    assert len({seen[b][0] for b in seen}) == len(seen)


# ---------------------------------------------------------------- 6. 探针视图


def test_policy_view_published_for_metrics(monkeypatch):
    impl = _impl("1:16,8:128", global_ktp=4)
    view = TileInferDecodeAttention._policy_view
    assert view == {"global": 4, 1: 16, 8: 128}
    # JSON 可序列化(在役 metrics dumper 直接 json.dump)
    import json

    json.dumps(view)
