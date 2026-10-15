"""CPU 契约测试:T6 配置驱动的预热规格推导(``tileinfer.integrations.prewarm``)。

目的:锁定「预烧脚本推导的规格集」与「引擎内联公式推导的规格集」**逐项等价**——只要两者
一致,out-of-process 预烧就一定命中引擎的编译 key,服务路径零编译(T6 的目标契约)。

引擎内联公式在 ``TileInferDecodeAttention.prewarm_prefill`` /
``_prewarm_prefill_ladder``(本测试**不 import torch 侧重物**,而是用同款公式在测试里独立
复算一遍作为"参考实现",再与 ``prewarm`` 模块比对)。

覆盖:
1. ``declared_cap`` 与 ``kernels...declared_pool_cap`` 一致(含池漂移 9425→9899 落同桶);
2. ``sub_pages``/``bucket_for``/``decode_buckets`` 边界;
3. ``prefill_ladder_tops`` 与引擎公式逐项一致(plain 与 split 两档);
4. ``prefill_specs`` 的完整 ladder(变体数、split 追加规则、元组顺序 = 9/10 字段);
5. ``build_prewarm_plan`` 端到端:decode 桶 ∩ prefill ladder 与引擎推导一致。
"""

from __future__ import annotations

import pytest

from tileinfer.integrations.prewarm import (
    KERNEL_PAGE_SIZE,
    build_prewarm_argv,
    build_prewarm_plan,
    bucket_for,
    decode_buckets,
    declared_cap,
    next_pow2,
    prefill_ladder_tops,
    prefill_specs,
    prewarm_plan_to_dict,
    sub_pages,
)

BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


# ------------------------------------------------------------------ 0. 漂移守护(对引擎真值)

#: 引擎侧常量/函数必须与 prewarm 模块一致,否则"单一真源"名存实亡(预烧静默不命中)。
#: 这组断言让常量漂移在 CI 就被抓住,而不是等到服务期发现预烧失效。
def test_engine_constants_match_prewarm_module():
    from tileinfer.integrations import vllm_ascend as engine

    assert engine.KERNEL_PAGE_SIZE == KERNEL_PAGE_SIZE
    assert engine.BATCH_BUCKETS == BUCKETS
    assert engine.PREFILL_MAX_TILES == 4096


def test_engine_helpers_match_prewarm_module():
    from tileinfer.integrations import vllm_ascend as engine

    for n in (0, 1, 2, 3, 7, 8, 64, 100, 1024, 4096):
        assert engine._next_pow2(n) == next_pow2(n), n
    for b in (128, 256, 2048, 4096):
        assert engine._sub_pages(b) == sub_pages(b), b
    for r in (1, 3, 8, 32, 33, 256):
        assert engine._bucket_for(r) == bucket_for(r, BUCKETS), r


def test_engine_declared_cap_matches_prewarm_module():
    from tileinfer.kernels.attention.paged_decode_ascend950 import declared_pool_cap
    from tileinfer.integrations import vllm_ascend as engine  # noqa: F401

    for pool in (1, 2, 3, 100, 9425, 9899, 2288, 65536):
        assert declared_pool_cap(pool) == declared_cap(pool), pool


# ------------------------------------------------------------------ 参考实现(引擎公式复刻)


def _engine_tiles_top(
    max_num_reqs: int, budget: int, group: int, prefill_max_tiles: int
) -> int:
    """``prewarm_prefill`` 的 plain 顶公式(逐行复刻)。"""
    limit = max_num_reqs if max_num_reqs else BUCKETS[-1]
    budget = budget or 8192
    top = next_pow2(-(-budget * group // KERNEL_PAGE_SIZE) + limit)
    return min(top, prefill_max_tiles)


def _engine_tiles_top_split(
    tiles_top: int, max_model_len: int, ktp: int, prefill_max_tiles: int
) -> int:
    """``prewarm_prefill`` 的 split 顶公式(逐行复刻)。"""
    if ktp <= 0:
        return tiles_top
    factor = 1
    if max_model_len:
        max_pages = -(-max_model_len // KERNEL_PAGE_SIZE)
        factor = -(-max_pages // ktp) + 1
    return min(next_pow2(tiles_top * factor), prefill_max_tiles)


def _engine_ladder_specs(kernel_batch, kvh, group, dim, pool, dtype, tiles_top, tiles_top_split, ktp, ms):
    """``_prewarm_prefill_ladder`` 的规格序列(逐行复刻)。"""
    specs = []
    tiles = 1
    while tiles <= max(tiles_top, tiles_top_split):
        plain = (kernel_batch, kvh, group, dim, KERNEL_PAGE_SIZE, pool, tiles, 128, dtype)
        specs.append(plain)
        if ktp > 0 and tiles <= tiles_top_split:
            specs.append(plain + (ms,))
        tiles *= 2
    return specs


# ------------------------------------------------------------------ 1. declared_cap


def test_declared_cap_matches_kernel_helper():
    """``prewarm.declared_cap`` 必须与内核侧 ``declared_pool_cap`` 同值(无 pin 时)。"""
    from tileinfer.kernels.attention.paged_decode_ascend950 import declared_pool_cap

    for pool in (1, 2, 3, 100, 9425, 9899, 2288, 65536):
        assert declared_cap(pool) == declared_pool_cap(pool), pool


def test_pool_drift_same_bucket():
    """池从 9425 漂到 9899(真实 vLLM 快照波动)必须落在同一声明容量(预烧跨运行命中)。"""
    assert declared_cap(9425) == declared_cap(9899)


def test_declared_cap_pin():
    assert declared_cap(100, 128) == 128
    with pytest.raises(ValueError):
        declared_cap(200, 128)


# ------------------------------------------------------------------ 2. sub_pages / bucket


def test_sub_pages():
    assert sub_pages(128) == 1
    assert sub_pages(2048) == 16
    assert sub_pages(256) == 2
    with pytest.raises(ValueError):
        sub_pages(100)


def test_bucket_for():
    assert bucket_for(1, BUCKETS) == 1
    assert bucket_for(3, BUCKETS) == 4
    assert bucket_for(32, BUCKETS) == 32
    assert bucket_for(33, BUCKETS) == 64
    with pytest.raises(ValueError):
        bucket_for(257, BUCKETS)


def test_decode_buckets():
    assert decode_buckets(8, BUCKETS) == [1, 2, 4, 8]
    assert decode_buckets(0, BUCKETS) == list(BUCKETS)  # 配置不可得:全建
    assert decode_buckets(1, BUCKETS) == [1]


# ------------------------------------------------------------------ 3. ladder 顶


@pytest.mark.parametrize("max_num_reqs,budget,group,model_len,ktp", [
    (32, 8192, 8, 0, 0),
    (32, 8192, 8, 32768, 64),
    (32, 8192, 8, 32768, 0),
    (4, 2048, 6, 0, 0),
    (0, 0, 8, 32768, 64),  # 配置缺省
    (256, 16384, 8, 65536, 128),
])
def test_ladder_tops_match_engine(max_num_reqs, budget, group, model_len, ktp):
    tops = prefill_ladder_tops(
        max_num_reqs=max_num_reqs,
        max_num_batched_tokens=budget,
        max_model_len=model_len,
        group_size=group,
        buckets=BUCKETS,
        prefill_max_tiles=4096,
        prefill_kv_tile_pages=ktp,
    )
    ref_plain = _engine_tiles_top(max_num_reqs, budget, group, 4096)
    ref_split = _engine_tiles_top_split(ref_plain, model_len, ktp, 4096)
    assert tops == (ref_plain, ref_split)


def test_ladder_top_is_pow2_and_bounded():
    top, top_split = prefill_ladder_tops(
        max_num_reqs=32, max_num_batched_tokens=8192, max_model_len=32768,
        group_size=8, buckets=BUCKETS, prefill_max_tiles=4096, prefill_kv_tile_pages=64,
    )
    assert top & (top - 1) == 0 and top <= 4096
    assert top_split & (top_split - 1) == 0 and top_split <= 4096
    assert top_split >= top


# ------------------------------------------------------------------ 4. 完整 ladder


@pytest.mark.parametrize("ktp,model_len", [(0, 0), (64, 32768), (128, 65536)])
def test_prefill_specs_match_engine(ktp, model_len):
    kw = dict(
        num_kv_heads=2, group_size=8, head_size=256, pool_pages=4096, dtype="bfloat16",
        max_num_reqs=32, max_num_batched_tokens=8192, max_model_len=model_len,
        buckets=BUCKETS, prefill_max_tiles=4096, prefill_kv_tile_pages=ktp, split_ms_compile=16,
    )
    got = prefill_specs(**kw)
    top, top_split = prefill_ladder_tops(
        max_num_reqs=32, max_num_batched_tokens=8192, max_model_len=model_len,
        group_size=8, buckets=BUCKETS, prefill_max_tiles=4096, prefill_kv_tile_pages=ktp,
    )
    ref = _engine_ladder_specs(
        bucket_for(32, BUCKETS) + 1, 2, 8, 256, 4096, "bfloat16", top, top_split, ktp, 16
    )
    assert got == ref


def test_prefill_specs_field_shapes():
    """plain 变体 9 字段、split 变体 10 字段(第 10 = max_splits),顺序与 _compile 解包一致。"""
    plain_only = prefill_specs(
        num_kv_heads=2, group_size=8, head_size=256, pool_pages=4096, dtype="bfloat16",
        max_num_reqs=32, max_num_batched_tokens=8192, max_model_len=32768,
        buckets=BUCKETS, prefill_max_tiles=4096, prefill_kv_tile_pages=0, split_ms_compile=16,
    )
    assert plain_only and all(len(s) == 9 for s in plain_only)
    # 批维 = bucket(32)+1 = 33(real 32 + 1 trash);block_q 恒 128
    assert all(s[0] == 33 and s[7] == 128 for s in plain_only)

    with_split = prefill_specs(
        num_kv_heads=2, group_size=8, head_size=256, pool_pages=4096, dtype="bfloat16",
        max_num_reqs=32, max_num_batched_tokens=8192, max_model_len=32768,
        buckets=BUCKETS, prefill_max_tiles=4096, prefill_kv_tile_pages=64, split_ms_compile=16,
    )
    n10 = [s for s in with_split if len(s) == 10]
    assert n10 and all(s[9] == 16 for s in n10)


def test_prefill_ladder_variant_count_within_cap():
    """canonical ladder 的变体数应远小于引擎变体上限 32(P3/D39)。"""
    specs = prefill_specs(
        num_kv_heads=2, group_size=8, head_size=256, pool_pages=4096, dtype="bfloat16",
        max_num_reqs=32, max_num_batched_tokens=8192, max_model_len=32768,
        buckets=BUCKETS, prefill_max_tiles=4096, prefill_kv_tile_pages=0, split_ms_compile=16,
    )
    assert len(specs) <= 32


# ------------------------------------------------------------------ 5. 端到端 plan


def test_build_prewarm_plan_qwen36():
    """Qwen3.6 真实形状:block=2048(sub=16)、池漂移落同桶、decode 桶 ∩ prefill ladder 自洽。"""
    plan = build_prewarm_plan(
        num_kv_heads=2, group_size=8, head_size=256, dtype="bfloat16",
        block_size=2048, engine_pages=600,  # 600*16=9600 子页 → cap 16384
        max_num_reqs=32, max_num_batched_tokens=8192, max_model_len=32768,
        buckets=BUCKETS, prefill_max_tiles=4096, prefill_kv_tile_pages=0, split_ms_compile=16,
    )
    assert plan.decode == [1, 2, 4, 8, 16, 32]
    assert plan.pool_pages == 16384
    assert plan.prefill and all(s[5] == 16384 for s in plan.prefill)
    assert plan.notes["sub_pages"] == 16


def test_build_prewarm_plan_pool_agnostic():
    """两次运行池不同但落同桶 → 预烧规格完全一致(跨进程命中)。"""
    kw = dict(
        num_kv_heads=2, group_size=6, head_size=128, dtype="bfloat16",
        block_size=128, max_num_reqs=32, max_num_batched_tokens=8192, max_model_len=2048,
        buckets=BUCKETS, prefill_max_tiles=4096, prefill_kv_tile_pages=0, split_ms_compile=16,
    )
    a = build_prewarm_plan(engine_pages=9425, **kw)
    b = build_prewarm_plan(engine_pages=9899, **kw)
    assert a.pool_pages == b.pool_pages
    assert a.decode == b.decode and a.prefill == b.prefill


def test_prewarm_plan_to_dict_is_json_safe():
    """plan → dict 必须可 JSON 序列化(CI/运维 diff 用),且字段与 plan 一致。"""
    import json

    plan = build_prewarm_plan(
        num_kv_heads=2, group_size=8, head_size=256, dtype="bfloat16",
        block_size=2048, engine_pages=600, max_num_reqs=32, max_num_batched_tokens=8192,
        max_model_len=32768, buckets=BUCKETS, prefill_max_tiles=4096,
        prefill_kv_tile_pages=64, split_ms_compile=16,
    )
    d = prewarm_plan_to_dict(plan)
    s = json.dumps(d)  # must not raise
    assert d["decode"] == plan.decode
    assert d["pool_pages"] == plan.pool_pages
    assert d["prefill"] == [list(x) for x in plan.prefill]
    assert d["prefill_plain"] + d["prefill_split"] == len(plan.prefill)
    assert d["prefill_split"] > 0  # ktp=64 → 有 split 变体
    assert isinstance(s, str) and s


# ------------------------------------------------------------------ 6. T6 收尾:启动器 argv


def test_build_prewarm_argv_roundtrips_to_cli():
    """argv 必须能被预烧脚本的 argparse 完整解析,且字段值一致(启动器→脚本不丢参)。"""
    import argparse

    argv = build_prewarm_argv(
        num_kv_heads=2, group_size=8, head_size=256, dtype="bfloat16",
        block_size=128, engine_pages=15589, max_num_reqs=32,
        max_num_batched_tokens=8192, max_model_len=32768, prefill=True,
        prefill_kv_tile_pages=64,
    )
    # 与脚本 args 定义等价的极简 parser(只校验我们产出的键都能被接受)
    p = argparse.ArgumentParser()
    for name in ("kv-heads", "group", "dim", "block-size", "num-engine-pages",
                 "max-num-reqs", "max-num-batched-tokens", "max-model-len",
                 "prefill-kv-tile-pages"):
        p.add_argument("--" + name, type=int)
    p.add_argument("--dtype")
    p.add_argument("--prefill", action="store_true")
    ns = p.parse_args(argv)
    assert ns.kv_heads == 2 and ns.group == 8 and ns.dim == 256
    assert ns.block_size == 128 and ns.num_engine_pages == 15589
    assert ns.max_num_reqs == 32 and ns.max_num_batched_tokens == 8192 and ns.max_model_len == 32768
    assert ns.dtype == "bfloat16" and ns.prefill is True and ns.prefill_kv_tile_pages == 64


def test_build_prewarm_argv_prefill_optional():
    """prefill=False 时不带 --prefill/--prefill-kv-tile-pages(decode-only 配方保持可用)。"""
    argv = build_prewarm_argv(
        num_kv_heads=2, group_size=6, head_size=128, dtype="bfloat16",
        block_size=128, engine_pages=9425, max_num_reqs=32,
        max_num_batched_tokens=8192, max_model_len=2048, prefill=False,
    )
    assert "--prefill" not in argv
    assert "--prefill-kv-tile-pages" not in argv
    assert "--num-engine-pages" in argv and "9425" in argv


