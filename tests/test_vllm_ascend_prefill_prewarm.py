"""设备测试:P3 prefill 预热覆盖(变体量化归一 + ladder 预热)。

P3 把 prefill 内核变体从「每步精确 (batch, num_tiles)」量化归一为
``(bucket(batch_cap) + 1 trash 行, 2 的幂 tiles)``:

* 批维 padding:meta 长出 trash 行(0 页、整页 last_page_len → kv_len=0),
  padding tile 指向 trash 行 → 内核页循环零次迭代,输出不读;
* tile 维 padding:q_len=0 的死 tile(打包/解包循环跳过,RowThr 全 -1)。

因此 ``prewarm_prefill`` 能用 ≤32 个 canonical 变体覆盖引擎可调度的全部 prefill
形状,新形状首步不再触发 ~110s 惰性编译窗(D22)。

覆盖:
1. 量化路径数值正确性:非 2 幂 tile 数 + 批 padding + append 语义 + decode 行
   混批,vs ``reference_attention``(TOL 3e-2,与其它内核测试同口径);
2. ladder 预热:构造小预算实例,prewarm 后任意落在 ladder 内的形状
   ``usable()`` 立即为真,且首见形状首步 prefill_step 立即接管(墙钟 <5s,
   即引擎验收「首见形状首步延迟 <5s」的 API 级等价)。

CPU 机器自动跳过;真实 head_dim=256 / group=8(Qwen3.6 full-attn 形状)。
所有用例共用恒定池形状(ENGINE_PAGES 个引擎页 → POOL_PAGES 个子页),
使 canonical spec 与 ladder spec 的 pool_pages 一致、可被磁盘缓存去重。
"""

from __future__ import annotations

import time

import pytest
import torch

from tileinfer.integrations.vllm_ascend import (
    KERNEL_PAGE_SIZE,
    PrefillShapeManager,
    TileInferDecodeAttention,
    _expand_page_table,
    _next_pow2,
    _pad_tile_schedule,
    _pool_sub_view,
    prefill_step,
)
from tileinfer.metadata import RaggedMetadata
from tileinfer.plan import plan_query_tiles
from tileinfer.testing.reference import reference_attention
from tileinfer.utils import is_npu_available

try:  # pragma: no cover
    import tilelang.ascend  # noqa: F401

    _HAVE_ASCEND950 = True
except Exception:  # pragma: no cover
    _HAVE_ASCEND950 = False

pytestmark = [
    pytest.mark.npu,
    pytest.mark.skipif(not is_npu_available(), reason="no Ascend NPU available"),
    pytest.mark.skipif(not _HAVE_ASCEND950, reason="tilelang.ascend (>=0.1.15) not importable"),
]

DTYPE = torch.bfloat16
TOL = 3e-2
ENGINE_BLOCK = 2048
SUB = ENGINE_BLOCK // KERNEL_PAGE_SIZE
KVH, GROUP, DIM = 2, 8, 256
HEADS = KVH * GROUP
SCALE = DIM**-0.5
ENGINE_PAGES = 4  # 恒定池:4 个引擎页 → 64 个 128-token 子页(spec 的一部分)
POOL_PAGES = ENGINE_PAGES * SUB
BATCH_CAP = 4  # canonical 批 = bucket(4)+1 = 5(真实批位 + trash 行)
CANONICAL_BATCH = BATCH_CAP + 1


# --------------------------------------------------------------------------- CPU 纯逻辑


def test_next_pow2_and_schedule_padding():
    assert [_next_pow2(x) for x in (1, 2, 3, 5, 8, 129, 512, 4096)] == [
        1, 2, 4, 8, 8, 256, 512, 4096,
    ]
    sched = plan_query_tiles(
        torch.tensor([264, 40]), torch.tensor([3, 1]), block_q=128, load_balance=False
    )
    assert sched.num_tiles == 4
    padded = _pad_tile_schedule(sched, 8, trash_seq=5)
    assert padded.num_tiles == 8
    assert padded.q_lens.tolist() == [128, 128, 8, 40, 0, 0, 0, 0]
    assert padded.seq_ids.tolist() == [0, 0, 0, 1, 5, 5, 5, 5]
    assert padded.kv_page_lens.tolist()[:4] == sched.kv_page_lens.tolist()
    assert padded.q_offsets.tolist()[:4] == sched.q_offsets.tolist()


# --------------------------------------------------------------------------- 设备用例


def _run_case(kv_lens: list, qo_lens: list, seed: int):
    """按恒定池形状构造一次 prefill_step 调用;返回 (served, 墙钟秒)。"""
    assert sum(-(-kv // ENGINE_BLOCK) for kv in kv_lens) <= ENGINE_PAGES, "池溢出"
    device = torch.device("npu", 0)
    gen = torch.Generator().manual_seed(seed)

    pool: list[torch.Tensor] = []
    tables: list[list[int]] = []
    fulls: list[torch.Tensor] = []
    for kv in kv_lens:
        row = list(range(len(pool), len(pool) + -(-kv // ENGINE_BLOCK)))
        pool.extend(torch.zeros(ENGINE_BLOCK, KVH, DIM, dtype=DTYPE) for _ in range(len(row)))
        tables.append(row)
        full = torch.randn(kv, KVH, DIM, generator=gen).to(DTYPE) * 0.3
        fulls.append(full)
        pool[row[0]].view(-1, KVH, DIM)[:kv] = full

    num_tokens = sum(qo_lens)
    table = torch.tensor(tables, dtype=torch.int32, device=device)
    # 池恒定为 ENGINE_PAGES 个引擎页:pool_pages 进入 canonical spec,填充后与 ladder spec 一致
    while len(pool) < ENGINE_PAGES:
        pool.append(torch.zeros(ENGINE_BLOCK, KVH, DIM, dtype=DTYPE))
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    qsl = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(qo_lens), 0)), dtype=torch.int32, device=device
    )
    q = (torch.randn(num_tokens, HEADS, DIM, generator=gen).to(DTYPE) * 0.2).to(device).contiguous()
    out = torch.empty_like(q)
    kc = torch.stack(pool).to(device)
    vc = torch.randn(kc.shape, generator=gen).to(DTYPE).to(device) * 0.3

    t0 = time.time()
    served = prefill_step(
        query=q,
        key_cache=kc,
        value_cache=vc,
        block_tables=table,
        seq_lens=seq_lens,
        query_start_loc=qsl,
        block_size=ENGINE_BLOCK,
        output=out,
        num_heads=HEADS,
        num_kv_heads=KVH,
        head_size=DIM,
        scale=SCALE,
        batch_cap=BATCH_CAP,
    )
    elapsed = time.time() - t0
    if not served:
        return None, elapsed

    # 参考值:与 wiring 测试同构(子页视图 gather + 逐请求 causal reference)
    table_exp = _expand_page_table(table, SUB).cpu()
    kc_sub = _pool_sub_view(kc, SUB).cpu().float()
    vc_sub = _pool_sub_view(vc, SUB).cpu().float()
    offset = 0
    for b, (kv, qo) in enumerate(zip(kv_lens, qo_lens)):
        pages = -(-kv // KERNEL_PAGE_SIZE)
        page_ids = table_exp[b, :pages].to(torch.long)
        kb = kc_sub[page_ids]
        vb = vc_sub[page_ids]
        sub_meta = RaggedMetadata(
            kv_indptr=torch.tensor([0, pages], dtype=torch.int32),
            kv_indices=torch.arange(pages, dtype=torch.int32),
            kv_last_page_len=torch.tensor(
                [kv % KERNEL_PAGE_SIZE or KERNEL_PAGE_SIZE], dtype=torch.int32
            ),
            page_size=KERNEL_PAGE_SIZE,
        )
        sub_meta.qo_indptr = torch.tensor([0, qo], dtype=torch.int32)
        qb = q[offset : offset + qo].cpu().float().permute(1, 0, 2).unsqueeze(0)
        exp = reference_attention(qb, kb, vb, sub_meta, causal=True)
        got = out[offset : offset + qo].cpu().float().permute(1, 0, 2).unsqueeze(0)
        diff = (got - exp).abs().amax(dim=(0, 1, 3))
        bad = (diff > TOL).nonzero().flatten().tolist()
        assert not bad, (
            f"request {b} (kv={kv}, qo={qo}) mismatch: max|diff|={diff.max().item()}; "
            f"bad rows ({len(bad)}/{qo}): {bad[:16]}"
        )
        offset += qo
    return True, elapsed


def _canonical_ready(tiles: int) -> bool:
    spec = (CANONICAL_BATCH, KVH, GROUP, DIM, KERNEL_PAGE_SIZE, POOL_PAGES, tiles, 128, "bfloat16")
    return PrefillShapeManager.usable(spec)


@pytest.mark.parametrize(
    "kv_lens,qo_lens,label",
    [
        ([300], [300], "single-nonpow2-tiles"),  # t=19 → T=32,批 1→5(trash 行)
        ([500, 129, 37], [100, 129, 1], "mixed-append-decode-row"),  # t=17 → T=32,批 3→5
    ],
)
def test_prefill_quantized_numerics(kv_lens, qo_lens, label):
    """量化 canonical 变体(q_len=0 死 tile + trash 行)服务的步骤数值必须对齐参考。"""
    seed = abs(hash(label)) % 1000
    deadline = time.time() + 900  # 冷缓存时一次 canonical 编译 ~110s
    got = None
    while time.time() < deadline:
        got, _ = _run_case(kv_lens, qo_lens, seed)
        if got is not None:
            break
        time.sleep(5)
    assert got is True, f"{label}: canonical 变体编译后仍未接管"


def test_prewarm_prefill_ladder_covers_first_seen_shape():
    """prewarm ladder 后:任意 ladder 内形状 usable() 即真;首见形状首步立即接管(<5s)。"""
    instance = TileInferDecodeAttention(
        num_heads=HEADS,
        num_kv_heads=KVH,
        head_size=DIM,
        scale=SCALE,
        device=torch.device("npu", 0),
        dtype=DTYPE,
        kv_tile_pages=0,
        max_num_reqs=BATCH_CAP,
        # 预算 256:ceil(256*8/128)+4 = 20 → ladder top = 32(变体 {1,2,4,8,16,32})
        max_num_batched_tokens=256,
    )
    key_cache = torch.zeros((ENGINE_PAGES, ENGINE_BLOCK, KVH, DIM), dtype=DTYPE)
    t0 = time.time()
    built = instance.prewarm_prefill(ENGINE_BLOCK, key_cache)
    elapsed = time.time() - t0
    ready = len(PrefillShapeManager._ready)
    assert ready <= PrefillShapeManager._cap, "变体数超上限"
    assert ready >= 6, f"ladder 预热后 ready 变体过少: {ready}"
    print(f"prewarm: built={built} ready={ready} elapsed={elapsed:.1f}s")

    # ladder 内任意非 2 幂形状的 canonical spec 都应即时可用(prefill_step 量化 t→2 的幂)
    for qo in (1, 17, 97, 257, 397, 512):  # t = 1,2,7,17,25,32 → T = 1,2,8,32,32,32
        t = _next_pow2(-(-qo * GROUP // 128))
        assert _canonical_ready(t), f"tiles={t} (qo={qo}) 的 canonical 变体未被预热覆盖"

    # 首见形状(qo=397,本进程从未跑过该 qo)首步:立即接管且墙钟 <5s(引擎验收 API 级等价)
    got, elapsed = _run_case([397], [397], 7)
    assert got is True, "预热后首见形状首步未接管"
    assert elapsed < 5.0, f"首见形状首步耗时 {elapsed:.2f}s(应 <5s,无惰性编译窗)"
    print(f"first-seen qo=397: served in {elapsed:.3f}s")
