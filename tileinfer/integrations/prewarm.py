"""配置驱动的预热规格推导(T6):把「引擎需要哪些内核变体」变成一条可复算的纯公式。

背景 / 为什么需要这个模块
------------------------
TileLang 内核按形状编译(冷编译 ~100 s/变体),因此工程上有两条铁律:

1. **服务路径上永不编译** —— 否则会把 EngineCore 卡住(看起来像引擎 hang)。做法是启动时
   把「引擎在整个服务期可能用到的全部变体」一次编译进 ``TILELANG_CACHE_DIR``;
2. **预烧(out-of-process)必须与引擎推导出完全相同的规格** —— 否则预热"命中不了",服务
   路径又会掉回惰性编译(FIA 回退)。

历史做法的问题(本模块要解决的):预烧脚本 ``scripts/prewarm-tileinfer-kernels.py`` 只
覆盖 decode,且参数(池大小/桶/ladder 顶)**要从日志里手抄**;引擎侧另有一份内联公式
(``TileInferDecodeAttention.prewarm_prefill``)。两处公式一旦漂移,预热静默失效——
这正是 T6 要消除的「隐式双真源」。

本模块把公式抽成**纯函数**(不 import torch / tilelang / vllm),让引擎与预烧脚本调用
同一份实现;形状常量(BATCH_BUCKETS / PREFILL_MAX_TILES / SPLIT_MS_COMPILE)以参数传入,
避免与集成层产生循环导入。

术语
----
* **bucket**:批量档位。批大小每步都变,内核按形状编译,所以把批量 padding 到最近的档位
  (``BATCH_BUCKETS``),每个档位编译一次。
* **ladder**:prefill 的 tile 维档位,取 2 的幂 1,2,4,...,top。tile 维 padding 到 2 的幂,
  使有限个 canonical 变体覆盖任意 prompt 长度(P3/D39)。
* **pool cap**:内核 ABI 的「声明容量」,对真实池向上取整到 2 的幂(或 ``TILEINFER_POOL_CAP``
  钉死)。池大小从 vLLM 内存快照推导、逐次运行会漂移,所以把它从内核/plan/预热 key 里
  移除(L2 池无关 ABI),预烧才能跨运行命中。
* **ktp**:KV pages per split tile。decode 侧按桶策略(H21);prefill 侧为全局
  ``TILEINFER_PREFILL_KV_TILE_PAGES``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

#: 内核页的 token 数:TyileLang 内核以 128-token 页为 N tile(见 vllm_ascend.KERNEL_PAGE_SIZE)。
KERNEL_PAGE_SIZE = 128

#: prefill 内核的 query 行块默认值(与 vllm_ascend 的 canonical 变体一致)。
PREFILL_BLOCK_Q = 128


def next_pow2(n: int) -> int:
    """最小的 ``>= n`` 的 2 的幂(1 保持 1);与 ``vllm_ascend._next_pow2`` 同义。"""
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def sub_pages(block_size: int) -> int:
    """一个引擎页含多少个 128-token 子页(>=1;1 表示无需变换)。

    混合模型(Qwen3.5/3.6 带 GDN 层)会把引擎页对齐到 mamba 状态页(如 2048 token),而内核
    的页必须是 128 token(64 KB L0 容量)。``block_size = k*128`` 的页在内存里就是 k 个连续
    的 128-token 子页,故池以零拷贝子页视图呈现、页表展开为子页 id。
    """
    if block_size % KERNEL_PAGE_SIZE != 0:
        raise ValueError(
            f"block_size={block_size} 不是 {KERNEL_PAGE_SIZE} 的整数倍;内核无法分页"
        )
    return max(1, block_size // KERNEL_PAGE_SIZE)


def bucket_for(num_reqs: int, buckets: Tuple[int, ...]) -> int:
    """第一个 ``>= num_reqs`` 的桶;超出最大桶则报错(与 ``vllm_ascend._bucket_for`` 同义)。"""
    for b in buckets:
        if b >= num_reqs:
            return b
    raise ValueError(
        f"batch={num_reqs} 超过最大桶 {buckets[-1]};请向 BATCH_BUCKETS 追加更大档位"
    )


def decode_buckets(max_num_reqs: int, buckets: Tuple[int, ...]) -> List[int]:
    """引擎会为哪些桶建 decode 计划:``<= max_num_reqs`` 的全部桶。

    ``max_num_reqs<=0`` 表示配置不可得,退回「所有桶都建」(命中的是磁盘缓存,代价一次性)。
    """
    limit = int(max_num_reqs) if max_num_reqs else buckets[-1]
    return [b for b in buckets if b <= limit]


def prefill_ladder_tops(
    *,
    max_num_reqs: int,
    max_num_batched_tokens: int,
    max_model_len: int,
    group_size: int,
    buckets: Tuple[int, ...],
    prefill_max_tiles: int,
    prefill_kv_tile_pages: int,
    default_budget: int = 8192,
) -> Tuple[int, int]:
    """prefill ladder 的两个顶:``(tiles_top, tiles_top_split)``。

    与 ``TileInferDecodeAttention.prewarm_prefill`` 逐项对齐:

    * 一个混合步最多带 ``budget = max_num_batched_tokens`` 个 token:其中 ``budget-n`` 个
      prefill token(每行 group 个 → /128 即 tile 数)加 ``n`` 个单行 decode append(各 1 tile);
      顶还要加上 ``limit``(桶)把逐请求开销算作上界 → ``next_pow2(budget*group/128 + limit)``;
    * split ladder 顶 = plain 顶 × 长上下文下的逐请求 split 因子(``max_pages/ktp + 1``,再取
      2 的幂);没有它,长上下文步会越过 ladder 顶去抢一段 ~3.5 min 的惰性编译。
    """
    limit = int(max_num_reqs) if max_num_reqs else buckets[-1]
    budget = int(max_num_batched_tokens) or default_budget
    tiles_top = next_pow2(-(-budget * group_size // KERNEL_PAGE_SIZE) + limit)
    tiles_top = min(tiles_top, prefill_max_tiles)
    tiles_top_split = tiles_top
    if prefill_kv_tile_pages > 0:
        factor = 1
        if max_model_len:
            max_pages = -(-int(max_model_len) // KERNEL_PAGE_SIZE)
            factor = -(-max_pages // prefill_kv_tile_pages) + 1
        tiles_top_split = min(next_pow2(tiles_top * factor), prefill_max_tiles)
    return tiles_top, tiles_top_split


def prefill_specs(
    *,
    num_kv_heads: int,
    group_size: int,
    head_size: int,
    pool_pages: int,
    dtype: str,
    max_num_reqs: int,
    max_num_batched_tokens: int,
    max_model_len: int,
    buckets: Tuple[int, ...],
    prefill_max_tiles: int,
    prefill_kv_tile_pages: int,
    split_ms_compile: int,
    block_q: int = PREFILL_BLOCK_Q,
) -> List[tuple]:
    """引擎会为哪些 prefill 变体建 spec(plain + split 两族,ladder 逐级 2 的幂)。

    与 ``TileInferDecodeAttention._prewarm_prefill_ladder`` 逐项对齐:
    ``kernel_batch = bucket_for(limit)+1``(真实批位 + 1 个 trash 行);ladder 从 1 到
    ``max(tiles_top, tiles_top_split)`` 逐级翻倍;split 变体在 ``tiles <= tiles_top_split``
    时追加(第 10 个字段 = ``max_splits``)。规格元组顺序必须与 ``PrefillShapeManager._compile``
    解包一致。
    """
    limit = int(max_num_reqs) if max_num_reqs else buckets[-1]
    kernel_batch = bucket_for(limit, buckets) + 1
    tiles_top, tiles_top_split = prefill_ladder_tops(
        max_num_reqs=max_num_reqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        group_size=group_size,
        buckets=buckets,
        prefill_max_tiles=prefill_max_tiles,
        prefill_kv_tile_pages=prefill_kv_tile_pages,
    )
    specs: List[tuple] = []
    tiles = 1
    while tiles <= max(tiles_top, tiles_top_split):
        plain = (
            kernel_batch,
            num_kv_heads,
            group_size,
            head_size,
            KERNEL_PAGE_SIZE,
            pool_pages,
            tiles,
            block_q,
            dtype,
        )
        specs.append(plain)
        if prefill_kv_tile_pages > 0 and tiles <= tiles_top_split:
            specs.append(plain + (split_ms_compile,))
        tiles *= 2
    return specs


@dataclass
class PrewarmPlan:
    """一次预烧的完整规格集:可打印、可测试、可直接喂给编译函数。"""

    buckets: List[int]
    pool_pages: int
    decode: List[int] = field(default_factory=list)
    prefill: List[tuple] = field(default_factory=list)
    notes: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        n_split = sum(1 for s in self.prefill if len(s) == 10)
        return (
            f"decode buckets={self.decode} | prefill variants={len(self.prefill)} "
            f"(split {n_split}) | pool_pages={self.pool_pages}"
        )


def build_prewarm_plan(
    *,
    num_kv_heads: int,
    group_size: int,
    head_size: int,
    dtype: str,
    block_size: int,
    engine_pages: int,
    max_num_reqs: int,
    max_num_batched_tokens: int,
    max_model_len: int,
    buckets: Tuple[int, ...],
    prefill_max_tiles: int,
    prefill_kv_tile_pages: int,
    split_ms_compile: int,
    pool_pages_override: Optional[int] = None,
) -> PrewarmPlan:
    """从服务配置推导完整预烧规格(单一真源,T6)。

    ``engine_pages`` 是 vLLM 的 KV 池页数(引擎页);内核按**声明容量**编译,故这里调用方
    传入的 ``engine_pages`` 会先经子页展开再交给 :func:`declared_cap`。``pool_pages_override``
    对应 ``TILEINFER_POOL_CAP`` 钉死。
    """
    pool_pages = declared_cap(
        engine_pages * sub_pages(block_size), pool_pages_override
    )
    plan = PrewarmPlan(buckets=decode_buckets(max_num_reqs, buckets), pool_pages=pool_pages)
    plan.decode = list(plan.buckets)
    plan.prefill = prefill_specs(
        num_kv_heads=num_kv_heads,
        group_size=group_size,
        head_size=head_size,
        pool_pages=pool_pages,
        dtype=dtype,
        max_num_reqs=max_num_reqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        buckets=buckets,
        prefill_max_tiles=prefill_max_tiles,
        prefill_kv_tile_pages=prefill_kv_tile_pages,
        split_ms_compile=split_ms_compile,
    )
    plan.notes = {
        "engine_pages": int(engine_pages),
        "sub_pages": sub_pages(block_size),
        "prefill_max_tiles": int(prefill_max_tiles),
        "prefill_kv_tile_pages": int(prefill_kv_tile_pages),
    }
    return plan


def declared_cap(pool_pages: int, pin: Optional[int] = None) -> int:
    """内核 ABI 的声明容量:``pin`` 优先,否则向上取整到 2 的幂。

    与 ``kernels.attention.paged_decode_ascend950.declared_pool_cap`` 同义(此处复刻以避免
    在纯 CPU 推导时 import torch 侧内核模块)。
    """
    pool_pages = int(pool_pages)
    if pin:
        pin = int(pin)
        if pin < pool_pages:
            raise ValueError(
                f"TILEINFER_POOL_CAP={pin} 小于真实池({pool_pages} 页);声明容量必须覆盖所有活页"
            )
        return pin
    return 1 if pool_pages <= 1 else 1 << (pool_pages - 1).bit_length()


def prewarm_plan_to_dict(plan: PrewarmPlan) -> dict:
    """把 :class:`PrewarmPlan` 转成可 JSON 序列化的字典(``--plan-only``/``--json`` 用)。

    这是「预烧规格」的机器可读形态:可在不编译的机器上复算、diff、入库,把预烧与引擎的
    一致性从"人工比对日志"变成"可复现的产物对比"。
    """
    return {
        "buckets": [int(b) for b in plan.decode],
        "pool_pages": int(plan.pool_pages),
        "decode": [int(b) for b in plan.decode],
        "prefill": [list(s) for s in plan.prefill],
        "prefill_plain": sum(1 for s in plan.prefill if len(s) == 9),
        "prefill_split": sum(1 for s in plan.prefill if len(s) == 10),
        "notes": {k: int(v) for k, v in plan.notes.items()},
    }


