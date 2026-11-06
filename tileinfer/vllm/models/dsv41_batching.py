"""把 vLLM 的扁平 token 批切成按请求的片段（P3 并发改造第 1 步）。

背景（`docs/notes/dsv41-perf-plan.md` §13）：我们的 `forward` 目前
`ids = input_ids.reshape(1, -1)` 且只取 `positions[0]`，等于把所有 token 当成一条序列。
当 `--max-num-seqs ≥ 2` 时 vLLM 会把多个请求的 token 拍平成一个 `[num_tokens]` 传进来，
必须按请求边界切开，每个请求用**自己**的 :class:`ModelState`。

本模块只做纯粹的边界计算，不碰模型，因此可以离线单测；`forward` 暂不改用
（避免在没有真机窗口的情况下动已验证路径），等 §13 的第 2、3 步一起接。

vLLM v1 的 `attn_metadata` 形态在不同版本里有差异，这里对三种常见字段都做兼容：

* ``query_start_loc``：``[num_reqs + 1]`` 的累积起点（vLLM v1 主用）；
* ``seq_lens`` / ``seq_lens_cpu``：每个请求的长度；
* 都没有时返回 ``None``，调用方退回"单序列"行为（保护已验证路径）。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

Slice = Tuple[int, int]


def _to_list(value) -> Optional[List[int]]:
    """把 tensor / numpy / list 统一成 python int 列表；失败返回 None。"""
    if value is None:
        return None
    try:
        if hasattr(value, "tolist"):
            out = value.tolist()
        else:
            out = list(value)
    except Exception:  # noqa: BLE001 - 任何异常都按"不可用"处理
        return None
    if not out:
        return None
    try:
        return [int(v) for v in out]
    except Exception:  # noqa: BLE001
        return None


def request_slices(attn_metadata, num_tokens: int) -> Optional[List[Slice]]:
    """算出每个请求在扁平批里的 ``(start, end)``；无法判断时返回 ``None``。

    返回 ``None`` 表示"按单序列处理"——这是**当前已验证**的行为，调用方必须据此走老路径。
    只有一个请求时返回长度为 1 的列表，语义与单序列一致。
    """
    if attn_metadata is None or num_tokens <= 0:
        return None

    # 1) vLLM v1 的累积起点
    starts = _to_list(getattr(attn_metadata, "query_start_loc", None))
    if starts is None:
        # 有些版本把它放在嵌套的 attention metadata 里
        for attr in ("attn_metadata", "impl"):
            inner = getattr(attn_metadata, attr, None)
            if inner is not None:
                starts = _to_list(getattr(inner, "query_start_loc", None))
                if starts is not None:
                    break

    if starts is not None and len(starts) >= 2:
        # 累积起点是 [0, l0, l0+l1, ...]；末项应等于 num_tokens（允许不等的容错）
        sl: List[Slice] = []
        for i in range(len(starts) - 1):
            a, b = starts[i], starts[i + 1]
            if b < a:
                return None
            sl.append((max(0, a), min(num_tokens, b)))
        # 丢弃空片段
        sl = [s for s in sl if s[1] > s[0]]
        return sl or None

    # 2) 退而用每个请求的长度
    lens = _to_list(getattr(attn_metadata, "seq_lens", None))
    if lens is None:
        lens = _to_list(getattr(attn_metadata, "seq_lens_cpu", None))
    if lens is None:
        return None
    total = sum(lens)
    if total != num_tokens:
        # decode 阶段 seq_lens 是"含历史的总长"，不是本批 token 数——这种情况我们不猜
        return None
    sl = []
    off = 0
    for n in lens:
        if n < 0:
            return None
        sl.append((off, off + n))
        off += n
    sl = [s for s in sl if s[1] > s[0]]
    return sl or None


def slices_from_counts(counts: Optional[Sequence[int]], num_tokens: int) -> Optional[List[Slice]]:
    """按每请求 token 数（与 ``num_scheduled_tokens`` 同序）算出片段；不一致时返回 ``None``。

    这是并发路径的**主判据**：`attn_metadata` 在我们这套栈里可能是 ``None``（没有 attention
    backend，见 perf-plan §16/§18），靠它判断并发会让多请求走进单序列路径并崩溃。
    """
    if not counts or num_tokens <= 0:
        return None
    try:
        vals = [int(c) for c in counts]
    except Exception:  # noqa: BLE001
        return None
    if any(c < 0 for c in vals) or sum(vals) != num_tokens:
        return None
    slices: List[Slice] = []
    off = 0
    for c in vals:
        if c > 0:
            slices.append((off, off + c))
        off += c
    return slices or None


def is_single(slices: Optional[Sequence[Slice]], num_tokens: int) -> bool:
    """是否等价于"整批就是一条序列"（可以走已验证的老路径）。"""
    if not slices:
        return True
    return len(slices) == 1 and slices[0][0] == 0 and slices[0][1] == num_tokens


def run_batched(
    slices: Sequence[Slice],
    ids,
    positions,
    acquire,
    forward_one,
) -> Tuple[List[object], List[object]]:
    """按请求逐段调用 ``forward_one``，返回 ``(每请求的输出, 每请求的最后一个 token)``。

    参数是**回调**而非具体对象，这样接线前就能用 fake 单测：

    * ``acquire(start, end)``：为该片段取一个 state（真实现里走 :class:`StateSlots`）；
    * ``forward_one(state, ids_slice, pos_slice)``：跑一个请求的前向，返回该段的 hidden。

    ``pos_slice`` 用**该段自己的**起始位置（并发时各请求的位置互不相同，这正是当前实现
    `positions[0]` 的错误来源）。

    任何一段拿不到 state（``acquire`` 返回 ``None``）都直接抛 ``RuntimeError``——
    宁可让请求排队/失败，也不能悄悄算错（与 §13 的风险说明一致）。
    """
    per_req = []
    last_tok = []
    for start, end in slices:
        if end <= start:
            continue
        state = acquire(start, end)
        if state is None:
            raise RuntimeError(
                "StateSlots 已满：无法为请求分配 state，请降低 --max-num-seqs 或调大 max_slots"
            )
        out = forward_one(state, ids[start:end], positions[start:end])
        per_req.append(out)
        last_tok.append(out[-1] if len(out) else out)
    return per_req, last_tok

