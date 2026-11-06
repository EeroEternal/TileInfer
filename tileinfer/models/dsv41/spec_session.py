"""投机解码的接线适配层（把"k 个候选一次前向"包装成 `generate` 需要的 step_fn）。

分层（`docs/notes/dsv41-perf-plan.md` §21）：

```
transformer_forward(model, tok_ids, state, start_pos)   # 一次吃 k 个候选
        ↓  capture["h_all"]（全部位置的 hidden）
logits_all_positions(model, h_all)                      # [1, k, vocab]
        ↓  argmax(-1)
spec_decode.generate(history, step_fn, ...)             # 提议 → 验证 → 接受
```

本模块只做最后一层适配，**不依赖具体模型**：外部注入 `forward_k(tokens) -> [k, vocab]`，
因此可以用 fake 前向完整单测（不需要 NPU、不需要真权重）。

状态推进（`start_pos += accepted + 1`）由调用方负责——适配层不持有 state，避免把
"一次验证"和"状态生命周期"耦在一起。
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from .spec_decode import generate


def make_step_fn(forward_k: Callable[[Sequence[int]], "object"]) -> Callable[[Sequence[int]], List[int]]:
    """把 ``forward_k`` 包装成 ``generate`` 需要的 ``step_fn``。

    ``forward_k(tokens)`` 返回 ``[k, vocab]`` 的逐位置 logits（list 或带 ``argmax`` 的对象）。
    ``step_fn(tokens)`` 返回每位置的贪心 argmax（list[int]）。

    关键点：喂进去的候选数量决定返回的 argmax 数量——这正是 teacher-forced 验证的语义，
    也是 `verify_prefix` 能逐位比对的前提。
    """

    def step_fn(tokens: Sequence[int]) -> List[int]:
        logits = forward_k(tokens)
        if logits is None:
            return []
        # 支持 torch 张量 / 嵌套 list
        if hasattr(logits, "argmax"):
            return [int(x) for x in logits.argmax(dim=-1).reshape(-1).tolist()]
        return [int(max(range(len(row)), key=lambda j: row[j])) for row in logits]

    return step_fn


def run_speculative(
    forward_k: Callable[[Sequence[int]], "object"],
    history: Sequence[int],
    max_new: int,
    propose_fn=None,
    k: int = 0,
) -> Tuple[List[int], dict]:
    """跑一段投机生成，返回 ``(tokens, stats)``。

    ``propose_fn`` 为 ``None`` 时退化为纯贪心（每步 1 个 token），用于对拍基线。
    """
    step_fn = make_step_fn(forward_k)
    return generate(history, step_fn, max_new, propose_fn=propose_fn, k=k)


def acceptance_rate(stats: dict) -> float:
    """接受率 ``accepted / proposed``（无提议时返回 0.0）。

    目标要求报告这个数：没有它就无法判断投机收益是来自"提议命中"还是"碰巧"。
    """
    proposed = int(stats.get("proposed", 0))
    if proposed <= 0:
        return 0.0
    return float(stats.get("accepted", 0)) / proposed
