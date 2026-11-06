"""投机解码（路线 B：n-gram / prompt-lookup）的提议器与接受判定。

为什么先做这条路线（`docs/notes/dsv41-perf-plan.md` §20）：

* **不需要任何新权重**——提议来自文本自身的历史 n-gram 匹配，而 MTP/DSpark 路线要忠实复现
  DeepSeek 的 markov 头提议算法，工作量大得多；
* 验证侧的能力已经就绪（`layer.py` 的分块解码：一次前向吃 k 个 token）；
* 全部逻辑是纯 Python，可离线单测，不占卡。

收益特征（必须实测，不能假设）：对重复/引用性文本（长文档问答、代码、模板化输出）有效，
对自由创作几乎为 0。所以启用后必须报告**接受率**（accepted / proposed）。

贪心解码下投机结果必须与逐 token 贪心**完全一致**：验证时用模型对 k 个候选位置各自算出的
argmax 与提议逐个比对，只接受**最长相同前缀**，第一个不一致的位置由模型自己的 argmax 顶上。
"""

from __future__ import annotations

from typing import List, Sequence


def propose_ngram(history: Sequence[int], k: int, n: int = 3) -> List[int]:
    """在 ``history`` 里找与末尾 n-gram 相同的最近一次出现，复制其后最多 ``k`` 个 token。

    返回的列表长度 0..k：
    * 历史长度 < n 或找不到匹配 → 空列表（不提议，退回普通解码）；
    * 找到多处时取**最近**一处（更可能重复）。

    纯整数运算，可离线单测。
    """
    if k <= 0 or n <= 0:
        return []
    hist = [int(x) for x in history]
    if len(hist) < n + 1:
        return []
    needle = hist[-n:]
    # 从倒数第 n+1 个位置往前找最近一次相同 n-gram
    for start in range(len(hist) - n - 1, -1, -1):
        if hist[start : start + n] == needle:
            tail = hist[start + n : start + n + k]
            if tail:
                return tail
            return []
    return []


def verify_prefix(proposed: Sequence[int], sampled: Sequence[int]) -> int:
    """贪心验证：返回 ``proposed`` 与模型 ``sampled`` 的最长相同前缀长度。

    ``sampled[i]`` 是模型在"第 i 个候选位置"上的 argmax。接受 ``accepted`` 个后，
    第 ``accepted`` 个位置改用 ``sampled[accepted]``（若存在），这样结果与逐 token 贪心一致。
    """
    cnt = 0
    for p, s in zip(proposed, sampled):
        if int(p) != int(s):
            break
        cnt += 1
    return cnt


def generate(history: Sequence[int], step_fn, max_new: int, propose_fn=None, k: int = 0):
    """跑一段贪心生成；``propose_fn`` 非空时启用投机。返回 ``(tokens, stats)``。

    ``step_fn(tokens) -> List[int]``：把 ``tokens`` 依次接到当前前缀后，返回模型在**每个位置**上的
    贪心 argmax（teacher-forced；长度等于 ``len(tokens)``）。这正是真实实现里"一次前向吃 k 个候选、
    取每个位置 argmax"的语义，所以本函数可以用 fake ``step_fn`` 完整单测。

    关键性质：启用投机后输出序列与 ``propose_fn=None`` 时**完全相同**——投机只减少前向次数。

    ``stats`` 记录 ``{"proposed": n, "accepted": n, "steps": n}``，用于报告接受率
    （目标要求：不接受率就无法判断收益来源）。
    """
    out: List[int] = []
    stats = {"proposed": 0, "accepted": 0, "steps": 0}
    while len(out) < max_new:
        prefix = list(history) + out
        cand: List[int] = []
        if propose_fn is not None and k > 0:
            cand = list(propose_fn(prefix))[:k]
            cand = cand[: max_new - len(out)]          # 不越过请求长度
        stats["proposed"] += len(cand)
        stats["steps"] += 1

        # 验证：把候选（为空时用 1 个占位）喂给 step_fn，拿各位置 argmax
        fed = cand if cand else [0]
        sampled = list(step_fn(fed))
        if not sampled:
            break

        accepted = verify_prefix(cand, sampled) if cand else 0
        stats["accepted"] += accepted
        out.extend(cand[:accepted])
        if accepted < len(sampled):
            out.append(sampled[accepted])              # 模型自己的 argmax 顶上
        out = out[:max_new]
    return out, stats


def split_accept(proposed: Sequence[int], sampled: Sequence[int]) -> tuple[List[int], int]:
    """返回 ``(实际采用的 token 序列, 被接受的提议数)``。

    采用序列 = 被接受的前缀 + 第一个不一致位置上的模型 argmax（若存在）。
    这保证"投机"不会改变贪心结果，只改变一次前向能产出几个 token。
    """
    accepted = verify_prefix(proposed, sampled)
    out = [int(x) for x in proposed[:accepted]]
    if accepted < len(sampled):
        out.append(int(sampled[accepted]))
    return out, accepted
