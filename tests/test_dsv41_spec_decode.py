"""`spec_decode`（路线 B：n-gram 投机解码）的离线单测。

关键性质：**贪心一致性**——采用序列必须与"逐 token 取 argmax"完全相同，
投机只影响一次前向产出几个 token，不改变结果。
"""

from __future__ import annotations

import pytest

from tileinfer.models.dsv41.spec_decode import (
    generate,
    propose_ngram,
    split_accept,
    verify_prefix,
)


# ---------------- 提议器 ----------------
def test_no_history_returns_empty():
    assert propose_ngram([], 3) == []
    assert propose_ngram([1, 2], 3, n=3) == []


def test_proposes_continuation_after_repeat():
    # 历史 [1,2,3,4,5,1,2,3] 末尾 3-gram 是 (1,2,3)，最近一次出现在 index 0
    # → 其后 3 个 token 是 [4,5,1]（跨越到旧内容的续写，属正确行为）
    hist = [1, 2, 3, 4, 5, 1, 2, 3]
    assert propose_ngram(hist, k=3, n=3) == [4, 5, 1]


def test_respects_k_limit():
    hist = [7, 8, 9, 1, 2, 3, 7, 8, 9]
    assert propose_ngram(hist, k=1, n=3) == [1]
    # k 大于可用续写长度时，返回能拿到的全部（这里跨到旧内容，共 6 个）
    assert propose_ngram(hist, k=10, n=3) == [1, 2, 3, 7, 8, 9]
    assert propose_ngram(hist, k=3, n=3) == [1, 2, 3]


def test_uses_most_recent_match():
    hist = [5, 6, 1, 2, 5, 6, 3, 4, 5, 6]
    # 末尾 (5,6)，最近一次出现在 index 4 -> 后续是 [3,4]
    assert propose_ngram(hist, k=2, n=2) == [3, 4]


def test_no_match_returns_empty():
    assert propose_ngram([1, 2, 3, 4, 5, 6], k=2, n=3) == []


def test_degenerate_arguments():
    assert propose_ngram([1, 2, 3], k=0) == []
    assert propose_ngram([1, 2, 3], k=2, n=0) == []


# ---------------- 验证与接受 ----------------
def test_verify_prefix_counts_matching_head():
    assert verify_prefix([4, 5, 6], [4, 5, 9]) == 2
    assert verify_prefix([4, 5], [9]) == 0
    assert verify_prefix([4, 5], [4, 5]) == 2
    assert verify_prefix([], [1]) == 0


def test_split_accept_matches_greedy_when_all_accepted():
    """全部提议被接受，且没有额外位置 → 结果就是提议本身（与逐 token 贪心一致）。"""
    got, acc = split_accept([4, 5, 6], [4, 5, 6])
    assert got == [4, 5, 6]
    assert acc == 3


def test_split_accept_replaces_first_mismatch_with_model_choice():
    """第一个不一致处用模型自己的 argmax，保证结果等于纯贪心。"""
    got, acc = split_accept([4, 5, 6], [4, 9, 1])
    assert acc == 1
    assert got == [4, 9]      # 接受 4，第 2 位用模型的 9


def test_split_accept_all_rejected():
    got, acc = split_accept([4, 5], [9, 1])
    assert acc == 0
    assert got == [9]


def test_split_accept_no_proposal_is_plain_greedy():
    got, acc = split_accept([], [7])
    assert got == [7]
    assert acc == 0


@pytest.mark.parametrize("proposed,sampled", [
    ([1, 2, 3], [1, 2, 3]),
    ([1, 2, 3], [1, 2, 0]),
    ([1, 2], [0, 0]),
    ([], [5]),
])
def test_result_equals_stepwise_greedy(proposed, sampled):
    """无论接受多少，采用序列都必须等于"逐 token argmax"的结果。"""
    got, acc = split_accept(proposed, sampled)
    # 纯贪心的期望：从 sampled 的第 0 位开始，逐位取（sampled 就是模型在各位置的 argmax）
    expected = list(sampled)
    # 但只有在 proposals 与 sampled 前缀相同时，get 才可能更短；对齐校验：
    assert got == expected[: len(got)]
    assert acc == verify_prefix(proposed, sampled)
    assert len(got) == min(len(sampled), acc + 1) or len(got) == len(sampled)


# ---------------- 生成循环：投机 == 纯贪心 ----------------
SEQ = [7, 8, 9, 7, 8, 9, 7, 8, 9, 1, 2, 3]


def _counter_step():
    """确定性的"模型"：第 i 个位置的贪心 argmax 就是 SEQ[i]，与喂入内容无关。

    这正好模拟 teacher-forced 验证：一次喂 k 个候选，返回 k 个位置各自的 argmax。
    """
    state = {"pos": 0}

    def step_fn(tokens):
        base = state["pos"]
        state["pos"] = base + len(tokens)
        return [SEQ[base + i] for i in range(len(tokens))]

    return step_fn


def test_generate_plain_greedy():
    got, stats = generate([], _counter_step(), 5, propose_fn=None)
    assert got == SEQ[:5]
    assert stats["proposed"] == 0
    assert stats["steps"] == 5          # 纯贪心：一步一个 token


def test_generate_speculative_matches_plain_greedy():
    """核心性质：启用 n-gram 投机后输出必须与纯贪心完全相同，且步数不增加。"""
    plain, _ = generate([], _counter_step(), 8, propose_fn=None)

    hist = [7, 8, 9]                     # 让 n-gram 能命中 SEQ 的重复结构
    spec, stats = generate(hist, _counter_step(), 8, propose_fn=lambda p: propose_ngram(p, 4, 3), k=4)

    assert spec == plain, (spec, plain)
    assert stats["steps"] <= 8
    assert stats["accepted"] <= stats["proposed"]
    assert stats["accepted"] > 0         # 这组数据里 n-gram 应当命中


def test_generate_respects_max_new():
    got, _ = generate([], _counter_step(), 3, propose_fn=lambda p: propose_ngram(p, 4, 3), k=4)
    assert len(got) == 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
