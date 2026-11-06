"""`spec_session`（接线适配层）的离线单测：用 fake 前向验证适配与生成语义。

不需要 NPU、不需要真权重：`forward_k(tokens)` 由测试注入。
"""

from __future__ import annotations

import pytest

from tileinfer.models.dsv41.spec_decode import propose_ngram
from tileinfer.models.dsv41.spec_session import (
    acceptance_rate,
    make_step_fn,
    run_speculative,
)

VOCAB = 12


def _onehot(argmax_ids):
    """把 argmax 列表变成 [k, vocab] 的 one-hot 行（模拟 logits）。"""
    rows = []
    for a in argmax_ids:
        row = [0.0] * VOCAB
        row[a] = 1.0
        rows.append(row)
    return rows


def _make_forward(seq):
    """确定性"模型"：第 i 个位置的 argmax 恒为 seq[i]（与输入内容无关）。"""
    state = {"pos": 0}

    def forward_k(tokens):
        base = state["pos"]
        state["pos"] = base + len(tokens)
        ids = [seq[base + i] % VOCAB for i in range(len(tokens))]
        return _onehot(ids)

    return forward_k


def test_step_fn_takes_argmax_per_position():
    step = make_step_fn(lambda toks: _onehot([3, 7, 1][: len(toks)]))
    assert step([5, 5, 5]) == [3, 7, 1]
    assert step([9]) == [3]


def test_step_fn_empty_input():
    step = make_step_fn(lambda toks: _onehot([]))
    assert step([]) == []


def test_run_speculative_plain_is_greedy():
    seq = [1, 2, 3, 4, 5, 6]
    got, stats = run_speculative(_make_forward(seq), [], 4, propose_fn=None, k=0)
    assert got == seq[:4]
    assert stats["steps"] == 4
    assert acceptance_rate(stats) == 0.0


def test_run_speculative_matches_plain_and_reports_rate():
    seq = [7, 8, 9, 7, 8, 9, 7, 8, 9, 2, 3]
    plain, _ = run_speculative(_make_forward(seq), [], 8, propose_fn=None, k=0)

    hist = [7, 8, 9]
    spec, stats = run_speculative(
        _make_forward(seq), hist, 8,
        propose_fn=lambda p: propose_ngram(p, 4, 3), k=4,
    )
    assert spec == plain, (spec, plain)
    assert stats["steps"] <= 8
    rate = acceptance_rate(stats)
    assert 0.0 <= rate <= 1.0
    assert rates_are_consistent(stats, rate)


def rates_are_consistent(stats, rate):
    if stats["proposed"] == 0:
        return rate == 0.0
    return abs(rate - stats["accepted"] / stats["proposed"]) < 1e-9


def test_acceptance_rate_edge_cases():
    assert acceptance_rate({"proposed": 0, "accepted": 0}) == 0.0
    assert acceptance_rate({}) == 0.0
    assert acceptance_rate({"proposed": 4, "accepted": 1}) == pytest.approx(0.25)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
