"""`dsv41_req_ids` 钩子的离线单测（P3 前置）。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tileinfer.vllm.models import dsv41_req_ids as rid


class FakeRunner:
    def __init__(self):
        self.calls = []

    def execute_model(self, scheduler_output, extra=None):
        self.calls.append(scheduler_output)
        return f"out:{len(scheduler_output.num_scheduled_tokens)}"


def make_sched(keys):
    return SimpleNamespace(num_scheduled_tokens={k: 1 for k in keys})


@pytest.fixture(autouse=True)
def _clean():
    rid.reset()
    yield
    rid.reset()
    rid.uninstall(FakeRunner)


def test_install_records_req_ids_and_keeps_return_value():
    assert rid.install(FakeRunner) is True
    runner = FakeRunner()
    out = runner.execute_model(make_sched(["r1", "r2"]))
    assert out == "out:2"                  # 返回值不被改变
    assert rid.current_req_ids() == ["r1", "r2"]


def test_install_is_idempotent():
    assert rid.install(FakeRunner) is True
    assert rid.install(FakeRunner) is False    # 第二次不重复包装
    runner = FakeRunner()
    runner.execute_model(make_sched(["a"]))
    assert rid.current_req_ids() == ["a"]


def test_ids_update_per_step():
    rid.install(FakeRunner)
    runner = FakeRunner()
    runner.execute_model(make_sched(["a", "b", "c"]))
    assert rid.current_req_ids() == ["a", "b", "c"]
    runner.execute_model(make_sched(["c"]))
    assert rid.current_req_ids() == ["c"]


def test_empty_schedule_sets_empty_list():
    """空调度（有字段但没请求）→ 记录空列表，而不是 None。"""
    rid.install(FakeRunner)
    runner = FakeRunner()
    out = runner.execute_model(SimpleNamespace(num_scheduled_tokens={}))
    assert out == "out:0"
    assert rid.current_req_ids() == []


def test_missing_field_sets_none():
    """字段完全缺失 → 钩子置 None，且不影响被包装方法自身的行为。"""

    class TolerantRunner:
        def execute_model(self, scheduler_output, extra=None):
            return "ok"

    rid.install(TolerantRunner)
    try:
        out = TolerantRunner().execute_model(SimpleNamespace())   # 没有 num_scheduled_tokens
        assert out == "ok"
        assert rid.current_req_ids() is None
    finally:
        rid.uninstall(TolerantRunner)


def test_hook_never_breaks_on_weird_scheduler_output():
    rid.install(FakeRunner)

    class Boom:
        @property
        def num_scheduled_tokens(self):
            raise RuntimeError("boom")

    runner = FakeRunner()
    # 钩子吞掉异常，但被包装方法本身仍会被调用（这里会因拿不到字段而报错，属预期）
    with pytest.raises(RuntimeError):
        runner.execute_model(Boom())
    assert rid.current_req_ids() is None


def test_uninstall_restores_original():
    rid.install(FakeRunner)
    assert rid.uninstall(FakeRunner) is True
    rid.reset()
    runner = FakeRunner()
    runner.execute_model(make_sched(["x"]))
    assert rid.current_req_ids() is None     # 还原后不再记录
    assert rid.uninstall(FakeRunner) is False


def test_install_without_vllm_returns_false_on_bad_class():
    assert rid.install(object) is False      # 没有 execute_model


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
