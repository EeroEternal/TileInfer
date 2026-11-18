import pytest

from tools.nemotron.check_c2_teacher import check


def _hf(steps):
    return {"p": {"steps": [[{"tok": "a", "logit": 5.0, "logprob": -1.0}] for _ in range(steps)]}}


def _svc(n, logprob=-1.01, token="a"):
    return [
        {
            "prompt": "p",
            "steps": [
                {"step": i, "tokens": [token], "top_logprobs": [{token: logprob}]}
                for i in range(n)
            ],
        }
    ]


def test_teacher_gate_accepts_small_delta_and_same_top1():
    result = check(_hf(2), _svc(2), top5_min=1)
    assert result["ok"] is True
    assert result["criteria"]["atol"] == 2e-2


def test_teacher_gate_rejects_logprob_delta():
    result = check(_hf(1), _svc(1, logprob=-1.03), top5_min=1)
    assert result["ok"] is False
    assert result["prompts"][0]["failures"][0]["logprob_delta"] == pytest.approx(0.03)


def test_teacher_gate_rejects_top1_mismatch():
    result = check(_hf(1), _svc(1, token="b"), top5_min=1)
    assert result["ok"] is False
    assert result["prompts"][0]["top1_mismatch"] == 1
