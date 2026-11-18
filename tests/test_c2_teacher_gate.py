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
    result = check(_hf(2), _svc(2))
    assert result["ok"] is True
    assert result["criteria"]["atol_diagnostic"] == 2e-2


def test_teacher_gate_reports_logprob_delta_as_diagnostic():
    result = check(_hf(1), _svc(1, logprob=-1.03))
    assert result["ok"] is True
    assert result["prompts"][0]["failures"] == []
    assert result["prompts"][0]["max_logprob_delta"] == pytest.approx(0.03)


def test_teacher_gate_rejects_top1_mismatch():
    result = check(_hf(1), _svc(1, token="b"))
    assert result["ok"] is False
    assert result["prompts"][0]["top1_mismatch"] == 1


def test_teacher_gate_matches_token_ids_for_byte_tokens():
    hf = {"p": {"steps": [[{"id": 42, "tok": "\ufffd", "logit": 5.0, "logprob": -1.0}]]}}
    svc = [
        {
            "prompt": "p",
            "steps": [
                {
                    "step": 0,
                    "token_ids": [42],
                    "tokens": [""],
                    "token_logprobs": [-1.0],
                    "top_logprobs": [{"": -1.0}],
                }
            ],
        }
    ]
    result = check(hf, svc)
    assert result["ok"] is True
    assert result["prompts"][0]["same_top1"] == 1


def test_teacher_gate_accepts_near_tie():
    hf = {
        "p": {
            "steps": [
                [
                    {"id": 1, "tok": "a", "logit": 5.0, "logprob": -1.0},
                    {"id": 2, "tok": "b", "logit": 4.9, "logprob": -1.1},
                ]
            ]
        }
    }
    svc = [
        {
            "prompt": "p",
            "steps": [
                {"step": 0, "token_ids": [2], "tokens": ["b"], "top_logprobs": [{"a": -1.0, "b": -1.1}]}
            ],
        }
    ]
    result = check(hf, svc)
    assert result["ok"] is True
    assert result["prompts"][0]["tied_top1"] == 1
