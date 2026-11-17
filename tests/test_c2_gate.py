import importlib.util
from pathlib import Path


def _load_tool():
    path = Path(__file__).resolve().parents[1] / "tools" / "nemotron" / "check_c2_logprobs.py"
    spec = importlib.util.spec_from_file_location("check_c2_logprobs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hf(ids):
    return {
        "results": [
            {
                "prompt": "p",
                "generated_token_ids": ids,
                "first_token_top5": [
                    {"token": "A", "logit": 10.0, "logprob": -0.1},
                    {"token": "B", "logit": 9.875, "logprob": -0.225},
                    {"token": "C", "logit": 8.0, "logprob": -2.1},
                ],
            }
        ]
    }


def test_tie_aware_gate_accepts_documented_tie():
    tool = _load_tool()
    service = [
        {
            "prompt": "p",
            "ids": [1, 3],
            "top_logprobs": [{"A": -0.1, "B": -0.225}],
        }
    ]
    steps = {
        "p": {
            "ids": [1, 2],
            "steps": [
                [
                    {"id": 1, "tok": "A", "logit": 10.0, "logprob": -0.1},
                    {"id": 2, "tok": "B", "logit": 9.875, "logprob": -0.225},
                ],
                [
                    {"id": 2, "tok": "B", "logit": 9.5, "logprob": -0.5},
                    {"id": 3, "tok": "C", "logit": 9.5, "logprob": -0.5},
                ],
            ],
        }
    }
    result = tool.check(
        _hf([1, 2]),
        service,
        steps,
        logprob_tol=0.5,
        top5_min=1,
        tie_logit_tol=0.125,
        require_sequence=True,
    )
    assert result["ok"] is True
    assert result["rows"][0]["strict_ids"] is False
    assert result["rows"][0]["sequence_ok"] is True


def test_tie_aware_gate_rejects_unexplained_divergence():
    tool = _load_tool()
    service = [
        {
            "prompt": "p",
            "ids": [1, 3],
            "top_logprobs": [{"A": -0.1, "B": -0.225}],
        }
    ]
    result = tool.check(
        _hf([1, 2]),
        service,
        None,
        logprob_tol=0.5,
        top5_min=1,
        tie_logit_tol=0.125,
        require_sequence=True,
    )
    assert result["ok"] is False
    assert "no per-step tie evidence" in result["rows"][0]["errors"][0]


def test_gate_requires_logprob_and_top5_agreement():
    tool = _load_tool()
    service = [
        {
            "prompt": "p",
            "ids": [1, 2],
            "top_logprobs": [{"A": -3.0, "Z": -1.0}],
        }
    ]
    result = tool.check(
        _hf([1, 2]),
        service,
        None,
        logprob_tol=0.5,
        top5_min=2,
        tie_logit_tol=0.125,
    )
    assert result["ok"] is False
    row = result["rows"][0]
    assert row["top5_overlap"] == 1
    assert any("top-1" in e for e in row["errors"])
    assert any("logprob delta" in e for e in row["errors"])
