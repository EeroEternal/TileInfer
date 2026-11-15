#!/usr/bin/env python3
"""CPU reference probe for the Nemotron M5 bring-up.

This is the no-vendor-baseline oracle: load the checkpoint with HF transformers on CPU,
run one greedy prompt, and dump text/token ids/first-token top-5.  It never downloads and
never touches NPU; if the checkpoint is absent it exits with a clear error.

Transformers 4.57 does not ship NemotronH.  Newer transformers (5.x) does; if a second
site-packages is available (``TILEINFER_HF_SITE`` or the DSH SGLang venv below), put it
ahead of the installed transformers and disable torchvision, which is not needed for a
text-only probe and may be ABI-incompatible with the CPU torch.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import os
import sys
from pathlib import Path

_DEFAULT_HF_SITE = "/home/dlb/sglang/.venv_npu/lib/python3.12/site-packages"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="local HF checkpoint directory")
    ap.add_argument("--prompt", default="Say hello in one sentence.")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--out", default="", help="JSON output path (default: stdout)")
    ap.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16"))
    ap.add_argument("--top-k", type=int, default=5)
    return ap


def _prepare_imports() -> None:
    """Import CPU torch first, then a transformers that knows ``nemotron_h``."""
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch  # noqa: F401  (must be the CPU build from this venv)

    site = os.environ.get("TILEINFER_HF_SITE", _DEFAULT_HF_SITE)
    if site and Path(site).is_dir() and site not in sys.path:
        sys.path.insert(0, site)
    # Text-only: make the vision import fail fast, rather than loading an ABI-mismatched
    # torchvision from the secondary site-packages.
    sys.modules.setdefault("torchvision", None)


def _load_model(model_dir: str, dtype: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = torch.float32 if dtype == "float32" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    kwargs = {
        "local_files_only": True,
        "attn_implementation": "eager",
    }
    if importlib.util.find_spec("accelerate") is not None:
        kwargs["low_cpu_mem_usage"] = True
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch_dtype, **kwargs)
    except TypeError:
        # transformers 4.x spelling
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch_dtype, **kwargs)
    model.eval()
    return tok, model


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_dir = Path(args.model)
    if not model_dir.is_dir():
        print(f"error: model directory not found: {model_dir}", file=sys.stderr)
        return 2

    try:
        _prepare_imports()
        import torch
        import transformers
        from transformers import AutoModelForCausalLM  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment-specific
        print(f"error: transformers/torch unavailable: {exc}", file=sys.stderr)
        return 2

    try:
        tok, model = _load_model(str(model_dir), args.dtype)
    except Exception as exc:  # pragma: no cover - environment-specific
        print(f"error: cannot load reference model: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    inputs = tok(args.prompt, return_tensors="pt")
    prompt_ids = inputs["input_ids"][0].tolist()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tok.eos_token_id,
        )

    seq = out.sequences[0]
    generated_ids = seq[len(prompt_ids) :].tolist()
    text = tok.decode(generated_ids, skip_special_tokens=False)
    topk = []
    if out.scores:
        first = out.scores[0][0].float()
        logprobs = torch.log_softmax(first, dim=-1)
        vals, idx = torch.topk(first, k=min(args.top_k, first.numel()))
        topk = [
            {
                "token_id": int(i),
                "token": tok.decode([int(i)]),
                "logit": float(v),
                "logprob": float(logprobs[int(i)]),
            }
            for v, i in zip(vals.tolist(), idx.tolist())
        ]

    payload = {
        "reference_id": "hf-cpu-nemotron-h-" + _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "model": str(model_dir),
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "dtype": args.dtype,
        "device": "cpu",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "transformers_site": next((p for p in sys.path if p.endswith("site-packages") and "sglang" in p), ""),
        "prompt_token_ids": prompt_ids,
        "generated_token_ids": generated_ids,
        "text": text,
        "first_token_top5": topk,
        "note": "CPU reference; no NPU and no vllm-ascend baseline",
    }
    text_out = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text_out, encoding="utf-8")
    else:
        print(text_out, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
