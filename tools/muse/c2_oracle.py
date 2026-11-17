"""Pure-torch float32 CPU oracle for the MuseGlimmer C2 sample.

This is an independent implementation of the text decoder's math (no vLLM, no
vllm-ascend). It reads the checkpoint with safetensors mmap one tensor at a time, runs a
float32 prefill for the prompt, and prints the next-token top-k. The NPU server is expected
to match it within the tolerance declared in the manifest's ``benchmark`` block.

    python tools/muse/c2_oracle.py --out /tmp/muse-c2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer


def rms(x, eps, weight=None, gemma=False):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        w = weight.float()
        if gemma:
            w = w + 1.0
        y = y * w
    return y


def rope(x, positions, theta):
    dim = x.shape[-1]
    half = dim // 2
    inv = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) * 2.0 / dim))
    freqs = positions.float()[:, None] * inv[None, :]
    cos = freqs.cos()[:, None, :]
    sin = freqs.sin()[:, None, :]
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class MuseRef:
    def __init__(self, model_dir: str):
        root = Path(model_dir)
        cfg = json.loads((root / "config.json").read_text())
        self.cfg = cfg.get("text_config", cfg)
        index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        self.shard_of = index
        self.handles = {
            shard: safe_open(str(root / shard), framework="pt")
            for shard in sorted(set(index.values()))
        }

    def raw(self, name: str) -> torch.Tensor:
        return self.handles[self.shard_of[name]].get_tensor(name)

    def weight(self, name: str) -> torch.Tensor:
        return self.raw(name).float()

    def forward(self, ids: list[int], topk: int = 5) -> dict:
        c = self.cfg
        eps = float(c["rms_norm_eps"])
        post_eps = float(c["post_norm_eps"])
        heads = int(c["num_attention_heads"])
        kv_heads = int(c["num_key_value_heads"])
        head_dim = int(c["head_dim"])
        group = heads // kv_heads
        scale = head_dim ** -0.5
        q_scale = float(c["qk_scale_factor"])
        theta = float(c["rope_parameters"]["rope_theta"])
        window = int(c["sliding_window"])
        layer_types = list(c["layer_types"])
        theta_by_layer = list(c["layer_rope_theta"])

        input_ids = torch.tensor(ids, dtype=torch.long)
        h = self.weight("model.language_model.embed_tokens.weight")[input_ids]
        h = rms(h, eps)
        positions = torch.arange(len(ids))
        causal = torch.triu(torch.ones(len(ids), len(ids), dtype=torch.bool), diagonal=1)
        offsets = torch.arange(len(ids))

        prefix = "model.language_model.layers."
        for layer in range(int(c["num_hidden_layers"])):
            p = f"{prefix}{layer}."
            residual = h
            x = rms(h, eps, self.raw(p + "input_layernorm.weight"), gemma=True)

            q = x @ self.weight(p + "self_attn.q_proj.weight").T
            k = x @ self.weight(p + "self_attn.k_proj.weight").T
            v = x @ self.weight(p + "self_attn.v_proj.weight").T
            q = rms(q.view(-1, heads, head_dim), eps) * q_scale
            k = rms(k.view(-1, kv_heads, head_dim), eps)
            v = v.view(-1, kv_heads, head_dim)
            if theta_by_layer[layer]:
                q = rope(q, positions, theta)
                k = rope(k, positions, theta)

            k = k.repeat_interleave(group, dim=1)
            v = v.repeat_interleave(group, dim=1)
            scores = torch.einsum("thd,shd->hts", q, k) * scale
            mask = causal
            if layer_types[layer] == "sliding_attention":
                mask = mask | ((offsets[:, None] - offsets[None, :]) >= window)
            scores = scores.masked_fill(mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out = torch.einsum("hts,shd->thd", probs, v)
            out = out.reshape(len(ids), heads * head_dim)

            gate = x @ self.weight(p + "self_attn.gate_proj.weight").T
            out = torch.sigmoid(gate) * out
            out = out @ self.weight(p + "self_attn.o_proj.weight").T
            out = rms(out, post_eps, self.raw(p + "post_attention_layernorm.weight"), gemma=True)
            h = residual + out

            residual = h
            x = rms(h, eps, self.raw(p + "pre_feedforward_layernorm.weight"), gemma=True)
            up = x @ self.weight(p + "mlp.up_proj.weight").T
            gate = x @ self.weight(p + "mlp.gate_proj.weight").T
            x = torch.nn.functional.silu(gate) * up
            x = x @ self.weight(p + "mlp.down_proj.weight").T
            x = rms(x, post_eps, self.raw(p + "post_feedforward_layernorm.weight"), gemma=True)
            h = residual + x

        h = rms(h, eps, self.raw("model.language_model.norm.weight"))
        logits = h[-1] @ self.weight("lm_head.weight").T
        logits = logits * float(c["output_multiplier"])
        cap = c.get("final_logit_softcapping")
        if cap:
            cap = float(cap)
            logits = cap * torch.tanh(logits / cap)
        logp = torch.log_softmax(logits, dim=-1)
        values, indices = logp.topk(topk)
        return {
            "prompt_ids": ids,
            "next_token_id": int(indices[0]),
            "top5": [[int(i), float(v)] for i, v in zip(indices.tolist(), values.tolist())],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/models/Muse-Glimmer-30B")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--ids", default="")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    ids = (
        [int(x) for x in args.ids.split(",")]
        if args.ids
        else tokenizer.encode(args.prompt)
    )
    result = MuseRef(args.model).forward(ids, args.topk)
    result["prompt"] = args.prompt
    result["prompt_text"] = tokenizer.decode(ids)
    result["top5_tokens"] = [tokenizer.decode([i]) for i, _ in result["top5"]]
    text = json.dumps(result, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
