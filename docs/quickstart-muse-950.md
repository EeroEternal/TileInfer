# Quickstart: MuseGlimmer-30B on Ascend 950PR (V3 bridge)

This is the self-contained recipe for the machine class we validated on 2026-11-17
(8 × Ascend950PR, 86 GB HBM each). It assumes the repository's `.venv` is the vendor stack
used for the Muse run; adjust paths if yours differs.

## 1. Prerequisites

| Item | Required value | Why |
|---|---|---|
| CANN | 9.3.0 (950 weekly) | the 9.1.0 system default is not the verified stack |
| NNAL / ATB | any build with `libatb.so` (validated: 9.2.0-beta.1) | `torch_npu._npu_rotary_embedding` loads it on the decode path |
| torch / torch_npu | 2.10.0+cpu / 2.10.0.post4 | the vllm-ascend 0.23.0-950 pairing |
| vLLM / vllm-ascend | 0.23.0 / 0.23.0 (950 wheel) | the bridge pins 0.23 |
| transformers / hub | 5.5.4 / ≥1.5,<2.0 (validated 1.33.0) | checkpoint tokenizer_class is `TokenizersBackend` |
| triton-ascend | 3.2.2 | block-table / decode kernels |
| glibcxx | GLIBCXX_3.4.30 available on `LD_LIBRARY_PATH` | CANN opmaster |
| weights | `/home/models/Muse-Glimmer-30B` | BF16, ~56 GiB, 2 shards |

TileInfer itself: `pip install -e .` (or `PYTHONPATH=<repo>`) in that venv; the only entry
point is `vllm.general_plugins -> tileinfer.bridge.install:install`.

## 2. Environment

```bash
source /path/to/cann-9.3.0/set_env.sh
source /path/to/nnal/atb/set_env.sh          # provides libatb.so
export ACL_OP_INIT_MODE=1
export LD_LIBRARY_PATH=/path/to/glibcxx:$LD_LIBRARY_PATH
export PYTHONPATH=/path/to/cann-9.3.0/python/site-packages:/path/to/cann-9.3.0/opp/built-in/op_impl/ai_core/tbe
export TILELANG_CACHE_DIR=$HOME/.cache/tilelang-ascend950   # writable local disk
export TILEINFER_MANIFEST=muse_glimmer
export ASCEND_RT_VISIBLE_DEVICES=<free card id>             # do not assume card 0
source <repo>/.venv/bin/activate
```

`npu-smi info` first: pick a card whose `HBM-Usage` is at the baseline (~4.5 GB), not one
held by another tenant.

## 3. Verify the weights before serving

A corrupted shard (zeroed header) fails late with `invalid JSON in header`. Check first:

```bash
python scripts/verify-safetensors.py /home/models/Muse-Glimmer-30B
```

It validates every `*.safetensors` header and the index; non-zero exit means re-upload /
re-download that shard.

## 4. Serve

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /home/models/Muse-Glimmer-30B --served-model-name muse \
  --trust-remote-code \
  --max-model-len 32768 --max-num-seqs 8 \
  --gpu-memory-utilization 0.85 \
  --reasoning-parser atem \
  --port 8360
```

Notes:
- Omit `--enforce-eager` to enable graph capture. Graph does **not** change TPOT but lowers
  TTFT (1K: 424→328 ms; 8K: 1998→1260 ms on the validation box). Use `--enforce-eager` only
  when memory is tight or when debugging.
- The ATEM parser is required for this checkpoint: content arrives in the `reasoning` channel
  until the model switches to the user channel.

## 5. Smoke

```bash
curl -s localhost:8360/v1/models
curl -s localhost:8360/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"muse","messages":[{"role":"user","content":"1+1=?"}],
  "max_tokens":64,"temperature":0}'
```

Expected: HTTP 200, non-empty `reasoning`/`content`.

C2 (numeric parity) uses the frozen first-token top-5 in the manifest:

```bash
python tools/muse/c2_oracle.py --out /tmp/muse-c2.json   # float32 CPU oracle (optional)
python tools/muse/check_c2.py --base-url http://127.0.0.1:8360
```

The checker exits 0 when the token set matches and the logprobs are inside the declared bf16
tolerance. See `docs/notes/muse-glimmer-v3-validation-20261117.md` §4.1.

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Free memory ... less than desired` | card already used by another tenant | pick a free `ASCEND_RT_VISIBLE_DEVICES` or lower `--gpu-memory-utilization` |
| `Tokenizer class TokenizersBackend does not exist` | transformers too old | install 5.5.4 (and hub ≥1.5) |
| `Could not load ... libop_plugin_atb.so` | NNAL/ATB not sourced | source the ATB `set_env.sh` |
| `invalid JSON in header` | truncated/zeroed safetensors shard | run §3, re-upload that shard |
| `'function' object is not subscriptable` in block_table | triton missing | install triton-ascend 3.2.2 |
| `ERR99999` on `import torch_npu` | CANN env not sourced (or wrong CANN) | source 9.3.0, not the 9.1.0 default |
