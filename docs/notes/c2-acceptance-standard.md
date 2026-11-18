# C2 acceptance standard: teacher-forced top-1/top-5 + declared logprob tolerance

Status: **standard updated; current single-card evidence still fails C2** (see §4).

## 1. What changed and why

The previous project text said "逐位比较对数（4 位小数口径）". That is a
same-stack determinism check: it is meaningful when two arms run the same
kernels/dtype on the same device. It is not a cross-backend reference check.
Tiny bf16 differences accumulate through 52 layers; a 1e-4 absolute bound then
reports a failure even when greedy behaviour is identical.

The common industry practice for API logprobs is not a universal 1e-4 bound.
OpenAI-compatible APIs document that logprobs are model-dependent and
approximate; cross-implementation checks normally compare **rank/top-k
agreement** and use a declared tolerance for the numerical delta. The project
already has such a convention: `benchmarks/two_arm.sh` calls
`ab_report.py --atol 2e-2` and `benchmarks/capture_logprobs.py` requires the
same token sequence before comparing logprobs.

C2 therefore now means:

1. **Teacher forcing.** Force the reference prefix token ids on the served
   model at every step, then compare the next-token distribution. Comparing
   two sampled greedy trajectories after their first divergence measures two
   different prefixes, not numerical parity.
2. **top-1 must match** the reference top-1. A reference top-1 tie is counted
   separately and accepted only when the served token is within the recorded
   0.125-logit tie window; it is not a free pass for an arbitrary token.
3. **top-5 overlap ≥ 4/5** at every step.
4. **Selected-token logprob absolute delta ≤ 2e-2** by default. The tolerance
   is declared next to the evidence; it comes from the existing
   `two_arm.sh` cross-arm convention, not from a new number invented for this
   model.
5. Re-running the same request must produce the same tokens/logprobs.

`1e-4` remains useful only as a **same-stack diagnostic**. It is no longer the
C2 pass/fail boundary.

Implementation: `tools/nemotron/check_c2_teacher.py`
(tests: `tests/test_c2_teacher_gate.py`). The old first-token helper
`tools/nemotron/check_c2_logprobs.py` is kept for prompt-level debugging, not
as the C2 verdict.

## 2. Reproduction

```bash
python3 tools/nemotron/reference_probe.py ...          # HF all-step top-k (CPU)
python3 - <<'PY'                                        # force HF prefixes on the service
# POST /v1/completions with prompt=<list of token ids>
PY
python3 tools/nemotron/check_c2_teacher.py \
  --hf-steps /tmp/m5-hf-c2-all-steps.json \
  --service /tmp/m5-service-c2-teacher.json \
  --out /tmp/m5-c2-teacher-gate.json
```

The teacher capture sends the reference prompt ids plus the reference
generated ids before each step, `max_tokens=1`, `temperature=0`, `top_p=1`,
`top_k=0`, `logprobs=5`. It does not sample a new trajectory.

## 3. Current single-card scope

TP=1 on Ascend950PR is the delivery scope. TP=8 is out of scope for this
acceptance and is not a blocker; the host HCCL rootinfo issue is recorded
separately in `m5-nemotron-blocker.md`.

## 4. Current evidence (fails)

Input: four prompts, 32 teacher-forced steps each, HF CPU bf16 reference.
Gate: `atol=2e-2`, `top5_min=4`, exact/tie top-1.

| prompt | top-1 same | top-1 tie | top-1 mismatch | top-5 ≥4 | max abs logprob delta |
|---|---:|---:|---:|---:|---:|
| The capital of France is | 31/32 | 0 | 1 | 23/32 | 4.7552 |
| def fibonacci(n): | 32/32 | 0 | 0 | 32/32 | 0.3662 |
| Explain in one sentence why the sky is blue: | 31/32 | 1 | 0 | 30/32 | 0.2453 |
| 北京是中国的 | 26/32 | 2 | 4 | 27/32 | 0.3136 |

Verdict: **C2 not accepted**. The failure is not a 1e-4 technicality:

- `The capital of France is` step 31: reference top-1 is `n` (HF logit 21.375
  in fp32, 19.625 in bf16), served top-1 is `The` with logprob -0.012. This is
  a real greedy flip, not a near-tie.
- The code prompt is top-1/top-5 stable for all 32 steps, but the first-token
  `\` logprob differs by 0.3662; that alone exceeds the declared 2e-2.
- Chinese and sky failures are mostly near-ties at the top-1, but the top-5
  overlap and logprob deltas still exceed the declared bound.

## 5. Fix attempts already ruled out

All on TP=1, same checkpoint, `--generation-config vllm`, request
`temperature=0`, `top_p=1`, `top_k=0`:

| change | result |
|---|---|
| `TILEINFER_MOE_FP32=1` | first-token deltas change, step-31 `The` flip remains |
| `--mamba-cache-mode none` | step-31 `The` flip remains |
| no model `generation_config` defaults | logprobs unchanged |
| Mamba2 wide work in bf16 (experiment) | step-31 `The` flip remains; reverted |

The next step is layer-level isolation: dump the last-token output of every
Mamba2 layer on both HF and the served path, find the first layer whose
last-token vector departs from the HF reference, then fix that operator/state
path. Until that is done, `servable=false` and MTP stays default-off.
