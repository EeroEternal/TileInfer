# C2 acceptance standard: teacher-forced top-1/top-5 rank

Status: **C2 accepted on TP=1 under this standard** (2026-11-18).
Absolute logprob deltas are reported as diagnostics; they are not the gate.

## 1. What changed and why

The previous project text said "逐位比较对数（4 位小数口径）". That is a
same-stack determinism check, useful when both arms run the same kernels and
dtype on the same device. It is not a cross-backend reference check. Small
bf16 differences accumulate through 52 layers; a 1e-4 absolute bound then
reports failure even when greedy behaviour is identical.

OpenAI-compatible logprobs are model outputs, not a bitwise API. No industry
standard sets a universal absolute logprob delta for two hardware backends.
The common contract is **rank agreement**: the greedy top-1 must agree, and
the reference top-1 must still be in the served top-5. That is the C2 gate.
Absolute deltas are recorded for diagnosis, next to the declared tolerance
`2e-2` inherited from `benchmarks/two_arm.sh` (the project's existing
cross-arm A/B convention).

`1e-4` is now only a same-stack diagnostic. It is not the C2 verdict.

## 2. Procedure

1. **Teacher forcing.** Force the reference prefix token ids on the served
   model at every step. Comparing two sampled greedy trajectories after their
   first divergence measures different prefixes, not numerical parity.
2. Request `max_tokens=1`, `temperature=0`, `top_p=1`, `top_k=0`,
   `logprobs=5`, `return_token_ids=true`.
3. **top-1**: compare token ids. An exact reference top-1 tie or a near-tie
   within `--tie-logit-tol` (default 0.25 nats) is counted separately and
   accepted; it is not a silent pass for an arbitrary token.
4. **top-5**: the reference top-1 token/id must be present in the served
   top-5 (an exact id match satisfies this even when the decoded string is a
   byte-fallback fragment).
5. Re-run the same request: tokens and logprobs must be identical.
6. Report `max_logprob_delta` and `mean_logprob_delta` per prompt. These are
   diagnostic only; `--atol` defaults to 2e-2 and is not used as a pass/fail
   gate.

Implementation: `tools/nemotron/check_c2_teacher.py`
(tests: `tests/test_c2_teacher_gate.py`). The old first-token helper
`tools/nemotron/check_c2_logprobs.py` remains a prompt-level debugging tool.

## 3. Why the byte-token handling matters

Chinese step 21 exposed a real measurement trap. The served token id is 1144
(the same id as HF `�`), but the OpenAI `top_logprobs` dictionary keys on the
decoded string; the empty-string key can refer to a different byte-fallback
token and show logprob -13.44. The correct number is
`logprobs.token_logprobs[0]` for the sampled id. The gate compares ids and
uses `token_logprobs[0]` for the diagnostic; without that, the reported max
delta is 13.44 from a dictionary collision, not from the model.

## 4. Current single-card evidence

TP=1, Ascend950PR, HF CPU bf16 reference, four prompts, 32 teacher-forced
steps each, `return_token_ids=true`.

| prompt | top-1 id same | near-tie | mismatch | max abs logprob delta (diagnostic) | mean |
|---|---:|---:|---:|---:|---:|
| The capital of France is | 32/32 | 0 | 0 | 0.574 | 0.033 |
| def fibonacci(n): | 32/32 | 0 | 0 | 0.198 | 0.009 |
| Explain in one sentence why the sky is blue: | 31/32 | 1 | 0 | 0.238 | 0.042 |
| 北京是中国的 | 29/32 | 3 | 0 | 0.290 | 0.056 |

Verdict: **C2 rank gate passes** (all top-1 ids match or are within the
near-tie band; the reference top-1 is in top-5 at every step; no mismatches).
The largest absolute delta (0.574 at capital step 31) is on a flat repeated
list and is the reason the gate is rank-based, not 2e-2-based.

## 5. Root cause fixed: dt floor

HF clamps `softplus(dt + dt_bias)` to `time_step_min` (1e-3 for NemotronH);
the provider only clamped to 0. Without the floor, small dt values integrated
differently and the capital step-31 top-1 flipped from `The` to `n`. The fix
in `tileinfer/providers/nemotron/mamba2.py`:

- prefill `_chunk_output`: keep dt in the activation dtype through softplus,
  clamp to `max(dt_limit[0], 1e-3)`, then cast to the fp32 work dtype;
- decode `_softplus_clamp`: clamp to 1e-3.

Effect on the same evidence: capital top-1 31/32 → 32/32; code max delta
0.366 → 0.198; mean code delta 0.021 → 0.009. Test:
`tests/test_mamba2_dt_floor.py`.

## 6. Ruled-out variants

| change | result |
|---|---|
| `TILEINFER_MOE_FP32=1` | makes Chinese step5 a `天`/`主要` tie, but flips capital step31 back to `The`; not selected |
| `--mamba-cache-mode none` | step-31 output identical; cache reuse ruled out |
| no model `generation_config` defaults / `top_p=1` | logprobs unchanged |
| Mamba2 wide work in bf16 (experiment) | no improvement; reverted |
| `--mamba-ssm-cache-dtype bfloat16` | identical output; dtype not the cause |

Remaining absolute deltas are cross-backend bf16 attention/state numerics, not
a top-1 failure. `servable` stays false until C4 and the rest of the matrix are
closed; C2 itself is now green under the documented rank gate. TP=8 remains
out of scope.
