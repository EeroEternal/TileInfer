# vLLM 0.23 source excerpts (citation anchors)

This directory holds **verbatim copies** of the vLLM 0.23.0 modules that this repository cites by
`file:line` — nothing here is compiled, imported, or shipped in the wheel.

## Why it exists

The native backend is written against vLLM 0.23's *official extension points*, not against
`vllm_ascend` internals. Every such claim in this repo ends with a citation like
`reference/vllm-0.23/vllm__v1__attention__selector.py:21-34`, and a citation is only worth
anything if the reader can open the file and check the line.

Earlier these excerpts lived in `reference/vllm-0.23/`, which `.gitignore` excluded — so every
citation in `tileinfer/vllm/platform.py` and friends pointed at a path that did not exist in a
fresh checkout. Moving them here makes the citations resolvable again.

## Naming

The path is flattened with `__` because these are single files lifted out of a package tree:

| here | upstream |
|---|---|
| `vllm__v1__attention__selector.py` | `vllm/v1/attention/selector.py` |
| `vllm__model_executor__layers__mamba__gdn__qwen_gdn_linear_attn.py` | `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` |
| `fla_fused_sigmoid_gating.py` | `vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py` (short alias, kept for historical `upstream_ref=` values) |

## Contents

Only the modules that are actually cited, 18 files, ~720 KB:

- **platform / worker contract** — `platforms__interface`, `config__parallel`,
  `v1__worker__worker_base`, `v1__worker__gpu_worker`
- **attention** — `v1__attention__selector`, `v1__attention__backend`,
  `v1__attention__backends__registry`
- **model runner** — `v1__worker__gpu_model_runner`
- **graph capture** — `compilation__cuda_graph`
- **GDN / FLA** — `mamba__gdn__qwen_gdn_linear_attn`, `mamba__ops__causal_conv1d`,
  `fla__ops__fused_gdn_prefill_post_conv`, `fla__ops__chunk`, `fla_fused_recurrent`,
  `fla_fused_sigmoid_gating`, `fla_solve_tril`, `fla_cumsum`, `fla_l2norm`

## Provenance and licence

Upstream: <https://github.com/vllm-project/vllm>, tag `v0.23.0`. vLLM is Apache-2.0; the same
licence covers this repository (see `../../LICENSE`), and the files are unmodified.

## Rules

1. **Do not edit these files.** They are line-number anchors; an edit silently invalidates every
   citation that points into them.
2. When upgrading vLLM, re-copy the cited modules from the new tag and re-check the line numbers
   the code cites. The upgrade checklist is in `../../docs/notes/vllm-api-0.23.md`.
3. If a citation cannot be re-verified, delete the claim rather than leaving a confident-sounding
   line that no longer matches the source.
