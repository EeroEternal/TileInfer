# Development rules (v3)

This file is normative for new work. It records the architecture decision made after the v2
operator tree was frozen; `docs/extending.md` remains the how-to guide, but where the two
disagree this file wins.

## 1. v2 is archived

`tileinfer/vllm/` was **v2 legacy**. On 2026-11-16 it was moved out of the package to
[`legacy/v2/tileinfer/vllm/`](../legacy/v2/tileinfer/vllm/). It is a source archive: it is not
installed, not collected by pytest, and **not importable as `tileinfer.vllm`**. v3 code never
imports it.

- Do **not** add features, fix behavior, or write new tests in `legacy/v2/`. It is history.
- Do **not** copy a v2 module and edit it in place. New code goes to the v3 layers.
- The installed package must not contain a `tileinfer/vllm` directory; `tileinfer.vllm` is not a
  valid import path any more.
- A pull request that adds business logic under `legacy/v2/` is rejected. Reviewers should ask
  for the v3 location instead.

The v2 → v3 implementation migration is recorded in
[`notes/m5-nemotron-v3-migration.md`](notes/m5-nemotron-v3-migration.md); the move itself is
recorded in [`notes/m1-delete-list.md`](notes/m1-delete-list.md).

## 2. Where new code goes

| Kind of code | Location | May import |
|---|---|---|
| cap template / implementation | `tileinfer/providers/` (L4) | torch, `tileinfer.contracts` |
| vLLM / vllm-ascend names and installation | `tileinfer/bridge/` (L5) | vLLM, vllm-ascend, providers |
| model facts | `tileinfer/manifests/` | contracts |
| capability decision | `tileinfer/resolve/` (L2) | contracts, manifests, providers |
| declarations and schemas | `tileinfer/contracts/` (L3) | stdlib only |

A provider declares coverage, graphability, parallelism, evidence, and `delete_when` before it
is useful. A provider that only re-implements something the vendor already serves is not a gap
and should not be added (`docs/extending.md` §3.0).

## 3. Installation is plan-driven

`tileinfer/bridge/install.py` is the only entry point. It calls the seams in
`tileinfer/bridge/seams.py`; each seam runs only when the plan names its access point.

For operator providers:

- the plan must contain a selection whose `source != "vendor"`;
- the plan must contain the `impl_table` access point;
- the provider name must have an explicit binder in `tileinfer/bridge/impl_table.py`;
- `Plan.servable=false` refuses the install, unless an explicitly labelled bring-up run sets
  `TILEINFER_ALLOW_UNSERVABLE=1`.  Such a run builds its plan with `--allow-candidates`; that
  selects the declared candidates (including a model-type candidate that overrides a generic
  vendor capability) but still writes `servable=false`.

There is no global `install_ops()` call in the v3 bridge and no "install everything then let the
model choose" fallback. A missing access point, an unknown provider, or a failed binder is
reported as failure and leaves the vendor path untouched.

## 4. Model-specific facts are not defaults

A new architecture needs a `ModelManifest` with its layer pattern, geometry, and special
behavior. Do not run it through another model's manifest and patch one family; that is how the
v2 path silently installed Mamba2 for an unrelated target. (`ModelProfile` is v2-only and lives
in the archive.)

## 5. Evidence

No provider is promoted to a default on synthetic or dummy evidence. The promotion bar is
target real-weight C1/C2, and graph capture must be measured separately (C3). A plan may be
built with `servable=false` during bring-up; that is a decision artifact, not a support claim.
