# legacy/v2 — frozen v2 snapshot (archived 2026-11-16)

This directory is the **v2 TileInfer backend**, moved out of the installed package so that the
v3 tree (`tileinfer/{contracts,resolve,manifests,providers,bridge,cli}`) is the only live path.

It is a **source archive**, not a runnable package:

- it is not installed (`pyproject.toml` packages only `tileinfer*`);
- it is not collected by `pytest tests`;
- `import tileinfer.vllm` fails — do not add `legacy/v2` to `sys.path` to “make it work”, because
  the installed `tileinfer` package shadows the partial `tileinfer/` directory here.

## Contents

| Path | What it is |
|---|---|
| `tileinfer/vllm/` | v2 platform, worker, attention backend, op families, profiles, plugin |
| `tests/` | v2-only tests (`test_v3_*` stayed in the main `tests/`) |
| `scripts/` | `serve_tileinfer.py`, `serve_qwen36.py`, `install-native-backend.sh`, examples |
| `benchmarks/` | the 17 benchmarks that import `tileinfer.vllm` |
| `tools/dsv41/` | the two v2-dependent scripts (`perf_spec_parity.py`, `s1_vllm_registry.py`); the rest of `tools/dsv41` stayed in the main tree because the v3 DSV41 parity tests import it |

## Why it is kept

v2 is the A/B reference: it runs, and it answers “is the vendor really unable to do this shape,
or is it just slow?”. The v3 decision rules are in
[`docs/development-rules.md`](../../docs/development-rules.md); the move is recorded in
[`docs/notes/m1-delete-list.md`](../../docs/notes/m1-delete-list.md).

## How to read or run it

- Read: open `legacy/v2/tileinfer/vllm/...` directly.
- Run: check out the commit before the move in a separate worktree; the archive is not wired for
  import in this tree.
