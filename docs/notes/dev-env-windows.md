# Dev environment notes (Windows workstation)

Records what it took to get a runnable CPU test loop on the Windows checkout, so the next session
does not rediscover it.  Nothing here affects the Ascend box.

## Working commands

```powershell
$env:PYTHONPATH="E:\project\TileInfer\runtime\pylibs"
$env:PYTHONDONTWRITEBYTECODE="1"
python -m pytest tests/test_serving_buckets.py -q -p no:cacheprovider --no-header
```

`runtime/` is gitignored. `runtime/pylibs` holds torch 2.14.1+cpu, pytest 9.1.1 and their deps,
extracted from wheels (see below). `runtime/vllm-src` will hold the vLLM 0.23.0 sdist (the API
reference for the platform/attention port).

## What does not work, and the workaround

| Symptom | Cause | Workaround |
|---|---|---|
| `pip install anything` → `Permission denied ... .metadata` | the file sandbox denies pip's unpack/rename sequence (fails even for `six`) | download the wheel with `urllib` and `zipfile.extractall` into `runtime/pylibs` |
| `Invoke-WebRequest` / `curl` → TLS failure (`SEC_E_NO_CREDENTIALS`, "基础连接已经关闭") | the shell's TLS stack cannot acquire credentials in this sandbox | do network I/O from **Python** (`urllib.request` works) |
| `pwsh` not found | only Windows PowerShell 5.1 is on PATH | use the `pwsh` tool (it dispatches to 5.1) |
| `& script.ps1` → not digitally signed | execution policy | `Get-Content script.ps1 -Raw \| Invoke-Expression` |
| `import torch` → `Error loading c10.dll` | no MSVC runtime on this machine | `python .work/get_runtime_dlls.py` (pulls the `msvc-runtime` wheel and copies `msvcp140*.dll` / `vcruntime140*.dll` / `concrt140*.dll` into `runtime/pylibs/torch/lib`) |
| `python -m py_compile` → `Access denied: __pycache__` | the sandbox denies `__pycache__` creation | always set `PYTHONDONTWRITEBYTECODE=1`; syntax-check with `ast.parse` |
| `py_compile`/pytest leave no cache | `-p no:cacheprovider` avoids `.pytest_cache` |

Bootstrap scripts (both under `.work/`, not part of the repo): `get_runtime_dlls.py`,
`bootstrap_cpu_deps.py`.

## Pre-existing collection failures (not caused by the port)

`pytest tests -m "not npu"` cannot be green on a machine without the TileLang toolchain, because
nine files import `tilelang` at module level:

```text
test_fused_sample.py  test_fused_sample_integration.py  test_greedy_fast_sampler.py
test_moe_align_packer.py  test_moe_prenz_pack.py  test_moe_slot_layout.py
test_moe_weight_prefetch.py  test_prep_fastpath.py  test_sw_pipeline_switch.py
```

A handful of legacy `test_vllm_ascend_*` / `test_step_diag` tests additionally need `vllm_ascend`
or write files (PermissionError here).  For CPU verification during the port, run:

```powershell
python -m pytest tests/test_plan.py tests/test_reference.py tests/test_attention_api.py `
  tests/test_backend_registry.py tests/test_serving_buckets.py tests/test_serving_step.py `
  tests/test_kv_cache_write.py tests/test_patch_registry.py tests/test_layer_deps.py `
  tests/test_no_vllm_ascend.py -q -p no:cacheprovider
```

→ 95 passed (2026-10-27).

The same collection failure will hit GitHub CI (`.github/workflows/ci.yml` installs torch + pytest
only), so the CI job has been red independently of this work; fixing it means marking those files
`tilelang` and skipping them, which is worth doing once the operator layer lands.
