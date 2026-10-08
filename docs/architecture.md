# Architecture

This document describes the pieces that are actually implemented: the plan/run contract, the
metadata model, the scheduler, and the invariants each kernel has to satisfy.  For the project
positioning, read [`design.md`](design.md) first.

## 1. The plan/run split

```
      host, per batch *shape*                     device, per step
 ┌──────────────────────────────┐         ┌──────────────────────────────┐
 │ plan()                       │         │ run()                        │
 │  · detect mode               │         │  · copy step metadata into   │
 │  · tile the work             │  plan   │    plan-owned buffers        │
 │  · LPT-order the tiles       │ ──────► │  · launch kernel(s)          │
 │  · allocate workspace        │         │  · (merge splits)            │
 │  · JIT-compile the kernel    │         │                              │
 └──────────────────────────────┘         └──────────────────────────────┘
        can be slow, runs once                must be allocation-free,
        per shape, never captured             compilation-free, sync-free
```

Three rules fall out of "the run stage is captured into an ACLGraph":

1. **No allocation in `run`.**  Everything the kernel writes into is owned by the plan
   (`plan.workspace`, `plan.backend_state`).  The `WorkspaceManager` pools buffers per signature,
   so steady-state serving memory is flat.
2. **No compilation in `run`.**  The kernel handle is resolved in `finalize_plan` — one JIT per
   (shape, page-capacity) combination.  `run` does a dict lookup.
3. **No host synchronisation in `run`.**  Shapes are static, so nothing has to be read back from
   the device.  Metadata values (page lists, sequence lengths) are only ever *copied* to the
   device, never read from it.

### Metadata ownership

`plan` stores *references* to the metadata tensors it was built from.  Serving engines update
those buffers in place between steps (vLLM's `block_tables` / `seq_lens` behave this way), so the
plan keeps working.  Engines that produce new tensors every step can instead pass them to `run`:

```python
out = attn.run(q, k_cache, v_cache, plan=plan,
               kv_indptr=indptr, kv_indices=indices, kv_last_page_len=last_len)
```

The TileLang backend copies the step metadata into plan-owned, statically shaped buffers.  Two
things fall out of that copy: the kernel's signature stays static (which is what makes capture
legal), and the `int32 → float32` cast that the vector unit needs for the tail mask happens for
free.

## 2. Metadata model

Two representations, one conversion:

```
engine side (dense)                      kernel side (ragged)
block_tables [B, max_pages]      ──►     kv_indptr     [B+1]   cumulative page counts
seq_lens     [B]                         kv_indices    [P]     physical page ids, logical order
block_size   int                         kv_last_page_len [B]  valid tokens in the last page
                                         page_size     int
```

`PageTable.to_ragged()` drops the padding, so a kernel block never iterates over unused table
entries — the dense form costs a data-dependent branch in the innermost loop, the ragged form
costs one extra tiny tensor.  The reverse conversion (`to_page_table`) exists for tests and for
engines that want to keep their own book-keeping.

`RaggedMetadata.validate()` is called at plan time and rejects the three mistakes that otherwise
surface as silent corruption: non-monotonic `kv_indptr`, indices pointing past the pool,
`kv_last_page_len` inconsistent with `page_size`.

## 3. Scheduling, and why LPT

On Ascend, kernel blocks are dispatched to AI cores roughly in `cid` order.  A serving batch is
badly skewed — one request at 100k tokens next to 32 requests at 200 — so a schedule that gives
one block per request leaves most cores idle while one grinds through the long request.

`plan.default_tiles` therefore:

1. **looks at the work per request** (`kv_tile_pages` sets the target work per tile);
2. **splits long requests along the KV axis** into separate tiles, each of which produces a
   *partial* output that a merge step combines (split-KV);
3. **orders tiles longest-processing-time first** (`lpt_order`), so the heavy tiles start on the
   first cores and the tail of the schedule is made of cheap tiles.

The merge contract is the standard one, and it is implemented twice on purpose — once in the
torch reference (`merge_partials`) and once (next) in a kernel:

```
partial_out[t] = Σ_k exp(s_k − m_t) v_k          (unnormalised numerator of tile t)
partial_lse[t] = m_t + log Σ_k exp(s_k − m_t)
out[b]         = Σ_t exp(lse_t − lse_b) · partial_out[t] / exp(lse_b)
```

`tests/test_reference.py::test_split_kv_decode_matches_single_pass` pins the two paths to each
other, so the scheduler can be developed without a device in the loop.

When a schedule contains no splits (`needs_merge == False`), the plan allocates **no** workspace
and `run` takes the single-pass path.  That is the common decode case today.

## 4. Kernel invariants

Every TileLang kernel in `tileinfer/kernels/` must satisfy the following; they exist because each
one has already been the source of a real bug in this class of kernel:

- **The tail page is masked before `exp`.**  Cache pages are allocated in full, so slots past
  `kv_last_page_len` contain garbage.  Masking after `exp` (or not at all) silently inflates the
  softmax denominator: the output stays in a plausible range and no test that only compares
  magnitudes will notice.  `tests/test_reference.py` poisons the padding region with `1e4` to make
  this failure loud.
- **Logical page order comes from `kv_indices`, never from arithmetic on the request index.**  A
  prefix-caching engine hands out physical pages in arbitrary order and shares them across
  requests; `tests` shuffle pages for the same reason.
- **The causal offset is `kv_len − qo_len` for an append step**, not `0`.  Chunked prefill and
  speculative decode both rely on this; `test_append_uses_correct_causal_offset` covers it.
- **Layout is NHD**, `[pages, block_size, kv_heads, dim]`.  This is the layout vLLM-Ascend and
  MindIE allocate, so kernels can be dropped in without a repack; `HND` exists in the reference
  implementation for completeness only.
- **GQA groups are contiguous**: query head `kv_head * group + i` maps to KV head `kv_head`.  The
  kernel exploits this to load each KV page once for all query heads of a KV head.

## 5. Backend registry

```
AttentionBackend (base)          name, is_available(), supports_mode(),
                                 finalize_plan(plan, k_cache, v_cache), run(...)
  ├── ReferenceBackend   name="reference"   torch, CPU+NPU, every mode, the oracle
  └── TileLangAscendBackend name="tilelang" Ascend, decode only (for now)
```

`AttentionBackend.plan`-level policy (mode detection, tiling, workspace sizing) lives in
`build_plan`, shared by every backend.  A new backend therefore only describes *mechanism*: which
kernel handles a plan, and how to launch it.  `get_backend("auto")` prefers `tilelang` when it is
available and falls back to `reference`, which keeps the same user code running on a laptop and on
the NPU.

## 6. Environment (Ascend)

The plain PyPI `tilelang` wheel has **no Ascend backend**.  The supported combination on an
Ascend 950 / CANN 9.1.x / Python 3.12 / x86_64 machine is:

```bash
# 1. CANN runtime + the NNAL ATB ops (libatb.so lives in nnal)
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 2. tilelang-ascend refuses to share a process with torch_npu without this: both runtimes
#    initialise the ACL op library, and the failure mode is *silently wrong results*, not an
#    error.  This is the first line of the toolchain's own set_env.sh.
export ACL_OP_INIT_MODE=1

# 3. a libstdc++ with GLIBCXX_3.4.30 in front of the system one
#    (openEuler 22.03 ships 6.0.28; CANN's opmaster needs 3.4.29, libtvm.so needs 3.4.30)
export LD_LIBRARY_PATH=/path/to/glibcxx:${LD_LIBRARY_PATH}

# 3. torch (CPU build is enough; torch_npu owns the device) + torch_npu
pip install torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install torch_npu==2.10.0.post4

# 4. the TileLang Ascend toolchain (GitHub release of tile-ai/tilelang-ascend):
#    branch ascendc_pto (Ascend C / PTO backend, the default) or npuir (MLIR backend)
pip install "tilelang-0.1.4+linux.cann910-cp312-cp312-linux_x86_64.whl"

# 5. TileInfer itself
pip install -e .
python -c "import torch, torch_npu; print(torch_npu.npu.device_count())"
```

Note that `torch_npu` needs membership in the group owning `/dev/davinci*` (usually `HwHiAiUser`):

```bash
usermod -aG HwHiAiUser <user>
```

`have_tilelang()` in `tileinfer/utils.py` checks for the Ascend pass-config keys rather than just
the import, so a CUDA-only TileLang install correctly reports "no Ascend toolchain" instead of
failing later inside the compiler.

### Known environment traps (Ascend 950PR / CANN 9.1.1)

These cost real debugging time; they are listed here so nobody pays for them twice.

| Symptom | Cause | Fix |
|---|---|---|
| `libtvm.so: version GLIBCXX_3.4.30 not found` | system libstdc++ too old | conda-forge `libstdcxx-ng 12.2.0`, put it first on `LD_LIBRARY_PATH` |
| kernel runs but output is garbage — **including the upstream examples** | `ACL_OP_INIT_MODE` unset while `torch_npu` is loaded | `export ACL_OP_INIT_MODE=1` |
| `[Bisheng] Launch kernel failure! ret 507000` | ACL `RT_INTERNAL_ERROR`; observed together with the row above | same |
| `expected Object but got str` from `script.ir_builder.tir.Arg` | kernel module uses `from __future__ import annotations` (PEP 563 turns annotations into strings), or keeps `T` / shape constants in an enclosing closure | keep PEP 563 off in kernel modules and compute every shape constant inside the jit function |
| a `+linux.cann910` wheel on a CANN 9.1.1 box | the release asset is built for CANN 9.1.0 | build the wheel from source against the local CANN |

Status of the last row on the dev machine: the source build was attempted; the tree is at
`third_party/tilelang-ascend` (submodules vendored, TVM patches applied) and the CMake step needs
one full-log run to be fixed.  Until then, device numerics cannot be trusted, which is why the
kernel test in `tests/test_tilelang_decode.py` is reported as "compiles and launches" rather than
"validated".

## 7. Testing strategy

| Layer | Where it runs | What it protects |
|---|---|---|
| `tests/test_metadata.py` | CPU | the engine contract (indptr/page-table conversions) |
| `tests/test_plan.py` | CPU | tiling coverage, split accounting, LPT actually balances |
| `tests/test_reference.py` | CPU | tiled == dense oracle, tail masking, causal offset, GQA |
| `tests/test_attention_api.py` | CPU | plan/run lifecycle, plan cache, metadata overrides |
| `tests/test_tilelang_decode.py` | NPU (auto-skip) | kernel == reference on the device |

The rule the suite encodes: **a kernel is never compared against another kernel.**  Every device
test compares against `reference_attention`, and every scheduling feature is proven on the CPU
first.  That way a failure tells you *which* layer broke.
