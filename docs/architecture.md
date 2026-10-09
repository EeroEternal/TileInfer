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
| `ACL_ERROR_RT_AICORE_EXCEPTION` (507015) on **every** kernel, including a plain elementwise add | the `tilelang-…+linux.cann910` release wheel is built for CANN 9.1.0 and emits binaries this device rejects | do not use the wheel; build from source against the local CANN |
| `TVMError: Unresolved call Op(tl.ascend_fill)` | the classic `ascendc` code generator has no handler for `tl.ascend_fill` (which `T.tile.fill` lowers to) — only `codegen_ascend_pto.cc` implements it | compile with `target="pto"` (see `tileinfer.kernels.attention.paged_decode.ASCEND_TARGET`, overridable via `TILEINFER_ASCEND_TARGET`) |
| `expected Object but got str` from `script.ir_builder.tir.Arg` | kernel module uses `from __future__ import annotations` (PEP 563 turns annotations into strings), or keeps `T` / shape constants in an enclosing closure | keep PEP 563 off in kernel modules and compute every shape constant inside the jit function |
| `3rdparty/pto-isa/.../TAssign.hpp: no member named 'assignData' in pto::Tile<Acc, …>` | the pinned `pto-isa` cannot express `T.assign` on a *small* Accumulator tile (observed with M=2 and M=4; M=16 compiles) | keep the GQA group (the first GEMM's M) at 16 or above, or avoid materialising a tiny L0C tile |
| `AclrtSynchronizeDeviceWithTimeout … 507014` (aicore timeout) | kernel deadlocks.  In practice this is what happens when a kernel written against the **classic `ascendc` CV model** (cube ↔ vector hand-off through a workspace tensor plus the automatic sync passes) is compiled with `target="pto"` | write the kernel against the PTO execution model, i.e. follow `examples/sparse_flash_attention/example_sparse_flash_attn_gqa_pto.py` (explicit `T.Scope("C")/T.Scope("V")` and cross-core flags) instead of `examples/flash_attention/paged_flash_attn_bhsd.py` |

`have_tilelang()` in `tileinfer/utils.py` checks for the Ascend pass-config keys rather than just
the import, so a CUDA-only TileLang install correctly reports "no Ascend toolchain" instead of
failing later inside the compiler.

### The working stack (verified 2026-10-09)

TileInfer's device path is **not** the `tile-ai/tilelang-ascend` fork.  The combination that works on
the reference box is the **official** TileLang wheel with a **newer CANN**:

| | |
|---|---|
| TileLang | `tilelang==0.1.15` from PyPI — it ships the **Ascend 950 backend** (`tilelang.ascend`, a DeepSeek-built dialect, `target="ascend"`, default arch `dav-3510`) |
| CANN | **9.3.0** (weekly build), installed side-by-side in a user prefix; the system 9.1.1 and the driver stay untouched |
| torch / torch_npu | 2.12.0 / 2.12.0.post2 |
| device | `Ascend950PR_9579` |

Verification, reproduced from a clean shell (see `docs/architecture.md` history for the failing
fork runs):

```
source <CANN 9.3.0>/set_env.sh        # + /home/lipi/glibcxx on LD_LIBRARY_PATH
python check_tilelang_npu_kernel.py   # upstream's Ascend 950 Quick Start: C = relu(A @ B^T)
  -> tilelang 0.1.15, npu: True (Ascend950PR_9579)
  -> "GEMM + ReLU passed. All check passed."
```

That kernel contains a real cube→vector hand-off (`T.dual_copy(C_l0c, C_ub)` then a
`T.SimtVF(threads=128)` region), i.e. exactly the primitive whose fork/PTO equivalent hung with
`aicore timeout` on this device for three different CANN versions (below).

What the upstream backend gives us, and what it does **not** (from
`examples/ascend/flash_attention/README.md`, which is the closest template):

| Have | Missing — i.e. TileInfer's job |
|---|---|
| Faster-than-SDPA MHA/GQA forward (320–362 TFLOPS on 950, bf16, `head_dim` 128, dense) | **paged KV** and any engine metadata contract |
| 128x128 tiles, online softmax, TMEM/L0 staging, automatic Cube/Vector sync | **decode** (one query row per request) and **ragged / variable lengths** |
| SIMT + SIMD mixing, `T.dual_copy` | **causal masking and padding masks** (upstream explicitly refuses them) |
| `T.Pipelined`, `T.Persistent`, GQA via flattened `q_len = S1 * G` | attention sinks, MLA/sparse, low precision KV |

**Port status (same day).**  The decode kernel is ported and validated: `tilelang-ascend950`
backend -> `kernels/attention/paged_decode_ascend950.py`, 5 device cases (full page, partial tail,
single token, ragged batch, group == M tile) match the torch reference to ~1e-3 with bf16 on
`Ascend950PR_9579`.  Three constraints of the dialect are worth knowing before writing another
kernel:

* the ND→NZ copy template asserts `ROWS % 16 == 0`, and `dual_copy` splits M across the two AIVs,
  so the M tile must be a multiple of 32 — a GQA group of 8 is padded to 32 (the caller passes
  zero-padded `Q`/`Out` and slices back);
* the kernel's page-index argument is a **fixed-size** ABI slot: pad `kv_indices` to the compiled
  pool size instead of recompiling per step (unused slots are never dereferenced);
* `from __future__ import annotations` breaks the eager builder too — it calls `get_type_hints`,
  which then cannot resolve the enclosing function's locals.

So the plan of record becomes: keep TileInfer's metadata / planner / plan-run / reference layers
(device-independent, already tested), and implement the kernels against `tilelang.ascend`, reusing
upstream's `examples/ascend/flash_attention/core.py` shape and tiling where it applies.

### The fork, kept for the record

The `tile-ai/tilelang-ascend` fork (branch `ascendc_pto`) was a dead end **for this device**, and
the reasons are worth keeping because they cost days to establish:

What is *known to work* under the fork, in order of how much it proves:

| Check (fork, `target="pto"` unless noted) | Result |
|---|---|
| upstream PTO GEMM example (`examples/gemm/example_gemm_pto_developer.py`, cube only) | ✅ `Kernel Output Match!` |
| trivial elementwise add (vector only) | ✅ max abs diff 0.002 (fp16 rounding) |
| **minimal cube→vector hand-off** (`benchmarks/probes/pto_cv_handoff.py`) | ❌ aicore timeout `507014` |
| upstream PTO attention examples (developer mode, and the explicit-scope one) | ❌ aicore timeout `507014` |
| TileInfer paged decode kernel | ❌ same aicore timeout (not its own bug) |
| upstream `examples/flash_attention/paged_flash_attn_bhsd.py` (`target="auto"`) | ❌ `Unresolved call Op(tl.ascend_fill)` |

The conclusion is uncomfortable but useful: **the cube→vector hand-off under the PTO target does
not work on this machine**, for upstream's kernels as much as for ours.  Cube-only and vector-only
kernels are fine.  So there is nothing to fix in TileInfer before upstream does; the minimal
reproducer lives in `benchmarks/probes/pto_cv_handoff.py` and the issue draft (with the full
environment matrix and questions) in
[`upstream-issue-pto-cv-hang.md`](upstream-issue-pto-cv-hang.md).

Options while that is open, in the order we would try them:

1. **`npuir` branch** — the second, MLIR-based backend route of the same repository
   (`target="npuir"`, explicit `T.Scope("Cube")` / `T.Scope("Vector")`, same `alloc_L1/L0C/ub`
   vocabulary).  Investigated on the reference machine: the tree is pushed and clean at
   `third_party/tilelang-npuir`, `bishengir-compile` **is** present in CANN 9.1.1
   (`/usr/local/Ascend/cann-9.1.1/bin/bishengir-compile`, plus an `-a5` variant), but
   `install_npuir.sh` additionally requires `python_packages/{bishengir,mlir_core}`, which the CANN
   toolchain does not ship — so it falls back to building **AscendNPU-IR from the vendored
   submodule**, i.e. an LLVM/MLIR-scale build needing Clang 15/LLD 15 (not installed on the box):
   1–2 hours, `--bishengir-path=` then points at `3rdparty/AscendNPU-IR/build/install`.
   Note the branch's own Docker recipe (`docs/Docker-README.md`) bakes **CANN 8.5.0 + Clang 15 +
   pre-compiled AscendNPU-IR** into an image, which would combine two of the experiments below
   *without touching the host* — often the cheapest way to answer this question.
2. **Vector-only decode kernel** — decode's M is the GQA group, so the whole online-softmax loop
   can in principle run on the vector core with no cube involvement, sidestepping the hand-off.
   Correct and runnable today; slower than the cube path, so it is a functional fallback rather
   than the target design.
3. **A different CANN** — see the section below on why "newer" is more promising than "older",
   and why any such change must be a side-by-side userspace install or a container.
4. **Wait for upstream**, using the reproducer to drive the fix.

### If you are tempted to change the CANN version

We tested it, so you do not have to.  The same 25-line canary
(`benchmarks/probes/pto_cv_handoff.py`) fails identically on three CANN versions, one of them
newer than ours and one of them the exact version the release wheel was built for:

| CANN | how it is installed on the reference box | `ACL_OP_INIT_MODE` | canary result |
|---|---|---|---|
| 9.1.1 | system (`/usr/local/Ascend/cann-9.1.1`) | 1 (CANN upgrades it to 2) | ❌ aicore timeout `507014` |
| 9.1.0 | side-by-side user prefix (by another user on the same box) | 2 | ❌ aicore timeout `507014` |
| 9.2.0-beta.2 | side-by-side user prefix | unset **and** 2 | ❌ aicore timeout `507014` |

So the cube→vector hand-off failure is **not** a CANN version issue, and upgrading (to 9.3.0 or
otherwise) is not a fix we would spend a maintenance window on.  Two useful side effects of that
sweep: switching CANN in a shell is genuinely risk-free (the user-local prefix pattern works, the
system install and the driver are untouched), and the *release wheel* can be tested against its
own target version if ever needed.

The next hypothesis worth a build is a **missing build option**: the PTO route's cross-core
synchronisation is likely implemented with the shared-memory (``shmem``) feature that
``install_ascend.sh --enable-shmem`` turns on, and the current build was made without it.  If that
does not fix the canary either, the remaining explanation is the SKU itself (``Ascend950PR_9579``
vs the ``_9599`` the toolchain knows about) and it belongs upstream.

If a CANN change is ever needed for another reason, keep the constraints below in mind.

Hard constraints for any such experiment: never modify the system install
(`/usr/local/Ascend/cann-9.1.1`) — another user's vLLM service runs against it on this box; CANN is
version-locked across three packages (`toolkit` + `nnal` + `Ascend-cann-950-ops`, ≈4.4 GB); and the
clean way is a side-by-side install under `/home/<user>` (or a container) selected per shell via
`source …/set_env.sh`, leaving the driver alone.  The decisive test is one variable at a time:
`python benchmarks/probes/pto_cv_handoff.py`.

Two environmental facts that surprised us and are worth keeping:

* The device reports **`Ascend950PR_9579`** while the toolchain's simulator paths only know
  `Ascend950PR_9599`/`Ascend910_9599`; platform detection (`"950" in name → A5`) does the right
  thing regardless — and that SKU difference is a prime suspect for the hand-off failure.
* On this box `ACL_OP_INIT_MODE=1` is silently upgraded by CANN to `2` (aclops disabled), which is
  the annotation CANN wants for custom kernels here — set it anyway, since with it unset the
  failure mode is *wrong numbers* rather than an error.

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
