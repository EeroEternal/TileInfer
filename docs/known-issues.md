# Known issues and post-mortems

## PM-1 — "kernel memory corruption" that was a bad gather index (resolved)

**Status:** resolved · **Lesson:** on Ascend, an out-of-range *gather* index is not a fault — it
returns whatever is at that address, which is indistinguishable from a kernel scribbling over
memory.

### What it looked like

A device test that builds a paged KV cache reported that a tensor the kernels only ever *read* came
back corrupted:

```
before: [15, 11, 3, 6, ..., 36, 32, 35, 34, 33, 37, 39, 38]
after : [15, 11, 3, 6, ..., 10573, 0, 42304, 0, 0, 8, 0, 42304]
```

request 0's page ids were intact, request 1's were replaced by garbage — and the *same* garbage
every run.  That fingerprint (deterministic "corruption", starting exactly at a tile boundary) sent
us looking for an out-of-bounds write in the split/merge kernels, and a gate was added that refused
`kv_tile_pages > 0` with `batch > 1`.

### What it actually was

The test's cache builder permuted each request's page list like this:

```python
perm = torch.randperm(end - start, generator=gen) + start   # wrong: global offset ...
indices[start:end] = indices[start:end][perm.to(device)]    # ... applied to a local slice
```

The `+ start` turned a 0-based permutation into global page ids, so the gather indexed a 32-element
slice with values 32..39.  On Ascend that silently returns unrelated device memory instead of
raising (the same mistake raised `IndexError` on the CPU path, which is why the *reference*
computation failed later with `index 42304 is out of bounds for dimension 0 with size 40`).

Fixing the permutation to be 0-based makes every case pass, including multi-request split
schedules, and the "corruption" disappears with no kernel change.

### Why it took so long to see

* The garbage was deterministic, so it looked like real data being written somewhere it should not
  be — rather than like uninitialised memory.
* Our debug hook (`.cpu()` between the two launches) reported the tensor as intact: torch copies
  are ordered on *torch's* stream, while the TileLang kernels are launched on their own, so the
  check ran before the kernels had finished.  `torch.npu.synchronize()` in the test then made the
  writes visible.
* Interleaving guard tensors around the plan's buffers made the symptom vanish — because the
  canaries changed the allocator layout — which "confirmed" a layout-dependent overrun.

### Rules that came out of it

1. **Always bounds-check indices on the host before they reach a device gather** (the test now
   asserts `kv_indices` integrity right after the run, so a future repeat fails loudly).
2. When a device test disagrees with a CPU reference about *values*, first check for
   `IndexError`-class mistakes that Ascend does not report: gather indices, out-of-range slices.
3. Synchronise the device (`synchronize`) before trusting a debug peek, not just the current stream.

Two changes made while chasing this are worth keeping and are now permanent: the kernels read
**plan-owned copies** of the page table (stable addresses, caller tensors can never be touched), and
the merge kernel stores a **full padded tile** (`BR` rows) rather than only the live `group` rows, so
every write is tile-aligned and in-bounds by construction.

## KI-2 — bundled backends did not register inside a vLLM EngineCore (resolved)

**Status:** resolved, and the first diagnosis was incomplete.  Two separate defects, both in
`_ensure_builtin_backends()`:

1. it returned early once `reference` was registered, so a failure while importing a later backend left
   the registry permanently incomplete (fixed first: each backend is now imported independently and its
   failure is collected in `_IMPORT_ERRORS` and quoted in the error message);
2. the "loaded" flag was set **before** the imports and nothing was synchronised.  Engines call this
   from several threads at once — vLLM builds one backend object per attention layer and the first
   decode step of every layer asks the registry simultaneously — so one thread returned immediately,
   looked up `tilelang-ascend950`, and raised `unknown backend ...; known: ['reference']` while another
   thread was still importing.  A single-threaded test could never see it, which is why the earlier
   "resolved" was wrong.  The import now runs under a lock and the flag is set after it, and
   `tests/test_backend_registry.py` races eight threads at the registry (verified to fail against the
   old ordering).

Both defects are the same user-visible message, which is what made this expensive to untangle: it took
one log line per import (the `_diag` calls, now at debug level) to see `registry=[]` and
`registry=['reference']` on consecutive lines.

The original note follows.

**History:** affects the vLLM integration only (the standalone library is fine).

In the vLLM plugin's own process every bundled backend registers
(`['reference', 'tilelang', 'tilelang-ascend950']`); inside the EngineCore `list_backends()` returns
`['reference']`, and asking for `tilelang-ascend950` raises `unknown backend ... known: ['reference']`
**without** any import error being recorded - which `_ensure_builtin_backends()` (independent imports,
failures collected in `_IMPORT_ERRORS` and quoted in the message) cannot produce as written, so
something about the engine process is *undone* rather than failing.

Eliminated already: a duplicate module identity from having TileInfer both editable-installed and on
`PYTHONPATH` (removed, no change).

Next: one probe that logs `list_backends()`, `_REGISTRY.keys()` and `_IMPORT_ERRORS` at plugin load
time and again inside `_build_plan()`.  That distinguishes "imported but unregistered" from "never
imported" and points at the line to fix.  Meanwhile the plugin falls back to FIA, which is why the
integration is still opt-in.

## KI-1 — split-KV wedges the device when several shapes share one process (open)

**Status:** open · **Affected:** `tilelang-ascend950` with `kv_tile_pages > 0` ·
**Default:** the backend **refuses** split plans unless `TILEINFER_ALLOW_SPLIT_KV=1`.

### Symptom

Running the serving sweep (`benchmarks/bench_attention.py --preset serving --kv-tile-pages 16`,
seven shapes in one process) dies on the third shape:

```
b64/kv512   -> ok          b16/kv4096 -> ACL_ERROR_RT_VECTOR_CORE_EXCEPTION (507035)
b32/kv2048  -> ok          b8/kv8192  -> every later op fails (device wedged)
```

The exception is reported *asynchronously*: it surfaces in whatever torch op runs next
(`aclnnInplaceNormal`, then a host copy), not at the launch that faulted.  After it, the device
rejects everything, so the sweep cannot continue.

### What is ruled out

* **The shape itself.**  `b16/kv4096` with `kv_tile_pages=16` (2 splits) run alone produces
  `max|diff| = 4e-4` against the CPU reference, and **20 consecutive launches of that same plan in
  one process are all clean**.
* **The arithmetic.**  All seven device tests pass, including split schedules (single request with
  4 splits, and a ragged 2-request batch with different split counts).
* **The harness' reference.**  The first sweep ran the reference on the NPU and failed the same way;
  moving the reference to the CPU (which is also what the probes do) did not change it.
* **A stale kernel cache.**  `tilelang`'s cache key is a SHA-256 of the generated function binary,
  so a shape cannot silently reuse another shape's kernel.  (The ABI check would also catch a
  size mismatch, as it did once with `kv_indices`.)

### Why we are not shipping it enabled anyway

A vector-core exception is not a wrong number: it wedges the device for the rest of the process, so
any engine that hit it would take the whole serving process down.  Until it is understood, split-KV
is opt-in, and the documented way to get the 3x is to run it per shape (or in a fresh process).

### Retracted: the vLLM EngineCore fault was not ours

An earlier revision of this file claimed a second reproducer - the first decode inside a vLLM
EngineCore faulting with the same 507035.  That is **withdrawn**: the same server, same model, same
request with TileInfer switched off (`TILEINFER_DISABLE=1`, so the stock Ascend FIA path) crashes
identically.  The model in that experiment was a hand-made tiny Qwen2 (2 layers, vocab 1024, random
weights) built to get `head_dim=128` on a box that has none; something in that configuration faults on
this stack regardless of the attention backend, and it is not our bug.

What *is* verified in vLLM is in [`integration-vllm-ascend.md`](integration-vllm-ascend.md): the
backend registers, is selected, is entered for decode-only batches, and declines configurations its
kernel cannot serve (Qwen2.5-0.5B's `head_size=64`) with a logged reason before falling back to FIA.

Three more hypotheses for KI-1 were tested and **refuted** while chasing this, and each one is a
result worth keeping:

| Hypothesis | Experiment | Result |
|---|---|---|
| caller buffers unaligned | slice every input out of a bigger allocation at 2 B / 16 B / 128 B / 512 B offsets | clean at every offset |
| pool larger than 2 GiB overflows 32-bit addressing | compile and run against pools of 0.5 / 2.4 / 7.3 GiB | clean at every size |
| the kernel ignores the caller's NPU stream | write the page on a non-default stream, launch immediately, ask which content it saw | saw the new content: ordering respected |

So KI-1 remains what it was: the split path, several shapes in one process, no reproducer outside the
sweep.

### Reproducer, and what has been ruled out (updated)

```bash
source <CANN 9.3.x>/set_env.sh
export TILEINFER_ALLOW_SPLIT_KV=1
python benchmarks/bench_attention.py --preset repro --backend tilelang-ascend950 \
       --dtype bf16 --kv-tile-pages 16
# b64/kv512   -> ok
# b32/kv2048  -> ok
# b16/kv4096  -> ACL_ERROR_RT_VECTOR_CORE_EXCEPTION (reported by the next torch op)
```

`--preset repro` is exactly the three-shape prefix that fails, and it is kept in the harness for
this purpose.

Ruled out, each by running the reproducer above with one thing changed:

| Hypothesis | Experiment | Result |
|---|---|---|
| the harness' NPU reference | `--no-check` | still faults |
| repeated launches / benchmark loop | `--warmup 0 --iters 1` | still faults |
| a stale disk-cached kernel | `TILELANG_CLEAR_CACHE=1` (forces a recompile) | still faults |
| a stale in-memory kernel | code read: the cache key is a SHA-256 of the compiled function, not its name | impossible |
| the shape itself | `b16/kv4096 kv_tile_pages=16` alone, and 20 consecutive launches of one plan | clean |
| "unsplit kernel first, then split" | a probe running that pair in both orders | clean |
| the exact case construction | a probe mirroring `make_case_tensors` (same `randn`, `shuffle_pages=True`, `qo_indptr`, three launches with per-launch sync) for the same three shapes | **clean** |

The last row is the interesting one: an exact mirror of the *cases* passes, so the trigger is not the
shapes, the arguments or the page table but some **process state the harness has and the probe does
not** (allocation history/layout is the obvious candidate - it is what made an earlier guard-tensor
experiment "prove" a layout-dependent overrun in PM-1 as well).  A device debugger, or
`ASCEND_LAUNCH_BLOCKING=1` plus per-launch attribution, is the way in.

Because a device fault takes the process with it, split-KV stays opt-in meanwhile.

### Next steps

1. Run the reproducer under `ASCEND_LAUNCH_BLOCKING=1` so the fault is attributed to the launch that
   causes it rather than to the next op; that alone should say whether it is the split kernel or the
   merge kernel.
2. Instrument the harness path to dump the device pointers and shapes of every tensor handed to the
   kernels, and diff that against the probe run that passes - if a *layout* difference is the trigger,
   the addresses will show it (e.g. a buffer landing next to an unmapped page).
3. Try the harness with the plan buffers allocated through a different allocator path (e.g. one extra
   dummy allocation between them) to test the layout sensitivity directly.
4. Ask upstream with the reproducer: a `T.SimtVF` + `dual_copy` kernel that faults only in some
   allocation layouts is worth their attention regardless of whose bug it is.


## PM-2 — multi-tile prefill was off by a row offset (resolved)

**Status:** resolved (with a documented cost) · `tilelang-ascend950` prefill path.

### Symptom

Prefill/append matched the reference for a single query tile, and was wrong by ~3.0 - the magnitude
of a *misplaced row*, not of a precision problem - as soon as a request spanned more than one tile.
The single-tile case that passed included the append offset `kv_len - qo_len`, so the mask, the
paging and the softmax were all fine; only the row mapping could be wrong.

### Root cause

The causal mask needs the row's *global* query position, and the number the kernel can compute inside
the vector region is the *per-AIV* row index: `dual_copy` splits the M tile over the two AIVs and each
one sees `ROWS = block_q // 2` rows **starting at 0**.  For a tile whose live rows all fall in the
first half, local == global (hence the single-tile pass); otherwise the second AIV computes its mask as
if it held the first half's rows - and its output is *not* discarded, it is written into the upper half
of the packed tile.

The dialect does not expose the AIV index to a `T.SimtVF` region.  It is available only through the
explicit mixed-kernel structure: `with T.Vector(vector=2) as sid:` (documented in
`tilelang/ascend/language/frame.py`, where `sid = asc_get_sub_block_id()`), which also implies
wrapping the cube work in `with T.Cube():`.

### Resolution

For now: **the caller plans its query tiles with `block_q // 2` rows**, so every live row is in the
first AIV half.  `forward_prefill` enforces it and explains why; the tests cover multi-tile prefill,
append, a partial tail page and a ragged batch.  The cost is that half of the M tile is padding, i.e.
~2x the cube work a prefill could need.

The proper fix - worth ~2x on prefill - is to restructure the kernel into explicit `T.Cube()` /
`T.Vector(vector=2) as sid:` regions and mask with `sid * ROWS + r`.  That is a bigger change (it also
moves the `dual_copy`s into the vector region) and it needs device iteration, so it is a standalone
task rather than a footnote to this one.
