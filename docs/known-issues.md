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

### Next steps

1. Bisect by shape pairs: compile+run b64/kv512 (unsplit) and then b16/kv4096 (split) in one
   process — if that is enough to fault, the trigger is "an unsplit kernel followed by a split
   kernel" rather than the number of shapes.
2. Set `TILELANG_CLEAR_CACHE=1` (or a fresh `TILELANG_CACHE_DIR`) for the sweep: if the fault
   disappears, the disk cache is involved after all.
3. Compare the generated CCE for the split kernel when compiled *after* an unsplit kernel in the
   same process (the JIT is in-process, so a compiler-state leak is plausible).
4. Reproduce under `ASCEND_LAUNCH_BLOCKING=1` to get the fault attributed to the launch that causes
   it instead of to the next op.

## WIP-1 — multi-tile prefill is off (open)

**Status:** open · `tilelang-ascend950`, prefill/append path.

`tests/test_tilelang_ascend950_prefill.py` covers four cases.  The **single-tile** one passes,
including the case that matters most for correctness — *append* (`qo_len=4` over 128 cached tokens),
where the causal offset `kv_len - qo_len` has to be right, plus a partial tail page.  Cases that span
**more than one query tile** (`qo_len * group > block_q`) are wrong by ~3.0, which is the magnitude of
a *misplaced row* rather than of a precision problem.

That isolates it nicely: the softmax, the paging, the causal mask and the append offset are all
exercised by the passing case; what is left is the tile/row mapping for tiles whose row offset is
non-zero, i.e. either

* the wrapper's packing/unpacking (`forward_prefill` maps `q_flat[b, bh, q_off : q_off + rows]`), or
* the kernel's row → query-position map `q_pos = (q_off + r) // group`.

Next step is a two-tile unit case with a hand-checked mask (e.g. `group=1`, `block_q=32`, `qo_len=64`
so that tile 1 starts exactly at a query position) which separates the two.

The kernel is kept in the tree, and the test asserts the single-tile configuration and `xfail`s the
rest, so the WIP cannot be mistaken for a working path.
