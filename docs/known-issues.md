# Known issues

## KI-1 — out-of-bounds device write in multi-request split-KV schedules

**Status:** open · **Affected:** `tilelang-ascend950` backend with `kv_tile_pages > 0` and
`batch > 1` · **Workaround:** the backend refuses that combination (`NotImplementedError`);
single-request split-KV is validated and enabled.

### Symptom

After the two launches (split attention, then merge), a tensor the *caller* owns and the kernel
only ever *reads* comes back corrupted.  Observed on `kv_indices`:

```
before: [15, 11, 3, 6, ..., 36, 32, 35, 34, 33, 37, 39, 38]   # valid page table
after : [15, 11, 3, 6, ..., 10573, 0, 42304, 0, 0, 8, 0, 42304]
                              ^^^^ request 0's entries are fine, request 1's are overwritten
```

The values (`10573`, `42304`) are not page ids; they look like fragments of other device buffers.
Note the boundary: the corruption starts exactly at `BR = 32` elements in, i.e. one padded M tile
worth of data, which is the granularity of the `dual_copy` epilogue writes (`out_ub[0:ROWS, ...]`
of two AIVs becoming `BR` rows, and the 1-D lse copy of the same shape).

### Reproduce

```bash
source <CANN 9.3.x>/set_env.sh
python -m pytest "tests/test_tilelang_ascend950_decode.py::test_paged_decode_matches_reference[2-2-8-kv_lens6-split-KV, ragged-8]"
# -> AssertionError: kv_indices corrupted after run: [...]
```

The test asserts integrity of `kv_indices` right after `run`, so the failure is a *memory* failure,
not a numerics failure.

### What has been ruled out

* **Not the arithmetic.** With a single request the same two kernels go through the public
  `plan`/`run` path and match the torch reference (this is the `split-KV, 4 tiles` test case), and
  `benchmarks/probes/split_kv_decode.py` merges 4 partials to within 2e-4 of the dense oracle.
* **Not the page table's values.** Rebuilding the same schedule with an identity page table
  corrupts nothing.
* **Not the plan's buffer sizes**, as far as reading the code goes: with `max_splits = 4`,
  `batch = 2`, `kv_heads = 2`, `BR = 32` the largest index written is
  `(slot * kv_heads + bh) = 9` into `PartOut` (16 entries) and `320` into the flat `PartLse`
  (512), and the merge writes `group = 8` rows at `bh * BR`.
* Calling `paged_decode_split` and `paged_decode_merge` directly with the plan's own buffers, the
  same schedule and a permuted page table leaves `kv_indices` intact — the corruption only shows up
  through the backend path, which is why this is still open rather than explained.

### Next steps

1. Instrument `forward_split` to snapshot `kv_indices` between the two launches (the probe already
   has the helper) and bisect which launch writes past its buffer.
2. Suspect the 1-D `dual_copy` for the lse vector: the source is `ROWS` floats per AIV while the
   destination range spans `BR`.  Replacing it with two explicit 1-D copies (or a GM store from a
   `BR`-row UB buffer) is a cheap experiment.
3. If neither shows up, capture the launch arguments per case and diff them against the
   single-request schedule that works — the difference is "more than one request in the slot
   table", so the slot arithmetic (`seq_ids * max_splits + split_ids`) deserves a second look for
   requests whose split counts differ (4 vs 1 here, so slots 5..7 are padding).
