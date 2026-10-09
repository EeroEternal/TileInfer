# Upstream issue draft: PTO cube→vector hand-off hangs on Ascend 950PR (CANN 9.1.1)

Status: **draft, not filed.**  This is the shortest form of a bug we hit while bringing TileInfer up
on the reference machine.  It blocks every attention kernel we have tried — ours *and* the two
upstream ones — so it is worth reporting upstream rather than working around silently.

Repo: `tile-ai/tilelang-ascend`, branch `ascendc_pto` (HEAD at the time of writing, 2026-10-08),
built from source against the locally installed CANN 9.1.1.

## Summary

Any PTO kernel in which the cube core hands a result to the vector core hangs:

```
GEMM (cube) -> L0C -> GM workspace -> UB (vector) -> elementwise add -> output
=> AclrtSynchronizeDeviceWithTimeout ... 507014 (aicore timeout)
```

Cube-only kernels work, vector-only kernels work, so the failure is specific to the **hand-off**,
not to attention or to our kernel in particular.

## Minimal reproducer

[`benchmarks/probes/pto_cv_handoff.py`](../../benchmarks/probes/pto_cv_handoff.py) — 25 lines of
kernel body, `target="pto"`, the standard auto-sync pass configs:

```python
with T.Kernel(1, is_npu=True) as (cid, vid):
    a_l1 = T.alloc_L1([64, 64], "float16"); b_l1 = T.alloc_L1([64, 64], "float16")
    c_l0c = T.alloc_L0C([64, 64], "float32")
    u = T.alloc_ub([64, 64], "float32"); o = T.alloc_ub([64, 64], "float16")
    T.copy(A, a_l1); T.copy(B, b_l1)
    T.gemm_v0(a_l1, b_l1, c_l0c, init=True)
    T.copy(c_l0c, ws)        # cube -> GM
    T.copy(ws, u)            # GM -> vector
    T.tile.add(u, u, 1.0)
    T.copy(u, o); T.copy(o, C)
```

## Evidence matrix (one machine, one session)

| Kernel | Target | Result |
|---|---|---|
| `examples/gemm/example_gemm_pto_developer.py` (upstream) | `pto` | ✅ `Kernel Output Match!` |
| elementwise add, UB only (vector-only) | `pto` | ✅ max abs diff 0.002 (fp16 rounding) |
| **minimal cube→vector hand-off (above)** | `pto` | ❌ aicore timeout `507014` |
| `examples/sparse_flash_attention/example_sparse_flash_attn_gqa_pto_developer.py` (upstream) | `pto` | ❌ aicore timeout `507014` |
| `examples/sparse_flash_attention/example_sparse_flash_attn_gqa_pto.py` (upstream, explicit `T.Scope("C")/("V")` + cross flags) | `pto` | ❌ aicore timeout `507014` |
| `examples/flash_attention/paged_flash_attn_bhsd.py` (upstream, classic target) | `auto` | ❌ `TVMError: Unresolved call Op(tl.ascend_fill)` (see below) |

The same kernels compiled with the released `tilelang-…+linux.cann910` wheel instead fail earlier
and differently: even a plain elementwise add raises `ACL_ERROR_RT_AICORE_EXCEPTION` (507015), i.e.
the wheel's binaries are rejected by this device.

## Environment

| | |
|---|---|
| NPU | `Ascend950PR_9579` (single card, `npu-smi` reports `Ascend950PR`) |
| CANN | 9.1.1 (`ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.1`) |
| Driver | 25.7.rc1 (`Innerversion=V100R001C10SPC102B224`) |
| Platform detection | `torch.npu.get_device_name()` → `Ascend950PR_9579` → `"950" in name` → platform `A5` |
| Python / torch | 3.12.14 / torch 2.10.0+cpu + torch_npu 2.10.0.post4 |
| `ACL_OP_INIT_MODE` | 1 (CANN logs that it auto-switches to 2: "aclops disabled" — as intended here) |
| Build | `install_ascend.sh` from source, `USE_ASCEND ON`, PTO-ISA submodule at the pinned commit |

Note that the toolchain's simulator paths only know `Ascend950PR_9599` / `Ascend910_9599`, while the
card reports `Ascend950PR_9579`. Platform detection still resolves to `A5`, but the SKU difference
is a plausible suspect for the hand-off failing at runtime.

## Secondary finding: classic `ascendc` target cannot lower `T.tile.fill`

On the same HEAD, `target="auto"` (classic code generator) dies with

```
TVMError: Unresolved call Op(tl.ascend_fill)
```

because `src/target/codegen_ascend.cc` has handlers for `ascend_fill_experiment` and `npu.fill` but
none for `tl::ascend_fill()`, while `src/target/codegen_ascend_pto.cc:881` does.  `T.tile.fill`
lowers to `tl.ascend_fill`, so upstream's own `examples/flash_attention/paged_flash_attn_bhsd.py`
cannot compile on HEAD with the default target.  Possibly intentional during the PTO migration, but
it makes `target="pto"` mandatory — worth documenting in the installation guide.

## Questions for upstream

1. Is the cube→vector hand-off under PTO supported on `Ascend950PR_9579` (as opposed to `_9599`)?
2. Does the pinned PTO-ISA require a newer CANN than 9.1.1 for the CV sync path?  **We can already
   answer half of this: no.**  The reproducer fails identically on CANN 9.1.0, 9.1.1 and
   9.2.0-beta.2, with and without `ACL_OP_INIT_MODE=2` — see the table in
   [`architecture.md`](architecture.md#if-you-are-tempted-to-change-the-cann-version).
3. Is the cross-core synchronisation expected to require a build with `--enable-shmem` (i.e. is the
   current failure an unsupported configuration rather than a bug)?
4. Should the classic `ascendc` target still support `T.tile.fill`, or is `target="pto"` now the
   only supported route on A5?
5. Is there a known-good commit/tag for A5 attention we should pin instead of HEAD?
