# Nemotron v2 → v3 migration (2026-11-16)

The Nemotron work was originally implemented on the v2 operator path
(`tileinfer/vllm/ops/*`). v2 is now frozen. This note records what moved and what still needs
hardware evidence.

## Canonical locations

| Concern | v3 location | Frozen v2 location |
|---|---|---|
| Mamba2 chunk scan + selective state update | `tileinfer/providers/nemotron/mamba2.py` | `tileinfer/vllm/ops/mamba2.py` (shim) |
| generic + non-gated `relu2` MoE compute | `tileinfer/providers/moe.py` | `tileinfer/vllm/ops/moe.py` (shim) |
| OOT `UnquantizedFusedMoEMethod` | `tileinfer/providers/fused_moe_oot.py` | `tileinfer/vllm/ops/fused_moe_oot.py` (shim) |
| non-gated FusedMoE guard / Ascend experts / shared alias | `tileinfer/bridge/nemotron_moe.py` | `tileinfer/vllm/ops/moe_guard.py` (shim) |
| `nemotron_h` config alias | `tileinfer/bridge/nemotron_config.py` | `tileinfer/vllm/config_compat.py` (shim) |
| provider declarations | `tileinfer/providers/nemotron/__init__.py` | — |
| plan-driven installation | `tileinfer/bridge/impl_table.py` | `tileinfer/vllm/ops/install.py` (not called by v3) |

## Behavior changes

1. `tileinfer/bridge/install.py` no longer imports or calls `v2 install_ops()`. Provider
   installation is the `impl_table` seam.
2. `seams.install_all()` runs a seam only when the plan lists its access point.
3. `impl_table` installs only non-vendor selections. No selection, missing access point,
   unknown provider, or `Plan.servable=false` fails closed; the vendor table stays in place.
4. `decide()` adds `impl_table` to `access_points` only when it actually selects a non-vendor
   provider, and records the gap verdict in `Plan.servable`.
5. The two Nemotron providers remain **candidates** (`tileinfer.mamba2_state`,
   `tileinfer.nemotron_moe`); they are not defaults and the target plan stays `servable=false`
   until real-weight C1/C2/C3 exists.

## Offline verification (no NPU)

- `tests/test_v3_impl_table.py`: plan gating, unservable refusal, unknown provider refusal,
  Mamba2 target binding, no v2 import in the v3 bridge, no v2 import in providers.
- `tests/test_v3_contracts.py`: plan round-trip including `servable`; access-point budget.
- `tests/test_nemotron_offline.py`: declarations, static audit, config normalization.
- `tests/test_mamba2_ops.py`, `tests/test_moe_ops.py`, `tests/test_fused_moe_oot.py`:
  numerically checked against the scalar/per-token reference on CPU.

These are offline checks only. C1 (real service), C2 (HF reference consistency), and C3
(graph capture) still require the NPU and explicit consent; no support claim is made here.
