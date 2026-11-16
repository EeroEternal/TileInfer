# Nemotron 热路径算子清单（静态，O1/O4 补充）

> 只读源码，不 import vLLM、不碰 NPU。结论是**候选缺口**，不是“不支持”的证明；
> 真正是否走到这些算子、是否已在 Ascend 上编译通过，必须上机验证。

---

## 1. 模型热路径

`vllm/model_executor/models/nemotron_h.py` 的 decoder 层有三类：

| 层 | 类 | 关键调用 |
|---|---|---|
| `M` | Mamba2 mixer | `MambaMixer2`（`:378`） |
| `*` | full attention | `Attention`（`:468`） |
| `-` / `E` | MLP / MoE | `FusedMoE`（`:217`）、`RMSNorm`（`:304/347/396/517`） |
| 公共 | embedding / lm head | `VocabParallelEmbedding`（`:559`）、`ParallelLMHead`（`:875`） |

`MambaMixer2` 内部（`mamba_mixer2.py:530`）主要走：

- `causal_conv1d_fn` / `causal_conv1d_update`
- `mamba_chunk_scan_combined_varlen`（`ssd_combined.py`）
- `selective_state_update`（`ssu_dispatch.py`）
- `rms_norm_gated`

---

## 2. Ascend 侧静态覆盖

| 算子/路径 | Ascend 证据 | 状态 |
|---|---|---|
| `causal_conv1d_update` | `patch/worker/patch_triton.py:10` 替换成 `causal_conv1d_update_npu` | ✅ 有替换 |
| `causal_conv1d_fn` | 只有 `_310p/ops/causal_conv1d.py` 的 PyTorch fallback | ⚠️ 主路径未确认 |
| `mamba_chunk_scan_combined_varlen` / `ssd_combined` | 全树搜索无 `mamba_mixer2`/`ssd_combined`/`mamba_chunk_scan` 命中 | ❌ **候选缺口** |
| `selective_state_update` | 全树无同名替换；`ssu_dispatch` 可能走 torch fallback | ⚠️ 未确认 |
| `rms_norm_gated` | 仅 KDA 路径有 `rms_norm_gated`；Mamba2 未确认 | ⚠️ 未确认 |
| `MambaMixer2` 本身 | 无 Ascend 专用替换/包装 | ⚠️ 走 vLLM 通用路径 |
| hybrid Mamba cache config | `patch/platform/patch_mamba_config.py:149` | ✅ 有 patch（GDN 取向） |
| MoE | `vllm_ascend/ops/fused_moe/` + patch | ✅ 有 Ascend 实现，形状待验 |
| attention backend | Ascend 有 attention backend，但 Mamba/attention 混合层未验 | ⚠️ 上机 |

---

## 3. 结论

- **最高风险**不是“Ascend 没有 Mamba”，而是 **Mamba2 的 SSD/chunk-scan 没有 Ascend 替换**；
  `causal_conv1d` 和 cache config 已有部分覆盖，不能因为它们存在就推断模型能跑。
- `Mamba2Contract`（O4）对应的是“补 SSD/mixer 执行”，不是补 `causal_conv1d`。
- 若上机 dummy 在加载/编译阶段就报 `ssd_combined`/`mamba_mixer2` 不可用，应按 P2 处理；
  若只是图捕获失败，先按 P5 graph hook 处理，不要误判为算子缺失。
- 本清单不替代 D8 的 `TILEINFER_AUDIT` 运行结果；D8 必须给出实际热路径命中。

---

## 4. 复现（只读）

```bash
grep -n "MambaMixer2\|FusedMoE\|Attention" \
  .venv_v3/lib/python3.12/site-packages/vllm/model_executor/models/nemotron_h.py
grep -n "causal_conv1d\|ssd_combined\|selective_state_update\|rms_norm_gated" \
  .venv_v3/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/mamba_mixer2.py
grep -R "mamba_mixer2\|ssd_combined\|mamba_chunk_scan" \
  .venv_v3/lib/python3.12/site-packages/vllm_ascend --include=*.py
```
