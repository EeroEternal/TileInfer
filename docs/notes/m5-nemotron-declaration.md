# Nemotron 模型声明草案（O2，离线）

> 状态：**草案，不是可部署声明**。目标 checkpoint 的 `config.json` 尚未到位；
> 下列字段里带 `TODO(config)` 的必须用真实 config 填，不能拿默认值冒充。
> 本文件不写“用谁的实现”；实现选择属于 plan，不属于声明。

---

## 1. 声明必须覆盖的字段

对齐 [`tileinfer/contracts/manifest.py`](../../tileinfer/contracts/manifest.py)：

| 字段 | 目标值/来源 | 当前状态 |
|---|---|---|
| `name` | `Nemotron-3.5-Lightning-30B-A3B` | 已定 |
| `manifest_id` | `nemotron-3.5-lightning-30b-a3b` | 草案 |
| `arch` | `NemotronHForCausalLM` | 待 config 确认 |
| `model_type` | `nemotron_h`（MTP 时 `nemotron_h_mtp`） | 待 config 确认 |
| `num_layers` | `TODO(config)` | 未知 |
| `layer_types` | 由 `hybrid_override_pattern` 展开 | 未知；见 §3 |
| `attention` | `num_attention_heads` / `num_key_value_heads` / `attention_head_dim` | 未知 |
| `mamba` / `linear` | Mamba2 参数（不是 GDN） | **schema 待扩展** |
| `moe` | 专家数、top-k、intermediate | 未知 |
| `quantization` | BF16 checkpoint → `none` | 预期，待确认 |
| `kv` | full attention + Mamba state 的混合 | 未知；见 §3 |
| `required_capabilities` | full attention、Mamba state、MoE、RMSNorm、RoPE、sampling | 待映射 |
| `benchmark` | 固定 prompt、max_tokens、first-token top5 | 参考实现到位后填 |
| `source` | 本任务书/静态 gap 文档 | 草案 |

---

## 2. 不能现在填死的字段

- **层数/pattern**：必须从目标 checkpoint 的 `config.json` 读；vLLM 的
  `NemotronHConfig` 默认值（52 层等）不是这个 checkpoint 的事实。
- **MTP**：只有 `num_nextn_predict_layers > 0` 且权重里确实有 MTP heads 才能声明；
  否则 D4 直接上报“无 MTP”，不能用 ngram 替代。
- **上下文**：`max_position_embeddings` 决定 8K/16K/32K/128K/262K 哪几档可测。
- **benchmark top5**：必须有参考实现输出；不能手写。

---

## 3. 声明 schema 的真实缺口（必须先解决）

当前 `ModelManifest` 只接受：

```python
unknown = sorted(set(layer_types) - {"full_attention", "linear_attention"})
```

而 NemotronH 的 pattern 是 `M`（Mamba2）/`*`（Attention）/`-`（MLP），
`NemotronHConfig` 的默认 pattern 还包含 `E`（MoE）。因此：

1. **不能把 Mamba2 冒充 `linear_attention`**：`LinearSpec` 是 GDN 的
   `num_k_heads/num_v_heads/...`，Mamba2 的参数是
   `mamba_num_heads/mamba_head_dim/ssm_state_size/mamba_n_groups/...`，语义不同。
2. **不能把 MLP 层丢掉**：`layer_types` 与 `num_layers` 必须等长；否则声明校验失败或
   层数对不上。
3. **provider 契约也不够**：`LinearAttentionContract` 是 GDN 的
   `chunk/recurrent(q,k,v,g,beta)`；Mamba2 需要 conv/SSM/state 的形状，不能直接复用。

**已实施的离线改动（2026-11-12，CPU 测试通过）：**

- `layer_types` 增加 `"mamba"`、`"mlp"`；
- 新增 `MambaSpec`（`num_heads/head_dim/state_size/n_groups/conv_kernel/expand/dtype`），
  与 `LinearSpec` 并列；
- 声明校验：有 `mamba` 层必须有 `MambaSpec`，有 `MambaSpec` 必须有 `mamba` 层；
  `mlp` 层不要求 attention spec；
- 测试：`tests/test_nemotron_offline.py` 覆盖“mamba+mlp 可表示 / 缺 spec 拒绝 /
  有 spec 无层拒绝”。

**仍未解决：**

- `required_capabilities` 的 Mamba 映射：可先复用 `state.linear`，但 provider 契约需要
  单独定义，不能改 GDN 的语义；
- provider 的 `LinearAttentionContract` 仍是 GDN 的 `chunk/recurrent(q,k,v,g,beta)`，
  Mamba2 不能直接复用；这是 O4 的剩余设计项。
- 真实 config 字段仍未到位；声明草案里的 `TODO(config)` 不能填默认值。

在这些剩余项落地前，O2 **不能**产出最终 Nemotron 声明；但 schema 已能表达
Mamba/MLP 层，不再需要用假字段绕过。

---

## 4. 声明草案（占位，不可直接部署）

```json
{
  "schema_version": 1,
  "name": "Nemotron-3.5-Lightning-30B-A3B",
  "manifest_id": "nemotron-3.5-lightning-30b-a3b",
  "arch": "NemotronHForCausalLM",
  "model_type": "nemotron_h",
  "num_layers": "TODO(config.num_hidden_layers)",
  "layer_types": ["TODO(expand hybrid_override_pattern)"],
  "attention": {
    "kind": "gqa",
    "num_heads": "TODO(config.num_attention_heads)",
    "num_kv_heads": "TODO(config.num_key_value_heads)",
    "head_dim": "TODO(config.attention_head_dim)",
    "rope": "TODO(confirm)"
  },
  "mamba": {
    "kind": "mamba2",
    "num_heads": "TODO(config.mamba_num_heads)",
    "head_dim": "TODO(config.mamba_head_dim)",
    "state_size": "TODO(config.ssm_state_size)",
    "n_groups": "TODO(config.mamba_n_groups)",
    "conv_kernel": "TODO(config.mamba_d_conv)",
    "expand": "TODO(config.mamba_expand)"
  },
  "moe": {
    "num_experts": "TODO(config)",
    "top_k": "TODO(config)",
    "intermediate_size": "TODO(config)"
  },
  "quantization": "none",
  "kv": {"TODO": "mixed full-attention + Mamba state; confirm MambaSpec"},
  "required_capabilities": ["TODO(map to the 8-contract vocabulary)"],
  "benchmark": {
    "prompt": "TODO(reference prompt)",
    "max_tokens": "TODO",
    "top5": ["TODO(reference top5)"]
  },
  "source": "docs/notes/m5-nemotron-static-gap.md"
}
```

---

## 5. 完成 O2 的判据

- 用真实 `config.json` 填完 §1，无 `TODO`；
- `ModelManifest.validate()` 通过（需要先完成 §3 的 schema 扩展）；
- `required_capabilities` 全部是 8 个已知能力之一；
- benchmark top5 来自参考实现，不是手写；
- 声明里没有出现“用 vllm_ascend / 用某 provider”的字样。
