# Nemotron 模型声明草案（O2，离线）

> 状态：**真实 config 已到手，声明结构已按 config 落成 draft；但 benchmark top5 未填、
> MTP 权重未验，所以不注册进 `MANIFESTS`、不作为可部署声明。**
>
> 事实来源：[`m5-nemotron-config.json`](m5-nemotron-config.json)（仅 config 快照，未下载权重）。
> 代码草案：[`tileinfer/manifests/nemotron_3_5.py`](../../tileinfer/manifests/nemotron_3_5.py)。

---

## 1. 真实 config 事实

| 字段 | 值 |
|---|---|
| `architectures` | `["NemotronHForCausalLM"]` |
| `model_type` | `nemotron_h` |
| `num_hidden_layers` | 52 |
| 层型 | 23 `mamba` + 23 `moe` + 6 `attention` |
| `max_position_embeddings` | 262144 |
| attention | 32 heads / 2 KV heads / head_dim 128；RoPE theta 10000；partial rotary 1.0 |
| Mamba2 | 64 heads / head_dim 64 / state 128 / groups 8 / conv 4 / expand 2 |
| MoE | 128 routed experts / top-6 / intermediate 1856 / 1 shared expert |
| `num_nextn_predict_layers` | 1 |
| MTP 层型 | `["attention","moe"]` |
| dtype / 量化 | bfloat16 / `none` |

**不是** vLLM `NemotronHConfig` 的默认值：默认 pattern 是 24 M / 24 MLP / 4 attention、
无 MoE，与目标 23/23/6 完全不同。不能拿默认值填声明。

---

## 2. config schema 兼容（必须先做）

目标 config 使用 `layers_block_type` / `mtp_layers_block_type`；vLLM 0.23 的
`NemotronHConfig` 只有只读属性 `layers_block_type`，期望 `hybrid_override_pattern`。
直接加载会：

```text
AttributeError: property 'layers_block_type' of 'NemotronHConfig' object has no setter
```

即使绕过，也会用默认 pattern，把 23 个 MoE 层建错。

[`config_to_manifest.py`](../../tools/nemotron/config_to_manifest.py) 的
`normalize_config()` 做机械转换：

```text
layers_block_type      -> hybrid_override_pattern   (mamba=M, attention=*, mlp=-, moe=E)
mtp_layers_block_type   -> mtp_hybrid_override_pattern
n_groups/conv_kernel/expand/chunk_size -> mamba_n_groups/mamba_d_conv/...
```

转换后 `NemotronHConfig` 可加载，层型计数与 MTP pattern 均与目标一致（有 CPU 测试）。
这是部署 shim，不是改上游源码；上游接受新 schema 后可删除。

---

## 3. 声明字段（draft）

| 字段 | 值 |
|---|---|
| `name` | `Nemotron-3.5-Lightning-30B-A3B` |
| `manifest_id` | `nemotron_3_5_lightning` |
| `arch` | `NemotronHForCausalLM` |
| `model_type` | `nemotron_h` |
| `num_layers` | 52 |
| `layer_types` | 23 `mamba` + 23 `moe` + 6 `full_attention` |
| `attention` | `gqa`, 32/2 heads, head_dim 128, rope |
| `mamba` | `MambaSpec(mamba2, 64 heads, 64 dim, state 128, groups 8, conv 4, expand 2)` |
| `moe` | 128 experts, top_k 6, intermediate 1856, shared 1 |
| `quantization` | `none` |
| `kv` | hybrid；6 full attention / 23 Mamba / 23 MoE；上下文 262144；MTP 1 层 |
| `required_capabilities` | 7 项：`attention.full`, `moe.routed`, `norm.rms`, `rotary`, `sampling.topk`, `state.linear`, `quant.verify` |
| `benchmark` | prompt/max_tokens 已定，`top5=[]`，`pending` |

**为什么没有 `attention.linear`**：`LinearAttentionContract` 是 GDN 的
`q/k/v/g/beta`，Mamba2 不是 GDN。Mamba2 走 `state.linear` + `Mamba2Contract`
子契约；不把 Mamba2 冒充 `attention.linear`。

---

## 4. 还不能做的事

- **不注册**：`BENCHMARK_PENDING=True`，`top5=[]`；注册会让 `find_manifest` 把
  draft 当成可部署声明。
- **不承诺 MTP**：config 有 1 层 MTP，但权重 key 未查；没有 MTP heads 就按 D4 上报，
  不用 ngram 替代。
- **不承诺 262K 可用**：config 支持 262144，但实际 HBM/图/性能要上机；验收先跑
  8K/16K/32K/128K，262K 作为条件档。
- **不写 provider 默认路径**：Mamba2 的 SSD/chunk-scan 仍是候选缺口，见
  [`m5-nemotron-provider.md`](m5-nemotron-provider.md)。

---

## 5. 完成 O2 的判据

- benchmark top5 由 HF/transformers 参考实现填上，`BENCHMARK_PENDING=False`；
- MTP 权重 key 用 [`mtp_probe.py`](../../tools/nemotron/mtp_probe.py) 查清，有/无都记录；
- 声明通过 `ModelManifest.validate()` 并注册进 `MANIFESTS`；
- 声明里没有出现“用 vllm_ascend / 用某 provider”的字样；
- config shim 的 `delete_when` 已记录：上游接受 `layers_block_type` 或升级后删除。
