# 模型 bring-up：vLLM 不支持时，到底要补什么

> 一句话：**算子/注意力只是 compute contract。** 一个模型要“可部署”，还要通过
> 注册、权重、显存/KV、并行、量化/投机、图捕获、输入/服务、验收这些契约。
> 对标准 dense AR 模型，算子可能只占小头；对混合/扩散模型，引擎改动才是大头。

---

## 0. 先分清四种“不支持”

| 情况 | 例子 | 缺的是什么 | 量级 |
|---|---|---|---|
| vLLM main 有，release pin 没有 | Flint | 把 model 文件/注册/权重加载 backport 到 0.30，适配 API | 天～周 |
| vLLM 有，vllm-ascend 没有 | K2-Horizon、Nemotron 3.5、LFM2.5 | Ascend 侧 model 注册 + 自定义 attention/SSM/conv 算子 + KV/graph | 周～月 |
| vLLM 没有，但仍是标准 AR | Raon-1 mini | 从 HF 实现移植成 vLLM 模型：图、权重、KV、并行、tokenizer | 周～月 |
| vLLM 没有，且不是 AR | Clover（block diffusion） | 引擎级：scheduler、mask、迭代去噪、cache、serving | 月级，通常超出 provider 范围 |

“vllm-ascend 追平 vLLM 就支持了”只对第一种接近成立；后三种都不是自动的。

---

## 1. 分层清单

| 层 | 要补什么 | vLLM 的通用件 | 模型特有 |
|---|---|---|---|
| 注册/配置 | `architectures` → 模型类；config 字段解析 | `ModelRegistry`、HF config | 新 `model_type`、config 字段、`hf_config_override` |
| 模型图 | embedding、layer 顺序、norm、RoPE、residual、logits | `Attention`、`FusedMoE`、`RMSNorm`、`RotaryEmbedding`、`LogitsProcessor` | 混合层序列、custom attention、gating、activation |
| 权重加载 | HF 参数名 → vLLM 参数名；QKV/gate_up 融合；专家堆叠；量化 scale | `WeightsMapper`、`AutoWeightsLoader`、`maybe_prefix` | `load_weights()`、TP/EP 切分、missing/unexpected 处理 |
| KV/显存 | KV cache 规格、block size、sliding window、state cache | `KVCacheSpec`、`MambaSpec`、`FullAttentionSpec` | attention 类型、conv/SSM state、MLA、cross-attention |
| 并行 | TP/PP/EP/DP 的切分与通信 | `ColumnParallelLinear`、`RowParallelLinear`、`VocabParallelEmbedding`、`FusedMoE` | PP missing layers、专家映射、all-to-all |
| 量化 | quant config、scale、kernel | `QuantizationConfig`、FP8/INT4/AWQ/GPTQ | 模型特有 quant 布局、Ascend 量化方法 |
| 投机/MTP | draft model、MTP heads、rejection sampler | `SupportsMTP`、`MTPModelTypes`、Eagle/DFlash | 模型自己的 MTP 层、`num_nextn_predict_layers` |
| 图/编译 | 自定义 op 必须 graph-safe；动态 shape | `support_torch_compile`、CudagraphDispatcher | op 分解、fallback、动态控制流 |
| 输入/服务 | tokenizer、chat template、stop token、tool/reasoning parser | OpenAI server、通用 tokenizer | 特殊 token、template、parser、多模态 processor |
| Ascend 后端 | attention/MoE/quant/SSM kernel、HCCL、内存 | `vllm_ascend/ops`、`attention`、`patch` | 新 attention backend、算子、KV layout |
| 验收 | logits parity、长上下文、图、投机、并发、冷启动、性能 | 测试框架 | 矩阵证据、基线比值 |

---

## 2. 具体例子

### Flint：vLLM main 有，0.30 没有

- vLLM main 注册了 `FlintForCausalLM`（`vllm/models/flint.py`），0.30 没有。
- 要做的是 **backport**：把 model 文件、registry、权重加载、依赖的 API 适配到 0.30；
  如果 Ascend 通用 attention/MoE 能覆盖，就不需要新数学。
- 但仍然要跑：权重加载、logits parity、KV、图、长上下文、并发。
- 结论：**版本窗口型缺口**，不是“从零实现模型”；厂商追平后可能自然消失。

### K2-Horizon：vLLM 有，Ascend 没有

- vLLM 0.30 有 `K2HorizonForCausalLM`；MoVA 用 Triton `fused_mova_impl`，
  还直接调 `vllm._custom_ops`（`moe_sum`、`rms_norm`）。
- Ascend 要补：
  1. `K2HorizonForCausalLM` 的注册/加载；
  2. MoVA attention 的 Ascend 实现（Triton-Ascend 或 AscendC）；
  3. `moe_sum`/`rms_norm` 等自定义 op 的 Ascend 等价；
  4. MoVA 的 KV cache 规格、block table、mask；
  5. 图捕获（FULL/PIECEWISE）与回退计数；
  6. 权重映射、TP/EP、tokenizer/template；
  7. 正确性/长上下文/并发/性能验收。
- 结论：不是“只写一个 kernel”，但也不至于改引擎；是**模型 + 算子 + KV + 图**的组合。

### Raon-1 mini：vLLM 没有，标准 AR

- 要从 HF `transformers` 实现移植成 vLLM 模型：
  config/model 类、layer 图、`load_weights`、KV spec、TP/PP/EP、tokenizer/chat template。
- 如果层都是标准 attention/MoE，可以大量复用 vLLM 层；但“复用”不等于“自动”：
  名字映射、融合 QKV/gate_up、专家切分、RoPE、norm 顺序都要自己对齐。
- 然后再做 Ascend 后端和验收。
- 结论：**完整 model bring-up**，比“给 vLLM 已支持的模型补 Ascend kernel”多一层。

### Clover：vLLM 没有，且不是 AR

- block diffusion：每个 block 从 mask 开始，迭代去噪，双向/块内注意力；
  vLLM 的 scheduler 假设“一次 forward 出一个 token”，KV cache 也是按自回归增长。
- 要补的不只是模型：scheduler、mask、迭代循环、cache 语义、stop 条件、serving 协议。
- 结论：**引擎级工作**，不应该塞进 provider；M5 不建议选。

---

## 3. 为什么“追平”不是免费

1. vllm-ascend 钉住一个 vLLM 版本；要等包含该模型的 release，或自己 rebase。
2. vLLM 模型可能用 CUDA/Triton/custom ops；Ascend 要重写或提供等价实现。
3. 即使模型是标准层，Ascend 也要过 attention backend、KV layout、graph capture、
   精度/性能验收；“代码合了”不等于“支持”。
4. 支持表是**验证过**的集合，不是 registry 有名字就算。

---

## 4. 对 M5 的含义

- 如果 M5 想验证“provider 机制”，优先选 **vLLM 已支持、Ascend 缺一个边界清晰的
  attention/算子** 的模型（K2-Horizon 是典型）；这样工作是可封装的。
- 如果 vLLM 完全没有该模型，那是 **model bring-up**，会新增模型文件/加载器/配置，
  很可能超出“兼容层 + provider”的边界，也会吃掉接入点预算。
- Clover 这类非 AR 模型是 engine bring-up，不建议作为 M5。
- Flint 是版本窗口，不是 durable gap；厂商追平后可能白做。
- 选型前先问三个问题：
  1. vLLM 有没有该 architecture？
  2. 缺的是“注册/权重”还是“新算子/新执行范式”？
  3. 厂商是否已经在做（PR/main）？

---

## 5. 最小 bring-up 顺序

1. config + registry，能 `AutoConfig` 解析。
2. eager BF16 模型图，能 forward。
3. `load_weights`，与 HF logits 对齐。
4. KV cache spec + 单请求生成。
5. TP/PP/EP，分布式一致性。
6. Ascend attention/MoE/quant kernel + 图捕获。
7. tokenizer/chat template/stop/tool parser。
8. 投机/MTP/量化（如果模型有）。
9. 跑 `docs/verification-matrix.md` 全表，写证据。

---

## 6. 对应代码位置

- vLLM registry：`vllm/model_executor/models/registry.py`
- 模型接口/能力：`vllm/model_executor/models/interfaces.py`
- 权重映射：`vllm/model_executor/models/utils.py`
- KV 规格：`vllm/v1/kv_cache_interface.py`
- attention：`vllm/model_executor/layers/attention/`
- MoE：`vllm/model_executor/layers/fused_moe/`
- 投机：`vllm/config/speculative.py`
- 编译：`vllm/compilation/decorators.py`
- Ascend 侧：`vllm_ascend/ops/`、`vllm_ascend/attention/`、`vllm_ascend/patch/`、
  `vllm_ascend/models/`、`vllm_ascend/platform.py`
