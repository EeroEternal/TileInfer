# 2026 年国外 5–40B 开放权重模型：vllm-ascend 支持性调研

> 调研时间：2026-11-12。范围：Hugging Face 上 2026-01-01 至 2026-11-12 创建、
> `text-generation`、总参数 5B–40B、非中国团队、开放权重。
> 热度用 HF `downloads`/`likes` 排序，数字会随时间变化。
>
> **这是静态调研，没有上卡实测。** "不支持"分两级：
> **A 硬阻塞**（vLLM 0.30 不注册该架构，或 Ascend 缺关键算子/明确未完成的支持项）；
> **B 未适配**（vLLM 已支持，vllm-ascend 没有模型级代码/验证，可能走通用路径，需 smoke test）。
> vllm-ascend 支持表自己也写明"Models not listed may be verified independently"，
> 所以 B 类不能直接断言"跑不了"。

---

## 0. 结论

**A 类（优先看）：**

1. `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B` — 31.6B，1.04M downloads，Mamba 混合架构；
   Ascend 的 Nemotron/Mamba 支持还是 open PR。
2. `IFM/K2-Horizon-MoVA-36B-A4B` — 37.4B，Apache-2.0，MoVA 自定义注意力；
   vLLM 有实现，但依赖 `_custom_ops` + Triton，Ascend 没有对应模型代码。
3. `LiquidAI/LFM2.5-24B-A2B` — 24B，ShortConv 混合架构；Ascend 的 ShortConv PR 已关闭，
   明确写"Liquid d1 support is not yet verified"，且图捕获未建立。
4. `RecursiveSI/flint` — 39.8B，coding agent 模型；vLLM **main** 才有 `FlintForCausalLM`，
   vLLM 0.30（vllm-ascend 0.30 的配对版本）没有；Ascend 无。
5. `Diffuspace/Clover-35B-A3B-Preview` — 34.7B，block diffusion，不是自回归；
   vLLM registry 没有 `CloverMoeModelLM`，只有独立 PyTorch/SGLang 路径。
6. `KRAFTON/Raon-1-mini-preview` — 28.8B，自定义 `Raon1MiniForCausalLM`；
   vLLM 0.30/main 都没有，Ascend 无。

**B 类（vLLM 支持，Ascend 未适配/未验证）：**

`poolside/Laguna-XS.2`、`CohereLabs/North-Mini-Code-1.0`、
`ibm-granite/granite-4.2-30b`、`sarvamai/sarvam-30b`、
`JetBrains/Mellum2-12B-A2.5B-Thinking`、`swiss-ai/Apertus-v1.5-8B`。

**对 M5 的含义：** 如果要找"厂商短期不会补、又足够热门"的真缺口，
**K2-Horizon-MoVA** 最合适；`Raon` 可作备选；`Clover` 是 diffusion，属于引擎级工作。
`Nemotron`、`Flint`、`LFM2` 厂商已经在跟进（PR 或 vLLM main），现在做容易变成版本滞后，
不满足"不要重复厂商已覆盖能力"的红线。

---

## 1. 口径

- 参数：`safetensors.total`；MoE 用总参数，不是激活参数。
- 热度：HF downloads / likes（2026-11-12 抓取），只作排序，不代表生产采用。
- "国外"：排除 Qwen、DeepSeek、GLM/Z.ai、MiniMax、Moonshot、StepFun、OpenBMB、
  Tencent、ByteDance、Xiaomi、BAAI、InternLM、Baichuan 等中国团队；
  保留 NVIDIA、poolside、Recursive、Liquid AI、IFM、Cohere、IBM、DiffuSpace、
  KRAFTON、JetBrains、Swiss AI、Sarvam 等。
- 排除微调/量化/合并/GGUF 变体，只保留原模型或 instruct 版本。
- 支持判断：
  1. vLLM 0.30.0 的 `registry.py` 是否有该 architecture；
  2. vllm-ascend 0.30.0 wheel 内是否有对应 model/patch/op；
  3. vllm-ascend 支持表、issue、PR。
- 本机 950 只有 `0.23.0-950` wheel；PyPI 的 0.30.0 是 A2 build。
  下表的 0.30 判断是**上游代码判断**，不代表本机可以直接跑 0.30。

---

## 2. A 类：硬阻塞

| 模型 | 总参数 | 发布 | 热度 | 许可 | 为什么现在不能部署 |
|---|---:|---|---:|---|---|
| [NVIDIA-Nemotron-3.5-Lightning-30B-A3B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) | 31.6B | 2026-08-01 | 1,043,242 dl / 272 likes | OpenMDW-1.1 | vLLM 0.30 有 `NemotronHForCausalLM`；模型是 Mamba2 + attention + MoE。vllm-ascend 0.30 没有 `nemotron_h` 模型代码；Nemotron 3 Super 的 Mamba SSD 支持仍是 open PR [#11656](https://github.com/vllm-project/vllm-ascend/pull/11656)。 |
| [K2-Horizon-MoVA-36B-A4B](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B) | 37.4B | 2026-09-01 | 14,924 dl / 433 likes | Apache-2.0 | vLLM 0.30 有 `K2HorizonForCausalLM`；实现直接 `from vllm import _custom_ops as ops`，MoVA 用 Triton `fused_mova_impl`。vllm-ascend 0.30 没有 `k2_horizon` 代码，也没有 Ascend 版 MoVA 算子。同架构的 [Jais-3-MoVA-36B-A4B](https://huggingface.co/IFM/Jais-3-MoVA-36B-A4B)（阿联酋，37.4B）同理。 |
| [LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) | 24.0B | 2026-10-12 | 37,945 dl / 607 likes | LFM Open License 1.0 | vLLM 0.30 有 `Lfm2MoeForCausalLM`，但大量层是 `ShortConv`（conv state）。vllm-ascend 的 ShortConv PR [#21682](https://github.com/vllm-project/vllm-ascend/pull/21682) 已关闭，正文写明"Liquid d1 support is not yet verified"，且现有 torch fallback "not established to support graph capture"。 |
| [Recursive Flint](https://huggingface.co/RecursiveSI/flint) | 39.8B | 2026-10-21 | 49,593 dl / 1,390 likes | Apache-2.0 | vLLM 0.30 registry **没有** `FlintForCausalLM`；vLLM main 才注册到 `vllm.models.flint`。vllm-ascend 0.30 与 vLLM 0.30 配对，因此现在不能部署。模型卡主要给 SGLang 路径。 |
| [Clover-35B-A3B-Preview](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview) | 34.7B | 2026-10-24 | 4,841 dl / 270 likes | Apache-2.0 | block diffusion LM，不是自回归；vLLM 0.30/main 都没有 `CloverMoeModelLM`。模型卡给的是独立 PyTorch 实现和 SGLang backend；上 vllm-ascend 不是"加个 provider"的量级。 |
| [Raon-1 mini preview](https://huggingface.co/KRAFTON/Raon-1-mini-preview) | 28.8B | 2026-10-21 | 4,376 dl / 240 likes | Apache-2.0（需申请） | 自定义 `Raon1MiniForCausalLM`，vLLM 0.30/main 都没有该 architecture；vllm-ascend 无。韩国 KRAFTON，训练语料以韩语/英语为主。 |

### 补充说明

- **Nemotron** 的 PR #11656 是 120B Super 的 BF16 支持，不是 30B Lightning；
  但二者共用 Nemotron-H 的 Mamba/attention/MoE 路径，PR 未合并前 30B 也没有验证过的 Ascend 路径。
- **Flint** 是"版本窗口"型缺口：vLLM main 已支持，等 vllm-ascend 追平 main 后可能自然消失。
  如果目标是"长期缺口"，它不是好选择；如果目标是"现在就要跑"，它是。
- **LFM2.5** 的 PR 已关闭，不代表厂商承诺支持；但说明厂商已经看过，后续可能重开。
- **Clover** 需要 diffusion 采样/块解码，和 vLLM 的自回归 scheduler 不是同一类问题；
  选它做 M5 会把"补 provider"变成"改引擎"。

---

## 3. B 类：vLLM 支持，Ascend 未适配/未验证

这些模型的架构在 vLLM 0.30 里有实现，vllm-ascend 0.30 没有模型级代码；
其中标准 Attention + MoE 的模型**有可能**走通用路径跑起来，但没有 Ascend 验证，
不能写成"支持"。

| 模型 | 总参数 | 发布 | 热度 | 许可 | 现状 |
|---|---:|---|---:|---|---|
| [Laguna-XS.2](https://huggingface.co/poolside/Laguna-XS.2) | 33.4B | 2026-04-23 | 808,540 dl / 320 likes | Apache-2.0 | vLLM 有 `LagunaForCausalLM`；Ascend 无 `laguna` 代码。结构是标准 Attention + MoE，理论上可能通用，但无验证。 |
| [North-Mini-Code-1.0](https://huggingface.co/CohereLabs/North-Mini-Code-1.0) | 30.5B | 2026-06-05 | 8,238 dl / 588 likes | Apache-2.0 | vLLM 有 `Cohere2MoeForCausalLM`；Ascend 无 `cohere2_moe` 代码。标准 Cohere2 MoE + sliding attention。 |
| [Granite 4.2 30B](https://huggingface.co/ibm-granite/granite-4.2-30b) | 29.3B | 2026-08-07 | 36,738 dl / 142 likes | Apache-2.0 | vLLM 有 `GraniteForCausalLM`；Ascend 无 granite 代码。标准 dense transformer，可能通用。 |
| [Sarvam-30B](https://huggingface.co/sarvamai/sarvam-30b) | 32.2B | 2026-03-03 | 3,840 dl / 235 likes | Apache-2.0 | vLLM 有 `SarvamMoEForCausalLM`（MLA + MoE）；Ascend 有 DeepSeek MLA 路径，但没有 Sarvam 模型级适配。 |
| [Mellum2-12B-A2.5B-Thinking](https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Thinking) | 12.1B | 2026-05-26 | 3,611 dl / 349 likes | Apache-2.0 | vLLM 有 `MellumForCausalLM`，复用 Qwen3MoE 层；Ascend 无 mellum 代码。标准结构，可能通用。 |
| [Apertus-v1.5-8B](https://huggingface.co/swiss-ai/Apertus-v1.5-8B) | 8.9B | 2026-07-24 | 12,646 dl / 187 likes | Apache-2.0（gated） | vLLM 有 `ApertusForCausalLM`；关键自定义激活 `XIELU` 是 `CustomOp`，有 Python fallback，但 Ascend 无 XIELU 实现/验证。 |

另外，`ibm-granite/granite-switch-4.1-30b`（30B，`granite_switch`）在 vLLM 0.30/main
都没有注册，更接近 A 类；但热度较低（4.8k dl / 46 likes），先放观察。

---

## 4. 为什么没有 Llama / Mistral / Gemma 4

- **Meta**：HF `meta-llama` 2026 年没有新建 5–40B 开放模型（检索结果为 0 条）。
- **Mistral**：2026 年的 Small-4 是 119B、Medium-3.5 是 128B、Large-4 是 1T，全部超过 40B。
- **Google Gemma 4**：12B / 26B-A4B / 31B 都在范围内，但 vllm-ascend 0.30 已把
  `Gemma4` 标为 🔵 Experimental 支持（A2/A3/950PR&950DT），所以不是缺口。
- **Apple / Microsoft** 的 2026 30B 级模型：`apple/SimpleSD-30B-instruct` 是
  `Qwen3MoeForCausalLM`，`microsoft/AesCode-32B` 是 `Qwen3VLForConditionalGeneration`，
  属于已支持架构的微调，不是新架构缺口。

---

## 5. 对 M5 选型的建议

> 如果 vLLM 本身没有该模型，"支持"就不只是算子/注意力，而是完整 model bring-up；
> 分层清单见 [`model-bringup-layers.md`](model-bringup-layers.md)。

1. **首选 K2-Horizon-MoVA-36B-A4B。**
   - 37.4B、Apache-2.0、非中国团队、有 HF/vLLM 生态；
   - 缺口明确：MoVA attention + MoE 自定义算子，Ascend 没有；
   - 厂商没有公开的 Ascend 支持 PR，短期不会变成"版本滞后"；
   - 代价：不是纯 provider，可能要补 Triton/自定义算子，正好检验"填缺口"能力。

2. **备选 Raon-1 mini 28.8B。**
   - 自定义架构、无厂商支持，缺口更"干净"；
   - 但热度低、需申请权重、韩语为主，作为 second model 的生态价值弱。

3. **Clover 只在接受"diffusion 引擎级工作"时选。**
   - 否则它会超出"provider"范围，变成改 scheduler/采样。

4. **Nemotron / Flint / LFM2 不建议作为"长期缺口"选题。**
   - Nemotron：Ascend PR 已开；
   - Flint：vLLM main 已支持；
   - LFM2：Ascend PR 已看 ShortConv。
   拿它们做 M5，等实现完厂商可能已经支持，正好踩"重复厂商工作"的红线。

5. **B 类先做 1 天 smoke test，再决定是否写 provider。**
   标准 Attention + MoE 的模型可能只差一个注册/验证，不需要新算子；
   在没有实测前，不要把它们写成"厂商不支持"。

---

## 6. 证据

- vLLM 0.30 registry：
  <https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/models/registry.py>
- vLLM main registry（Flint 在这里）：
  <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py>
- vllm-ascend 0.30 支持表：
  <https://docs.vllm.ai/projects/ascend/en/v0.30.0/user_guide/support_matrix/supported_models.html>
- Nemotron Ascend PR #11656：
  <https://github.com/vllm-project/vllm-ascend/pull/11656>
- LFM2 ShortConv Ascend PR #21682：
  <https://github.com/vllm-project/vllm-ascend/pull/21682>
- K2-Horizon vLLM 实现：
  <https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/models/k2_horizon.py>
- vllm-ascend 0.30 release：
  <https://github.com/vllm-project/vllm-ascend/releases/tag/v0.30.0>
