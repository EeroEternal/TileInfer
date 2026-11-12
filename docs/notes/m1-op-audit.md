# M1 算子级回退审计：厂商实现到底有没有被选中

**结论先说**：在 Qwen3.6-35B-A3B / vllm-ascend 0.23.0(A5) / vLLM 0.23.0 上，稳态解码路径**没有**
未解释的算子级 torch 回退——所有吃算力的类都是厂商实现。图级 `fallbacks=0` 不是假象。

但这不证明"一句都没漏"，只证明"这个模型厂商全覆盖"。审计本身的局限写在 §4，比结论更重要。

---

## 1. 为什么需要它

验收里 `fallbacks=0` 数的是 **CudagraphDispatcher 的 NONE**：一个 batch 没能用上捕获好的图。
它看不见另一种回退——图照样重放、但图里某个算子是纯 torch 的 `forward_native`。两种回退在日志里
都是静默的，前者我们已计数，后者此前完全没有观测点（违反 I4：没有观测点的结论不算证据）。

所以审计要回答的不是"能不能跑"，而是：

> 模型建出来的每一个算子实例，最终执行的是 `vllm_ascend.*`，还是 vLLM 的 torch 默认实现？

## 2. 机制

挂两个**构造期**入口（不是调用期），因此默认零热路径开销：

| 入口 | 覆盖 | 记录什么 |
|---|---|---|
| `CustomOp.dispatch_forward` | 39 个 `CustomOp` 类 | 实例类名、类模块、最终选中的 `forward_*` 及其定义模块 |
| `PluggableLayer.__new__` | 20 个 `PluggableLayer` 类 | 实际实例化的类名与模块 |

`impl` 的判定只看**真正会执行的 forward 定义在哪个包**：`vllm_ascend.*` → `vendor`，否则
`torch`（`CustomOp`，会落到 `forward_native`）或 `vllm`（`PluggableLayer`，整类没有被替换）。

判定依据是 `vLLM 0.23 CustomOp.__new__` 按**类名**查 `op_registry_oot`；查不到就静默用
`forward_native`，不打日志。这是审计能成立的前提。

开关是环境变量，默认关：

```bash
TILEINFER_AUDIT=/tmp/tileinfer-ops.json      # 或 =1 用默认路径
```

数据在 `SIGUSR1` 和进程退出时落盘（`tileinfer/observ/opaudit.py`），与图计数器同一套习惯：
建了算子的进程才写，API server 进程写空文件会被跳过。

实现位置：`tileinfer/bridge/seams.py:install_op_audit`（复用已有的 `method_wrap` 机制，
**不新增接入点**）。

## 3. 结果

跑法（2026-11-12 16:22，单卡 Ascend950PR，端口 8322，cap 32）：

```bash
TILEINFER_AUDIT=/tmp/tileinfer-ops.json bash /tmp/run_v3_audit.sh
curl -s -X POST localhost:8322/v1/completions -d '{"model":"qwen36","prompt":"Give me a short introduction to large language models.","max_tokens":32,"temperature":0}'
kill -USR1 <EngineCore pid>
```

证据：`/tmp/tileinfer-ops.json`（23 类 / 903 实例 / 723 厂商）。

| 类 | impl | 实例 | calls | 读法 |
|---|---|---:|---:|---|
| `AscendRowParallelLinear` | vendor | 135 | – | 稠密主体 |
| `AscendGemmaRMSNorm` | vendor | 101 | 81 | 层归一化 |
| `AscendMergedColumnParallelLinear` | vendor | 100 | – | gate/up 融合 |
| `AscendReplicatedLinear` | vendor | 80 | – | router 等 |
| `AscendColumnParallelLinear` | vendor | 58 | – | qkv/下投影 |
| `AscendFusedMoE` + `AscendMoERunner` | vendor | 40+40 | – | 专家层 |
| `AscendSiluAndMul` | vendor | 40 | 680 | 激活，最热 |
| `AscendQKVParallelLinear` | vendor | 37 | – | qkv |
| `AscendRMSNormGated` | vendor | 30 | 30 | GDN 的 gated norm |
| `AscendApplyRotaryEmb` | vendor | 29 | 27 | rope |
| `AscendMMEncoderAttention` | vendor | 27 | 27 | 视觉塔（文本不用） |
| `AscendVocabParallelEmbedding` / `AscendParallelLMHead` / `AscendLogitsProcessor` | vendor | 1+1+1 | 0 | 首尾 |
| `AscendMRotaryEmbedding` / `AscendRotaryEmbedding` / `AscendConv3dLayer` | vendor | 1+1+1 | 0/0/1 | 单例 |
| **`QwenGatedDeltaNetAttention`** | **vllm** | **30** | – | 见下 ① |
| **`ChunkGatedDeltaRule`** | **torch** | **30** | **0** | 见下 ② |
| **`AscendUnquantizedFusedMoEMethod`** | **torch** | **40** | **0** | 见下 ③ |
| `UnquantizedFusedMoEMethod` / `MoERunner` | torch / vllm | 40+40 | 0 | 基类记账 |

三个非 vendor 行的解释（都不是回退）：

1. **GDN：类没换，方法换了。** 厂商在
   `vllm_ascend/patch/worker/patch_qwen3_5.py:197-209` 直接把
   `QwenGatedDeltaNetAttention.forward / _forward_core` 替换成 `AscendGatedDeltaNetAttention`
   的实现。类名仍是 vLLM 的，所以类级审计只能看到 `vllm`——**这是本审计看不见方法级 monkey patch
   的直接证据**，也是最值得记住的一条局限。
2. **chunk rule：建了但没跑。** `ChunkGatedDeltaRule` 30 个实例，本次请求 0 次调用。它是
   prefill 的分块路径；这条 8-token prompt + 32 步解码没走到。
3. **MoE method：vLLM 的 `forward_native` 是壳。** `UnquantizedFusedMoEMethod.forward_native`
   只做 `self.moe_kernel.apply(...)`，而 `apply` 正是 Ascend 子类重写的方法；且该对象 0 次调用
   （MoE 走 `apply` 而非 `__call__`）。`suspicious` 标记在这里是误报，判定规则已写进
   `opaudit.py` 的 docstring。

## 4. 这个审计**不能**证明什么

1. **看不到方法级 patch。** 见 ①。类级全绿不代表没有 patch 替换。
2. **图模式下 calls 严重欠采样。** 32 步解码、40 层，`AscendGemmaRMSNorm` 只记到 81 次
   （应为 ~1300 量级）。原因是捕获后 replay 不经过 Python。所以 `calls` 列只对 prefill / eager
   有意义，**不能**用来判断 decode 每步调了几次。
3. **PluggableLayer 没有 calls。** linear / MoE / GDN / embedding 的 `forward` 是各自类定义的，
   没被统一包；它们的"频率"目前只能用实例数近似（每 token 每实例一次）。
4. **只覆盖一条短请求。** 长 prefill（chunk rule 真正工作的区间）没跑。

要把"decode 每个 op 都是厂商 kernel"说死，得用 device 侧 profiler（`torch_npu.profiler` /
msprof）看实际下发的 kernel，而不是 Python 计数器。这是下一步，不是本轮结论。

## 5. 对"什么时候补"的意义

这次审计的产出不是一个新 provider，而是一条**可复现的判据**：

- 实例表里出现 `impl=torch` 且 `calls` 不为 0，且该类在模型热路径上 → 才进入"要不要补"的讨论；
- 进入讨论后先问"能不能让厂商实现被选中"（注册名、形状、开关、配置），因为厂商的 Ascend kernel
  我们大概率打不过；
- 只有"厂商确实没有 / 有但覆盖不了 / 有但数值不对 / 有但上不了图"这四种缺口，才写 provider。

对照 `docs/acceptance.md` §2.1：Qwen3.6 属于"厂商能正确部署"，因此它的缺口表本来就是空的，
审计全绿是预期结果，不是成绩。
