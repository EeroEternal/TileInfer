# LFM3-1.7B 在 TileInfer 上的适配报告

> 本文档包含三部分：模型与适配说明、验证与性能结果、效果评估。
> 代码与原始日志在 `xjx1-dev` 工作区；技术细节证据见
> [`docs/notes/lfm3-adaptation.md`](notes/lfm3-adaptation.md)。

## 0. 摘要

选择 `LiquidAI/LFM3-1.7B` 作为“vllm-ascend 完全跑不了”的路线 O 目标，完成了从模型声明、契约扩展、短卷积 provider、vLLM 模型实现到 bridge 接入的整条链路。

最终结果：C1 能跑、C3 能上图、C4 在 conc=1 对 HF 参考达标；C2 为容差内通过（非逐位）。模型以真实 `vllm.general_plugins` 入口加载，`tileinfer doctor` 全 PASS。

## 1. 模型介绍

### 1.1 基本信息

| 项 | 值 |
|---|---|
| 模型 | `LiquidAI/LFM3-1.7B`（Liquid AI，LFM3 系列） |
| 架构 | `Lfm3ForCausalLM` / `model_type=lfm3` |
| 参数量 | 约 17 亿（bf16 权重 3.46 GB） |
| 层数 | 24 |
| 隐藏维度 | 2048 |
| MLP 维度 | 7680（SwiGLU） |
| 注意力 | 16 Q / 4 KV，head_dim 128（GQA，4 组） |
| 词表 | 128000 |
| 上下文 | 最大位置 262144，滑动窗口 256 |
| 权重位置 | `/home/models/LFM3-1.7B`（ModelScope 下载，25 个文件） |

### 1.2 结构特点

LFM3 不是普通 dense Transformer，而是 **卷积 + 注意力混合** 结构。24 层按固定模式排列：

```text
层号:  0   1   2    3   4      5   6    7   8      9  10   11 12      ...
类型: conv conv full conv sliding conv full conv sliding conv full conv sliding ...
```

统计：

| 类型 | 数量 | 说明 |
|---|---|---|
| `conv` | 13 | 带门控的 depthwise 因果短卷积，kernel=7，状态长度 6 |
| `full_attention` | 6 | 全注意力，GQA 16/4 |
| `sliding_attention` | 5 | 窗口 256 的滑窗注意力 |

每个注意力层还有：

- QK RMSNorm（对每个 head 的 128 维做归一化）；
- **per-head sigmoid 输出门**：`attn_output * sigmoid(gate_proj(x))`；
- 两套 RoPE：
  - full 层：θ=1e6，partial rotary 0.5，采用 **零尾（zero-tail）** 方式；
  - sliding 层：θ=1e4，全维度旋转。
- sandwich norm：attention/conv 后和 FFN 后各有一个 RMSNorm；
- 绑定词嵌入（`tie_word_embeddings=true`）。

一句话：它用短卷积承担大部分 token mixing，用少量全注意力/滑窗注意力补长程信息，目标是长上下文下比纯注意力更省算力。

## 2. 为什么需要适配（特殊点）

| # | 特殊点 | 为什么通用路径不行 |
|---|---|---|
| 1 | vLLM 0.23 没有 `Lfm3ForCausalLM` | 原生注册表无该类；只能走路线 O 自己实现 |
| 2 | transformers backend 回退也不通 | 回退能加载权重，但 `modeling_lfm3.py` 的 attention 假设 batch 维，vLLM 传扁平 token 布局，在 `attn_output * gate` 处形状崩溃 |
| 3 | 短卷积状态不是 GDN | 厂商的 `state.linear` 实现是 GDN；LFM3 需要 kernel=7 的 depthwise 因果卷积 + 6 槽滚动状态 |
| 4 | full/sliding 混合注意力 | 每层窗长不同，不能全用 full，也不能全用滑窗 |
| 5 | 双 RoPE、零尾 partial | vLLM 的 partial rotary 是“缩短旋转维度 + NeoX 配对”；LFM3 是“保留完整 128 维 rotate_half，只把前 64 维的频率保留、尾部频率置零”，两者数值不等价 |
| 6 | QK RMSNorm + per-head gate + sandwich norm | 不是标准 Llama 结构，必须逐项复现 |
| 7 | Ascend 页大小对齐 | vllm-ascend 断言 SSM 页大小必须等于 `128 × head_dim × kv_heads × dtype_size`；短卷积状态太小，无法与 128 token 的 attention 页对齐 |
| 8 | FIA 要求 value 连续 | vLLM 的 qkv split 产生非连续 view，Ascend FIA 直接报 “value tensors must be contiguous” |
| 9 | transformers 5.16/5.17 的配置兼容性 | 新版会把 `conv` 重映射成 `linear_attention`，并对嵌套 `rope_parameters` 报 `Unrecognized keys`；5.12.1 又缺少模型代码使用的 `use_kernel_func_from_hub_with_fallback` |
| 10 | 图捕获与动态状态索引 | 最初逐请求 `int(state_indices[i])` 在 capture 时触发 host 同步，图捕获失败 |

## 3. 改了哪些地方、怎么改、效果

### 3.1 总览

| 层/文件 | 改动 | 为什么 | 效果 |
|---|---|---|---|
| `tileinfer/manifests/lfm3_1_7.py` + registry | 新增模型声明：24 层类型、16/4 头、head_dim 128、窗口 256、双 RoPE、短卷积 kernel 7、benchmark、perf、模块式 `model_impl` | 项目要求“声明先于实现”，所有决策从声明出发 | `tileinfer plan build` 可产出计划；缺口报告可计算；与 Qwen3.6 互不影响 |
| `tileinfer/contracts/manifest.py` | 新增 `StateSpec`；`AttentionSpec` 增加 `sliding_window`、`rope_by_layer`；`ModelManifest` 增加 `state`、模块式 `model_impl`；允许 `conv`/`sliding_attention` | 原 schema 只能表达 full/linear + GDN | 能合法表达 LFM3 的卷积状态、滑窗、双 RoPE 和路线 O；合并 upstream 后 Qwen plan id 为 `079341ffd3f17b0d` |
| `tileinfer/resolve/decide.py` | 按 `state.kind` 区分 GDN/short_conv；`model_impl` 保留在 `Plan` 上并请求 `model_table`，不再生成 `model.arch` 计划行 | 厂商 `state.linear` 是 GDN，直接选会“假装覆盖”；模型注册由 `model_table` 统一负责 | LFM3 计划出现 `tileinfer.short_conv`，`plan.model_impl=tileinfer.bridge.models.lfm3`；Qwen 仍 8 条全厂商 |
| `tileinfer/providers/catalog.py` | 新增 route-O 作用域的 short-conv provider 声明（覆盖、可图、并行、证据、删除条件） | 这是真实缺口，不是速度问题，也不能变成全局默认 | 计划可以选到 `tileinfer.short_conv`；`graphable=True` |
| `tileinfer/providers/state/short_conv.py` | 纯 torch 的因果 depthwise 卷积与滚动状态；prefill/step/batch 三种入口 | 厂商没有短卷积状态实现 | 数学可独立在 CPU 测试；模型层只负责状态缓存/元数据 |
| `tileinfer/bridge/seams.py` | `install_model_table()` 按 upstream 约定导入 `model_impl` 模块并调用 `register_config()` / `register_models()`；`impl_table` 识别“provider 被路线 O 模型内部消费”；`model_table` 在计划请求时作为关键 seam | vLLM 在创建 `ModelConfig` 时就会做架构回退，不先注册会去 import HF remote code 并失败 | 插件路径 `skipped=none`，所需 seam 全 ok；模型类在配置解析前就位 |
| `tileinfer/bridge/models/lfm3.py` | `register_config()` 应用 transformers/rope 兼容补丁并关闭 vllm-ascend 全局 cos/sin 缓冲；`register_models()` 按类路径注册 `Lfm3ForCausalLM` | 与 upstream 的 `model_impl` 机制保持一致；LFM3 特例不再硬编码在 bridge 里 | 两种模型（MuseGlimmer / LFM3）走同一套注册路径 |
| `tileinfer/models/lfm3/model.py` | 完整 vLLM 模型实现：短卷积层（接 vLLM short-conv 状态元数据）、滑窗/全注意力、QK RMSNorm、per-head gate、零尾 RoPE、sandwich norm、SwiGLU、qkv/w13 权重合并、状态页对齐填充 | vLLM 无 LFM3 实现，HF remote code 与 vLLM 布局不兼容 | 权重正常加载（3.38 GB），C1/C3/C4 跑通 |
| `tileinfer/models/lfm3/compat.py` | transformers 5.16/5.17 兼容 shim：阻止 `conv` 重映射、补齐/清理嵌套 rope 参数 | 新版 transformers 的配置规范化与 LFM3 自定义 `conv`/per-layer rope 冲突 | `ModelConfig` 能解析 LFM3；不导入 HF `modeling_lfm3.py`，绕开布局冲突 |
| `tileinfer/bridge/models/lfm3.py` 中的 `_patch_ascend_rope()` | 关闭 vllm-ascend 的全局 cos/sin 缓冲更新 | 厂商只维护一份按模型级 partial factor 定尺寸的缓冲；LFM3 同时有 64/128 两种旋转维度，必然错位 | 不再触发 `update_cos_sin` 形状断言；模型使用自己的逐层 RoPE |
| 短卷积状态填充 | 状态形状从 `(6,2048)` 填充到 `(32,2048)`，前向只取前 6 槽 | 满足 Ascend 的 SSM/attention 页大小对齐断言 | 通过 `Cannot align ssm_page_size and attn_page_size` 检查，不改变卷积数学 |
| 短卷积 decode 优化 | 用 7 次向量乘加替代 grouped `F.conv1d`；状态 gather/scatter 改到连续原始布局 | Ascend 上 2048 组 depthwise conv 每层约 8 ms，13 层约 110 ms | TPOT 从 ~138 ms → 5–7 ms |
| `tests/test_lfm3_manifest.py` | 声明/计划/GDN 不误选/短卷积数学/批量 prefill 测试 | 固化关键假设 | CPU 守护测试全绿 |
| `docs/notes/lfm3-adaptation.md` | 技术证据、复现命令、C1–C4 状态、限制 | 项目要求“没有证据不能算在役” | 可复现的适配记录 |

### 3.2 关键实现细节

1. **路线 O 注册**：模型声明里的 `model_impl="tileinfer.bridge.models.lfm3"` 保留在计划字段中；bridge 在 `ModelConfig` 创建前导入该模块，先跑 `register_config()`（transformers/rope 兼容 + Ascend rope 补丁），再跑 `register_models()`（`ModelRegistry.register_model()`），避免 vLLM 走 transformers 回退。
2. **短卷积状态**：vLLM 的 `ShortConv` 自定义算子只提供 CUDA/Triton 路径，Ascend 上的 `forward_native` 是空实现；因此模型层继承 `MambaBase`，自己用 vLLM 的 short-conv 元数据（`state_indices_tensor_*`、`query_start_loc_*`、`has_initial_states_p`）驱动纯 torch 卷积。
3. **双 RoPE**：没有使用 vLLM `get_rope(partial_rotary_factor=0.5)`，因为它的配对方式与 HF 不同；在模型内实现了 HF 等价的 `Lfm3RotaryEmbedding`（零尾 inv_freq + 完整 128 维 `rotate_half`）。
4. **RMSNorm**：使用 float32 方差、与 HF 一致的实现，减少数值差异。
5. **权重映射**：`conv` → `short_conv`；`q/k/v_proj` → `qkv_proj`；`w1/w3` → `w13`；`lm_head` 与 `embed_tokens` 绑定。
6. **可图性**：短卷积 prefill/decode 全部改成张量索引与批量运算，不再读取 `.item()`；C3 捕获成功。

## 4. 验证与性能

### 4.1 C1：能跑

真实插件入口（没有手工调用 `install()`）。合并 upstream `model_impl` 机制后已用新计划在 LFM3 真权重上重跑：

```text
[tileinfer] TileInfer v3 bridge: plan=497db0c9c88f09b3 model=LFM3-1.7B
selections=7 ... access_points=6 |
seams=config_gate,graph_hook,attn_select,impl_table,model_table | skipped=none
OUT 0 ' Paris.\n\nThe capital of Japan is Tokyo.\n\nThe capital of Australia is Canberra'
```

`tileinfer doctor`：python / vllm / vllm_ascend / device_type / plan / layer_guard / bridge / graph_metrics 全 PASS。

### 4.2 C2：跑对（容差内）

同一请求、温度 0，参考实现为同机 HF transformers 5.17（厂商没有基线）：

| 排名 | HF logprob | vLLM logprob |
|---|---|---|
| ` Paris` | -0.1842 | -0.1854 |
| ` the` | -2.6842 | -2.6854 |
| `:\n\n` | -4.3717 | -4.3729 |
| `Paris` | -4.4967 | -4.4354 |
| `:` | -5.4342 | -5.3729 |

top-5 集合一致，每个 logprob 在声明容差 `0.1` 内；新计划下第 3/4 名不再互换。贪心生成前 2 个 token 一致，之后分叉（`What is ...` vs `The capital of ...`），原因是 Ascend FIA 与 HF SDPA 的累积顺序不同。

逐层对比（HF decoder layer hook vs vLLM hook）显示差异是渐进的，没有某层突变：

```text
layer 00 conv  max=0.00098 mean=0.000029
layer 10 full  max=0.01562 mean=0.000583
layer 18 full  max=0.06250 mean=0.002304
layer 22 full  max=0.05029 mean=0.005642
```

最后一层 pre-norm 输出 RMS：HF 0.2393，vLLM 0.2391；1.81 RMS 是 final norm 之后的值。结论：**不是权重/卷积/RoPE 的 bug，是 FIA vs SDPA 的数值差**。要逐位一致，需要写一个按 HF SDPA 累积顺序的 attention provider。

### 4.3 C3：能上图

短卷积改成张量操作后，图模式实测（新计划 C1 运行）：

```text
capture_calls=1  captured_graphs=8  (PIECEWISE 4 + FULL 4)
decode_calls=16  hits=16  fallbacks=0  errors=0
```

更早的长跑记录为 `decode_calls=567 / hits=567 / fallbacks=0`。batch=2、~1K token、`max_num_batched_tokens=512`（chunked prefill）也正常，说明 `has_initial_states_p` 的续接状态路径可用。

### 4.4 C4：跑得快

C1/C2/C3 的重跑只改了模型注册机制，没有改模型数学；下表沿用同一实现的测量，未在新计划下重做 p50/p90 门禁。新计划冒烟（1024 prompt + 64 output，conc=1）单请求 0.534 s，约 120 tok/s，与下表一致。

64 output tokens，512 引擎，1/4/16 并发：

| prompt | conc | TTFT p50/p90 | TPOT | 吞吐 |
|---|---|---|---|---|
| 1024 | 1 | 195.8 / 195.8 ms | 6.29 ms | 107.9 tok/s |
| 1024 | 4 | 89.6 / 89.7 ms | 7.37 ms | 461.4 tok/s |
| 1024 | 16 | 285.9 / 363.4 ms | 7.90 ms | 1293.9 tok/s |
| 4096 | 1 | 73.7 / 73.7 ms | 5.27 ms | 157.7 tok/s |
| 4096 | 4 | 246.2 / 320.6 ms | 7.51 ms | 352.3 tok/s |
| 4096 | 16 | 766.0 / 1226.3 ms | 14.28 ms | 610.7 tok/s |
| 7680 | 1 | 138.7 / 138.7 ms | 5.30 ms | 135.3 tok/s |
| 7680 | 4 | 396.9 / 474.0 ms | 9.16 ms | 264.9 tok/s |
| 7680 | 16 | 1296.9 / 2126.2 ms | 23.29 ms | 364.3 tok/s |

HF 参考（conc=1）：TTFT 349.7 / 78.9 / 148.9 ms，TPOT 13.77 / 11.69 / 11.73 ms（1K/4K/8K）。

判定：

- conc=1：TTFT 和 TPOT 均 ≤ 1.10× HF，**C4 通过**；
- conc>1：没有 HF 参考，按绝对吞吐记录；1K/4K/8K 的 conc16 分别为 1293.9 / 610.7 / 364.3 tok/s；
- 4K/8K conc16 的 TPOT 升到 14.28 / 23.29 ms，是下一步优化点；
- 1K conc1 的 TTFT 高于 conc4，属于首轮图重放/缓存效应，趋势以 TPOT 和吞吐为准。

一个重要的过程数据：第一版短卷积 decode 用 grouped `conv1d`，TPOT 约 **138 ms/token**（比 HF 慢 10 倍）。改成 7 次向量乘加 + 连续状态布局后降到 **5–7 ms**，这是本次性能达标的关键。

### 4.5 复现命令

```bash
cd /home/userlist/ww/TileInfer_xjx1
/home/userlist/ww/py_venv/bin/pip install --no-deps --no-build-isolation \
    --target /tmp/tileinfer_pkg .
/home/userlist/ww/py_venv/bin/python -m tileinfer.cli.plan build \
    --model /home/models/LFM3-1.7B --out /tmp/lfm3-plan.json

PYTHONPATH=/tmp/tileinfer_pkg \
TILEINFER_PLAN=/tmp/lfm3-plan.json \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
/home/userlist/ww/py_venv/bin/python -c "... LLM(...) ..."
```

要求：transformers 5.16+（实测 5.17）；单卡；NPU 空闲。

## 5. 效果评估

**已经做到的：**

- LFM3 从“vLLM 完全不认识、厂商没有实现”变成可加载、可生成、可上图、性能达标的模型；
- TileInfer 的路线 O 第一次跑通：声明 → 计划 → 模型替换 → 验证；
- 短卷积 provider 是真实缺口，不是重复实现厂商已有能力；
- 性能在 conc=1 对 HF 参考达标，并发吞吐可用。

**还不够的：**

- C2 不是逐位一致，只有容差内；
- 4K/8K 高并发的 TPOT 明显上升（14–23 ms），需要 profile；
- 单卡、无 TP/量化；
- 依赖 transformers 5.17，属于较新的版本组合；
- 没有覆盖 prefix caching、投机、量化、长上下文阶梯的完整验证矩阵。
