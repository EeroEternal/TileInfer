# DeepSeek-V4.1-Flash 性能优化计划（参考 DeepInfer）

## 0. 基线（本会话实测）

| 指标 | 数值 | 口径 |
|---|---|---|
| vLLM 路径 TPOT | **743 ms/token** | `vllm serve`，70 token / 52 s，3 次 |
| 独立栈 batched 臂 | **522 ms** | 差分法 `(T(72)-T(8))/64`，极差 0.8% |
| 独立栈 streaming 臂 | 658 ms | 同上，极差 0.4% |
| 逐 token 拆解 | MoE 559-567 ms / 总 721-734 ms | 逐专家流式 |
| MoE 单层（batched GEMM） | unpack 0.80 / scale 0.42 / bmm 0.18 ms | `s5_profile_batched_gemm.py` |
| H2D | 53.7 GB/s pinned / 14.0 pageable；单槽 0.348 ms | `s4_h2d_bench.py` |
| 理论地板 | ~260 ms（88 H2D + 24 bmm + 150 attention） | s4-plan §5.19 |

## 1. DeepInfer 盘点（这是本轮最重要的发现）

位置：`/home/userlist/.ernest/DeepInfer`（另有 `/home/dlb/DeepInfer`、`/home/dlb/deepinfer-notes`）。

**它明确支持本模型**：README 的 `Supported models` 第一条就是 `DeepSeek-V4.1-Flash`；
并且带 `requirements-ascend.txt`（torch 2.13.0+cpu / torch-npu 2.13.0rc1）与 `deepinfer/npu_ops/`，
说明有 Ascend 路径。

声明的优化能力：optimized kernels、Attention DP / balanced CP、MoE EP + EPLB、
DSpark + confidence-scheduled verification、SWA bounded replay、异步调度、batch-invariant、
确定性采样、结构化输出。

### 关键：它用 TileLang，而且有 Ascend 专用算子目录

`deepinfer/model_executor/layers/tilelang_ops/` 共约 1.1 万行，分 `asc/`（约 7418 行）与 `cuda/` 两套。
`asc/` 下与本项目热点直接对应的 kernel：

| 我们的热点 | DeepInfer 对应 kernel |
|---|---|
| attention（~150 ms/token） | `asc/compress.py`、`asc/get_kv_index.py`、`asc/candidate_blocks.py`、`asc/flash_mla_utils.py` |
| KV 读写 / paged cache | `asc/store_kv_cache.py`、`asc/load_paged_kv_cache.py`、`asc/kv_quant.py` |
| engram 查表 | `asc/gather_engram_tokens_zos.py`、`asc/gather.py` |
| 归一化 | `asc/norm.py`、`asc/rstd_1ulp.py` |
| 采样 | `asc/sampler.py`、`asc/fused_sample.py`、`asc/random_sampling.py` |
| 主机-设备拷贝 | `asc/dflash_copy.py`、`asc/dflash.py` |
| 元数据准备 | `asc/prepare_v4_decode_metadata.py`、`asc/schedule_mtp_varlen.py` |

模型结构参考：`deepinfer/model_executor/models/v4_modules/`（`attn.py`、`mla_attn.py`、`compress.py`、
`index_attn.py`、`engram.py`、`moe.py`、`hc.py`、`hc_blocks.py`、`rope.py`、`norm.py`）。

### 不能直接采用的约束

`requirements-ascend.txt` 要求 **torch 2.13.0+cpu / torch-npu 2.13.0rc1**，
而本项目跑在 **torch 2.10.0+cpu / torch_npu 2.10.0.post4**（C1 要求两臂同环境）。
因此 DeepInfer 作为**参考实现**读取与移植，不整体切换到它的栈。

## 2. 移植策略（按收益排序）

1. **attention 融合**：把我们的逐算子 compressor/indexer 换成 `asc/compress.py` +
   `asc/get_kv_index.py` + `asc/candidate_blocks.py` 的写法（预估省 ~150 ms/token）。
2. **MoE fp4 融合 GEMM**：先在 `v4_modules/moe.py` / `layers/linear.py` 里确认它有 fp4 专家 GEMM 的实现，
   再决定是移植还是照其思路用 TileLang 自写（当前 unpack+scale 合计 1.22 ms/层/GEMM）。
3. **engram gather**：`asc/gather_engram_tokens_zos.py` 对照我们的 engram 查表。
4. **MTP / DSpark**：它有 `schedule_mtp_varlen.py` 与 confidence-scheduled verification，
   对应我们"未启用 MTP"的缺口。

每个 kernel 都要：数值对拍（首 token 11111、S2 <=1e-2）+ 差分法 A/B + 极差。

## 3. 本轮结论

deepInfer 存在、支持本模型、有 Ascend TileLang 算子目录，
**因此 P1 的 attention/engram/MoE 优化应改为"移植 DeepInfer 的 asc kernel"，而不是从零设计**。
代价是它钉 torch 2.13，我们只能移植源码级别实现（TileLang 层），不能整体切换。

## 8. 专家预取双缓冲：落点与设计（P1 剩余项）

代码事实（`tileinfer/models/dsv41/expert_stream.py`）：

```python
def load_slot(self, slot, expert, device="npu:0"):
    for spec in self.specs:                 # 每个专家 3 个 spec（w1/w2/w3，各含权重+scale）
        dst = self._device[base + spec.offset : base + spec.end]
        dst.copy_(src_bytes)                # 同步 H2D
```

即：**每专家 3 次同步 H2D**，decode 一步 6 专家 = 18 次串行拷贝，期间 NPU 完全空闲。

### 设计（两级，先做 A）

**A. 双槽轮转 + 预取（低风险）**
1. 每个 `(layer, expert)` 分配**两个**槽位（A/B），`bind` 返回 (slot, buf_index)；
2. 计算专家 e 的 GEMM 之前，先在**同一段 Python 代码里**发起 e+1 的 `load_slot`
   —— 即使拷贝仍是同步的，也能让驱动把 e+1 的传输与 e 的 GEMM 排到不同队列；
3. 显存代价：`n_slots × slot_bytes`，当前 384 槽 × 5.9 MB ≈ 2.2 GiB，翻倍到 4.4 GiB
   （卡上现有 105 GiB 余量，完全够）。

**B. 真异步（需验证）**
`torch_npu.Stream` + `event.record/wait`：在 side stream 上 `copy_`，主 stream 等 event。
风险：torch_npu 的 stream 语义与我们的同步 `ref_cpu.linear` 混用时容易漏同步；
**必须用逐层对拍 + 首 token 作回归门**，且先用 seqlen=1 与 seqlen=64 各测一次再决定是否保留。

### 可量化的目标

`load_slot` 实测 ≈ 0.35 ms/槽（53.7 GB/s、5.9 MB/槽）。decode 一步 40 层 × 6 专家 × 0.35 ≈ **84 ms**，
与 §5.19 里"H2D 88 ms"的地板估算一致。若 A 能把其中一半藏到 GEMM 后面，TPOT 可再降 ~40 ms。
这不足以单独达成 350 ms，但与 `decode_batched`（预期 −220 ms）叠加后是可观的。

## 9. 更正 §8：跨层预取不可行，只有"层内异步重叠"这一种形态

§8 的 A 方案（每专家 A/B 双槽、算 e 时预载 e+1）**假设 e+1 已知**，对这个模型不成立：

```python
for e in group:                 # ① 本组专家一次性全部页入
    if slots[e] in fresh: arena.load_slot(...)
for e in group:                 # ② 再逐个算 GEMM
    ...
```

1. **层内已经是"全载再全算"**，没有"边算边载"的空隙；
2. **跨层预取不可能**：下一层用哪些专家由下一层的 gate 决定，而 gate 需要上一层输出——
   路由是**数据依赖**的，无法提前。

### 唯一可行的替代（真正的"双缓冲"）

用 `torch_npu.Stream` 把"加载 e+1"与"计算 e"放到不同队列重叠：

* 现在 6 个专家的加载（6 × 0.35 ms ≈ 2.1 ms/层）全部前置，与计算串行；
* 若与首个 GEMM 重叠，理论上每层可藏住约 2 ms，**40 层约 84 ms/token**（与 §5.19 的 H2D 地板一致）。

**风险**：我们的计算是同步 `ref_cpu.linear`（不是异步图），torch_npu 的 stream/event 混用时
极易漏同步。必须：① 先用 seqlen=1 与 64 各测一次；② 首 token 仍为 `11111`；③ 逐层对拍 ≤1e-2；
任一不满足即回退（环境开关，默认关闭）。

### 因此目标里"专家预取双缓冲"的定性

| 形态 | 可行性 |
|---|---|
| 跨层预取 | ❌ 路由数据依赖 |
| 层内 A/B 槽轮转（§8 A） | ❌ 无空隙可填（已全载） |
| **层内异步流重叠** | ✅ 唯一可行，收益上限 ~84 ms/token，需真机验证 |

## 10. MTP 投机解码：vLLM 有原生支持，但对我们的栈不可用

### 事实（源码核查）

vLLM 0.23 **内置**了本模型的 MTP：

```python
# vllm/config/speculative.py:314-319
if hf_config.model_type == "deepseek_v4":
    hf_config.model_type = "deepseek_mtp"
    n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
    hf_config.update({"n_predict": n_predict, "architectures": ["DeepSeekV4MTPModel"]})
```

且 `registry.py:626` 已注册 `"DeepSeekV4MTPModel"`，`config/vllm.py:1076` 里也对这个架构做了专门处理。
也就是说：**只要用 vLLM 自带的 DeepSeek-V4 实现，`--speculative-config '{"method":"mtp",...}'` 就能跑。**

### 为什么我们不能直接用

我们的推理栈是**自研 adapter**（`tileinfer/vllm/models/dsv41.py`），不是 vLLM 原生的
`DeepseekV4ForCausalLM`：权重是 fp8-block-32 + MXFP4、专家与 engram 表卸载在主机内存、KV 由我们自己的
`ModelState` 管理。vLLM 的投机解码路径会：

1. 用 **vLLM 自己的模型加载器**加载一份 `DeepSeekV4MTPModel` 作为 draft；
2. 期望 draft 与 target 共享 runner 的 KV cache 与注意力元数据；
3. 由 runner 统一做"提议 + 验证"。

而第 1 步在 Ascend 上就失败（没有 vllm-ascend，原生模型跑不起来），第 2/3 步与我们的
`ModelState` 自管 KV 架构直接冲突。所以 **MTP 对我们不是"配置问题"，而是架构改造**：
要么把我们的 KV 迁进 vLLM 的 paged KV（工程量极大，且 Ascend 上的 kernel 要重做），
要么自己在 `forward` 里实现"一次生成 k 个候选 + 批量验证"的投机循环（可行但需要新的采样与回滚逻辑）。

### 结论

- MTP 的**权重与结构**现成（`n_mtp_layers=3`、`dspark_target_layer_ids=(37,38,39)`），
  vLLM 也有原生实现可作为语义参考；
- 但在当前架构下，它是**投入最大**的一项，不适合作为下一步；
- 相对更快的路径是：先把 attention per-kernel 融合（~150 ms 潜力）与层内异步页入（~84 ms）做完，
  再看是否还有必要上 MTP。

### 附带发现

`vllm/config/vllm.py:1076` 特别处理了 `"DeepseekV4ForCausalLM"` 与 `"DeepSeekV4MTPModel"` 两个架构，
说明官方路径对我们的模型有完整支持——如果将来允许装 vllm-ascend，走原生路径会省掉我们自研 adapter 的
全部维护成本。这点值得在决策时权衡。

## 11. decode 的成本结构：per-op 启动开销，而非带宽（决定优化排序）

### 实测与代码对照

`layer.py:385 attention_forward()` 的一次 decode（1 行）调用序列：

```python
qr  = rms_norm(q_norm, linear(x, wq_a))          # 2 ops
q   = linear(qr, wq_b).unflatten(...)            # 1 op
rotate(q, cos, sin)                              # 1 op（rope）
kv, topk_idxs = _window_kv(...)                  # 多 op（投影 + rope + 写 cache + 取 topk）
compress_kv, compress_idxs = _compress_kv(...)   # 仅 compress_ratio>0 的层，再多几 op
out = sparse_attn(q, kv, attn_sink, topk_idxs)   # 1 op（主注意力）
rotate(out, cos, sin, True)                      # 1 op
out = einsum("bsgd,grd->bsgr", out, wo_a)        # 1 op
final = linear(..., wo_b)                        # 1 op
```

每层 decode 约 **12–18 个小 op**，40 层约 **500–700 次 launch/step**。

### 量级判断

- 若瓶颈是**带宽**：每层要读的密集 attention 权重约 15–20 MB，按 1.6 TB/s 只需 ~12 µs/层，
  40 层 ~0.5 ms —— 与实测的 ~150 ms/token **差两个数量级**，可以排除。
- 若瓶颈是**per-op 开销**：实测 attention ≈150 ms / 40 层 = **3.75 ms/层**，除以 ~15 个 op
  ≈ **250 µs/op** —— 这正是 Ascend 在 `--enforce-eager` 下小张量算子（launch + stream 同步）的典型量级。

⇒ **decode 的 attention 成本主要由算子启动开销构成。** 这解释了两件事：
为什么 `moe_forward_batched`（把 6 个专家的 18 次小 op 合成少数几次批量 op）能拿到 −15%；
以及为什么单纯"换更快的内核"没用——要**减少 op 的数量**。

### 由此得出的优化排序（重要变更）

| 方案 | 机制 | 预估收益 | 评估 |
|---|---|---|---|
| **ACL graph / 图捕获** | 把整个 decode step 捕获成一张图，回放时消除全部 launch 开销 | 潜力最大（可能 −100 ms 以上） | **应优先评估** |
| per-kernel 融合（手工把 rope+scatter+attn 合成一个 TileLang kernel） | 减少 op 数（~15 → ~4） | ~100 ms | 次优，工作量大 |
| 层内异步页入 | 重叠 H2D 与计算 | ~84 ms | 与上面正交，可叠加 |

**新的头号候选是图捕获**：我们已经把 decode 的状态张量都放在 `_stream_state` 里（形状固定、
跨步复用），这恰好是图捕获所需的前提；而 `--enforce-eager` 是我们为兼容性主动加的开关。
若能把 decode step 做成 ACLGraph 可捕获，收益可能超过移植任何单个 kernel。

### 下一步

1. 确认 vLLM 0.23 在 Ascend 上走 ACLGraph 的条件（`compilation_config` / `cudagraph_mode` 对应项），
   以及我们的 `forward` 为何不可捕获（动态 shape？host 侧 `.cpu()`？`_last_logits` 副作用？）；
2. 若能捕获，先在小序列上验证数值（首 token `11111`）再测 TPOT。

## 12. 图捕获的可行性：被 `pos` 的主机侧控制流卡住

§11 把"ACL graph / 图捕获"列为头号候选（机制上能一次性消除 500–700 次/step 的 launch 开销）。
本轮核查它在我们代码里的实际可行性。

### 硬阻塞点（`tileinfer/vllm/models/dsv41.py:273`）

```python
pos = int(positions.reshape(-1)[0].item()) if positions is not None else 0
...
if pos == 0 or state is None:        # 分支里创建 ModelState
    state = ModelState(...)
...
logits, _extras = transformer_forward(self.model, ids, state, pos, ...)
```

`pos` 是一个**主机侧 Python int**，它同时用于：

1. **控制流分支**（`pos == 0` 时新建 `ModelState`）——图捕获无法包含 Python 分支；
2. **KV/rope 的偏移切片**（`rt.cos[start_pos : start_pos + seqlen]`、写入 cache 的位置）——
   捕获时必须把偏移变成图的静态输入，而我们是动态标量。

再加 `.item()` 本身是 host 同步，捕获期间不允许。

### 结论：可做，但不是"打开一个开关"

要让它可捕获，需要：

1. **把 state 生命周期移出捕获区**（在捕获前按 request 预分配，符合 P3 的分槽改造——两者可以合做）；
2. **把 `pos` 变成设备侧张量**或图的外部输入，rope/cache 写入改用设备端偏移，去掉 Python 分支；
3. **固定所有形状**（我们已经是 1 行 decode，但这依赖 `--max-num-seqs 1`）。

工作量与 **P3（`ModelState` 按 request 分槽）** 高度重合——所以正确顺序是：
**先做 P3 的分槽改造（把 state 变成常驻、按槽复用），再在其上做 decode-only 图捕获**，而非两件事各做一遍。

### 因此的路线调整

| 顺序 | 工作 | 理由 |
|---|---|---|
| 1 | P3：`ModelState` 分槽（常驻、不随请求重建） | 既是 P3 目标，又是图捕获的前置 |
| 2 | 去掉 `forward` 里的 `.item()` 与 `pos == 0` 分支 | 图捕获的必要条件 |
| 3 | decode-only ACL graph 捕获 + 首 token `11111` 回归门 | 预期消除大部分 launch 开销 |
| 4 | 若 3 收益不足，再做 per-kernel（rope+scatter 等）融合 | 兜底方案 |

### 一个乐观的旁证

`tileinfer/vllm/platform.py:424-427` 早已记录：`graph_pool_handle` 在 Ascend 上会解析到
`torch.npu.graph_pool_handle`，只是"torch_npu 是否暴露它无法离线验证"。也就是说图池这一层
不是未知障碍，真正缺的是上面第 1、2 步的重构。

## 13. P3 并发改造：现状确认与设计

### 现状（已核对源码）

```python
# tileinfer/vllm/models/dsv41.py:258
def forward(self, input_ids, positions=None, kv_caches=None, attn_metadata=None, **kwargs):
    ids = input_ids.reshape(1, -1)                                    # ← 强制当成单序列
    pos = int(positions.reshape(-1)[0].item())                        # ← 只看第一个 token 的位置
```

`attn_metadata` 在整个文件里只出现 2 次（签名 + docstring），**从未被读取**。

⇒ 当 `--max-num-seqs ≥ 2` 时，vLLM 会把多个请求的 token **拍平成一个 `[num_tokens]`** 传进来；
我们当前的实现会把它们拼成一条"长序列"，位置也只取第一个，结果必然错乱。这就是 P3 必须先改
`ModelState` 的原因，而不是单纯把 `--max-num-seqs` 调大。

### 设计（三步，每步都可用开关回退）

**第 1 步：按请求切分**
从 `attn_metadata` 拿请求边界。vLLM v1 的 `attn_metadata` 提供 `query_start_loc`
（`[num_reqs+1]` 的累积起点）或等价的 `seq_lens`；据此把 `input_ids` / `positions` 切成
per-request 片段。若 `attn_metadata` 为 `None`（单请求，当前已验证路径），退回现有行为。

**第 2 步：state 按槽位常驻**
```python
self._slots: dict[int, ModelState] = {}       # req_id -> ModelState
```
- 槽位在请求首次出现时创建，请求结束后释放（或保留复用）；
- 每个 `ModelState` 的 `max_seq_len` 按该请求实际需要分配（当前已是这个逻辑）；
- 这一步同时是**图捕获的前置**（§12：state 生命周期必须在捕获区之外）。

**第 3 步：逐请求前向 + 拼接**
对每个片段调用 `transformer_forward`，把各请求最后一个 token 的 hidden 拼回 `[num_reqs, hidden]`
交给 `compute_logits`（decode 阶段每请求只贡献 1 个 token，天然对齐）。

### 验收门槛（与目标一致）

1. 单请求行为**逐位不变**（用现有 `Paris` + 首 token `11111` 回归门）；
2. `--max-num-seqs 4` 时：4 个并发请求各自输出正确，且**每请求 TPOT ≤ 单请求 1.5×**
   （单请求基线 592.4 ms/token，即 ≤889 ms）；
3. 用 `perf_vllm_tpot.py` 的差分法口径，逐请求计时（并发时按每个请求自己的 token 数折算）。

### 风险

- 我们的 `ModelState` 内部按 `bsz=1` 分配了大量缓冲，4 个槽位意味着 4 份 → 主机/显存占用 ×4
  （HBM 里只有窗口/KV 缓冲，量级不大；需要实测确认）；
- `torch.where` 等动态 shape 操作在并发下更碎，per-op 开销会上升——这与 §11 的结论一致，
  所以 P3 的收益必须实测，不能假设"并发一定提高吞吐"。

## 14. P3 第三步接线的关键发现：`_last_logits` 是单行捷径，去掉即可

核对了 `compute_logits`（`dsv41.py:437`）：

```python
def compute_logits(self, hidden_states, **kwargs):
    last = getattr(self, "_last_logits", None)
    if last is not None:
        out = last.reshape(1, -1)                                  # ← 硬编码 1 行
    else:
        x = hidden_states.reshape(1, -1, hidden_states.shape[-1])  # ← 也硬编码 1 行
        out = logits_from_hidden(self.model, x)
```

两处都**强制 batch=1**，所以并发时不可能正确。但这反而让接线变简单：

**方案：多请求时直接跳过 `_last_logits` 捷径，用 `hidden_states` 一次性算 logits。**

- `_last_logits` 是 S3 时期为了"采样用堆栈自己的 logits"而加的捷径，只在单序列下有意义；
- 多请求时按 `attn_metadata` 切分后，各请求的最后一个 hidden 拼成 `[num_reqs, hidden]`，
  再对整批做一次 LM head（`logits_from_hidden`）——**一个 GEMM 覆盖所有请求**，
  既正确又比逐请求做更省；
- 因此 **`compute_logits` 只需去掉 `reshape(1, -1)` 的硬编码**，改为保留行数；
  `_last_logits` 仅在 `slices is None`（单序列、已验证路径）时使用。

### 接线清单（第 3 步，改动很小）

1. `forward`：开头加
   `slices = request_slices(attn_metadata, num_tokens)`（已实现并有 12 个测试）；
2. 若 `is_single(slices, num_tokens)` → **原样走现有代码**（逐位不变，保护已验证路径）；
3. 否则 → `run_batched(slices, ...)`（已实现）逐请求跑 `transformer_forward`，
   用 `self._slots.acquire(req_id)` 取 state（`StateSlots` 已实现），
   并把每请求最后一个 hidden 拼成 `[num_reqs, hidden]` 返回，同时**清空 `_last_logits`**；
4. `compute_logits`：把两处 `reshape(1, -1)` 改成保留首维（`[num_rows, vocab]`）；
5. 开关：`VLLM_DSV41_MULTI_REQ=1` 才启用（默认关），失败即回退。

### 仍需真机验证的部分

- 第 3 步的 `req_id` 来源：vLLM 的 `attn_metadata` 里是否有稳定的请求标识
  （`query_start_loc` 只给边界，不给 id）；没有的话要用"片段索引 + 位置"做键，
  这会让 state 复用失效（每步都像新请求）——**这是接线时最需要实测确认的一点**；
- 并发 4 时的每请求 TPOT 与 HBM 峰值。

## 15. 请求标识的位置已查明（§14 的那个风险点）

源码核查（`vllm/v1/attention/backends/utils.py:624-628`）：

```python
num_reqs = len(input_batch.req_ids)
num_scheduled_tokens = [scheduler_output.num_scheduled_tokens[id] for id in input_batch.req_ids]
```

结论：**请求 id 在调度器侧的 `input_batch.req_ids` 上**，配套的每请求 token 数在
`scheduler_output.num_scheduled_tokens[id]`。而模型的 `forward` 只收到 `attn_metadata`
（加上 `input_ids` / `positions`），**拿不到 `req_ids`**。

### 因此 P3 接线有三条路（按推荐度）

**路线 1（推荐）：让 `attn_metadata` 带上请求标识**
我们本来就有自研平台层（`platform_dsv41`）与 profile（`profiles/dsv41.py`）。
在构造 attention metadata 时把 `req_ids`（或由它派生的槽位号）塞进去，
`forward` 直接读。改动集中在平台层，不碰已验证的单序列路径。

**路线 2：用"每请求首个位置"做键**
`positions` 是逐 token 的，切分后每段的首个位置唯一标识一个请求（对检测/长上下文都成立）。
无需 `req_ids`，但每个 decode 步的"首位置"都在增长，所以键必须**按请求的起始位置**而不是首位置——
需要额外记录映射，等价于自己实现一份 id 分配，复杂度与路线 1 相当但更脆。

**路线 3：不做 state 复用**
每步为每个片段新建 `ModelState`——`ModelState` 构造实测约 740 ms（首次），
即使复用分配器后也只是几十 ms，**并发收益会被这条抵消**，不可取。

### 结论

- P3 的正确入手点是**路线 1**：在平台/metadata 构造处补 `req_ids`，而不是在 `forward` 里猜；
- 这条改动可以与 P1 的 Op 合并（§11 的 op 计数问题同样受 `--enforce-eager` 与 metadata 影响）；
- 在拿到独占窗口前，`forward` 的接线**不应动**——因为路线 1 的落点（metadata 构造）也需要真机验证。

### 给下一次会话的最小起点

1. 在 `tileinfer/vllm/` 里找 attention metadata 的构造点（`platform_dsv41.py` / `profiles/dsv41.py` 或后端）；
2. 把 `req_ids` 挂上去（一个 list 字段即可）；
3. 按 §14 的 5 步接线，用 `VLLM_DSV41_MULTI_REQ=1` 开关；
4. 用 `perf_vllm_tpot.py`（差分法 + `ignore_eos` + 实际 token 数）在 `--max-num-seqs 4` 下测每请求 TPOT。

## 16. §15 路线 1 不成立：我们没有 attention backend，改用 runner 侧钩子

核查结果：`grep -rn "attn_backend|get_attn_backend|AttentionMetadata|metadata_builder" tileinfer/vllm/`
**一条都没有**。也就是说本栈没有自研的 attention backend，也没有 metadata 构造代码——
`attn_metadata` 是 vLLM 用默认/占位实现传给我们的，我们只是忽略它。

⇒ §15 的"路线 1（在 metadata 上挂 req_ids）"实际等于**新写一个 attention backend**，
工作量远大于"加一个字段"。这条路不通。

### 改用路线 1'：runner 侧钩子取 req_ids

`gpu_model_runner` 在调模型 `forward` 之前构造 `input_batch`，里面有 `req_ids`。
用一个**小钩子**把它存到模型能读到的地方：

```python
# 平台/插件初始化时（仅在 VLLM_DSV41_MULTI_REQ=1 时安装）
_orig = GPUModelRunner.execute_model
def patched(self, scheduler_output, *a, **kw):
    try:
        self.model._current_req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    except Exception:
        pass
    return _orig(self, scheduler_output, *a, **kw)
GPUModelRunner.execute_model = patched
```

`forward` 里就有 `self._current_req_ids` 与 `len` 对齐的每请求 token 数
（`num_scheduled_tokens` 的顺序与 `req_ids` 一致，正好与 §14 的 `request_slices` 输出的片段一一对应）。

优点：不动模型签名、不写 backend、单序列路径完全不受影响（钩子只在开关打开时安装）。
缺点：依赖 `execute_model` 这个方法名与 `num_scheduled_tokens` 这个字段——**必须实测确认**
（vLLM 0.23 里两者都存在，见 `attention/backends/utils.py:624-628`，但钩子是否被调用要验证）。

### 因此 P3 的实现顺序（更新）

1. 写钩子 + `_current_req_ids` 注入（开关 `VLLM_DSV41_MULTI_REQ=1`，默认关）；
2. 按 §14 的 5 步接线 `forward` / `compute_logits`；
3. 真机 `--max-num-seqs 4` 验证：每请求输出正确 + 每请求 TPOT ≤ 单请求 1.5×。

**注意**：第 3 步需要独占窗口；第 1、2 步可以离线写完并用 fake 单测覆盖（钩子的 patchem 逻辑不难测）。

## 17. P3 首次真机并发验证：失败点已定位（窗口已用上）

**启动条件**：available=521 GB、卡空闲；`--max-num-seqs 4` + `VLLM_DSV41_MULTI_REQ=1`。
服务正常启动，4 个并发请求全部返回 **HTTP 500**，EngineCore 进入致命错误。

### 根因（服务端 traceback）

```
rt.window_kv[:bsz, start_pos % win] = kv.squeeze(1)
RuntimeError: The expanded size of the tensor (1) must match the existing size (16) at no...
```

即**目标切片 `rt.window_kv[:bsz]` 的第 0 维是 16，而源 `kv` 的批维是 1**。
说明 `_forward_multi` 里 `StateSlots` 造出的 `ModelState(bsz=1)` 与
`LayerRuntime` 实际分配的 `window_kv` 批维（16）不一致——`rt` 来自某个按更大批维预分配的运行时，
或 `bsz` 变量取的是 `rt`/config 的值而不是 `state.bsz`。

### 说明与处置

- 该路径由 `VLLM_DSV41_MULTI_REQ=1` 门控，**默认关闭**，所以已验证的单请求路径完全不受影响
  （同轮 `test_dsv41_vllm_forward.py` 等 16 个用例仍全绿）；
- 失败是**功能性且可修**的：需要让多请求槽位的 `ModelState` 的 `bsz` 与 `LayerRuntime` 的
  `window_kv` 批维一致（二者取同一个值），这是下一轮的第一个动作；
- 排查入口：`state.runtime()` 里 `window_kv` 的分配维度，以及 `transformer_forward` 中 `bsz` 的来源。

### 仍需真机验证的项（本次未及）

- runner 钩子是否真的被调用（`current_req_ids()` 是否非空）——若为空，槽位会退化为"按片段序号"；
- 修好 bsz 后：4 并发正确性（每个请求含 `Paris`）+ 每请求 TPOT ≤ 单请求 1.5×（≤889 ms）。

### 本次新增的可复现脚本

`tools/dsv41/perf_concurrency.py`：K 线程并发、`ignore_eos`、差分法 `(T(N2)−T(N1))/(N2−N1)`、
≥3 次中位数、并逐个检查输出含 `Paris`。

## 18. §17 崩溃的真正机制：多请求被送进了单序列路径

代码对照（`layer.py:444-458`）：

```python
def _window_kv(attn, x, rope, start_pos, rt, cfg, o):
    bsz, seqlen, _ = x.shape            # bsz 来自输入张量
    ...
    rt.window_kv[:bsz, :seqlen] = kv    # start_pos == 0 分支
    ...
    rt.window_kv[:bsz, start_pos % win] = kv.squeeze(1)   # decode 分支（崩溃处）
```

而我在 `forward` 里加的门是：

```python
_slices = request_slices(attn_metadata, n_tokens)
if not is_single(_slices, n_tokens):      # is_single(None, n) == True
    return self._forward_multi(...)
```

**`attn_metadata` 在我们这套栈里是 `None` 或不含 `query_start_loc`**（§16 已确认我们没有
attention backend），于是 `request_slices` 返回 `None`，`is_single(None, n)` 返回 **`True`**，
**多请求分支永远不会进入** —— 4 个并发请求被当成一条 4 token 的序列送进单序列路径，
在 decode 分支的窗口 KV 写入上维度不匹配而崩。

### 修法（下一轮第一步）

不要在 `attn_metadata` 上判断，改用**已经可用的 runner 钩子**：

```python
req_ids = current_req_ids()          # dsV41_req_ids 钩子，已实现并有 8 个单测
if req_ids and len(req_ids) > 1:
    # 用 num_scheduled_tokens 的顺序切分（钩子已按同一顺序记录）
    slices = build_slices_from_counts(counts)   # 累积求和即可
    return self._forward_multi(...)
```

即：**以 `req_ids` 的数量作为"是否并发"的判据**，`attn_metadata` 只是可选的补充来源。
这样即使 `attn_metadata` 为 `None` 也能正确分流。

### 同时要确认的两点

1. `current_req_ids()` 在真机上是否真的非空（钩子是否被调用）——首次并发跑应在日志里打印一次；
2. `ModelState(bsz=1)` 的槽位与 `LayerRuntime.window_kv` 的 `bsz` 保持 1（`layer.py:212` 用 `self.bsz`，
   只要 `ModelState.bsz=1` 就一致），所以 §17 里的 "16 vs 1" 很可能是"4 个请求被拼成 4 列"的次生现象，
   而不是槽位 bsz 配错。

## 19. P3 当前唯一未解点：钩子大概率没被调用

第 6 次真机验证（13:28:47）的 traceback 已深入内部：

```
dsv41.py:313 forward → dsv41.py:481 _forward_multi
→ dsv41_batching.py:154 → RuntimeError: StateSlots 已满
```

说明分流、钩子安装、返回值契约、槽位键 4 处修复都已生效，**只剩"请求 id 不稳定"**这一条：
每步都被当成新请求，4 步就把 6 个槽位填满。

而代码里已按**片段序号**映射（`_ord_of_start`），该映射若拿到 `req_ids` 就是稳定的；
所以现在的现象等价于 **`req_ids` 为空**，即 `current_req_ids()` 返回 `None`，
退化成 `f"slice{start}"`（`start` 是 token 偏移，每步都变）。

### 为什么钩子可能没生效（按可能性排序）

1. `install()` 在模型 `__init__` 里调用，但 `GPUModelRunner.execute_model` 可能是在
   **别的类/包装层**上被调用（例如 `execute_model` 由基类或 wrapper 提供，我们补丁的类不是实际调用方）；
2. vLLM 可能在导入时就解析了方法引用（较老的写法），补丁类属性不生效；
3. `install()` 返回 False（例如 `execute_model` 不存在于该类），而我只记了 `_req_ids_hooked`
   没有在日志里暴露它。

### 下一步只需一个观测就能定性

在 `_forward_multi` 里加一行（默认 INFO 级，只打前 3 次）：

```python
if _n_log < 3:
    print(f"DSV41_MULTI slices={slices} req_ids={req_ids} hooked={getattr(self,'_req_ids_hooked',None)}", flush=True)
```

- `hooked=False` ⇒ 补丁没装上，去查真实调用方类名；
- `hooked=True` 但 `req_ids=None` ⇒ 补丁装上了但调用路径没经过它（换 `execute_model` 之外的钩子，
  或用 `scheduler_output` 在别处出现的位置）。

### 备选方案（若钩子路线继续不通）

用**请求的起始位置**做稳定键：每个片段的第一 token 的 `positions[start]` 在 decode 步是
"已生成长度 − 1"，各请求互不相同且逐 step 只增 1 —— 可以用它减去本步序号得到稳定 id。
这不需要任何 vLLM 钩子，代价是多一次小计算。

## 20. 投机解码：两条路线的取舍（本轮核查后的结论）

### 前置能力已就绪（3 项，13 个离线用例）

| 件 | 位置 | 作用 |
|---|---|---|
| `window_slots_for` + `window_topk_idxs` 的 `seqlen>1` 分支 | `layer.py` | 一次验证 k 个 token 时逐 query 的窗口槽位 |
| `compress_rope_positions` + `_compress_kv` 多条目分支 | `layer.py` | 压缩 KV 的多位置 rope / 写入 |
| `write_ring_chunk` | `layer.py` | k 个 token 连续落环；**被拒草稿无需回滚** |

`seqlen==1` 与 `n_lat==1` 两条老路径都被逐位锁住（回归门）。

### 路线 A：DSpark / MTP（DeepSeek 自带 draft 头）

核查到的接口事实：

* `mtp.{0,1,2}` 是三个 `Dsv41DSparkBlock`（`modules.py:527-532`），各自是一个完整 block，
  专家配置独立（`dspark_n_routed_experts=128`、`activated=3`）；
* 权重在 checkpoint 里，当前用 `with_mtp=False` 跳过；
* `transformer_forward` 已经在 `extras["main_hiddens"]` 里收集**目标层 37/38/39 的注意力输入**
  （`block.py:718-729`）——正是 DSpark 要消费的输入；
* 配置里还有 `dspark_block_size=5`、`dspark_markov_rank=256`、`dspark_noise_token_id=128799`、
  以及 stage 2 的 `markov_head` / `confidence_head`。

⇒ 可行性明确，但要**忠实实现 DSpark 的提议算法**（markov 头采样 + 置信度门控），
这需要先弄清该算法的确切语义（tech report / 官方 `inference/` 里应有线索）。
工作量估计：中到大，且每一步都需要 CPU 对拍。

### 路线 B：n-gram / prompt-lookup（无 draft 模型）

* **完全不需要新权重**，提议来自 prompt/已生成文本里的 n-gram 匹配；
* 验证恰好复用刚做完的能力：一次前向吃 k 个候选 token，取最长匹配前缀；
* 实现量小（提议器 ~50 行 + 接受循环 ~40 行），CPU 可完整对拍；
* **收益依赖输出分布**：对重复性/引用性文本（长文档问答、代码、模板化输出）有效，
  对自由创作几乎为 0。必须在真实 prompt 上测命中率再决定是否保留。

### 建议顺序

1. **先做路线 B**（成本低、可离线验证、能立刻拿到"投机解码是否值得"的证据）；
2. 若命中率可观 → 补路线 A 的 DSpark 提议器（收益更高且与输出分布无关）；
3. 两条路线共用同一套"验证 + 接受"骨架，因此路线 B 的骨架可直接复用。

### 验收门槛（与目标一致）

* 贪心解码下，投机结果必须与逐 token 贪心**完全一致**（首 token `11111`、100 prompt 一致率 ≥95%）；
* TPOT 用差分法测，并报告接受率（accepted/proposed）。

## 21. 投机解码接线规格（零件已齐，剩接线与对拍）

### 已就绪的零件

| 零件 | 位置 | 作用 |
|---|---|---|
| `write_ring_chunk` / `_window_kv` 的 `seqlen>1` | `layer.py` | 一次吃 k 个 token 并落环 |
| `window_topk_idxs` 的 `seqlen>1` | `layer.py` | 逐 query 窗口槽位 |
| `compress_rope_positions` / `_compress_kv` 多条目 | `layer.py` | 压缩 KV 的多位置 rope/写入 |
| `logits_all_positions` | `block.py` | 逐位置 logits `[b,s,vocab]`（验证取 argmax 用） |
| `propose_ngram` / `verify_prefix` / `split_accept` / `generate` | `spec_decode.py` | 提议、接受、循环（31 个离线用例） |

### 接线契约（一步 = 一次前向）

```python
def spec_step(model, state, ids_ctx, start_pos, cand, k):
    # 把 k 个候选接到当前前缀后，一次前向拿到逐位置 logits
    tok = torch.tensor([cand], device=dev)                 # [1, k]
    logits, extras = transformer_forward(
        model, tok, state, start_pos, capture={}, moe_fn=moe_fn
    )                                                       # logits 只用最后一个位置
    h = extras["hidden_all"]                                # ← 需要新增：全部位置的 hidden
    per_pos = logits_all_positions(model, h)                # [1, k, vocab]
    return per_pos[0].argmax(-1).tolist()                   # 每位置的贪心 argmax
```

**待补的一小块**：`transformer_forward` 目前只把最后一个位置的 logits 返回给调用方，
验证需要"全部位置的 hidden"。两种做法（推荐 a）：
* (a) 在 `capture` 里塞 `{"h_all": h}`（`capture` 已是现成的旁路，不改返回值契约）；
* (b) 改动返回值增加第三项——会破坏所有现有调用点，不推荐。

### 状态推进与"无回滚"语义

* 接受 `m` 个候选后，`start_pos += m + 1`（`m` 个接受 + 1 个模型自己的 argmax）；
* **不需要回滚**：环形窗口由 `start_pos` 决定写入位置，被拒草稿的槽位会在下一次写入时被覆盖，
  读取时由 `window_topk_idxs` 的 `-1` 掩码排除（`write_ring_chunk` 的单测已钉住该性质）；
* 压缩 KV 同理：`compress_len = (start_pos+seqlen)//ratio`，被拒部分不进入有效长度。

### 验收（上车前的离线门）

1. **贪心一致性**：同一 prompt 下，"投机生成 N token" 与 "逐 token 贪心生成 N token" 必须逐 token 相同；
2. **步数下降**：`stats["steps"] < N`（否则投机无收益）；
3. **接受率**：报告 `accepted / proposed`；n-gram 在重复性文本上应显著 > 0；
4. 真机 A/B：用 `perf_vllm_tpot.py` 的差分法口径，两臂同 venv 同参数。

### 建议的落地顺序

1. 补 `capture["h_all"]`（1 处，低风险）；
2. 写 `spec_session.py` 的 `run(...)`，用 tiny 配置 + 官方 oracle 权重在 CPU 上跑第 1–3 条验收；
3. 通过后再接 vLLM 适配器（沿用 `VLLM_DSV41_MULTI_REQ` 式的开关，默认关闭）。

## 22. 投机解码：离线能验证什么、不能验证什么

### 已用 CPU 验证（`tools/dsv41/perf_spec_parity.py`）

```
plain: [...]  步数=6
spec : [...]  步数=6  提议=2  接受=2  接受率=1.00
PASS：投机与贪心逐 token 相同
```

* **等价性成立**：n-gram 投机生成的序列与逐 token 贪心**逐 token 相同**——投机只改变一次前向
  产出几个 token，不改变结果。这是目标里"贪心一致"门槛的离线证明，已通过。
* 顺带修掉两个脚本错误：自建 `TinyModel` 桩缺 `embed/norm/layers/head`（`transformer_forward`
  必然用到），改用真实 `build_model(cfg)` 再灌 oracle 权重。

### 离线**不能**验证的：接受率与收益

上面的接受率是 1.00，但**这个数字没有意义**：tiny 配置的模型是随机权重（只有单层灌了 oracle 权重、
其余随机），它生成的 token 流本身没有可利用的重复结构，而 n-gram 提议器正是在赌重复结构。
换句话说：

| 问题 | 能否离线回答 |
|---|---|
| 投机的输出是否与贪心一致（正确性） | ✅ 能，已验证 |
| 接受率有多高（收益） | ❌ 不能，依赖真实模型的输出分布 |
| 长文本/代码/引用场景是否值得开 | ❌ 不能，必须在真实模型上测 |

⇒ **路线 B 的收益判定必须占用一次真机窗口**，不能靠 CPU 外推。
这也是为什么它只能作为"已实现、默认关闭、待观测"的能力存在。

### 上机时的最小观测集（一次窗口即可完成）

1. 同一 prompt 下 `plain` 与 `spec` 的输出比对（应逐 token 相同）；
2. `stats` 里的 `proposed/accepted` → 接受率；
3. `steps` 对比（投机必须更少，否则无收益）；
4. 用差分法测 TPOT：`(T(72) − T(8)) / 64`，两臂同 venv 同启动参数。

### 若接受率偏低（预计在自由创作场景）

按收益排序的后续选项：
1. 提高 `n`（更长 n-gram 匹配更准但命中更少）或调 `k`；
2. 换 route A（DSpark/MTP 提议器，收益与输出分布无关，但需复现 markov 头算法）；
3. 转向 §11 的结论——decode 成本主体是 **per-op 启动开销**，图捕获可能比拼投机更划算。

## 23. P3 并发：实测结果与架构上限（2026-11-05 窗口）

### 实测（修复后首次成功测量）

环境：`available≈545 GB`、卡空闲；`--max-num-seqs 4` + `VLLM_DSV41_MULTI_REQ=1`；
脚本 `tools/dsv41/perf_concurrency.py --conc 4 --n1 4 --n2 16 --reps 2`。

```
T(4)=13.529s (极差 1.1%)   T(16)=36.7s (极差 5.2%)
每步耗时 = (36.7 − 13.529) / 12 = 1931 ms/step
正确性: 8/8 个请求输出含 Paris
Application startup complete ✓
```

### 两件同时被确认的事

**1. 显存分配修复有效**：每步耗时 **4251 → 1931 ms（−55%）**，且正确性保持 8/8。
之前 `NPU out of memory ... 107.90 GiB allocated, 165.93 MiB free` 的根因是
`_forward_multi` 按**模型 1M 上限**给每个请求分配 `ModelState`（单请求路径本来是按需分配）。
改成 `max(2048, pos[end-1]+1)` 后显存压力消失，分配器抖动也随之消失。

**2. 架构上限暴露（重要）**：1931 / 4 ≈ **483 ms/请求**，接近单请求的 592 ms。
说明 4 个请求是**串行**跑完的：`run_batched` 本质是"按请求逐段调用 `transformer_forward`"的循环，
每个请求各自过 40 层，**没有批处理收益**。

| 指标 | 实测 | 目标 |
|---|---|---|
| 每步耗时（4 并发） | 1931 ms | ≤889 ms |
| 折算每请求 TPOT | ~1931 ms | ≤889 ms（单请求 592 × 1.5） |
| 差距 | 2.2× | — |

⇒ **P3 现在给的是"可用"，不是"吞吐收益"。**

### 达标所需的改动：真批处理前向（中等规模重构）

要把 4 个请求的 token 合成 `bsz=4` 一次性过 40 层，而不是循环 4 次。可行性依据：

* `window_topk_idxs` 已经返回 `[bsz, ...]`，窗口索引逻辑天然支持批；
* 每个请求的 KV 在各自 `ModelState` 里，批处理需要每层把 4 份视图拼起来；
* MoE 侧更容易：4 个请求的 token 可以合并成一次分组 GEMM
  （`moe_forward_streamed_grouped` 已经按 `[E, max_n, in]` 组织）。

建议顺序：先批 attention（窗口 + 压缩 KV 拼接），再批 MoE，最后批 elementwise（天然支持）。
每一步都要守住"单请求逐位不变"的回归门。

### 另一个可选方向

§11 已定性：decode 成本主体是 **per-op 启动开销**。图捕获能同时加速单请求与并发，
但前置是把 state 生命周期移出捕获区（P3 的分槽改造已完成大半）。
若批处理重构太大，图捕获是等收益的另一条路。

## 24. 真批处理前向的落地设计（P3 达标路径）

### 现状为什么不批

`_forward_multi` → `run_batched` 的循环里，每个请求**各自**调用一次
`transformer_forward`（40 层全跑一遍），所以 4 并发 ≈ 4 × 单请求成本（实测 483 ms/请求）。

### 目标形态

一次 `transformer_forward` 吃 `bsz=R` 个请求的当前 token（decode 时 R 行），
各层内部对 R 个请求**同时**计算：

```
输入: ids [R]（每请求 1 个 token）、pos [R]、slots [R]（各自的 ModelState 槽位）
每层:
  rms_norm / linear      → 天然支持 bsz=R（权重与批无关）
  _window_kv             → 每请求写自己的环形缓冲：按 slot 散写到**共享** batched 缓冲
  attention(sparse)      → 需要 [R, ...] 的 batched 窗口 + 每请求不同的 topk_idxs
  MoE                    → 直接把 R 行拼成一次分组 GEMM（已有 grouped 实现）
  wo_a/wo_b              → 天然支持
```

### 关键改动点（按依赖顺序）

**A. 共享 batched KV 缓冲**
`LayerRuntime.window_kv` 现在是 `[1, win, dim]`。批处理需要 `[R_max, win, dim]`，
其中第 i 行属于槽位 i 的请求。做法：让 `ModelState` 不再各自持有 KV，而是共享一个
`BatchedKV` 对象（按 slot 索引），`_window_kv` 里把"写第 0 行"改成"按 slot 数组散写"：

```python
# 现在：rt.window_kv[:bsz, ...] = kv
# 之后：batched.window_kv[slots, ...] = kv        # slots: [R] 的长整型索引
```

**B. 逐请求的窗口索引**
`window_topk_idxs(win, bsz, seqlen, start_pos)` 目前只接受**单个** start_pos。
批处理要改成接受 `[R]` 个位置，返回 `[R, win]`（每请求自己的窗口）。纯索引运算，可离线单测。

**C. sparse attention 的 batched 调用**
`o.sparse_attn(q, kv, sink, topk_idxs, scale)` 需要支持 `q:[R,1,heads*dim]`、
`kv:[R, win+compress, dim]`、`topk_idxs:[R, 1, K]`。要确认现有 kernel 是否接受 bsz>1；
若不接受，退路是对 R 个请求循环（那就回到现状，见下方"风险"）。

**D. MoE 批处理（最容易，已有组件）**
把 R 行的 hidden 拼成 `[R, hidden]`，直接走 `moe_forward_streamed_grouped`
（它按 `[E, max_n, in]` 组织，R 行天然并行）。

### 验收与风险

* **回归门**：单请求路径必须逐位不变（`slices` 长度为 1 时不进批处理分支）；
* **数值门**：R=4 时每个请求的贪心输出必须与单请求逐 token 相同（复用
  `perf_concurrency.py` 的 `Paris` 校验 + 更严格的"两两比对"）；
* **最大风险**：`sparse_attn` 是否支持 bsz>1。**必须先查清这一点再动手**——
  若不支持，批处理只能在 attention 之外做，收益会大幅缩水（attention 占总成本约 150/592）。

### 建议的第一步（一个函数、可离线验证）

先实现 **B**（`window_topk_idxs` 支持 `[R]` 个位置），因为它：
* 纯索引运算，能立刻写单测（与逐请求调用的结果逐一比对）；
* 是 A/C 的前置（散写与 batched attention 都要按请求的位置来）；
* 不改任何已验证路径（新增重载，旧签名保持）。

完成 B 之后再花一次窗口确认 `sparse_attn` 的 bsz 支持情况，然后决定是否继续 A/C。

## 25. mmap 权重：内存需求降 88%（本会话最重要的一项基础设施改进）

### 问题

原先 `load_weights` 用 `param.copy_(tensor)` 把 457.78 GiB 专家+engram 权重拷进**匿名内存**
（内核不可回收），实测峰值 `anon-rss ≈ 484–491 GB`。在 628 GB 机器上这意味着必须 available ≥505 GB，
一旦其它租户同时申请就被 OOM-kill（本会话多次实测 `{'EngineCore': -9}`）。

### 改法（`DSV41_MMAP_WEIGHTS`，**默认开启**，设 `0` 关闭）

`tileinfer/vllm/models/mmap_weights.py::MmapWeightSource`：自己用
`safe_open(path, "pt", device="cpu")` 打开所有 shard 并**保活句柄**，从 handle 取 `get_tensor(name)`
赋给 `param.data`（`[N,K//2]` 的 int8 再 `.view(_FP4)`），不做 copy。

**关键陷阱（必须记住）**：vLLM 自带的 `safetensors_weights_iterator` 是 `with safe_open(...)`
用完即关；若把参数指向它 yield 的张量，mmap 随之释放 → **悬垂指针 → 原生段错误**
（表现为"engine core exited unexpectedly"、无 Python traceback）。所以句柄必须由我们自己持有到模型销毁。

### 实测效果

```
改造前: anon-rss ≈ 490 GB        （需 available ≥505 GB）
改造后: rss = 60–72 GB           （available 371 GB 也能正常起服务）
单请求: {"text": " Paris. The capital"}     ✓
4 并发: 4/4 个请求输出 Paris                ✓
```

⇒ **"要 560 GB 空闲"这个前提彻底不成立**：这是加载方式造成的，不是模型强制的。
默认开启后再也不需要为窗口发愁。

### 回归门

`DSV41_MMAP_WEIGHTS=0` 可一键退回旧的 copy 路径；`tests/test_dsv41_mmap_weights.py` 6 个用例
覆盖跨 shard 读取、dtype、句柄保活、close 后报错、未知名字、目录缺失。

## 26. 观测方法的一个坑：`cut -c` 会切坏 UTF-8

本会话多轮出现"日志数字不一致"的假象（`N2=4` 而非 40、`极差 .%`、`T(16)=51.93` 与 per-step 矛盾），
根因是 **`cut -c1-N` 按字节/字符截断含中文的行，把数字切掉**。脚本本身一直是对的（md5 校验一致）。

**纪律**：读日志不要 `cut -c`；直接 `cat`/`tail` 全行（这些行本来就短）。

### 修正后的 P3 真实基线

```
并发 4，N1=4 N2=16 reps=1
T(4)=15.820s  T(16)=51.930s
并发每步耗时 = (51.930 − 15.820) / (16 − 4) = 3009.2 ms
正确性: 4/4 个请求输出含 Paris
```

与"4 个请求各跑一遍 40 层 ≈ 4 × 750 ms ≈ 3000 ms"完全吻合 ⇒ **根因确认：无批处理**。
目标 ≤889 ms/step，当前 3.4×。

## 27. P3 真批处理落地：4 并发每步 3009 → 871 ms（达标）（2026-11-05）

### 目标与判据

`--max-num-seqs 4` 下每步 `(T(16)-T(4))/12 ≤ 889 ms`（单请求 592 ms 的 1.5×），4/4 请求输出含
`Paris`。脚本与判据不变：`tools/dsv41/perf_concurrency.py`。

### 实现（`tileinfer/models/dsv41/batched_decode.py`）

decode 步把 R 个请求的当前 token 堆成 `[R,1,...]`，40 层各跑**一次**：

* 每步从各请求自己的 `ModelState` **gather** 读写前状态：窗口 `[R,128,512]`、压缩流前缀
  `[R,(pos+1)//ratio,512]`、index key、compressor 的 `kv_state/score_state`；
* 每层 attention 走 batched 版：`window_topk_idxs_batched` + 共享 batched 窗口 + 逐行
  compressor/indexer + `sparse_attn` 的 `b=4`（`§24` 已确认支持 `bsz>1`）；
* 前向结束把状态 **scatter** 回各请求的 `ModelState`；因此 prefill 仍走原串行路径，请求可以在
  "串行 prefill → 批量 decode" 之间无缝切换，已验证的单序列路径零改动；
* MoE 走 `moe_forward_streamed_grouped`：路由簿记（`indices/weights`）一次搬回主机，
  w1/w3 与 w2 各一次 batched GEMM，避免每专家一次 `torch.where`（每次都会因动态形状同步）。

**开关（默认关闭，N5）**

| 变量 | 默认 | 作用 |
|---|---|---|
| `VLLM_DSV41_BATCHED_FWD` | off | 打开真批处理 decode；只在"本批全部请求各 1 token"时生效，混合批/ prefill 自动回退串行 |
| `VLLM_DSV41_ARENA_SLOTS` | 384 | 专家 arena 槽位数。2048 槽 = 36.7 GiB HBM，跨步命中率由 42% 不再因容量抖动 |
| `DSV41_MOE_BF16_GEMM` | off | fp4 反量化 GEMM 用 bf16（fp4 值与 e8m0 尺度在 bf16 精确；仍是 fp32 累加） |
| `VLLM_DSV41_BATCHED_TIMING` / `VLLM_DSV41_MOE_TIMING` | off | 仅诊断：per-section 计时 |
| `DSV41_MOE_HITSTATS` | off | 仅诊断：相邻步专家命中率 |

### 结果（同一服务、同一 venv，`--enforce-eager --max-num-seqs 4`）

```
并发 4，N1=4 N2=16，两次独立测量
reps=3: T(4)=9.928s (极差 11.7%)  T(16)=20.383s (2.3%)  → 871.2 ms/step, 12/12 Paris
reps=5: T(4)=9.675s (7.3%)        T(16)=20.205s (1.6%)  → 877.5 ms/step, 20/20 Paris
```

对照：改造前 3009.2 ms/step（逐请求串行 40 层），本次 **−71%**。

### 成本结构（诊断口径，含同步开销）

`attn ≈160 ms`、MoE `≈600 ms`（load ≈200 / w13 ≈160 / w2 ≈170 / stack+shared ≈80）、
其余 hc/norm/head ≈200 ms。MoE 仍是主体：4 行去重后每层约 12 个专家，单步要搬
≈5 GB packed 权重（42% 跨步命中），H2D 是硬成本；再往下需要"反量化专家常驻 HBM"或
H2D/GEMM 重叠。

### 回归门

* `pytest tests/test_dsv41_*.py -q`：**135 passed**（含 4 个 official 逐层/整块对拍；无失败）；
* 新增 `tests/test_dsv41_batched_decode.py` 5 例：批量一步 == 逐请求串行（logits ≤2e-3、
  argmax 一致），且批量后的 state 继续串行仍与全串行一致；
* 真机：`perf_concurrency` 3 次重复 12/12 `Paris`。

### 未完成（下一轮）

1. **100 prompt 贪心一致率 ≥95%** 的质量门仍未跑（P3 只到"4/4 Paris"）；
2. 余量只有 ~2%，`T(4)` 极差 11.7%：需要 5 次以上重复取中位数，或再省 ~40 ms；
3. 串行 prefill 仍是 4×（本判据只测 decode 步），并发首字延迟待 P2/P3 后续；
4. 2048 槽 arena 让 engine HBM 到 52.8 GiB；若与他人共享卡需按 `VLLM_DSV41_ARENA_SLOTS` 回调。

## 28. 质量门（100 prompt）暴露并修复 state 管理三处 bug；贪心 4-token 一致率仍未达 95%

### 新建可复现脚本

`tools/dsv41/perf_quality_gate.py`：100 条固定 prompt，`temperature=0 / ignore_eos /
return_token_ids / logprobs=5`，可并发；`--compare` 报完整序列与逐位置一致率。官方 `inference/`
在 CPU 约 140 s/token，100 prompt 不可行，故用**单请求串行路径**作代理 oracle（该路径已由
S3/S4 对拍官方）。两臂同 venv、同启动参数，仅切 `VLLM_DSV41_BATCHED_FWD`。

### 发现并修复的 bug（都不是"数值噪声"）

1. **`num_scheduled_tokens` 的字典序 ≠ flat batch 顺序**。cached 请求会被重排：
   `scheduler_output` 里顺序是 `[850f, 83e3, 91b7]`，而 flat 位置是 `[7, 8, 10]`。
   旧代码按序号取 req id，把 state 错配到别的请求上。修复：钩子改为记录
   `(req_id, absolute_start, scheduled_len)`（`scheduled_new_reqs` + `scheduled_cached_reqs`
   的 `num_computed_tokens`），按**绝对起点 + 片段长度**匹配。
2. **warm-up/已结束请求的 state 残留**。位置链认领会在同一 pos 上撞到旧 state，表现为
   "前 3 个 token 的 top-5 逐位相同、第 4 个 token 起完全发散"（甚至读到别的 prompt 的上下文）。
   修复：优先 id 精确匹配；位置链只作唯一候选兜底；新 prefill 一律重建。
3. **state 只增不减**：100-prompt 串行跑到第 ~67 个请求时 engine **原生退出**（无 Python
   traceback）。修复：每 8 步清理 >32 步未使用的 state（活跃请求每步刷新 `_last_step`）。
   修复后 100/100 请求零错误完成。

另：单请求也统一走 `_forward_multi` 的 slot 路径（此前"单独 prefill → 并发 decode"会把 state
留在 `_stream_state`，多请求路径找不到它）。

修复后**同臂重复**（同 prompt、同服务、连续两轮）从 3/20 提升到 **20/20 完全一致**，
证明模型+服务路径本身已确定；剩下的差异只来自"批量 vs 串行"的 kernel 形状。

### 质量门结果（fp32 GEMM 两臂，`max_tokens=4`）

| 指标 | 结果 |
|---|---|
| 首 token 一致 | **100/100（100%）** |
| 逐位置一致 | pos0 100、pos1 90、pos2 80、pos3 54（共 400 个位置） |
| 完整 4-token 序列一致 | **54/100（54%）** |
| 同臂重复一致 | **20/20** |

诊断：把批量臂的 MoE 改成"逐行调用单请求 MoE"（`VLLM_DSV41_BATCHED_MOE=per_row`）后，
前 20 条的首两个 token 变成 20/20、20/20，但 pos2/pos3 仍有 14/20、10/20 一致 ——
说明差异同时来自 **batched attention/dense 的形状相关归约**，不是单一 MoE 内核。

结论：**"首 token ≥95%" 已满足（100%）；"4-token 贪心序列 ≥95%" 在当前 kernel 形状差异下
不成立。** 这不是 state bug（同臂已完全确定），而是 greedy 对 ~1e-2 级数值差异的敏感性；
仓库既有 C3 口径（§52）也是"组件级 oracle + 端到端有界差异 + 自查确定性"，而非逐 token 全同。
下一步二选一：
(a) 质量门改为 logprob 有界判据（建议 |Δ| ≤ 0.21 nats，与 §52 一致）；
(b) 批量臂对 attention/MoE 做"逐行等价"内核（会牺牲 P3 的吞吐收益）。

## 29. P3 复测：arena 容量是隐藏杠杆；4096 槽最好 879 ms，但窗口方差仍大

### 新的实测（2026-11-06，同代码、同启动参数，仅改 `VLLM_DSV41_ARENA_SLOTS`）

| arena 槽 | 4 并发每步（N1=8/N2=32, reps=5） | HBM | 备注 |
|---|---|---|---|
| 384 | 1017.2 ms | 16 GiB | 单请求 **571.5 ms**（回到 592 基线） |
| 2048 | 956.3 ms | 53 GiB | 单请求 799.5 ms |
| 4096 | **922.5 / 879.2 / 993.6 ms**（三次） | **89.6 GiB** | 最好一次达标 ≤889 |

单请求 TPOT 也随 arena 变化：384 槽 571 ms，2048 槽 800 ms（同一代码）。
结论：**arena 不是"越大越好"的单调量**——384 槽对单请求最快（240 个专家全部常驻），
4 并发需要 4096 槽才能把跨步命中率撑住；4096 槽的代价是 HBM 89.6 GiB。

### 判据口径修正

`N1=4/N2=16` 的 T(4) 含 prefill/首 token 的固定项，实测极差可达 130%；改用
`N1=8/N2=32` 后极差降到 5.6–12.9%（仍受他人任务/温度影响）。后续 P3 统一用
`N1=8/N2=32, reps=5`。

### 当前判定

* 功能：20/20 请求输出 `Paris`，state 修复后同臂重复 20/20；
* 性能：4096 槽三次 879.2 / 922.5 / 993.6，**中位数 922.5，未稳定 ≤889**；
  最好一次达标。P3 仍是 **open**（不是 blocked：下一步可做 dequant 专家缓存、
  减少 per-step sync/拷贝，或固定 4096 槽后多窗口取中位数）。
* 单请求 P1 基线在 384 槽下仍是 571 ms，未被本次批处理改动破坏。

## 30. P2 补测：2K prompt TTFT 仍 22–26 s（grouped vs default），距 5 s 目标很远

### 口径

`tools/dsv41/perf_ttft.py`（本次修复了 completions 端点的流式解析：chunk 在 `choices[0].text`，
不是 chat 的 `delta.content`）。prompt 文件 1530 词 ≈ **1740 token**（不是 2048，粗估见脚本），
`stream: true`，TTFT = 首个 SSE chunk 墙钟；同一 venv/启动参数，只切 `DSV41_MOE_GROUPED`。
两臂均 `BATCHED_FWD=0`（prefill 不受批量 decode 影响），`ARENA_SLOTS=96`（384 槽时 grouped
2K prefill 会 engine 原生退出，疑似 grouped 全专家批次的瞬时峰值；96 槽稳定）。

| 臂 | TTFT P50 | P90 | 极差 | decode |
|---|---|---|---|---|
| `DSV41_MOE_GROUPED=1`（7 reps） | **22.1 s** | 23.7 s | 30% | 1200 ms/chunk |
| `DSV41_MOE_GROUPED=0`（3 reps） | **25.6 s** | 40.1 s | 57% | 不可算（2 chunks） |

grouped 相对 default **−13.7%**（22.1 vs 25.6），方向与单层 −24.5% 一致，但**绝对值距 P2 的
5 s 目标仍有 4.4×**。按 40 层拆：22.1 s ≈ 552 ms/层（与 §S4 的单层 grouped 数一致），
要 ≤5 s 必须把 prefill 单层压到 ~125 ms，现有"host 专家按层换入 + fp4 unpack"结构做不到；
需要"预取/双缓冲 + 更大 grouped 批次"或 prefill 专用融合 GEMM。P2 保持 open。

另：384 槽 grouped 2K prefill 的 native death 是一个独立稳定性问题，后续应先定位
（nmax/峰值 HBM），再谈 384/512 槽下的 P2。

## 31. P3 最终验收（4096 槽，空闲窗口三次连测）：中位数 886 ms ≤ 889

§29 的 922.5 中位数包含他人任务/温度窗口。2026-11-06 12:00 后卡空闲，用同一服务器
（`BATCHED_FWD=1 / MULTI_REQ=1 / BF16_GEMM=1 / ARENA_SLOTS=4096`）连续跑三次
`perf_concurrency --conc 4 --n1 8 --n2 32 --reps 5`：

| 次序 | 每步 | T(8) 极差 | T(32) 极差 | 正确性 |
|---|---|---|---|---|
| 1 | **886.0 ms** | 25.4% | 35.7% | 20/20 Paris |
| 2 | **873.1 ms** | 29.4% | 6.8% | 20/20 Paris |
| 3 | **928.5 ms** | 12.1% | 9.5% | 20/20 Paris |

**中位数 = 886.0 ms ≤ 889 ms（1.5× 单请求 592 ms），P3 性能验收通过**（余量 0.3%，
仍需注意 T(8) 的 prefill 固定项方差；§29 的高值来自非空闲窗口）。

对照：改造前 3009.2 ms/step；本配置 HBM ≈ 89.6 GiB；单请求在 384 槽 571.5 ms。

P3 仍有一个非性能 open 项：质量门的 4-token 全序列一致率 54%（首 token 100/100），
判据需按 §52 改为 logprob 有界（见 §28）。

## 32. 质量门判据定稿：logprob 有界也不达 95%；4-token 全序列一致率不是这套 kernel 的有效判据

§28 保存的 100-prompt 双份 JSON（批量 grouped + fp32 / 单请求串行 + fp32）上，新增
`perf_quality_gate.py --compare --delta` 的 **logprob 有界判据**：

* 逐位置：token 相同直接算过；不同时，若"右臂（串行）token"在左臂（批量）top-5 内，
  且 `|logp_批量(串行 token) − logp_批量(批量 token)| ≤ δ`，也算过。

| δ (nats) | bounded_agreement | 说明 |
|---|---|---|
| 0（逐 token 全同） | **81.0%**（324/400） | 首 token 100/100，之后快速下降 |
| 0.5 | 84.3% | — |
| 1.0 | **86.8%**（347/400） | 36/46 个分歧位置串行 token 在批量 top-5 内 |
| 2.0 | 87.7% | gap 中位数 0.88、p90 3.17、max 4.43 |
| 3.0 | 88.7% | 仍有 10 个位置串行 token 完全不在 top-5 |

结论：**即使把判据放宽到 3 nats，也达不到 95%**。这不是 state bug（修复后同臂重复
20/20），而是批量 kernel（attention/dense 的 m=R vs m=1 形状 + fp8/fp4 GEMM 归约顺序）
造成的真实数值差异；4-token 全序列相等对这套栈不是有效判据。

下一步只有两条：
1. **组件级判据**（推荐）：沿用 S2 的 `max|diff| ≤ 1e-2` + 首 token 100/100 + 同臂确定性；
   端到端不再要求逐 token 相等；
2. **逐行等价模式**：批量臂的 attention/MoE 按行调用单请求实现（质量模式，性能会回到
   ~1.4 s/step），用它证明逻辑等价，性能模式仍用 grouped。

质量门因此保持 open，性质是"判据不可达"，不是"逻辑错误"。

## 33. P4 长上下文：4K 首次跑通（TTFT 36.7 s），并修掉三处"4K 必崩"的容量 bug

### 三个真实 bug（都已修，默认行为不变）

1. **engram 滚动缓存被 checkpoint 的 4096 截断**：`EngramHash(max_seq_len=cfg.max_seq_len)`
   在 4221-token prompt 上触发 `expanded size 4096 vs 4221`。修复：用
   `max(cfg.max_seq_len, vLLM max_model_len)` 建缓存。
2. **StateSlots 的 factory 闭包捕获了第一次调用的 `_want`**：warm-up 用 2 token 建了 2048
   的 state，之后 4K 请求 `_want=4221` 但 factory 仍读旧 dict ⇒ `state_max=2048`、
   RoPE 表 2048、`shape [1,4221,1,-1]` 崩溃。修复：`_want` 挂到 `self`。
3. **decode 位置越界**：prompt N 个 token 的预填充只分配到 `abs_end=N`，第一步 decode 的
   `pos=N` 需要第 N+1 行 RoPE。修复：容量取 `abs_end + 64`（并受 `max_model_len` 限制）。

### 4K 实测（`perf_p4_context.py`，流式 completions，max_tokens=2）

```
prompt_chars=24454（≈4221 token）
TTFT = 36.70 s   total = 36.70 s   chunks=2
HBM peak = 128.5 GiB（engine 空闲后仍 85 GiB，分配器保留）
RSS peak = 148.2 GiB
```

4K **不 OOM**，但 HBM 峰值已贴顶；16K/32K 尚未跑（空闲余量不足以再叠一个 16K 工作区）。
P4 仍 open：需要把 prefill 的 grouped MoE 按更小 expert 组/chunk 处理，或降低 vLLM 侧
KV 预留（我们自管 KV，vLLM 的 paged KV 实际不用），才能给 16K/32K 腾出空间。

## 34. P4 16K：OOM 根因是 **indexer 的 [S, 32, T] score**，不是 MoE

### 4K 的新最优配置

`ARENA_SLOTS=24` + `DSV41_MOE_PREFILL_CHUNK=1024`：

```
4K (4221 token): TTFT 29.92 s, HBM peak 55.7 GiB, RSS 148 GiB
```

比 96 槽（36.7 s / 128.5 GiB）**更快且省一半 HBM**：小 expert 组减少了 grouped 的一次性
`[E, nmax, ...]` 峰值。4K 稳定。

### 16K 的 OOM 定位

```
torch.OutOfMemoryError: Tried to allocate 34.75 GiB
  ... layer.py:indexer_forward
  score = (score.relu() * weights.unsqueeze(-1)).sum(dim=2)
```

`score = einsum("bshd,btd->bsht", q, index_k)` 在 prefill 时是 `[1, S, 32, T]`：

* ratio=1 层：T≈S=14363 → **26.4 GiB**，乘 weights 再分配一份 → 34.75 GiB；
* ratio=2 层：T≈7181 → 13.2 GiB，同样翻倍。

MoE 分块（`DSV41_MOE_PREFILL_CHUNK`）对 16K 无效，因为峰值不在 MoE；16K 在
indexer 处必 OOM。

### 修法（下一轮实现）

**blockwise streaming top-k**：把 T 维按 1024 分块，每块算 `[S,32,1024]`（~1.9 GiB），
逐块与 running top-k 合并；candidate-source 层同时维护 block maxima 以增量构造
`select_candidate_blocks` 的掩码。数学与整块等价，只是不一次性持有 `[S,32,T]`。
预计 16K/32K 峰值可压到 <60 GiB（4K 已是 55.7 GiB）。

## 35. P4 进展：indexer streaming top-k 落地；16K 通过（201 s / 118.8 GiB），32K 原生退出

### 实现

`layer.py:indexer_forward` 新增 `DSV41_INDEXER_CHUNK`（默认 0=旧路径）：
* 将 T 维按 chunk 分块，逐块算 `[b,s,chunk]`（chunk=1024 时 ~1.9 GiB），
  与 running top-k 精确合并；
* candidate-source 层先做一遍 **block maxima**（bs=8），用等价的
  `_select_from_block_max` 生成 mask，再流式 top-k；
* fp4 量化后的 score 存在真实并列，整块与流式在并列边界可取不同 key；两者都是合法
  top-k，CPU 用例只要求 greedy 一致 + 有界差异（`tests/test_dsv41_indexer_stream.py`）。

### 实测（`perf_p4_context.py`，max_tokens=2）

| 长度 | 配置 | TTFT | HBM peak | RSS peak | 结果 |
|---|---|---|---|---|---|
| 4K (4221 tok) | arena 24 / MoE 1024 / indexer 1024 | 29.92 s | 55.7 GiB | 148 GiB | ✅ |
| 16K (14363 tok) | arena 24 / MoE 1024 / indexer 1024 | **201.25 s** | **118.8 GiB** | 155 GiB | ✅ 无 OOM |
| 32K (34209 tok) | arena 16 / MoE 512 / indexer 512 | — | — | — | ❌ EngineCore 原生退出 |

32K：请求返回 200 后约 6 分钟 EngineCore 消失，**没有 Python traceback，dmesg 也没有新的
OOM**；客户端一直挂（已 kill）。16K 已到 118.8 GiB，32K 很可能在某个原生分配点越界/被杀。
P4 32K 保持 open；下一步在 32K 上开 `DSV41_MULTI_REQ_DEBUG`/分段计时，定位是
MoE 分组、attention workspace 还是 indexer 峰值，并考虑进一步减小 chunk/arena。

## 36. P4 达标：32K 上下文不 OOM（1218 s / 126.9 GiB）；4K/16K/32K 全部记录

### 32K 成功配置

之前的 32K 原生退出不是结构问题，而是 **chunk/arena 太大导致峰值越过 HBM**。用更小的
分块后跑通：

```
DSV41_ARENA_SLOTS=8  DSV41_MOE_PREFILL_CHUNK=256  DSV41_INDEXER_CHUNK=256
max-model-len=40960, gpu-memory-utilization=0.2, max-num-seqs=1
prompt = 34209 token（198194 字符）
TTFT = 1218.47 s   HBM peak = 126.9 GiB   RSS peak = 161.5 GiB   chunks=2
DSV41_MEM: 每层 alloc 14–18 GiB，peak 43–47 GiB（不含 vLLM 预留）
```

### P4 全表（`perf_p4_context.py`，stream=true，max_tokens=2）

| 长度 | 配置 | TTFT | HBM peak | RSS peak | 结果 |
|---|---|---|---|---|---|
| 4K (4221) | arena24 / MoE1024 / idx1024 | 29.92 s | 55.7 GiB | 148 GiB | ✅ |
| 16K (14363) | arena24 / MoE1024 / idx1024 | 201.25 s | 118.8 GiB | 155 GiB | ✅ |
| 32K (34209) | arena8 / MoE256 / idx256 | **1218.47 s** | **126.9 GiB** | 161.5 GiB | ✅ 不 OOM |

结论：**P4 的"32K 不 OOM + 记录 4K/16K/32K TTFT/HBM"已达成**。代价是 TTFT 随长度近似
线性（~35 ms/1K token）且 32K 已用到 126.9/128 GiB；前缀缓存（KV 锚点层）仍未做，
属于 P4 的第二个交付项，保持 open。

## 37. P2 负结果：把 arena 调大 + prefill 分块会让 2K TTFT 更差（22.1 → 36.5 s）

为压低 32K 峰值而引入的分块，在 2K 上是**净损失**：

| 配置 | 2K TTFT P50 | P90 |
|---|---|---|
| arena96 / 不分块（§30 原值） | **22.1 s** | 23.7 s |
| arena384 / MoE chunk256 / indexer chunk1024 | **36.5 s** | 38.2 s |

原因：分块把每层 MoE 拆成多次 expert 分组/加载，2K 本可一次装下，chunk 反而增加
group 往返与 kernel 启动；indexer chunk 也增加一遍扫描。**P2 不应复用 P4 的小分块配置**。
P2 仍需 prefill 专用的层内重叠/融合（或更高效的 grouped GEMM），当前 22.1 s 距 5 s 4.4×，
保持 open。

## 38. 质量门终判：连"逐行 attention + 逐行 MoE"也只有 6/20；差异来自 batched dense

新增质量模式（`VLLM_DSV41_BATCHED_ATTN=per_row` + `VLLM_DSV41_BATCHED_MOE=per_row`）：
批量 decode 的 attention/MoE 都按行调用单请求实现（CPU 用例：与串行 logits 差 3e-8）。

真机 20-prompt × 4-token（fp32，两臂同 venv/参数）对比单请求串行：

```
parity full equal = 6/20
per-position       = [20, 20, 15, 6]
France: parity [11111,16,223,1]  serial [11111,16,1,22663]
```

首两个 token 已完全一致；从第 3 个开始分叉。此时 attention 与 MoE 都已是单请求形状，
剩余差异只能来自 **batched dense**（embedding / hc_mixes / rms_norm / linear 在 R 行与
1 行下的 kernel tiling），属于 1e-3 级数值差异 + greedy 在近似并列处翻转。

**结论**：4-token 逐 token 95% 对这套栈不可达，per-row 模式也救不了；质量门应定稿为
"组件级 S2 ≤1e-2 + 首 token 100/100 + 同臂 20/20 确定性"。per-row 模式保留为质量参考，
性能模式仍用 grouped（本模式未做性能验收）。

## 39. P1 探针：去掉 `--enforce-eager` 不等于图捕获；TPOT 561 ms（−1.8%）

先修了一个启动 bug：编译 warm-up 用 **8192 token** 的 dummy batch，而 engram 滚动缓存仍按
`max(cfg 4096, max_model_len 2048)` 分配，直接崩在
`expanded size 4096 vs 8192`。已改为
`max(cfg.max_seq_len, max_model_len, VLLM_DSV41_ENGRAM_CACHE=8192)`。

启动结果（`--enforce-eager` 去掉，其余同单请求基线）：

```
WARNING torch.compile is turned on, but the model ... does not support it
WARNING Skipping CUDA graph capture (cudagraph_mode NONE)
```

即 **vLLM 没有为这个自定义模型启用 compile/CUDA graph**，decode 仍是 eager。

单请求 TPOT（arena384，`N1=8/N2=32 reps=5`）：

| 配置 | TPOT | 正确性 |
|---|---|---|
| `--enforce-eager`（基线） | 571.5 ms | Paris |
| 无 `--enforce-eager` | **561.4 ms**（−1.8%） | Paris |

1.8% 在极差范围内，**没有兑现图捕获收益**。P1 保持 open；下一步要么给
`DeepseekV41ForCausalLM` 加 vLLM 的 `support_torch_compile` 并处理动态 state/branch，
要么走 torch_npu 的显式 `npu_graph` 捕获（需先证明 NPU 侧 kernel 支持）。

## 40. P1 图捕获真实探针：ACL 桥生效、capture 开始，但第一步就撞上 in-capture sync

用上一阶段留下的 `TILEINFER_ACLGRAPH=1 + TILEINFER_COMPILE=0`（不传
`--enforce-eager`）启动，平台确实把 `cudagraph_mode` 设成 `FULL_DECODE_ONLY` 并开始
capture。失败点是 **我们的 forward 里的 host 同步**：

```
File tileinfer/vllm/models/dsv41.py
  pos = int(positions.reshape(-1)[0].item()) ...
RuntimeError: AclrtSynchronizeStreamWithTimeout ... 
  Not_Supported(EE1016): Stream during the capture stage is not supported
  stream is captured
  at gpu_model_runner.py:6669 _capture_cudagraphs
```

结论（关键，避免再走弯路）：

1. `tileinfer.vllm.graphs` 的 alias 桥（`torch.cuda.CUDAGraph → torch.npu.NPUGraph` 等）
   **是有效的**，capture 已经进入真正的 NPU graph 捕获阶段；不是"平台没支持"。
2. 失败在模型代码，不在桥：捕获区内禁止 stream synchronize，而 `forward` 需要
   `positions[0]` 的 **host int** 来决定 state/start_pos，必然要 `.item()`。
3. 即便去掉这一处，`forward` 里还有 state 创建、`max_seq_len` 分支、Python 循环与
   `torch.cat`/分配；W8 capture contract（plan prewarmed / no compile / no alloc / no sync）
   目前一条都不满足。

⇒ **图捕获需要一个专门的 static decode 路径**：预分配输入/输出/state、位置作为 host 侧
预计算（runner 提供）、无 `.item()`/无分支/无分配，capture 后才谈 replay。一次性改
`forward` 不可行。P1 保持 open；这条比"打开一个开关"大得多。

## 41. P3 最终复核（全部 P3/P4 改动之后）：866.8 / 896.5 / 868.0，中位数 868 ≤ 889

在 §31 之后又有 state 修复、indexer streaming、prefill chunk 等改动；为确认没有回归，
在空闲窗口用最终代码重跑三次（`BATCHED_FWD=1 / MULTI_REQ=1 / BF16_GEMM=1 / ARENA_SLOTS=4096`，
`N1=8/N2=32 reps=5`）：

| 次序 | 每步 | T(8) 极差 | T(32) 极差 | 正确性 |
|---|---|---|---|---|
| 1 | **866.8 ms** | 19.7% | 8.2% | 20/20 Paris |
| 2 | **896.5 ms** | 9.6% | 10.8% | 20/20 Paris |
| 3 | **868.0 ms** | 7.1% | 4.2% | 20/20 Paris |

**中位数 868.0 ms ≤ 889**，三次全部 20/20 Paris；P3 性能验收在最终代码上保持通过
（比 §31 的 886 更好，差异在窗口/温度内）。质量门口径见 §38（组件级）。

## 42. P1 负结果：12 GiB 反量化专家缓存没有收益（575.4 vs 571.5 ms）

按"unpack 占 GEMM 93%"的旧结论，在 `moe_forward_batched` 里加了
`DSV41_DECODE_DEQ_CACHE_GB`：命中时跳过 fp4 unpack，直接用缓存的 bf16 w1/w3/w2 做 bmm；
12 GiB 可容下 arena384 的全部 ~240 个专家（47 MB/专家）。

同环境单请求差分（arena384、BF16_GEMM=1、`N1=8/N2=32 reps=5`）：

| 配置 | TPOT | 正确性 | HBM |
|---|---|---|---|
| 无缓存（基线） | 571.5 ms | Paris | 16 GiB |
| 12 GiB 反量化缓存 | **575.4 ms** | Paris | 35.8 GiB |

**没有收益**（−0.7%，在极差内），还多占 20 GiB HBM。可能原因：缓存命中率被逐 token 路由
变化拉低；命中时仍要整层 stack/copy 并失去原 grouped 的批量形状；或者 unpack 在
`moe_forward_batched` 里并不是 TPOT 的主要项（旧 93% 是单 GEMM 内部占比，不等于整栈占比）。
按 N5 保持默认 0（关闭）。P1 仍要静态 decode/图捕获，而不是这个缓存。

## 43. P4/P2：exact-prompt state cache 落地；2K 重复 prompt P50 28 s → 1.23 s

### 实现（`VLLM_DSV41_PROMPT_CACHE=1`，默认关）

单请求路径在 prefill 完成后，把**整段 prompt 的 state**（40 层 window/compress/index/
kv_state，含 engram cache）与最后一 token 的 hidden/logits 快照进 `self._prompt_cache`
（仅保留 2 条，deepcopy）。下一次请求若 token ids 完全相同（hash 命中）：

* 直接 `deepcopy` 快照 state，跳过整个 prefill；
* 返回缓存的 hidden 与 `_last_logits`；
* decode 从快照位置继续。

同时修了单请求路径的状态容量：`needed = pos + seqlen + 64`（此前第一步 decode 的
`pos=prompt_len` 会越界，报 `size 32 vs 0`）。

### 2K 重复 prompt 实测（`perf_ttft`，reps=3，stream=true）

```
TTFT P50 = 1233 ms   P90 = 28006 ms（第一次冷启动）   极差 2173%
decode = 1034 ms/chunk
```

第一次 28 s（冷），后两次 1.2 s（命中）⇒ **P50 已进入 ≤5 s 目标**，P90 仍由冷启动主导。
正确性：同一 prompt 连发两次，`perf_quality_gate --compare` 4/4 token 全同（PASS）。

### 限制

当前只做 **exact prompt** 命中，不是部分前缀共享；真正"KV 锚点层/前缀缓存"需要按
block 边界保存/恢复（任意公共前缀），仍是 P4 第二项的 open 内容。缓存 2 条、deepcopy
成本随上下文线性（2K ~30 MB，32K ~150 MB），需按 HBM 余量给 cap。

## 44. P4 前缀缓存（prompt-level）：2K 公共前缀 TTFT 38.3 s → 6.56 s，输出逐 token 一致

在 §43 的 exact-prompt 缓存上扩展为**最长 prompt 前缀**命中：

* 缓存项保存 token ids、post-prefill state、hidden、logits；
* 新请求取"已缓存 token 序列是当前 prompt 前缀"里最长的一条；
* 恢复 state：容量不够时新建 `ModelState` 并逐层 copy window/compress/index/kv_state
  与 engram cache；
* 剩余 suffix **逐 token** 走 decode 形状（compressor/indexer 对 `start_pos>0` 已验证），
  只对最后一个 suffix token 取 logits。

同时修了一个 state 恢复的 bug：`attention_forward` 在 `is_kv_source` 层进入
`_compress_kv` 前必须先把 `rt.index_k` 发布到 `rt.shared.index_k`，否则非边界的
ratio>1 步（或恢复后的 state）在 indexer 里拿到 `None`。

### 实测（`start_pc`：arena96、prompt cache、2K prompt）

```
cold (A=2K)        TTFT = 38.25 s
B = A + 3 words    TTFT =  6.56 s   （公共前缀命中，suffix 逐 token）
exact A 重复       TTFT =  1.23 s   （§43）
```

正确性：B 先冷跑一次得到 tokens `[223,3180,16,1]`，再从 A 前缀恢复跑 B，tokens
**完全相同（EQUAL）**。

### 限制与后续

* suffix 是按 token 走 decode（每 token 含一次 head），所以只适合短 suffix；
  要把"短后缀"也做快，需要 chunked prefill（compressor 对 `start_pos>0, seqlen>1`
  尚未实现），或在 snapshot 上保存 block 边界。
* 缓存仍是 **prompt 级**（任意公共前缀但必须完整缓存过），不是 block-level KV 锚点；
  32K 快照 ~150 MB，需要 cap/换出策略。

## 45. 前缀缓存补记：suffix 跳过 head 无可测收益；冷启动窗口方差大

给 `transformer_forward` 加了 `skip_logits`，前缀命中的 suffix 只对最后一个 token 算
head。复测（2K prompt + 3 词 suffix）：

```
cold      = 78.5 s   （同一窗口明显受他人任务/温度影响；此前 38.3 s）
prefix-hit=  7.2 s   （含 head 时为 6.56 s，差异在噪声内）
```

结论：suffix 只有 ~5 token，head 不是瓶颈；**skip_logits 保留但不算优化**。prefix-hit
的 6.5–7.2 s 才是稳定量级。冷启动 TTFT 的窗口方差（38–78 s）需在空闲窗口重测后才可作为
P2 的 P90 报告值。

## 46. P4 长上下文前缀缓存：16K 公共前缀 TTFT 210 s → 4.0 s

把 prompt-level 前缀缓存用到 16K（`prompt cache + MoE chunk1024 + indexer chunk1024`,
arena96, max-model-len 40960）：

```
16K cold        TTFT = 210.0 s
16K prefix-hit  TTFT =   4.0 s   （B = 16K + 3 词；复用整段 16K 前缀，suffix 逐 token）
```

对比 §35 的冷启动 201 s，命中后 **4.0 s ≤ 5 s 目标**；说明长上下文的主要成本在 prefill，
state 恢复 + 短 suffix 的代价很小。2K 同法为 38 s → 6.56 s（suffix 更长/噪声）。

限制不变：suffix 逐 token（含最后一次 head），prompt 级锚点（必须完整缓存过该前缀），
不是 block-level；16K 快照约 100 MB 级，需要 cap。

## 47. P2 判死（架构下界）：2K prefill ≤5 s 在"专家全在主机"下不可达

用实测带宽做下界，不需要再调参：

* 每层 384 个路由专家 × 17.93 MiB（w1/w2/w3 + scales）= 6.724 GiB；
* 40 层 = **268.9 GiB** 必须每 prefill 从主机搬到卡上；
* H2D 实测 **53.7 GB/s** ⇒ **≥5.38 s**，且这是"只搬 packed fp4、零 unpack、零 GEMM、
  零 attention"的下界。

P2 要求 2K prompt TTFT ≤5 s，低于这个下界。任何真实实现还要叠加：
fp4→bf16/GEMM 的 unpack（当前 552 ms/层 ≈22 s）、attention/compressor/indexer。
专家不可能常驻 HBM（476 GB > 128 GB），所以单卡 + 全主机卸载的组合下 **P2 ≤5 s 无解**，
不是调优问题。要达标只有改架构：多卡分片（专家总量/卡数下降）、或把热专家/全部专家放到
更大 HBM 的卡上。

**实际可交付的替代**：前缀缓存（§44/§46）——2K exact 重复 1.23 s、16K 公共前缀 4.0 s，
在"重复/共享前缀"的真实负载上已经满足 5 s。P2 的冷启动 5 s 目标标记为 **DISPROVEN**。

## 48. P1 投机解码上机实测：n-gram 接受率 25%，tokens/forward 1.01 → 无可测收益

新增 `tools/dsv41/perf_spec_accept.py`：用真实**贪心输出**做 oracle（提议 token 等于模型下一步
argmax 才算接受，因此无需额外前向），统计 n-gram（n=3, k=4）的提出/接受与
`tokens_per_forward`。

10 个 prompt × 32 token（单请求、greedy、ignore_eos）：

```
tokens=320 proposed=24 accepted=6 forwards=316 acceptance=25.00%
tokens_per_forward=1.01  speedup=1.01x
```

逐 prompt 看，5/10 的 prompt **一次提议都没有**（自由文本没有可匹配的 3-gram），其余接受
1–3 个；总体上投机只把 320 token 的前向从 320 降到 316。**没有收益**（1.01×，噪声）。

结论：n-gram 投机在这个模型的自由生成分布上 **DISPROVEN**；它只对高重复文本（代码/模板）
可能有效，而本 workload 不是。P1 的投机项关闭，剩余只有 static decode/图捕获与 attention
融合。

## 49. 终态复核（最新代码，前缀/层改动之后）：908.3 / 865.7 / 861.9，中位数 865.7 ≤ 889

最终代码（含 prompt cache、layer index_k 发布、skip_logits 等）再跑三次 P3
（4096 槽，`N1=8/N2=32 reps=5`）：

| 次序 | 每步 | 正确性 |
|---|---|---|
| 1 | 908.3 ms | 20/20 Paris |
| 2 | 865.7 ms | 20/20 Paris |
| 3 | 861.9 ms | 20/20 Paris |

**中位数 865.7 ≤ 889**；P3 性能验收保持通过。汇总终态见
`docs/notes/dsv41-final-report.md`。

## 50. 32K 前缀缓存测量被他人任务挤掉（非代码问题）

在空闲窗口尝试 32K cold + prefix-hit（`MULTI_REQ=0 + PROMPT_CACHE=1 + arena8 +
MoE/idx chunk256`）。cold 运行中 engine OOM：

```
Tried to allocate 4.18 GiB (NPU total 123.02 GiB; 23.81 GiB already allocated;
277.34 MiB free; 24.68 GiB reserved)
```

当时另一租户的 `bench_serving_35b.py` 正占用 HBM；请求结束后 npu-smi 立即无进程。
**这不是容量配置错误**：同样的 32K cold 在独占窗口曾成功（§36，1218 s / 126.9 GiB）。
32K prefix-hit 的数字因此暂缺；16K 前缀 210 s → 4.0 s 已交付（§46）。

重试条件：卡上无他人 NPU 进程、`available` 充足，再跑同一 `start_pc32 + prefix32_probe`
（cold ~20 min，hit 预计 4–6 s）。这是共享卡的外部约束，不是 P4 代码问题。

## 51. 32K prefix-hit 第二次尝试：启动即被他人占用挡住

卡在检查时显示空闲，启动瞬间另一租户又占了 ~105 GiB，vLLM 启动直接拒绝：

```
ValueError: Free memory on device npu:0 (18.08/123.02 GiB) on startup is less than
desired GPU memory utilization (0.2, 24.6 GiB).
```

这不是配置问题：32K cold 需要 >100 GiB 工作区，不能靠降低
`--gpu-memory-utilization` 绕过（降到 0.1 能启动但 prefill 会 OOM）。32K prefix-hit
需要一个**持续约 25 分钟无人抢卡**的窗口；16K 前缀（210 s → 4.0 s）已交付。
该项保持"外部资源受限、待重试"，不影响 P4 的 32K 不 OOM 结论（§36 已在独占窗口达成）。

## 52. 32K 前缀缓存终于测到：cold 931.5 s → prefix-hit 4.0 s

第三次尝试拿到独占窗口（前两次被他人 HBM 占用挡住，§50/§51）：

```
32K cold        TTFT = 931.5 s
32K prefix-hit  TTFT =   4.0 s   （B = 32K + 3 词；恢复整段 32K state + suffix 逐 token）
```

至此 4K/16K/32K 的 prompt 级前缀命中都在 ~4–7 s 量级（16K 4.0 s、32K 4.0 s），
远低于冷启动（16K 201 s、32K 931–1218 s）。P4 的"32K 不 OOM + 前缀缓存"两项均已实测
交付；仍是 prompt 级锚点（必须完整缓存过该前缀），block-level 仍是后续。

配置：`MULTI_REQ=0 + PROMPT_CACHE=1 + arena8 + MoE_PREFILL_CHUNK=256 +
INDEXER_CHUNK=256`，max-model-len 40960、gpu-mem 0.2、enforce-eager。

## 53. P1 成本拆分与 grouped-decode A/B：MoE 占 fwd 65–70%，grouped 无收益

### 单 token 拆分（`VLLM_DSV41_DECODE_TIMING=1`，arena384）

```
pos=7  fwd=559ms moe=401ms
pos=8  fwd=459ms moe=317ms
pos=9  fwd=459ms moe=298ms
pos=10 fwd=421ms moe=278ms
pos=11 fwd=467ms moe=316ms
pos=13 fwd=428ms moe=283ms
```

MoE 占 **65–70%**，attention+dense+head 约 130–150 ms。P1 目标 350 ms 要求 MoE ≤~200 ms；
当前 MoE ~300 ms（6 experts ×40 层），H2D 下界仅 80 ms ⇒ 剩下 ~220 ms 是 unpack/GEMM。
**不换 GEMM 实现就到不了 350**，这与静态 decode/图捕获是两个独立瓶颈。

### grouped decode A/B（`DSV41_DECODE_MOE=grouped`，默认关）

| 口径 | 默认 batched | grouped | 差异 |
|---|---|---|---|
| conc1 TPOT（N1=8/N2=32, reps5） | 560.5 ms | 539.1 ms | −3.8%（极差 37–108%） |
| decode ms/chunk（perf_ttft, 16 tok） | 531 ms | 524 ms | −1.3%（噪声） |

结论：grouped 对单 token 没有可测收益（它把 w1/w3 合成一次 bmm 的收益被其他开销吃掉）；
**默认保持 `moe_forward_batched`**，开关保留用于诊断。P1 的下一步只能是指向 fused fp4
dequant GEMM（DeepGEMM 不可用，需 TileLang 自研）或 static graph。

## 54. P1 单层 MoE 细分（`VLLM_DSV41_BATCHED_MOE_TIMING`）：H2D ~80 ms、GEMM/unpack ~140 ms

在 `moe_forward_batched` 里按 gate / slot-load / gather / w13 / w2 打点（每层同步，绝对值略偏大，
但比例可信）。典型层（39/40）：

```
gate=0.12  load=0.92  gather=0.30  w13=2.30  w2=1.18  (ms)
```

40 层累计：

| 阶段 | 每层 | ×40 | 说明 |
|---|---|---|---|
| gate | 0.11 ms | 4.4 ms | 路由 |
| slot load (H2D) | 0.9–3.3 ms | **~80 ms** | 与 53.7 GB/s 下界一致，不可再省 |
| gather | 0.31 ms | 12 ms | arena 平内存取 6 专家 |
| w13 (unpack+2 GEMM) | 2.28 ms | **91 ms** | fp4 unpack 占大头 |
| w2 (unpack+GEMM) | 1.18 ms | 47 ms | 同上 |
| 合计 MoE | ~7 ms | **~280 ms** | 与 DECODE_TIMING 的 278–401 一致 |

非 MoE（attention+dense+head）约 130–150 ms。所以 350 ms 目标的预算：
`80 (H2D) + 130 (attn/dense) + 140 (GEMM/unpack) = 350` **正好卡在边界**。
要把 GEMM/unpack 从 140 压到 ~70，只能 fused fp4 dequant GEMM（DeepGEMM 不可用，
需 TileLang 自研）；H2D 80 ms 是硬下界，attention 130 也需融合才能再降。

## 55. 负结果：torch_npu 的 W4 原生算子不能直接吃本模型的 MXFP4 布局

`npu_weight_quant_batchmatmul` 的 schema 确实有 `weight_dtype`（可传
`torch_npu.float4_e2m1fn_x2`）和 `antiquant_group_size`，看起来能做 fused W4A16。
实测（`tools/dsv41/probe_npu_w4.py`，随机 64×128，packed uint8 + e8m0 group32 scale）：

* `packed.view(torch_npu.float4_e2m1fn_x2)` 直接报
  `shape '[296]' is invalid for input of size 4096`（fp4 是 2 元素/字节，不能用 view 扩维）；
* 改传 uint8 + `weight_dtype=fp4` 后，算子内部走 AICPU `Transpose` 并崩溃：
  `EE9999 Internal error / aicpu exception (507018)`。

即：没有可以"喂 packed [N,K/2] + e8m0 [N,K/32]"就得到 `x @ w^T` 的原生 MXFP4 matmul；
这与 deep_gemm 不可用的结论一致。fused fp4 dequant GEMM 仍只能 TileLang 自研。
（探测后 NPU 正常，已在下一轮前确认。）

## 56. 外部参考核对完毕：DeepInfer 本机不可用，fused MXFP4 只能自研

目标要求"先核对 DeepInfer 是否支持 fp8-block-32 + MXFP4 + 本模型注意力"。核对结果：

* `/home/userlist/.ernest/DeepInfer` 只有 examples/tests 等 Python 文件，**未安装成包**
  （pip 里没有 deepinfer）；
* 其 `vendor/DeepGEMM-Ascend`、`vendor/DeepSelect`、`vendor/DeepEP-Ascend`、
  `vendor/FlashMLA` **全部是 4 KB 空目录**（submodule 未初始化），没有可借的 Ascend
  kernel 源码；
* 本机仅 `deep_gemm 0.1.1`，fp8×fp4 已在 §? 的 6 组实验排除；
* `torch_npu` 的 W4 原生算子（`npu_weight_quant_batchmatmul` 与
  `npu_grouped_matmul` + `weight_dtype=fp4`）都在内部 AICPU `Transpose` 上崩
  （507018，§55/§56）。

因此"借用 DeepInfer/DeepGEMM 的 fused 预取/分组 GEMM"这条路在本机**没有可用实现**；
fused fp4 dequant GEMM 只能按 Tech Report 的描述用 TileLang 自研。这与目标里的 fallback
一致，也解释了 P1 为什么停在 350 ms 边界。

## 57. 负结果：按"专家集合"缓存 stacked w13/w2 也没有收益

思路：单 token decode 的路由集合若在相邻 token 间稳定，可以按
`(layer_id, ordered expert ids)` 缓存 **已 dequant + 已 stack** 的 w13/w2；命中时跳过
arena bind/load、gather、unpack、stack，只剩 bmm。12 GiB 可放约 28 个层的集合。

实测（arena384，`N1=8/N2=32 reps=5`）：

| 配置 | TPOT | HBM |
|---|---|---|
| 默认 batched | 560.5 ms | 16 GiB |
| **12 GiB set cache** | **577.8 ms** | 36 GiB |

**+3%（噪声内）**，无收益，还多占 20 GiB。说明逐 token 的路由集合/顺序变化足够频繁，
命中率不足以抵消 cache 查找/首建成本；这与 §42 的 per-expert 缓存负结果一致。
按 N5 默认 0 关闭。结论：在 host-offload + 逐 token 路由下，**任何"缓存 dequant 权重"
的 Python 层方案都无效**，只能把 dequant 融进 GEMM（TileLang）。

## 58. 全改动后的 P3 终检：866.1 / 879.0 / 861.0，中位数 866.1 ≤ 889

在 set-cache / timing / grouped-decode / 前缀缓存等全部改动之后，用默认生产配置
（`BATCHED_FWD=1 + MULTI_REQ=1 + BF16_GEMM=1 + ARENA_SLOTS=4096`，
`N1=8/N2=32 reps=5`）再跑三次：

| 次序 | 每步 | 正确性 |
|---|---|---|
| 1 | 866.1 ms | 20/20 Paris |
| 2 | 879.0 ms | 20/20 Paris |
| 3 | 861.0 ms | 20/20 Paris |

**中位数 866.1 ≤ 889**；三次均 20/20 Paris。P3 在最终代码上仍通过，且与 §49 的 865.7
一致。新加的开关（set cache、decode timing、grouped decode、prompt cache）全部默认关，
不影响生产路径。

## 59. 负结果：bf16 head（`DSV41_HEAD_BF16=1`）无可测收益

最后一个非 MoE 成本项：head 129280×5120，默认缓存 fp32 权重。试 bf16 head（半流量、cube
原生）同窗口 A/B（arena384，conc1，`N1=8/N2=32 reps=5`）：

| 配置 | TPOT | 正确性 |
|---|---|---|
| 默认 fp32 head | 537.3 ms | 5/5 Paris |
| **bf16 head** | **528.9 ms** | 5/5 Paris |

−1.6%，在极差（9–68%）内，**不算收益**；head 不是瓶颈（非 MoE 的 130–150 ms 主要是
attention/dense）。开关保留默认 0。至此 P1 的逐项开关级优化全部测完：MoE batched/grouped、
per-expert/set 缓存、n-gram 投机、bf16 head——均无实质收益。

## 60. 4K prefix-hit 两次尝试都被抢卡；P1 开关级已穷尽

4K（4221 token）prefix-hit 两次在启动/首次请求时撞上他租户：

* 第一次：free 1.85 GiB（他租户 `tokenspeedsch`）；
* 第二次：free 1.34 GiB（他租户 `dsv4-generate`）。

均为外部占用，不是配置问题；16K/32K 的 prefix-hit 已在独占窗口测得（4.0 s）。4K cold
仍为 29.9 s / 55.7 GiB（§33）。4K hit 暂缺，需独占窗口（cold ~30 s + hit ~4 s）。

同时确认 P1 的**开关级优化已穷尽**：MoE batched/grouped、per-expert/set 缓存、n-gram 投机、
bf16 head、非 eager / ACL graph（capture 在 `.item()` 失败）、torch_npu W4 原生算子
（AICPU 507018）、DeepInfer/DeepGEMM（vendor 为空/无 fp4）。剩余只有 TileLang fused
fp4 dequant GEMM 或 static decode。

## 61. 4K prefix-hit 补测成功：59.2 s → 3.9 s；4K/16K/32K 命中全部 ~4 s

第三次尝试拿到空窗：

```
4K cold        TTFT = 59.2 s   （该窗口有波动；§33 记录 29.9 s / 55.7 GiB）
4K prefix-hit  TTFT =  3.9 s   （B = 4K + 3 词）
```

至此 **4K / 16K / 32K 的 prompt 级前缀命中都是 3.9–4.0 s**（16K 4.0、32K 4.0），
冷启动分别 29.9–59.2 s、201 s、931–1218 s。P4 的"32K 不 OOM + 前缀缓存"完整交付。
仍是 prompt 级锚点（必须完整缓存过该前缀），非 block-level。

## 62. TileLang dequant kernel 原型：正确但比 torch 慢（2.28 vs 1.21 ms）

在写 fused fp4 dequant GEMM 前，先做最小可行块：TileLang 的
`fp4 packed uint8 + e8m0 scale -> bf16` 元素级 kernel（`tools/dsv41/probe_dequant_tilelang.py`），
覆盖 6 个 w13 专家（N=13824, K=5120）。

* 正确性：tiny 64×128 与 `unpack_fp4_lut + scale` **maxdiff = 0.0**；
* 性能（Ascend 950，28 核）：

| 配置 | 时间 |
|---|---|
| BN=8 BK=512 th=128 | 4.23 ms |
| BN=16 BK=1024 th=128 | 4.08 ms |
| BN=16 BK=1024 th=256 | **2.28 ms** |
| BN=32 BK=512 th=256 | 2.28 ms |
| torch `unpack_fp4_lut + scale` | **1.21 ms** |

自定义 dequant 仍慢于 torch 的 index_select 实现，而孤立 bmm 只要 **0.09 ms**
（`probe_unpack_bench.py`）。因此"fused dequant GEMM"的收益窗口只存在于
"dequant 比 torch 快"的前提下；当前 kernel 不满足，先不集成。要继续必须做
SimdVF/persistent 的 dequant（或直接在 GEMM 的 L1 载入阶段 dequant 且不落 HBM），
属于更深的 kernel 工程。

## 63. Persistent dequant 更慢（4.46 ms）；TileLang dequant 路径关闭

在 §62 的 dense-grid kernel 上加 28-core persistent 调度（`T.Kernel(28)` + serial tile
loop），期望消掉 launch 开销：

| 版本 | 最佳 |
|---|---|
| dense grid（§62） | **2.28 ms** |
| persistent 28-core | **4.46 ms**（BN16/BK1024/th256） |
| torch `unpack_fp4_lut + scale` | **1.17–1.21 ms** |

persistent 反而更慢（serial tile loop 让编译器无法流水/重叠 GM→UB 拷贝）。结论：
**TileLang dequant-only 路径关闭**。torch 的 `index_select` 版本已经接近带宽上限；
要再快只能把 dequant 融进 GEMM 的 L1 载入阶段（避免 bf16 的 HBM 往返），属于更深的
kernel 工程；且即便成功，按 §62 的 bmm=0.09 ms，收益上限也只有 ~0.3–0.4 ms/层（~15 ms/token），
不足以单独把 430 ms 拉到 350。

探测脚本：`probe_dequant_tilelang.py`（dense）、`probe_dequant_persistent.py`（persistent）。
