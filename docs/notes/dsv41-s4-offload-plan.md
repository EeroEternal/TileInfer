# DeepSeek-V4.1-Flash 单卡卸载方案（S4 施工计划）

> 状态：计划稿（2026-11-03，第 19 轮）。前置事实见 [`dsv41-facts.md`](dsv41-facts.md)
> §1（分账）、§17（卸载预算）、§21–§25（加载/对拍的实测数据）。
> 目标：专家权重 + engram 常驻**主机内存**，HBM 稳定在设定水位以下，换入换出不崩（S4 验收）；
> 之后再进 S5（30 分钟混合长度 + 性能曲线 + 同环境基线对比，C1）。

---

## 0. 为什么必须卸载（实测数字，不是估算）

| 项 | 值 | 来源 |
|---|---|---|
| 模型总量 | **475.2 GiB**（510.3 GB） | 48 分片头部求和 |
| 路由专家（40 层 × 384） | **268.9 GiB**（56.6%） | facts §1 |
| engram 两张表（层 1、14） | **189.1 GiB**（39.8%） | facts §1 |
| 其余全部（attn/gate/共享专家/embed/head/vision/MTP 等） | **17.2 GiB**，其中 MTP 专家 7.2 GiB | facts §1 |
| 单个专家 | **18.80 MB**（w1+w2+w3 各 5.90 + 三份 scale 1.10） | facts §1 |
| HBM | 131,072 MB（128 GiB） | `npu-smi` |
| 主机内存 | 628 GB（611 GB 可用） | `free -g` |

结论：把 **465 GiB（专家 + engram）放主机**，HBM 只留 ~10 GiB 稠密 + KV + 激活 + 专家热缓存。

**流量下限**（决定 S5 的 TPOT 门槛）：

| 场景 | 每次搬运量 |
|---|---|
| decode 1 token，全 miss | 40 层 × 6 专家 × 18.80 MB = **4.51 GB** |
| decode 1 token，命中率 38%（110 GiB 缓存） | ≈ **2.8 GB** |
| 长 prefill（每层 384 专家都激活） | 最坏 **288.8 GiB**（整个专家集合过一遍） |
| engram | 2 层 × 24 行 × ~264 B ≈ **12.7 KB/token** |

⇒ `TPOT ≥ 4.51 GB / 有效 H2D 带宽`：50 GB/s → 90 ms/token；100 GB/s → 45 ms/token。
**S5 的"可稳定对话"必须以此为前提设定门槛**，并且必须同时有：专家热缓存、预取（copy/compute overlap）、
batching（batch>1 摊薄每 token 的专家搬运）。

---

## 1. 内存布局

### 1.1 主机侧（一次性分配，进程生命周期内不释放）

```
host_arena/
  experts[layer][expert]  ->  w1 (fp4 packed, 5.90 MB) + w1.scale (e8m0)
                               w2 (5.90 MB) + w2.scale
                               w3 (5.90 MB) + w3.scale
  engram[layer]           ->  embed.weight (rows × 256, fp8) + embed.scale (rows × 8, e8m0)
```

* 复用已经验证过的加载器（`tileinfer/models/dsv41/weights.py`），它已经把 fp4 专家、19 亿行 engram
  表按 checkpoint 的原始 dtype 读进参数树；S4 只是把**这些参数的 device 从 cpu 变成"cpu 常驻 + HBM 槽位"**。
* 建议 pinned host memory（`torch_npu` 的 pinned allocator）以提高 H2D 带宽；**必须先实测**
  pinned vs pageable 的差异（见 §5 第 1 步）。

### 1.2 设备侧（HBM 预算 128 GiB）

| 用途 | 预算 |
|---|---|
| 稠密权重（attn 4.7 + gate 0.16 + 共享专家 1.3 + embed/head 2.5 + norm/hc/engram.wkv ~0.3） | **~9.0 GiB** |
| **engram 表 = 0**（只在查表时搬 24 行/token/layer） | 0 |
| KV（1M 上下文 ≈ 3.4 GiB；64K ≈ 0.2 GiB） | 0.2–3.4 GiB |
| 激活 + workspace + 双缓冲 | ~3 GiB |
| **专家热缓存（可调水位）** | **≤ 110 GiB** |

水位：`TILEINFER_DSV41_HBM_WATERMARK_GIB`（默认 110），超水位时按 LRU 逐出、**不得**逐出正在飞行中的块。

---

## 2. 专家换入换出（核心机制）

### 2.1 数据结构

```python
class ExpertCache:
    slots: dict[(layer, expert)] -> slot_index          # 常驻 HBM 的专家
    free:  list[int]                                     # 空槽（预分配）
    lru:   OrderedDict[(layer, expert), None]
    inflight: set[(layer, expert)]                       # 正在 H2D 的块，禁止逐出
    slot_bytes = 3 * 5.90 MB (weights) + 1.10 MB (scales) = 18.80 MB
```

* 槽位在启动时按水位**预分配**（`torch.empty` 一次，避免碎片与 OOM），不动态申请。
* 键是 `(layer, expert)`；同一层同一专家在多 token/batch 中复用（decode 的命中率主要来自这里）。

### 2.2 一次 decode 的流程（单层）

```
gate(x) -> 6 个 expert id
for e in ids:
    if (layer, e) in cache:  slot = cache.slot((layer,e)); hit += 1
    else:
        slot = cache.evict_one()            # LRU，跳过 inflight
        enqueue H2D copy (host_arena -> slot)   # 异步流
for e in ids: 绑定 slot 上的权重 -> fp4 GEMM（等待对应 copy 完成）
```

* **双缓冲**：第 k 个专家的 copy 与第 k-1 个专家的 GEMM 重叠（`torch_npu` 的 stream + event）。
* 单请求 decode 每层只有 6 个专家 → 每层一个 copy 批次（6 × 18.8 = 113 MB），40 层约 4.5 GB。
* 合并搬运：把 6 个专家的 w1 连续拼成一个大块（host 侧预排布），减少 H2D 小包开销。

### 2.3 prefill

prefill 每层最多 384 个专家 → 不逐一换入，**整层扫描**：把该层专家（384 × 18.8 MB = 7.22 GB）
按批（例如每批 32 个专家 = 600 MB）搬入、算完即弃，HBM 峰值只占一个批次。
代价是最坏 288.8 GiB 的全量搬运，但 prefill 是一次性的；S5 需要把这条路径单独计时（TTFT）。

---

## 3. engram 换入换出

* 查表 id 在 `embed` 之后即可算出（`EngramHash.forward`，与权重无关）→ **可以在第 1 层开始前就预取**
  第 1 层与第 14 层的 24 个 id。
* 每个 id 取 256 B（fp8 行）+ 8 B（scale）；24 个 = 6.4 KB/layer/token。
* 主机侧用 `torch.index_select` 取出 24 行（在 CPU 上做）→ 6.4 KB H2D（异步，与前面的层重叠）。
* **不要**把 h2d 的粒度做到单行（24 次 256 B 的拷贝开销远大于一次 6.4 KB）。

---

## 4. 正确性红线（S4 验收要覆盖的失败模式）

| 失败模式 | 要求 |
|---|---|
| 逐出正在 H2D 的块 | 禁止（inflight 集合） |
| 槽位不足时死锁 | 逐出 LRU；若全部 inflight 则等待最早的事件 |
| H2D 与 GEMM 竞争同一 slot | 每 slot 一个 event，GEMM 前 `wait_event` |
| 超水位 | 触发逐出直到水位以下，并把 `hbm_watermark_events` 计数暴露出来 |
| 专家 id 越界 / 未分配 | 启动时断言 `n_routed_experts`、`n_local_experts` 与 config 一致 |
| engram 行越界 | `hash < rows`（S0 已验证 `sum(primes) == rows`，加断言即可） |

数值红线：**与 CPU oracle 的贪心首 token 一致**（已验证为 `11111 ' Paris'`），
逐层 hidden 允许 1–2 bf16 ULP 的归约差异（facts §25.12）。

---

## 5. 施工顺序（每步都有可复现脚本与判据）

1. **实测 H2D 带宽**（pinned / pageable、1 MB / 32 MB / 128 MB 块）→ `tools/dsv41/s4_h2d_bench.py`。
   判据：得到 GB/s 曲线，作为 TPOT 下界公式的输入。

   **已实测（2026-11-03，`torch_npu 2.10.0.post4`，卡上有他方服务占 95 GB 时的小块基准）**：

   | 传输块 | pinned | pageable |
   |---:|---:|---:|
   | 1 MB | 31.5 GB/s | 10.1 GB/s |
   | 4 MB | 47.7 GB/s | 9.8 GB/s |
   | 32 MB | 53.0 GB/s | 13.9 GB/s |
   | 128 MB | 53.4 GB/s | 14.7 GB/s |
   | 512 MB | **53.7 GB/s** | 14.0 GB/s |

   ⇒ **必须用 pinned**（3.8×）：小块（1 MB）只有 31 GB/s，4 MB 以上饱和到 ~54 GB/s，
   所以**合并搬运**（把 6 个专家拼一大块 ≈ 113 MB）能拿到满带宽。
   代入流量公式：`4.51 GB ÷ 53.7 GB/s ≈ 84 ms/token`（全 miss，≈12 tok/s）；
   命中率 38% 时 `2.8 GB → 52 ms`（≈19 tok/s）。**这就是 S5 TPOT 的物理下界基线。**

2. **专家槽位缓存 + 单层验证**：单层 6 专家换入 → 与"专家常驻"结果逐位一致；
   测量命中/未命中路径的耗时。
3. **整栈 decode（1 token）**：40 层 + 换入换出，HBM 峰值 ≤ 水位，输出 token == 11111。
4. **整栈 prefill**：按批扫描 384 专家，记录 TTFT 与 HBM 峰值。
5. **长上下文**：KV 1M 上下文（3.4 GiB）+ 卸载，确认不 OOM、token 不串。
6. **稳定性**：30 分钟混合长度请求，记录 TTFT/TPOT/HBM/RSS 曲线（S5）。
7. **基线对比**：与同环境（同 venv、同 torch/torch_npu/vLLM/CANN/tilelang、同启动参数，C1）
   的基线臂对比；**两臂不同时占卡**（C2）。

---

## 5.1 第 3 步（专家换入）实测：批量 arena 跑满带宽

脚本：`tools/dsv41/s4_page_in.py`（逐张量拷）、`tools/dsv41/s4_batched_h2d.py`（单块 arena）。
两者都先加载**真实 layer 0**（`copied=2333 / 7.4 GB / missing=extra=shape=dtype=0`），并在加载不干净时
**直接拒绝继续**（否则会拿未初始化内存比出"假的 0 误差"——踩过一次）。

| 项 | 实测 |
|---|---|
| 单专家 `w1/w2/w3 + scales` | 17.9 MiB |
| 逐张量 `.to(dev)`（36 次） | 10.4 ms（0.29 ms/次），折算 ≈1.7 GB/s |
| **合并成 107.6 MiB 单块 H2D** | **2.11 ms = 53.6 GB/s**（= 第 1 步 53.7 GB/s 上界） |
| 主机侧拼块（pinned arena 填充） | 3.7 ms / 6 专家（可异步，与上一层计算重叠） |
| arena 一次性预分配 | 1823 ms（启动期成本，不计入 TPOT） |
| 数值 | 从设备 arena 按偏移切片还原 `w1` 后与 CPU 参考比：5 个专家 `0.000e+00`，1 个 `4.883e-04`（≤1 bf16 ULP） |

推论：**必须批量**（4.9×）；扁平 arena + 偏移切片（`view(uint8) → view(float4_e2m1fn_x2)` /
`view(float8_e8m0fnu)`）是正确布局，不需要每专家独立张量；外推 40 层 `40 × 2.11 ms ≈ 84 ms/token`，
与 `4.51 GB ÷ 53.7 GB/s` 的下界公式**交叉验证一致**。


### 5.2 第 3c 步进展：arena 布局与设备验证通过，`moe_forward_streamed` 尚差 9.3

**已通过（真实 layer 0）**

| 项 | 结果 |
|---|---|
| `ExpertArena`（`tileinfer/models/dsv41/expert_stream.py`） | 6 个专家打包 → 一次 H2D → 偏移切片：`w1` 与 CPU 参考 **0.000e+00** |
| 单槽字节 | 17.93 MiB × 6 = 107.6 MiB；`pack` 主机侧 3.0 ms（≈36 GB/s）；整块 H2D **2.53 ms / 44.5 GB/s** |
| 6 个 CI 用例（`tests/test_dsv41_expert_stream.py`） | 偏移无空洞、单槽往返、双槽独立、LRU 逐出、inflight 保护、尺寸不符必报错 |
| 单层 MoE：`paged` vs `resident` | **bit-exact = True**（两者相对 CPU 参考同为 7.588e-02，即设备 fp4 GEMM 的固有精度） |
| 时序 | resident 逐专家 `.to(dev)` 一次性 1.8 s → 摊薄后每层 `pack + H2D ≈ 4 ms`（含暖跑 11.6 ms） |

**尚未通过**：把上述路径封装成 `block.py::moe_forward_streamed()` 后，与 CPU `moe_forward()`
相比 `max|diff| = 9.34`、`mean = 1.49`（而内联的 paged 路径是 bit-exact）。试过两种权重位置
（w2 前 / w2 后）都在 9.3 附近，说明**不是**乘法顺序问题。

线索：`moe_forward()` 含共享专家，其贡献本身是 `max|diff| 27.88`；9.3 既不是"漏加"也不是"多加"。
下一步最有效的调试方式：对**单个 token、单个专家**打印两边的中间量（`gate/up`、`silu*up`、
`w2` 输出、`weights[idx, top]`），逐项对照，而不是继续在整层输出上猜。


### 5.3 逐专家隔离：设备数学与 `expert_forward()` 一致，残差只在"共享专家/累加"这一步

同一层、同一批 token、同一组真实权重，**逐专家**（不加路由权重）比较"设备 arena 视图"与
CPU `expert_forward()`：

```
expert  13: max|diff| = 1.192e-07      expert 181: 0.000e+00
expert 146: 0.000e+00                  expert 185: 3.906e-03
expert  14: 0.000e+00                  expert  54: 0.000e+00
```

⇒ **arena 视图 + 设备 fp4 GEMM 的逐专家数学是正确的**（4 个专家逐位相同，2 个 ≤4e-3，即 1 ulp 级）。

因此 `moe_forward_streamed` 与 `moe_forward` 的 9.336 残差**不在专家数学里**，只剩两处：
1. 逐专家结果的**散射累加**（`y[idx] += ...`）；
2. **共享专家** `expert_forward(moe.shared_experts, flat)` 的拼接。
参考量级：共享专家单独贡献 `max|diff| 27.88`，而 9.336 既不是 0 也不是 27.88，说明是其中一步的
输入/累加口径不同（下一步只需对**单个 token**打印两条路径的 `y_routed`、`y_shared` 各一次即可定位）。


### 5.4 残差细分：共享专家相同、逐专家数学相同，问题在**路由部分的量级**

```
per-expert (device arena view vs expert_forward):  13: 1.2e-07  146/14/181/54: 0.0  185: 3.9e-03
shared part magnitude:      cpu 2.112e+01   streamed 2.112e+01      ← 完全相同
routed part magnitude:      cpu 1.819e+01   streamed 1.338e+01      ← 差 26%
row-wise |streamed - cpu|:  [7.6875, 4.7812, 9.3359, 7.8271]
```

三条硬事实：
1. **逐专家数学正确**（4 个逐位、2 个 ≤1 ulp）；
2. **共享专家两条路径完全一致**（2.112e+01）；
3. 但**路由部分的量级**（不含共享专家）差 26%，且四个 token 都有 5–9 的偏差。

⇒ 不是数值精度问题，而是**累加口径**问题：路由和里少了/错配了某一份贡献。已排除：
权重乘法位置（w2 前 9.305 / w2 后 9.336，几乎不变）、`idx` 是否搬到设备（改成 `idx.to(xd.device)`
后数字完全不变）。

下一步（精确到打印，10 分钟）：对**一个专家**（如 13）打印两条路径的
`idx`、`top`、`weights[idx, top]`、`out`（w2 之后）与 `y[idx]` 的贡献，逐项 diff。
最可能的候选：`moe_forward` 里 `weights[idx, top, None]` 的广播形状 [n,1] 与我的 [n,1] 不同源，
或某个专家的 `slots[e]` 与 `slot_of` 不一致（尽管逐专家检查用的是同一批 views）。


### 5.5 设备侧整块试验被"卡被他方占满"打断（外部依赖，非代码问题）

`tools/dsv41/s4_block_stream_device.py` 已把稠密权重（attn/norm/hc/gate/共享专家，**6.91 GiB**）搬到卡上、
路由专家留在主机、经 arena 换入——正是 S4 目标形态。脚本开跑时 `npu-smi` 报 **122.5 GiB free**，但真正分配时
崩溃：

```
torch.OutOfMemoryError: NPU out of memory. Tried to allocate 20.00 MiB
  (123.02 GiB total capacity; 2.98 GiB already allocated; 16.54 MiB free; 3.49 GiB reserved)
```

⇒ 在他方进程占用下，卡上可用显存从 122 GiB 掉到 **16 MiB**。S4 后续每一步（整块、整栈 decode、prefill、
S5 曲线）都需要 ~15 GiB 以上的稳定可用 HBM，因此**必须先按 C2 与占用方协调**，或改用显存更小的形态
（例如只跑单层 MoE 段、把 attention 留在 CPU）。

已修完的、与显存无关的问题（本轮）：`moe_forward_streamed` 里 gate 在设备上跑时，
`torch.where(indices == e)` 的索引与 `y[idx] +=` 的散射必须留在 CPU（三处 device 边界已修）；
`block_forward` 新增 `moe_fn` 钩子，可在不改其它代码的前提下替换 MoE 步骤。


### 5.6 整栈设备运行 + 连续 decode 实测（S4 收口 / S5 起点）

```
prompt 'The capital of France is' -> [671, 6102, 294, 8760, 344]
loaded in 73s: 93378 张量 / 502.7 GB / missing=extra=shape=dtype=0
on card 10.42 GiB 稠密 | on host 457.78 GiB（专家 268.9 + engram 189.1）
HBM free after placement 111.9 GiB；arena 96 槽 = 1.68 GiB
prefill 2.6-3.9 s → token = 11111 (' Paris')，top5 首位 21.627
decode 8 步：TPOT 765 ms（1.3 tok/s），HBM 稳定 105 GiB free，无 OOM
arena: misses 796 / evictions 700 / failed 0
```

**一个关键的性能修复**：第一版每次换入都把**整个 arena**（1.68 GiB）拷一遍，每层一次 →
67 GiB/token，TPOT 1962 ms。改成 `to_device_slots()` **只拷本层刚填的几个槽**后 → **765 ms**（2.6×）。

**剩下的差距（765 ms vs H2D 下界 84 ms）**已定位：不是搬运，而是每次 fp4 GEMM 都要
`unpack_fp4` 把 5.9M 元素解成 bf16（每 token 240 次专家 × 3 个 GEMM），属**算子实现问题**，
需要 TileLang kernel 才能消掉——这是 S5 之后的性能工作，不是卸载机制的问题。

**S4 判定：达成**。卸载生效（10.42 GiB 稠密常驻 / 457.78 GiB 主机）、换入换出不崩
（failed=0，700 次逐出）、HBM 稳定在 105 GiB free、数值与 CPU oracle 一致（token 11111）。

**S5 起点数据**：TPOT 765 ms、prefill 2.6–3.9 s、HBM 105 GiB free。仍需：30 分钟混合长度、
TTFT/TPOT/HBM/RSS 曲线、同环境基线对比（C1/C2）。


### 5.7 长度扩展与分组建页（S5 起点）

预填长度从 5 增到 16 个 token，专家扇出随之增大，因此把 `moe_forward_streamed` 改成**分组建页**：
每层要用的专家按 `arena.n_slots` 切组，逐组 pack → 只拷本组槽位 → 计算 → `cache.drop()` 归还槽位。

| 预填 token | 前向 | 建页次数 | 搬运 | TPOT | HBM free |
|---|---|---|---|---|---|
| 5 | 2.6–3.9 s | 796 | 13.9 GiB | 765 ms | 105.3 GiB |
| 16 | 3.6 s | 1333 | 23.3 GiB | 721 ms | 105.4 GiB |

16 token 的实际输出：`' The capital of'`（连续、语义连贯的英文）——**"正常出字"达成**；
`evictions=0 / failed=0 / free_slots=96`（组用完即还），HBM 恒定 105 GiB free。

结论：预填成本近似线性于"不同专家的个数"，单 token decode 约 0.72 s；长预填（数百 token）会触及
每层 384 个专家的上限（≈6.9 GiB/层 × 40 层），需要更大 arena 或预填分块，这是 S5 剩余工作的核心。


### 5.8 30 分钟混合长度稳定性（S5 验收主项）——通过

`tools/dsv41/s5_soak.py --minutes 30`，一次加载、稠密常驻卡上、专家与 engram 全程主机内存：

```
requests = 276   corrupt = 0   arena.failed_acquires = 0
TTFT  1853–6768 ms（均值 3410，随 prompt 长度 5→42 token 增长）
TPOT  686–2961 ms（均值约 730；>1.2 s 的尖峰只出现在他方进程抢卡时）
HBM used 17.7–121.8 GiB（中位 17.7；121.8 是他方进程占用，不是本进程）
RSS   460.4–460.6 GiB（全程平坦，无泄漏）
累计搬运 11.9 TiB / 660,882 次建页，evictions 0（每组用完即还）
RESULT: OK
```

**结论**：
1. **无 OOM、无 token 损坏**（`corrupt=0`：固定 prompt 的 `11111` 每次都对）、**无换入失败**；
2. **RSS 完全平坦**（460.4–460.6 GiB，30 分钟内漂移 0.2 GiB）→ 没有泄漏；
3. **HBM 稳态 17.7 GiB used / 105 GiB free**，远低于水位；
4. 唯一的 TPOT 尖峰（2.96 s）与 HBM 121.8 GiB 同刻出现，是**他方进程抢占单卡**造成的（C2 禁止两臂同时占卡）。

**基线对比（C1 口径下的可行部分）**：同代码同权重的 CPU oracle 为 140 s/token，本实现 0.73 s/token
（**约 190×**）；H2D 下界为 `4.51 GB ÷ 53.7 GB/s ≈ 84 ms`，当前 730 ms 的差距来自 fp4 unpack + 小 GEMM
派发（算子实现，需 TileLang kernel），已在 §5.6 定位。


### 5.9 性能归因（S5）：unpack 占单次 GEMM 的 93%

`tools/dsv41/s5_profile_decode.py` 在卡上实测（真实 fp4 专家权重 2304x5120）：

| 项 | 耗时 | 备注 |
|---|---|---|
| H2D 单个 17.93 MiB 槽位 | **0.348 ms**（50.3 GB/s） | 已达 53.7 GB/s 上界的 94% |
| `unpack_fp4`（5.9M 值） | **0.239 ms** | 反量化 |
| `linear(packed w1)` | 0.402 ms | 含 unpack |
| `linear(预解包 w1)` | 0.027 ms | 纯 GEMM |
| unpack 占比 | **93%** | 瓶颈所在 |

外推：每层 6 专家 x（3 GEMM + 1 槽位）= 9.3 ms → 40 层 373 ms；实测端到端 730 ms，
差额约 350 ms 是 per-layer Python 派发 + attention/compressor/indexer/gate。

**两条可选的优化路径（都已量化）**
1. **主机侧反量化成 bf16 再分页**：每专家 `0.72 ms(H2D 35.9 MiB) + 3x0.027 = 0.80 ms`
   vs 现在 `0.35 + 3x0.402 = 1.55 ms` → **约 2x**；代价是 arena 与搬运量翻倍（水位需重算）。
2. **TileLang 融合反量化 GEMM**：把 0.239 ms 的 unpack 消进 GEMM，理论可把 373 ms 压到 ~30–60 ms，
   逼近 84 ms 的 H2D 下界；这是 S5 之后的主要性能工作。

**当前判定（按目标口径）**：单请求正常出字 ✓、长上下文不 OOM ✓（1M → 13.27 GiB）、
30 分钟混合长度可稳定对话 ✓（276 请求 / 零失败 / RSS 平坦）。性能仍为 0.73 s/token，
若要求交互级速度（>10 tok/s）需走上面第 2 条路径。


### 5.10 路径 1 被实测否决；瓶颈已一分为二

**路径 1（主机侧反量化成 bf16 再分页）实测**：

```
host unpack          18.076 ms/expert   ← CPU 上解 5.9M 值，比卡上慢 75x
H2D 67.5 MiB          1.255 ms
per layer 6 experts   8.0 ms -> 40 layers 321 ms   （fp4 分页：355 ms）
```

⇒ 只快 10%，**否决**。原因：主机侧 unpack 太慢（18 ms），且搬运量翻倍（17.9→67.5 MiB）。
唯一有效的是**路径 2：TileLang 融合反量化 GEMM**（把卡上的 0.254 ms unpack 并进 GEMM）。

**同时得到一个更重要的分解**：专家路径外推 355 ms，而实测 TPOT 730 ms ⇒ **约 375 ms（51%）
是 attention/compressor/indexer/gate + 每层 Python 派发**，任何分页优化都碰不到这部分。

因此要把 TPOT 压到交互级，必须同时做两件事：
1. 融合反量化 GEMM（吃掉 355 ms 里的 ~93%）；
2. 减少 per-layer 派发（例如把 40 层循环里的多次小算子合并 / 用图执行）。


### 5.11 专家缓存跨 token 复用：命中率 19.3%，TPOT 765 → 683 ms

把 arena 开成"整层容得下"（400 槽 = 7.0 GiB），并在扇出不超过槽数时**不再归还槽位**
（`if len(groups) > 1` 才 drop），使同一个 prompt 的后续 token 能命中上一轮装进来的专家：

```
slots=400, arena 7.0 GiB, HBM free 100.0 GiB
hits=431 / misses=1805 → hit_rate 19.3%
TPOT mean 683 ms（96 槽、用完即还时 765 ms，-11%）
token 仍为 11111（数值不受影响），failed_acquires=0
```

结论：跨 token 复用**确实生效**（19.3% 命中），但收益有限（-11%）——因为单 token 只有
40x6=240 次访问、而不同 prompt 的专家集合不同（soak 里混长度请求会互相冲掉）。
要拿到更大收益要么按会话固定专家集（命中率可到 ~90%），要么走 §5.10 的算子路径。


### 5.12 逐段计时：MoE 占 decode 的 **75%**（我此前把派发估高了）

给 `block_forward`/`transformer_forward` 加了 `timings` 钩子（仅计时，不影响数值），实测 4 个 decode 步：

```
step 1: moe 624 ms of 823 ms    step 3: moe 621 ms of 800 ms
step 2: moe 507 ms of 680 ms    step 4: moe 571 ms of 757 ms
TPOT mean 765 ms；token 仍正确，RESULT: OK
```

⇒ **MoE 占 75%（≈570 ms），attention/compressor/indexer/gate 等合计约 25%（≈195 ms）**。
结合 §5.9（GEMM 时间的 93% 是 `unpack_fp4`），得到可执行的优化预算：

| 优化项 | 当前 | 优化后（估计） | 手段 |
|---|---|---|---|
| MoE 里的 unpack | ≈480 ms | ≈30 ms | TileLang 融合反量化 GEMM |
| 纯 GEMM + 搬运 | ≈90 ms | ≈90 ms | 已接近带宽/算力 |
| attention 等 | ≈195 ms | ≈195 ms | 后续再做（需要 per-kernel 分析） |
| **TPOT** | **765 ms** | **≈300 ms** | |

这修正了 §5.10 里"51% 是派发"的粗估：派发不是主因，**MoE 的 unpack 才是**。


### 5.13 直连拷贝 -8%；新瓶颈是**小张量派发延迟**，不是带宽也不是 unpack

把"先 pack 进 pinned arena 再整块 H2D"改成 `ExpertArena.load_slot()` **从主机张量直接拷进设备槽位**
（省掉每层 108 MiB 的主机内 memcpy）：

```
TPOT 765 -> 704 ms（-8%），MoE 570 -> 539 ms，token 仍 11111，RESULT: OK
```

但收益有限，说明 pack 不是主因。把账算清（每 token、40 层、每层 6 专家）：

| 项 | 实测/推算 |
|---|---|
| 6 次槽位拷贝（6 x 0.35 ms） | ≈ 2.1 ms/层 → 84 ms/token |
| 6 专家 x 3 次 GEMM（含 unpack 0.09 + GEMM 0.027） | ≈ 2.3 ms/层 → 92 ms/token |
| **实测 MoE** | **13.5 ms/层 → 539 ms/token** |
| 差额 ≈ 1.5 ms/专家 | **小张量逐算子派发延迟**（每个专家约 8 个小算子，1 token 的张量上每次 ~0.2 ms） |

另外单独实测：`unpack_fp4` 只需 **0.091 ms**（此前 0.254 ms 是含首次调用的暖机），用字节级 LUT
（`unpack_fp4_lut`，**逐位等价**）也还是 0.092 ms —— **unpack 不是瓶颈，此前的归因偏了**。

**结论（修正后的优化路线）**：剩下的大头是**每专家 8 个小算子的派发**，正确解法是把 6 个专家
**批量执行**：arena 的槽位布局是统一的，可以把 6 个槽位看成 `[6, ...]` 张量，
用 `baddbmm`/grouped GEMM 一次算完 6 个专家的 w1/w3/w2，把 48 次派发压到 8 次（预计 TPOT → ~250 ms）。
这一条不需要 TileLang，是下一步性价比最高的改动。


### 5.14 批量 decode MoE 已实现且**逐位等价**；TPOT 复测被"机器被他人占用"打断

**实现**（`block.py::moe_forward_batched` + `_fp4_linear_batched`）：decode 单 token 时，把该 token 的
6 个专家在设备上 gather 成 `[6, ...]`，三次 GEMM 各用**一次 `bmm`** 完成（unpack 走字节级
`unpack_fp4_lut`，块 scale 用 `repeat_interleave(32)` 批量展开）。

**等价性（真实 layer 0、真实权重、设备上）**：

```
batched vs moe_forward: max|diff| = 0.000e+00   mean = 0.000e+00     ← 逐位相同
arena: misses=6, fails=0
```

**TPOT 复测未完成**：`--batched` 的整栈运行在**加载阶段**被 OOM killer 终止（`anon-rss 367 GB` 时被杀），
因为此刻机器上有**他人负载**：主机已用 276 GB（剩余 350 GB，装不下 458 GB 的模型），
卡上也被占 **121,857 / 131,072 MB**。日志里可见两次 `Out of memory: Killed process`。

⇒ 这不是代码问题（CPU 侧调用 `moe_forward_batched` 与 `moe_forward` 的等价性、以及设备侧
逐位一致都已验证）。**下一步**：机器空闲时重跑
`s4_full_stack_device.py --gen 5 --slots 400 --batched`，对比 704 ms 的基线；按 §5.13 的分析预期
MoE 从 ~540 ms 降到 ~150-200 ms。


### 5.15 批量 decode MoE 实测：TPOT 704 → **619 ms**（-12%）；剩余成本在 **scale 展开**

机器空闲后跑通 `--batched`（真实整栈、真实权重）：

```
token = 11111 (' Paris')      RESULT: OK
step 1: moe 475 ms of 642 ms | step 2: moe 421 of 579 | step 3: moe 487 of 659 | step 4: moe 437 of 595
TPOT mean 619 ms (1.6 tok/s)      （非批量路径 704 ms；最初 765 ms）
```

**优化曲线**：765（arena 整块拷贝）→ 704（`load_slot` 直连）→ **619 ms（6 专家批量 `bmm`）**。

**为什么只 -12% 而不是预期的 2.5x**：批量只消掉了"派发次数"，没消掉**数据量**。逐项看
每 token 每层仍要做：
* 3 个权重各 `unpack` 6 份（6x2.95M 字节 gather）；
* **块 scale 展开**：`scale_to_float(scale).repeat_interleave(32)` 要在 `[6, 2304, 5120]`（70M 元素）
  上物化一次再相乘 —— 这与逐专家版本的总工作量相同（6 x 11.8M），是本轮剩下的最大单项；
* 3 次 `bmm`（真正的计算，很小）。

⇒ 要把 TPOT 再往 84 ms 的下界压，**必须让 scale 与 unpack 在 GEMM 内部完成**（TileLang 融合
反量化 GEMM）。这条结论现在有三次独立测量支撑（§5.9 unpack、§5.13 派发、本条 scale 展开）。


### 5.16 用广播代替 scale 展开：TPOT 619 → **566 ms**

`_fp4_linear_batched` 里原来是 `w * scale.repeat_interleave(32, dim=-1)` —— `repeat_interleave` 会
**物化** `[6, 2304, 5120]` 的 fp32 张量（约 280 MB）再相乘。改成把权重 reshape 成
`[k, out, in//32, 32]` 后用 `scale.unsqueeze(-1)` **广播**相乘（数值完全相同，省掉 280 MB 的分配与回读）：

```
token = 11111 (' Paris')   RESULT: OK
step 2: moe 356 ms of 497 ms   step 3: moe 424 of 564   step 4: moe 404 of 549
TPOT mean 566 ms (1.8 tok/s)
```

**完整优化曲线**：765 → 704（直连槽位拷贝）→ 619（6 专家批量 bmm）→ **566 ms（广播 scale）**，
累计 **-26%**。剩余单项仍是 `unpack` + 逐块 scale 的乘加本身（合计仍要在 70M 元素上走一遍），
要再往下压必须把它们放进融合 kernel（TileLang）。


### 5.17 批量路径的稳定性复检（8 分钟 soak）

`tools/dsv41/s5_soak.py --minutes 8 --batched`（decode 走批量、prefill 走流式）：

```
requests = 81   corrupt = 0   arena.failed_acquires = 0
TTFT  1912-7097 ms（均值 3560）
TPOT  566-1082 ms（中位约 580；尾峰与 HBM 116.5 GiB 同刻，是他方抢占）
HBM   稳态 17.7 GiB used / 105.3 GiB free
RSS   460.6-460.6 GiB（完全平坦）
建页 194,036 次 / eviction 193,940（96 槽按层换出）/ 零失败
RESULT: OK
```

与非批量 30 分钟 soak（TPOT 均值 ~730）相比：**TPOT 均值 ~580，约 20% 提升**，稳定性指标不变。
输出文本连贯（`' The capital of Italy'`、`' The capital of France'`）。


### 5.18 批量 GEMM 三相剖析：unpack+scale 只占 0.17 s，剩下的在 gather/派发

`tools/dsv41/s5_profile_batched_gemm.py`（k=6、真实形状 2304x5120，无需加载模型）：

| 阶段 | 耗时 | 占单次 GEMM |
|---|---|---|
| `unpack_fp4_lut`（70.8M 值） | 0.80 ms | 57% |
| 块 scale 广播乘 | 0.42 ms | 30% |
| `bmm`（fp32 输入） | **0.18 ms** | 13% |
| 合计 | **1.40 ms** | |

外推 3 GEMM x 40 层 = **0.17 s/token**，而实测 MoE 为 0.40–0.49 s ⇒ **0.23–0.32 s 不在这三个 GEMM 里**。
结合 §5.13，这部分是每层的 `gather`（6 次 stack x 3 权重 + 3 scale）、`act_quant`、`silu/权重乘`、
以及 Python 派发。

**结论（修正 §5.15/§5.16 的表述）**：下一步不该再优化 unpack，而应
1. 把 6 个专家的 6 个 part 直接**在 H2D 时堆成 `[6, ...]`**（省掉 6 次设备侧 stack）；
2. 每层只做一次 `act_quant`（当前 3 次 GEMM 各做一次）；
3. 把若干小算子（silu、乘权重、sum）合并。

顺带记录两个 NPU 限制：`repeat()` 不支持 `float4_e2m1fn_x2` / `float8_e8m0fnu`，
必须 `.view(torch.uint8)` 后再 repeat、再 view 回来。


### 5.19 单请求 decode 的实际地板：≈400 ms，不是 84 ms

把 566 ms 拆到"不可避免"与"可优化"两栏（每 token，40 层）：

| 项 | 现状 | 说明 | 可否消除 |
|---|---|---|---|
| 专家 H2D | ≈88 ms | 240 专家 x 17.93 MiB = 4.3 GiB @ 50 GB/s | 否（缓存命中率只有 ~19%） |
| gather 设备侧 stack | ≈16 ms | 6 次/层 x ~214 MB | 可（H2D 时直接堆叠） |
| unpack + scale | ≈170 ms | 3 GEMM x 40 层 x (0.80 + 0.42) | 需融合 kernel |
| bmm | ≈24 ms | 3 x 40 x 0.18 | 否（已是纯算力） |
| attention / compressor / indexer / gate | ≈150 ms | 每层 ~3.7 ms | 需 per-kernel 分析 |
| 其他派发 | ≈120 ms | 每层若干小算子 | 部分可合并 |

⇒ **设计层面的地板**：`88（H2D）+ 24（bmm）+ 150（attention）≈ 260 ms`；
若再用融合 kernel 消掉 unpack/scale 的 170 ms 与 gather 的 16 ms，单请求 decode 约落在
**260–300 ms**（约 3.5 tok/s），这是"6 个专家零复用 + 单张卡 + 主机内存换入"这一形态的物理极限。

**§5.1 的 84 ms 是纯 H2D 下界、不是可达到的 TPOT**——它假设 unpack/attention 都不存在。
63 轮前把它当作目标是个记录错误，这里更正。

结论：TPOT 从最初 765 ms 降到 **566 ms** 已经把"派发与数据布局"这层挖干净；
再往下的收益必须来自 (a) 融合反量化 kernel、(b) attention 侧 per-kernel 优化。


### 5.20 按 `perf-comparison-rules.md` 重测 TPOT（差分法）+ 同环境 A/B

规则要求 `tpot = (T(N2)-T(N1))/(N2-N1)`、N1=8/N2=72、同一 prompt、非流式、多次取中位数，
并明确否定逐 token 计时。`tools/dsv41/s5_wall_tpot.py` 按此实现（含极差、环境头、非物理解拒收）。

| 臂 | T(8) | T(72) | **TPOT** | 极差 |
|---|---|---|---|---|
| A 批量 decode（现状） | 6.167 s | 39.570 s | **522 ms/token**(1.92 tok/s) | 0.8% |
| B 逐专家流式 | 7.390 s | 49.531 s | **658 ms/token**(1.52 tok/s) | 0.4% |

* 两臂同一 venv（torch 2.10.0+cpu / torch_npu 2.10.0.post4 / vllm 0.23.0），只差 MoE 一步
  （`--streaming` 开关），满足 C1；极差 <1%，远低于 10% 噪声门槛；
* 批量路径比流式快 **26%**；此前逐 token 计时的 566 ms 与差分法的 544/522 ms 相差 4–8%，
  说明旧数字只是"指示值"，**结论以 522 ms 为准**。


### 5.21 S2 判据达成：设备整块 vs CPU 参考 **7.812e-03 < 1e-2**

用 `s4_block_stream_device.py`（稠密常驻卡上、路由专家经 arena 换入）复测：

```
moved 6.91 GiB dense to the card; routed experts stay on the host
streamed block (device) vs CPU reference: max|diff| = 7.812e-03  mean = 2.146e-07
pre_mix: max|diff| = 2.980e-07      RESULT: OK
```

⇒ S2 的验收门槛（设备对 CPU、bf16、`max|diff| < 1e-2`）在**整块**口径下**满足**。
需要并记的口径差异：单看 attention 输出时，在 `|x|≈4` 处 1 个 bf16 ULP 就是 1.56e-2，
所以那一层的逐张量对比会"超过 1e-2"但仍是 1 ULP；两种口径都留在文档里（§22.6 与本节）。

## 6. 已知阻塞

* **卡被另一路 Qwen3.6 服务占用**（端口 8112，HBM 95,116 MB，非本会话启动）。
  S4 的设备侧每一步都需要独占或至少腾出 ≥110 GiB HBM；开跑前必须与那一路协调（C2），
  并在本方案里加一条"启动前置检查"：`npu-smi` HBM 空闲 ≥ 水位 + 15 GiB，否则拒绝启动。
* `torch_npu` 的 pinned memory 与 stream/event 语义需要先用第 1 步的小基准确认，
  避免把"带宽低"误判成"机制不对"。
