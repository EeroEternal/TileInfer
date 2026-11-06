# MoE 页入路径优化 A/B（第 13–14 轮）

> 脚本：`tools/dsv41/s4_layer_moe_stream.py`（单层 + arena，同一 CPU 参考）
> 复现：`DSV41_MOE_FAST=auto DSV41_MOE_W13=1 .venv/bin/python tools/dsv41/s4_layer_moe_stream.py --seqlen 64`

## 1. 三臂实测

| `seqlen` | A 默认（host 累加） | B `DSV41_MOE_FAST=1`（device 累加） | C B + `DSV41_MOE_W13=1` |
|---|---|---|---|
| **64（prefill 型）** | 912.2 ms | **538.0 ms（−41.0%）** | **515.4 ms（−43.5%）** |
| **1（decode 型）** | **60.2 ms** | 62.6 ms（+4.0%） | 67.6 ms（+12.3%） |

三臂数值均为 **`max|diff| = 0.000e+00`**（对同一 CPU 参考，逐位一致）——
比预期更好：改动只动累加位置与索引扫描设备，fp32 下结果完全一致。

## 2. 两处改动

### 2.1 device 侧累加（`_moe_forward_streamed_device_accum`）

默认路径：

```python
y = torch.zeros(flat.shape, dtype=torch.float32)      # host
for e in group:
    idx, top = torch.where(indices_h == e)            # CPU 扫描
    ...
    y[idx] += out.float().cpu()                       # 每个专家一次 D2H + host 散加
```

prefill 可触及全部 384 专家 ⇒ **384 次 D2H + 384 次 host 散加/层**。快路径：

```python
y = torch.zeros(flat.shape, dtype=torch.float32, device=device)
for e in group:
    idx, top = torch.where(indices_d == e)            # device 扫描
    ...
    y.index_add_(0, idx, out.float())                 # 全程 device，结束一次 D2H
```

并把路由权重用 gate 已在 device 上的副本（省掉每专家一次 H2D）。

### 2.2 w1/w3 合并反量化（`DSV41_MOE_W13`）

`w1`(gate) 与 `w3`(up) 共享同一个 `xsel` 与输入维，沿输出维拼接后**一次** `ref_cpu.linear`：

```python
cat  = torch.cat([w1.view(torch.uint8),  w3.view(torch.uint8)],  dim=0).view(w1.dtype)
scat = torch.cat([s1.view(torch.uint8),  s3.view(torch.uint8)],  dim=0).view(s1.dtype)
gate_out, up = ref_cpu.linear(xsel, cat, scat).split([w1.shape[0], w3.shape[0]], dim=-1)
```

拼接必须走 `uint8` 视图——torch_npu 不支持 fp4/e8m0 dtype 上的 `cat`（与 `repeat` 同一限制）。

## 3. 为什么需要 `auto` 策略

prefill 大赢、decode 略亏（decode 只 6 专家，D2H 从 6 次变 1 次收益有限，而 device 侧
`index_add_` 与更大的中间张量反而更贵）。所以：

```python
if _fast == "auto":
    _fast = "1" if (x.numel() // x.shape[-1]) > 1 else ""
```

实测 `DSV41_MOE_FAST=auto DSV41_MOE_W13=1`、`seqlen=64`：**615 ms**（对照组 912 ms），
策略生效；默认路径（不设环境变量）**一位未改**，满足 C1 的两臂同环境要求。

## 4. 对目标的影响

- **P2（TTFT）**：prefill 侧 −43% 直接改善长 prompt 的首字延迟——这是 P2 的主要瓶颈。
- **P1（TPOT）**：decode 不受益。TPOT 的下一步杠杆是
  **把 1 行的 decode 分派到已有的 `moe_forward_batched`**（S5 实测该臂 **522 ms**，而当前 vLLM 路径
  走 streamed 是 743 ms），以及 MTP 投机解码。
- 注意：`moe_forward_batched` 之前被"一律走 streamed"覆盖，是因为当时在追 wo_a 数值 bug；
  该 bug 已由 `_resolve` 的修复解决（见 `dsv41-vllm-integration-plan.md` §29），所以它可以在
  **数值验收通过后**重新启用。

## 5. 验收待办

1. 真机跑 `s4_full_stack_device.py --gen 2`，确认首 token 仍为 `11111`（需 ≥510 GB 主机内存）；
2. 若通过，把 vLLM 适配器的 1 行分派改为 `moe_forward_batched`，用差分法重测 TPOT，目标 ≤350 ms。

## 6. 受控三臂 × 3 次重复（同一窗口连续跑，取中位数）

上面的三臂数据跨了两个窗口、机器负载不同，绝对值不可比。本节是在**同一窗口内连续跑**
三臂 × 3 次重复的结果（`s4_layer_moe_stream.py`，同一 CPU 参考）：

### prefill 型（seqlen=64）

| 臂 | rep1 | rep2 | rep3 | **中位数** | 相对默认 |
|---|---|---|---|---|---|
| A 默认 | 692.0 | 771.1 | 731.7 | **731.7 ms** | — |
| B `FAST+W13` | 568.2 | 595.9 | 578.5 | **578.5 ms** | **−20.9%** |
| C `GROUPED` | 552.2 | 561.7 | 551.2 | **552.2 ms** | **−24.5%** |

极差：A 11.4%，B 4.8%，C 2.0% —— 分组 GEMM 不仅更快，**波动也更小**。
（A 的极差较大是因为它每专家一次 D2H，受主机↔设备带宽争用影响更明显。）

C 相对 B 再快 4.5%，说明"把组内各专家的 w1/w3 堆叠成一次批量反量化 GEMM"确实又省掉了 2E 次
`ref_cpu.linear` 的固定开销。

### decode 型（seqlen=1）

| 臂 | rep1 | rep2 | 中位数 |
|---|---|---|---|
| A 默认 | 63.6 | 49.7 | ~50–64 ms |
| B `FAST+W13` | 65.3 | 68.0 | ~65–68 ms |
| C `GROUPED` | 69.8 | 66.8 | ~67–70 ms |

结论与 §3 一致：**decode 侧这三项优化都不划算**（只 6 个专家，D2H 本来就只有 6 次；
而 padding 到 `max_n`、设备侧累加的固定开销反而更明显）。所以 `auto` 只对多行生效是对的。

## 7. 结论

- **prefill 侧确定收益 −24.5%（中位数，受控重复）**，数值逐位一致（`max|diff| = 0`）；
- **decode 侧无收益**，`DSV41_MOE_FAST=auto` 会在 1 行输入时自动跳过这些优化；
- TPOT（P1）要改善，必须靠 decode 专属手段：`moe_forward_batched` 分派（待整模型验证）或 MTP。

## 8. TPOT 差分法的一个陷阱（实测踩到，已修）

第一次跑两臂 TPOT 时，`default` 臂给出 **1.5 ms/token** —— 一个"看起来像巨大优化"的废数据：

```
T(8)=4.046s   T(72)=4.143s
TPOT = (4.143 - 4.046) / (72 - 8) = 1.5 ms/token
```

原因：prompt 是 `"The capital of France is"`，模型在 `" Paris."` 之后**自然停止**（`finish_reason=stop`），
于是请求 8 个和 72 个 token 的**实际生成长度几乎一样**，差分法的分子趋近 0，除出来的数毫无意义。

### 修法（`tools/dsv41/perf_vllm_tpot.py`）

1. 请求体加 `"ignore_eos": True`，强制生成满 `max_tokens`；
2. 分母改用服务端返回的 **`usage.completion_tokens`**，并打印
   `实际生成 tokens: N1=.. N2=..（请求 8/72）`；
3. 若 `ct2 <= ct1` 直接判"差分法不可用"并退出，**不产出数字**。

### 为什么值得记下来

这个陷阱对所有"短 prompt + 差分法测 TPOT"的场景都成立：**只要模型会早停，差分法就会给出
偏小甚至为负的 TPOT**。之前基线里的 743 ms/token 是逐 token 计时得到的，不受此影响；
但一旦改用差分法（验收要求），就必须同时锁 `ignore_eos` 与用实际 token 数做分母。

### 顺带确认的正面结果

同一轮里，两臂的 `Paris` 数值门都通过：

```
== 臂 A：默认分派 ==                 首 token 校验 PASS（输出含 Paris）
== 臂 B：DSV41_DECODE_BATCHED=1 ==   首 token 校验 PASS（输出含 Paris）
```

即 **`moe_forward_batched` 在 vLLM 端到端输出正确**（首 token 未被破坏）。缺的只是性能数字。

## 9. vLLM 路径双臂 TPOT 实测（decode 优化终于量化）

方法：`tools/dsv41/perf_vllm_tpot.py`，差分法 `(T(72) − T(8)) / (72 − 8)`，`temperature=0`、
`ignore_eos=true`（否则早停，见 §8）、分母用服务端返回的**实际 completion_tokens**、3 次重复取中位数。
两臂同 venv、同启动参数（`--enforce-eager --max-model-len 2048 --max-num-seqs 1
--gpu-memory-utilization 0.3 --no-enable-prefix-caching --no-enable-chunked-prefill`），满足 C1。

| 臂 | 实际生成 tokens | TPOT | 极差 T(8) / T(72) | `Paris` 门 |
|---|---|---|---|---|
| 默认（streamed decode） | 8 / 72 | **697.7 ms/token** | 10.5% / 2.4% | PASS |
| `DSV41_DECODE_BATCHED=1` | 8 / 72 | **592.4 ms/token** | 3.5% / 4.6% | PASS |

**相对差异：−15.1%（batched 更快）。**

### 结论与处置

- decode 侧把 1 行输入分派到 `moe_forward_batched`（六个专家合成一次批量 GEMM）**确有收益**，
  且两边输出都是 `Paris`（首 token 未被破坏）；
- 因此已把该分派**设为默认**（`DSV41_DECODE_BATCHED` 的默认值改为生效，设 `0` 可退回）；
- 与基线的 743 ms/token 对比：今天的默认臂测到 697.7 ms（机器负载差异），
  受控 A/B 的相对收益 −15.1% 才是可引用的数字。

### 与目标（≤350 ms）的差距

592.4 → 350 仍需 **−41%**。剩余候选按预估收益排序：
1. **异步流把专家页入与 GEMM 重叠**（上限 ~84 ms/token，见 perf-plan §9）；
2. **attention per-kernel 融合**（~150 ms/token，需移植 DeepInfer 的 asc TileLang）；
3. **MTP 投机解码**（潜力最大，需接 vLLM speculative 路径）。
