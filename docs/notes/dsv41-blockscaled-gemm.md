# Ascend MXFP4 block-scaled GEMM（P1 突破）

> 2026-11-07。目标机 `tileinfer-102`，TileLang 0.1.15 / torch_npu 2.10.0.post4 / CANN 25.7.RC1。
> 结论：**不再需要自研 fused fp4 dequant kernel**——Ascend TileLang 已暴露
> `T.gemm_blockscaled`，硬件在 MAD 内直接消费 packed FP4 权重 + e8m0 block scale。

## 1. 关键 API

```python
A: int8 [M, K/2]                 # packed FP4, 低 nibble 在前
B: int8 [N, K/2]
SA / SB: uint16 [K/64, M] / [K/64, N]   # 每 64 K 两个 e8m0 字节
a_l1 = T.view(A, (M, K), "float4_e2m1fn")
T.gemm_blockscaled(a_l1, b_l1, acc, sa_l1, sb_l1, transpose_B=True)
```

约束：`K % 64 == 0`；L1 输入只支持 NT（`transpose_A=False, transpose_B=True`）；
`BN` 必须整除 `N`。`float4_e2m1fn` 与 `float4_e2m1fn_x2` 的字节数相同，`T.view` 按 bit 数校验。

激活量化用 `torch_npu.npu_dynamic_mx_quant(x, dst_type=296, block_size=32)`：
返回 `q uint8 [r, K/2]` 和 `s uint8 [r, K/64, 2]`，把 `s` 按
`s.reshape(r, K/64, 2).view(uint16).reshape(r, K/64).T` 变成 kernel 的 `SA`。
它和 checkpoint 的权重量化格式一致，单次 32x5120 约 **11 µs**。

## 2. 数值等价（实测）

| 测试 | 形状 | maxdiff |
|---|---|---|
| FP4 x FP4，CPU pack 的 q/s | 32x32x256 | **0.0** |
| FP4 x FP4，跨 10 个 K chunk | 32x2304x5120 | **0.0** |
| NPU `npu_dynamic_mx_quant` + kernel vs 同 q/s 的 dequant 参考 | 32x2304x5120 | **0.0** |

即：只要 A/B 的 q 与 scale 相同，kernel 与 `unpack_fp4_lut + scale + bmm` 逐位一致。

## 3. 性能

单个 expert tile（`M=32`），事件计时，50 次：

| 形状 | block-scaled | torch unpack + bmm | 倍数 |
|---|---|---|---|
| N=2304, K=5120（w13 半） | **6.75 µs** | 244 µs | **36×** |
| N=5120, K=2304（w2） | **8.60 µs** | 235 µs | **27×** |
| N=576, K=2304 | 11.25 µs | — | — |

单 expert 的 kernel 已接近 launch 下限，继续优化空间在调用次数与 Python 派发。
集成进 `moe_forward_batched`（`DSV41_MOE_BLOCKSCALED=1`，默认关）：

| 指标 | baseline | block-scaled |
|---|---|---|
| 单请求 TPOT（N1=1/N2=33，3 次） | 571.5 ms | **410.3 ms** |
| 单请求 TPOT（N1=8/N2=72，3 次） | — | **423.1 ms** |
| 首 token（20 prompt） | 20/20 | **20/20** |
| 4-token 全序列一致 | — | 10/20（50%） |
| 逐位置一致 | — | 82.5% |
| logprob 有界 δ=1 nat | — | 86.3% |

实现：一次 kernel 算所有 routed expert 的 w1/w3（同一 token，拼成 `N=E*2*hid`），
w2 按 expert 逐个调用（激活不同）。w13 的 bmm 被完全消除。
剩余差距（423 vs ≤400）主要是 6 次 w2 调用 + 每层 Python 派发；下一步合并 w2
（grouped/block-diagonal）或把 `npu_swiglu_mx_quant` 融进激活量化。

## 4. 质量权衡

- **非残差**：激活直接 FP4，相对误差 16.2%（当前 FP8 激活 + FP4 权重为 11.9%）。
  首 token 仍 20/20；4-token 序列与串行臂只有 50% 相同，差异集中在 EOS 的 1-token 翻转。
- **两段残差**（`DSV41_MOE_BLOCKSCALED_RESIDUAL=1`）：`x ≈ x0 + x1`，两个 FP4 GEMM；
  激活误差降到 **1.37%**，但当前实现用 `ref_cpu.unpack_fp4_lut` 在每层反解 x0，
  TPOT 升到 **547.7 ms**（20 prompt 全序列 11/20）。残差路径默认关，仅作质量参考。
- 与既有 batched decode 的 4-token 54% 同一量级：**逐 token 相等不是这套 kernel 的有效判据**；
  继续沿用“组件级 + 首 token + 同臂确定性 + 任务级准确率”。

## 5. 复现

```bash
cd /home/userlist/zxs/TileInfer
.venv/bin/python -m tools.dsv41.probe_blockscaled_gemm
.venv/bin/python -m pytest tests/test_dsv41_blockscaled.py -q

# 集成臂
DSV41_MOE_BLOCKSCALED=1 ... --max-num-seqs 1 --enforce-eager --port 8123
.venv/bin/python tools/dsv41/perf_vllm_tpot.py --port 8123 --n1 8 --n2 72 --reps 3
```

## 6. 逐层成本与 100-prompt（2026-11-07 补充）

`VLLM_DSV41_BATCHED_MOE_TIMING=1` 下 40 层 decode 的单层均值：

| 项 | ms/layer | 说明 |
|---|---|---|
| gate | 0.1 | 路由 |
| **load** | **2.5–4.5** | 6 专家 × 17.93 MiB 的 H2D；理论 107.6 MiB / 53.7 GB/s ≈ 2.0 ms |
| gather | 0.3 | 路由簿记 |
| w13 | 0.25–0.3 | 一次 blockscaled GEMM（N=E·2·hid） |
| w2 | 0.7–1.2 | 6 次 blockscaled GEMM |
| shared | 0.24 | FP8 shared expert（未改） |

合计约 4–6 ms/layer，×40 ≈ 160–240 ms/token；**H2D 仍是 MoE 的主要成本**，w13 已接近
launch 下限。TPOT 复测（N1=8/N2=72，3 次）**419.9 ms**，相对 baseline 571.5 为 −26.5%。

shared expert 是 FP8 + 32×32 block scale，不是 MX per-row，不能直接用 `gemm_blockscaled`；
尝试过“bf16 权重常驻缓存”，TPOT 432.0（更差，且多占 2.8 GiB），已回退。

100-prompt 质量门（`perf_quality_gate.py --limit 100 --max-tokens 4`，同 venv/同参数，仅切
`DSV41_MOE_BLOCKSCALED`）：

| 指标 | 结果 |
|---|---|
| 首 token 一致 | **100/100** |
| 逐位置一致（pos0..3） | 100 / 92 / 84 / 65（85.3%） |
| 完整 4-token 一致 | **63/100（63%）** |
| logprob 有界 δ=1 nat | 91.5%（366/400） |

4-token 逐 token 仍未达 95%，与既有 batched 臂同量级；首 token 硬门槛已满足。

## 7. 下一步

1. 合并 6 次 w2 调用（grouped block-scaled 或 block-diagonal），目标 ≤400 ms；
2. 把 `npu_swiglu_mx_quant` 用于 SwiGLU + 激活量化融合，省一次 quant；
3. 把该路径接进 P3 batched decode（`bsz=R`，当前只接了单请求 `moe_forward_batched`）；
4. 100-prompt 首 token + 任务级准确率；S2 组件级对拍；
5. 若质量需要，做 fused residual kernel（避免 Python/LUT 反解），把残差路径压回 ≤420 ms。
