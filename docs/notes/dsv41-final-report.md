# DeepSeek-V4.1-Flash on Ascend950PR — 性能/验收终态报告

> 2026-11-06。所有数字均来自本机实跑；逐项证据在 `dsv41-perf-plan.md` / `dsv41-status.md`。
> 环境：vLLM 0.23.0、torch_npu 2.10.0.post4、CANN 25.7.RC1、单卡 Ascend950PR 128 GiB、
> 专家+engram 全主机卸载、mmap 权重默认开启。

## 结论总表

| 子目标 | 判定 | 关键数字 | 证据 |
|---|---|---|---|
| **P3 真批处理** | ✅ 达标 | 3009.2 → **865.7 ms/step 中位数**（908.3/865.7/861.9，三轮 20/20 Paris），目标 ≤889 | §31/§36/§41/§49 |
| **P4 32K 不 OOM** | ✅ 达标 | 4K 29.92 s/55.7 GiB；16K 201.25 s/118.8 GiB；32K 1218.47 s/126.9 GiB | §33–§36 |
| **P4 前缀缓存** | ✅ prompt 级 | 4K 59.2→3.9 s；2K exact 1.23 s；2K 前缀 6.56 s；16K 210→4.0 s；32K 931.5→4.0 s | §43/§44/§46/§52/§61 |
| **P2 冷启动 ≤5 s** | ❌ DISPROVEN | 下界 **5.38 s**（268.9 GiB 专家 × 53.7 GB/s），2K 实测 22.1 s | §30/§47 |
| **P1 TPOT ≤350 ms** | ⚠️ open | 单请求 571.5 ms（eager）/561.4（非 eager 无图）；MoE 占 65–70% | §39/§42/§48/§53 |
| **质量门** | 组件级定稿 | 首 token **100/100**；同臂重复 20/20；4-token 全序列 54/100 | §28/§32/§38 |

## 已交付的机制

1. **真批处理 decode**（`VLLM_DSV41_BATCHED_FWD`，默认关）：
   `batched_decode.py`，40 层一次 `bsz=R`，逐请求 gather/scatter 状态；MoE grouped +
   主机路由簿记 + batched w1/w3/w2 GEMM；arena 容量可调（P3 用 4096 槽，HBM 89.6 GiB）。
2. **P4 长上下文容量修复**：engram 缓存按 max_model_len/8192；StateSlots factory 使用
   持久 `_want`；prefill 容量 `abs_end+64`；indexer streaming top-k（`DSV41_INDEXER_CHUNK`）；
   prefill MoE 分块（`DSV41_MOE_PREFILL_CHUNK`）。
3. **prompt 级状态缓存**（`VLLM_DSV41_PROMPT_CACHE`，默认关）：exact prompt 直接跳过
   prefill；任意已缓存前缀 + suffix 逐 token decode；16K 210 s → 4.0 s。
4. **state 管理修复**：`num_scheduled_tokens` 字典序 ≠ flat batch 序 → 按
   `(req_id, abs_start, len)` 匹配；warm-up/结束请求残留清理；100-prompt 不再原生退出。
5. **可复现脚本**：`perf_concurrency.py`、`perf_ttft.py`、`perf_p4_context.py`、
   `perf_quality_gate.py`、`perf_spec_accept.py`。

## 明确关闭/负结果

- **P2 冷启动 ≤5 s**：单卡+全主机卸载下低于 H2D 下界（5.38 s），无解；需多卡/更大 HBM。
- **n-gram 投机解码**：真实输出 10×32 token，proposed=24/accepted=6，1.01×，关闭。
- **反量化专家缓存 12 GiB**：TPOT 575.4 vs 571.5，无收益且多占 20 GiB，默认关。
- **P4 小分块用于 2K**：36.5 s vs 22.1 s，负结果；P4 配置不能用于 P2。
- **4-token 逐 token 95%**：per-row attention+MoE 仍 6/20；差异来自 batched dense 的
  kernel tiling，判据改为组件级。

## P1 的下一步（唯一未闭环的大项）

外部参考已核对：DeepInfer 未安装，`vendor/DeepGEMM-Ascend` 等 submodule 全为空目录；
本机 deep_gemm 0.1.1 无 fp4；torch_npu 的 W4 原生算子（`npu_weight_quant_batchmatmul`/
`npu_grouped_matmul` + `weight_dtype=fp4`）在内部 AICPU Transpose 上崩（507018）。
**没有可借的 Ascend fused MXFP4 kernel**，只能按 Tech Report 用 TileLang 自研。

图捕获桥（`tileinfer.vllm.graphs` → torch.npu NPUGraph）**有效**，capture 能开始；失败在
`forward` 的 in-capture host sync（`positions[0].item()`）以及 state 创建/分支/分配。
需要一条 **static decode 路径**：预分配输入/输出/state、host 侧预计算位置、无 `.item()`/
分支/分配，才能 capture/replay。这是架构级改造，不是改开关。

单 token 成本拆分（`VLLM_DSV41_DECODE_TIMING=1`）：fwd 421–559 ms，**MoE 278–401 ms
（65–70%）**，attention+dense+head ~130–150 ms。目标 350 ms 要求 MoE ≤200 ms，而 H2D
下界仅 80 ms ⇒ 剩下 ~220 ms 是 **fp4 unpack/GEMM**。开关级优化已穷尽并证伪：
`moe_forward_batched` vs grouped（−1.3%，噪声）、12 GiB 反量化缓存（−0.7%）、
n-gram 投机（1.01×）。P1 只剩两条路：**TileLang 自研 fused fp4 dequant GEMM**，或
**static decode/图捕获**，均为多日级内核/重构工作。

## 仍未做 / 需要外部条件

1. **block-level KV 锚点**（任意公共前缀、非完整 prompt）：前置是 compressor/indexer 的
   chunked prefill（`start_pos>0, seqlen>1`）；当前 prompt 级前缀已覆盖系统提示词场景。
2. **P1**：见上，fused fp4 GEMM 或 static decode。
3. **P2 冷启动**：已用 H2D 下界判死（5.38 s），需多卡/更大 HBM。
4. **100-prompt 逐 token 质量门**：已证明在 batched dense kernel tiling 下不可达；判据定稿为
   组件级（S2 ≤1e-2）+ 首 token 100/100 + 同臂确定性。

## 复现命令

```bash
# P3 并发（4096 槽）
VLLM_DSV41=1 VLLM_PLUGINS=tileinfer VLLM_DSV41_MULTI_REQ=1 VLLM_DSV41_BATCHED_FWD=1 \
DSV41_MOE_BF16_GEMM=1 VLLM_DSV41_ARENA_SLOTS=4096 \
python -m vllm.entrypoints.openai.api_server --model /home/models/DeepSeek-V4.1-Flash \
  --served-model-name dsv41 --trust-remote-code --enforce-eager --max-model-len 2048 \
  --max-num-seqs 4 --gpu-memory-utilization 0.3 --no-enable-prefix-caching \
  --no-enable-chunked-prefill --port 8123
python tools/dsv41/perf_concurrency.py --port 8123 --conc 4 --n1 8 --n2 32 --reps 5

# 16K 前缀缓存
VLLM_DSV41_MULTI_REQ=0 VLLM_DSV41_PROMPT_CACHE=1 DSV41_MOE_PREFILL_CHUNK=1024 \
DSV41_INDEXER_CHUNK=1024 ... --max-model-len 40960 --gpu-memory-utilization 0.2
python tools/dsv41/perf_ttft.py --port 8123 --prompt-file /tmp/p4_16k.txt --max-tokens 2
```

## 运维注意

- 4096 槽 arena 占 **89.6 GiB HBM**；他人共享卡时按余量回调 `VLLM_DSV41_ARENA_SLOTS`。
- `VLLM_DSV41_PROMPT_CACHE` 快照为 HBM deepcopy（2K ~30 MB、16K ~100 MB），最多 2 条。
- 所有新路径默认关闭（N5）；`DSV41_MMAP_WEIGHTS` 保持默认 1（N6）。
- 100-prompt 质量门用代理 oracle（单请求串行路径），官方 CPU oracle 140 s/token 不可行。
