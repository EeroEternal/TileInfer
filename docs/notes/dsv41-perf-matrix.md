# 端到端性能矩阵（2026-11-08）

> 口径：流式 `/v1/completions`，`temperature=0`、`ignore_eos=true`；TTFT 为首个非空内容 chunk；
> ITL 为相邻 token chunk 间隔；TPOT = `(last-first)/(n-1)`。同一台 Ascend950PR 单卡。
> 表中 `out` 是实际流式统计到的输出 token/chunk 数；请求 `max_tokens` 另注。
> 没有的格子写“未测”，不同配置的数字单独标 config。

## 1. 原生 vllm-ascend：支持 v41 的 0.30 卡在 A5/A2 设备类型

独立目录 `.venv_ascend` / `.venv_ascend30`（复制并改写 shebang，未污染原 `.venv`）：

- `vllm-ascend 0.23.0` 与 `0.23.0.post1` 只认 `deepseek_v4`，不认 `deepseek_v41`；
- `vllm-ascend 0.30.0` **已包含 `vllm/models/deepseek_v41` 和 `DeepseekV41Config`**，
  `vllm 0.30 + transformers 5.14.1` 能正确解析 v41 config；
- 但 0.30 启动需要 `torch-npu==2.10.0.post7 + triton-ascend==3.6.0.post2`，公开索引上的
  post7 是 **A2 构建**，当前卡是 **A5（950PR）**，启动报
  `Current device type: AscendDeviceType.A5 does not match ... AscendDeviceType.A2`；
- 强制 A5 + V2 runner 后，带 CPU offload 报 `get_accelerator_view_from_cpu_tensor` 不支持；
  去掉 offload 报 `DeepseekV4 attention requires a CUDA device`（上游 vllm 0.30 注意力未完全替换）。

因此**原生 vllm-ascend 没有可比的性能数字**。原 `.venv` 的误装已卸载；`.venv_ascend*` 未跟踪。
详见 `dsv41-native-ascend-attempt.md`。

## 2. 单并发

### 2.1 短上下文 / 2K

配置：`max-model-len 40960`、`gpu-memory-utilization 0.35`；优化臂
`BLOCKSCALED=1 + ARENA=1536`，baseline `BLOCKSCALED=0 + ARENA=384`。

| 输入 tokens | 臂 | 请求 max_tokens | 实际 out | TTFT | TPOT | ITL P50 | ITL P90 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 127 | 优化 | 8 | 8 | 8.44 s | 306 ms | 402 ms | 431 ms |
| 127 | baseline | 8 | 8 | 7.97 s | 492 ms | 580 ms | 603 ms |
| 2045 | 优化 | 8 | 5 | 27.29 s | 427 ms | 509 ms | 576 ms |
| 2045 | baseline | 8 | 5 | 19.86 s | 521 ms | 671 ms | 741 ms |

### 2.2 长上下文

配置：优化臂 4K 用 `ARENA=1024 + BLOCKSCALED=1`、util 0.45；16K 用
`ARENA=96 + MOE_PREFILL_CHUNK=1024 + INDEXER_CHUNK=1024 + BLOCKSCALED=1`、util 0.2；
baseline 用 `ARENA=96 + 同样 chunking`、util 0.2。

| 输入 tokens | 臂 | 请求 max_tokens | 实际 out | TTFT | TPOT | ITL P50 | ITL P90 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4096 | 优化 | 8 | 5 | 29.81 s | 408 ms | 537 ms | 559 ms |
| 4096 | baseline | 8 | 6 | 29.92 s | 680 ms | 821 ms | 921 ms |
| 16381 | 优化 | 8 | 5 | 97.22 s | 805 ms | 636 ms | 1949 ms |
| 16381 | baseline | 8 | 5 | 91.12 s | 604 ms | 795 ms | 827 ms |

> 16K 的 blockscaled 优化臂反而比 baseline 慢（TPOT 805 vs 604），因为该配置 arena 只有 96，
> blockscaled 的专家换入 miss 高；16K 的 block-cache 命中路径是另一套配置（4.65 s，见
> `dsv41-p4-chunked-prefill.md`）。

## 3. 4 并发

配置：`max-num-seqs 4`、`max-model-len 4096`、util 0.35；优化臂
`MULTI_REQ=1 + BATCHED_FWD=1 + ARENA=4096`，baseline `MULTI_REQ=1 + BATCHED_FWD=0 + ARENA=384`。

| 输入 tokens | 臂 | 请求 max_tokens | 实际 out | TTFT（4 请求中位） | TPOT | ITL P50 | ITL P90 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2045 | 优化 | 8 | 4 | 86.68 s | 1290 ms | 1402 ms | 1414 ms |
| 2045 | baseline | 8 | 5 | 87.81 s | 3126 ms | 3825 ms | 4851 ms |
| 127 | 优化 | 16 | 未记录 | 22.71 s | 1022 ms | 941 ms | 1009 ms |
| 127 | baseline | 8 | 未记录 | 29.86 s | 1551 ms | 1985 ms | 2002 ms |

历史差分验收：4 并发每步 **3009.2 → 878.6 ms**（127-token 档，20/20 `Paris`）。

## 4. 未测 / 不可得

- 4 并发 4K/16K（P3 配置 max-len/arena 不足，未跑）；
- 32K 冷启动全矩阵（单次 15–20 min，且当前 32K 只验证过 prompt 缓存/不 OOM）；
- 原生 vllm-ascend：host 容量、packed MXFP4、mmap、逐专家 offload 均已打通，
  但 profile 卡在 `aclnnHcPre` 的 950 kernel 缺 `tilingKey 1003`（预编译
  `hc_pre.json` 只有 1000/1001），无可测 TTFT/TPOT/ITL；详见
  `docs/notes/dsv41-native-mmap.md` §5；
- baseline 4K/16K 的 `max_tokens` 更大档位（本轮只测 8）。

## 5. 复现

```bash
.venv/bin/python tools/dsv41/perf_matrix.py --port 8123 \
  --tokens 2048 --max-tokens 8 --conc 1 --reps 1 --label <arm>
```

各臂启动脚本：`.work/start_matrix_opt.sh`、`start_matrix_base.sh`、
`start_matrix_opt_long.sh`、`start_matrix_base_16k.sh`、`start_p3_matrix.sh`、
`start_p3_base_matrix.sh`、`start_matrix_opt_16k.sh`。
