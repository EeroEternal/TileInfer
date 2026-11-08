# 端到端性能矩阵（2026-11-08，进行中）

> 口径：流式 `/v1/completions`，`temperature=0`、`ignore_eos=true`；TTFT 为首个非空内容 chunk，
> ITL 为相邻 token chunk 的间隔，TPOT = `(last-first)/(n-1)`。同一台 Ascend950PR，单卡。
> 本文件只记录本轮实跑；没有的格子写“未测”，不拿别轮不同配置的数字冒充。

## 1. 原生 vllm-ascend：暂时无法启动

在独立目录建了 `.venv_ascend`（从 `.venv` 复制并改写 shebang，未污染原 `.venv`）：

- `vllm 0.23.0 + vllm-ascend 0.23.0 + transformers 5.5.4` 可 import，`torch.npu.is_available()=True`；
- 但 `deepseek_v41` 不在 transformers/vLLM 的 config 表里：原生只认 `deepseek_v4`；
- 把 config 改成 `deepseek_v4 + DeepseekV4ForCausalLM` 后，vLLM 报
  `text_config ... does not have num_attention_heads`；
- `vllm-ascend 0.30.0` 要求 `torch-npu==2.10.0.post7`、`triton-ascend==3.6.0.post2`、
  `transformers==5.14.1`，与当前 `post4` 环境不兼容，需要另一套 CANN/torch-npu，不能只换 venv。

因此**原生 vllm-ascend 没有可比的性能数字**；原 `.venv` 已 `pip uninstall vllm-ascend` 清理。

## 2. 单并发 / 短上下文

本轮配置：`max-model-len 40960`，`gpu-memory-utilization 0.35`，优化臂
`DSV41_MOE_BLOCKSCALED=1 + ARENA_SLOTS=1536`，baseline 臂 `BLOCKSCALED=0 + ARENA_SLOTS=384`。

| prompt tokens | 臂 | TTFT | TPOT | ITL P50 | ITL P90 |
|---:|---|---:|---:|---:|---:|
| 127 | 优化 | 8.44–9.41 s | 306–394 ms | 402–414 ms | 430–469 ms |
| 127 | baseline | 7.97 s | 491 ms | 580 ms | 603 ms |
| 2045 | 优化 | 27.29 s（19.4–35.2） | 427 ms | 509 ms | 576 ms |
| 2045 | baseline | 19.86 s | 521 ms | 671 ms | 741 ms |

> 注意：这里的 TTFT 比历史文档的 3.8–4.3 s 高，是因为本轮统一用了 `max-model-len 40960`
> 和大 arena；历史 3.8 s 是 `max-len 2048 + arena 384` 配置。**不同配置的 TTFT 不能直接横比**。

## 3. 单并发 / 长上下文

| prompt tokens | 臂 | TTFT | TPOT | ITL P50/P90 |
|---:|---|---:|---:|---:|
| 4096 | 优化 | **30.05 s** | **482 ms** | 522 / 560 ms |
| 4096 | baseline | 29.9–59.2 s（历史） | 未测 | 未测 |
| 16381 | 优化 | 未完成（engine 退出） | 未测 | 未测 |
| 16381 | baseline | 201–210 s（历史） | 未测 | 未测 |

16K 本轮请求导致 engine shutdown，未拿到 TTFT/TPOT；历史 block-cache cold 223.42 s、
命中 4.65 s 见 `dsv41-p4-chunked-prefill.md`。

## 4. 4 并发 / 短上下文

配置：`max-num-seqs 4`、`max-model-len 2048`、127-token prompt、8 output tokens。

| 臂 | TTFT（4 请求中位） | TPOT | ITL P50 | ITL P90 |
|---|---:|---:|---:|---:|
| P3 优化（`MULTI_REQ=1 + BATCHED_FWD=1 + ARENA 4096`） | 22.71 s | 1022 ms | 941 ms | 1009 ms |
| P3 前 baseline（`MULTI_REQ=1 + BATCHED_FWD=0 + ARENA 384`） | 29.86 s | 1551 ms | 1985 ms | 2002 ms |

历史差分验收：4 并发每步 **3009.2 → 878.6 ms**（20/20 `Paris`），见
`dsv41-p3-batching.md`；本轮流式 TTFT/ITL 只测了 127-token 这一档。

## 5. 未测

- 4 并发 2K / 长上下文；
- baseline 的 4K/16K TTFT/TPOT/ITL（同本轮配置）；
- 16K 优化臂（engine 退出）；
- 原生 vllm-ascend（模型架构/版本卡住）；
- 32K 全矩阵。

## 6. 复现

```bash
.venv/bin/python tools/dsv41/perf_matrix.py --port 8123 \
  --tokens 2048 --max-tokens 8 --conc 1 --reps 2 --label <arm>
```

服务器启动脚本：`.work/start_matrix_opt.sh`（优化）、`.work/start_matrix_base.sh`
（baseline）、`.work/start_p3.sh`（P3 优化）、`.work/start_p3_base.sh`（P3 baseline）。
