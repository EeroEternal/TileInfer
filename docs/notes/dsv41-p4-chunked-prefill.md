# P4 chunked prefill / block-level KV 锚点（进行中）

> 2026-11-07。目标：任意公共前缀命中 ≤5 s，4K/16K 先验收，32K 窗口可用时补测。

## 1. 当前进展

### 1.1 P3 回归（不破坏）

同一 4 并发配置（`BATCHED_FWD=1 / ARENA_SLOTS=4096 / max-num-seqs 4`）：

```
T(8)=11.843s  T(32)=32.930s  →  878.6 ms/step
20/20 请求输出含 Paris
```

≤889 ms，P3 未回归。blockscaled MoE 默认关闭，P3 路径未改。

### 1.2 compressor chunked prefill（已完成）

`compressor_forward` 增加 `start_pos>0 && seqlen>1` 分支，要求 chunk 与 `ratio` 对齐：

- `start_pos % ratio == 0` 且 `seqlen % ratio == 0`；
- 每个 chunk 产出 `seqlen/ratio` 个完整 latent，不携带 partial-block state；
- 与整段 prefill 逐位一致（fp32 softmax 聚合顺序相同）。

新增 CPU 用例 `tests/test_dsv41_chunked_compress.py`：

| 用例 | 结果 |
|---|---|
| ratio=2，8 token 整段 vs 2×4 chunk | `torch.equal` ✅ |
| ratio=4，16 token 整段 vs 4×4 chunk | `torch.equal` ✅ |

### 1.3 indexer chunked prefill（已完成）

`indexer_forward` 在 `start_pos>0 && seqlen>1` 时：

1. 新 latent 的 RoPE 位置：对齐 chunk 用 `start_pos + i*ratio`（不是 decode 的
   `start_pos+1-ratio`）；
2. `index_k` 写入 `start_pos//ratio : +n_lat`；
3. `compress_lens` 变成 `[b,s]`：`(arange(s)+start_pos+1)//ratio`，score 按 query 逐行 mask；
4. topk 用 `min(index_topk, end_pos//ratio)`，再按每行 `compress_lens` 把越界项置 -1；
5. candidate source 的 `select_candidate_blocks` 对 `[b,s,T]` 已支持，且 score 仍覆盖
   **全部历史 T**，因此每个 chunk 重算候选是精确的，不需要跨 chunk 合并。

新增 CPU 用例 `tests/test_dsv41_chunked_indexer.py`：

| 用例 | 结果 |
|---|---|
| ratio=2，8 token 整段 vs 2×4 chunk | index_k 与 top-k 逐位相等 ✅ |
| ratio=1，8 token 整段 vs 2×4 chunk | index_k 与 top-k 逐位相等 ✅ |

注意：不同 chunk 的 `topk` 宽度可以不同（例如前 4 个 token 只有 2 个压缩块）；每个 chunk
独立调用 `sparse_attn` 即可，不需要拼接成统一宽度。

### 1.5 chunked suffix 已接进 prefix cache（默认关）

`VLLM_DSV41_PROMPT_CACHE=1 VLLM_DSV41_PREFIX_CHUNK=128` 时，prompt 级前缀命中后的 suffix
走 `_prefill_suffix_chunked`：先对齐到偶数位置，再按偶数 chunk 调 `transformer_forward`
（`start_pos>0, seqlen>1`），尾随单 token 回退 decode 形状。

实测（A=421 token，B=A+7 token，同 venv/同参数）：

| 臂 | 输出 token | 文本 | 耗时 |
|---|---|---|---|
| A 冷 prefill（缓存 A） | `[1,0,5,223]` | `# ` | 11.7 s |
| B 前缀命中 + chunked suffix | `[455,6102,294,14251,344,16235,16,455]` | ` The capital of Italy is Rome. The` | 8.89 s |
| B 冷 prefill（对照） | **同上，8 个 token 逐位相同** | 同上 | 14.85 s |

说明 chunked suffix 的数学与整段 prefill 一致；候选源层的 `compress_lens` 需要
`unsqueeze(-1)` 才能与 `[b,s,T]` 的 block 维广播（已修）。

### 1.6 block-level 锚点（原型，默认关）

`VLLM_DSV41_BLOCK_CACHE=1 VLLM_DSV41_BLOCK_SIZE=512` 时，cold prefill 按 512 token 分块，
每块结束把 `window_kv/compress_kv/index_k/kv_state/score_state/engram` 的**已用切片**存到
CPU（设备端快照在 4K 时触发 507035，CPU 快照稳定），key 为 token 前缀哈希。

实测（A=4K cold，C=3.5K 公共前缀 + 8 token）：

| 项 | 结果 |
|---|---|
| A cold（含 7 个 block 快照） | 77.0 s |
| C block hit（恢复 3584 + chunked suffix 8） | **5.66 s** |
| C 输出首 token | 与 cold C 一致（`455,6102,294,14251`） |
| 128-block / 421-token 小测 | hit 8.05 s vs cold 11.9 s，8 token 逐位一致 |

目标 ≤5 s，当前 5.66 s（差 13%）：恢复本身约 2–3 s，suffix 的 chunked prefill 约 2–3 s。
把短 suffix 改回逐 token decode 反而更慢（8.49 s，prefill 上下文下每 token 都触发 6 专家换入），
因此保留 chunked suffix。下一步压缩恢复成本（复用 state、只拷贝必要层）或把块大小/恢复做成
增量式。

### 1.7 设计要点

- block 大小取所有 `compress_ratio` 的公倍数（本模型 ratio ∈ {1,2}，block=256 即可）；
- prefill 每完成一个 block，对 `window_kv/compress_kv/index_k/kv_state/score_state` 做快照，
  以 token block 哈希为 key；
- 新 prompt 找最长命中 block 前缀，恢复快照，剩余 suffix 用 chunked prefill（而非逐 token
  decode）补齐；
- 快照占 HBM：每 block 的 state 约 ``window + compress + index``，需按 4K/16K 实测预算。

## 2. 复现

```bash
cd /home/userlist/zxs/TileInfer
.venv/bin/python -m pytest tests/test_dsv41_chunked_compress.py -q
.venv/bin/python -m pytest tests/test_dsv41_*.py -q   # 144 passed
```

## 3. 下一步

1. 完成 `indexer_forward` 的 `seqlen>1` 分支 + CPU 对拍；
2. 把 chunked prefill 接进 `_restore_prefix_state` 的 suffix 路径；
3. block 快照/哈希/恢复 + 4K 命中验收，再跑 16K。
