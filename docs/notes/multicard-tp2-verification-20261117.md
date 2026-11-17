# 多卡场景 A 验证：Qwen3.6 TP=2（2026-11-17）

> 结论：**通过**。TileInfer 侧（计划并行 / `doctor` 门 / bridge 在 TP 进程安装）和端到端
> （健康、完整请求、图捕获 6 张、回退 0）都过了。
>
> 环境上第一次被 HCCL `RootInfoDetect` 阻塞；根因是 `/etc/hccl_rootinfo.json` 是一个**空目录**。
> 临时移走后 TP=2 通过，**测试结束已恢复原目录**。这是宿主问题，不是 TileInfer 代码。

## 0. 机器与代码

| 项 | 值 |
|---|---|
| 机器 | `192.168.235.16`，8 × Ascend950PR |
| 卡 | 物理 2、3（`ASCEND_RT_VISIBLE_DEVICES=2,3`）；测试时 8 卡全空闲 |
| 端口 | 8137 |
| 代码 | `/home/ernest/TileInfer`，未提交的 Phase 0 改动 |
| CANN | `/home/zxs/Ascend/cann-9.3.0`（`source /home/zxs/TileInfer/scripts/env-ascend950.sh`） |
| Triton | `/home/zxs/triton-ascend-3.2.2`（必须加入 `PYTHONPATH`） |
| 栈 | torch 2.10.0+cpu / torch_npu 2.10.0.post4 / vLLM 0.23.0 / vllm-ascend 0.23.0 |

> 本机不能用系统 CANN 9.1.0。用错时先报 `libcust_opmaster_rt2.0.so` 的
> `GLIBCXX_3.4.29` 缺失，再报 `triton has no attribute next_power_of_2`；换成上面的
> CANN 9.3 + triton-ascend 后这两个错误消失。

## 1. TileInfer 侧：通过

```bash
source /home/zxs/TileInfer/scripts/env-ascend950.sh
export PYTHONPATH=/home/ernest/TileInfer:/home/zxs/triton-ascend-3.2.2:$PYTHONPATH
cd /home/ernest/TileInfer

python -m tileinfer.cli.main plan build \
  --model /home/models/Qwen3.6-35B-A3B --out /tmp/ti-qwen-tp2/plan.json --tp 2
# -> plan_id=d6f1c7e5ce668498 parallel=['tp'] servable=True sources=['vendor']

TILEINFER_PLAN=/tmp/ti-qwen-tp2/plan.json TILEINFER_TP_SIZE=2 \
  python -m tileinfer.cli.main doctor --json
# -> ok=true；plan/parallel/layer_guard/bridge 全 PASS
```

真实 TP 进程里的 bridge 日志（API server + EngineCore 都出现）：

```text
[tileinfer] TileInfer v3 bridge: plan=d6f1c7e5ce668498 model=Qwen3.6-35B-A3B
selections=8 (vllm_ascend.attention=1, ... vllm_ascend.sample=1) access_points=4
fallback=forbid parallel=tp | seams=config_gate,graph_hook,attn_select
skipped=none | metrics=/tmp/ti-qwen-tp2/metrics.json
```

说明：计划里的 `parallel=tp` 被真实 TP 进程读到；bridge 正常安装，且**没有**注入任何
非厂商实现（8 项全 vendor）。

## 2. 端到端结果：通过

启动日志（TP=2，两块卡各一个 worker）：

```text
Available KV cache memory: 34.26 GiB
GPU KV cache size: 2,028,105 tokens
Free memory on device (79.29/79.81 GiB) ... Actual usage: 32.85 GiB for weights ...
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 3/3
Capturing CUDA graphs (decode, FULL): 3/3
Application startup complete.
```

`/health` 在 65 s 内返回 200。

greedy 请求（temperature 0，`The capital of France is`，16 tokens）：

```text
text:  ' Paris, a city renowned for its rich history, culture, and iconic landmarks.'
usage: prompt_tokens=5 completion_tokens=16 total_tokens=21
```

对 EngineCore 先发 `SIGUSR1` 得到全 0（图捕获发生在 worker 进程），再对两个 `VLLM::Worker`
发 `SIGUSR1`，得到真实计数：

```json
{
  "plan_id": "d6f1c7e5ce668498",
  "capture_calls": 1,
  "captured_graphs": 6,
  "capture_keys": {"PIECEWISE": 3, "FULL": 3},
  "cudagraph_mode": "FULL_AND_PIECEWISE",
  "capture_done": true,
  "decode_calls": 15,
  "decode_graph_hits": 15,
  "decode_fallbacks": 0,
  "none_after_capture": 1,
  "attention_backend": "vllm_ascend.attention.attention_v1.AscendAttentionBackend",
  "errors": 0
}
```

`decode_calls=15` 全部命中图，`decode_fallbacks=0`，`errors=0`，后端是厂商
`AscendAttentionBackend`。

> 观测提示：TP>1 时图计数在 **worker** 进程里；只给 EngineCore 发 `SIGUSR1` 会读到全 0。
> 这解决了 M5 §3 里"短跑 decode_calls 显示 0"的同一类困惑。

## 3. 环境阻塞与解决（HCCL RootInfoDetect）

第一次带正确 CANN/triton 运行时，两个 worker 都建好了 HCCL 组：

```text
world_size=2 rank=0 ... backend=hccl
world_size=2 rank=1 ... backend=hccl
```

随后第一次 all-reduce 失败：

```text
HCCL function error: hcclCommInitRootInfoConfig(...), error code is 4
Communication_Error_Ranktable_Detect(EI0015): ... RootInfoDetect failed.
```

已排除（均无变化）：全新 `VLLM_CACHE_ROOT`、`--skip-mm-profiling`、
`HCCL_SOCKET_IFNAME=enp136s0f4u1u1`、`HCCL_SOCKET_IFNAME=lo`（`HCCL_CONNECT_TIMEOUT=120`）、
8 卡全空闲重试。

根因：`/etc/hccl_rootinfo.json` 是一个**空目录**，HCCL 把它当 JSON 文件读写，`RootInfoDetect`
因此失败。这与 [`m5-nemotron-blocker.md`](m5-nemotron-blocker.md) 记录的现象一致（当时该路径
被描述为"空目录"）。

处理（经用户同意、可回滚）：

```bash
mv /etc/hccl_rootinfo.json /etc/hccl_rootinfo.json.bak-empty   # 空目录
# 重跑 TP=2 -> 通过
mv /etc/hccl_rootinfo.json.bak-empty /etc/hccl_rootinfo.json   # 已恢复
```

测试结束已恢复，且没有生成新文件；没有改其他系统文件，没有 kill 他人进程，没有删共享 cache。
**注意**：恢复后宿主又回到原状态，未来 TP 运行仍会踩到；真正修复应由宿主/环境负责人处理
（删除或修正该空目录，或提供合法 `RANK_TABLE_FILE`）。

## 4. 判定

| 项 | 结果 |
|---|---|
| 计划并行记录（`parallel=tp`） | ✅ |
| `doctor` 并行门 | ✅ |
| bridge 在 TP 进程安装、全厂商选择 | ✅ |
| C1 能跑（health + 完整输出） | ✅ |
| C3 能上图（6 图 / 15 命中 / 0 回退） | ✅ |
| C2 跑对（与 TP1/HF 逐位或容差） | 未做（本轮只发了一个请求，没有对照） |
| 性能（TTFT/TPOT/吞吐） | 未测 |
| 我方 provider 的 TP 数值 | 未涉及（全厂商路径） |

## 5. 证据路径

- 计划：`/tmp/ti-qwen-tp2/plan.json`
- doctor：`/tmp/ti-qwen-tp2/doctor.json`
- 响应：`/tmp/ti-qwen-tp2/response.json`
- 指标：`/tmp/ti-qwen-tp2/metrics.json`
- 成功日志：`/tmp/ti-qwen-tp2/serve9.log`
- 失败日志（环境阻塞证据）：`serve6.log`、`serve7.log`、`serve8.log`
- `/tmp` 会被清理；结论与命令以本文件为准。

## 6. 下一步

1. **C2**：同一请求在 TP=1 与 TP=2 下比较 token / logprobs（或与 HF 参考），补上"跑对"；
2. 再谈 Phase 1（我方 provider 的 TP 数值正确性）——本轮没有触发任何非厂商实现。
