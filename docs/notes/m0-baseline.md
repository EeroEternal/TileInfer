# M0 基线冻结（v2，2026-11-12）

> 目的：在动任何 v3 代码之前，把"现在的 v2 是什么样"钉死到可复现的数字上。
> 之后每一条"没变差 / 变好了"都拿这一页做对照。
>
> 机器：单卡 Ascend950PR（131072 MB HBM），CANN 9.3.0（`/home/userlist/zxs/TileInfer/.work/cann930`），
> Python 3.12.14，vLLM 0.23.0，vllm-ascend 0.23.0（A5 build），torch 2.10.0+cpu，
> torch_npu 2.10.0.post4。
>
> 代码：`9670c6000a2a662d3012e260b500a33e1ca8174c`（最后一个代码提交是
> `449d8bfa5019dd15f17fc13411b049cd4270a8cd`，之后只有 docs 提交）。

## 1. 环境与命令

```bash
cd /home/userlist/zxs/TileInfer
source scripts/env-ascend102.sh          # 唯一的 CANN/环境定义
export LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/9.1.1/atb/cxx_abi_1/lib:${LD_LIBRARY_PATH}
```

两个 v2 环境：

| 环境 | 用途 | 关键包 |
|---|---|---|
| `.venv_native213` | CPU 测试、`ti_trace`、v2 服务 | torch 2.13；**不装** vllm-ascend |
| `.venv_baseline` | vllm-ascend 基线 A/B | torch 2.10 + torch_npu 2.10.0.post4；vllm-ascend A5 build |

`.venv_baseline` 原本装的是 A2 build，启动时报
`Current device type: AscendDeviceType.A5 does not match ... A2`；本基线用
`/home/userlist/ww/py_venv/lib/python3.12/site-packages/vllm_ascend`（A5）覆盖后修复，
`_build_info.py` 现在为 `__device_type__ = 'A5'`。这不是代码改动，是环境修复。

## 2. 逐文件 CPU 测试（冻结值）

命令（每个文件一个全新进程，避免进程内状态互相污染）：

```bash
# 脚本与完整输出
/tmp/m0_perfile_baseline.sh
# -> /tmp/m0-perfile-native.tsv, /tmp/m0-perfile-native.log
```

结果：**88 个测试文件，74 个 `rc=0`，9 个 `rc=5`，5 个 `rc=1`。**

`rc=5` 是 pytest 的"没有选中用例"（文件里全是 `npu`/`tilelang` marker，被
`-m 'not npu and not tilelang'` 过滤），不是失败：

`test_boot_rehearsal.py`, `test_decode_token_split_device.py`, `test_native_prefill_oracle.py`,
`test_sw_pipeline_device.py`, `test_tilelang_ascend950_decode.py`,
`test_tilelang_ascend950_moe.py`, `test_tilelang_ascend950_prefill.py`,
`test_tilelang_decode.py`, `test_worker_contract.py`。

`rc=1` 的 5 个，逐个定性：

| 文件 | 现象 | 定性 |
|---|---|---|
| `test_attn_contract.py` | `TileInferAttentionBackend.get_name() == 'CUSTOM' != 'TILEINFER'` | v2 代码/测试漂移，先于本次工作存在 |
| `test_model_profiles.py` | dummy profile 的 "MLA" 分支不抛异常 | 同上 |
| `test_moe_ops.py` | 非法 expert id 未抛 `ValueError` | 同上 |
| `test_platform_contract.py` | 能力总数 62 != 65 | 同上 |
| `test_no_vllm_ascend.py` | 扫描到 `tileinfer/bridge/*` 的 vllm_ascend import | **本次 S1 的分层反转**，见下 |

最后一条要说明：跑基线时 v3 骨架已经落地，旧的"全仓库禁止 vllm_ascend"守卫把
`tileinfer/bridge/` 判成违规，所以 `rc=1`。S1 把守卫改成"只有 `bridge/` 可以 import"后，
该文件 `rc=0`（2 passed）。这是**预期中的唯一一次基线红转绿**，不是回归。

## 3. v2 端到端 smoke（`ti_trace`）

```bash
.venv_native213/bin/python scripts/ti_trace.py run --preset smoke
# 2026-11-12 14:49:32 -> trace_out/20261112-144932-smoke/summary.md
```

| 项 | 2026-11-12（本次） | 2026-11-10（文档旧值） |
|---|---|---|
| 就绪 | 57 s | 57 s |
| TTFT | **4963.9 ms** | 814.7 ms |
| TPOT | **134.28 ms** | 106.62 ms |
| 吞吐 | **3.45 tok/s** | 7.77 tok/s |

本次参数为 `--attention-backend CUSTOM --enforce-eager --max-model-len 9216
--max-num-batched-tokens 9216 --max-num-seqs 32 --gpu-memory-utilization 0.75`，端口 8111。

**怎么读这组数**：首次请求的 TTFT 约 5 s，远大于稳态 90 ms 量级，和"热身后 814.7 ms"
不是同一口径；最可能的解释是 v2 在首请求路径上还有 TileLang/JIT 编译（或 cache 冷）。
这正好是 v3 硬约束"不在请求路径上编译"的反面证据。M0 只冻结事实，不在这里解释完；
M2 的预编译门会用同一 preset 复测，要求首请求 TTFT 回落到稳态量级。

## 4. vllm-ascend 基线服务（A 臂）

启动器 `/tmp/run_baseline_ascend.sh`：

```bash
source scripts/env-ascend102.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/9.1.1/atb/cxx_abi_1/lib:${LD_LIBRARY_PATH}
.venv_baseline/bin/python -m vllm.entrypoints.cli.main serve /home/models/Qwen3.6-35B-A3B \
  --served-model-name qwen36 --port 8321 --trust-remote-code \
  --max-model-len 9216 --max-num-seqs 32 --gpu-memory-utilization 0.88
```

启动证据（三次启动一致）：`init engine` 23.37 / 23.44 / 23.84 s（编译 12.1–12.4 s），
权重 65.52 GiB，KV 39.09–39.77 GiB，capture sizes `[1,4,16,32]`，
`FULL_AND_PIECEWISE`，HBM 117730/131072 MB。

### 4.1 长跑 A/B（A1、A2 两次，间隔重启）

`/tmp/ab_short.py`（已复制为 `benchmarks/v3_ab.py`），每场景 3 次，中位数；两臂同 venv 依赖、同参数、同端口模型名。
`long4k` 名义场景实际 prompt 只有 **534 tokens**（脚本里的长文并没有 4K），4K 另测，见 §5。

| 场景 | A1 TTFT/TPOT/吞吐 | A2 TTFT/TPOT/吞吐 | 中位数 |
|---|---|---|---|
| ttft_short_1tok | 96.29 / – / – | 97.58 / – / – | TTFT 96.94 ms |
| decode_128tok | 97.00 / 8.77 / 114011 | 95.98 / 8.72 / 114729 | 96.49 / 8.745 / 114370 |
| decode_256tok | 94.51 / 8.78 / 113943 | 96.85 / 8.73 / 114552 | 95.68 / 8.755 / 114248 |
| long4k（实际 534 tok） | 95.11 / 8.82 / 113399 | 96.33 / 8.76 / 114219 | 95.72 / 8.79 / 113809 |
| conc_4 | 193.68 / 10.84 / 325.92 | 187.91 / 10.84 / 326.10 | 190.80 / 10.84 / 326.01 |
| conc_16 | 196.34 / 13.29 / 1078.13 | 190.52 / 13.28 / 1088.98 | 193.43 / 13.285 / 1083.56 |

TTFT/TPOT 单位 ms，吞吐为输出 tok/s。A1→A2 漂移：TTFT ≤3%，TPOT ≤0.8%，吞吐 ≤1%。
这台机器在 10 分钟窗口内的噪声足够小，1.10× 的门有可判性。

### 4.2 规范输出（C2 的模型侧 oracle）

请求 `/tmp/req_qwen36.json`：

```json
{"model":"qwen36","messages":[{"role":"user","content":"The capital of France is"}],
 "max_tokens":16,"temperature":0,"top_p":1,"logprobs":true,"top_logprobs":5}
```

基线响应 `/tmp/baseline_fia.json`：文本
`"Here's a thinking process:\n\n1.  **Analyze User Input:**"`，usage 15/16，
首 token `Here` logprob `-0.0302655640989542`，top-5 为
`Here -0.030266 / Thinking -3.530266 / The -8.655266 / Okay -9.030266 / 用户 -10.030266`。
这组数已写进 `tileinfer/manifests/qwen3_6.py` 的 `benchmark.top5`。

## 5. 真 4K prompt（补充，M1 验收用）

`/tmp/long_ttft.py` + `/tmp/long4k_prompt.txt`（56 段重复，**3939 prompt tokens**，
`max_tokens=32`，3 次中位数）：

| 臂 | TTFT | TPOT |
|---|---|---|
| vllm-ascend 基线 | **262.16 ms** | **8.911 ms** |
| TileInfer v3（同一 vllm-ascend 栈 + bridge） | **262.24 ms** | **8.902 ms** |
| 比值 | **1.000** | **0.999** |

## 6. 冻结声明

* 本页数字在 `9670c60` + 上述环境上可复现；换 CANN/驱动/模型/KV 参数即失效。
* v2 的四项旧账（图捕获没接、TPOT 2.53×、TTFT 11.8×、MTP 未做）在本页之后仍然成立；
  M1 的 v3 结果见 [`m1-v3-bridge.md`](m1-v3-bridge.md)，不要拿两页混着比。
* 逐文件基线是"每文件独立进程"口径；整套 suite 一起跑会有跨文件污染，v2 文档里的
  80 failures 是那个口径，两者不矛盾。
