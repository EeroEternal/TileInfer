# 性能候选登记表（candidate registry）

本表是 **E4 / C2** 的落点：每一项已落地的性能改动都必须能回答四个问题 ——
它叫什么、怎么关掉、默认是开还是关、实测 ratio 是多少、怎么测出来的。
数字出处一律是 `docs/performance.md` 的对应小节；本表只做索引与汇总，**不重复论证**。
ratio > 1 表示"开启比关闭快"，单位是倍。

## 开关的三种形态

| 形态 | 例子 | 用途 |
|---|---|---|
| 环境变量（每次调用都读） | `TILEINFER_MOE_SORTED` | 常规回退 |
| 环境变量（构造期读一次） | `TILEINFER_PREFILL` | 路径级切换 |
| 文件开关（`TILEINFER_FLAG_WATCH=1` 时生效） | `/tmp/tileinfer_flag/moe_sorted=1` | **同进程配对 A/B**，见 §27 |

文件开关由 `tileinfer/serving/flags.py` 提供；默认关闭，关闭时**不触碰文件系统**。

## 已登记、默认**开**（有 ratio）

| 候选 | 开关 | measured_ratio | 怎么测的 | 出处 |
|---|---|---|---|---|
| MoE 专家循环：一次排序 + 连续段 | `TILEINFER_MOE_SORTED` / `moe_sorted` | **2.569**（op，同进程） | `benchmarks/moe_dtype_ab.py`；对 fp32 oracle 与旧循环逐位相同 | §21 §29 |
| 融合 SwiGLU（`npu_swiglu`） | `TILEINFER_MOE_SWIGLU` / `moe_swiglu` | 3.0（op）；**1.043**（端到端配对） | `benchmarks/swiglu_ab.py`；两形状下与 torch 形式逐位相同 | §29 §30 |
| GDN decode 常量缓存 | `TILEINFER_GDN_CACHE` / `gdn_cache` | **1.231**（op） | `benchmarks/gdn_cache_ab.py`；输出与状态 `torch.equal` 为真 | §31 |
| 4-tap 卷积：向量化 + 免同步 | `TILEINFER_GDN_CONV_FAST` / `gdn_conv_fast` | **1.241**（端到端配对，4/4 为正） | `benchmarks/ab_conv_pair.sh` | §32 |
| GDN `solve_tril` Newton 倍增求逆 | `TILEINFER_GDN_SOLVE_BATCH=0` 回退 | 孤立 **45.6×**；整链 1.91×；端到端 **1.04×**（噪声带内） | `benchmarks/bench_gdn_stages.py`；129 个 oracle 双状态通过 | §58 §59 |
| prefill 走 paged-prefill 内核 | `TILEINFER_PREFILL`（`=0` 回退参考实现） | 内核占比 **1.000**（100 step，reference=0） | §18.4 同环境 A/B + 三方 oracle | §18 |
| MoE 全部专家批量化（两次 `bmm`） | `TILEINFER_MOE_BMM` / `moe_bmm` | **1.480**（fp32）/ **5.587**（配 bf16） | `benchmarks/moe_bmm_ab.py`；对 oracle 与循环路径逐位相同（fp32 下） | §53 §54 |
| MoE matmul 用输入 dtype（bf16） | `TILEINFER_MOE_FP32=1` 可恢复 fp32 | 循环上 1.251；配 bmm 合计 **5.587** | `benchmarks/moe_bmm_ab.py`；对 oracle 8.0e-5（fp32 是 5.3e-5） | §54 |
| MoE 校验缓存（不每次校验 id） | `TILEINFER_MOE_STRICT=1` 可恢复每次校验 | **1.107**（端到端配对，4/4） | `benchmarks/paired_tpot.py` | §42 |

## 已登记、默认**关**（负结果或未采用，保留证据）

| 候选 | 开关 | 默认 | measured_ratio | 结论 | 出处 |
|---|---|---|---|---|---|
| 卷积走分组 `conv1d` | `TILEINFER_GDN_CONV_LIB` / `gdn_conv_lib` | **关** | **1.004**（无效应） | 与向量化形式等价，保留更简单的实现 | §42 |
| GDN 批次元数据主机缓存 | `TILEINFER_GDN_HOST_CACHE` / `gdn_host_cache` | **关** | **0.999** | 无效应；同步单价被高估 | §38 §42 |
| GDN decode 免主机同步（设备侧 gather） | `TILEINFER_GDN_NO_SYNC` / `gdn_no_sync` | **关** | **0.453**（慢 2.2×） | 三次 MB 级搬运 ≥ 一次同步 | §33 |
| 按步共享 ragged triple | `TILEINFER_ATTN_STEP_SHARE` | **关** | 无可测收益（<1%） | 小张量算子合并低于分辨率 | §24 |
| MoE 零主机读取解码路径 | `TILEINFER_MOE_DEVICE_PATH` | **关** | **0.818**（慢 22%） | 补齐 bmm 的算术量吃掉同步收益（§78 同时修掉一个真 bug） | §70 §78 |
| 卷积状态单次连续写入 | `TILEINFER_GDN_CONV_STATE1` | **关** | **0.926**（同进程配对） | 曾误报 2.77×，是机器漂移（§83） | §82 §83 |
| 卷积更新免主机读取 | `TILEINFER_GDN_CONV_CAPTURESAFE` | **关** | 未测（为图捕获准备） | 捕获未成立，暂不启用 | §72 |
| GDN 卷积融合内核 | `TILEINFER_GDN_CONV_KERNEL` / `gdn_conv_kernel` | **关** | 内核级 **42.5×**；**端到端 0.995（无收益）** | 输出 0.37%、状态逐位相同；配对 4/4 有效，极差 2.3% | §88 §89 §93 |
| ACLGraph 捕获 | `TILEINFER_ACLGRAPH` + `TILEINFER_COMPILE` + `TILEINFER_DYNAMO_OPAQUE` | **关** | — | 八层阻塞，第八层是平台限制（捕获期禁止流同步）；且需先付 22% 成本 | §66–§80 |

## 用这张表的方式

1. 怀疑某个数字时，先按"出处"跳过去看**怎么测的**（同进程 op 级 / 端到端配对 / 跨运行）——
   **跨运行的绝对值不可比**（§35、§83），只有配对差可引用；
2. 线上出问题要回退时，按"开关"列逐个关；每一项都保证"关掉 == 旧行为"，且都有对应测试；
3. 新增候选时：先挂开关（默认关）→ 同进程配对/op 级测 → 记录 ratio → 再决定默认值 → 补进本表与
   `performance.md` 的对应小节。
4. **铁律（§83.4）**：没有"同一进程内交替 A/B ≥4 对"的配对差，不得写入 ratio。
