# Performance

Numbers from the reference machine (single `Ascend950PR_9579`, CANN 9.3.0, official
`tilelang==0.1.15`, bf16).  Nothing here is a promise — it is a baseline with the methodology and
the known limits written down, so the next person can tell an improvement from a regression.

> **2026-10-13(权威数字)**:当前主力场景 **Qwen3.6-35B-A3B**(混合 GDN + d256 full-attn)图模式
> 双源 A/B 见下方「图模式双源全指标」节;完整流程、逐场景 P50/P99 与原始 JSON 索引在
> `.agent_loop/PERFORMANCE_REPORT.md`(第 5 节,TEMPORARY_PASS 比值口径)。本文件其余章节为
> Qwen2.5-1.5B/d128 时代的库级基线,方法论仍有效,绝对值不可与 Qwen3.6 数字混读。
>
> **2026-10-14(阶段 3 更新)**:下表比值仍为 E15 干扰期 TEMPORARY 口径(结构未变,短上下文
> 结论健康窗复核一致,见 P1 节);其列出的长上下文落差已按阶段三 L1-L3 逐项处置——KI-1 已修
> (P6)、池无关 ABI 就绪(P7)、prefill 首见编译窗已消除(P3,110s→<5s)、split-KV opt-in
> 在役兑现 8K TPOT 1.411x/16K 2.037x(P8 节)。终版汇总见 PERFORMANCE_REPORT §10。
>
> **2026-10-14(P0 权威口径基线,当前对照原点)**:下节为按权威命令口径固化的双侧
> (FIA / TileInfer 最优配置)全场景 P50/P90/P99 基线——后续所有性能提升以此表为对照原点。

## P0 权威口径基线(Qwen3.6-35B-A3B,FIA vs TileInfer,P50/P90/P99,2026-10-14)

**口径**: `vllm bench serve --backend openai --endpoint /v1/completions --dataset-name random
--random-range-ratio 0 --num-prompts 32 --ignore-eos --temperature 0 --metric-percentiles
50,90,99`(场景=L/O/C 矩阵);serve 参数 `--max-model-len 32768 --max-num-seqs 32
--gpu-memory-utilization 0.90 --language-model-only`(图模式 FULL_AND_PIECEWISE,bucket
[1,4,16,32],chunked prefill 默认)。**off 侧** = `TILEINFER_VLLM=0`(FIA,vLLM 原始昇腾算子
路径);**on 侧** = `TILEINFER_VLLM=1 TILEINFER_KV_TILE_PAGES=16 TILEINFER_ALLOW_SPLIT_KV=1`
(TileInfer 当前最优配置:decode+prefill 接管 + split-KV opt-in)。环境:2026-10-14
16:25-16:41 单卡独占,双侧背靠背;原始 JSON 在 `.agent_loop/run/bench_results/p0_*.json`
(gitignored)。**在役探针(数字有效前提)**:on 侧 fallbacks=0/exceptions=0/capture_calls=40/
ready_plans=6/prefill_calls=4400/prefill_variants=9;off 侧 serve 日志零 TileInfer 接管。

### 吞吐(输出 tok/s)

| 场景 | FIA | TileInfer | TI/FIA(>1 优) |
|---|---|---|---|
| 128/128/C1 | 105.8 | 93.9 | 0.887 |
| 128/128/C8 | 412.4 | 452.3 | 1.097 ✓ |
| 128/512/C8 | 427.0 | 487.7 | 1.142 ✓ |
| 8K/128/C8 | 159.7 | 34.0 | 0.213 |
| 16K/128/C8 | 95.0 | 11.5 | 0.121 |

### TTFT (ms)

| 场景 | FIA P50 | FIA P90 | FIA P99 | TI P50 | TI P90 | TI P99 | TI/FIA P50 |
|---|---|---|---|---|---|---|---|
| 128/128/C1 | 93.7 | 99.5 | 139.8 | 118.5 | 124.3 | 182.5 | 1.264 |
| 128/128/C8 | 188.3 | 208.8 | 209.1 | 328.2 | 343.5 | 344.1 | 1.743 |
| 128/512/C8 | 190.6 | 194.9 | 200.2 | 336.1 | 343.6 | 352.6 | 1.764 |
| 8K/128/C8 | 1059.8 | 3014.4 | 4426.9 | 5836.9 | 16497.9 | 25123.5 | 5.507 |
| 16K/128/C8 | 1720.5 | 5632.3 | 8704.0 | 14450.2 | 48402.0 | 77516.2 | 8.399 |

### TPOT (ms)

| 场景 | FIA P50 | FIA P90 | FIA P99 | TI P50 | TI P90 | TI P99 | TI/FIA P50 |
|---|---|---|---|---|---|---|---|
| 128/128/C1 | 8.77 | 8.78 | 8.78 | 9.88 | 9.88 | 9.88 | 1.126 |
| 128/128/C8 | 18.5 | 18.6 | 18.6 | 15.2 | 15.2 | 15.2 | 0.823 ✓ |
| 128/512/C8 | 18.8 | 18.9 | 18.9 | 15.8 | 15.8 | 15.8 | 0.840 ✓ |
| 8K/128/C8 | 41.1 | 42.7 | 43.8 | 192.9 | 201.7 | 203.1 | 4.696 |
| 16K/128/C8 | 69.9 | 71.9 | 72.5 | 589.3 | 602.5 | 611.9 | 8.429 |

### ITL (ms)

| 场景 | FIA P50 | FIA P90 | FIA P99 | TI P50 | TI P90 | TI P99 | TI/FIA P50 |
|---|---|---|---|---|---|---|---|
| 128/128/C1 | 8.78 | 8.99 | 9.61 | 9.88 | 9.96 | 10.2 | 1.125 |
| 128/128/C8 | 18.4 | 19.1 | 20.1 | 15.2 | 15.3 | 15.7 | 0.826 ✓ |
| 128/512/C8 | 18.7 | 19.4 | 20.2 | 15.8 | 16.4 | 16.7 | 0.846 ✓ |
| 8K/128/C8 | 19.2 | 130.6 | 144.3 | 29.9 | 740.0 | 1177.7 | 1.562 |
| 16K/128/C8 | 20.8 | 146.2 | 157.4 | 44.6 | 1670.3 | 2122.4 | 2.143 |

**解读(E41)**:①短上下文 C8 净正(吞吐 +9.7~14.2%、TPOT/ITL 0.82-0.85x)——TileInfer decode
接管在并发档即有收益;C1 轻微落后(TTFT 1.264x/TPOT 1.126x,与 E40 短上下文税同向)。
②**长上下文×并发引擎级病态(新暴露的头号弱势)**:C8 下 8K/16K 全指标 4.7-8.4x 落后
(duration 8K 120s vs 26s、16K 355s vs 43s)——与 E40 单流结论(TPOT 1.09-2.04x 改善、TTFT
持平)方向相反;病态负载结构 = chunked prefill 混合调度下 prefill chunk 与 decode 交织,
「单流内核级健康(重扫消除 1.14-1.26x)」在 C8 引擎路径上不成立。W1 重新定义为
「长上下文×并发引擎级 prefill/decode 交织」,机理候选与取证计划见 EVIDENCE E41。

## 图模式双源全指标(Qwen3.6-35B-A3B,自研源+官方源,TI/FIA 比值)

**口径注记(读数前必读)**:①测量期存在外部全盘扫描干扰(E15/H6,禁 kill 的他人进程),图模式
每步主机路径整体退化(~130ms 量级),**绝对值失真,以下均为双侧同条件 TI/FIA 比值**(TEMPORARY
口径,环境恢复后应按正常口径重跑);②split-KV 关闭(KI-1)、M-MoE 未接入(0.86x)、gdn gating/
chunk-wu 未接入(opt-in 机制就绪)——即对比的是"TileInfer decode+prefill 接管 vs FIA 全原生",
比值不含上述未启用项的贡献;③比值 = TileInfer/FIA,吞吐类 <1 为落后、TTFT/TPOT 类 >1 为落后;
④在役探针通过是数字有效前提(6 plan READY、capture_calls=40、fallbacks=0、markers 随 replay
增长、logprobs 逐位一致)。

自研源(中位;场景 1tok/128tok/512tok/8K/16K/并发 4/16/32):

| 场景 | 指标 | FIA | TileInfer | TI/FIA |
|---|---|---|---|---|
| 单 token(33 in) | TTFT ms | 563.4 | 628.9 | 1.12 |
| 128 tok 生成 | TPOT ms | 130.8 | 137.9 | 1.05 |
| 128 tok 生成 | 吞吐 tok/s | 7.64 | 7.25 | 0.95 |
| 512 tok 生成 | TPOT ms | 131.0 | 142.5 | 1.09 |
| 8K in+128 gen | TTFT / TPOT | 5470 / 136.7 | 15917 / 269.2 | **2.91 / 1.97** |
| 16K in+128 gen | TTFT / TPOT | 11298 / 142.9 | 46910 / 402.4 | **4.15 / 2.82** |
| 16K 纯 prefill | TTFT ms | 11298 | 46863 | **4.15** |
| 并发 4 / 16 / 32 | 吞吐 tok/s | 24.21 / 78.69 / 126.55 | 23.01 / 72.69 / 115.33 | 0.95 / 0.92 / 0.91 |

官方源(vllm bench serve,9 场景×3 中位;输出 tok/s、请求 req/s、TTFT/TPOT/ITL P50/P99 完整表
见 PERFORMANCE_REPORT §5.4 与 §8 终版汇总)关键行:1tok TTFT P50 1.05;128/512tok 吞吐
0.94-0.95、TPOT P50 1.06;并发 4/16/32 吞吐 0.96/0.95/0.94;8K 吞吐 **0.33**/TTFT P50 **4.04**;
16K 吞吐 **0.21**/TTFT P50 **5.76**。

解读:短上下文与并发(≤512 输入、≤32 并发)双源一致 TI≈FIA;长上下文 2-6x 为测量期结构性
(KI-1 split-KV 未启用 + D22 prefill 编译窗 + chunked prefill KV 重扫)——阶段三已处置:
KI-1 修复(P6)+ 首见编译窗消除(P3)+ split-KV opt-in 兑现 decode 侧 1.41x/2.04x(P8);
纯 prefill 剩余差距中的 **chunked prefill KV 重扫**已消除(见下方「Prefill KV 重扫截断」节,
prefill 内核级 1.14-1.26x);引擎级端到端 TTFT 复核待补。

## P1/H10 短/长上下文长度自适应分流(gdnwu 融合核,引擎同侧 A/B,2026-10-14)

**方案**:prefill 总 token 数 T ≤ 阈值(缺省 128,微基准交叉点 E26)走 `gdn_chunk_wu` 融合核,
超阈值回退原 chunk 链;`TILEINFER_OPS` 显式含 `gdnwu` 才安装(opt-in,默认路径零改动),
`TILEINFER_GDNWU_MAX_TOKENS` 可覆盖阈值。机制经五轮在役探针验证零异常
(gdn_wu_calls/long_skips/fallbacks 语义全部符合设计,如健康成对轮 900/2040/0)。

**验收数字(健康成对,环境对齐判据:双侧 decode 同代码,8K TPOT 比值 1.0000;
wuoff=原链 vs wuon=融合核,背靠背)**:

| 场景 | wuoff | wuon | wuon/wuoff |
|---|---|---|---|
| 1tok(33 in)TTFT | 110.6 ms | 109.4 ms | 0.989 |
| 128tok(33 in)TTFT / TPOT | 108.2 / 9.26 | 109.1 / 9.28 | 1.008 / 1.002 |
| 512tok TTFT / TPOT | 108.6 / 9.55 | 108.6 / 9.56 | 1.000 / 1.001 |
| 8K TTFT / TPOT | 1487.6 / 17.16 | 1474.9 / 17.16 | 0.991 / 1.000 |
| 16K TTFT / TPOT | 3870.6 / 25.19 | 3868.8 / 25.20 | 1.000 / 1.000 |
| 并发 4/16/32 TTFT | 232.8/315.9/461.2 | 226.3/307.0/455.6 | 0.972/0.972/0.988 |

**目标形状补测(138-token prefill 进融合核,`TILEINFER_GDNWU_MAX_TOKENS=192`;同轮内用
原链场景做暖机漂移内校正)**:138tok 融合核 0.926 vs 同轮 522tok 原链 0.926 → 校正后
**收益 = 1.00x(零)**。微基准预期(E24:T128 省 ~6ms/步≈6%)未在引擎端到端兑现。

**结论**:①长上下文零劣化 proven(全部 |Δ|≤1.5%);②短上下文引擎级收益 ≈0(健康成对
−1.1%~+0.8%;目标形状校正后 1.00x),未达 ≥5% 验收判据——微基准收益 ≠ 引擎收益
(与 M-MoE E19/gating E21 同型教训,在役 A/B 为最终裁决);③处置:**不默认接入**,机制/
内核/测试保留为 opt-in(`TILEINFER_OPS=...,gdnwu`),交用户裁决是否重启(路线=E24 判读 3
平方链全 cube 化)。VERDICT: TEMPORARY_PASS。

方法注记:①健康判定看门狗(`.agent_loop/run/run_gdnwu_ab_health.sh`)以环境对齐比值
(8K TPOT |Δ|≤3%)+ 机制探针前提自动产出成对数据;②背靠背冷启动 TTFT 探针存在暖机漂移
(后跑侧长 prefill 快 8-18%),成对结论必须同轮内校验;③早期"健康带"判据(8K TTFT<700ms)
系误判——drift 轮 462ms 实为 prefill_fallbacks=980(全回退 FIA)的病态数字,TI 接管稳态即
~1700ms(E30 修正)。

## P3 prefill 预热覆盖:canonical 变体量化 + ladder 预热(2026-10-14)

**问题**:prefill 内核按 `(batch, kv_heads, group, dim, page, pool_pages, num_tiles, block_q,
dtype)` 编译变体,真实流量形状空间近连续(任意 prompt 长度×任意并发),首见形状惰性编译
~110s(D22)形成 TTFT 编译窗,是图模式 A/B 长上下文 2-6x 落差的结构性成因之一。

**方案(D39)**:变体量化归一——批维 padding 到 `bucket(max_num_seqs)+1` 行(末行为 trash 行:
0 页、整页 last_page_len→kv_len=0;padding tile 指向它→内核页循环零迭代、输出不读);
tile 维向上量化到 2 的幂(padding tile `q_len=0`:打包/解包跳过、RowThr=-1 全掩码)。
canonical 变体 = `(B, 2^⌈log2 t⌉)`,prewarm 按 token 预算(`max_num_batched_tokens`)推导
ladder:顶 = `next_pow2(ceil(B·group/128)+max_reqs)` 与 `PREFILL_MAX_TILES` 取小 →
变体数 = log2(top)+1 ≈ 11 ≤ 上限 32,覆盖引擎可调度的**全部** prefill 形状。

**验收数字(E34/E34b)**:

| 项 | 数字 | 判据 |
|---|---|---|
| 设备测试(量化路径数值 vs reference_attention,TOL 3e-2) | 4/4 绿(非 2 幂 t=19→32、批 1→5 trash 行、append、decode 行混批) | 通过 |
| ladder 预热后任意 ladder 内形状可用性 | qo∈{1,17,97,257,397,512} 全部即时 usable | 通过 |
| 首见形状首步(API 级,qo=397) | **0.008s** | <5s ✓ |
| 引擎首发首见形状 ~700/~1500/~2600 tok TTFT | **1.43 / 2.59 / 1.54 s** | <5s ✓ |
| 引擎在役探针 | prefill_calls=30(3 请求×10 层,首发即接管)、prefill_fallbacks=0、prefill_exceptions=0 | 通过 |
| decode 侧不受影响 | fallbacks=0、capture_calls=40、ready_plans=6、markers 增长、exceptions=0 | 通过 |

**成本注记**:预热单价 ~109s/变体(与 D22 惰性编译同价)× ladder ≈11 → 冷启动一次性
~16-20min(实测 ladder 9 新编译 16min;T=512/1024 由启动 dummy 的后台编译并行完成),
磁盘缓存持久化后热启 0.2s;量化 waste ≤2x tile 数但 padding tile 指零页行,实际零 KV
计算;超 ladder 形状(配置变更)仍走惰性编译回退,行为同 D22。VERDICT: PASS。

Reproduce with:

```bash
.venv/bin/python -m pytest tests/test_vllm_ascend_prefill_prewarm.py -q -s
bash .agent_loop/run/probe_p3_prefill_warm.sh
```

## P4 M-MoE 一轮重试(最终轮):结构性瓶颈定位,FINAL TEMPORARY 关闭(2026-10-14)

**背景**:M-MoE 融合内核 E19 收口于 0.86x 设备时间(TEMPORARY);本轮为契约内的最终重试,
攻击面=块调度合并 / FFI 主机批处理 / E 权重预 NZ 布局,门槛 ≥1.0x aclnn(设备时间)才接入。

**干净环境基线复现(E35)**:E19 的 chrome-trace 绝对值系当期环境干扰虚高 ~15x(E15 期),
当日干净环境重测——decode_b32:TI 793µs/步 vs aclnn 链 603µs/步(**0.76x**);prefill_400:
1277.7 vs 1002.8µs(**0.78x**);墙钟 geomean 0.898x(与 E19 的 0.891x 一致,比值口径稳定)。

**结构定位(E36)**:
1. 流水无关:K 主循环 T.Pipelined 双缓冲(去守卫+哨兵 tile 重构后确认真流水化,内核源码
   比对)对设备时间零影响(gate_up 474.1 vs 475µs)——串行延迟链不是瓶颈。
2. 带宽记账:TI 读权重 ~955MB/步@793µs≈1.2TB/s;aclnn 同权重 @603µs≈1.62TB/s。差距在
   GM→L1 **nd2nz 格式转换吞吐**(~50GB/s/核):W traffic 每步固定不可压缩(每 (tile,nt)
   单元必读一次专家权重切片)。
3. 预 NZ 直载在内核层不可表达:tilelang 的 `T.copy` GM→L1/UB 恒定发射 nd2nz;NZ 标注
   (`make_ascend_compact_nz_layout`)仅支持 UB 内 vsstb SIMD 逐行散转(更慢);L0B 64KB
   容量封顶 W-op 几何(Z3 调度器实证)。追平 aclnn 需要「权重以免转换形态进入 MTE」=
   **vLLM 集成层预 NZ 权重布局**(内核签名之外),超出本轮内核重试范围。
4. 旋钮扫(7 配置,数值全零差):最优 spec = gate_up `block_n=256` + down
   `block_n=64, block_k=512`(down 单 K 迭代、W2 128KB 全连续读)。
5. 其余攻击面评估:块调度=persistent 28 核+空块早退已在役(E19),守卫移除实验证明调度
   开销非因子;FFI 主机批处理=今日主机仅 ~23µs/步(wall 0.816 vs device 0.793ms)且图模式
   replay 无主机路径,不改变设备时间门槛。

**最终数字(E36b,最优 spec 已固化进 benchmarks/bench_moe_grouped.py)**:

| 口径 | 场景 | aclnn 链 | TileLang 融合 | 比值 |
|---|---|---|---|---|
| 设备时间(µs/步,chrome trace) | decode_b32 | 602.1 | **740.9** | **0.813x** |
| 设备时间(µs/步) | prefill_400 | 1002.5 | **1191.2** | **0.842x** |
| 墙钟 geomean(7 场景) | — | — | — | **0.974x**(b1 1.654 / b4 1.160 / b16 0.843 / b32 0.859 / p128 0.851 / p400 0.850 / skew 0.829) |

**判定**:设备时间门槛 ≥1.0x 未达(内核内部优化空间实测 ~6%,缺口 ~19-24% 为结构性的)→
按预设契约**不接入,M-MoE FINAL TEMPORARY 关闭,此后不再重试,交用户裁决**。重启路线=
集成层预 NZ 权重布局(权重加载时一次转 NZ,内核免 nd2nz)+ 本轮 bench 已固化的最优 spec。

Reproduce with:

```bash
source .agent_loop/run/env.sh
.venv/bin/python benchmarks/bench_moe_grouped.py --out .agent_loop/run/p4_final_wall.json
.venv/bin/python benchmarks/bench_moe_devtime.py   # 设备时间口径(门槛判据)
```

## P8 split-KV 引擎启用 + 长上下文 A/B(2026-10-14,opt-in 兑现)

**机制**(E39):plan 的 schedule 以「满额」构建(dummy_lens=表宽,每请求
max_splits=ceil(width/kv_tile_pages)),split 内核把每 tile 页循环钳到运行期真实页数
(eff=min(plen, req_pages−start)),空 tile 经 l→1 写 0/−1e30 槽被 merge 零权重吸收——
同一预烧 plan 服务任意页数分布,满足图捕获固定 plan 约束;finalize_plan 预编译 split+merge
(否则 capture 窗内首编 ~2.5min)。kv_tile_pages=16。

**引擎 A/B**(2026-10-14 10:14-11:00,NPU 空闲窗,双侧背靠背,r=3 取中位):

| 场景(自研源,ms) | plain(基线) | split-KV on | 比值 |
|---|---|---|---|
| long8k_128tok TPOT | 17.62 | **12.49** | **1.411x**(−29.1%) |
| long16k_128tok TPOT | 25.60 | **12.57** | **2.037x**(−50.9%) |
| long8k TTFT | 1556.6 | 1523.7 | 1.022x(噪声带) |
| long16k TTFT | 3938.2 | 3942.7 | 0.999x |
| decode_128tok TPOT(b1) | 9.28 | 9.38 | 0.989x(税 ~1%) |
| decode_512tok TPOT(b1) | 9.56 | 9.91 | 0.965x |
| conc_16 TPOT | 14.20 | 15.90 | 0.893x(并发税,如实记录) |
| conc_32 TPOT | 17.30 | 21.26 | 0.814x(并发税,如实记录) |

官方源(vllm_bench_serve,含并发):8K TPOT 59.0→54.0(**1.093x**)/16K 153.6→140.2
(**1.095x**),TTFT ±2.4%;conc16 1.05x/conc32 1.04x/tok512 1.09x。

**探针**:双侧 fallbacks=0、exceptions=0、capture_calls=40、6 桶 split plan READY、
prefill_calls=980/markers 50960 同构——split 路径真实在役。启动 15min(满额 split plan
首见真实编译 2.7min/桶,一次性;预烧缓存命中后 ~3min)。

**判定**:H16 **proven**。长上下文收益双源兑现(E39 期望带 30-60% 内:8K 29%/16K 51%),
内核级 2.8x(b1/64K)在引擎 16K 收窄至 2.04x(全注意力仅 10/40 层);TTFT 全场景无劣化。
**部署形态=opt-in**:`TILEINFER_KV_TILE_PAGES=16 + TILEINFER_ALLOW_SPLIT_KV=1` 显式开启
服务长上下文部署;短上下文税(b1 ~1%/并发档 11-19%)与编译窗(首见 15min)已定量入档,
全局默认不开启。

Reproduce with:

```bash
source .agent_loop/run/env.sh
PORT=8355 bash .agent_loop/run/p8_ab.sh on    # split 侧
PORT=8356 bash .agent_loop/run/p8_ab.sh off   # plain 基线侧
```

Reproduce with:

```bash
source <CANN 9.3.x>/set_env.sh
python benchmarks/bench_attention.py --preset serving --backend tilelang-ascend950 --dtype bf16
```

## H17/E-B 收口:weight_nz_mode=2 不可用 + dense NZ 零收益(2026-10-14,iter-037)

**结论:官方「集成层预 NZ」路线在 Qwen3.6 + vllm-ascend 0.23.0 上 disproven,不再投入**;
dense GEMM(E-B,设备占比 17.3%)首次评估随之收口 = 零收益。

1. **dense 微基准**(E42):aclnn linear ND vs NZ 权重,`qkv M=512 K=2048 N=6144`
   ratio 0.988/0.995、`o_proj M=512 K=4096 N=2048` ratio 0.999/1.002,maxdiff=0(逐位一致)
   ——aclnn 算子无内核内 nd2nz 惩罚,E36 的 nd2nz ~50GB/s 是 TI 自研 TileLang 内核
   GM→L1 路径特性。
2. **引擎级裁决**(E43):`--additional-config '{"weight_nz_mode": 2}'` 启动即 abort——
   `vllm_ascend/ops/linear.py:98 maybe_trans_nz → npu_format_cast(ACL_FORMAT_FRACTAL_NZ)`
   报 `aclnnNpuFormatCastCalculateSizeAndFormat, error code 161002`(稠密权重形状不被算子
   支持;与 E42 的 npu_grouped_matmul bf16 组合受限同源)。零代码官方开关不可用。
   nz0 背靠背对照与 P0 基线全场景偏差 ≤6.5%(环境无漂移)。
3. **路线意义重定位**:NZ 权重唯一剩余价值 = TI 自研 MoE 内核的 pre-NZ 直载输入
   (E-A 重启候选;P4 节的重启路线在基线侧无兑现可能,自研内核侧另需设备验证)。

Reproduce with:

```bash
bash .agent_loop/run/h17_nz_ab.sh          # nz0/nz2 背靠背 5 场景(nz2 预期启动 abort)
.venv/bin/python .agent_loop/run/e42_nz_gemm_probe.py   # dense ND/NZ 微基准
```

## W1-C8 病态 2×2 隔离:归因 TI prefill 路径(2026-10-14,iter-037,E44)

P0 基线暴露的「长上下文×并发」引擎级病态(16K C8 吞吐 0.121x/TTFT 8.4x/TPOT 8.4x)
经 `TILEINFER_OPS` 域隔离(16K C8 权威口径单景,on 侧配置=split opt-in):

| 配置 | TTFT p50 (ms) | TPOT p50 (ms) | 吞吐 (tok/s) |
|---|---|---|---|
| off(FIA 全部) | 1720 | 69.9 | 95.0 |
| on(TI 全接管) | 14450 | 589.3 | 11.5 |
| **probeA `TILEINFER_OPS=decode`**(TI split-KV decode + FIA prefill) | **1573(健康)** | 84.5 | 80.4 |
| **probeB `TILEINFER_OPS=prefill`**(TI prefill + FIA decode) | **19198(病态)** | 537.3 | 11.8 |

- 病态 100% 随 TI **prefill** 接管域走;decode/split 侧无责(probeA 探针全绿:
  capture_calls=40/ready_plans=6/fallbacks=0;剩余 ~15% 吞吐差=已知 W2 并发税)。
- probeB 的 TPOT 537ms(FIA decode)≈ on 侧 589ms → decode 慢是被同引擎混合步的
  TI prefill 慢步摊派,非 decode 自身问题;E41 候选③(图切换)/④(batched tokens
  交互)出局,剩余候选=①多行 chunk 形状的内核执行开销 vs prefill_step 逐层主机
  准备链(q_pad ~553MB/层/步、每层 2 次主机同步、plan 重建;C8 短 chunk×多行放大)。
- 下一步(H18,已登记):profiler 逐内核+主机段分解(16K C8 负载)→ 修复
  (跨层元数据共享或内核调度)→ 门槛:16K/8K C8 TTFT/TPOT 收窄至 ≤2x、短场景 |Δ|<3%。

Reproduce with:

```bash
PROBES=A bash .agent_loop/run/w1iso_probe.sh   # 或 B;单景 16384x128_c8
```

## W1-C8 权威口径 A/B:H20 split-KV prefill(ktp=64)工程闭环但性能门失败,W1-C8 主因改判(2026-10-15,iter-048)

### 背景
- P0 基线(iter-036)定位头号新弱势:长上下文×C8,8K 吞吐 0.213x/16K 0.121x(TTFT 5.51x/8.40x)。
- 归因链:E44 2×2 隔离(病态 100% 随 TI prefill 接管域)→ E45 主机链(E46 步守卫修复)→
  E46 profiler(mac=0.02、aiv_vec 0.96)→ E47 多行因子否证(线性扩展)→ H19/H20 split-KV
  prefill(双内核 head+frontier,iter-039..047)。

### 本轮工程闭环(提交 ed8d1b3 + d6aa3f6)
- 数值:槽位碰撞根因修复(iter-047 稠密 per-tile 槽位)后设备套件 6/6、探针 7/7 全绿。
- 在役:发射键量化(半区死 tile 填充至 kernel_tiles,SPLIT_MS_COMPILE=16 统一缓存键)+
  split 阶梯预热(顶按最大上下文 split 因子扩展至 1024)后,权威口径 on 侧
  **prefill_calls=4400 全接管、零回退、零异常**、capture_calls=40、ready_plans=6。

### 权威口径 A/B 数字(ktp=64,P50;off 侧 fresh 与 P0 锚点一致,环境无漂移)
| 指标 | 8K/128/C8 | 16K/128/C8 |
|---|---|---|
| 吞吐 TI/FIA | 0.226x | 0.130x |
| TTFT P50 比 | 5.22x | 7.53x |
| TPOT P50 比 | 4.30x | 7.84x |
- 对照:plain prefill 时代(P0)0.213x/0.121x——split 增益 ~+4%,门(≤2x)失败。
- 短场景 TI 接管净正不变:吞吐 1.13x/1.15x,TPOT 0.82x/0.84x;C1 TTFT 0.980x。

### 判定(W1-C8 主因改判,三证据收拢)
1. E46:prefill 内核计算时间 88% 在 main_kernel,mac 利用率 0.02(向量链主导、页串行依赖)。
2. E47:同功多行形状 1.00-1.11x(时间随 tiles 线性=吞吐界)。
3. E63:split 提并行 ×3 引擎无增益——C8 饱和下并行度不是瓶颈。
→ **W1-C8 主因=TI prefill 内核单位功效率(vs FIA 的算法/内核级差距 ~8x),非并行度/
   主机链/重扫**(后两者已修但占比小)。split-KV 路线对该弱势项已到顶。

### 处置
- `TILEINFER_PREFILL_KV_TILE_PAGES` 保持 opt-in 默认 0(默认路径=P0 行为);ktp>0 供
  延迟敏感场景(单 chunk 微基准 stair 1.75x)与后续实验。VERDICT: TEMPORARY_PASS
  (W1 原始命题未达成,证据链完整,交用户决策;后续路线见 .agent_loop/state/DECISIONS.md D57:
  ①长上下文 prefill 步分流回退 FIA(零内核成本,期望 →~0.45x+)②内核效率重写(需新证据))。

## W2 收口:split-KV per-bucket ktp 分带(H21)——并发税归零,推荐部署配置(2026-10-15,iter-056)

### 结论
`TILEINFER_KV_TILE_PAGES_POLICY="1:16,2:16,4:32,8:128"`(未列桶回退全局,≥16 桶回退 0=plain)
在权威口径 A/B 下**同时达成**:并发/短上下文零税 + b1 长上下文 decode 收益保留。
机理(E61/E64 波次模型):split 税=空 tile 与实 tile 同占 AIV 波槽的整波量子化;
分带使 batch×splits 恒 ≤28(1 波)→ 短上下文同波数,税归零;长上下文小批 splits 换并行。

### 权威口径 A/B(h21 链,5 场景 P50,ti0=plain 基线 vs tip=分带;探针双侧 capture_calls=40/fallbacks=0/ktp_policy 在役)
| 场景 | ti0 吞吐 | tip 吞吐 | 吞吐比 | ti0 TPOT | tip TPOT | TPOT 比 |
|---|---|---|---|---|---|---|
| 128/128/C8 | 368.6 | 503.5 | (1.366)† | 19.20 | 13.51 | (0.704)† |
| 128/512/C8 | 422.8 | 544.9 | (1.289)† | 18.39 | 14.11 | (0.767)† |
| 8K/128/C8 | 33.3 | 33.6 | 1.009 | 196.37 | 195.46 | **0.995** |
| 16K/128/C8 | 11.4 | 11.5 | 1.009 | 600.52 | 592.89 | **0.987** |

† 链上短场景比值**判无效**:ti0 侧恰逢主机负载峰(负载 5.3,较 P0 锚点 -21%),tip 侧超锚点 +8%;
E65 矩阵 D 内核级证明 b8/ktp128=16 tiles 1 波 ≈ plain+merge,不可能快 36% → ti0 侧环境凹陷。

### 短场景复跑裁决(h21_fast.sh,双侧背靠背对称窗,E72)
| 场景 | fast-ti0 吞吐/TPOT | fast-tip 吞吐/TPOT | 吞吐比 | TPOT 比 |
|---|---|---|---|---|
| 128/128/C8 | 461.2 / 13.50 | 465.1 / 13.49 | 1.008 | 0.999 |
| 128/512/C8 | 543.2 / 14.11 | 543.6 / 14.10 | 1.001 | 0.999 |

短门 |Δ|≤3% **PASS**(实测 |Δ|≤0.8%);复跑侧带=默认 {1:16,2:32,4:64,8:128}(H21_POLICY 未传),
活动桶 b8 与终带同为 ktp=128,短上下文全带 1 波零税(波次模型),结论迁移至终带成立。
长门(不劣化 1.005)**PASS**(0.995/0.987);b1 长上下文收益 ≥1.3x 由 E65(内核级 1.75x 阶梯)
与 E40(引擎级 8K 1.411x/16K 2.037x)双源支撑。**H21 proven,GATE_OVERALL: PASS。**

### 使用指引(推荐部署配置)
- 启用:`export TILEINFER_KV_TILE_PAGES_POLICY="1:16,2:16,4:32,8:128"`(集成层按策略含 split
  自动置位 TILEINFER_ALLOW_SPLIT_KV;探针 metrics 出现 ktp_policy 视图即在役)。
- 默认开关:**维持 opt-in 不翻默认**(无 env=plain,D41 编译窗依据;窗口预算充足的建议启用,
  b≤8 长上下文 decode 收益 1.4-2.0x,短上下文/并发零代价)。
- 带整定依据:E65 矩阵 D 60 点(32 tiles 软悬崖=merge 底价;b2:16/b4:32 支配候选档,b8:128
  取绑定门裕度)。调整带仅需改 env 串,免重编(不同 ktp 不共享 plan,首见桶首步编译 ~2.5min,
  prewarm 阶梯覆盖)。

## W1 收口:D57-① 长上下文 prefill 步分流回退 FIA——0.226x/0.130x → 0.754x/0.758x(2026-10-15,iter-056)

### 机制与配置
`TILEINFER_PREFILL_MAX_CTX=4096`(token,默认 0=零行为变化):页×128 严格大于阈值的
prefill 步分流回退 FIA(`_prefill_ctx_shunt` 纯函数 + `_PREFILL_SKIP_GUARD` 同步守卫 +
探针计数 `prefill_long_skips`);零内核成本,短 prefill 仍由 TI 接管(prefill_calls=520)。

### 权威口径结果(w1 链 shunt 侧 5 场景 P50,vs P0 锚点;探针 fallbacks=0/capture 40/skips=377)
| 场景 | w1s 吞吐 | p0_ti(基线) | p0_fia | 吞吐 vs FIA(基线→现) | 时长 |
|---|---|---|---|---|---|
| 8K/128/C8 | 120.8 | 36.3(0.226x) | 160.2 | **0.754x**(0.226x→) | 112s→34s |
| 16K/128/C8 | 71.9 | 12.3(0.130x) | 94.9 | **0.758x**(0.130x→) | 355s→57s |
| 128/128/C1 | 94.1 | 95.5 | 105.7 | TPOT 9.86 vs 9.87(1.000) | — |
| 128/128/C8 | 502.9 | 466.3 | 411.7 | TPOT 13.53(短无损†) | — |
| 128/512/C8 | 544.0 | 491.8 | 426.7 | TPOT 14.11(短无损†) | — |

† 短 C8 TPOT 较昨日锚点优 11%,经 E72 fast 对标定=当日窗口漂移(今日 plain 即 13.5,
w1s 不触 decode),分流对短场景零影响;c1 TPOT 比 1.000 直接达标。
**W1/D57-① 裁决 PASS(超预期:0.75x vs 期望 ~0.45x);残余 8K/16K 缺口(0.75x)在 decode
侧(TPOT 56.6/94.8 vs FIA 41.0/69.3),与 prefill 无关。**

### 推荐组合部署(当前最优配置)
`TILEINFER_PREFILL_MAX_CTX=4096` + `TILEINFER_KV_TILE_PAGES_POLICY="1:16,2:16,4:32,8:128"`
(shunt 修长上下文 prefill + 分带保短上下文零税并留 b≤4 长上下文 decode 收益);
两者默认均 opt-in(默认路径=P0 行为)。D57-② 内核效率重写维持暂缓。

## W1 靶点修正:0.75x 残余缺口 100% 在 decode 侧长上下文(2026-10-17,iter-085,E116)

权威口径原始 JSON 复盘(iter-056 w1 链 + P0 锚点):**FIA 侧 decode ITL p50 对上下文长度
平坦**(128ctx→8K→16K = 19.05/19.05/20.69ms),**TI 侧线性劣化**(19.2/37.9/62.8ms);
TTFT p50 8K TI 943 vs FIA 1050ms(shunt 后 prefill 已对齐)。故 D57-② 原表述「prefill
内核效率重写」不解决 0.754x/0.758x——缺口唯一主因 = TI 长 ctx decode attention:

- 机理:W2 proven 分带 b8:ktp=128 → 8K(64 页)仅 1 split,grid=batch×kv_heads=16 块
  (<28 核),每块串行 64 页 × ~30µs/页(K/V 拷贝+4 小 GEMM+SIMT softmax+rescale)≈
  1.87ms/层 ×10 层 = 18.7ms/步,与 ITL 差值精确吻合;带宽下界 ~2.5µs/页 → ~12x 余量。
- 结构盲区:分带按 batch 桶而非 ctx 长度(E40 全局 ktp16 长 ctx 1.41x/2.04x 有效 →
  E61 b8 短 ctx 税 4.5x → E65 分带取零税牺牲长 ctx 并行)。
- 行动(H42):①SIMD softmax(内核自述 decode shape 2.6x,默认 off)零代码 A/B;
  ②per-page 成本微基准分解;③内核双缓冲/并行化改造(门槛 ≥1.5x 现役+短形状零税)。

## P-PROF 流程画像:FIA 侧设备分段占比 + decode 设备-bound 判定(2026-10-15,iter-061,E79)

方法:`p_prof_side.sh`(vLLM 0.23 内置 profiler + torch_npu Text 导出)对 FIA 侧
(纯净 vLLM 基线)采集双窗 timeline——prefill 窗 8192x4_c8、decode 窗 128x256_c8;
`p_prof_summarize.py` 按阶段×层类型×内核聚合(GDN 规则含 w_u)。聚合明细:
`.agent_loop/run/prof_summary_fia.md`(TI 侧补采见 pprof_ti_chain,等设备窗)。

### decode 窗(257 步,设备总 4951.1 ms = 19.3 ms/步;与 h21 TPOT 19.20 ms 同量级)
判定:**TPOT≈设备时间,decode 端到端为设备-bound(主机已隐藏)**——后续 decode 优化
必须以设备时间为靶,主机侧收益上限极小。

| 层类型 | 设备耗时(ms) | 占比 | 最大内核 |
|---|---|---|---|
| MoE | 2663.9 | 53.8% | GroupedMatmul 2397.0(48.4%,avg 116.6µs×80/步)+ routing 等 266.9(5.4%) |
| dense+胶水 | 1566.5 | 31.6% | MatMulV3 1038.8(21.0%,avg 14.9µs×271/步:shared expert 80/步+attn/GDN 投影+router) |
| GDN | 573.7 | 11.6% | fused_gdn_gating 301.9(6.1%)+ RecurrentGatedDeltaRule 164.8(3.3%) |
| full-attn | 47.7 | 1.0% | — |
| other | 99.3 | 2.0% | — |

### prefill 窗(设备总 4095.9 ms)

| 层类型 | 占比 | 最大内核 |
|---|---|---|
| MoE | 35.6% | GroupedMatmul 1237.8(30.2%,avg 429.8µs) |
| dense+胶水 | 29.0% | MatMulV3 603.7(14.7%) |
| GDN | 18.9% | ChunkGatedDeltaRuleFwdH 316.8(7.7%)+ recompute_w_u 241.4(5.9%) |
| full-attn | 6.5% | FusedInferAttentionScore 216.2(5.3%) |
| other | 10.1% | — |

### 热点归因与战略推论(E79)
1. **MoE routed GEMM(FIA)48.4%** 是 decode 第一热点——即 M-MoE 内核线(H27 preNZ)
   对标的替换对象;模型含 shared expert(config shared_expert_intermediate_size=512,
   40 层×2 GEMM/步)是 dense 段 MatMulV3 21% 的重要组成部分。
2. **30% 端到端目标的可达域**:decode 19.3 ms/步中 routed MoE 段仅 ~9.3 ms;H27 preNZ
   达标(内核 0.813x→1.0x aclnn)单独折算端到端 ~3%(MoE 内核段×16-19% 税),不足以
   单独撑起 30% TPOT;其余可达域=TI 路径特有胶水/slot/主机段(待 TI 侧画像定量)、
   dense 段 shared-expert 融合(DeepSeek 模式候选)、GDN(已收口)。**瓶颈证明
   (终止条件 b)暂不触发,等 TI 侧画像补全后裁决。**
3. 派生假设 H28(shared expert 融合)已登记候选池,立项与否待 TI 侧画像确认 dense 段
   在 TI 路径中的占比后定。

## T6 warm-up / batch bucketing 一等公民化:配置驱动的单一真源(2026-10-15)

**问题(隐式双真源)**:TileLang 内核按形状编译(冷编译 ~100 s),服务路径上必须**零编译**
(否则卡住 EngineCore)。做法是启动前 out-of-process 预烧进 `TILELANG_CACHE_DIR`。但历史实现里
"要编译哪些变体"由**两处独立公式**决定:引擎内联的 `TileInferDecodeAttention.prewarm_prefill`
(含 prefill ladder)与 `scripts/prewarm-tileinfer-kernels.py`(只覆盖 decode,且池大小/桶/ladder
顶**要从日志手抄**)。两处一旦漂移,预烧静默失效 → 服务路径掉回惰性编译 → FIA 回退。

**方案(`tileinfer/integrations/prewarm.py`)**:把公式抽成**纯函数**(不 import torch/tilelang/vllm),
引擎与预烧脚本调用同一份实现,消除双真源:
- `bucket_for` / `decode_buckets`:批量向上取整到 `BATCH_BUCKETS`;
- `prefill_ladder_tops` / `prefill_specs`:prefill 的 canonical ladder(`kernel_batch=bucket+1` 的
  trash 行、tiles 取 2 的幂到顶、split 变体按 `max_splits` 追加),顶由 token 预算与 `max_model_len`
  推导;
- `declared_cap`:内核 ABI 的声明容量(2 的幂上取整 / `TILEINFER_POOL_CAP` 钉死),池从内存快照
  漂移也落同桶(L2 池无关 ABI),预烧跨运行命中。
- `build_prewarm_plan`:`(配置) → (decode 桶, prefill ladder spec)` 的唯一入口,`--prefill` 落地到
  预烧脚本(此前 prefill 完全未预烧)。
- `prewarm_plan_to_dict` + `--plan-only --json`:预烧规格的机器可读产物,可在不编译的机器上复算、
  diff、入库——把"预烧与引擎一致"从人工比对日志升级为可复现的产物对比。
- `build_prewarm_argv` + 启动器 `TILEINFER_PREWARM=1`:`scripts/vllm_tileinfer_launch.py` 从 serve
  argv 与模型 `config.json` 推导参数,**serve 启动前自动预烧**(经已测脚本,不重复实现编译);池大小
  由 `TILEINFER_PREWARM_PAGES` 显式给出(服务器快照值,见启动日志 `GPU KV cache size`)。

**验收数字(CPU 契约,`tests/test_prewarm_specs.py`,24 用例全绿)**:

| 项 | 数字 | 判据 |
|---|---|---|
| **漂移守护**:引擎常量/函数 ≡ prewarm 模块 | `KERNEL_PAGE_SIZE`/`BATCH_BUCKETS`/`PREFILL_MAX_TILES`/`_next_pow2`/`_sub_pages`/`_bucket_for` 逐值相等 | 一致 |
| `declared_cap` ≡ 内核 `declared_pool_cap` | 池 {1,2,3,100,9425,9899,2288,65536} 逐值相等 | 一致 |
| 池漂移同桶(9425→9899,真实 vLLM 快照波动) | `declared_cap` 同值 | 跨运行命中 |
| `prefill_ladder_tops` vs 引擎内联公式 | 6 组配置(含缺省/长上下文/split 档)逐项相等 | 一致 |
| `prefill_specs` vs 引擎 `_prewarm_prefill_ladder` | 3 组 ktp×model_len 规格序列逐项相等 | 一致 |
| 变体字段形状 | plain=9 字段、split=10 字段;批=bucket+1、block_q=128 | 与 `_compile` 解包一致 |
| ladder 变体数 | ≤32(P3/D39 上限) | 通过 |
| Qwen3.6 端到端推导 | decode=[1..32]、prefill 11 变体(1..1024 tiles)、pool=16384 | 通过 |
| `plan → JSON` 可序列化 | 字段与 plan 一致、split>0(ktp=64) | 通过 |
| 全量 CPU 回归 | `pytest tests -m "not npu"` 130 passed | 无回归 |

**预烧真编译(NPU 实测,2026-10-15 16:34)**:`scripts/prewarm-tileinfer-kernels.py --prefill`
按真实引擎口径(block=128、池 15589 页→cap 16384、seqs=32、budget=8192、len=32768)实跑:

| 阶段 | 数字 | 判据 |
|---|---|---|
| 冷编译(首次,磁盘缓存未命中) | decode 6 桶(1..32)+ prefill 11 变体(1..1024 tiles)全成功，**~25 min**(单变体 ~90-115s) | 全绿 |
| **热编译(缓存命中,同一配置) | **rc=0，15 s** 完成 6 桶+11 变体 | **命中率 100%** |
| 编译是否需空闲 HBM | **否**——冷编译在 HBM 62% 被外部 serve 占用下仍全成功(host 侧 CCE 代码生成) | 已证 |

即:预烧脚本按配置推导出的规格集,与引擎编译键**逐一对齐**;服务启动前预烧完成后,服务路径
**零编译**(15s 命中 vs 冷 25min)。

**serve 端到端零编译核对**:`RESULT=PREWARM_OK_SERVE_BLOCKED_ENV` —— serve 启动仍被
**环境侧 GDN custom-op 回归**(`aclnnCausalConv1d` `activation_mode`,见 D47/`DECISIONS.md`)阻断,
与 T6 改动无关(自动监听编排 `.agent_loop/run/t6_verify_chain.sh` 已把该分支单列,不误判为 T6 失败)。
custom-op 环境恢复后,同一编排可直接补 serve 侧 `grep "compiling prefill kernel for shape" = 0` 核对。

Reproduce with:

```bash
# 自动监听:等 HBM 空闲稳定窗 → 推导校验 → 预烧真编译 → best-effort serve 零编译核对
setsid bash .agent_loop/run/t6_verify_chain.sh 7 >> .agent_loop/run/t6_verify_chain.log 2>&1 &
cat .agent_loop/run/t6_t5_verify/RESULT
.venv/bin/python -m pytest tests/test_prewarm_specs.py -q            # CPU 契约(24,含漂移守护)

# 预烧规格(CPU,不编译;可 diff/入库):
python scripts/prewarm-tileinfer-kernels.py --kv-heads 2 --group 8 --dim 256 \
    --block-size 2048 --num-engine-pages 600 --max-num-reqs 32 \
    --max-num-batched-tokens 8192 --max-model-len 32768 --prefill --plan-only --json
# 真编译(NPU 空闲时执行,产出落 TILELANG_CACHE_DIR):
TILELANG_CACHE_DIR=.work/tl_cache_950 python scripts/prewarm-tileinfer-kernels.py \
    --kv-heads 2 --group 8 --dim 256 --block-size 2048 --num-engine-pages 600 \
    --max-num-reqs 32 --max-num-batched-tokens 8192 --max-model-len 32768 --prefill
```

## T4/T5 内核机制离线取证:full M tile(已完成)与 SIMD softmax(机制可表达)(2026-10-15)

两条 P1 内核路线都属"机制是否可表达"一类,结论读自生成的 CCE,**无需设备**(仿 D45 的离线下探
方法:`benchmarks/probes/*_lower.py` + `compile_kernel_source(target="ascend")`)。

### T4 prefill full M tile = ✅ 已完成(commit `5730b66`,PM-2)
`next-tasks.md` 曾把 full M tile 列为未做,实为**过期文档**。实际方案(见 [`known-issues.md`](known-issues.md#pm-2--prefillappend-multi-tile-wrong-prefill-multi-tile-misplaced-rowspm-2)):
因果掩码需要**全局**行号,而向量区只见本 AIV 的行(`dual_copy` 把 M 半分);解法不是取 AIV 索引,
而是 **host 预算每行因果阈值** `q_pos + shift` 写入 `RowThr`(超出行数的填 -1),内核像 `s_ub` 一样
`dual_copy` 它——两半各得自己的阈值,掩码在两半都正确。`block_q // 2` 规划限制**已移除**,全 tile
端到端使用(`BR = block_q`)。CPU 契约:`tests/test_offline_lower_probes.py::test_prefill_uses_full_m_tile`。

### T5 SIMD softmax = 🟡 机制可表达 + **数值等价已证**,性能待集成
**动机**:当前 decode/prefill 的 online softmax 用 `T.SimtVF` + `T.alloc_reducer`(几行而非上百行,
是正确的首版取舍),但它把向量时间留在关键路径;上游 FlashAttention 用打包的 `T.simd.*` 微 API。

**离线取证**(`benchmarks/probes/simd_softmax_lower.py`,32×128 tile):

| 路径 | 向量内在函数 | 标量 `for` |
|---|---|---|
| 现状(SIMT,`T.SimtVF`+reducer) | **0** | **11** |
| 候选(SIMD,`T.simd.*` in `T.SimdVF()`) | **7**(`vlds_norm`/`vcmax`/`vsub`/`vexp`/`vcgmax`/`vmul`/`vsts_norm`) | **1** |

**判读**:纯 SIMD softmax(减行 max → exp → 乘行和的倒数)在本后端能完整 lower 成向量内在函数
——**机制可表达已证**。CPU 契约:`tests/test_offline_lower_probes.py` 两个用例(SIMD 降到向量内在
函数、SIMT 降到标量循环)。

**设备数值等价(2026-10-15 实测,`benchmarks/probes/simd_softmax_device.py`,两档宽度)**:

| 宽度 | 路径 | max\|diff\| vs fp32 oracle | max\|rowsum−1\| | 判定 |
|---|---|---|---|---|
| BC=64(单寄存器) | SIMT(`T.SimtVF`+reducer) | 0.00000 | 0.00000 | PASS |
| BC=64 | **SIMD**(`T.simd.*`) | **0.00000** | 0.00000 | PASS |
| **BC=128(真实 page_size,跨 2 寄存器)** | **SIMD** | **0.00000** | 0.00000 | PASS |
| BC=128 | SIMT | (信息性*) | — | — |

\* 合成 harness 只 `T.copy` 单块稠密 tile,不复现真实内核的 `dual_copy`/GEMM 输出布局,故其 SIMT
reducer 在 BC>64 会错(与 SIMD 无关);真实 SIMT 基准由设备内核套件
`tests/test_tilelang_ascend950_decode.py` 提供。**SIMD 在 BC=64 与 BC=128 都与 fp32 oracle 逐位一致**。

**数值等价已证(含跨寄存器归约)**。关键语义(读自 `simd_inst.h`):`vcmax`/`vcadd` 是**整向量**归约、
结果落 lane 0,须 `vdupv(POS_LOWEST)` 广播回所有 lane;`vcgmax` 是**按 datablock** 的 max(非 sum——
离线探针早先用错过,设备探针已改用 `vcadd`);行宽 >64 时用 `vmax`/`vadd` **跨寄存器合并**两段的
lane-0 归约结果(`vmax(vcmax(r0),vcmax(r1))` / `vadd(vcadd(e0),vcadd(e1))`)——这是 BC=128 集成的关键。
CPU 契约:`test_offline_lower_probes.py::test_simd_device_softmax_lowers`(锁 BC=64 与 BC=128 两档)。

**分页 masked 数等价(2026-10-15 实测,`benchmarks/probes/simd_softmax_paged_device.py`)**:decode 内核
的 softmax 是**分页 + masked + 运行期页循环**,比上表更复杂。该探针在设备上验证了这一组合:

| 场景(tail=最后一页有效列数) | max\|diff\| vs fp32 oracle | 判定 |
|---|---|---|
| tail=128(整页) | **0.00000** | PASS |
| tail=64(半页) | **0.00000** | PASS |
| tail=13(奇数尾) | **0.00000** | PASS |

覆盖:运行期 `for p in T.serial(pages)` 页循环、运行期 per-request `last_len`(从 UB 读)、尾页
mask(`vcmps`+`vsel`)、跨寄存器归约、two-pass(行 max → exp/sum → `o/l`,与内核 online 形式等值)。

**集成必须绕开的两个后端限制**(已定位):
1. **SIMD_VF 不能读全局内存**——`kv_last_page_len` 须先 `T.copy` 进 UB 再在 SimdVF 内用;
2. **bisheng 不支持 f32 `select`/`sitofp`**——mask 比较用 **int32 lane 索引 vs int 标量**
   (`vci(int32)`+`vcmps` int 重载),不要用 float `T.if_then_else`(会降成 f32 select)。
CPU 契约:`test_offline_lower_probes.py::test_simd_paged_softmax_lowers`。

**逐页 online SIMD 核心(= decode 内核形态,2026-10-15 实测,`simd_softmax_online_device.py`)**:
内核是**逐页 online**(running `(m,l)` + 每页 rescale),非 two-pass。该探针以真实形状(rows=16、
page128=2 寄存器、8 页)验证 online 核心(SIMT 侧不碰 o 的 rmw):

| 量 | max\|diff\| vs fp32 oracle | 判定 |
|---|---|---|
| `p/l`(=softmax) | **0.00148** | PASS(bf16 容差内) |
| 行和 `l` | 0.00003 | PASS |

**速度(真实 decode 形状 rows=16/page128/8 页,静态 kernel 单跑)**:

| 路径 | µs/iter | 命中 |
|---|---|---|
| SIMT(`T.SimtVF`+reducer) | **12.4** | oracle 0.00000 |
| **SIMD** | **4.7** | oracle 0.00000 |
| **加速** | **2.6×** | 两者均 bit-exact |

**T5 的速度收益已证**(真实形状 SIMD 比 SIMT 快 2.6×,且均与 fp32 oracle 逐位一致)。
一个**集成警告**:`o = o*alpha` 与 `o += P@V` 若在**同一 SimdVF 块内相邻**读改写同一 UB,会踩
hazard(实测 `o` diff=1.0 而 `m/l/alpha` 精确);真实内核里二者被 cube GEMM 隔开,是安全的——集成时
保持这一分离(softmax 块只写 p/alpha/l,o 的 scale 单独成块)。

**剩余**:把逐页 online 核心按 opt-in(`TILEINFER_SIMD_SOFTMAX`)接进 `paged_decode_ascend950.py` 的
decode softmax 块,跑 `scripts/run_device_ci.sh` + 在役探针确认真实端到端增益。

**全量 CPU 回归**:`pytest tests -m "not npu"` **138 passed**。


Reproduce with:

```bash
.venv/bin/python benchmarks/probes/simd_softmax_lower.py      # 离线:两路径指令混合
.venv/bin/python benchmarks/probes/simd_softmax_device.py     # 设备:数值对拍(NPU)
.venv/bin/python -m pytest tests/test_offline_lower_probes.py -q
```

## Prefill KV 重扫截断(prefill kernel KV-rescan truncation,2026-10-14)

**问题**:prefill 内核每个 query tile 都循环该请求的**全部** KV pages(`for p in T.serial(req_pages)`),
但因果掩码意味着一个 tile 只能看到它的因果前沿(`q_pos + causal_shift`)之前的 KV。长 prompt 下
后续 tile 的绝大部分页被掩码成 -1e30 却仍被读入、做 QK^T、softmax、PV——即 O(tiles×pages) 的
**冗余 KV 扫描**,是长上下文纯 prefill TTFT 落后的结构性成因之一。

**方案**:host 侧按 tile 预计算 `TilePageLens = floor(frontier / page_size) + 1`(frontier =
该 tile 末行 query 位置 + append shift;空/padding tile 为 0),内核页循环改为 `for p in T.serial(plen)`
并把 `plen` 钳到该请求真实页数。跨页 `limit` 与行阈值 `RowThr` 保持不变——被跳过的尾部页原本就被
掩码成 -1e30,故数值**逐位等价**(复用了 P8 split 内核的动态截断先例)。

**验收数字**(微基准,单请求,group=8,d128,page=128,median of 10;`benchmarks/bench_prefill_rescan.py`):

| 场景 | 全扫基线 ms | 截断 ms | 加速 | max abs diff vs oracle |
|---|---|---|---|---|
| 8K prefill | 178.0 | 156.5 | **1.14x** | 0.0078(两侧相同) |
| 16K prefill | 553.7 | 449.4 | **1.23x** | 0.0078 |
| 32K prefill | 1953.8 | 1552.7 | **1.26x** | 0.0156 |
| chunk 512 over 8K 前缀 | 14.3 | 13.1 | 1.09x | 0.0002 |
| chunk 2048 over 8K 前缀 | 56.8 | 54.8 | 1.04x | 0.0002 |

截断与全扫的 max abs diff **完全一致**,证明被跳过的页确实只贡献掩码值。收益随上下文变长单调上升
(纯 prefill 前半 tile 几乎不扫,后段 tile 扫得越来越少);append/chunk(有缓存前缀,每个 tile 都要看
前缀)收益较小但为正。

**回归**:设备测试 34/34 绿(新增 3 个多 tile 长上下文截断用例:2048/2200-append/4096);CPU 套件 60/60 绿。
引擎级端到端 TTFT 复核(vLLM serve 长上下文)待补。

Reproduce with:

```bash
source .agent_loop/run/env.sh
python benchmarks/bench_prefill_rescan.py                                   # 截断(在役)
TILEINFER_PREFILL_NOSCAN_OFF=1 python benchmarks/bench_prefill_rescan.py    # 全扫基线
```

## Decode baseline (paged, GQA 32q/8kv, head_dim 128, page 128)

| batch | kv_len | ms | GFLOP/s | GB/s | max abs err vs fp32 oracle |
|---|---|---|---|---|---|
| 64 | 512 | 0.982 | 547 | **137** | 2.0e-3 |
| 32 | 2048 | 1.847 | 581 | **145** | 9.8e-4 |
| 16 | 4096 | 1.831 | 587 | **147** | 4.9e-4 |
| 8 | 8192 | 2.144 | 501 | **125** | 4.9e-4 |
| 4 | 16384 | 2.785 | 386 | **96** | 2.4e-4 |
| 1 | 32768 | 2.652 | 203 | **51** | 2.4e-4 |
| 1 | 65536 | 5.230 | 205 | **51** | 1.2e-4 |

*Correctness holds across the whole grid* (bf16 vs the fp32 torch reference, max abs error ≤ 2e-3),
which is the first thing to check before believing any of the timings.

### How to read it

**Decode is memory bound, so `GB/s` is the metric, not `GFLOP/s`.**  One query row per request
reads K and V exactly once, so the floor is `2 · batch · kv_heads · kv_len · head_dim · 2` bytes
plus the page table; the harness computes that and divides by the median of 10 runs.

* The best point (147 GB/s at b16/kv4096) is where the grid is wide enough to use the cores:
  the kernel launches `batch × kv_heads` blocks, so b16 gives 128 blocks.
* The worst points are `batch == 1`: 8 blocks for 32 cores, one core streaming 32k–64k tokens of
  K and V alone.  That is a *scheduling* limit, not a bandwidth limit — and it is exactly what the
  planner's split-KV (`kv_tile_pages > 0`) was built for: the reference implementation and its
  tests already exercise the split-and-merge contract, the kernel has yet to consume it.
* For context on the other axis: upstream's dense **prefill** FA reaches 320–362 TFLOP/s on this
  device (compute bound, 4096-row tiles).  A decode kernel is never going to look like that, which
  is why the column above is GB/s.

## Known limits of this baseline

| Limit | Effect | Where the fix belongs |
|---|---|---|
| One block per `(request, kv_head)` | small batches and long contexts starve the cores (51 GB/s at batch 1) | split-KV along pages (planner support exists, `kv_tile_pages`), then a merge kernel; and/or `T.Persistent` over a fixed core count |
| M tile padded to 32 rows | GQA group 8 wastes 3/4 of the QK^T M dimension | pack several requests per tile when they share a page, or accept it — cube time is not the bottleneck at these shapes |
| SIMT softmax | a few `µs` of vector time per page, not overlapped with the next page's load | move to the upstream `T.simd.*` packed-softmax path once correctness is locked (that is what upstream's FA does) |
| ~75 s compile per new shape | first request after a shape change is slow | warm up the shapes a deployment uses; the plan/run split already keeps compilation out of `run` |
| Numbers measured with another vLLM instance holding ~113 GB of the 128 GB HBM | possible interference, no exclusive HBM | re-measure on an idle device before publishing |

## Split-KV: 3x on the starved shapes (measured, merge validated)

The `51 GB/s at batch 1` row above is a *scheduling* limit: the kernel launches
`batch × kv_heads` blocks, so one long request occupies 8 of 32 cores.  Splitting the KV axis —
which the planner has supported all along (`kv_tile_pages`, with the merge contract implemented and
tested in the torch reference) — fills the grid instead:

| shape | tiles / blocks | ms | GB/s |
|---|---|---|---|
| b1 / 32768 tok, no split | 1 / 8 | 2.596 | **51.7** |
| b1 / 32768 tok, 16 pages per tile | 16 / 128 | 0.876 | **153.2** |
| b1 / 32768 tok, 8 pages per tile | 32 / 256 | 0.900 | 149.2 |
| b1 / 65536 tok, 16 pages per tile | 32 / 256 | 1.741 | **154.2** |
| b64 / 512 tok (wide control) | 64 / 512 | 0.886 | 151.4 |

Through the **public API** (`BatchAttention.plan(..., kv_tile_pages=16)` with
`TILEINFER_ALLOW_SPLIT_KV=1`, one shape per process — see KI-1):

| shape | splits | ms | GB/s | vs unsplit |
|---|---|---|---|---|
| b1 / 65536 tok | 32 | 1.866 | **143.9** | 51 -> **2.8x** |
| b4 / 16384 tok | 8 | 1.853 | **144.9** | 97 -> +49% |
| b8 / 8192 tok | 4 | 1.859 | **144.4** | 125 -> +15% |

All three match the CPU oracle to <= 2e-4.  The 2.8x at the long-context single-request case is the
whole point: that is the shape a coding agent or a long-document request produces.

**3.0x** on the long-context single-request case, and the whole grid now sits at ~150 GB/s, i.e. the
split kernel *at the wide shape* is already as fast as the unsplit one — the extra partial/lse
traffic (~3% of the K/V traffic) is not visible.

Correctness of the split path: a 4096-token request split into 4 tiles, partials merged on the host
with the reference's weighting (`out = Σ exp(lse_t − lse_all) · partial_t`, weights summing to 1),
matches the dense oracle to **2e-4**.

Reproduce: `python benchmarks/probes/split_kv_decode.py` (host-side merge, used to validate the
contract) — for the end-to-end path use the API with the opt-in:

```bash
TILEINFER_ALLOW_SPLIT_KV=1 python -c "..."   # BatchAttention.plan(..., kv_tile_pages=16)
```

Status: **split kernel + device merge kernel are landed, wired through the backend, and validated**
by `tests/test_tilelang_ascend950_decode.py` (single request with 4 splits; ragged 2-request batch
with different split counts).  Split-KV is **opt-in** (`TILEINFER_ALLOW_SPLIT_KV=1`) because it has
twice wedged the device with a vector-core exception when several shapes are compiled and run in one
process — not reproducible for a single shape, nor across 20 consecutive launches of one plan; see
[`known-issues.md`](known-issues.md#ki-1--split-kv-wedges-the-device-when-several-shapes-share-one-process-open).
Measurements are therefore taken one shape per process.

## What is *not* measured yet

* prefill / append (the kernel does not exist yet — decode only);
* end-to-end serving (TTFT / TPOT / throughput inside vLLM-Ascend);
* the CANN FIA baseline on the same shapes (the harness has the hook, it needs a working FIA call
  on this stack — see `run_fia`).

## End-to-end: TileInfer against the stock FIA operator inside vLLM

Qwen2.5-1.5B-Instruct (hidden 1536, 12 heads, 2 KV heads, head_dim 128, GQA group 6) served by
vLLM 0.23.0 + vLLM-Ascend 0.23.0, `--enforce-eager`, `--max-model-len 2048`, KV pool pinned with
`--num-gpu-blocks-override 8192`, one prompt, greedy.  TileInfer's decode plan was built **inside the
EngineCore** (`TileInfer plan READY ... backend=tilelang-ascend950`) and the measured window had
**zero fallbacks to FIA**; the baseline is the same server started with `TILEINFER_DISABLE=1`, so
vLLM-Ascend serves through the stock FIA operator.

| workload | TileInfer | stock FIA | ratio |
|---|---|---|---|
| 1-token request (TTFT + one step), median of 5 | 23 ms | 22 ms | 1.05x |
| 128-token greedy completion, 4 runs | 75.6 / 82.9 / 81.7 / 89.0 tok/s | 87.3 / 88.8 / 85.9 / 91.7 tok/s | 0.94x |
| 4 concurrent requests, 64 tokens each | 223.9 tok/s | 319.6 tok/s | 0.70x |

Reading it honestly: for a single stream our decode path is within ~6 % of the production operator,
and at four concurrent requests it is ~30 % behind.  Both sides run eager (no ACLGraph) and the
context never exceeds 2048 tokens, so this is the *short-context* regime — the one where our kernel
has no structural advantage:

* the split-KV path (2.8x standalone at 32k-64k tokens) only pays off from roughly 8k tokens up and is
  still off by default because of KI-1, so nothing here exercises it;
* at batch 4 the difference is scheduling, not numerics: our plan gives each request one work tile per
  step, while FIA's operator packs the batch into its own tiling.  That is what the batch-bucketing
  work (T6) is for;
* the prefill M-tile fix and the SIMD softmax (T4/T5) are the two known ~2x items still on the table.

Method notes that mattered: the kernels must be pre-warmed **out of process** (a compile inside the
engine stalls it, see `architecture.md`), the KV pool must be pinned (`--num-gpu-blocks-override`) so
the pre-warm key is reproducible, and a bare `wait` in the harness waits for the server process too -
which is what made three earlier "engine hangs" look like engine hangs when the engine was idle.

## MoE 专项终态:preNZ align 直载线终裁关闭,内核 0.813x/0.842x 为技术栈上限(FINAL TEMPORARY,2026-10-16 iter-067)

**结论**:MoE 层专项的 30% 端到端目标未达成,以 FINAL TEMPORARY 交用户裁决
(PERFORMANCE_REPORT §13;优化台账 PERF_OPTIMIZATION 共 19 次尝试)。

**终裁证据链(E88)**:H30 发射竞态白盒破案并修复后(E83/E85,955bd21→e43a18e:
dst 写访问让调度器合成 RAW/WAR 事件边 + MTE2/MTE1 pipe_barrier,生成源逐 token 等价,
CPU 契约 23/23 绿),设备 numeric 复验(2026-10-16 02:10,空闲窗完整跑通)双 pack
全 FAIL——transposed rel_y=3.818/3.24、identity rel_y=3.851/4.597(TOL 3e-2),与
修复前(E82)同量级;同 run 生产 nd 内核 0.0019-0.0028 全绿。恒等 pack 与转置 pack
同量级错=决定性判据:**nd2nz 重排是 MTE copy 引擎(asc_copy_gm2l1_align)固有语义,
主机侧预打包在 tilelang call_extern 路径不可绕过**。

**MoE 内核线上限的构成**(全攻击面已裁决,详见 PERFORMANCE_REPORT §13.2):
- 融合内核本体 0.813x(decode_b32)/0.842x(prefill_400) vs aclnn 链,缺口=GM→L1
  nd2nz 转换带宽(~50GB/s/核,E36),内核内不可修;
- 官方参照线全部负结果:DeepGEMM-Ascend m_grouped 0.565x(E74)、weight_nz_mode=2
  启动 abort(E43)、工作树 prenz 编译失败(E66)、H24 两内核合一收益 ~2% 降级(D69);
- 端到端数学上界:内核缺口 16% × decode MoE 占比 53.8%(E79)≈ TPOT ~8.6%,
  30% 目标在 MoE 域内不可达。

**代码终态**:生产在役路径零变更;`tileinfer/kernels/moe/align.py` 的同步修复保留
(独立正确性改进);D73 平内存 pack/unpack + `moe_prenz` 接入层封存
(`TILEINFER_OPS=moe` 门控,默认不激活——numeric 门槛未过,禁止接入在役)。

**本专项已落地的引擎级收益**(基线=vLLM 原始昇腾算子路径 FIA,详见 §12/P8):
W1 长上下文 prefill 分流(8K/16K C8 吞吐 0.226x/0.130x → 0.754x/0.758x)、
W2 per-bucket ktp 分带(并发税归零,TPOT 0.995/0.987)、split-KV 长上下文 decode
(1.41x/2.04x)。遗留候选:H28 shared expert 融合(TPOT 上界 ~6%,等 TI 侧画像,
pprof_ti_chain_v2.sh 重试链在挂)。

## P-PROF 补全:TI 侧画像(双窗)+ 双侧对照——MoE/dense/GDN 分段占比与 H28 上界确认(2026-10-16,iter-071,E91)

采集条件:vLLM 内置 torch profiler,TI 侧(推荐部署配置:ktp 分带+shunt),
prefill 窗=8192×4/C8、decode 窗=128×256/C8;与 FIA 侧(iter-061/E79)同口径。
失败复盘:02:49 首跑失败真因=链进程缺 CANN 环境 source(E89/E90),10:19 环境完整
继承后一次成功(E91)——画像工具链(P-PROF)就此双侧收口。

### TI 侧设备分段占比(per-window 真值,E91;与 FIA 同表结构)
| 窗 | 设备总 | MoE | dense+胶水 | GDN | other(自研 attention) | full-attn |
|---|---|---|---|---|---|---|
| prefill 8K×4/C8 | 4463.6ms | 32.8% | 26.9% | 17.5% | 17.1% | 5.7% |
| decode 128×256/C8 | 4345.9ms | 38.5% | 37.2% | 13.3% | 11.0% | ~0% |

Top-5 归因(TI decode 窗):GMM 32.3%(MoE)/MatMulV3 23.9%(dense,含 shared expert)/
main_kernel 8.7%(TI decode attention)/fused_gdn_gating 7.0%/RecurrentGatedDeltaRule 3.8%。

### 双侧对照(decode 窗,TI vs FIA)与结论
| 段 | TI | FIA(E79) | 解读 |
|---|---|---|---|
| MoE | 38.5% | 53.8% | TI dense/GDN 段相对更重,MoE 内核 0.84x 的端到端杠杆 TI 侧 ≈ TPOT ~5-6% |
| dense | 37.2%(MM3 23.9%) | 31.6%(MM3 21.0%) | shared expert 融合(H28)TPOT 上界 ~5-8%,双侧一致 → 立项候选维持 |
| GDN | 13.3% | 11.6% | 一致(gating/recurrent 为主,融合线 H8 已按证据关闭) |
| full-attn/other | ~0%+11.0%(main_kernel) | 1.0% | TI 自研 attention 计入 other;绝对值小,非瓶颈 |

画像派生假设处置:**H28(shared expert 融合进 grouped)维持「立项候选归档」**——
TI 画像确认其 TPOT 上界 5-8%(E91.4),但 MoE 专项以 FINAL TEMPORARY 收口
(终止条件 b 已触发),H28 留作用户决策的下一个优化靶点,不在本专项内展开。

注:聚合器脚本对 TI 双窗 CSV 存在并集显示缺陷,本表数字以 per-window 真值
(E91)为准;原始数据 .agent_loop/run/prof/ti/ 双目录 CSV。

## 系统级专项 H31:shared expert multistream 官方开关(零代码),短/中场景 TPOT -1.5%~-2.3%(2026-10-16 iter-072,E94-E98)

用户裁决 D84 开系统级优化专项后首项落地。机制=vllm-ascend 0.23.0 内置
`multistream_overlap_shared_expert`:shared expert 的 gate_up+act+down 切到独立 NPU
stream(`fused_moe.py:734` npu_stream_switch),与 routed dispatch/GMM1/GMM2/combine
事件级重叠,bf16/TP1 走 `_shared_experts_part1/2` 分支;启动时官方
`_validate_shared_expert_consistency` 断言 split 与 integrated 数值等价(内建 oracle)。

**占比真相(E97,离线深挖 E91 decode 窗 kernel_details.csv 形状聚类)**:shared expert
两 MatMul(gate_up 1.49% + down 1.08%)合计 **2.57%** 设备时间——H28 立项口径
5-8% 为含 shared_gate/胶水的理想化上界;multistream 完美重叠的理论上界即 ~2.6%。

**权威口径单变量 A/B**(TI 最优配置 KTP16+split-KV 固定,± 开关;双侧 rc=0,
5 场景×32 请求 completed/failed 全绿;图模式 FULL_AND_PIECEWISE capture [1,4,16,32];
TI 在役探针 ready_plans=6/kernel_markers=73970):

| 场景 | TPOT_P50 off→on(ms) | ΔTPOT | Δ吞吐 |
|---|---|---|---|
| 128x128_c1 | 9.882→9.656 | **-2.28%** | +1.94% |
| 128x128_c8 | 15.252→14.992 | **-1.70%** | +1.43% |
| 128x512_c8 | 15.792→15.553 | **-1.52%** | +1.54% |
| 8192x128_c8 | 192.750→193.526 | +0.40%(<±5% 红线) | -0.15% |
| 16384x128_c8 | 590.991→589.354 | -0.28% | +0.46% |

**结论**:H31 proven(PERF_OPTIMIZATION 优化尝试 20)。短/中场景 TPOT/吞吐双向收益、
长场景中性、无场景越过不劣化红线 → 开关纳入推荐配置:
`--additional-config '{"multistream_overlap_shared_expert": true}'`(与 split-KV/KTP16
正交可叠加;与 fused_mc2 互斥,单卡不涉及)。长场景中性原因:prefill 段占比高、
decode 段 shared 摊薄。原始 JSON:.agent_loop/run/bench_results/h28_h28ms_*.json(gitignored)。

**攻击序修正(E97,「份额×可达增益」底册首个量化重排)**:dense 31.6% 的最大子项
实为 GDN 投影 MM3 族(in_proj_qkvz 7.84% + w/u 4.97% 等,合计 ~10.6%),其次 lm_head
采样路径 3.74%(per-step);shared 线已收。上游动向(E96):vllm-ascend 近 90 天
shared expert multistream 活跃(Kimi K3 驱动 PR #18818/#18904/#18950),量化+SP 场景
有踩坑史(#18894),我方 bf16/TP1 路径独立于其上。

## 系统级专项 H32:GDN in_proj_ba→in_proj_qkvz 权重融合(S-GDN 投影族第一刀,2026-10-16 iter-073)

E97 重排后 dense 域最大子项(GDN 投影 MM3 族)的可执行子项。机制=每 GDN 层
`in_proj_ba`(M=16,K=2048,N=64,0.26MB 权重)是纯内核启动延迟税(in-engine
3.8µs/次×30 层 ≈ 0.67% TPOT,E97/E100),把 ba 权重 cat 进 qkvz 权重尾部成单 GEMM
(N=12352),输出零拷贝 strided 切片喂原 `fix_query_key_value_ordering`(其内部
view 仅做末维拆分,对非连续张量合法,E101),下游 rearrange/cat 拷贝与现状一致。

**取证(E100,设备微基准,互异权重轮转流式口径)**:
- 峰值标定:950PR 纯读 1.486TB/s / triad 1.445TB/s(官方 1.6TB/s 峰值,E99);
- 融合收益:qkvz+ba 分离 36.0µs vs 融合 27.1µs(ba 边际成本≈0);
- 布局变体零收益:NN 预转置/NZ 权重在真实 decode 形状(M=8/16)除 out_proj/attn_qkv
  微基准 +11% 外均持平(NZ 与 E42 一致),不构成推翻 vendor 的双侧证据;
- lm_head 623µs=1.63TB/s 贴顶(字节削减需量化,不在授权范围,搁置);
- in-engine vs 微基准存在 1.4-1.7x 上下文缺口(L2 全污染/MoE 流干扰),内核本体
  非短板——机制取证列为后续候选;
- torch split-K 代理(双 GEMM+add)失败(发射开销),单内核 split-K 留 kernel 轮。

**实现**(74c106e,`TILEINFER_OPS=gdnproj` 显式 opt-in):类级包装
`QwenGatedDeltaNetAttention.forward_cuda`;懒构建 fused 权重(预热期、图捕获前,
+1.5GB 显存);守卫(TP1/无量化/interleaved/普通权重)不满足或运行期异常即回退原路径;
探针计数 `gdnproj_calls/fallbacks` 入 metrics dumper。数值契约=TOL 3e-2 级等价
(N 变化致 MatMulV3 tiling 不同,bf16 ULP 级累加序差,E100.1;非逐位等价)。

**A/B 结果**:(待 h32_wait_chain 门控链跑完填入)

**结论**:(待 A/B)

## H43 棒:SIMD softmax 权威口径单变量 A/B PASS——长场景 TPOT -50~67%/吞吐 +93~188%,短场景持平(2026-10-19 iter-192,E236/E237)

**机制**:`TILEINFER_SIMD_SOFTMAX=1` 把 decode/split/prefill 内核逐页 online softmax 从
SIMT reducer 形态换成 packed `T.simd.*` 形态(paged_decode_ascend950.py,设备数值等价
由 `tests/test_tilelang_ascend950_decode.py` off/on 各 25 用例+`benchmarks/probes/
simd_softmax_*_device.py` 保证);内核级自述 decode shape ~2.6x(E116)。

**判据**:8K/16K ITL/TPOT 改善 ≥5%、短场景 |Δ|<3%、双侧 fallbacks=0
(h43_summarize.py;A/B 唯一差异=该环境变量,双侧同图模式 FULL_DECODE_ONLY、
同 TILEINFER_OPS/分带 D97 最优配置)。

**完整 A/B(off=11:28 会话 3 场景+13:32 补跑 2 场景;on=13:23 会话 5 场景;
双侧 10/10 json completed=32 failed=0,metrics fallbacks=0 capture=40)**:

| 场景 | TPOT(ms) off→on | ΔTPOT | ITL Δ | 吞吐 off→on | Δ吞吐 |
|---|---|---|---|---|---|
| 128x128_c1 | 9.65→9.44 | +2.20% | +2.14% | 96.0→96.1 | +0.07% |
| 128x128_c8 | 14.99→15.04 | -0.36% | -0.43% | 459.1→415.9 | -9.40%* |
| 128x512_c8 | 15.55→15.12 | +2.77% | +2.84% | 495.3→509.1 | +2.78% |
| 8192x128_c8 | 191.35→96.53 | **+49.55%** | +42.84% | 34.2→66.0 | **+92.73%** |
| 16384x128_c8 | 584.71→194.30 | **+66.77%** | +56.92% | 11.6→33.3 | **+187.55%** |

VERDICT: PASS(脚本内置判据:长场景最小改善 42.84% ≥5%,短场景最大劣化 0.36% <3%)。

*c8 吞吐 -9.40% 跨会话波动澄清(iter-192 复查):同会话双臂复跑(13:45-13:50,
r2)=off TPOT 16.29/吞吐 415.3 vs on TPOT 16.36/吞吐 425.3 → **on 吞吐 +2.39%、
TPOT +0.39%(不劣化)**;off 臂自身跨会话吞吐 459.1(11:29)→415.3(13:47)波动
-9.5%,与 r1 差值同量级 → r1 的 -9.40% 为会话间机器状态波动,非 SIMD softmax 劣化;
两次对照 TPOT 均持平(+0.34%/+0.39%),判决维持 PASS。

**接入口径(D92.3 单变量保护)**:代码默认值**保持 off**——在役下游链 off 基座
依赖「无该变量=off」语义,默认翻转将破坏 h45/h46/h47/h48/h49 等棒的双侧单变量;
接入方式=「当前最优配置」含 `TILEINFER_SIMD_SOFTMAX=1`(D97 口径),后终权威口径
验收跑分显式携带,环境变量保留为回退通道。

**对 30% 总目标的贡献**:16384x128_c8 吞吐 +187.55%(11.6→33.3 tok/s)、
8192x128_c8 吞吐 +92.73%,远超单域 30% 门槛;短场景(c1/c8/512)持平。

**注意(口径更正)**:本节的 off/on 两侧同时开启了 `gdnproj/gdnsmall/greedy/gdnreorder`
四个 opt-in(见 `.agent_loop/run/h43_baseline.sh:47`),而这四项**各自的 A/B 尚未出数**
(H32 off 臂启动失败、H34/H37 等窗超时、H35 六次无窗)。因此本节的绝对数字含这些未验证项
的未知贡献,只能作为「SIMD 单变量差值」使用;在 H51-E3 leave-one-out 归因完成前,
不得把它们写成 proven。

## H52 prefill 打包/解包去 D2H(实现完成,设备验证待窗口)

**问题(源码审查发现,E235-7)**:`forward_prefill` 的 Q 打包与结果解包是逐 tile 的 Python
循环:每 tile 一次 `torch.nonzero(live).flatten().tolist()` + 3 次 `.item()`(各自一次 D2H
同步),再加 `num_tiles × kv_heads` 次切片赋值发射。L=8192 时约 256 个 live tile
→ **~1.5k 次设备同步 + ~2k 次发射**,每个 prefill call 付一次,直接进 TTFT。

**改动**(`tileinfer/kernels/attention/paged_decode_ascend950.py` `forward_prefill`):
调度三元组(`seq_ids/q_offsets/q_lens`)本就在设备上,因此打包改为一次
`index_select`(设备侧 gather)+ `masked_fill`(padding 行归零),解包改为一次
`index_copy_`(masked scatter)。越界行先 `clamp` 再 mask,保证 gather 不越界。
**零 D2H**,发射次数从 O(tiles×heads) 降到 O(1)。数学与旧实现逐元素等价:
- 行索引 = `(seq_id × kv_heads + kv_head) × (max_qo_len × group) + q_offset + r`;
- `valid = r < q_len`(padding tile 的 `q_len==0` 天然全掩,与旧 `live` 跳过等价)。

**已验证**:设备数值 `tests/test_vllm_ascend_prefill_wiring.py` **3/3 通过**
(prefill-no-cache / chunked-append+fresh / mixed-decode-row 三形态),说明打包/解包
索引语义与旧实现一致。

**待验证(H51 窗口)**:中上下文(1K-4K)TTFT 改善幅度。注意 8K/16K prefill 已被
`TILEINFER_PREFILL_MAX_CTX=4096` shunt 回 FIA,本项收益面是 **shunt 阈值以下**的 prefill
步骤,不在长场景口径内。

## H53 H42-③ 页流水微基准的编译失败根因(环境,已修)

**现象**:`h44_probe`(H42-② 页循环成本矩阵)在编译期失败,报
`asc_pack_to_high / asc_pack_to_low / asc_storealign_pack_quarter / asc_set_gm2l1_nz_para /
asc_load_l2_cache_mode` 未声明。

**根因**:不是内核逻辑,而是**链脚本没 `source .agent_loop/run/env.sh`**——tilelang 因此
退回系统 `cann-9.1.1`,其 `simd_inst.h` / codegen 不含 9.3 才有的这些 intrinsic。
与 D26 记录的历史坑完全同型。

**修复**:`h44_wait_chain.sh`、`h46_wait_chain.sh`、`h33_wait_chain.sh`、
`h33v2_wait_chain.sh` 四处补 `source .agent_loop/run/env.sh`。

**验证**:修复后在 9.3 工具链下
`build_decode_kernel(..., pipeline=True)` **编译通过**(此前必失败),
`TILEINFER_SIMD_SOFTMAX=1` 的 plain 内核同样编译通过。这意味着 H42-②/③ 的微基准
此前从未真正跑过,其"未出数"是环境问题而非性能结论。

## H58 待实现项的作用域盘点:几乎全是 decode,与当前最大缺口(prefill)错配

**背景**:H57 的 P0 重锚显示 decode 已领先(ITL +9.2%/+13.6%),唯一负项是 prefill
(TTFT 慢 1.53x/1.71x)。逐项核实剩余待实现项的作用域后,发现一个结构性错配。

| 待实现项 | 作用域 | 代码依据 | 对当前缺口的贡献 |
|---|---|---|---|
| E1 capture size=8 | **decode** | cudagraph batch 桶只覆盖 decode 图;c8 不再 padding 到 16 | 间接(降低 c8 的 batch padding 税) |
| E2 ktp 分带重扫 | **decode** | `kv_tile_pages` 仅用于 decode 的 split-KV(`paged_decode_ascend950.py:1297`);prefill 走另一条 path | 无 |
| H33 L2 缺口 | **decode** | HYPOTHESES 标题即「decode GEMM in-engine vs 微基准 1.4-1.7x」,小 M = decode | 无 |
| E3 opt-in 归因(gdnproj/gdnsmall/greedy/gdnreorder) | **decode** | GDN decode 路径 + 采样阶段 | 无 |
| H52 prefill D2H 向量化 | **prefill** | `forward_prefill` 的打包/解包 | **有(唯一直接项)** |
| H42-②/③ 页流水 | **prefill**(shunt 阈值以下) | per-page 成本矩阵用于 prefill 内核 | 有 |

**结论**:
1. 队列里 6 项中只有 2 项(prefill D2H、H42 页流水)作用于 prefill,且 H42 只覆盖
   shunt 阈值(4096)以下的中上下文。
2. 真正影响 8K/16K TTFT 的,**目前队列里没有任何已立项的抓手**。
3. 因此下一步必须**新立 prefill 专项假设**,而不是继续消化 decode 侧队列。

**建议的 prefill 攻坚路径**(按证据成本排序):
1. **启用已有的 D57-① shunt(H59,本轮已实现)**:代码早就有"长上下文 prefill 交还
   FIA"的机制,且注释引用了 E41/E46/E47/E63 三条独立证据说明 TI prefill 内核在长上下文
   必然劣化。默认值从 0 改为 4096(`TILEINFER_PREFILL_MAX_CTX=0` 可显式关闭)。
   预期:8K/16K 的 TTFT 直接回到 FIA 水平(16384 从 3146ms → ~1836ms,约 −42%),
   TPOT 从 117.5 → ~78ms 转正;decode 优势(ITL 已领先)完整保留。
2. **profile 定位**(零代码,一次窗口):`p_prof_side.sh` 双窗采集对比 TI vs FIA 的
   `op_statistic.csv`,确认 shunt 后 prefill 段已无 TI 算子。
3. **H52 D2H 实测**:同批 profile 读出打包/解包主机段,验证零 D2H 是否兑现。
4. **若仍想赢下 prefill**:需针对 `aiv_vec_ratio=0.96 / mac=0.02`(E46)重做内核,
   属新立项范畴(split 路线已由 E63 证否)。

**方法论提醒**:TPOT 在 `output=128` 的场景被 TTFT 主导
(`TPOT ≈ (TTFT + ITL×127)/128`),因此「ITL 领先」不能抵消「TTFT 落后」。
后续所有长上下文场景的判据应以 **TTFT 和 ITL 分列**呈现,不能只看 TPOT。

## H57 设备端到端首次跑通:vendor 算子包 9.1→9.2 修复 + P0 重锚结果(2026-10-19 18:00-19:23)

### H57.1 阻塞 5 小时的真因:CANN 自定义算子包的编译版本不匹配

现象:所有 serve 在 `Capturing CUDA graphs (decode, FULL)` 阶段崩溃:
```
RuntimeError: call aclnnCausalConv1d failed
Invalid_Argument(EZ0026): Parameter activation_mode of CausalConv1d has incorrect value.
Reason: only supports "silu" or "none".
```

排查轨迹(每条都验证过):
1. 不是 TileInfer 改动——`TILEINFER_VLLM=0`(完全不加载插件)**同样崩溃**;
   临时 `git stash` 回滚全部 4 个源码改动后**仍然崩溃**;
2. 不是 kill 缓存——`torch_compile_cache` 备份清空后仍复现;
3. 不是 CANN 版本——`.work/cann930` 与 `/home/dlb/Ascend/install/cann-9.3.0`
   的 `libopapi.so` **md5 完全相同**;
4. 不是 LD_PRELOAD——`unset LD_PRELOAD` 后同样崩溃;
5. 不是 cudagraph 模式——`FULL_DECODE_ONLY` 与默认 `FULL_AND_PIECEWISE` 都崩。

**真因**:`vllm_ascend/_cann_ops_custom/vendors/custom_transformer/version.info` 写的是
`custom_opp_compiler_version=9.1.0` —— 这个 vendor 自定义算子包是用 **CANN 9.1** 编译的,
其 `aclnnCausalConv1d` 接受 `activation_mode: int`(1=silu);但在 CANN 9.3 运行时下,
内置 `CausalConv1d` tiling 要求**字符串** `"silu"`/`"none"`,签名不兼容 → Invalid_Argument。

成功对照:dlb 的 `/home/dlb/vllm-ascend`(源码可编辑安装)使用
`custom_opp_compiler_version=9.2.0` 的 vendor 包(404952B vs 我们的 400808B),
其 serve 正常启动。

**修复**:把 dlb 的 9.2 vendor 包覆盖到本栈(原 9.1 版备份为 `custom_transformer.bak_910`)。
覆盖后:
- `Application startup complete` ✅
- `CausalConv1d` 报错 0 条 ✅
- TileInfer 探针:`fallbacks=0, capture_calls=40, ready_plans=6,
  ktp_policy={1:16,2:16,4:32,8:128}` ✅

### H57.2 H51-E4 P0 重锚(TI 当前最优 vs FIA,2026-10-19 18:11-19:09,同会话背靠背)

配置:TI = `TILEINFER_OPS=decode,prefill,greedy` + `SIMD_SOFTMAX=1` + ktp 分带
`1:16,2:16,4:32,8:128` + multistream;FIA = `TILEINFER_VLLM=0`。
两侧 `cudagraph_mode=FULL_AND_PIECEWISE`,capture `[1,4,16,32]`,32 请求全绿。

| 场景 | TI TPOT | FIA TPOT | ΔTPOT | TI 吞吐 | FIA 吞吐 | Δ吞吐 | TI TTFT | FIA TTFT |
|---|---|---|---|---|---|---|---|---|
| 128x128_c1 | 9.44 | 8.53 | **−10.59%** | 98.8 | 107.4 | −8.05% | 105.2 | 103.9 |
| 128x128_c8 | 15.05 | 18.31 | **+17.81%** | 480.1 | 184.8 | **+159.77%** | 210.7 | 200.8 |
| 128x512_c8 | 15.13 | 18.45 | **+18.00%** | 514.6 | 434.5 | +18.46% | 223.2 | 199.8 |
| 8192x128_c8 | 52.11 | 40.44 | **−28.87%** | 124.1 | 162.1 | −23.46% | 1592.4 | 1039.6 |
| 16384x128_c8 | 117.50 | 69.00 | **−70.29%** | 56.4 | 72.0 | −21.70% | 3146.2 | 1836.4 |

(ΔTPOT = (FIA−TI)/FIA,正值 = TI 更快)

**关键解读(必须成对看 ITL 与 TTFT)**:

| 场景 | TI ITL | FIA ITL | ITL 结论 | TI TTFT | FIA TTFT | TTFT 结论 |
|---|---|---|---|---|---|---|
| 8192 | **16.95** | 18.66 | **TI 快 9.2%** | 1592 | 1040 | TI 慢 53% |
| 16384 | **19.09** | 22.10 | **TI 快 13.6%** | 3146 | 1836 | TI 慢 71% |

→ **TileInfer 的 decode 内核确实更快(ITL 领先 9–14%),但 prefill 慢 1.5–1.7 倍,
把 TPOT 整体拉负。** 这是本轮最重要的定量结论:**优化重心必须转向 prefill**。

### H57.3 【更正】prefill shunt 从未被启用——原"排除假设"结论无效

**原记录有误,已作废**。原表把「shunt=4096」与「shunt=0」当作两个配置对比,实际上:

| 项 | 实际值 |
|---|---|
| `_PREFILL_MAX_CTX` 默认值 | **0(关闭)** |
| E4 TI 侧启动脚本(`.work/tiserve/run.sh`) | **未设置该变量 → 0** |
| E5 "noshunt" 脚本 | 显式 `TILEINFER_PREFILL_MAX_CTX=0` |
| E4/E5 的 `prefill_long_skips` 计数 | **两次都是 0** |

→ **E4 与 E5 的 prefill 配置完全相同**,两次测量的 TTFT 差异(1592 vs 1608 / 3146 vs 3428)
只是运行间噪声。原结论"shunt 开关对 TTFT 无影响"**不成立**,因为从未有过 shunt 开启的测量。

**真实结论**:E4 中 8K/16K 的 TTFT 全部由 **TileInfer 自研 prefill 内核**产生
(`prefill_calls=4400, prefill_long_skips=0`),这就是它比 FIA 慢 1.53x/1.71x 的原因。

**而代码里已有针对性机制**(`vllm_ascend.py:203-209` 的 D57-① 注释):
> TI prefill 内核的单位功效率随 KV 上下文线性劣化(E46 aiv_vec_ratio=0.96/mac=0.02 +
> E47 同功线性扩展),C8 饱和下 8K/16K 场景整体吞吐仅 0.213x/0.121x(E41),
> split 路线亦无增益(E63)

即:**作者早已知道并实现了 shunt,只是默认关闭、且 E4 没打开**。原 H57.4 第 2 条
"prefill 本体是瓶颈、无抓手"应更正为:**抓手就是打开 shunt(或在其基础上改进)**。


### H57.4 本轮结论与下一步

1. **首胜**:短并发场景 TI 已明显领先 —— c8 TPOT +17.81%、吞吐 +159.77%(FIA 在该场景
   有异常,需复测确认);128x512_c8 +18.00%/+18.46%。
2. **待解**:长场景 TTFT 慢 1.53x/1.71x 是当前最大缺口。**抓手已定位(D57-① 已实现但
   默认关闭)**:让 KV 上下文超过阈值的 prefill 步交还 FIA。H59 把它改为默认启用。
3. **c1 场景小幅落后(−10.59%)**:单请求无并发摊薄,decode 优势不足以覆盖固定开销。
4. 下一步优先级:**prefill 路径攻坚**(H52 的 D2H 向量化已在代码里,需实测其 TTFT 收益;
   以及 H42-②/③ 的 per-page 成本矩阵用于定位 prefill 内核瓶颈)。

## H56 三个 opt-in 在役从未生效(挂点/守卫缺陷,已修)——H32/H34/H35 数据全部作废重跑

**这是本轮最重要的发现**:H32(gdnproj)/H34(gdnreorder)/H35(gdnsmall)三项此前所有
"已实现、待 A/B"的表述都建立在一个错误前提上——它们**在役一次都没执行过**。

证据(2026-10-19,`serve_h43simd_on.log` + metrics):

| 项 | 现象 | 根因 |
|---|---|---|
| H32 gdnproj | `gdnproj_calls=0`(off/on 双侧) | 挂点包的是 `QwenGatedDeltaNetAttention.forward_cuda`;而 vllm-ascend `patch_qwen3_5.py:207` 把 `forward` 换成了 `AscendGatedDeltaNetAttention.forward`(gdn.py:67),该类**内联** in_proj 段、从不调用 `forward_cuda` |
| H34 gdnreorder | `gdnreorder_calls=0` | 同一挂点问题;另 `reorder_rows_plan` 只按 interleaved 布局推导,本模型 `gqa_interleaved_layout=False` |
| H35 gdnsmall | 启动即打印 `not installed: npu_causal_conv1d_custom=False` | `npu_causal_conv1d_custom` 是 CANN 自定义算子,由 `ASCEND_CUSTOM_OPP_PATH` 在引擎初始化期加载,晚于插件 `install()`;旧代码要求"三处全就绪否则整体放弃" |
| H36 greedy | 安装成功,但计数未验证 | 守卫含 `max_num_logprobs is not None` 等 9 项,需在窗口内以 `greedy_fast_calls` 实测确认 |

**另有一个布局缺口**:`build_fused_inproj_weight` 要求 `gqa_interleaved_layout=True`
(Qwen3-Next),而 Qwen3.6-35B-A3B 是 `False`(Qwen3.5 布局,`qwen3_5.py:142`)——
即使挂点正确,守卫也会拒绝,融合永不发生。

**修复**(本轮):
1. H32/H34 挂点改到实际被调用的 `forward`;
2. `fused_forward_cuda` 增加 Qwen3.5 分支(`[q|k|v|z]`/`[b|a]` 段切,与上游
   `forward_cuda:936-944` 逐行同构),`build_fused_inproj_weight` 去掉 interleaved 硬门;
3. `build_reordered_inproj_weight` 显式守卫:仅 interleaved 布局做行重排,否则交回 H32;
4. gdnsmall 改为**部分安装 + 惰性补装**:rearrange/l2norm 立即生效,conv1d 包装在
   首次 GDN forward 时若算子已注册则补上;
5. `h51_ab.sh` 增加 `OPTIN_INACTIVE` 自检:本侧声明开启的接管若 `calls==0`,直接告警,
   避免再产出"看起来是 A/B、实际两侧同配置"的无效数据。

**验证**:CPU 契约 `test_gdn_proj_fusion.py` / `test_gdn_reorder.py` /
`test_gdn_small_fusion.py` / `test_gdn_gating.py` 共 33 项通过;新增
`test_qwen35_layout_fused_slices_match_separate` 锁住本模型的段切等价性;
`install()` 后 `QwenGatedDeltaNetAttention.forward` 确为 TileInfer 包装,
且本模型守卫通过(fused 权重 `(12352, 2048)`)。

**处置**:H32/H34/H35 的历史 A/B 数字(含 `h32_ab_done=DONE(off=1 on=0)`、
`h34_ab_done=DONE(wait=upstream_timeout)`、`h35_ab_done=DONE(off=124 on=124)`)
**全部作废**;须在 H51 窗口以修复后的代码重跑。此前把 `gdnproj,gdnsmall,greedy,gdnreorder`
写进"当前最优配置"的做法已撤销(见 `h51_ab.sh` 基座注释)。

## H54 RoPE cos/sin「一次物化」在本栈不适用(disproven,未实施)

上游 vllm-ascend #18986 报告「RoPE 派生 cos/sin cache 一次物化」在 910B4 上 decode 步
−6.4%/吞吐 +7.0%(E232)。逐条核查本栈 0.23.0 + Qwen3.6-35B-A3B 后**否决**:

1. **该优化对象不存在**:#18986 针对的是 `vllm_ascend/ops/rotary_embedding.py` 里
   `update_cos_sin()` 每步 `index_select + view + repeat + chunk` 重建 `_cos/_sin`。
   但该函数开头即 `if _cos_sin_cache is None or _cos is None or _sin is None: return`,
   而 `_cos/_sin` 只在 `set_cos_and_sin()` 的
   `elif not is_vl_model(vllm_config) and has_rope(vllm_config)` 分支分配。
   本模型 `config.json` 含 `vision_config`(Qwen3_5MoeForConditionalGeneration),
   `is_vl_model` 为 True → **`_cos/_sin` 恒为 None** → `update_cos_sin()` 每步是
   一次空函数返回,零开销可省。
2. **RoPE 已被融合**:FIA 侧 profile(`op_statistic.csv`)里没有独立的 rope 算子,
   只有 `split_qkv_rmsnorm_mrope_kernel`(decode 窗 40892µs / 5.68%,已把
   split + RMSNorm + MRoPE 合成单核)与 `vllm::triton_split_qkv_rmsnorm_mrope` 包装;
   `aten::index_select` 累计仅 12µs。不存在"每步重物化"的落点。
3. 因此本项在本栈的**可达收益为 0**,不实施;若未来换非 VL 模型(纯文本 Qwen3-Next
   同架构)再评估。

## H55 统一设备窗口验证编排(h51,脚本就绪)

为把"已实施未验证"的项一次窗口内跑完,新增三个脚本(均在 `.agent_loop/run/`,gitignored):

| 脚本 | 作用 |
|---|---|
| `h51_ab.sh` | 参数化权威口径 A/B 单臂执行器。基座固定为「当前最优配置」(H31 multistream + SIMD + ktp 分带 + 四个在役 opt-in),两侧差异完全由 `H51_OFF_ENV` / `H51_ON_ENV`(环境变量)与 `H51_*_SERVE_EXTRA`(serve 参数)声明。取代 h32..h49 各自一份的 baseline 脚本。 |
| `h51_window_chain.sh` | 串行编排:E4 P0 重锚 → E1 capture size 8 → E3 leave-one-out → E2 SIMD 后 ktp 重扫。每臂前后校验窗口(无他人 serve + HBM<8%),失败自动跳过并记 done。 |
| `h51_summarize.py` | 按实验判据出判决表:E1 需 c8 ΔTPOT≥3% 且长场景 ≥−5%;E2 需长场景 ≥3% 且短场景 |Δ|≤3%;E3 输出逐项贡献,关掉反而更快(>+0.5%)者判 REVIEW 并建议移出推荐配置。 |

启动方式(tmux,设备空闲后):

```bash
bash .agent_loop/run/h51_window_chain.sh
# 或指定子集:EXPERIMENTS="E4 E1" bash .agent_loop/run/h51_window_chain.sh
```




## H59 长上下文 prefill 默认交还 FIA(D57-① 默认启用 + 决策前移,2026-10-19)

### H59.1 背景:代码早有机制,但默认关着且 A/B 从未打开
E238-2 的 P0 重锚显示:8K/16K 的 TTFT 为 1592/3146ms,而 FIA 为 1040/1836ms,
慢 1.53x/1.71x,把 TPOT 拖成 −28.87%/−70.29%。同一批数据里 **ITL 却领先 9.2%/13.6%**
(16.95 vs 18.66 / 19.09 vs 22.10)。

而 `vllm_ascend.py` 里早有 D57-① 机制,注释引用了三条独立证据:
> TI prefill 内核单位功效率随 KV 上下文线性劣化(E46 aiv_vec_ratio=0.96/mac=0.02 +
> E47 同功线性扩展),C8 饱和下 8K/16K 整体吞吐仅 0.213x/0.121x(E41),
> split 路线亦无增益(E63)

**即作者早已定位到问题并实现了分流,只是 `_PREFILL_MAX_CTX` 默认 0(关闭),
历次 A/B 基座也没有打开它** —— 所以这个负项一直挂在账上。

### H59.2 改动
| # | 内容 |
|---|---|
| ① | `_PREFILL_MAX_CTX` 默认 **0 → 4096**:≤4K 仍走 TI 内核,>4K 交还 FIA。显式 `=0` 恢复旧行为。 |
| ② | **决策前移**:原实现在 `_expand_page_table` + `ragged_from_page_table_device` **之后**才判 shunt,被分流的步白付一次页表展开(含一次 D2H)。改为用 `seq_lens` 在展开前估上界 `ceil(seq_len/128)`:估 ≤ 阈值 → 真实值只会更小,安全继续;估 > 阈值 → 保守分流。保留展开后的兜底复查(防御性)。 |

为什么用上界是安全的:页表推导的 real page count 可能因稀疏/尾部留空**小于** `ceil(seq_len/128)`,
所以"估算 ≤ 阈值"蕴含"真实值 ≤ 阈值"(不会漏分流到长上下文);
"估算 > 阈值"最多是**多分流**一个边界情况,代价只是回退 FIA(不会更慢)。

### H59.3 契约与预期
- CPU 契约:`test_vllm_ascend_prefill_ctx_shunt.py` 8 项通过
  (含新增 `test_h59_default_shunts_long_context_only`);prefill 全族 **32 项通过**。
- 预期(待 H51 窗口实测):16384 TTFT 3146 → ~1836ms(−42%),TPOT 117.5 → ~78ms 转正;
  8K 同理;≤4K 短场景行为不变(仍在 TI 内核),decode 的 ITL 优势完整保留。
- 判据:8K/16K ΔTPOT ≥ +15%,短场景 ΔTPOT 在 ±3% 内,`prefill_long_skips > 0`。

## H60 prefill 解包尾部省一次全量搬运(2026-10-19)

### H60.1 问题
`forward_prefill` 的收尾原本是:

```python
result = out_flat.view(...).permute(...).reshape(...).contiguous()  # 全量拷贝 #1
if out is not None:
    out.copy_(result)                                                # 全量拷贝 #2
    return out
return result
```

调用方(引擎路径)几乎总是传入 `out`,于是每次 prefill 都要做两次整块搬运。
以 16384 输入、`kv_heads×group×dim` 计,这是与 token 数成正比的纯带宽开销,直接进 TTFT。

### H60.2 改动
保留 `permute` 的 strided 视图,只在**真正需要物化**时才拷贝:

```python
result = out_flat.view(...).permute(...)          # 视图,零拷贝
if out is not None:
    out.copy_(result.reshape(...))                # 一次,直落调用方缓冲
    return out
return result.reshape(...).contiguous()           # 无 out 时才物化
```

正确性:`permute` 只改 strides;`(kv_heads, q_pos, group) → (qo_heads, q_pos)` 的
`reshape` 本就要求物化,`copy_` 按其逻辑形状逐元素读取,与旧路径结果相同。

### H60.3 契约
`test_h60_unpack_tail_contract`(纯 CPU 静态检查)锁定:
- 有 `out` 时 `copy_` 直接吃 strided view,**不得**先 `contiguous()` 到临时变量;
- 无 `out` 时仍保留 `contiguous()` 兜底(调用方要求连续张量)。
prefill 全族 **41 项 + 9 项**通过(含设备数值等价 3 项)。

### H60.4 与 H52 的关系
H52 消除了**同步**开销(O(tiles) 次 D2H),H60 消除了**带宽**开销(1 次全量搬运)。
两者叠加后,`forward_prefill` 的 host 侧只剩:
- 1 次 D2H(与 `qo_host` 合并的 `seq_lens`,见 H59-②);
- 2 次向量化 gather/scatter(`index_select` + `index_copy_`);
- 1 次结果搬运(有 `out` 时)。

## H61 关键战略结论:TI 自研 prefill 内核在所有实测场景都慢于 FIA(0/5)

### H61.1 数据(E238-2 的 TTFT 逐场景)
| 场景 | TI TTFT | FIA TTFT | ΔTTFT |
|---|---|---|---|
| 128x128_c1 | 105.2 | 103.9 | **−1.18%** |
| 128x128_c8 | 210.7 | 200.8 | **−4.94%** |
| 128x512_c8 | 223.2 | 199.8 | **−11.70%** |
| 8192x128_c8 | 1592.4 | 1039.6 | **−53.17%** |
| 16384x128_c8 | 3146.2 | 1836.4 | **−71.32%** |

**TI 在 5 个场景中 0 个胜出。** 差距从 −1.2% 单调放大到 −71.3%,与 E46 的
"单位功效率随 KV 上下文线性劣化"完全一致。

### H61.2 推论
1. **TI 自研 prefill 内核没有可辩护的性能优势区间**——至少在 C8 饱和、
   `block_q=128`、当前 `ktp=0` 配置下如此。
2. 因此 shunt 阈值 4096 是**保守值**:它只切除大亏(8K/16K),保留小亏(短场景)。
3. 若短场景也稳定落后,正确做法是把阈值下调到极小值 → **prefill 全部交还 FIA**,
   让 TileInfer 只在 decode 侧生效(那里 ITL 已领先 9~18%)。
   这**不需要改代码**,只调 `TILEINFER_PREFILL_MAX_CTX`。
4. split 路线已被 E63 证否(0.226x/0.130x),不要重走。

### H61.3 待窗口决策
| 配置 | 预期 |
|---|---|
| `MAX_CTX=0`(全 TI 内核) | 8K/16K 大亏(已实测) |
| `MAX_CTX=4096`(H59 默认) | 8K/16K 回到 FIA;短场景仍有 −1~−12% |
| `MAX_CTX=1`(全部交还 FIA) | prefill 完全等于 FIA;decode 保留 TI 优势 |

判据:在 H51 窗口跑 `MAX_CTX ∈ {0, 4096, 1}` 三档 × 5 场景,选 **总吞吐最高**的一档。
若 `MAX_CTX=1` 与 `=4096` 在短场景无显著差异,则采用 1(更简单、无维护面)。

### H61.4 终审判决(E244,2026-10-19 21:01-21:38 同会话三档 A/B)——默认改为 1

三档全部实测(在役探针 + 图模式端到端,json 有效性自检全过,c1 档
capture=40/fallbacks=0/kernel_markers=74030/prefill_long_skips=432):

| 场景 | c4096 TPOT | c1 TPOT | ΔTPOT | c4096 TTFT | c1 TTFT | c4096 吞吐 | c1 吞吐 |
|---|---|---|---|---|---|---|---|
| 128x128_c1 | 9.44 | 9.44 | +0.0% | 110.2 | **92.4** | 98.3 | **99.6** |
| 128x128_c8 | 15.05 | 15.05 | −0.0% | 235.8 | **186.8** | 476.6 | **486.5** |
| 128x512_c8 | 15.13 | 15.14 | −0.1% | 230.2 | **191.3** | 514.4 | **516.6** |
| 8192x128_c8 | 39.24 | 39.05 | +0.5% | 1053.4 | 1041.7 | 166.7 | **167.5** |
| 16384x128_c8 | 69.03 | 67.73 | +1.9% | **1711.3** | 1817.2 | 96.4 | **96.8** |

- **TPOT/吞吐:两档恒等**(±1.9% 内;长场景跨 serve 方差 ~6% 大于档间差);
- **短场景 TTFT:c1 档 −16~−21%**(TI ≤4K prefill 对 FIA 的真实劣势,E238-2 同向);
  8K/16K TTFT 两档都走 FIA(>4096),差异为调度噪声(16K +6% 单样本);
- 按 H61.3 预登记判据(总吞吐最高档;1 与 4096 无显著差异则取 1)
  → **`TILEINFER_PREFILL_MAX_CTX` 默认 4096 → 1**(vllm_ascend.py,H61 注释)。
  恢复 TI prefill:显式 `=4096`(H59 档)或 `=0`(全 TI)。
- 战略含义:TI prefill 线(H52/H60/H58)收益面收窄为「显式开启时」;H58 剩余靶心
  =TI ≤4K prefill TTFT −16~−21% 劣势(仅在显式恢复时兑现)。
- 契约:`tests/test_vllm_ascend_prefill_ctx_shunt.py` 9/9 绿(默认 1 契约+
  全量分流语义);wiring 全家另跑。
- 口径注:E241 中 c4096 的 8192/16384 TPOT(36.82/62.88)与本轮 39.24/69.03
  为跨 serve 方差(同码同配置),档内配对比较不受影响。

## H64/E1 cudagraph capture 档 [1,4,16,32]→[1,4,8,32]:c8 域 TPOT −22%/吞吐 +25-28%,零劣化(2026-10-19 iter-200,E251)

**变量**:唯一差异=`--cudagraph-capture-sizes "1 4 8 32"`(serve 参数,零代码);
其余全同(TI 基座:H31 multistream + SIMD softmax + ktp 分带 + 在役 opt-in,
`TILEINFER_VLLM=1`)。权威口径 bench 5 场景 ×32 prompts 双侧全绿。

| 场景 | TPOT off(16 档) | TPOT on(8 档) | ΔTPOT | 吞吐 off | 吞吐 on | Δ吞吐 |
|---|---|---|---|---|---|---|
| 128x128_c1 | 9.34 | 9.35 | +0.02%(中性) | 98.4 | — | — |
| 128x128_c8 | 15.05 | 11.69 | **−22.34%** | 477.8 | 599.2 | **+25.40%** |
| 128x512_c8 | 15.13 | 11.70 | **−22.65%** | 514.3 | 660.6 | **+28.45%** |
| 8192x128_c8 | 36.99 | 33.69 | −8.92% | 165.9 | 177.6 | +7.10% |
| 16384x128_c8 | 62.35 | 59.65 | −4.33% | 96.8 | 99.9 | +3.17% |

- **机理**:vllm-ascend 在 A5 上将 7 档 capture 剪成 4 档,c8 请求落 bucket=16 图,
  每步 8 实+8 pad 重放;补 8 档后 padding 减半,decode 图重放工作量与 host 判桶
  成本同步下降。长场景增益递减(KV 长度主导设备时间);c1 两档同在 bucket=1,不变。
- **判决**:proven,全场景无劣化(最差 c1 +0.02%)。capture [1,4,8,32] 并入当前
  最优基座(D135),后续权威口径 A/B 双侧统一携带。
- **战略含义**:c8 域对 FIA 的领先幅度(TI16 档已 +17.8~18.5%,E238-2)经本项叠加
  后预计达 +36%(TPOT 口径)级——待新基座 FIA 重锚确认;**30% 总目标的 c8 分量
  基本关闭,主缺口转移至 c1 场景**(当前对 FIA −9.4%,E243-4)。
- 数据:`bench_results/h51_E1_{off,on}_*.json`(2026-10-19 22:50-23:03,gitignored)。

## H50 dense-A 线判决:GEMV 亲和路径无一点 ≥1.1x,A 线关闭(2026-10-19 iter-200,E255)

h50_dense_bench.py(E172 修正形状,M∈{1,8}×9 形状,HBM-cold 流式口径,设备实测):
- 全 18 测量点 speedup 0.877–1.047,门槛(任一 ≥1.1x)未命中 → **dense-A 线关闭**;
- MatMulV3(F.linear)与 matmul 视图路径同水平,vendor 小 M GEMM 已最优;
- dense 域后续攻击面=域内融合(H28 shared expert/相邻算子合并)与发射数削减。
数据:`.agent_loop/run/h50_relay.log`(gitignored 过程产物)。

## capture8 新基座权威重锚:c8 域正式达标(TPOT −34.0%/吞吐 +45.9%),c1 为唯一缺口(2026-10-20 iter-201,E258)

**口径**:h65 双侧全 5 场景 ×32 prompts。FIA=baseline 全默认;TI=当前最优基座
(TILEINFER_VLLM=1+decode/prefill/greedy+SIMD softmax+ktp 分带+H61 默认 MAX_CTX=1
+capture [1,4,8,32],D135)。在役探针 fallbacks=0,双侧 rc=0。

| 场景 | FIA TPOT | TI TPOT | ΔTPOT | FIA 吞吐 | TI 吞吐 | Δ吞吐 |
|---|---|---|---|---|---|---|
| 128x128_c1 | 8.77 | 9.66 | **+10.2%**(慢) | 105.6 | 97.1 | −8.1% |
| 128x128_c8 | 18.01 | 11.88 | **−34.0%** | 411.8 | 600.8 | **+45.9%** |
| 128x512_c8 | 18.39 | 12.06 | −34.4% | 426.9 | 638.2 | +49.5% |
| 8192x128_c8 | 38.60 | 33.99 | −12.0% | 160.3 | 176.6 | +10.2% |
| 16384x128_c8 | 63.65 | 59.92 | −5.9% | 95.0 | 98.9 | +4.0% |

- **判决**:按达成标准「c1/c8 吞吐 ≥30% 或 TPOT −≥30%,其余 ±5% 内」——
  **c8 双指标达标**(TPOT −34.0%/吞吐 +45.9%);其余场景全改善零劣化;
  **c1 未达标(+10.2%)=唯一缺口**。c8 达成=capture8(H64)+ TI decode 内核优势叠加兑现。
- c1 缺口 0.89ms/步;py-spy 定位见下节。

## H63-② b1/c1 host 段判决:主采样点=per-replay full sync,但消除三路径全零增益——非可回收成分(2026-10-20 iter-201,E257/E259/E261)

**取证链**:py-spy 在役采样(c8:1174 样本,c1:1389 样本,250Hz)——
`torch.npu.synchronize@acl_graph.py:256` 单点占 33.0-35.8%(调用链 100% ←
`_model_forward` ← `execute_model`),为 vllm-ascend FULL 图模式每次 replay 前的
host 排空屏障(防 update_attn_params 与 replay 的 param 槽位竞争)。

**消除实验**(TI capture8 基座,权威 5 场景):
| 手段 | c1 TPOT | vs 基线 | 结论 |
|---|---|---|---|
| h65 基线(host sync) | 9.66 | — | — |
| h66 ENPU_ENABLE(sync 换位到 update 前) | 9.68 | +0.2% | 零增益 |
| h68 FG event 化(#11915 移植,update 等 replay-done event) | 9.60 | −0.6% | 零增益 |

- **判决**:sync 等待时长=设备必要工作时间(汇合点),不构成步长可回收成分;
  「消除 sync ⇒ 收益」链路 disproven。h68 8192 场景 json 收割串档(数据无效,
  其余 4 场景有效;8192 补测待下轮)。
- **vendor patch 备案**(D137):PR #11915 移植保留于 .venv(默认关=零行为变化,
  备份+还原命令在 `.agent_loop/run/pr11915/backup/` 与 DECISIONS D137)。
- **c1 下一步**:per-step perf_counter 打点分解(H65 候选),精确记账 ~2.9ms
  host 暴露段后重排攻击序。

## H65 c1 步长 host 暴露段打点分解:prep 1.126ms=最大 host 单段(<1.5ms 门槛),c1 让位 dense/H28(2026-10-20 iter-202,E264)

**手段**(step_diag 新诊断模块,零 vendor 改动,`TILEINFER_STEP_DIAG=1` 门控默认关):
monkeypatch 打点 `EngineCoreProc._process_engine_step`/`Scheduler.schedule`/
`NPUModelRunner.execute_model`/`sample_tokens`/`_update_full_graph_params_if_needed`/
`ACLGraphWrapper.__call__` 与 "prepare input"/"forward"/"post process" 三个
record_function 块,JSONL 落盘;diag 开销验证 +0.2%(9.677 vs 9.66,噪声内)。

**c1 decode 每步 9.662ms 权威记账**(run2,4161 decode 步):

| 段 | median ms | 占周期 | 性质 |
|---|---|---|---|
| graph(replay+sync=设备链含上步 sampler/D2H 排空) | 7.871 | 81.5% | 设备必要 |
| prep("prepare input" 块) | 1.126 | 11.6% | **host 暴露,设备空闲** |
| sample_worker | 0.217 | 2.2% | host(发射) |
| fwd 残差 + exec 开销 + post + sched + upd_out | ≈0.32 | 3.3% | host 散段 |
| 引擎循环 gap(period−engine_step) | 0.013 | 0.1% | — |

- **判决**:最大 host 暴露单段=prep 1.126ms(<1.5ms 立项门槛)→ H65 主体
  **disproven**(归因 proven);**c1 缺口按 D105-3 让位 dense/H28 线**。
  prep 内部(py-spy 互证)≈`_prepare_inputs` 0.7ms+`_build_attention_metadata`
  0.5ms(gdn builder 在内)。host 暴露合计 ≈1.78ms,全消除理论下限 −18.4%。
- **战略账**:c1 −30%(≤6.14ms vs FIA 8.77)必须叠加 kernel 域:H28 shared
  expert 融合(≈0.4ms)+ dense 域 + H66 图内气泡(新未知数:graph 墙钟 7.871
  vs 画像核时+采样+D2H ≈7.05,差 ~0.6-0.8ms 待 kineto 归因)。
- **8192 补测闭环**(h69,FG=1):TPOT p50 36.56 vs 基线 36.82(−0.7% 噪声内)
  → FG event 同步四场景全零增益,E261 判决加严(sync 非可回收成分)。
- 工程 3 文件:`tileinfer/integrations/step_diag.py`(新)+ install() 挂钩 +
  `tests/test_step_diag.py`(11 用例 CPU 绿);契约=默认关零行为变化、双侧可用。

## 基座勘误与 capture8×multistream 互斥定谳(2026-10-20 iter-203,E266/E268)

### 勘误:h65 系采集 multistream(H31)未生效
h65_anchor.sh 及 h63_clean/h65d/h66 系 serve 未传 `--additional-config`,而
`multistream_overlap_shared_expert` 仅能经 additional_config 开启(ascend_config.py:139,
无 env fallback)⇒ E254-E265 全部 serve 实际 ms-off。双侧对称同缺,TI−FIA 相对结论
(H63/H65/H66)不受影响;绝对基线偏慢 c1 ~2.7%/c8 ~1.1%(h65ms 反测)。此后权威 A/B
双侧一律显式带 ADDCFG(如需 ms-on 口径)并 grep serve log 生效打印。

### capture8 与 multistream 互斥(h65ms 双侧 A/B,32 prompts 权威口径)
| 基座 | c1 TPOT p50 | c8 TPOT p50 |
|---|---|---|
| capture8 + ms-off(h65,权威基座) | 9.66 | 11.88 |
| capture8 + ms-on | **8.995(−6.89%)** | **16.70(+40.58% 崩)** |
| capture16 + ms-on(E98) | −2.28% | −1.70%(c8≈14.79) |
- 模式:bucket1 图 ms-on 大赚、bucket8 图 ms-on 结构性冲突、bucket16(E98)正常。
- 结论:capture8(H64,c8 +25%)与 multistream(H31,c1 −2.3%)互斥;**权威基座维持
  capture8+ms-off**。c1-only 部署可选 capture8+ms-on 变体(非默认)。

### H41(E+1 shared expert 融合)gate 关闭判决
ms-on 有效形态反测 shared 串行段上界 0.665ms/步(c1);E+1 融合消除该段但 GMM 工作量
+12.5%(c1 +0.28ms)→ 净上界 ≈0.38ms/步 < 0.5ms 立项门槛;sglang #40350 同型同 TP
反面证据与上游 #18927/#18902 revert 警示不变 ⇒ H41 维持 registered 不立项。

### H66 c1 设备链归因判决(iter-203,E268)
双侧 kineto 对差(h66 ti Σkernel 9.415ms/步=100.5% 闭合设备链 vs h62 fia 同配置参照):
- TI−FIA 差集中:**batch-1 decode attention 主核 ~0.34ms/步**(TI main_kernel 0.472 vs
  FIA vendor paged attention 0.133)+ 胶水 datamove +0.083 + norm +0.012;
- dense_gemm/moe_gmm/gdn 大核族全部零差(±0.007)——kernel 计层面 TI=FIA;
- 端到端设备链差 0.86ms 的余项 ~0.42ms=图内气泡/同步(可回收性未证)。
判决:归因 proven(≤2 域);单点可削面 0.44ms < 0.5ms 门槛 → 不立消除假设;
attention 主核 batch-1 劣势登记 **H67 候选**(c1 缺口抓手,H28/dense 后启动)。

## H70 prep 域 host 固定税削减·第一批判决(2026-10-20 iter-206,E280/E282/E283)

### 取证(E280/E282,子段 stub 二批,c1 权威 8448 步,TPOT p50 9.660=diag 开销≈0)
prep 块(vendor model_runner_v1 L2027-2269)全归账画像:

| 段 | ms/步 | 说明 |
|---|---|---|
| prep_inputs | 0.549 | bt_commit 4发0.087+bt_slotmap triton 0.084+snap_copy 3发0.049+prep_ids 0.037+未归账小 H2D 族 0.278 |
| **build_attn_meta** | **0.532** | **E277 残差悬案主体**(prep 块内 L2231,未分解) |
| prep_upd | 0.057 | |
| 杂项 | 0.13 | prep_det 0.027/async_state_update 0.020/tokens 列表 |
| cos_sin | 0.0005 | 本模型全局 cache 未注册早退,手法判死 |

### 第一批 A/B 判决(E283,配对三臂,同会话)
| 臂 | c1 TPOT p50 | c8 TPOT p50 | 判决 |
|---|---|---|---|
| OFF(基线复现) | 9.6683(+0.09%) | 11.897(+0.14%) | 环境无漂移 |
| fastpath 全开(commit+snap+slotmap) | 10.3306(+6.94%) | 12.6194(+6.22%) | **有害** |
| commit+slotmap(snap 关) | 9.6665(+0.07%) | 11.9009(+0.18%) | 中性,净收益≈0 |

**判决:H70 第一批 disproven**——bt_commit skip/bt_slotmap CPU fast path(PR#19033 同构手法)
在役生效但发射税节省被 array_equal 比较/numpy/H2D 成本抵消;snap mirror(torch.npu.Event 门
+持久 pinned 镜像)单独有害,禁用。代码保留 opt-in(`TILEINFER_PREP_FASTPATH`,默认关)作
负面结果遗产。E281 上游 PR#19033(−7.3%)收益经本轮剥离确认非 prep 域三手法主体。

**prep 域剩余抓手 = build_attn_meta 0.532ms/步(H70 第二批,待三批 stub 分解)**。

## H70-b/H71 取证与判决:_build_attention_metadata=gdn_build 主体(0.377ms)定谳,GDN 元数据 memo TPOT 中性,prep host 域 TPOT 杠杆排除(2026-10-20 iter-207,E285/E286)

### 取证(三批/四批 step_diag 桩,默认关)
- 三批(c1 稳态 4277 步,桩开销≈0):`build_attn_meta` 0.532ms 内部 = **gdn_build 0.377(60%)** + 组循环胶水 0.214(copy(cm_base)×4 组+bt 重取)+ fa_build 0.030 + bam_init/enc_lens 0.008(Qwen3.6 hybrid 3 GDN 组+1 full 组;3 GDN builder × build/步)。
- 四批(gdn_build 内部):pad 0.129(fill+D2D×3)+ att_d 0.088 + treat 0.062 + ctx 0.048 + split/胶水残差 0.159。

### H71 实现(GDN 元数据稳态 memo,opt-in `TILEINFER_GDN_META_MEMO=1`,默认关)
- 纯 decode 稳态(无 spec、mamba_cache_mode=none)下 GDN builder build 输出内容仅由请求集+qsl+purity 决定,逐步重算为冗余税;按全 CPU 侧键 `(req_ids, num_reqs, num_actual_tokens, attn_state, purity)` memo,任一前置破坏清槽重建(prefill 重建覆写持久缓冲,stale 命中=写错行,契约测试锁定)。
- 全量 pytest 绿;提交 b29fd01/bcaa4ed(后者修捕获期 5 位置入参调用点)。

### 配对 A/B 判决(h71ab,同会话 OFF→ON,32 prompts 权威)
| 场景 | OFF TPOT p50 | ON TPOT p50 | Δ | 吞吐 Δ |
|---|---|---|---|---|
| c1 (128/128,C=1) | 9.6321 | 9.5991 | **−0.34%** | +0.37% |
| c8 (128/128,C=8) | 11.9004 | 11.8867 | **−0.11%** | +0.12% |

- **VERDICT: H71 disproven(TPOT 杠杆)**——memo 生效但理论 0.38ms host 削减仅兑现 ~0.03ms TPOT(衰减 ~10x);与 E283 第一批(三手法中性)互证:**c1 decode 步长设备侧 bound,prep host 块与上一步设备执行 overlap,host 削减仅在超窗口时计入 TPOT**。
- prep 域全部已测手法对 c1 TPOT 均中性 ⇒ c1 host 域正式排除;剩余在役可执行线 = W1-D57-② prefill 内核重写;30% 总目标移交 FINAL TEMPORARY 评估(c8 已达标 −34.0%,c1 缺口=设备域 vendor 同水平+host 非临界)。
