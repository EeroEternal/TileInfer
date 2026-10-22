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

### H64 落地缺口(2026-10-20 iter-208,E287):参数已判 proven 但执行器未携带

D135 要求"后续所有权威口径 A/B 的双侧 serve 一律带 `--cudagraph-capture-sizes
"1 4 8 32"`",但核查发现 **h51 链的通用执行器 `h51_ab.sh` 实际没有携带**——
只有 `h65_anchor.sh` / `h65ms_anchor.sh` / `h65d_diag_c1.sh` 等少数脚本带。

后果:凡用 `h51_ab.sh` 跑的臂(E1/E2/E3)实际用默认档 `[1,4,16,32]`,c8 请求落
bucket=16 图、每步 8 实 + 8 pad。这解释了 E238-2(c8 TPOT 15.05) 与 E251(11.69)
之间约 22% 的差距——**两者不是同一 capture 档**,不可直接比较。

已修正:执行器补 `CAPTURE_EXTRA`,**仅 TI 臂**携带(与 `h65_anchor.sh` 一致;
FIA 臂保持 vLLM 原生默认档,代表原生基线不应被增强);同时把该执行器的
`TILEINFER_PREFILL_MAX_CTX` 默认由 4096 改为 **1**(对齐 E244/H61 与权威基座),
启动日志增 `maxctx=` / `capture=` 字段便于事后核验臂配置。

验证脚本 `h64_verify.sh`(三臂 fia/ti16/ti8)已就绪,但本轮设备被外部栈长期占用
(EngineCore 静默退出,见 E288),三臂未能出数——待窗口补测。

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

## H72 短序列 prefill occupancy 自适应(2026-10-20 iter-208)

### 问题
H61 三档实测(同一批数据)给出 TI prefill 内核相对 FIA 的**纯劣势**:

| 场景 | TI 内核 TTFT | 交还 FIA | 劣势 |
|---|---|---|---|
| 128x128_c1 | 103.4 | 92.4 | **+11.85%** |
| 128x128_c8 | 227.9 | 186.8 | **+22.01%** |
| 128x512_c8 | 218.2 | 191.3 | **+14.04%** |

短场景稳定慢 12-22%,与长场景 53-71% 同向,但幅度小得多——说明**不是同一机理**。

### 归因:并行度不足
内核 grid = `num_tiles × kv_heads`,`num_tiles = Σ ceil(qo_len · group / block_q)`。
本模型 `num_attention_heads=16`、`num_key_value_heads=2` → `group=8`、`kv_heads=2`:

| 场景 | num_tiles | **grid blocks** |
|---|---|---|
| q=128, b=1 | 8 | **16** |
| q=128, b=8 | 64 | 128 |
| q=512, b=8 | 256 | 512 |
| q=8192, b=8 | 4096 | 8192 |

c1 只有 **16 个 block**,远低于 950 可用核数。配合 E46 画像
(`aiv_vec_ratio=0.96`、`aic_mac_ratio=0.02`、MTE≈0——逐页串行向量链,
非访存非 cube 瓶颈),"加 block"是该形状下唯一可动方向。

### 实现
- 新增 `TILEINFER_PREFILL_BLOCK_Q`(0=自适应 / 正整数=固定值,便于单变量 A/B)
- 新增 `TILEINFER_PREFILL_SHORT_CTX`(默认 1024)
- 上下文 ≤ 1024 的 prefill 步用 `block_q=64`(tile 数翻倍),否则 128
- `block_q` 统一参与 **tile 数计算 / shape spec / schedule 规划 / split 规划** 四处,
  并存入 `shared` 跨层复用

一致的四处是硬约束:任一处用不同值,编译变体与内存布局就会失配。测试
(`test_vllm_ascend_prefill_block_q.py`)逐行扫描源码,禁止残留 `block_q=128` 硬编码
——它在实现时确实抓出了 split 分支遗漏的一处 `127) // 128`。

### 状态
契约 33 项通过。设备验证待窗口(预期短场景 TPOT ≥3%、长场景 |Δ|≤3%)。
**翻案条件**:若 16 blocks 其实已饱和(单 block 时间随 grid 线性变化),
说明 occupancy 非主因,应转 H42-③ 的内核效率方向。

---

## H73 shared_expert_gate 发射合并(2026-10-20 iter-208,opt-in)

### 问题:4 KB 权重花 100 µs
E269 的 dense 逐形状记账指出一个极端浪费点:

```
shared_expert_gate = ReplicatedLinear(hidden=2048, out=1)   # 40 层
E269 实测:100 µs/步(c1 TPOT 的 2.8%),权重 4 KB,effBW ≈ 0.00
纯带宽需求 = 4 KB / 1.6 TB/s = 0.0025 µs
```

**99.9% 是发射税**,单位权重成本 2.5 µs / 4 KB,是全部 271 次发射里最贵的一次。

上游同构:ROCm `#56789` 记录 `(1,K) skinny GEMM` 占 decode step **13.7%**。
本仓库 EVIDENCE:2026 曾判定「950PR 无 skinny 缺失问题,零行动」——该判定被
E269 的 950 实测数字推翻。

### 为什么这次能立住(H68-a / H69 都失败了)
| 已证否路线 | 失败根因 | H73 是否受影响 |
|---|---|---|
| **H68-a**(GDN qkvz+ba) | 融合输出以 strided slice 喂 conv1d,走 AscendC 慢路径(引擎 +16~17%) | **否**:下游是 SiluAndMul / sigmoid / 逐元素乘,**无 conv1d** |
| **H69**(router+shared_gate+gate_up) | 三族分属不同层,需跨 FusedMoE 边界重构,且 shared 族已有 separate-stream overlap | **否**:同在 `Qwen2MoeMLP` 内,只合并一对 |

### 方案
`Qwen2MoeMLP.forward` 中 `expert_gate(x)` 与 `gate_up_proj(x)` 的**输入完全相同**:

```python
# 原:两次发射
gate_up, _ = self.gate_up_proj(x)      # hidden → 2I
out = self.act_fn(gate_up)
out, _ = self.down_proj(out)
out = F.sigmoid(self.expert_gate(x)[0]) * out   # hidden → 1   ← 单独一发

# 融合:一次发射
# MerkedColumnParallelLinear.output_sizes: [I, I] → [I, I, 1]
fused, _ = self.gate_up_proj(x)        # hidden → 2I + 1
gate_up, gate_logit = fused[..., :-1], fused[..., -1:]
out = F.sigmoid(gate_logit) * self.down_proj(self.act_fn(gate_up))[0]
```

GEMM 每列独立,拼接不改变任何列的结果 → **逐位等价**(由测试锁定)。

### 实现与开关
`tileinfer/integrations/shared_gate_fuse.py`:
- 包装 `Qwen2MoeMLP.__init__`(改结构,仅当 gate 单输出且无 bias)
- 包装 `load_weights`(checkpoint 里 gate 权重仍是独立前缀,必须接住并搬进第 3 段
  ——这是本方案唯一的正确性风险点)
- 包装 `forward`(一次 GEMM + 切分,任何异常回退原实现)

| 开关 | 作用 |
|---|---|
| `TILEINFER_SHARED_GATE_FUSE=1` | 启用(默认关 = 零行为变化) |
| `TILEINFER_SHARED_GATE_FUSE_DRYRUN=1` | 只记日志不改结构(部署前自检) |

### 状态
契约 8 项通过(含逐位列等价 + 端到端语义等价两项核心数值验证)。
判据:**c1 TPOT ≥1%**(取低于理论 2.8% 的保守门槛,因图模式可能已在 host 侧
预发射吸收,同 E286 的 prep 域 ~10x 衰减机理)。c8 |Δ|≤3%。
**翻案条件**:实测无收益 → 归档为负面结果,并在 docs 记录"图模式 host 预发射
已吸收该发射"这一结论。

### H73 设备首验:实现链三处致命缺陷(2026-10-20 E293,全部已修)

首轮设备 A/B 直接暴露了三个**实现级**问题——融合此前从未真正跑起来过,
契约测试锁的都是"看起来对"的静态结构。三处按发现顺序记录:

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| ① | serve 日志 `shared gate fuse not installed: type object 'Qwen2MoeMLP' has no attribute 'load_weights'` | 本栈 `Qwen2MoeMLP` **没有** `load_weights`(权重由模型级 `stacked_params_mapping` + `named_parameters()` 映射);`install()` 无条件访问该属性抛 AttributeError,被调用方吞成一条 warning | 删掉该包装,`install()` 不再依赖它 |
| ② | ON 臂请求全 500,`npu_swiglu` 报 `Invalid_Argument_Tensor_Shape [2048, 1025]`(split 轴必须偶数) | shared expert 在本栈走 vllm-ascend 的 `AscendFusedMoE._shared_experts_part1/part2`(拆成 gate_up GEMM / 激活+down_proj+gate 两段做 overlap),**不调用** `Qwen2MoeMLP.forward`;只挂 forward 时融合出的 2I+1 列被整块喂进激活 | 增挂 `_shared_experts_part2`(两个实现文件 `fused_moe.py` / `fused_moe_0_23_0.py` 都挂) |
| ③ | ON 臂 serve 启动成功后**首个请求即静默杀死 EngineCore**(无 traceback) | 绑定点在首次 forward:此时 `AscendUnquantizedLinearMethod.process_weights_after_loading` 已把 `gate_up_proj.weight` 转成 FRACTAL_NZ;对 NZ 张量按 `[-1:, :]` 切片早已不是"最后一行",就地拷贝写坏内存 | 绑定前移到 `UnquantizedLinearMethod.process_weights_after_loading` **入口**;兜底路径显式拒绝 `ndim != 2` |

判别证据(③):同一 `h73_ab.sh` 用 `H73_DRYRUN=1`(只记日志、不改结构)跑 ON 臂,
**5/5 场景 32/32 全绿**(c1 9.694 / c8 11.983 / 512_c8 11.998 / 8K 36.688 / 16K 66.037,
与 OFF 臂逐项差 <0.4%);关掉 dryrun 的真实改造臂则在首个请求崩溃。
⇒ 崩溃来自**结构改造**(NZ 后的错位写),不是融合算术。
dryrun 数据已改名 `bench_results/h73_dryrun_on_*`,不与正式 A/B 混用。

契约同步加固(`tests/test_shared_gate_fuse.py`,12 项):
- 新增「install() 不得依赖 `Qwen2MoeMLP.load_weights`」回归锁;
- 新增「必须挂 `_shared_experts_part2`」锁;
- 新增「绑定必须在 NZ 转换前 + NZ 张量必须拒绝绑定 + 只绑一次」锁;
- 新增「`fused[..., :-1]` 必须 `.contiguous()` 再喂 `npu_swiglu`」锁(H68-a 同型陷阱);
- 新增「`expert_gate` 必须按签名 bind 取参」锁 + 位置传参行为验证。

### 第四处缺陷:只读 kwargs 会静默漏融合(2026-10-20 自查,已修)

`_make_mlp_init` 原本用 `kwargs.get("expert_gate")` 取参,但 `expert_gate` 是
`Qwen2MoeMLP.__init__` 的**第 6 个位置参数**。vLLM 目前全用关键字调用,所以能工作;
一旦上游改成位置传参,融合会**静默失效**(不报错、指标恒 0)——这正是 H56 踩过的
"挂点从未生效"同型缺陷。已改为 `inspect.signature(orig_init).bind(...)` 取参,
关键字/位置两种调用都正确,并有行为测试锁定。

同时修正了 `_make_mlp_init` 的过时文档串(它还在说"必须同时改 `load_weights`",
而那条路径早已删除)。

**状态**:实现缺陷已全部修复,契约 12 项绿;**off 臂 5/5 已出**(9.700/11.964/11.995/
36.788/65.287,与 14:21 那次逐项一致);**on 臂 17:43-17:46 已出(32/32 全 200)——
判决 PASS,见下节**。

### H73 设备 A/B 判决:PASS(2026-10-20 17:4x,E299)

on 臂(`TILEINFER_SHARED_GATE_FUSE=1`,17:43-17:46,32/32 全 200)vs off 臂
(17:17-17:20,32/32 全绿,与 14:21 会话逐项差 <0.4% ⇒ 基座无漂移):

| 场景 | TPOT off | TPOT on | dTPOT | 吞吐 off | 吞吐 on | dthr | 判定 |
|---|---|---|---|---|---|---|---|
| 128x128_c1 | 9.700 | 9.598 | **-1.05%** | 96.6 | 98.0 | +1.54% | **PASS(c1 主判据 ≥1%)** |
| 128x128_c8 | 11.964 | 11.909 | -0.46% | 593.8 | 598.2 | +0.74% | OK |
| 128x512_c8 | 11.995 | 11.938 | -0.47% | 647.1 | 651.2 | +0.64% | OK |
| 8192x128_c8 | 36.788 | 36.646 | -0.39% | 175.2 | 175.9 | +0.36% | OK |
| 16384x128_c8 | 65.287 | 65.497 | +0.32% | 98.2 | 98.2 | -0.00% | OK(噪声内) |

- c1 主判据 dTPOT -1.03%~-1.06%(与 14:21 off 基线比 -1.06%,与同会话 off 比
  -1.05%),达标;与 E269 预估(发射税 ~100µs/步 = c1 TPOT 2.8%)量级相符——
  差值说明图模式 host 预发射吸收了一部分(E286 同机理)。
- **融合生效的反证链**:旧缺陷②(part2 未挂)的症状是 ON 臂全 500(EZ0009);
  本轮 5 场景 32/32 全 200 ⇒ part2 已挂、2I+1 结构改造数学正确。
- **installed_hits=0 的定性**:shared_gate_fuse 的 `getLogger(__name__)` 无 handler,
  install() 的 INFO 在 serve 进程走 root(lastResort 只出 WARNING+)⇒ 日志静默。
  **功能无缺陷,可观测性问题**;后续判定安装改用契约探针/metrics,不 grep 日志。
- 单变量核查:on 臂 serve 17:41:52 启动,加载的是 17:23 加固版 shared_gate_fuse;
  off 臂 17:17 启动,加载 17:0x 版——两者仅差 `inspect.signature` 加固(纯取参方式,
  行为等价、零性能面);H74 内核形态改动(17:45/17:49)晚于两臂 ⇒ **A/B 单变量成立**。

**终态**:H73 proven(保守门槛);`TILEINFER_SHARED_GATE_FUSE` 建议在 c1 敏感部署
开启(默认关不变)。相对 30% 总目标贡献 ~1%(c1 TPOT)。

### H73 首轮 A/B 中观察到的口径问题(已修脚本)

`vllm bench serve` 在**全部请求失败**时仍返回 `rc=0`,原 `h73_ab.sh` 只按 rc 记日志,
导致 EngineCore 被外部栈挤死后整臂白跑(两轮如此)。已改为:
①按 json 的 `completed/failed` 判有效;②每个 case 后探活,serve 已死即中止重试;
③臂结束时 `all_ok=1` 才写 done。这个修正对后续所有 A/B 脚本都适用。


## 勘误:H49 并未"被 E283 覆盖"(2026-10-20 iter-208)

此前文档与轮次记录中出现过一个错误结论:「H49(slot_mapping triton→numpy 宿主削减)
已被 `prep_fastpath.py` 覆盖,且 E283 已判净收益≈0,故不再实现」。**该结论错误。**

### 两者数据来源不同

| | `prep_fastpath._slotmap_cpu_fast`(现有) | H49 原补丁(#19033 最小子集) |
|---|---|---|
| positions 来源 | `runner.input_batch.num_computed_tokens_cpu[...] + runner.query_pos.np` **重算** | 调用方直传 `positions_np`(`_prepare_inputs` 内已算好) |
| 依赖 | 需 `_prepare_inputs` 包装注入 threadlocal `runner` | 无 |
| pad 区 | 按 n 变化懒刷新(fill PAD) | 调用方给精确切片,无需刷新 |

现有实现比原补丁**多做了三件事**:threadlocal 读写、positions 重算、pad 懒刷新。
E283 记录的"发射税节省被 array_equal 比较 / numpy / H2D 成本抵消",恰好可以由
这三点解释——**被证否的是"该实现方式",不是"该优化方向"**。

### 处置
已在 `compute_slot_mapping` 增加 `positions_cpu` / `query_start_loc_cpu` 可选入参与
`_slotmap_from_cpu_args` 直接路径(优先于 runner 路径),公式与
`_compute_slot_mapping_kernel` 的 CP==1 分支逐项同构、逐位等价。
契约 11 项通过(新增 4 项:直接路径 vs oracle 逐位一致 / 优先于 runner / CP 门控让位 /
公式逐项核对)。提交 `1cbc224`。

**调用点改动**(让 model_runner 真的传参)仍留待设备窗口,不在未做引擎验证时
修改 vendor 调用点。

---

## H42-③ 双缓冲页流水:off 臂基线已出,on 臂待窗(2026-10-20 E295)

`h42c_ab.sh` 的 off 臂(权威基座:capture8 + SIMD softmax + ktp 分带 + `MAX_CTX=1`)
5/5 场景 32/32 有效(14:21-14:24,同会话):

| 场景 | TPOT(ms) | 吞吐(tok/s) |
|---|---|---|
| 128x128_c1 | 9.697 | 96.9 |
| 128x128_c8 | 11.971 | 593.5 |
| 128x512_c8 | 11.995 | 647.4 |
| 8192x128_c8 | 36.650 | 175.8 |
| 16384x128_c8 | 65.183 | 98.5 |

- 与 H73 off 臂(9.694/11.971/11.995/36.65/65.18)逐项一致 → 基座可复现,无漂移;
- on 臂(`TILEINFER_PAGE_PIPELINE=1`)两次进场都被外部 ww/reef 栈挤掉,未能出数;
- **判据不变**:8K/16K TPOT 或 ITL 改善 ≥5%,短场景 |Δ|≤3%,`fallbacks=0`/`capture=40`。

### H42-③ 假设被微基准推翻(E296,2026-10-20 设备实测)

`h42_page_cost_bench.py` 在**生产配置**(`TILEINFER_SIMD_SOFTMAX=1`)下跑通完整矩阵:

| 场景 | plain ktp=0 | split128 | split32 | split16 |
|---|---|---|---|---|
| b8 × 8K | **0.598 ms / 897.5 GB/s** | 0.652 / 823.4 | 0.770 / 696.9 | 1.026 / 523.1 |
| b8 × 16K | **1.166 ms / 921.2 GB/s** | 1.247 / 860.8 | 1.220 / 880.2 | 1.484 / 723.5 |
| b8 × 1 tok | **0.107 ms** | 0.132 | — | 0.585 |

三点结论:

1. **plain 路径已达 897–921 GB/s**,接近 HBM 峰值(公开 paged MQA decode 上界
   2006 GB/s 的一半,但那是不同 cache 布局/更小 head_dim 的对照)。H42-③ 立项时
   引用的"每页 ~30 µs vs 带宽下界 2.5 µs = 12x 余量"是 **SIMD softmax 之前**的老
   数据;**加上 SIMD 后余量已基本消失**。
2. **所有 ktp 切分都是负收益**(8K: 0.652/0.770/1.026 vs 0.598;16K 同型)。
   额外的并行度换来的 merge 开销与重复 K/V 读取超过了收益。
3. **split32/split16 出现数值误差** `err=0.047/0.091`(split128 为 0.0000)。
   这是**独立于性能的缺陷**:多 split 路径在 >1 split/请求 时精度不达标,须先
   查清才能考虑启用。

⇒ **H42-③ 判决:disproven**。`TILEINFER_PAGE_PIPELINE` 保持默认关,不再投入设备窗口。
新的瓶颈定位回到 **H67**(c1 短上下文 grid 仅 2 blocks),但注意:上表 b8 场景
plain 已贴带宽顶,**H67 的"并行度不足"只在 c1/b1 成立**,不能外推到 b8。

### 附带修好的两个探针/脚本缺陷

- `h42_pipeline_bench.py`:`q_pad` 在 CPU 上分配后 `.to(npu)` → tilelang 报
  `allocator anchor has the wrong device type; expected: 1, got: 12`;同时该探针只
  导入 `tileinfer.kernels.*` 子模块,未触发包根的设备解析。两处已修(直接在 NPU 上
  分配 + 显式导入 `tileinfer` 包根),但**该探针的判定已被 page_cost 数据取代**,
  不再需要它出数。
- `h42pp_wait_chain.sh`:窗口判据用 `npu-smi info | awk '/HBM-Usage/ {print $3}'`,
  而新版 npu-smi 把 HBM 拆成 `HBM Usage Rate(%)`/`HBM Capacity(MB)` 两行,`$3` 解析
  出 `|` → `[ "|" -lt 26214 ]` 恒假 → **空等 8 小时 188 轮**。已改用 `-t usages`
  的 key:value 格式并做数值校验(实测 5243MB,判据正常触发)。

---

## H74 DeepInfer 对标梳理(2026-10-20 iter-209,仅调研不实现)

DeepSeek 于 2026-10-20 开源 [DeepInfer](https://github.com/deepseek-ai/DeepInfer)
(Apache-2.0),**明确支持 Huawei Ascend 950 NPU,且内核用 TileLang 写**——与我们同栈。
已 sparse-clone 到 `/tmp/deepinfer_probe` 逐文件核对。

### 最有价值的发现:他们全员用 `T.Persistent`,我们全是 `T.serial`

`deepinfer/model_executor/layers/tilelang_ops/asc/` 共 **26 个**算子文件,统计如下
(`persistent` = `T.Persistent` 出现次数,`pipelined` = stage 相关行数):

```
sampler.py                  persistent=7  pipelined=14
load_paged_kv_cache.py      persistent=3  pipelined=8
get_kv_index.py             persistent=3  pipelined=11
prepare_v4_decode_metadata  persistent=2  pipelined=18
swa_index.py                persistent=1  pipelined=2
fused_sample.py             persistent=1  pipelined=6
...
```

**每一个**算子都是这个骨架:

```python
cores = tile_kernels.get_num_vec_cores()
with T.Kernel(cores) as pid:
    T.annotate_buffer_versions({buf: stages})        # 新版 API
    for task in T.Persistent([...], cores, pid, group_size=1, num_stages=stages):
        ...                                          # 每个核连续吃多个 task,跨 task 流水
```

而我们的 `tileinfer/kernels/attention/paged_decode_ascend950.py`:

```python
for p in T.serial(num_pages):        # 纯串行,0 个 Persistent,0 个 stage
```

**版本核对(本地实测)**:tilelang `0.1.15`,`T.Persistent` **存在**,
`num_stages` 参数**生效**(源码 docstring:"The number of pipeline stages for double
buffering. If 0, pipeline is disabled.");`T.annotate_buffer_versions` **不存在**
(更新版 API),但 `T.Persistent(..., num_stages=N)` 是等效入口。

**这条恰好解释了 H42-③ 的失败与 DeepInfer 的从容**:我们手写双缓冲
(`paged_decode_pipeline`)是在**手工摆 buffer + 手工发预取**,而编译器本来就提供了
`num_stages` 这个正式入口。手写路线被 E296 判 disproven,不等于"流水无收益",
而是"手工实现这条路走错了"。

### 其余可借鉴项(按与我们靶面的相关度排序)

| DeepInfer 文件 | 机制 | 我们的对应物 | 可借鉴度 |
|---|---|---|---|
| `get_kv_index.py` / `kv_index_cat.py` | Persistent+SIMD 做页索引展开/拼接 | `_expand_page_table` + ragged 三段重建(每步都跑) | 实测**不适用**(见 H74-②) |
| `load_paged_kv_cache.py` | 页式 KV 加载 + FP4/FP8 量化 + block 32/64 | `_sub_pages` / `pool_view` | 中(量化暂不需要) |
| `fused_sample.py` (673 行) | temperature+softmax+top-p+top-k+采样**单核融合** | H36 只做 greedy 跳过 cast | **高**(未实现) |
| `swa_index.py` (62 行) | `T.SimdVF` + `T.simd.vsel` 窗口索引 | 我们 `T.simd.*` 的用法样板 | 中(写法参考) |
| `fused_transform.py` (899 行) | norm+rope+hadamard+cast+store_kv 五合一 | H32/H34/H35 分散小算子 | **低**(见下) |

---

## H74 实现与判决:编译器页流水对 attention 主核无收益(2026-10-20)

按上一节的梳理实现了第一项(**只实现 DeepInfer 仓涉及的优化,不动 agent_loop 的其它待办**)。

### H74-① 实现:页循环 `T.serial` → `T.Pipelined(num_stages=N)`

`tileinfer/kernels/attention/paged_decode_ascend950.py`:

```python
# 改前
for p in T.serial(num_pages):
# 改后
for p in T.Pipelined(num_pages, num_stages=sw_pipeline):
```

开关 `TILEINFER_SW_PIPELINE=N`(默认 0)。三条实现约束都是踩出来的:

1. **`num_stages=0` 天然等价串行**——tilelang 的 docstring 明确写
   "If 0, pipeline is disabled",所以**一个写法覆盖开/关**。这样避开了 eager builder
   的两个坑:
   - 不能把 frame 先赋给变量再迭代(`TypeError: a TIR expression or buffer is not
     iterable (got Var)`);
   - 局部函数体在重放时不可见(`NameError: name 'page_body' is not defined`)。
2. **`sw_pipeline=0` 不进 cache key**(默认路径不因多一个键而重复编译);
   非 0 值必须进(否则流水版与串行版共用一个编译产物)。
3. 开关从 env 一路透传:`_sw_pipeline_stages()` → backend → `build_decode_kernel`,
   每一环都有契约锁(防"挂点从未生效",H56 同型)。

契约:`tests/test_sw_pipeline_switch.py` 17 项(env 解析健壮性 / cache key 三种情况 /
源码锁 / 接线锁);相关回归 46 项全绿。

### H74-① 实测:阶段深度 0→16 全程无拐点

`benchmarks/probes/h74_stage_sweep.py`(SIMD=1 生产配置,各档重复测两次取最小):

| 场景 | stages=0 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| b1 × 128 | 0.0190 ms | 0.0190 | 0.0190 | 0.0190 | 0.0190 |
| b1 × 2048 | 0.0589 | 0.0589 | 0.0590 | 0.0589 | 0.0589 |
| b1 × 8192 | 0.1872 | 0.1903 (0.98x) | 0.1876 | 0.1875 | 0.1873 |
| b8 × 128 | 0.0563 | 0.0565 | 0.0563 | 0.0563 | 0.0564 |
| b8 × 8192 | 0.5879 | 0.6575 (**0.89x**) | 0.5873 | 0.5901 | 0.5875 |

- **数值全部 `err=0.0000`,逐位等价**——流水重排没有改变计算顺序语义。
- **没有任何一档超过 1.01x**;stages=2 在 b8×8K 还有 11% 回归。
- 短形状零税(0.0563 vs 0.0564),E61 红线通过。

数值与边界的设备契约另有 17 项(`tests/test_sw_pipeline_device.py`),覆盖单页请求、
ragged batch、tail 页、head_dim=256 的 D-split,断言是 `torch.equal`(**逐位**,
不是"小于容差"),全绿。

### 为什么没做引擎级 A/B(以及为什么它本来就分辨不出来)

尝试过(`.agent_loop/run/h74_ab.sh`),但两次都被外部栈污染而作废:
- s0 臂 19:14–19:20 出数:c1 8.779 / c8 16.673 / 8K 39.904 / 16K 68.770;
- 与权威基线(9.700 / 11.964 / 36.788 / 65.287)比,**偏差 −9.5% ~ +39.4%**——
  远超任何可接受的单变量偏离,该窗口已被外部 serve 污染,数据已删除;
- 根因:外部权威栈(port 8000)会自动重启,退出后残留显存不立即释放,
  我们启动时只看到 43.6 GiB 空闲 < 0.9×110.72 GiB 需求,`Engine core init failed`。

更关键的是**分辨力**问题——就算拿到干净窗口也测不出来:

```
case              TPOT ms  kernel ms  占比   stages=2 折算到端到端
b1 × 128 (c1)       9.70     0.0190   0.20%        0.02%
b8 × 8192          36.79     0.5879   1.60%        0.18%
b8 × 16384         65.29     1.1070   1.70%        0.19%
```

attention 主核只占 TPOT 的 **0.2%~1.7%**;流水最坏影响 11%,折算到端到端仅
**0.02%~0.19%**,比引擎 A/B 的噪声底(约 ±0.5%)还小一个量级。

⇒ 以**内核级微基准**为判决依据是恰当的,不必再投引擎窗口。这一点也反过来验证了
H42-③ 的 disproven 判决:那条路即使实现正确,端到端最多也只有 ~0.2% 的量级。

### 为什么无效(与 DeepInfer 的真实差异)

DeepInfer 的 `T.Persistent` 本质是**任务摊平器**:固定 `num_cores` 个核轮流吃
`domain` 里的多个 task。我们用 `T.Kernel(batch * kv_heads)` 已经把 grid 铺满,
**每个 block 只处理一个 (batch, kv_head),页循环是 block 内的串行链**:

```
K copy → QK^T GEMM → softmax → P copy → V copy → PV GEMM
```

`k_l1` 既被本轮 K 拷贝写、又被本轮 QK^T 读,`v_l1` 同理——**跨迭代没有可重叠的
独立数据流**,编译器即使加 stage 也找不到能提升的拷贝。这与 DeepInfer 的
`load_paged_kv_cache`(纯拷贝任务,天然可流水)场景不同。

⇒ **H74-① 判决:disproven(对 attention 主核)**。开关保留(默认关,零成本),
但**不应启用**;也不再投入设备窗口。

### H74-② 页索引 kernel 化:实测后判定不适用

DeepInfer 用 `get_kv_index` 这类 Persistent 内核替换 Python 侧页表展开。先量化我们
的 glue 成本(`benchmarks/probes/h74_glue_cost.py`,纯 torch,不启动引擎):

| 环节 | c1(width=64) | c8(width=64) | c8(width=4096) |
|---|---|---|---|
| `_expand_page_table` | 0.07 µs | 0.05 µs | 27.5 µs |
| `ragged_from_page_table_device` | **149.4 µs** | 148.9 µs | 189.6 µs |
| 三段 ragged 重建(decode 内联版) | 42.4 µs | 42.3 µs | 41.8 µs |

关键发现:`ragged_from_page_table_device` 的 149 µs **与 batch/width 完全无关**——
它是 `masked_select()` 产生**动态形状**输出导致的固定 host 同步。但它只被
**prefill** 路径调用(`vllm_ascend.py:571`),而该路径的代码注释已明确:

> "Unlike decode, a prefill step always runs eager (never inside an ACLGraph capture),
> so the host-side work here — per-request loops, `.item()` syncs, a `masked_select`
> page-table walk — is acceptable"

**decode 热路径(`TileInferDecodeAttention.run`)早已是固定形状 scatter**,没有这个开销。
⇒ **H74-② 判决:不实现**(优化对象不在热路径上;做成内核只会多一次 launch 税)。

### 未实现(留给后续,按价值排序)

- `load_paged_kv_cache.py` 的 KV 量化路径:暂不需要(当前 bf16)。
- `swa_index.py` 的 SIMD 写法:仅作参考。

---

## H76 fused_sample 移植(2026-10-20):它是什么

上一节的第 ③ 项现在实现了。这一节解释它**到底在做什么**,以及为什么值得做。

### 原生路径在做什么(vLLM Ascend 采样)

```
probs = softmax(logits)           # ① 全词表 float32 物料化, V≈248K
sorted_probs, idx = probs.sort()  # ② O(V log V) 全量排序
cumsum(sorted_probs)              # ③ 前缀和找 top-p 截断点
sorted_probs[k-1]                 # ④ 第 k 大作 top-k 阈值
multinomial(probs)                # ⑤ 再抽一次随机数
```

每一步都要把 248K 个 float32 在 HBM ↔ UB 之间搬一遍。**②的排序是最贵的**,
但它的产出其实只有一个**标量阈值**——排在后面的顺序没人用。

### DeepInfer 的做法:不排序,直接二分出那个阈值

关键洞察:**top-p 和 top-k 都只是「按概率值切一刀」**。既然只要阈值,就不需要
有序排列。

于是它在值域上二分:

```
每轮: 取 [lo, hi] 的 1/4、1/2、3/4 三个点 (q0, q1, q2)
       扫一遍全词表,同时统计三组:
         (prob >= q0 的质量和, 数量)
         (prob >= q1 的质量和, 数量)
         (prob >= q2 的质量和, 数量)
       根据结果把区间缩到 1/4
16 轮 ≈ 4^-16 ≈ 2.3e-10 精度
```

**一轮扫描同时测三个点**,所以 16 轮只需要 16 遍扫描,却把区间缩小到 1/4^16。
这比排序的 O(V log V) ≈ 248K × 18 ≈ 450 万次比较要省得多——虽然扫描多遍,
但每遍是纯 SIMD 比较(64 lane 并行),没有排序的随机访存。

还有一个坑:**阈值附近的同值项**。多个 token 概率完全相同时,top-p 的"切在
哪里"会影响保留质量。DeepInfer 用「边界同值项配额 + 前缀扫描」精确控制保留
几个。我们的移植用了更保守的策略(见下)。

### 我们的移植:在 exp 空间搜索 + 简化同值处理

**① 在 exp 空间搜索,省掉 248K 次除法。**

令 `e_i = exp(x_i/temp − max)`,则 `p_i = e_i / total`。比较 `p_i >= q` 等价于
`e_i >= q·total`。于是搜索区间固定为 `[MIN_PROB, 1.0]`,top-p 目标质量写成
`top_p · total`——**全程零除法**。

**② 累积用 `alloc_local` 可变寄存器。**

tilelang 0.1.15 的两条硬约束(都是编译错误逼出来的):
- `MODE_MERGING` 必须赋给可变 SIMD 寄存器(`alloc_local`),不能直接当 `vsts` 参数;
- 向量不能在 `T.serial` 迭代间自赋值(`Immutable variable ... used outside its
  defining region`)。

**③ 选择阶段按向量归约 + 单向量内标量尾扫。**

不做全词表标量循环(248K 次串行不可用):每 64-lane 向量先 `vcadd` 求和,
目标不在本向量就整段跳过(一次标量加法);只有目标所在向量才逐 lane 扫 64 次。

### 与 H36 的关系(互补,不是替代)

| | H36(`greedy.py`) | H76(`fused_sample`) |
|---|---|---|
| 覆盖场景 | 全批 greedy(temperature=0) | 有 top-p/top-k 的真实采样 |
| 做法 | **跳过** fp32 cast,直接 argmax | temp+softmax+阈值+采样**融成一个 kernel** |
| 省掉什么 | 全词表 cast + 处理器链 | 全词表 `sort`(O(V log V)) |

**守卫会让全 greedy 批次走 H36**(H36 更轻,连 softmax 都不做),只有真正存在
`top_p < 1` 或 `top_k > 0` 的请求时才由本内核接管。

### 实现状态

- 内核:`tileinfer/kernels/sampling/fused_sample.py`(`do_sample=True/False`
  两个分支均编译通过并**在设备上数值验证**);
- 集成:`tileinfer/integrations/fused_sample.py`,7 条守卫 + 逐调用回退;
- 契约:`tests/test_fused_sample.py`(13 项)+ `tests/test_fused_sample_integration.py`
  (23 项),**36 项全绿**;
- 设备数值验证:`benchmarks/probes/fused_sample_device_test.py`,9 项判据
  (greedy 逐位 / 保留集 / top-k 语义 / 分布拟合 / 边界)**全部 PASS**
  (V=256 与 V=16384 多 chunk 两种配置均验证)。

### 设备实测:数值正确,但**只在 batch ≥ 20 时才有性能收益**

`benchmarks/probes/fused_sample_perf.py`,V=248320(真实词表),中位延迟:

| batch | 融合内核 | torch `sort+cdf` | 比值 |
|---|---|---|---|
| 8 | 2.882 ms | 0.978 ms | 0.34×(慢 2.9×) |
| 32 | 2.880 ms | 4.436 ms | **1.54×(快)** |

**判决(有条件通过)**:融合内核的延迟**与 batch 无关**(每行一个 block,
并行跑),而排序路径随 batch 线性增长,交叉点在 batch ≈ 20。
- batch ≥ 20(高并发解码):**值得启用**,32 时快 1.54×;
- batch < 20(低并发 / 单请求):**不值得**,应回退原生 sort。

因此集成层的守卫**必须包含 batch 门槛**,不能只要 `top_p<1 / top_k>0` 就接管。

### 端到端验证:尚未完成(2026-10-21 尝试,被环境阻塞)

**未验证项**(下面三项都还没有端到端数字):

| 验证项 | 现状 | 缺什么 |
|---|---|---|
| 引擎级 TPOT 收益 | 只有内核级 2.88ms 微基准 | 需要 off/on 两侧同 prompt A/B |
| `fused_sample_calls` 实际命中数 | 未在服务里观测过 | 需要一次成功启动 + 请求 |
| 真实 batch 分布下的收益 | 假设 batch≈32 | 生产 batch 可能远低于交叉点 |

**本次尝试(3 次启动,全部失败)**:

根因是**主机内存 pin 配额不足**,与内核无关:

```
# dmesg
[ascend] [ERROR] [svm] [svm_pin_user_npages_fast 87] Get_user_pages_fast fail.
[ascend] [ERROR] [svm] [svm_pin_host_pa 124] Pin host addr failed. (ret=-22)
```

```
$ ulimit -l
65536                    # 64 MB
$ ulimit -Hl
65536                    # 硬限制同样是 64MB
$ ulimit -l unlimited
bash: ulimit: max locked memory: cannot modify limit: Operation not permitted
```

权重 64.7GB 在 torch.compile 阶段需要 pin 大块主机内存,64MB 配额必然失败,
EngineCore 静默退出(日志无 traceback,只有 APIServer 侧
`Engine core initialization failed`)。把 `--gpu-memory-utilization` 从 0.90
降到 0.75 仍然失败——瓶颈是锁定内存配额,不是显存。

**附带发现**:失败后残留的 `VLLM::EngineCore` 会成为孤儿进程卡在驱动层,
`kill -9` 也不能立即生效,占住 119GB HBM。清理后降到 5.4GB。下次跑 A/B 前
应先 `npu-smi info` 确认无残留进程。

**恢复验证的前置条件**:在 `ulimit -l` 至少 ≥ 8GB 的会话/容器里启动服务
(需要 `--ulimit memlock=` 的容器参数或宿主机 limits.conf,当前 shell 无法自行提升)。

### 踩坑记录(每条都是实测定位,不是推测)

1. **UB 越界 → `507035`**:`vals_ub` 之外还有 `vecsum_ub`(`n_vec` 个)、
   `em_ub`/`vec_ub`(各 64 个)和一堆标量缓冲。`ub_cap=16384` 时叠加后越界。
   最终默认 **8192**(40 chunk),实测稳定且比 4096/2048 都快(扫描次数更少)。
2. **int32 的 `vcadd` + `vsts` 写 int32 shared 会崩**。所有归约(token 计数)
   统一用 float32 —— 词表 ≤ 2^24 时 float32 可精确表示每个整数。
3. **`MODE_MERGING` 语义不可依赖**:掩码累加改用「先 `vsel` 置零、再无条件
   `vadd`」,并直接赋给 `alloc_local` 寄存器。
4. **`vcadd` 不能省**:逐 lane 累加后必须 `vcadd` 再 `vsts`,否则只写出 lane 0
   的部分和(`total`/`keep_mass` 偏小 64 倍)。
5. **16 轮搜索精度不够**:搜索区间是 float32 位模式上的 `[2^-32, 1]`,跨约 2^31,
   16 轮后只剩 `2^-1` → 阈值偏差 5%、保留集偏大。**改为 32 轮**后阈值与 torch
   逐位吻合。
6. **count 被覆盖**:三分位搜索里 `stat_ub[3..5]`(三档数量)曾被直接赋值而非累加,
   `n_chunks>1` 时只剩最后一个 chunk 的计数 → top-k 完全失效。必须累加。
7. **`for g` 内穿插 SIMD 块 + 标量块 → 多 chunk 时结果错乱**(SIMD 写与标量读
   跨迭代重排)。改成「每 chunk 一个纯 SIMD 块 + 一次 `n_vec` 标量判断」。
8. **动态 global 偏移的 `T.copy` 会错位**(`logits[row, pick:pick+64]`)。第二遍
   定位改为按 chunk 顺序重放、只在目标 chunk 生效。
9. **全词表标量前缀和不可用**:248K 次串行 → 47ms,比 sort 慢 48 倍。

### 一个必须说清的限制

内核**不保证与原生 `multinomial` 逐个随机数相同**——随机数流实现不同,这种比较
没有意义。验证走统计口径:大样本频率 vs 过滤后概率,以及阈值一致性。



### 明确不可借鉴

- **FlashMLA / DeepGEMM / DeepEP**:MLA 架构专用。我们是 Qwen 的 GQA,内核不通用。
- **SWA bounded replay**(`YOCO_PREFILL_TRUNCATION_MODE=compact`):依赖 DeepSeek V4
  的 provider/consumer 分层(`yoco_boundary_layer_idx`),在边界层把历史 token 切掉
  只喂 consumer 层。Qwen 无此结构,没有对应插点。
- **MTP/DSpark 投机解码**:需要模型自带 draft head,当前 Qwen3.6 部署未启用。
- **异步调度**:DeepInfer 用 C++ (`csrc/engine/scheduler.hpp`) 实现,但我们已在
  **H50** 证实 "vLLM 的 `async_scheduling` 默认已开",这不是新杠杆。
- **`fused_transform` 的融合思路**:我们 H68-a 已实测——融合后 strided slice 喂
  conv1d 走 AscendC 慢路径,引擎面 **+16.1% 恶化**。DeepInfer 能做到是因为其下游是
  MLA(无 conv1d),不能直接照搬。

### 建议(待裁决,本轮不实现)

1. **用 `T.Persistent(num_stages=2~4)` 重构 `paged_decode_ascend950` 的页循环**——
   这是唯一能解释"同栈、同硬件,DeepInfer 能贴带宽而我们 897 GB/s"的结构差异。
   注意:H42-③ 已证明 b8 场景 plain 已接近带宽顶,所以收益应主要在
   **c1/b1 短上下文**(那里 grid 只有 2 blocks,正是 H67 的靶面)。
2. `_expand_page_table` + ragged 重建的 Persistent 化(每步固定税)。
3. H36 升级为 kernel 级融合采样(参考 `fused_sample.py` 的分块 + 双 CDF 设计)。

---

## H67 batch-1 decode attention 主核:取证棒就绪(2026-10-20 iter-208)

### 已知归因(E268/E266,双侧 kineto 对差)
TI c1 的 attention 主核(`main_kernel`,tilelang 自研)单步 **0.472 ms × 19.9 发射**;
FIA 的 attention 族合计 **0.133 ms** → **单步差约 0.34 ms**,是 c1 缺口的最大单点。
另 datamove **+0.083 ms**(q_pad / output 胶水)为第二项。

H66 当时因「可削面 0.44ms < 0.5ms 立项门槛」未立消除假设,登记为 H67 候选。
但该门槛是**当时的优先级判据**(前提是还有更高优先级的项),不是"无法优化"的证明。
在 c1 已成为唯一负项的现在,前提已不成立(见 D144-3)。

### 取证棒
`.agent_loop/run/h67_bench.py`(等窗执行,不改在役代码):
- 真实形状扫描:ctx ∈ {128, 512, 2048, 8192} × ktp ∈ {0, 16, 32, 64, 128}
- 与 SDPA 参照对照,得到每档 ratio

**预登记判据**:
| ratio | 判决 |
|---|---|
| ≤ 1.5 | **H67 立项**(存在明确消除路径) |
| ≥ 2.5 | **H67 关闭**(内核已达该架构上限,缺口应归因到 vendor 专用指令) |
| 中间区 | 记录最优配置,作为 H42-③ 双缓冲的补充靶面 |

## H67 根因定谳:短上下文 split-KV 不触发(2026-10-20 iter-208,E291)

### 之前的归因是半对的
H66 的双侧 kineto 对差给出:TI `main_kernel` **0.472 ms/步 × 19.9 发射**,
FIA attention 族 **0.133 ms/步** → 单步差 ~0.34 ms。
当时记为"batch-1 decode attention 主核慢",并因 0.44ms < 0.5ms 门槛未立项。

### 真正的原因:并行度只有 2
反推单次调用成本:TI `0.472 / 19.9 = 0.024 ms`,FIA `0.133 / 19.9 = 0.007 ms`
→ **TI 是 vendor 的 3.4 倍**。这个数字与代码注释精确吻合
(`tilelang_ascend950.py:152` 写明 split-KV「measured **3x** at batch 1」)。

而 split-KV 的切分单位是 **KV 页**(`paged_decode_split` 里 `BC = page_size = 128`),
调度器按 `kv_tile_pages` 整页切分。基座 ktp 分带 `1:16,2:16,4:32,8:128`,c1 走 ktp=16:

| ctx | pages | splits | needs_merge | grid |
|---|---|---|---|---|
| 128 | 1 | 1 | **False** | **2 blocks** |
| 512 | 4 | 1 | **False** | **2 blocks** |
| 2048 | 16 | 1 | **False** | **2 blocks** |
| 8192 | 64 | 4 | True | 8 blocks |

**c1/c8 的短序列永远只有 1 个 split,grid 只有 `1 tile × 2 kv_heads = 2 blocks`。**
0.34ms 的主体是并行度不足,不是 tilelang 内核算得慢。

### 三条改造路径都需设备验证(已排除离线实施)
| 路径 | 障碍 |
|---|---|
| ① 页内切分 | 内核 `page_starts/page_lens` 是页粒度;`limit` 只能截断**尾页**,不能跳过前部 |
| ② 降低 `KERNEL_PAGE_SIZE`(128→32) | 20+ 处依赖(池视图 / shunt 阈值 / ragged 展开 / dummy 形状) |
| ③ 10 层合并单次发射(game 2→20 blocks) | 需内核支持 KV 指针数组 + 改图捕获节点 |

### 判别实验(已就绪)
`.agent_loop/run/h67_bench.py` 用**不同有效页大小**构造不同 split 数,回答唯一问题:
**并行度不足,还是内核效率差?**

| 观测(ps 128→32 的耗时变化) | 判决 |
|---|---|
| 下降 ≥25% | 并行度主导 → **H67 立项**(方向=token 级 split) |
| 变化 <10% | 内核效率主导 → **H67 关闭**(转 H42-③) |
| 中间 | 记录曲线,作为 H42-③ 补充靶面 |

### 胶水项(0.083ms)已到带宽上限
q copy 0.08MB/步 + out copy 0.33MB/步 = 0.41MB/步,按 0.083ms 反推 **4.93 TB/s**
(接近峰值),不是发射税。`out_pad` 的 br=32 行中仅 group=8 行有效(4x 冗余),
但切片后 `copy_` 已是最小搬运,优化空间 <0.02ms。

## H67 实现:token 级 decode split(2026-10-20 iter-208)

### 为什么分带(ktp)解决不了 c1
`TILEINFER_KV_TILE_PAGES_POLICY` 按 **bucket** 分带,而 c1 的约束在 **width(页数)** 维度:

```
splits = ceil(pages / ktp)
c1 @ 128 token → pages = 1 → 无论 ktp 调到 1，splits 恒为 1
→ grid = 1 tile × 2 kv_heads = 2 blocks
```

用 E64 波次模型核算:

| ctx | pages | ktp=1 | ktp=2 | ktp=4 | ktp=8 | ktp=16 | ktp=32 |
|---|---|---|---|---|---|---|---|
| **128** | **1** | 1t | 1t | 1t | 1t | 1t | 1t |
| 512 | 4 | 4t | 2t | 1t | 1t | 1t | 1t |
| 2048 | 16 | 16t | 8t | 4t | 2t | 1t | 1t |

**分带调的是"每刀切多宽",但这里只有 1 页可切。** 必须改切分粒度。

### 实现
1. **调度器** `plan_decode_tiles_token_split`:按 `tokens_per_split` 个 token 切块,
   每块记录 `token_starts/token_lens` 及其覆盖的页范围。
2. **内核** `paged_decode_split`:页循环内把 token 窗口换算成页内 `[lo, hi)`,
   施加到 softmax 掩码(越界置 -1e30)。
3. **等价性**:未启用时 `tok_start=0, tok_len=page_lens*page_size`
   → `lo=0, hi=BC`,与旧版**逐位一致**(由构造保证,非测试保证)。

### 效果
| 场景 | 页级 split | token 级(32/块) |
|---|---|---|
| c1 @ 128 token(1 页) | 1 tile、**grid 2** | **4 tiles、grid 8** |
| c1 @ 512 token(4 页) | 1 tile、grid 2 | **16 tiles、grid 32** |

### 契约(8 项,`tests/test_decode_token_split.py`)
- `tokens_per_split == page_size` 时与页级切分(ktp=1)tile 数**一致**(回归基线)
- 1 页场景可切出 4 个 tile(核心用例)
- tile 数 ≈ `ceil(total / tokens_per_split)`
- 各 tile 的 token 窗口**互不重叠且并集完整**
- 每个 tile 的页范围**包含**其窗口所需的所有页
- 多请求独立切分 / 拒绝非法参数 / 空输入

**全量回归 263 项通过。**

### H67 实现缺陷:token 窗口只在 SIMD 分支生效(2026-10-20 E294,已修)

自查发现 `paged_decode_split` 的 token 窗口 `[lo, hi)` 只写在
`_simd_softmax_enabled()` 分支里;非 SIMD 的 softmax 分支仍用原来的
`c < limit` 掩码,**完全忽略 token_starts/token_lens**。后果:

- 同时开 `token_split` + 关 SIMD softmax → 每个 split 都算全量 KV,
  再被 merge 加权 —— **结果错误且不报错**(最坏的一类缺陷);
- 默认路径(`token_split=False`)不受影响,所以既有 263 项回归全绿。

修复:非 SIMD 分支补同一套 `lo/hi` 换算,`c < hi && c >= lo`;
`hi` 先 clamp 到 `limit`,因此 `c < hi` 严格覆盖原 `c < limit`(逐位一致)。

**教训**:实现"双分支内核"的新语义时,必须两个分支同时改;
本轮契约测试只覆盖调度层(planner),不覆盖内核分支,所以漏检。
后续 H67 设备数值验证须显式跑 **SIMD on/off 两组**。

### 状态与下一步
**默认关闭**(`token_split=False`)。启用前必须按序完成:

1. `h67_bench.py` 判别实验(确认并行度主导而非内核效率)
2. 数值验证(token 分块 vs 页级/单遍 oracle,含尾页、跨页、ragged,**SIMD on/off 两组**)
3. 引擎 A/B,门槛:**短场景 attention 主核 ≥1.5x 且 8K 不劣化**
4. 达标后接入集成层(按 bucket 选 `tokens_per_split`)

## H28 shared expert 恒选 E+1 折叠(iter-210 实现,等窗验证)

机制:shared expert(gate_up+swiglu+down ≈1.2ms/步串行 dense 链,E79/E91)折叠进
routed grouped 路径作第 E+1 号恒选专家;sigmoid(gate_logit) 门控折叠进 topk_weights
末列;gate_logit 经 router gate 权重扩 1 行免费产出。开关 `TILEINFER_SHARED_EXPERT_FOLD`
(默认关=零行为变化),`TILEINFER_SHARED_EXPERT_FOLD_DRYRUN` 自检。

实现面(E300 vendor 审查):在役 BF16/TP1,multistream 被 vendor `platform.py:32`
硬编码关闭 ⇒ 重叠路线作废,折叠为唯一主路线。三注入点:
①`AscendUnquantizedFusedMoEMethod.process_weights_after_loading` 前置拼接(NZ/pad 前,
H73 形态剔除 gate 末行);②模块级 `select_experts` 包装(renorm 后追加列,门控语义
独立于 routed renorm);③gate 扩列在首个 forward(warmup,先于 capture)懒完成。
shared 串行链整体 bypass(`_forward_shared_experts` 返回零张量,合加语义不变)。

验证状态:CPU 契约 12 项绿(纯函数数学/折叠时序/扩列降级/bypass/install 幂等);
相关集成回归(shared_gate_fuse+optin_gate)18 项绿。

设备链(iter-209,提交 d084bd2):
- **install 接线补齐**:fold.install() 挂进集成 install()(显式 opt-in,默认关);
  metrics dumper 合并 fold_layers/fold_forward/fold_skipped(在役探针判据,E299-3:
  模块 logger 无 handler,安装判定不靠 grep 日志)。
- **残留四项审查(E300-5 定谳)**:a) `w13_weight_list` 仅在 fused_mc2+dynamic_eplb
  创建(fused_moe.py L129-134),在役配置不触发,免扩 list;b) 合加点=bypass 返回
  zeros,合加语义不变;c) eplb/tid2eid:TP1/EPLB-off 下恒等(tid2eid=None 通道),
  sglang #39580(EPLB rebalance 输出错乱)反向佐证——EPLB+fold 组合未来须重审;
  d) 非 internal router:GEMV 降级路径保留(`_tileinfer_fold_ext` 未置位时走
  `expert_gate(hidden)`)。
- **设备 oracle 2/2 绿**(m∈{1,8,32},TOL 3e-2):折叠 grouped(E+1) vs 分解路径,
  GMM 用在役原语(moe_mlp unquant 链同参);gate 扩列契约过。
- **微基准门槛 PASS**:`h28_fold_bench.py` 200 iters 流水口径——b1 **1.400x** /
  b8 **1.117x** / b32 **1.072x** / b128 1.038x,**decode geomean 1.188x ≥1.0x**(H28 线门槛)。
  产物 `.agent_loop/run/h28_fold_bench.json`。

**集成接缝修复(iter-212/213,d1d2185+064cb00)**:
- E310:Qwen3.6 非 internal router(gate 无 weight_fp32)⇒ router_logits [M,256] 无 E+1 扩列,
  旧 wrapper shape 判据静默 passthrough(fold_forward=0)。修复=双态判据(列数==fold_expert_id[E]
  →plain 透传+expert_gate GEMV 现算 sigmoid 门控,与 0_23_0 L355-357 逐点等价;==E+1→extended 切片)。
- E312:折叠激活后 ids [M,k+1]=18432 与 dispatcher `active_num=num_tokens×td.top_k`=16384 失配
  (aclnnMoeInitRoutingV3 EZ0002,图捕获期 EngineCore init 必炸)。修复=comm 单例 td.top_k
  k→k+1 补升(token_dispatcher.py:405 唯一消费点;与 expert_num E→E+1 同点同法,全局幂等;
  层本体 top_k 保持 k=8)。fail-closed:top_k 缺失或单例不匹配即 fold_skipped 回退。
  契约 20/20 绿。

**权威 A/B(iter-213,gen4 单变量,E313)——PASS,纳入 TI 当前最优配置**:
on 臂 STARTUP_OK,40/40 层折叠,在役探针 fold_layers=40 / **fold_forward=18280** /
fold_skipped=0 / fallbacks=0 / capture_calls=40(图模式端到端)。

| 场景 | TPOT off | TPOT on | dTPOT | 吞吐 Δ | 判定 |
|---|---|---|---|---|---|
| 128x128_c1 | 9.69 | 9.49 | **−2.04%** | +1.00% | PASS(≥1%) |
| 128x128_c8 | 11.93 | 11.65 | **−2.29%** | +1.74% | OK(≤3%) |
| 128x512_c8 | 11.98 | 11.69 | **−2.38%** | +2.09% | OK(≤3%) |
| 8192x128_c8 | 36.49 | 36.59 | +0.27% | −0.34% | OK(<5%) |
| 16384x128_c8 | 65.71 | 66.34 | +0.96% | −0.76% | OK(<5%) |

收益落点 ~0.24-0.29ms/步(E300 上界 0.5-0.7ms 的 ~40%,预期内)。四门槛全闭合:
oracle 2/2 → 微基准 geomean 1.188x ≥1.0x → opt-in → 权威 A/B PASS ⇒ **H28 线收官**。
`TILEINFER_SHARED_EXPERT_FOLD=1` 计入 TI 侧当前最优配置(权威口径后续 A/B 均含)。
战略记账:c8 对 FIA 原始基线 18.01→11.65 = **−35.3%**(≥30% 保持);c1 对 FIA 锚点
8.766 仍 +8.2% 落后(batch-1 弱点),为 30% 总目标唯一残余缺口。

## H77 dense b1 大形状权重物理预排 NT→NN(2026-10-21 iter-215,实现落地等设备门槛)

机制(E297 立论):c1 档 dense GEMV(MMV3 族)是最大单一设备域(E267:3.532ms/步,
37.5%)。E255 微基准×E269 引擎逐形交叉:qkvz[2048→12288]/attn_qkv[2048→9216]/
o_proj[4096→2048] 三大形状引擎内有效带宽 1.30/1.28/0.94TB/s vs 微基准流式
1.62-1.65TB/s——引擎缺口 1.24-1.3x,聚合 ~450μs/步 ≈ c1 设备链 4.8%。已排除轴:
换后端(E255 matmul 视图 0.877-1.047 全败)、NZ 布局(E42/E100 零收益,且
weight_nz_mode=2 引擎启动 abort、默认=1 bf16 不转 NZ ⇒ 在役权重实态 ND-NT)。
未测正证据轴=E100 的 **ND 物理连续 NN 布局**(w_nn=[K,N],x@w_nn):M=16 微基准
out_proj/attn_qkv +11%,M=1 与引擎内从未测过。

实现(opt-in,默认关=零行为变化):
- `tileinfer/integrations/dense_nn.py`:PWAL 基类钩子(UnquantizedLinearMethod.
  process_weights_after_loading,与 H73 各自独立 marker 互不吞包装)→ 形状族识别
  (12288×2048=qkvz/9216×2048=attnqkv/2048×4096=oproj,与 H28/H73 域零交集)→
  加载期一次 `w.t().contiguous()` 预排 + 实例级 quant_method.apply 替换:
  `x.shape[0]≤TILEINFER_DENSE_NN_MAXM(默认 16,E100 证据域)` 走 matmul NN,
  否则回落原 apply(prefill 大 M 保持 NT)。TP≠1 全族跳过(RowParallel apply 含
  all-reduce,直替破坏通信语义)。
- 开关:`TILEINFER_DENSE_NN`(1=三族全开,亦可 qkvz,attnqkv,oproj 子集)/
  `TILEINFER_DENSE_NN_MAXM`/`TILEINFER_DENSE_NN_DRYRUN`。HBM 代价 ≈+2.55GB
  (三族预排副本;可子集降档)。
- 在役探针:metrics dumper 并入 dense_nn_modules/forward/fallback(ON 臂判据
  dense_nn_forward>0)。

### H79 v2=weight-swap 重做(E320 v1 失败定谳,2026-10-21 iter-234)

v1 引擎 A/B 判决无效(E320):on≈off 全场景 ±1.12% 噪声级,生效探针三重失守
(installed_hits=0/dense_nn_forward=0/fallback=80=仅 prefill eager)——vllm 0.23
默认 cudagraph_mode=FULL_AND_PIECEWISE,FULL graph decode 全程不经 python,
实例级 qm.apply 闭包被整体旁路。**微基准 1.128x 与引擎 0 收益并存=集成层失败,
不构成假设证否**。

v2 机制(同一假设的零分支实现,`dense_nn.py` 全量重写):
- PWAL 后置钩子(core+leaf 双类同挂:UnquantizedLinearMethod 与
  AscendUnquantizedLinearMethod,独立 marker 互不吞包装;后置=orig 完成后再换形,
  保证 swap 是 maybe_trans_nz(E43 bf16 恒等)之后的最后写者)→
  weight.data 物理替换为 `w.t().contiguous().t()`(逻辑 [N,K] 不变、存储 [K,N]
  连续、stride=(1,N))→ vendor apply(unquantized_gemm→F.linear)按 stride 自动
  走 NN 内核。零热路径分支、零 python 计数依赖,图 capture 前定形(地址稳定);
  幂等 marker+stride 自愈复核(被改写时重换形)。
- E321 stride 敏感性微基准(h79_stride_bench,引擎精确链,HBM-cold 流式):
  o_proj M=1/8/128 = **1.169/1.143/1.160**(全段 +14~17%)| qkvz M=1=0.919
  (decode 段亏损)| attn_qkv M=128=0.876(prefill 段亏损)⇒ **部署白名单=oproj
  单族**(静态族粒度是图模式唯一安全面;M 级分支=v1 死路),A/B 挂
  `TILEINFER_DENSE_NN=oproj`。HBM 代价 ≈+0.64GB(40×16MB)。
- 生效探针(图模式可见面):metrics dumper 并入 dense_nn_modules/stride_ok/
  stride_bad(ON 臂判据 stride_ok=modules=40;v1 forward 计数面作废)。
- 契约测试重写:CPU 14 项绿(门控/形状族/换形布局与数值/qm 零触碰/自愈/
  双类链/探针面)+ 设备 oracle 6 项绿(M∈{1,8}×三族,stride 断言+allclose 2e-2)。
- 权威 A/B(h79_ab.sh):基座=现役最优(capture8+SIMD+ktp 分带+H61+H28 fold),
  唯一变量=`TILEINFER_DENSE_NN=oproj`;off 臂复用同日 h77_off 基线(D162 臂级
  复用先例);判据 c1 TPOT ≥−1%,其余 |Δ|≤3%,短场景 |Δ|<3%。

判决(iter-234,E324,**两轮 A/B 交叉,2×2 独立 serve**):
- 首轮(off=同日 h77_off):c1 9.50→9.44 −0.71%,c8 三场景 −0.24~−0.33%,
  长上下文 −0.36%/−0.39%,吞吐 5/5 全正(+0.39~+1.40%);
- 配对复现(同会话新跑 off/on):c1 9.54→9.42 **−1.21% PASS**,其余 −0.26~−0.32%
  (16384 +1.00% 在 ±3% 带内),吞吐 4/4 正;
- 交叉判读:on 臂 c1 两独立 serve 9.42/9.44 非重叠低于 off 臂 9.50/9.54,
  效应真实,**均值 ~−0.96%(≈90μs/步)略低于假设 ≥100μs(记限定语)**;
  c1 吞吐 +1.4~+1.7%;生效探针三次 serve 全绿(modules=40/stride_ok=40/bad=0)。
- **VERDICT: PASS(带限定:幅度 ~90μs<100μs 估计)**;
  `TILEINFER_DENSE_NN=oproj` 自 iter-234 起计入在役最优配置面(D164),
  后续权威 A/B 基座两臂同含;opt-in 环境变量语义不变,默认仍关。

### H77 v1 原始实现存档(被 H79 v2 取代,2026-10-21 iter-215)

验证状态:
- CPU 契约 14 项绿(tests/test_dense_nn.py:门控/形状族/预排数值/shim 语义/
  wrapper 链与 H73 marker 独立性);
- 设备数值 oracle(tests/test_dense_nn_device.py,M∈{1,8}×三族 vs F.linear NT
  TOL 2e-2 + 大 M 回落逐位一致)与四轴微基准(`h77_nn_bench.py`:
  nt_nd/nn_nd/nz_eng/lin_nz,HBM-cold 流式口径,门槛=任一大形状 M=1 NN/NT
  ≥1.05x 且 allclose)——**等设备窗口**(哨兵 h77_gate_chain.sh 已武装,随
  iter-214 npu 套件退出自动补跑);
- 微基准门槛通过后才进权威 A/B(`h77_ab.sh`,基座=capture8+fold,唯一变量
  DENSE_NN;判据 c1 TPOT ≥−1%,其余 |Δ|≤3%)。

## H47 图模式拷贝链:关闭(可达性,2026-10-21 iter-235,E327)

取证(iter-235,capture 期探针两轮):
- v1(多进程默认)暴露 V1 架构盲区:EngineCore 为子进程,主进程 profiler 仅见客户端侧
  6 计数微操作 → 修复=`VLLM_ENABLE_V1_MULTIPROCESSING=0` 单进程模式(vllm 0.23 官方
  支持路径),capture 期算子栈落进主进程;取证口径对齐在役最优基座
  (capture8 `cudagraph_capture_sizes=[1,4,8,32]` + c8 权威并发);
- 出数:capture 期 `aten::copy_` 43902 次/9227ms、`aten::slice` 59086 次/74.8ms、
  `vllm::unified_attention_with_output` 190 次/2960ms、dynamo 全图编译 44 次
  (`fullgraph_capture` 1.6s)——拷贝链为 capture/compile 期高频 host 操作,
  与 E128 设备侧 240 次/步 replay 口径互为表里;
- `with_stack=True` 在 torch_npu/profiler 组合下不产出 Call stack(23GB trace 实测
  grep=0),python 栈级归因不可用(KI-9 记档),改走静态源码路径。

判决(三重证据,**closed**):
1. **份额量化**:拷贝链主体为 vendor 图模式共性行为(TI 240 vs FIA 216 次/步,E128);
   TI 独有面=10 层 full-attn×2 次 copy_(q_pad 入 pad + out_pad→out 写回,
   tilelang_ascend950.py:329-349)≈20-24 次/步,全消 ~50-58μs/步 ≈ TPOT 0.3%,
   远低于预登记「Slice ≥50% 削减→≥1.5%」门槛;
2. **共享面口径红线**:大头(~90%)位于 vllm 主仓文件(qwen3_next.py:416
   `torch.empty_like`+`output=` 写回 + runner bucket 填充),修改将同步作用于 FIA
   基线臂,污染「vs vLLM 原始路径」权威口径;
3. **带宽顶**:E291 已判 TI 胶水拷贝链 0.083ms 反推 4.93TB/s,近峰值无税可回收。
- **VERDICT: 不可达关闭**(机制存在但 TI opt-in 域内数学不可达);
  翻案条件=vendor 上游自行消除写回拷贝,或共享面口径经用户裁决变更。

## 全域瓶颈证明汇总(系统级专项终止条件 b)素材,c1 唯一残余缺口)

30% 大账现状(权威口径 vs vLLM 原始昇腾算子路径,E258 重锚+H28/H73/H79 入役后):
| 场景 | TPOT 变化 | 吞吐变化 | 判定 |
|---|---|---|---|
| 128x128_c8 | **−34~−35.3%** | +45.9% | **≥30% 达标** |
| 128x512_c8 | −34.4% | +49.5% | 达标 |
| 8192x128_c8 | −12% | 改善 | 不劣化 ✓(改善) |
| 16384x128_c8 | −5.9% | 改善 | 不劣化 ✓(改善) |
| 128x128_c1 | 9.42 vs 8.766(+7.5%) | 落后 | **不达,证据链判不可达** |

c1 全域攻击面终态(每域均穷尽取证):
| 域 | 终态判决 | 证据 |
|---|---|---|
| dense b1 GEMV | H79 兑现 ~90μs;其余族 NN 化证否;后端不可解三重互证 | E321/E324/E143/E144/E255 |
| MoE | H28 兑现(−2~2.4%);域上限 8.6%,19 尝试归档 | E313/D81/§13 |
| attention decode | token-split closed(编译病态)/页 split 净负/head-split 机理否决/双缓冲+流水 disproven | E295/E296/E325-1 |
| attention 胶水 | 0.083ms@4.93TB/s 近带宽顶 | E291 |
| 拷贝链(图模式) | 共享面口径红线+独有面 0.3% 不可达 | E327(本轮) |
| GDN | decode E20 封死;prefill H10 TEMPORARY;vendor AscendC 域 | E20/E134-2 |
| host prep | H65/H70/H71 disproven;period=设备链,host 全遮盖 | E265/E280-E286 |
| sync/FG event | replay 排空=设备必要时间;FG event 四场景零增益 | E257/E261/E263 |
| 图内气泡 | 0.42ms 不可分解 | E276 |
| prefill(W1) | D57-① shunt 兑现 0.754x/0.758x;字面重写 <0.2% 上限;残值=栈升级外部依赖 | E300/D147/D160 |
| capture 桶 | capture8 已入役(c8 −22% 域) | E251/E258 |
| 采样 | greedy 权威口径;topk opt-in 保留 | E138-5 |

复调研台账:vllm/vllm-ascend/sglang/flashinfer 四仓 Atom 增量连续 70+ 轮零可映射
(最新 E323/E325,SINCE 进位 vllm 22:06:55Z/flashinfer 22:03:21Z/sglang 22:03:47Z/
vllm-ascend 13:28:14Z),相邻领域(cannbench 算子库/CANN 869 opapi 枚举/编译器 pass/
调度策略/图优化/L2 hint/预取)均有专项取证档(E131/E132/E141/E143/E100/E33/H37/H39/H43)。

**结论**:c8 分量 ≥30% 达标且全场景不劣化;c1 分量要求 TI 设备链比 vendor 原生路径
快 35%+(TI 9.368 vs FIA 8.512ms),而全部 12 个攻击域的终态证据均指向「与厂商算子
同水平或消除路径不可达」,已兑现优化(capture8/H28/H61/H73/H79/split-KV/SIMD/shunt)
全部入役后 c1 仍 +7.5%。按任务书终止条件 b),系统级专项以 FINAL TEMPORARY 收口,
交用户裁决。

## H82 / L-A KV FP8 存储量化:CPU 数值域 oracle 判停(iter-238,E341)

用户裁决解禁 KV FP8(2026-10-21)后,按精度 oracle 纪律先做数值域门槛验证
(fp32 sdpa oracle,TOL=3e-2 相对,per-(layer,head) amax 校准,E4M3 round-trip;
GQA 16Q/2KV、head_dim 256、decode q_len=1,均匀+相关双分布,存证
`tests/test_kv_fp8_oracle.py`):

| 场景 | rel_F | per-query max | 判 |
|---|---|---|---|
| S=8192 相关(尖锐 attention) | 2.616e-2 | 3.293e-2 | 贴线 PASS |
| S=16384 相关 | 2.661e-2 | 3.651e-2 | 贴线 PASS |
| S=8192 均匀 | **3.800e-2** | 4.239e-2 | **超差** |
| S=16384 均匀 | **4.053e-2** | 4.620e-2 | **超差** |
| 量化 round-trip 基础误差 | 2.642e-2 | — | — |

**判决:per-head amax 校准 E4M3 的精度乐观上界(上帝视角校准)在均匀分布域超差、
相关域贴线无裕度;按预登记「任何场景超差即停、禁止降 TOL」红线,L-A 首方案
disproven 关闭**,设备 oracle 未做(乐观上界已超,无信息量)。
翻案路径记档(未立项,待用户裁决或新假设轮):①scale 粒度 per-token/per-block
(KV 行级 fp32 scale,存储 +0.4%,读取减半保留,vLLM/flashinfer fp8-KV 生产实现
同族);②dequant-in-kernel(E334 bf044ba 同思想)仅在①之后有意义。

## H84 / L-A 终局判停:per-block MX 粒度同样超差,scale 粒度细化被证死(iter-243,E353/D174)

E341 翻案路径①(per-block scale)经 H84 立项验证(MX 标准:E4M3+per-32 连续元素
E8M0 2 的幂 scale,存证 `tests/test_kv_fp8_mx_oracle.py`;实现通路已备=E352
tilelang 0.1.15 昇腾后端原生 blockscaled MAD `ascend_mad_mx`,dequant-in-MAD 零额外内核):

| 场景 | MX per-block rel_F | per-head(H82 对照) | 判 |
|---|---|---|---|
| S=8192 均匀 | **3.843e-2** | 3.800e-2 | **超差** |
| S=16384 均匀 | **3.867e-2** | 4.053e-2 | **超差** |
| S=8192 相关 | 2.784e-2 | 2.616e-2 | 贴线 PASS |
| S=16384 相关 | 2.667e-2 | 2.661e-2 | 贴线 PASS |
| raw round-trip(mx vs per-head,同 seed) | 2.653e-2 | 2.642e-2 | **零增益** |

**判决:按 H84 预登记红线,L-A 终局判停**。决定性证据=同输入下 per-32 粒度相对
per-head 单 scale raw round-trip 零改进(±5% 口径)——i.i.d. 分布下每 32 元素块
amax 与整头 amax 同分布,scale 粒度不改变 E4M3 三位尾数的舍入噪声地板
(gaussian ~2.4e-2/uniform 端到端 ~3.8e-2);任何更细粒度(per-token/per-16)同理无效。

**终局结论:「KV FP8(E4M3 族,任意 scale 粒度)在 TOL 3e-2 下对均匀域不可达」,
交用户裁决**(选项:a) 降 TOL 需明示口径变更;b) K/V 混合精度分置(误差主导度
分析先行);c) 放弃 KV 量化轴,L-B 稀疏选择为「读得少」唯一在途路径)。
存证语义=expect_pass 守卫(uniform 反向断言),未来若回落 TOL 内测试即失败=重开信号;
两轮穷尽(H82 iter-238+H84 iter-242/243)满足循环终止条件 b) 的 L-A 单线口径。

## H83-① prefill PV 通路单 pass 化:dim=256 编译期判死——L0B 物理容量为硬约束,D2 split 是唯一可行形态(iter-244,E356/D176)

H83 立项账(iter-240)判 PV 单 pass「恰满可行」;设备编译冒烟直接证伪:

```
Z3 loop scheduling failed: scope "shared.l0b" capacity 65536 is insufficient
for minimum buffer usage 98304
```

| 量 | 现役 D2 split | 单 pass(H83-①) |
|---|---|---|
| L0B = qk_b + pv_b | 32KB + 32KB = **64KB 恰满** | 32KB + 64KB = **96KB > 64KB 拒** |
| L0C = qk_acc + pv_acc | 64KB + 64KB = 128KB 恰满 | 64KB + 128KB = **192KB > 128KB 拒** |

机理:立项账把 L0B 当 pv_b 独占(64KB 恰满),漏算同作用域 qk_b 并存——Z3 调度
按最小并存合计且不做跨 buffer 生命周期合并,单 pass 在 dim=256 数学不可行。
**现役 D2 split 正是 L0B/L0C 双恰满约束下的唯一可行形态**(发射税已是被约束
最小化的形态,H83 预登记红线「编译失败→disproven,H80 退回仅 NZ K/V 布局轴」兑现)。

处置:pv_single_pass 编译期分支保留(默认 False 零影响,存证基座);契约测试
`tests/test_prefill_pv1pass_contract.py` 9/9 绿,smoke[256]/numeric 转
pytest.raises 守卫(未来容量扩展/调度器支持生命周期合并使编译通过=测试失败=重开
信号)。低优先翻案路径记档(不烧窗):gemm C 切片复用(L0C)+L0B 手工分段复用,
依赖编译器对切片 gemm accum 的支持性取证。

## H83-③ NZ K/V 布局轴:分析性 disproven——E88「nd2nz=MTE 固有语义」判据同构覆盖,布局转换税消除维度全域关闭(iter-245,E360/E361/D177)

E356 曾判 NZ 轴「不受影响,退回 H80 继续」;本轮以 MoE preNZ 终裁(E88)证据链
收口修正:

1. **判据同构**:现役 K/V 通路(`paged_decode_ascend950.py:1223` K GM→L1、
   `:1324` V transpose GM→L1)与 MoE 权重加载同为 tilelang T.copy GM→L1 表达、
   同一 codegen 发射层、同一 MTE 拷贝引擎。E88 决定性判据=「nd2nz 为 MTE 固有
   语义、主机预打包不可绕过(恒等 pack 同量级错为设备实证)」⇒ K/V cache 预排
   NZ 存储后 T.copy 仍恒发 nd2nz → 二次重排 → 数值错乱,与内核特异性质无关。
2. **免转换原语已三次设备证死**:raw `asc_copy_gm2l1_align`(唯一豁免路径)
   物理序=转置连续而非 compact-NZ(E76)+多段发射竞态(E83/E88),H26/H27/H30
   全灭;attention 域无任何特异豁免理由。
3. **判决**:分析性 disproven,不烧设备窗重复实验(工具链层结论,设备重复无
   信息量;H30 先例=同一堵墙撞三次才死透)。翻案条件=CANN 提供受控免转换直载
   原语或 tilelang 升级使 align 多段发射受控(D73 平内存 pack 与
   `kv_nz_layout.py` 资产届时可复用)。
4. **影响(asc_copy 全家族边界,E361 汇编)**:GM→L1(nd2nz 固有/align 未受控)、
   UB→L1(H22/E66 死)、L1→UB(codegen 无原语,E74)三通路均关;可信优化面
   收敛为「少搬运」(dim=256 域已由 E356 判死)与「读得少」(H81 稀疏选择,唯一
   在途)两轴。
5. **P-B 重写线收窄**:两个已知可动面(NZ 布局+D2 split 发射减半)双死 ⇒ 当前
   栈内无可实现面清单;命运由 H58 归因(A/B 域 blocked)定谳内核份额后裁决——
   份额不足以支撑 ≥1.5x → 重写线关闭交用户 FINAL TEMPORARY;份额足 → 须登记
   新可动面假设(当前未知)。

## L-D 采样封存实验恢复评估:终局判决(D168 不满足)——引擎内激活已证,稳态收益 ~1.9%<5% 门槛,保留 opt-in 不进默认(iter-248,E369/E365-E366+ld9a/ld9b)

评估链闭环(安装面→激活→A/B→判决):

1. **安装面根因与修复(E369,addf026)**:三计数全 0 的终解=守卫双必杀——
   ①`dtype!=fp32`(引擎 logits=bf16 恒拒);②`non_argmax_invariant 非空`
   (vLLM 初始化即注册 LogitBias/MinTokens,列表恒非空)。修复=bf16 接受
   (wrapper cast fp32)+守卫改前置实际应用处理器(`apply_logits_processors`,
   语义与原生 forward 同序);no_penalties/bad_words/allowed_mask 守卫保留。
   采样域 42 用例绿。
2. **激活实锤**:ld9b on 臂 fused_sample_calls=127(=t1k50 全部 128 步−1),
   skipped=fallbacks=0;greedy_c8(batch 8<MIN_BATCH 守卫)零触发符合设计。
3. **权威 A/B(单变量=OPS sampling,off 基线=v8 收割同夜数据)**:

| 场景 | off | on(ld9b) | Δ | 判据 |
|---|---|---|---|---|
| t1k50_c32 ITL p50 | 29.18 | 28.63 | **−1.88%** | <5% 不满足 |
| t1k50_c32 TPOT p50(per-req) | 29.22 | 29.48 | +0.88% | <5% 不满足 |
| greedy_c8 TPOT p50(守卫) | 11.61 | 11.65 | +0.37% | ≤3% ✓ |

   ld9a(首跑)ITL −1.69% 佐证一致;其样本含**一次性 ~3s fused kernel JIT
   编译停顿**(TTFT +30%/std_itl 272ms/throughput −42% 同源,不随磁盘缓存
   消除=每 serve 进程首次激活成本)。
4. **归因**:内核级 1.54x(E344,4.43→2.88ms@b32)不传递到引擎级的原因=
   wrapper host 面( cast+处理器应用+参数提取)+原生 sort+cdf 路径与其他
   工作重叠而融合核串行;净稳态收益上界 ~1.9%。
5. **判决与建议(交用户)**:D168 主判据不满足→L-D 收口;fused_sample
   **保留为 opt-in 资产**(OPS sampling 非默认,默认路径零影响,无回归);
   价值面=top-k/top-p 真实采样批(batch≥20)~1.9% ITL;废弃不建议(资产
   零成本);不进默认路径维持。原始 JSON:bench_results/ld9a_*/ld9b_*。

## H58 prefill 归因定谳与 P-B/D57-② 重写线关闭(D178,D177 预登记兑现;iter-248,E367/E368)

1. **fia 侧双窗画像(E367,p_prof 修复后首采)**:prefill 窗(8192/4/C8)
   设备总 4096.5ms——**full-attn 6.5%**/MoE 35.6%/dense+胶水 29.0%/GDN 18.9%;
   decode 窗 full-attn 仅 0.9%/MoE 53.0%。Top 内核=GroupedMatmulV5 30.2%/
   MatMulV3 14.7%/ChunkGatedDeltaRuleFwdH 7.8%。
2. **重写天花板**:内核重写 1.5x ⇒ prefill 窗设备时间收益 ≤6.5%×(1−1/1.5)
   ≈**2.2%**;且 H61 在役默认 `TILEINFER_PREFILL_MAX_CTX=1` 下 TI prefill
   内核**不在役**(份额=0),重写对权威口径 TTFT 零作用面。
3. **判决**:P-B(微基准 ≥1.5x 现役 TI prefill)按 D177 预登记「份额不足→
   重写线关闭」执行;H80 终态 closed(两可动面 E356/E360 双死+本条份额)。
   资产封存不变(kv_nz_layout.py+pv_single_pass 存证分支;翻案条件=tilelang/
   CANN 栈升级或用户改在役口径)。任务书「8K/16K prefill 落后 1.53-1.71x」为
   iter-196 pre-shunt 旧账(H59/H61 shunt 后已修平,D57-① 0.754x/0.758x),
   现状以权威矩阵 A/B 复核为准(见矩阵节)。
4. **TI 侧 p_prof 记阻(E368)**:on 臂第四次失败根因=未 prewarm 的
   split-kernel 形态在 forward 内同步 tilelang JIT 编译(py-spy 实锤,卡
   7min+;E348「TI 侧从未导出」真因修正,profiler/图捕获非根因)。降优先,
   prewarm 形态列表补齐或 env 对齐权威口径后转 H62(b1 固定开销)输入再采。

## 权威矩阵 A/B 全量复核(iter-248,2026-10-22 02:2x–02:4x,同机同日成对,v9b 链 failbits=0)

口径:`vllm bench serve` random 数据集,temperature 0,num-prompts 32,percentile p50;
off=`TILEINFER_VLLM=0`(纯 FIA 原始昇腾路径),on=TI 当前最优配置(VLLM=1+OPS
decode,prefill,greedy+SIMD softmax+ktp 分带+MAX_CTX=1+fold+DENSE_NN+capture8)。
原始 JSON:`.agent_loop/run/bench_results/mtx_{off,on}_*.json`。

| 场景 | TTFT p50 off→on | ΔTTFT | TPOT p50 off→on | ΔTPOT | 对照目标 |
|---|---|---|---|---|---|
| 128x128_c1 | 100.4→92.4 | **−7.98%**(TI 快) | 8.77→9.42 | +7.40% | c1 已知弱点(域内 +7.5% 一致,判不可达档) |
| 128x128_c8 | 205.3→185.5 | **−9.62%**(TI 快) | 16.69→11.62 | **−30.38%** | ≥30% ✓ |
| 128x512_c8 | 200.1→185.7 | **−7.21%**(TI 快) | 17.02→11.66 | **−31.53%** | ≥30% ✓ |
| 8192x128_c8 | 1052.1→1070.6 | **+1.76%** | 39.56→36.46 | −7.85% | 目标② TTFT ≤5% ✓;目标① TPOT 未达 |
| 16384x128_c8 | 1839.8→1980.7 | +7.66% | 68.29→65.35 | −4.31% | 目标② ✗(>5%);目标① 未达 |

结论:
1. **短场景 TTFT 全部 TI 显著更快**(−7~−9.6%,优向超 |Δ|≤3% 带,非劣化);
   c8 档 TPOT −30.4%/−31.5% 保持 ≥30%(对今日新鲜 FIA 基线;基线较 docs 旧账
   18.01→16.69 漂移 −7%,TI 11.62 与在役档案 11.65 一致)。
2. **目标②(8K/16K 纯 prefill TTFT ≤5%)**:8192 PASS(+1.76%);16384 未达
   (+7.66%,+141ms)——shunt 全分流下双侧同为 FIA 内核,差值=TI host 胶水面
   (chunked prefill 每 chunk 集成开销);短场景无此面暴露。残值面=P-C 再校准
   或胶水减薄,与 E327 独有面 0.3% 口径需区分(16K 域未单独画像,记 NEXT)。
3. **目标①(8K/16K×C8 TPOT ≥30%)**:−7.85%/−4.31%,远未达;与任务书
   「当前 +12%/+5.9%」同量级(基线漂移内)。唯一在途改善路径=L-B 稀疏选择
   (H81/H85,dump 判据待真实 trace);L-A 已终局判停(D174)。
4. 大账(今日口径):c8 −30.4%/−31.5% ✓;c1 +7.4%(不可达档,E327 证据链);
   8K −7.9%/16K −4.3%(长上下文档)。

## L-B 页级 top-k 稀疏选择:真实 trace 判据 LB_DISPROVEN——centroid 选择器对真实分布无判别力,「读得少」维度全域关闭(iter-249,E373/D180)

装备:dump hook(`TILEINFER_LB_DUMP` env 门控,`tileinfer/integrations/vllm_ascend.py` 模块级
`_lb_dump_collect/_lb_dump_record/_lb_dump_flush` + `forward_impl` DecodeOnly 分支接入;单请求
decode 步逐 full-attn 层累积,一步一文件,E364 格式;CPU 契约 7 用例 `tests/test_lb_dump_hook.py`)。
采集:dump 臂 = 权威 ti 臂 env + `--enforce-eager`(eager 下每步 Python 执行;plan miss =
后台编译 + 本步 FIA,无 E368 型 capture 窗卡死面);单请求 8192/8 与 16384/8,6 文件全收
(8K ctx8193/8194 双步,16K ctx16385-16388 四步)。

判据(HYPOTHESES H81/H85 预登记门槛;sparse_select_quality.py --load,三 dump 一致):

| 判据 | 8K | 16K | 门槛 | 判决 |
|---|---|---|---|---|
| H81 top-50% 页 min-cover(全层×全 head) | 0.23% / 0.21% | 0.50% | ≥99% | **LB_DISPROVEN** |
| H85 共享选择器最小可行预算 f* | None | None | ≤0.5 可行 | **SHARE_DISPROVEN** |
| top-10% 页 softmax 质量占比 | 38.3% | 49.0% | —(均匀=10%) | 分布集中度不足 |
| 最大单页质量份额 | ~19% | ~21% | —(均匀 1.5%/0.8%) | 有集中但远不足 |

结论:
1. **H81/H85 双关闭**:centroid(q·K 均值)打分与真实注意力分布几乎无关——个别 Q-head
   top-50% 页覆盖低至 0.2%(选择反向);cover_mean 也仅 0.55-0.70。要覆盖 99% 质量需
   ~95%+ 页,「读减半」前提不存在。
2. **DeepInfer 索引器训练件必要性反向确认**(深读线收口):E362 定位「centroid=训练-free
   等价物」,本轮实证该等价物在真实分布上不成立——DeepInfer DeepSelect 的收益来源是训练
   判别力而非选择结构,不可映射定谳。深读七轮(E330/E335/E339/E345/E357/E362/E371)
   累计登记 H80/H81/H83/H85,至此四假设全部终态关闭。
3. **「读得少」维度全域关闭**:L-A KV FP8(per-head/per-block MX 双粒度判停,E341/E353)
   + L-B(本轮)⇒ 长上下文 TPOT 在当前栈内的可达值 = 在役 split-KV+SIMD softmax 配置
   (权威矩阵 c8 档 −30.4%/−31.5%);进一步改善需新维度(训练件索引器/硬件 KV 压缩),
   超出本轮授权口径。
4. 口径声明:判据以 random-input trace 出数;若生产负载分布显著更尖锐,重开需带生产
   trace 重验(dump 装备保留,复验成本=一次 dump 臂 serve+一次 --load)。
5. dump 资产:6 个真实 trace(run/lb_dump/,gitignored);oracle 两处修正(非页对齐 S 尾页
   pad;多 real 层 verdict 取全层最小);E337 教训再证(dump 臂首跑漏 source env.sh 即
   E30009,补挂即通)。

## 16K 域 TTFT 残值归因:差值全在 host 面——SyncStream 101 次 vs 1 决定性,H86 立项(iter-249,E374/E375)

16K prefill 窗(16384/4/C8×8 条)双臂 p_prof 对照(on=TI 权威口径 env+enforce-eager /
off=FIA),prof_summary_{ti16k,fia16k}.md:

| 口径 | on(ti16k) | off(fia16k) | Δ |
|---|---|---|---|
| 设备总耗时 | 8459.9 ms(171391 条) | 8369.5 ms(170252 条) | **+1.08%(噪声内)** |
| 设备 full-attn | 10.7%(904.9 ms) | 10.8%(905.6 ms) | ≈0 |
| host API 总 | 6524.2 ms | 6225.7 ms | **+298.5 ms** |
| aclrtSynchronizeStream | 161.6 ms × **101** | 0 ms × 1 | **+161.6 ms(决定性)** |
| aclrtMemcpyAsync | 1931.1 ms × 3352 | 1767.1 ms × 3283 | +164.0 ms |
| 发射类(launch/LaunchKernel) | 1966.1 ms | 1673.3 ms | +92.8 ms |
| 胶水小算子(Copy/Cat/Fill/Zero) | 683.6 ms | 606.3 ms | +77.3 ms |

结论:16K TTFT +141ms(D179)不在设备(同 FIA 内核),在 TI 集成层 host 面——每步 2-3 次
流同步(prefill 判据 D2H:vllm_ascend.py L603 分流判据 cat().cpu().tolist() / L695 页数 /
L718 last-page-len)及其引发的 MemcpyAsync 排队。**H86 立项**:MAX_CTX=1 全分流在役口径下
分流判据恒真可短路(零设备信息需求),消除同步后预期 16384 TTFT ≤5%(8192 +1.76% 不回退)。
full-attn 16K 份额 10.7%(较 8K 窗 6.5% 升,序列平方域符合预期)——若未来重启 attention
域工作,16K 域份额输入以此为准。

## H86 判决:同步消除落地但 16384 p50 不动——p50 差值实为波次边界双稳统计,全分布+吞吐口径 TI 无劣势(iter-250,E376/D181)

H86 实施(默认开,语义严格等价的判据前移):`0 < TILEINFER_PREFILL_MAX_CTX < 128`(在役
默认 =1)下,任意非空 prefill 步页数上界 ≥1 页 → `max(pages)·128 > MAX_CTX` 恒真,分流判决
不需要设备信息——`prefill_step` 在 D2H 之前直接 refuse(同一计数器/skip 守卫/refuse 语义,
vllm_ascend.py `ctx shunt (all)` 分支)。CPU 契约 4 用例(零 D2H 断言/二Layer 守卫/4096 与 0
档不短路)。h86 权威矩阵 A/B(off/on 成对,failbits=0 全 32/32):

| 场景 | p50 TTFT off→on | Δp50 | Δmean | Δp90 | Δp99 | TPOT Δp50 | 判据 |
|---|---|---|---|---|---|---|---|
| 128x128_c1 | 93.7→97.3 | +3.78% | +5.13% | −1.48% | +35.4% | +7.65% | 短场景(噪声带内,历轮 −7.98%~+3.78%) |
| 128x128_c8 | 193.6→195.8 | +1.12% | +1.95% | +7.30% | +7.44% | **−30.09%** | PASS |
| 128x512_c8 | 189.7→193.4 | +1.94% | +2.01% | −0.26% | +1.12% | **−31.43%** | PASS |
| 8192x128_c8 | 1062.7→1070.7 | **+0.76%** | +2.46% | +1.01% | +0.76% | −6.87% | TTFT PASS(≤5%) |
| 16384x128_c8 | 1848.2→1982.7 | **+7.28%** | **+0.34%** | **−3.94%** | **−2.13%** | −3.95% | p50 未达;mean/p90/p99 全 ≤5% 且 p90/p99 反超 |

在役探针:prefill_long_skips=431(短路计数生效)/prefill_fallbacks=4400(≈10 层×步)/
fallbacks=0。**与 iter-248 矩阵逐行对照:零回归**(c8 TPOT −30.09%/−31.43% 保持,8K/16K
TTFT/TPOT 全在历轮抖动带内)。

判决(H86 DISPROVEN,按预登记门槛):
1. 同步消除**已生效但 16384 p50 差值不动**(+141ms→+134.5ms):E375 的「串行暴露×C8 排队」
   模型不成立——host 线程有裕量(设备队列深度>0),host API 时间差不转化为请求时延。
2. **p50 差值定性改写:波次边界双稳统计**。16384/C8/32 prompts = 4 波×8 并发;p50 恰落在
   波1末/波2首的边界上(≈总 prefill 时长 1848ms 处),TI decode 快 30% → 波次推进更快,
   中位请求落入波2(≈+一个 prefill 波的排队),而波3/波4 请求(p90/p99)在 TI 臂**更早**
   获得首 token(−3.94%/−2.13%);mean +0.34%、吞吐 +2.63%(req/s)同向。p50 是该场景下
   最不稳定 的统计量(两轮 +7.66%/+7.28% 复现的是波次结构而非性能差)。
3. 处置:去同步代码**保留在役**(零成本严格改进,全部既有门槛保持);目标②的 16384 p50
   残差不再归因 TI 内核/胶水效率。下轮登记单波判决实验(H87 方向:C=8×num_prompts=8 单波
   16384,p50 应与 off 打平;以及 num-prompts=16 整双波对照)以定谳波次边界机制。

## H87 单波判决(iter-251,2026-10-22):16384 p50 残差=波次边界统计伪影定谳,E380/D182

实验(h87 链,off/on 成对,failbits=0,环境逐字继承权威口径):16384×128×C8 以
num_prompts ∈ {32(4 波参照),16(2 波),8(单波判决行)}+8192×128×C8×p8(旁证)双臂消波对照:

| 场景 | 波次 | TTFT p50 off→on | Δp50 | Δp90 | Δp99 | Δmean | Δ吞吐 | 判定 |
|---|---|---|---|---|---|---|---|---|
| 16384x128_c8_p32 | 4 波 | 1833.9→1867.8 | +1.85% | +2.78% | +2.08% | +4.00% | +1.97% | iter-250 +7.28% 当日不复现(双稳) |
| 16384x128_c8_p16 | 2 波 | 1855.2→1859.7 | +0.24% | +0.54% | +0.49% | +0.24% | +2.72% | 残差未复现 |
| **16384x128_c8_p8** | **1 波** | **5136.8→5213.1** | **+1.49%** | +0.81% | +0.79% | +1.17% | +2.67% | **主判据 PASS(≤5%)** |
| 8192x128_c8_p8 | 1 波 | 2576.5→2599.4 | +0.89% | +0.99% | +1.06% | +0.98% | +7.85% | 8K PASS 域旁证 |

判决(H87 proven,主判据):单波消波后 16384 TTFT 全分布差距收敛至设备噪声量级
(+1.49%,对照 E375 设备侧 Δ+1.1%)⇒ 权威口径 p50 残差是**波次边界上的双稳统计伪影**
(32 prompts=4 波,p50 恰落波1末/波2首,中位请求归属随 run 波动:iter-248/250 +7.66%/+7.28%
vs 本轮 p32 +1.85%);伴随预测(p16/p32 按波数稳定重现残差)证伪——残差幅度不随波数稳定,
进一步排除固定差值源。目标②线穷尽取证四件套齐(E375 设备 Δ+1.1%/E376 去同步不动/E380
消波 +1.49%+全分布持续不落后):当前栈内无系统性 TTFT 差值源可攻击。在役配置不动
(MAX_CTX=1 维持,P-C 无收窄条件);双专项整体 FINAL TEMPORARY 交用户
(PERFORMANCE_REPORT §17)。

## H88 / L-A2 per-token scale 翻案判停:粒度不变性 token 轴再证,L-A 线永久关闭(iter-252,E381/D183)

用户裁决 D147「未实现的都接着实现」复活 L-A 最后一档翻案路径:per-(head, token)
行级 amax 校准(token 间动态范围解耦,存储开销 +0.4%),同 oracle 门槛
(TOL 3e-2 相对,fp32 sdpa,uniform+correlated 双域,S∈{8192,16384}):
存证 `tests/test_kv_fp8_oracle_per_token.py`(5 passed):

| 场景 | per-token rel_F | per-head(H82 对照) | per-block MX(H84 对照) | 判 |
|---|---|---|---|---|
| S=8192 均匀 | **3.816e-2** | 3.800e-2 | 3.843e-2 | **超差** |
| S=16384 均匀 | **3.907e-2** | 4.053e-2 | 3.867e-2 | **超差** |
| S=8192 相关 | 2.609e-2 | 2.616e-2 | 2.784e-2 | PASS |
| S=16384 相关 | 2.579e-2 | 2.661e-2 | 2.667e-2 | PASS |
| raw round-trip | 2.611e-2 | 2.642e-2 | 2.653e-2 | 同地板 |

**判决:按任务书 L-A2 预登记红线「超差即永久关闭,不再留翻案」→ L-A 线
(KV FP8 一切 scale 粒度变体)永久关闭**。三粒度(per-head/per-block MX/per-token)
数值逐项 Δ<5% 互证:E353「i.i.d. 粒度不变性」机理在 token 轴精确兑现——
每 token 行 amax 与整头 amax 同分布,scale 粒度不改变 E4M3 三位尾数的舍入噪声
地板(round-trip 2.61e-2 三粒度同水平);correlated 域异常 token 的 per-token
归一改善不落在误差主项(softmax 输出误差由全体 token 平均量化噪声决定)。
「读得少」维度至此四重判死(H82/H84/H88+L-B E373)。

## H89 / L-B2 分层 KV(池化余页):真实 trace oracle 判死——Jensen/页内方差地板超差 3.7-5.4x,达标记账需 f≥0.75(oracle 幻想上界),L-B2 全族关闭(iter-253,E384)

L-B2 重设计首选候选(任务书候选列首):未选页 128 个 K/V token 压缩为页均值
landmark(K̄,ṽ)+ 质量保持项 log(n_pg) 单虚拟 token,选中页保精确 token,混合
softmax;读取量 = f·S+(1−f)·S/128。判据(预登记,f=0.5 立项门):真实 decode
trace(权威 ti 臂 env+enforce-eager,random-input,E373 同一批 dump)逐 (层,Q-head)
相对 L2 误差 max ≤ TOL 3e-2,选择器双口径 = oracle-mass(真实页质量 top-f,数学上界)
+ centroid(E373 已证伪,对照)。存证 `benchmarks/probes/lb2_hier_kv_oracle.py` +
`tests/test_lb2_hier_kv_oracle.py`(6 passed;合成契约在真实判据前捕获 landmark
缺 log 计数项的真 bug——混合 softmax 中未选页被系统性低估 128 倍,f=0 全池化统一
抵消而混合 f 错误重分配,修正后误差随 f 单调下降):

| 预算 f(oracle-mass 上界) | 8K rel_max(10 层) | 16K rel_max(10 层) | 读取比 | 判 |
|---|---|---|---|---|
| 0(纯池化) | 0.177-0.883 | 0.272-1.654 | 0.0078 | 页内方差/Jensen 地板实锤 |
| 0.25 | 0.343 | 0.097* | 0.256 | 超差 |
| 0.5(**立项门**) | **0.163** | **0.110** | 0.504 | **超差 5.4x/3.7x** |
| 0.75 | 0.0269(贴线) | 0.0172 | 0.752 | 达标但幻想上界 |
| 0.875 | 0.0122 | 0.0074 | 0.876 | 无经济性 |
| centroid 对照 @f0.5 | 0.647 | 0.229 | 0.504 | 与 E373 反向判别一致 |

*16K f0.25 行为 JSON 记录代表值。**判决:f=0.5 预登记门双 dump 超差 ⇒ H89
disproven,L-B2 分层变体关闭**。关闭论证三点:①误差主体=池化近似的二阶
Jensen 间隙 × 页内注意力权重方差(f=0 行证明页内 token 级结构显著),与选择器
质量解耦——oracle-mass 是选择器侧数学上界,上界超差 ⇒ 异常值强制保留/重要性
累积/按头差异化预算等一切选择器/预算变体同族封死(它们只能改变「哪些页池化」,
不改变「池化本身的地板」);②TOL 达标需 f≥0.75 且为 oracle 幻想口径,可实现
选择器(centroid 实测 0.65)所需 f* 更高 ⇒ 读取节省 ≤25%(幻想)/≤10%(现实),
物理上不可能承载 ≥30% TPOT 目标(即便 KV 读取占 TPOT 100%);③与 L-B(丢弃型,
E373)独立双死,「读得少」维度五重判死(L-A 三粒度+L-B 丢弃+L-B2 池化)。
翻案条件=生产 trace 示出页内权重尖锐化(随机输入口径声明继承 E373)。

## H32 gdnproj 权威 A/B 终局判决:disproven——strided 零拷贝消费路径劣化(iter-255,2026-10-22,E390/H90)

lc 链(iter-252 武装,iter-255 收割)H32 棒双臂:off=在役最优基座,on=+gdnproj,
5 场景×p32,图模式+在役探针。off 臂 5/5 场景 32/32;on 臂 4/5(16384 见附记)。

| 场景 | TPOT p50 Δ | TTFT p50 Δ | THRU Δ |
|---|---|---|---|
| 128x128_c1 | **+16.98%**(9.412→11.011ms) | −8.10% | −12.91% |
| 128x128_c8 | **+12.74%** | −15.59% | −8.27% |
| 128x512_c8 | **+13.45%** | −10.24% | −11.07% |
| 8192x128_c8 | **+4.61%** | +1.37% | −3.67% |
| 16384x128_c8 | 缺失(on 臂请求流中断) | — | — |

- 接管成立:on 探针 gdnproj_calls=30、fallbacks=0(off=0),判决基于真实效应。
- 预登记判据「任一场景劣化 >5% 即 disproven」兑现:3 场景 >12%,**H32 disproven**。
- 归因:off 基线与历史权威口径吻合(c1 9.41ms);单变量脚本+启动形态对称;劣化幅值
  全场景常数 +1.3~1.6ms/步(8K KV 主导域也 +4.6%)→ 非注意力域,每层 ~50µs 固定
  开销;TTFT 反向改善证明融合 GEMM 本体方向正确(E100:27.1 vs 36.0µs),劣化锁定
  **strided 零拷贝切片喂 fix_query_key_value_ordering 的下游消费路径**(微基准覆盖面
  不足的教训:只测 GEMM 本体未测 strided 消费)。
- 处置:opt-in `TILEINFER_OPS=...,gdnproj` 维持不默认(回退面在役,翻转 env+重启即回退,
  E389 审查确认);翻案唯一通道=H90 物化变体(融合输出连续 buffer,放弃零拷贝),走
  oracle→微基准→A/B 全链,若仍劣化则 gdnproj 全族(含物化)永久关闭。
- 附记:on 臂 16384 场景客户端 32 连接 7s 内全断(TransferEncodingError),服务端引擎
  同期正常推进 prefill 无崩溃;on 臂独有+长场景,归因记档不阻塞判决。

## L-C 收割:H34 判死 + H36/H37 on 臂零激活定谳(iter-259,2026-10-22,E402/D185)

### H34(load 期行重排)closed——disproven at engine gate
- 数据:lc 链补跑(STICKS=h34,off 臂产物复用跳过)failbits=0 五场景 completed=32;
  TPOT Δ(c1/c8/512/8k/16k)=+0.54/−0.69/−0.71/+0.03/+0.30%,全噪声带。
- 探针:gdnreorder_fallbacks=30/30、gdnproj_calls=30(未归零)→ reorder 每次 forward
  均回退(`build_reordered_inproj_weight` 引擎内懒构建失败→`_tileinfer_reordered_inproj_failed`
  缓存→永久 decline),on 臂从未激活。
- 处置(D185):不修守卫不重挂——H34 仅为 H32 翻案通道,同目标 H90 已完整武装
  (契约 9 passed+微基准/链就绪),基座 H32 经济学已判死(E390),修复懒构建不改变
  fused GEMM 基线。翻案条件=H90 亦败且新证据指向行重排路径。

### H36(greedy 快路径)arm invalid——引擎内零激活,根因取证中
- 数据:双臂 failbits=0,TPOT Δ=−0.17/+0.17/+0.19/−0.22/−0.20% 全噪声带
  (=死钩点差异,无假设信息量);探针 h36_on 零 greedy 键(calls/fallbacks 双无)。
- 离线三重取证:install 链通(load_general_plugins 后 AscendSampler.forward=tileinfer
  包装)、机制通(离线直调 calls=1+输出正确)、引擎静态链通(0.23.0
  NPUModelRunner:333 sampler=AscendSampler→_sample:2626 self.sampler(...)→类属性
  forward,无实例遮蔽)。守卫逐条对照 0.23.0 SamplingMetadata 字段理论全过。
- 盲区与修复:守卫不满足路径零计数(静默 None,仅异常计 fallbacks)→ 死钩点/守卫
  拦截不可区分。已落地 greedy_fast_guard_skips 计数+_first_guard_fail 逐因命名
  (10 守卫)+warning_once;tests 11 用例全绿。

### H37(MoE 权重 prefetch)arm invalid——物化一次死亡
- 数据:双臂 failbits=0,TPOT Δ=−0.09~−0.89% 噪声(≪2% 门槛);探针
  moeprefetch_fallbacks=1+active=0=_materialize 首次异常即 _dead 永久失活。
- 疑根因:model_runner init set_weight_prefetch_method 与惰性替位时序冲突或升级后
  API 失配;已加物化失败首次 warning_once(异常类型+消息入 serve 日志)。
- 价值风险(E109 在录):950PR L2-bypass 可能架空预取假设,≥2% 门槛预期低。

### 推论与后续
- 权威基座 ops 的 greedy token 自始惰性(死钩点,h36 噪声为证),零行为影响,基座不改。
- 后续(等窗):单探针 serve(ops=decode,prefill,greedy,moeprefetch,PORT 8153,
  128x128_c1 单场景 ~2min)读计数定死钩点/守卫拦截→守卫可修即修→STICKS="h36 h37"
  重挂 on 臂(off 产物自动跳过)→H36/H37 终判。此二棒终判前 L-C 不算收割完毕。

### L-D DeepSelect 臂终态补记(iter-260,D186/E404)
- **定谳:上游内核不存在,臂关闭**。DeepInfer(同硬件参考实现)
  `npu_ops/deep_select.py` 源码实证:采样路径(`return_value=True`,即我方
  `topk.py` 唯一用法)官方直接 fallback `torch.topk`,注释明示「Sampler is
  not supported yet」——D167-3「等外部补齐库或语义」两通道均关闭:装库也无
  内核,与静默失效等价零收益;不装 deep_select 包(非官方源不可得+无收益)。
- 「torch.topk-on-logits 变体重开」分析性判死(不烧设备窗):权威口径全批
  greedy→apply_top_k_top_p 零暴露;单步替换收益必低于同域 fused_sample 全链
  引擎级 ~1.9%(已判 <5% 门槛)。
- 处置:`tileinfer/kernels/sampling/topk.py` 静默失效即终态(import 失败→
  交还原生,现行为正确,零维护成本);`_OPS_ALL` "sampling" 键与 fused_sample
  接线维持(H76 终态建议仍待用户裁决)。**L-D 两臂至此全部终态,L-D 线关闭。**
