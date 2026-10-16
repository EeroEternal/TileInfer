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
