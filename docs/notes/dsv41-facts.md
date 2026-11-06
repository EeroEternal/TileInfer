# DeepSeek-V4.1-Flash 架构事实核准（S0）

> 状态：S0 完成（2026-11-03）。本文是 S1–S5 的唯一事实基准：**所有形状、公式、量化语义都来自
> 官方只读实现 `/home/models/DeepSeek-V4.1-Flash/inference/`（`model.py` / `engram.py` /
> `kernel.py` / `vision.py` / `image_processor.py` / `convert.py` / `config.json`）与
> **checkpoint 头部实测**，不从记忆或论文推断。
>
> 可复现脚本：`tools/dsv41/`（见 §19）。验收证据：§16。
>
> 上游实现仅供参考、**不是部署目标**（C4）；本仓任何代码不 import `inference/`。

---

## 0. 取证与口径

| 项 | 值 | 来源 |
|---|---|---|
| 权重目录 | `/home/models/DeepSeek-V4.1-Flash` | `du -sh` = 476 GiB |
| 分片 | `model-00001-of-00048.safetensors` … `-00048`（含 `model.safetensors.index.json`） | 头部实测 |
| 张量总数 | **96,085** | 48 个 safetensors 头部 |
| 逻辑字节 | **510.3 GB（十进制）= 475.2 GiB** | 头部 `data_offsets` 求和 |
| dtype 分布 | `F8_E4M3` 357 · `I8` 47,232 · `F8_E8M0` 47,589 · `BF16` 520 · `F32` 387 | 同上 |
| `config.json`（HF 顶层） | `architectures=[DeepseekV41ForCausalLM]`、`model_type=deepseek_v41`、`text_config.model_type=deepseek_v41_text` | 模型根 |
| `inference/config.json` | `ModelArgs` 的字段全集（扁平命名），本文公式一律按它 | `inference/` |
| 官方运行时 | `torch>=2.10`、`tilelang==0.1.8`、`transformers`、`tokenizers`、`sympy`、`safetensors>=0.7` | `inference/requirements.txt` |
| 本机环境 | torch 2.10.0+cpu / torch_npu 2.10.0.post4 / vLLM 0.23.0 / tilelang 0.1.15 / CANN 25.7.RC1 / 单卡 Ascend950PR（131,072 MB HBM）/ 628 GB RAM / 16 核 | `npu-smi`、`pip list` |

顶层 `config.json` 与 `inference/config.json` 的字段是**一一映射**，命名差异必须记住：

| HF `text_config` | `inference` `ModelArgs` | 值 |
|---|---|---|
| `hidden_size` | `dim` | 5120 |
| `moe_intermediate_size` | `moe_inter_dim` | 2304 |
| `num_hidden_layers` | `n_layers` | 40 |
| `num_nextn_predict_layers` | `n_mtp_layers` | 3 |
| `num_attention_heads` | `n_heads` | 64 |
| `num_key_value_heads` | —（不参与，MLA 单 latent） | 1 |
| `qk_rope_head_dim` | `rope_head_dim` | 64 |
| `head_dim` | `head_dim` | 512 |
| `n_routed_experts` / `num_experts_per_tok` | `n_routed_experts` / `n_activated_experts` | 384 / 6 |
| `scoring_func` / `topk_method` | `score_func` | `sqrtsoftplus` / `noaux_tc` |
| `routed_scaling_factor` | `route_scale` | 1.5 |
| `rms_norm_eps` | `norm_eps` | 1e-20 |
| `sliding_window` | `window_size` | 128 |
| `kv_source_layer_ids` | `kv_source_layers` | `[2, 8, 14, 20]` |
| `index_source_layer_ids` | `index_source_layers` | `[2, 8, 14, 20, 24, 28, 32, 36]` |
| `candidate_source_layer_id` | `candidate_source_layer` | 20 |
| `compress_ratios`（43 项） | 同名 | `0,0, 2×18, 1×20, 0,0,0` |
| `engram_num_embeddings` | 同名 | `[384006168, 384016682]` |
| `rope_theta` / `rope_scaling.factor` / `original_max_position_embeddings` | `rope_theta` / `rope_factor` / `original_seq_len` | 10000 / 16 / 65536 |
| `compress_rope_theta` | 同名 | 160000 |
| `quantization_config.weight_block_size=[32,32]`、`scale_fmt=ue8m0`、`expert_dtype=fp4` | `dtype=fp8`/`expert_dtype=fp4` | — |
| `max_position_embeddings` | —（`max_seq_len` 是运行时参数） | 1048576 |

`compress_ratios` 完整值（下标 = 层号，40 主干 + 3 DSpark）：

```
layer  0  1 |  2  3 ... 19 | 20 21 ... 39 | 40 41 42 (mtp.0/1/2)
ratio  0  0 |  2  2 ...  2 |  1  1 ...  1 |  0  0  0
```

---

## 1. 权重分账：**96.4% 是专家 + engram**

实测（`tools/dsv41/inventory.py`，按名字归类，单位 GiB）：

| 组 | 张量数 | GiB | 占比 |
|---|---:|---:|---:|
| 路由专家（40 层 × 384） | 92,160 | 268.9 | 56.6% |
| engram 查表（第 1、14 层） | 12 | 189.1 | 39.8% |
| DSpark MTP（3 段，各含 128 专家） | 2,401 | 7.4 | 1.6% |
| attention 稠密（40 层） | 520 | 4.7 | 1.0% |
| token embedding / 输出 head | 2 | 2.5 | 0.5% |
| 共享专家（40 层） | 240 | 1.3 | 0.3% |
| vision tower（ViT 32 block） | 259 | 0.8 | 0.2% |
| MoE gate（40 层） | 120 | 0.1 | 0.0% |
| Hyper-Connection mix（40 层） | 240 | 0.1 | 0.0% |
| aligner（2 个 Linear） | 4 | 0.1 | 0.0% |
| indexer（8 层）/ compressor（4 层）/ 各 norm / 图像分隔符 | 123 | <0.05 | ~0 |
| **合计** | **96,085** | **475.2** | 100% |

推论（直接决定 S4 方案）：

* **常驻 HBM 的"非专家非 engram"集合只有约 10.0 GiB**（attention 4.7 + embed/head 2.5 +
  共享专家 1.3 + vision 0.8 + gate/hc/engram-wkv 等）。
* 专家 268.9 GiB + engram 189.1 GiB + MTP 专家 7.2 GiB = **465 GiB 必须在主机内存**（本机 628 GB
  RAM 放得下），HBM（128 GiB）只剩 ~110 GiB 可做专家热缓存 + KV + 激活。
* 单个路由专家 = **18.80 MB**（w1 5.90 + w2 5.90 + w3 5.90 + 三份 scale 1.10）。
  → **每 decode 一个 token 需搬 40 层 × 6 专家 × 18.8 MB = 4.51 GB**；若 HBM 专家缓存 110 GiB
  （约 5,850/15,360 槽 = 38% 命中），仍需 ~2.8 GB/token。**这是 TPOT 的物理下界**（见 §10）。
* 单个稠密层（含 gate/共享专家/hc/norm，不含路由专家）= 0.17 GB；每层合计 7.39 GB。

---

## 2. 张量形状全表

记 `D=5120`（dim）、`H=64`（头数）、`Dh=512`（head_dim）、`Dr=64`（rope）、`Dn=448`（nope）、
`Rq=1280`（q_lora）、`Ro=1024`（o_lora）、`G=8`（o_groups）、`E=384`（路由专家）、`Ei=2304`
（专家中间维）、`V=129280`（词表）。

### 2.1 主干逐层（40 层各一份）

| 张量 | dtype | shape | 说明 |
|---|---|---|---|
| `attn.wq_a.weight/.scale` | F8_E4M3 / F8_E8M0 | [1280, 5120] / [40, 160] | q LoRA 下行，block-32 |
| `attn.q_norm.weight` | BF16 | [1280] | RMSNorm(1280) |
| `attn.wq_b.weight/.scale` | F8 / F8_E8M0 | [32768, 1280] / [1024, 40] | 64×512=32768，MP 按输出维切 |
| `attn.wkv.weight/.scale` | F8 / F8_E8M0 | [512, 5120] / [16, 160] | KV latent（单头 512 维） |
| `attn.kv_norm.weight` | BF16 | [512] | RMSNorm(512) |
| `attn.wo_a.weight/.scale` | **BF16**（convert 反量化后） / — | [8192, 4096] | 分组输出：`[G*Ro, H*Dh/G]` |
| `attn.wo_b.weight/.scale` | F8 / F8_E8M0 | [5120, 8192] / [160, 256] | 行并行，输入 8192 |
| `attn.attn_sink` | F32 | [64] | 每头一个注意力汇（加在分母） |
| `ffn.gate.weight` | BF16 | [384, 5120] | 路由打分（fp32 计算） |
| `ffn.gate.bias` / `.bias_vl` | F32 | [384] / [384] | 仅影响**选择**，不影响权重 |
| `ffn.experts.{e}.w1.weight` | **I8 = fp4 打包** | [2304, 2560] | 逻辑 [2304, 5120]，K 维 2 值/字节 |
| `ffn.experts.{e}.w1.scale` | F8_E8M0 | [2304, 160] | 每 32 个 K 一个 scale（5120/32） |
| `ffn.experts.{e}.w2.weight/.scale` | I8 / F8_E8M0 | [5120, 1152] / [5120, 72] | 2304/2=1152、2304/32=72 |
| `ffn.experts.{e}.w3.*` | 同 w1 | | SwiGLU up 支 |
| `ffn.shared_experts.w1/w3.weight` | F8_E4M3 | [2304, 5120] | 共享专家是 **fp8** 不是 fp4 |
| `ffn.shared_experts.w2.weight` | F8_E4M3 | [5120, 2304] | |
| `ffn.shared_experts.w*.scale` | F8_E8M0 | [72, 160] / [160, 72] | |
| `attn_norm.weight` / `ffn_norm.weight` | BF16 | [5120] | eps=1e-20 |
| `hc_attn_fn` / `hc_ffn_fn` | F32 | [24, 20480] | `mix_hc=(2+4)*4`，`hc*dim=4*5120` |
| `hc_attn_base` / `hc_ffn_base` | F32 | [24] | |
| `hc_attn_scale` / `hc_ffn_scale` | F32 | [3] | pre/post/comb 三组 |

### 2.2 只在 source 层出现

| 张量 | 出现层 | dtype | shape |
|---|---|---|---|
| `attn.compressor.wkv.weight` | 2, 8, 14（ratio 2） | BF16（模型内升 fp32） | [512, 5120] |
| `attn.compressor.wgate.weight` | 2, 8, 14 | BF16（模型内升 fp32） | [512, 5120] |
| `attn.compressor.norm.weight` | 2, 8, 14, **20**（ratio 1） | BF16 | [512] |
| `attn.compressor.wkv.weight` | 20（ratio 1） | BF16（保持 bf16，无 gate） | [512, 5120] |
| `attn.indexer.wk.weight` | 2, 8, 14, 20（owns_k） | BF16 | [128, 512] |
| `attn.indexer.k_norm.weight` | 2, 8, 14, 20 | BF16 | [128] |
| `attn.indexer.wq_b.weight/.scale` | 2,8,14,20,24,28,32,36 | F8 / F8_E8M0 | [4096, 1280] / [128, 40] |
| `attn.indexer.weights_proj.weight` | 同上 8 层 | BF16 | [32, 5120] |
| `engram.embed.weight` | 1, 14 | F8_E4M3 | [384006168, 256] / [384016682, 256] |
| `engram.embed.scale` | 1, 14 | F8_E8M0 | [384006168, 8] / [384016682, 8] |
| `engram.wkv.weight/.scale` | 1, 14 | F8_E4M3 / F8_E8M0 | [25600, 6144] / [800, 192] |
| `engram.q_weight` / `k_weight` | 1, 14 | BF16 | [4, 5120] |

张量计数交叉验证（**HF 侧**计数；转换后 `wo_a.scale` 被 `convert.py` 反量化掉，故每层少 1 个）：
普通层 `attn` 13 个；第 2/8/14 层 21 个（+compressor 3 + indexer 5）；
第 20 层 20 个（+compressor 2 + indexer 5）；第 24/28/32/36 层 16 个（+indexer 5）。
每层 `ffn` = 2304（专家）+ 3（gate）+ 6（共享）= 2313。

### 2.3 非逐层

| 张量 | dtype | shape |
|---|---|---|
| `embed.weight` / `head.weight` | BF16 | [129280, 5120] / [129280, 5120]（head 在模型内是 **fp32**） |
| `norm.weight` | BF16 | [5120] |
| `image_start` / `image_end` / `image_newline` | BF16 | [5120] ×3 |
| `vision.patch_embed.proj.weight/.bias` | BF16 | [1024, 588] / [1024]（588 = 3×14×14） |
| `vision.blocks.{i}.norm1/norm2.weight` | BF16（模型内 fp32） | [1024] |
| `vision.blocks.{i}.attn.wqkv.weight/.bias` | BF16 | [3072, 1024] / [3072] |
| `vision.blocks.{i}.attn.wo.weight/.bias` | BF16 | [1024, 1024] / [1024] |
| `vision.blocks.{i}.mlp.w1.weight` | BF16 | [5632, 1024]（gate+up 打包，2×2816） |
| `vision.blocks.{i}.mlp.w2.weight` | BF16 | [1024, 2816] |
| `vision.norm.weight` | BF16 | [1024] |
| `aligner.w1.weight/.bias` | BF16 | [5120, 9216] / [5120]（9216 = 1024×3²） |
| `aligner.w2.weight/.bias` | BF16 | [5120, 5120] / [5120] |

### 2.4 DSpark（`mtp.0/1/2`，层号 40/41/42）

| 张量 | 出现 | dtype | shape |
|---|---|---|---|
| `mtp.{s}.attn.*`（wq_a/q_norm/wq_b/wkv/kv_norm/wo_a/wo_b/attn_sink） | 0,1,2 | 同主干 | 同主干（**无** compressor/indexer） |
| `mtp.{s}.ffn.experts.{e}.w*` | 0,1,2 | I8(fp4)/F8_E8M0 | 同主干，e ∈ [0,128) → **MTP 只有 128 专家** |
| `mtp.{s}.ffn.gate.weight/.bias/.bias_vl` | 0,1,2 | BF16/F32 | [128, 5120] / [128] |
| `mtp.{s}.ffn.shared_experts.*` | 0,1,2 | F8_E4M3 | 同主干 |
| `mtp.{s}.hc_*` / `attn_norm` / `ffn_norm` | 0,1,2 | 同主干 | 同主干 |
| `mtp.0.main_proj.weight/.scale` | 0 | F8 / F8_E8M0 | [5120, 15360] / [160, 480] |
| `mtp.0.main_norm.weight` | 0 | BF16 | [5120] |
| `mtp.2.norm.weight` | 2 | BF16 | [5120] |
| `mtp.2.markov_head.embed.weight` / `.head.weight` | 2 | BF16 | [129280, 256] / [129280, 256] |
| `mtp.2.confidence_head.proj.weight` | 2 | BF16（模型内 fp32） | [1, 5376]（5120+256） |

`mtp.*.embed.weight` / `mtp.*.head.weight` **不在 checkpoint**：`convert.py` 主动丢弃，运行时三个阶段
共用主干的 `embed` / `head`（`Transformer.__init__` 直接赋值）。

---

## 3. 数值约定（fp8 / fp4 / scale）

1. **稠密线性层 = FP8 block-32**：权重 `e4m3`，每 32×32 权重块一个 `e8m0`（power-of-2）scale；
   激活每 32 个 K 元素一个 `e8m0` scale。`scale_dtype = torch.float8_e8m0fnu`，`fp8_block_size=32`。
2. **专家 = MXFP4**：权重 `float4_e2m1fn_x2`（2 值/字节，K 维打包），每 32 个 K 一个 `e8m0` scale；
   激活仍是 fp8 block-32，kernel 内 fp4→fp8 后做 fp8×fp8 GEMM（`kernel.py:fp4_gemm_kernel`）。
   `fp4_gemm` 断言 `b_s.shape == (N, K//32)`，`a_s` 每 32 或 128 一个。
3. **checkpoint 里 expert 权重是 `I8`**：`convert.py` 用 `param.view(torch.float4_e2m1fn_x2)` 解释，
   不做数值变换（`--expert-dtype fp4`）。`--expert-dtype fp8` 时才走 `cast_e2m1fn_to_e4m3fn`
   （把 fp4 表值 ×2^k 展开成 e4m3，scale 变成 `2^(6-k)` 的 e8m0，逐块 lossless）。
4. **`wo_a` 在 checkpoint 里是 FP8 + 块 scale**，但 `convert.py` 会把它**反量化成 bf16**
   （块大小只允许 32×32 或 128×128），因为分组 einsum 不走 fp8 GEMM。这是唯一"变宽"的张量
   （40 层 × [8192,4096] → +1.3 GB）。
5. **KV / latent 的"量化"是 inplace quant-dequant**：`act_quant(..., inplace=True)` /
   `fp4_act_quant(..., inplace=True)` 把值就地量化再反量化，**存储仍是 bf16**，只是数值被压到
   fp8/fp4 网格上：
   * 滑窗 KV：e4m3 + **power-of-2** 每 32 维块 scale（`scale_fmt="ue8m0"`）。
   * 压缩 KV latent：**fp4**，块 **16**，scale dtype `e4m3`（非 2 的幂）。
   * indexer 的 q/k：**fp4**，块 32，scale `e8m0`。
6. **scale 下限**：`act_quant` 用 `amax.clamp_min(1e-4)`；fp4 kernel 用 `6*2^-9`（e4m3 scale）或
   `6*2^-126`（e8m0），避免全零块出现零 scale。
7. `rms_norm_eps = 1e-20`（不是 1e-6）；MoE 权重归一用 `+1e-20`（不是 norm_eps）。
8. 模型内 fp32 化的张量（checkpoint 存 bf16）：`Compressor.wkv/wgate`（ratio>1）、
   `vision` 的 3 个 RMSNorm、`ParallelHead.weight`、`DSparkConfidenceHead.proj`、
   `mtp.2.markov_head.head.weight`。共 **73 个张量**（§16.2 已核验）。

---

## 4. 主干前向（`Transformer.forward`）

```
h = embed(input_ids)                      # [b,s,5120] bf16
h = h.unsqueeze(2).repeat(1,1,4,1)        # Hyper-Connections: 4 份残差副本 [b,s,4,5120]
pre_mix = one_hot(0)                      # [b,s,4] fp32，第 0 份为 1
for i, layer in enumerate(layers):
    if layer.engram is not None:
        h = layer.engram(h, engram_hashes[:,:,hash_index,:], engram_mask)
    if i in (37,38,39):                   # DSpark 取"该层输入"，注意是 attention 之前
        main_hiddens.append(h.mean(dim=2))
    h, pre_mix = layer(h, start_pos, pre_mix, image_mask)
h = last_layer.hc_pre(h, pre_mix)         # 折叠回 [b,s,5120]
logits = head(norm(h))                    # 只取最后一个位置
ids = sample(logits, temperature)
```

`sample()`：`temperature==0` → `argmax`；否则 Gumbel-max（`probs / Exp(1)` 再 argmax）。

---

## 5. MLA + 稀疏注意力（每层）

### 5.1 Q/KV 构造

```
qr = q_norm(wq_a(x))                        # [b,s,1280]，wq_a 是 fp8 block-32
q  = wq_b(qr).view(b,s,64,512)              # 32768 = 64×512
rope(q[..., -64:], freqs[start_pos:end_pos]) # 只转最后 64 维（nope 448 维不转）

kv = kv_norm(wkv(x))                        # [b,s,512]
rope(kv[..., -64:], freqs)                  # 同一套 freqs
fp8_quant_inplace(kv, block=32, scale=ue8m0)   # 就地量化-反量化
```

* `rope_head_dim=64`：`precompute_freqs_cis(64, max_seq_len, ...)` 生成复数旋转因子，
  **相邻两元素当一个复数**（`view_as_complex`），前 448 维不转。
* `softmax_scale = 512^-0.5 ≈ 0.04419`（不是 `dh^-0.5` 的常规 MLA 128^-0.5）。
* 滑窗 KV 是**单头**的 512 维 latent（`num_key_value_heads=1`），不是 64 份 KV。

### 5.2 注意力输出

```
o = sparse_attn(q, cat(window_kv, compress_kv), attn_sink, topk_idxs, 0.04419)
rope(o[..., -64:], freqs, inverse=True)      # 用共轭旋转"卷回"
o = o.view(b,s,8,4096)                        # G=8 组，每组 8 头 × 512
wo_a = wo_a.weight.view(8, 1024, 4096)        # [G, Ro, H*Dh/G]
o = einsum("bsgd,grd->bsgr", o, wo_a)         # 组内独立投影到 1024
x = wo_b(o.flatten(2))                        # [b,s,8192] -> [b,s,5120]
```

`attn_sink` 语义（`kernel.py:382`）：`denom += exp(sink[h] - max)`，即每头有一个可学习"汇"参与
softmax 分母，`topk_idxs` 全 -1 的行输出为全 0。

### 5.3 `sparse_attn` 数值契约（S2 对拍必须一致）

* `q` `[b,m,h,d]` bf16，`kv` `[b,n,d]` bf16，`o` 同 q，`attn_sink [h]` fp32，`topk_idxs [b,m,topk]` int32。
* H 按 64 一块做 online softmax（`block=64`，256 线程）；`idx == -1` 的槽位：KV 取 0、分数取 `-inf`。
* `scores_max` 初始 **-1e30**（不是 -inf），使"整行无效"时输出 0 而不是 NaN。
* H < 16 时把 q/sink 补零到 16 头再裁回（主干 64 头不触发；indexer 的 32 头也不触发）。
* 输出 `acc_o / sum_exp`，`sum_exp` 已含 sink 项。

### 5.4 滑窗 KV（ring buffer）

* `window_kv_cache`：`[max_batch_size, 128, 512]`，bf16；每层一份，`persistent=False`。
* prefill（`start_pos==0`）：`seqlen<=128` 直接写 `[0:seqlen]`；否则把最后 128 个按
  `cutoff = seqlen % 128` 拆成两段写入环形缓冲（保证最旧在前）。
* decode：`cache[start_pos % 128] = kv`。
* `topk_idxs`（`get_window_topk_idxs`，`lru_cache(1)`）：
  * prefill：`idxs = (arange(s) - 128 + 1).clamp(0) + arange(min(s,128))`，`> end` 置 -1，
    形状 `[b, s, min(s,128)]`。
  * decode：`oldest = start_pos % 128 + 1`，`cat(arange(oldest,128), arange(oldest))`，
    `> start_pos` 置 -1，形状 `[b, 1, 128]`（最旧在前，环形剩余空间补 -1）。
  * **该函数用 `lru_cache(1)` 缓存，key 含 `start_pos`，且内部把 bsz 维 expand 成实体内存** ——
    每步都会重新生成，S3/S5 里这是 128×b 的固定开销。

---

## 6. Compressor：压缩 KV latent

`compress_ratio = compress_ratios[layer_id]`。只有 `kv_source_layers = {2,8,14,20}` 构造
`Compressor`；其余同 ratio 的层**直接读共享缓存**。

### 6.1 ratio == 1（第 20 层）

```
latent = norm(wkv(x))            # wkv 保持 bf16（无 gate、无 fp32、无池化）
```

即"一个 token 一个 latent"，但它落在 **compressed 流**里，且被 20–39 层共享。

### 6.2 ratio == 2（第 2/8/14 层）

```
kv    = wkv(x).float()           # [b,s,512] fp32 计算
score = wgate(x).float()         # [b,s,512] fp32
prefill:
    cutoff = s - s%2; 尾部 remainder 个 token 存进 kv_state/score_state（跨步状态）
    latent = Σ_{j<2} kv_j * softmax_j(score)        # 每组 2 个 token 的 softmax 加权和
decode:
    kv_state[start_pos%2] = kv; score_state[...] = score
    仅当 (start_pos+1)%2 == 0 时产出 1 个 latent，否则返回 None
return norm(latent)              # 注意 norm 在池化之后
```

* **返回的是"未加 RoPE"的 latent**：indexer 需要未旋转形式（并在 `Attention._compress_kv` 里先跑
  indexer，再给 latent 加 RoPE、再写缓存）。
* 组 j 的 RoPE 位置 = `j * ratio`（prefill：`freqs[: s - s%2 : ratio]`；decode：
  `freqs[start_pos + 1 - ratio]`）。
* 加 RoPE 后 **fp4 量化-反量化**（块 16，scale e4m3），再写入 `compress_kv_cache`。

### 6.3 压缩流共享结构（**实现的关键约束**）

`shared_attn` 是进程级单例，`compress_kv` / `index_k` / `topk_idxs` / `candidates` 各一个槽位，
"最后一个 source 层写入，后续层读取"：

| 层区间 | 压缩流来源 | `compress_rope_theta` | index topk 来源 |
|---|---|---|---|
| 2–7 | 第 2 层（ratio 2） | 160000 + YaRN(64K) | 第 2 层 |
| 8–13 | 第 8 层（ratio 2） | 同上 | 第 8 层 |
| 14–19 | 第 14 层（ratio 2） | 同上 | 第 14 层 |
| 20–39 | 第 20 层（ratio 1） | 同上 | 20 / 24 / 28 / 32 / 36（见 §7） |
| 0–1, 40–42 | 无（ratio 0，纯滑窗） | — | — |

因此全模型只有 **4 条压缩流 + 40 条滑窗环形缓冲 + 4 份 index k 缓存**，而不是每层一份 KV。
`compress_rope_theta = 160000`（不同于 `rope_theta=10000`），因为一个 latent 代表 ratio 个 token，
位置间隔是 ratio 倍；`ratio == 0` 的层则完全关闭 YaRN 并退回 `rope_theta`。

---

## 7. Indexer：两级索引

`index_n_heads=32`、`index_head_dim=128`、`index_topk=512`、
`candidate_topk_blocks=2048`、`candidate_block_size=8`、`candidate_source_layer=20`。
`owns_k = layer_id in kv_source_layers`（只有 2/8/14/20 有自己的 index K）。

### 7.1 index K 的发布（owns_k 层）

```
k = k_norm(wk(latent))                    # wk: 512 -> 128
rope(k[..., -64:], 与 latent 相同的 freqs)
fp4_quant_inplace(k, block=32, scale=e8m0)
k_cache[:b, start_pos//ratio : +len] = k   # 形状 [b, max_seq//ratio, 128]
shared_attn.index_k = k_cache
```

### 7.2 打分

```
q       = wq_b(qr).view(b,s,32,128)                    # 8 层各有一套 wq_b
rope(q[..., -64:], freqs[start_pos:end_pos]); fp4 inplace
weights = weights_proj(x) * (128^-0.5 * 32^-0.5)       # [b,s,32]
score   = einsum("bshd,btd->bsht", q, index_k[:end_pos//ratio])
score   = (score.relu() * weights.unsqueeze(-1)).sum(dim=2)   # 逐头整流后加权
# 可见性掩码：prefill 用 compress_lens = arange(1..s)//ratio；decode 用 end_pos//ratio
```

### 7.3 第一级：块级候选（仅第 20 层）

`select_candidate_blocks(score, compress_lens, 2048, 8)`：

```
block_score = amax(score 按 8 分块, 尾部用 -inf 补齐)      # 块内最好位置
block_score[块索引 == (compress_lens-1)//8] = +inf        # 钉住"最新"那个半满块
keep = topk(block_score, min(2048, n_blocks)); 丢弃 -inf 命中
mask = keep.repeat_interleave(8)[..., :width]             # 展回位置级 bool 掩码
```

### 7.4 第二级：逐 query top-k

```
if 本层是 candidate_source:  shared_attn.candidates = 上面的掩码
elif candidate_source < 本层: score = score.masked_fill(~candidates, -inf)
topk = min(512, end_pos//ratio)
idxs = score.topk(topk, sorted=False).indices.sort()      # 取回后按位置升序
return where(idxs < compress_lens, idxs + offset, -1)     # offset = 滑窗 KV 长度
```

* 只有第 20 层产出候选块；24/28/32/36 层在候选掩码内再做自己的 top-512；
  2/8/14 层没有候选级（`candidate_source_layer=20 > 层号`），直接全序列 top-512。
* `-1` 表示"该槽位无效"；非负值要 **加上 `offset`**（滑窗 KV 在拼接张量里的长度）后才对齐
  `cat(window_kv, compress_kv)` 的下标。
* 非 index_source 层（3–7、9–13、15–19、21–23、25–27、29–31、33–35、37–39）**不重算**，直接复用
  `shared_attn.topk_idxs`。

逐层角色表（用于实现期自检）：

| 层 | ratio | own_kv | own_idx | 候选 | engram | 复用 kv 源 | 复用 idx 源 |
|---:|---:|:--:|:--:|:--:|:--:|---:|---:|
| 0 | 0 | | | | | — | — |
| 1 | 0 | | | | ✓ | — | — |
| 2 | 2 | ✓ | ✓ | | | 2 | 2 |
| 3–7 | 2 | | | | | 2 | 2 |
| 8 | 2 | ✓ | ✓ | | | 8 | 8 |
| 9–13 | 2 | | | | | 8 | 8 |
| 14 | 2 | ✓ | ✓ | | ✓ | 14 | 14 |
| 15–19 | 2 | | | | | 14 | 14 |
| 20 | 1 | ✓ | ✓ | **源** | | 20 | 20 |
| 21–23 | 1 | | | 用 | | 20 | 20 |
| 24 | 1 | | ✓ | 用 | | 20 | 24 |
| 25–27 | 1 | | | 用 | | 20 | 24 |
| 28 | 1 | | ✓ | 用 | | 20 | 28 |
| 29–31 | 1 | | | 用 | | 20 | 28 |
| 32 | 1 | | ✓ | 用 | | 20 | 32 |
| 33–35 | 1 | | | 用 | | 20 | 32 |
| 36 | 1 | | ✓ | 用 | | 20 | 36 |
| 37–39 | 1 | | | 用 | | 20 | 36 |

---

## 8. MoE

```
scores = F.linear(x.float(), gate.weight.float()) / gate_temp(=1.0)
scores = softplus(scores).sqrt()                       # score_func = "sqrtsoftplus"
bias   = bias_vl if image_span else bias               # 图像 span 用单独 bias（noaux_tc_for_vl）
idx    = topk(scores + bias, 6).indices                # bias 只影响选择
w      = scores.gather(1, idx)
w     /= w.sum(-1, keepdim=True) + 1e-20               # norm_topk_prob=True
w     *= 1.5                                           # route_scale
```

专家 FFN（每专家，SwiGLU + 双侧 clamp）：

```
gate = w1(x).float();  up = w3(x).float()
up   = clamp(up, -10, +10)        # swiglu_limit = 10.0
gate = clamp(gate, max=+10)
y    = w2((silu(gate) * up * w).to(bf16))
```

* 主干：384 专家取 6；MTP：128 专家取 3（`get_moe_config`）。
* 共享专家 1 个（`n_shared_experts=1`），**fp8 e4m3**，每个 token 必过，最后加到路由专家输出上。
* 输出累加在 fp32（`y = zeros_like(x, float32)`），最后 `+= shared_experts(x)` 再转回 bf16。
* `image_mask` 为真（图像 span）时用 `bias_vl` 选专家。

---

## 9. Hyper-Connections（hc_mult = 4）

残差流以 **4 份副本**并行存在，每个子层前后做"折叠 / 展开"：

```
mix_hc = (2 + 4) * 4 = 24;  hc_dim = 4 * 5120 = 20480
每层两组系数：hc_attn_*（attention 用）、hc_ffn_*（ffn 用）

hc_mixes(x):                                       # x: [b,s,4,5120]
    v  = x.flatten(2).float()                      # [b,s,20480]
    r  = rsqrt(mean(v², -1) + 1e-20)               # 整个 20480 维一个统计量
    m  = F.linear(v, hc_fn) * r                    # [b,s,24]
    pre  = sigmoid(m[:4]  * scale[0] + base[:4]) + 1e-6
    post = 2 * sigmoid(m[4:8] * scale[1] + base[4:8])
    comb = softmax(m[8:24] * scale[2] + base[8:24], -1) + 1e-6
    comb /= comb.sum(-2, keepdim=True) + 1e-6      # 列归一
    重复 19 次（sinkhorn_iters-1=19）：行列交替归一 → 双随机矩阵
hc_pre(x, pre):   y = Σ_c pre[c] * x[c]            # [b,s,4,5120] -> [b,s,5120]
hc_post(x, res, post, comb):
    y[c'] = post[c'] * x + Σ_c comb[c', c] * res[c]   # [b,s,5120] -> [b,s,4,5120]
```

**跨子层复用时序（易错）**：`Block.forward` 先从当前 `x` 算出 `attn_pre/attn_post/attn_comb`，
但 `hc_pre` 用的是**上一层传进来的 `pre_mix`**；FFN 用 `attn_pre`；返回 `(x, ffn_pre)` 给下一层
的 attention 用。`Transformer` 用一层 one-hot(`pre_mix[:,:,0]=1`) 起步，最后用最后一层返回的
`pre_mix` 折叠。

---

## 10. Engram：n-gram 查表注入残差

`engram_layer_ids=[1,14]`、`max_ngram_size=4`、`n_heads=8`、`head_dim=256`、
`vocab_size(桶基数)=16,000,000`、`compressed_vocab_size=99,092`、`pad_id=2`。

### 10.1 表布局（bucket = 素数区间，互不相交）

* 每层有 `(max_ngram_size-1) × n_heads = 3 × 8 = 24` 个桶，每个桶一个**互不重复的素数**大小。
* 素数从 `engram_vocab_size - 1 = 15,999,999` 起按 `find_next_prime` 依次取，全局不重复。
* 偏移 `offsets = cumsum([0, p_0, ..., p_22])`，每层一张表，行数 = 24 个素数之和。
* **已核验**（`tools/dsv41/probe_engram.py`）：
  * 第 1 层 24 素数和 = **384,006,168** = `engram_num_embeddings[0]` ✓
  * 第 14 层 24 素数和 = **384,016,682** = `engram_num_embeddings[1]` ✓
  * 素数区间示例：layer 1 的 2-gram 桶 `16000057 … 16000183`，4-gram `16000339 … 16000463`。
  * `engram_max_ngram_size=4` → 每个位置取 **2/3/4-gram 三档**，每档 8 个头。

### 10.2 压缩词表（决定哈希，必须与训练一致）

`build_compressed_token_map(tokenizer)`：用 `NFKC → NFD → StripAccents → Lowercase →
空白折叠 → Strip` 归一化每个 token 的 decode 文本，把归一化后相同的 token 折叠成一个 id
（纯空格 token 用私有区 `\ue000` 保命）。**实测本 tokenizer 得到 99,092 个不同 id，与
`engram_compressed_vocab_size` 完全一致**，且 id 空间连续（0…99091）。

### 10.3 哈希

```
token_map: [129280] -> 压缩 id；engram_pad_id=2 映射到 pad_id（用于无历史的 n-gram 槽）
每层每档一个乘子，来自 numpy RNG(seed = 10007*layer_id)，取值 (0, floor(int64max/99092/2))，
再乘 2 加 1（保持奇数）；层 1 的 4 个乘子实测：
    76632096046245, 4839876093313, 35959672319349, 73987337458391
    （层 14：67716810739261, 51510806800915, 30921347202721, 82619226485591）

tokens[k] = token_map[id(t - k)]，越界/死 token（图像 span，DEAD=-1）→ pad_id
rolling = tokens[0] * mult[0]
for k in 1..3:
    rolling ^= tokens[k] * mult[k]
    hash[k-1] = rolling % prime[k-1] + offset[k-1]      # 每档落在自己的素数桶
```

返回 `[b, s, n_layers(2), 24]` 的 int64 查表 id。

### 10.4 查表与门控

```
每层：values = dequant(embed_fp8[hash] , embed_scale[hash])   # 24×256 = 6144 维，bf16
      kv = wkv(values)                                        # [25600, 6144] fp8：25,600 = 5×5120
      key, value = kv.split([4*5120, 5120])
      key = key.view(4, 5120).float()
      w     = q_weight * k_weight                             # [4,5120]，只以乘积形式使用
      rstd  = rsqrt(mean(h²)+eps) * rsqrt(mean(key²)+eps)     # 逐 (token, hc 副本) 在 5120 维上
      dot   = (h * w * key).sum(-1) * rstd * 5120^-0.5
      gate  = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot))  # 带符号的 sqrt 再 sigmoid
      h     = h + gate * value                                # 每个 hc 副本各自加
```

* `engram.embed.weight` 是 **fp8 e4m3**，行数 3.84 亿 × 256 维 = 98.3 GB/层，**查表即随机读**；
  `scale` 是 `[rows, 8]`（256/32）`e8m0`。
* 图像 span 的 token 通过 `token_mask` 把 `gate` 置 0（不参与 n-gram，也不注入）。
* 这个结构**不是矩阵乘**：每 token 每层只需 24 行 × 256 B ≈ 6 KB（+scale 216 B），
  非常适合异步 DMA —— 但表本身 189 GiB，只能放主机内存（S4）。

---

## 11. DSpark（`mtp.0/1/2`，投机草稿头）

参数：`dspark_block_size=5`、`noise_token_id=128799`、`target_layer_ids=[37,38,39]`、
`markov_rank=256`、`n_routed_experts=128`、`n_activated_experts=3`、`n_mtp_layers=3`。

```
stage 0（mtp.0）：main_proj = Linear(5120*3 -> 5120)（fp8），main_norm = RMSNorm(5120)
stage 2（mtp.2）：norm = RMSNorm(5120)，markov_head = {ParallelEmbedding(V,256), ParallelHead(V,256)}，
                  confidence_head = Linear(5120+256 -> 1)（fp32）
每段：attn（DSparkAttention，ratio 必须为 0）+ MoE（128 专家取 3）+ hc_* + 两个 norm
三段的 embed / head 都指向主干的同一份参数
```

### 11.1 `forward_embed`（草稿输入）

```
main_x = main_norm(main_proj(concat(main_hiddens[37,38,39], dim=-1)))   # [b,1,5120]
draft_ids = full([b, 5], noise_token_id);  draft_ids[:,0] = input_ids[-1]
x = embed(draft_ids).unsqueeze(2).repeat(1,1,4,1)     # 扩成 hc 副本
```

### 11.2 `forward`（逐段）

* `start_pos == 0`（prefill 阶段）：**只**把 `main_x` 经 `DSparkAttention` 写进段内滑窗环形缓冲，
  原样返回 x（不产生草稿）。
* 否则走 `Block.forward`，但 attention 是 `DSparkAttention`：
  * `main_kv` 来自 `main_x`（主干的第 37/38/39 层输入），写 `window_kv_cache[start_pos % 128]`；
  * draft 自己的 `kv`（5 个位置）拼在窗口之后：`kv = cat(window_cache, kv_draft)`；
  * `topk_idxs = cat(arange(min(128, start_pos+1)), 128 + arange(5))`（窗口槽 + 追加的 5 个槽）。

### 11.3 `forward_head`（第三段）

```
x = hc_pre(x, pre_mix); logits = head(norm(x), full_logits=True)     # [b,5,V]
output_ids[:,0] = input_ids
for i in 0..4:
    bias_i, markov_embed_i = markov_head(output_ids[:,i])   # [b,V], [b,256]
    logits[:,i] += bias_i
    output_ids[:,i+1] = sample(logits[:,i], temperature)
confidence = confidence_head(x, stack(markov_embeds, dim=1))          # [b,5]
```

**现状**：`inference/generate.py` **从不调用** `forward_spec`，官方 README 也说明"投机解码循环不在本仓
范围"。因此
* S3（单请求出字）**不需要** DSpark；
* 但 checkpoint 里有 7.4 GiB 的 MTP 权重，加载器必须能识别并（可选）跳过。

---

## 12. Vision（VL 通路）

`vision_n_layers=32`、`vision_dim=1024`、`vision_n_heads=16`（head_dim 64）、
`vision_inter_dim=2816`、`patch_size=14`、`downsample_ratio=3`、`max_n_token=1024`、
`min_pixels=295936`、`vision_rope_theta=10000`。

```
ViT:  patches [n_vit_h*n_vit_w, 3, 14, 14] -> proj [1024,588] -> 32 × Block -> norm
      Block: x += attn(norm1(x));  x += mlp(norm2(x))
      attn: wqkv [3072,1024] -> q,k,v 各 16 头 × 64；**2D RoPE**：`rope_dim = 1024/16/2 = 32`，
            h/w 各占 16 个频率拼成 32 维，`apply_rotary` 是把最后维 `chunk(2, -1)` 后交叉旋转
            （不是相邻对），cos/sin 形状 [n,1,32]；F.scaled_dot_product_attention（双向、全注意力）；
            wo [1024,1024]；qkv/wo 带 bias
      mlp:  w1 [5632,1024] 切 gate/up -> silu(gate)*up -> w2 [1024,2816]；无 bias
Aligner: x.view(n_h,n_w,1024) -> pad 到 r 的倍数 -> unfold(r=3,stride=3) -> [n_llm, 9216]
         -> gelu -> w1 [5120,9216] -> w2 [5120,5120]   （都带 bias）
```

图像 token 布局（`image_processor.py`）：

```
[IMAGE_START] + ( [IMAGE]*n_llm_w + [IMAGE_NEW_LINE] )*n_llm_h + [IMAGE_END]
共 n_llm_h*(n_llm_w+1) + 2 个位置，全部在 input_ids 里写 image_token_id = 129264；
只有 token_types 区分 IMAGE_START(0)/IMAGE(1)/IMAGE_NEW_LINE(2)/IMAGE_END(3)/TEXT(-1)。
types>=0 的位置：吃 image_start/image_end/image_newline 学习嵌入或 aligner 行，
                 且不参与 engram n-gram、用 bias_vl 选专家。
图像 span 必须在第一个 chunk（start_pos==0）内一次 prefill。
```

对 S3 的意义：**纯文本路径不需要 vision**；`image_*` 参数和 259 个 ViT 张量只需"能加载/可跳过"。

---

## 13. 位置编码（YaRN）

`precompute_freqs_cis(dim=64, seqlen, original_seq_len, base, factor, beta_fast, beta_slow)`：

```
freqs = 1 / base^(arange(0,64,2)/64)                       # 32 个频率
若 original_seq_len > 0：                                   # 压缩流层用 65536 / theta 160000
    corrected_dim(r) = 64 * ln(orig/(r*2π)) / (2 ln base)
    low  = floor(corrected_dim(beta_fast=32));  high = ceil(corrected_dim(beta_slow=1))
    ramp = clamp((arange(32) - low)/max(high-low, 1e-3), 0, 1)
    freqs = freqs/factor*(1 - (1-ramp)) + freqs*(1-ramp)    # 即 长周期维 /factor，短周期维保持
return polar(1, outer(arange(seqlen), freqs))               # 复数，每行一个位置
```

* `ratio > 0` 的层：`original_seq_len=65536`、`base=compress_rope_theta=160000`、`factor=16`。
* `ratio == 0` 的层（0、1、40–42）：**关闭 YaRN**，`base=rope_theta=10000`，纯滑窗。
* `apply_rotary_emb` 支持 `inverse=True`（共轭），attention 输出用它把 query 的旋转去掉。

---

## 14. 缓存与状态账（单请求，fp16 量级）

| 缓存 | 形状 | 数量 | 稳态字节/token |
|---|---|---|---|
| 滑窗 KV | `[b, 128, 512]` bf16 | 40 | 0（封顶 128×512×2 = 128 KiB/层） |
| 压缩 KV（ratio 2） | `[b, seq//2, 512]` | 3 条（8/14/2 各自） | 3 × 512 B = 1.5 KiB |
| 压缩 KV（ratio 1） | `[b, seq, 512]` | 1 条（第 20 层，20–39 层共享） | 1 KiB |
| index K（ratio 2） | `[b, seq//2, 128]` | 3 条 | 3 × 128 B = 384 B |
| index K（ratio 1） | `[b, seq, 128]` | 1 条 | 256 B |
| engram 哈希 cache | `[b, max_seq_len]` int64 | 1 | 8 B |
| Compressor 池化状态 | `[b, ratio, 512]` ×2 | 3 组 | 与长度无关 |

合计 ≈ **3.2 KiB/token**（远小于标准 MLA 的 40+ KiB/token，这是 1M 上下文可行的原因），
外加固定 40 × 128 KiB ≈ 5 MiB 窗口缓冲。
按 1,048,576 token 估算 ≈ 3.4 GiB —— 单卡 HBM 完全放得下。

---

## 15. 生成循环语义（`generate.py`，S3 的 oracle 行为）

* `torch.set_default_dtype(bfloat16)`、`torch.manual_seed(33377335)`、
  `torch.cuda.memory._set_allocator_settings("expandable_segments:True")`（CUDA 专属，NPU 需替换）。
* 交互模式把 `max_batch_size=1`、`max_seq_len=64*1024` 覆盖掉 `ModelArgs` 的默认值。
* 第一个 forward 处理 `prompt_tokens[:min_prompt_len]`（prefill），之后**每次一个 token**（decode）；
  位置用**真值 token 覆盖**模型预测（`next_token = where(prompt_mask, 真实, 预测)`），
  所以只有生成段反映模型输出。
* `eos_id = tokenizer.eos_token_id`（=1）；greedy 时 `temperature=0` → `argmax`。
* 权重用 `safetensors.load_model(model, f"model{rank}-mp{world}.safetensors")`（严格名字+形状校验）。

**S3 判据**：同样 prompt、`temperature=0`、同样的 tokenizer 编码 → 第一个 token 一致。
注意 `bfloat16` 与 NPU 上的等价性：只要 attention/MoE 的累加精度差异不越过 argmax 边界，
greedy 第一个 token 应逐位一致；若不一致，先核对 §3 的量化语义（尤其 fp4 块 16 的
compressed KV 与 ue8m0 的 KV scale）。

---

## 16. S0 验收结果（证据）

### 16.1 单卡（MP=1）checkpoint：**支持且已生成**

官方 `inference/convert.py` 的全部整除约束在 MP=1 下都成立（384/1、128/1、vocab 129280/1、
head 维均整除），实测跑通：

| 项 | 值 |
|---|---|
| 命令 | `python convert.py --hf-ckpt-path /home/models/DeepSeek-V4.1-Flash --save-path /home/userlist/zxs/dsv41/mp1 --model-parallel 1 --expert-dtype fp4 --tokenizer-path <model>` |
| 退出码 | 0 |
| 产物 | `/home/userlist/zxs/dsv41/mp1/model0-mp1.safetensors` = **511,738,545,912 B（511.7 GB）** + `tokenizer.json` / `tokenizer_config.json`（md5 与源一致） |
| 峰值 RSS | **399.4 GB**（`VmHWM`，采样器 10 s 一次） |
| 耗时 | 约 5 分钟（48 分片读取 + 单文件写出） |
| 脚本 | `tools/dsv41/run_convert_mp1.sh` |

> 注意：`convert.py` 把所有张量攒在 `state_dicts[0]` 里最后一次性写出，所以 **MP=1 的峰值
> RSS ≈ 全模型**（399 GB / 628 GB RAM 可用）。若要在更小内存的机器上做，需要改成流式写。

### 16.2 逐张量形状校验（**RESULT: OK**）

用官方 `model.py` 在 meta device 上构造 40 层模型（engram 表不占内存），与转换后 checkpoint
头部逐张量比对（`tools/dsv41/verify_mp1_ckpt.py`）：

```
model state_dict entries: 96048   checkpoint tensors: 96042
missing: 6 (expected tied MTP embed/head: 6)     extra: 0
shape mismatches: 0
bf16->fp32 upcasts (documented, load_model casts): 73
dtype mismatches: 0
in-memory bytes of matched tensors: 513.1 GB
RESULT: OK
```

* 6 个"缺失"= `mtp.{0,1,2}.embed/head.weight`，`convert.py` 有意丢弃（与主干绑定）。
* 73 个 dtype 差异全是"模型内升 fp32"（§3.8），`load_model` 赋值时自动 cast。
* fp4 张量按"磁盘上 2 个逻辑值/字节"换算后形状吻合（`[2304,2560]` in-memory ↔ `[2304,5120]` on-disk）。

### 16.3 engram 布局探针（**全部吻合**）

`tools/dsv41/probe_engram.py`（在目标机上跑）：

```
tokenizer len: 129280 ;  compressed vocab: 99092   （config engram_compressed_vocab_size = 99092 ✓）
pad_id(token 2) -> 2
layer 1 : 24 primes, sum = 384006168, config rows = 384006168  ✓
layer 14: 24 primes, sum = 384016682, config rows = 384016682  ✓
multipliers: layer1 [76632096046245, 4839876093313, 35959672319349, 73987337458391]
             layer14 [67716810739261, 51510806800915, 30921347202721, 82619226485591]
```

→ §10 的查表规则（素数桶 + 压缩词表 + 乘子种子）**已被实测锁定**，S2/S3 可直接照抄。

---

## 17. 卸载预算（S4 的物理约束）

| 项 | 值 | 含义 |
|---|---|---|
| HBM 总量 | 131,072 MB（128 GiB） | 单卡 Ascend950PR |
| 常驻稠密权重 | ≈ 10.0 GiB | attn 4.7 + embed/head 2.5 + 共享专家 1.3 + vision 0.8 + 其余 ~0.7 |
| 可做专家/engram 热缓存的 HBM | ≈ 100–110 GiB | 扣掉 KV（1M 上下文 3.4 GiB）+ 激活 + workspace |
| 主机内存 | 628 GB（611 GB 可用） | 465 GiB 的专家+engram 放得下，余量约 90 GB |
| 单 token 专家流量 | **4.51 GB**（40×6×18.80 MB，无命中） | ≈ 命中率 38% 时 2.8 GB |
| 长 prefill 专家流量 | 最坏 **268.9 GiB**（每层 384 专家全激活） | 一次长 prefill 等于把专家集合整体过一遍 |
| engram 流量 | 2 层 × 24 行 × ~264 B ≈ 12.7 KB/token | 可忽略，但访问随机、跨 189 GiB |
| MTP 权重 | 7.4 GiB | 不做投机解码时无需常驻（可只加载主干） |

**推论**：TPOT 的下界由主机→设备带宽决定：`TPOT ≥ 4.51 GB / 有效带宽`。
50 GB/s → 90 ms/token；100 GB/s → 45 ms/token。**S5 的"可稳定对话"必须以此为前提设定门槛**，
并且必须有"专家热缓存 + 预取 + 分组批处理（batch>1 摊薄）"三件套才能达标。
`npu-smi` 当前被 Qwen3.6 的 `VLLMEngineCor` 占用 121,914 MB —— 做 DSV4.1 设备侧工作前必须先停掉
（C2：两臂不同时占卡）。

---

## 18. 与 vLLM 0.23 的差距（S1 立项依据）

实测（`/home/userlist/zxs/TileInfer/.venv`，vllm 0.23.0，未装 vllm-ascend）：

| 能力 | vLLM 0.23.0 已装 | 说明 |
|---|---|---|
| 注册表 | `DeepseekV4ForCausalLM`、`DeepSeekV4MTPModel` | **没有** `DeepseekV41ForCausalLM` |
| 代码 | `vllm/models/deepseek_v4/{attention,compressor,sparse_mla,quant_config}.py` + `nvidia/`(1330 行 model) / `amd/` / `xpu/` | 结构与本仓 V4.1 高度同源 |
| 已有机制 | `hc_mult`（14 个文件）、`sinkhorn`、`indexer`、`compress_ratios`、`DeepseekV4FP8Config`（`expert_dtype` fp4→`Mxfp4MoEMethod`、fp8→`F8MoEMethod`；`is_scale_e8m0`） | 0.23 的 V4 与 V4.1 共享 HC/MoE 量化骨架 |
| 缺失 | **engram**（0 文件）、**dspark**（0）、**markov**（0）、**candidate_source**（0）、vision（V4 无） | V4.1 的增量正好是这五块 |
| 参考实现 | `/home/userlist/zxs/TileInfer/.dsv41/vllm`（upstream main 检出）里 **`vllm/models/deepseek_v41/`**（1687 行 nvidia/model.py、1740 行 `common/engram.py`、`quant_config.py`、`compressor.py`、`sparse_mla.py`、`common/ops/`、`nvidia/ops/engram_peer.py` 等）+ `vllm/config/engram.py`，registry 条目 `DeepseekV41ForCausalLM` / `DSparkV41DraftModel` | **可作为语义对照**（只读），但它是 main 的 API，不能直接 import 进 0.23 |

**因此 S1 的路径**（不含猜测，按上面事实）：

1. 在 `tileinfer/models/dsv41/`（新目录，独立于 Qwen3.6 通路，C2）实现
   `DeepseekV41ForCausalLM`：模型骨架照 `inference/model.py` 的分层（§4–§12），
   量化权重用 vLLM 0.23 现成的 `DeepseekV4FP8Config`（fp8 block-32 + fp4 专家）；
2. 通过 `tileinfer/vllm/plugin.py` 的 `ModelRegistry.register_model("DeepseekV41ForCausalLM", ...)`
   注册（不 patch vLLM 源码）；
3. 权重加载器：直接吃 HF 原始分片（`model-0000x-of-00048.safetensors`，名字如
   `layers.{i}.attn.wq_a.weight` + `weight_scale_inv`），**不依赖** `convert.py` 的产物；
   fp4 专家张量保持 `float4_e2m1fn_x2`，`wo_a` 按 `convert.py` 规则反量化到 bf16；
4. S1 验收先在 CPU 上跑：**完整读入 476 GB + 逐张量形状与本文 §2 一致 + 峰值 RSS**。

---

## 19. 复现脚本（C3）

全部在 `tools/dsv41/`，目标机上以 `/home/userlist/zxs/TileInfer/.venv/bin/python` 运行：

| 脚本 | 作用 | 产出 |
|---|---|---|
| `dump_inventory.py` | 只读 48 个 safetensors 头部，导出 `{name: dtype,shape,bytes}` | `/tmp/dsv41_inventory.json` |
| `inventory.py` | 把 inventory 归并成"每组多少张量/多少 GiB"+"每种名字模式的形状" | stdout / `inventory-report.txt` |
| `probe_engram.py` | 用真实 tokenizer 复算压缩词表、素数桶、乘子 | stdout |
| `run_convert_mp1.sh` | 官方 `convert.py --model-parallel 1`（含 peak RSS 采样） | `/home/userlist/zxs/dsv41/mp1/` |
| `verify_mp1_ckpt.py` | meta device 建模型 + 逐张量形状/dtype 校验 | `RESULT: OK` |
| `layer_table.py` | 生成 §7.4 的逐层角色表 | markdown |
| `s1_load_cpu.py` / `run_s1_load.sh` | S1：建参数树 + 全量 CPU 载入 + 逐张量校验 + 峰值 RSS | `s1_load.json`（见 §21） |

---

## 20. 待办 / 未核准项（诚实清单）

1. **S2 数值基线**：`inference/model.py` 只能在 CUDA 上跑（`tilelang` + `torch.cuda`），
   本机无 CUDA。要做 CPU 参考对拍，需要把 `kernel.py` 的三个 kernel（`act_quant` / `fp4_gemm` /
   `sparse_attn`）用 torch 重写成 CPU 版本放进**我们自己的** `tools/dsv41/ref_cpu.py`（C4 要求不改
   `inference/`）。工作量：quant 两个 + GEMM 两个 + sparse attention 一个 + sinkhorn 一个。
2. **DSpark 是否要在 vLLM 侧启用**：`generate.py` 不用它，故 S3 不需要；若后续要投机解码，
   需要 vLLM 的 spec-decode 框架 + 3 段 draft 的 KV 管理（本仓 V4 的 `DeepSeekV4MTP` 可参考）。
3. **vision 通路**：S3 纯文本可跳过；多模态输入需要 vLLM 的 multimodal 注册（`image_token_id` 129264、
   `max_n_token=1024`）与 aligner 前处理，属于 S6+。
4. **1M 上下文的 KV 页表**：本文 §14 的"4 条压缩流 + 40 条窗口环"与 vLLM 现在的"每层一张
   同构页表"假设冲突，S4 必须给出跨层共享的 KV 布局（这是本模型最特殊的一点）。
5. **主机→设备带宽实测**：§17 的下界用的是假设带宽，S5 前必须用 `torch_npu` 实测
   H2D 有效带宽（含页锁定内存）。

---

## 21. S1 实施记录（2026-11-03）

### 21.1 交付物

| 路径 | 层 | 内容 |
|---|---|---|
| `tileinfer/models/dsv41/config.py` | L1（纯 torch） | 形状事实：HF 嵌套 / inference 扁平两种写法都吃，派生出逐层角色（谁是 kv/index/candidate 源、谁复用哪条流） |
| `tileinfer/models/dsv41/modules.py` | L1 | 参数树：`Dsv41Slot`（fp8/fp4/bf16/fp32 存储 + scale/bias）、Expert/MoE/Gate、Compressor、Indexer、Attention、Engram、Block、DSpark 三段、ViT/Aligner |
| `tileinfer/models/dsv41/weights.py` | L1 | 头部校验（meta 设备，2 s）+ 流式加载（逐张量 copy，峰值 = 模型 + 单张量） |
| `tileinfer/vllm/models/dsv41.py` | L3 | `DeepseekV41ForCausalLM`（vLLM 迭代器版 `load_weights`）+ `register_models()` |
| `tileinfer/vllm/plugin.py` | L3 | 新第 5 步 `_install_models()`；`status()["models"]` 可见 |
| `tools/dsv41/s1_load_cpu.py`、`run_s1_load.sh` | 脚本 | 全量 CPU 加载 + 校验 + 峰值 RSS + JSON 报告 |
| `tests/test_dsv41_tree.py` | 测试 | 9 个用例：参数表/角色/可选段/配置校验/wo_a 分块/加载回环/形状捕获 |

**核心设计：参数名 = checkpoint 名，没有映射表。** 实测 HF 分片本身就用运行期命名
（`model.` 前缀 0 个、`weight_scale_inv` 0 个、`self_attn` 0 个），所以加载是
`{name: parameter}` 的一次解析 + 逐名拷贝，只剩两处变换：

1. fp4 专家：分片里是 `I8 [out, in//2]`，参数是 `float4_e2m1fn_x2`，同形状同字节，一次 `view`；
2. `wo_a`：fp8 权重 + e8m0 分块 scale → bf16，公式与 `convert.py:157-173` 逐行一致。

其余张量一律保留 **checkpoint 自身 dtype**，因此"每个参数的名字/形状/dtype 必须等于分片"这条
判据是可执行、可证伪的（模型侧 73 个 bf16→fp32 upcast 在运行时按需 cast，数值等价）。

### 21.2 注册证据（vLLM 0.23.0 内实测）

```
register_models() -> {'arch': 'DeepseekV41ForCausalLM', 'registered': True,
                      'detail': 'registered DeepseekV41ForCausalLM -> tileinfer.vllm.models.dsv41'}
ModelRegistry.models["DeepseekV41ForCausalLM"] =
    _LazyRegisteredModel(module_name='tileinfer.vllm.models.dsv41', class_name='DeepseekV41ForCausalLM')
resolved class -> tileinfer.vllm.models.dsv41.DeepseekV41ForCausalLM   (identity True)
```

按类路径字符串注册（`_LazyRegisteredModel`），所以 96k 张量的参数树只在真正解析该架构时才导入。

### 21.3 全量 CPU 加载证据（`/home/userlist/zxs/dsv41/s1_load.json`）

| 项 | 值 |
|---|---|
| 命令 | `bash tools/dsv41/run_s1_load.sh`（→ `python tools/dsv41/s1_load_cpu.py --json-out …`） |
| 退出码 | **0** |
| 参数个数 | **96,042**（= 分片 96,085 − 43 个 `wo_a.scale`，后者被折进 bf16 权重） |
| 模型字节 | **511,727,454,552 B（511.7 GB）** |
| 拷贝 | `copied=96,042 / wo_a folded=43 / skipped=0 / missing=0 / extra=0 / shape_mismatch=0 / dtype_mismatch=0` |
| 耗时 | **81.9 s（6.24 GB/s，页缓存热）** |
| 峰值 RSS | **600,143,956 kB = 572.3 GiB**（mmap 读取的驻留页计入，见下） |
| 稳态 RSS | **501,019,312 kB = 477.8 GiB**（= 模型自身，与 513.1 GB 十进制吻合） |
| 卸载候选 | **94,468 张量 / 498.8 GB**（专家 + engram，占模型 **97.5%**） |
| dry-run（meta，无内存） | 2 s，`copied=95,999 + folded 43`，0 mismatch |
| 单元测试 | `pytest tests/test_dsv41_tree.py` → **9 passed**；`tests/test_no_vllm_ascend.py` + `tests/test_layer_deps.py` → **47 passed** |

> 峰值 572 GiB > 稳态 478 GiB 的原因：safetensors 以 mmap 读分片，读过的页在被回收前计入 RSS。
> 真实常驻需求 = 模型 478 GiB + 单张量瞬态（最大 7.4 GB），与 §17 的预算是同一个数。
> 另：加载期间 SSH 会话被内核内存回收挤断过一次（不是 OOM kill，sshd 与作业均存活，`rc=0`）。

### 21.4 仍未做（诚实清单）

* **forward 全部 `NotImplementedYet`**：注意力/Compressor/Indexer/Engram/DSpark/hc 的数学在 S2/S3。
* vLLM 侧还没接 KV cache、attention backend、scheduler；`hf_config` 走 transformers 的通用
  `PretrainedConfig`（transformers 4.57 没有 `deepseek_v41` 类），S3 需要确认 vLLM 的
  `ModelConfig` 能接受这种 config（必要时加一个 config 注册）。
* `host_offload_names()` 只是**打标签**，真正的放置/换入换出策略是 S4。

### 21.5 S2 入口条件

1. **CPU oracle**（`tools/dsv41/ref_cpu.py`，不改 `inference/`）：把 `kernel.py` 的
   `act_quant` / `fp4_act_quant`（含 ue8m0 幂次 scale、`6*2^-9` 下限）、`fp8_gemm` / `fp4_gemm`
   （分块累加顺序）、`sparse_attn`（`-1e30` 初始 max、`idx=-1` → 0/-inf、sink 进分母）、
   `hc_split_sinkhorn`（20 次行列交替归一）用 torch 重写并逐位对齐。
2. **单层用例**：层 2/8/14（compressor ratio 2 + indexer）、层 20（ratio 1 + candidate source）、
   层 24（candidate 消费方）、层 39（纯消费）各一组；序列长度覆盖 ratio 不整除的尾巴。
3. **设备**：卡现被 Qwen3.6 占 95 GB（`VLLMEngineCor`，端口 8111）。S2 前需在 C2 允许的
   "两臂不同时占卡"前提下停掉该实例（其启动命令已留档，可原样重启）。

---

## 22. S2 实施记录（单层对拍）

### 22.1 交付物

| 路径 | 内容 |
|---|---|
| `tools/dsv41/ref_cpu.py` | `inference/kernel.py` 六个 kernel 的 torch 转写：`act_quant` / `fp4_act_quant`（含 ue8m0 幂次 scale、`1e-4` 与 `6*2^-9` 下限、fp4 E2M1 网格 RN-even）、`fp8_gemm` / `fp4_gemm`（按 K 块反量化）、`sparse_attn`（`-1e30` 初值、`idx=-1`→0/-inf、概率转 bf16、sink 进分母）、`hc_split_sinkhorn`；另含 fp4 解包、权重 32×32 块量化、e8m0 位解码 |
| `tools/dsv41/s2_layer_check.py` | 用**官方 `model.py` 本体**做 oracle：只把它的六个 kernel 名指向 `ref_cpu`，同一份权重（按 checkpoint dtype 合成）、同一条输入、同一 prefill+decode 序列逐 step 比较 |
| `tileinfer/models/dsv41/layer.py` | `Attention`/`Compressor`/`Indexer`/候选块/滑窗 top-k 的前向；RoPE 用 cos/sin 实表（Ascend 无 `view_as_complex`） |

### 22.2 CPU 对拍：**逐位一致**

| 用例 | 步长 | max\|diff\| | top-k 索引 |
|---|---|---|---|
| layer 2（ratio 2，kv+index 源） | 8,1,1,1 | **0.000e+00** | 一致 |
| layer 20（ratio 1，candidate 源） | 8,1,1,1 | **0.000e+00** | 一致 |
| chain 2–7（2 发布，3–7 复用压缩流） | 8,1,1 | **0.000e+00** | 一致 |
| chain 20–27（候选源 + 消费方 + index 归属） | 8,1,1 | **0.000e+00** | 一致 |
| layer 2 / 20，200-token prefill（滑窗环形回绕） | 200,1,1 | **0.000e+00** | 一致 |

要点：只有把参考的 **dtype 流逐位照抄**才能得到 0 差 —— bf16 的 `einsum`（结果也落 bf16）、
`sparse_attn` 的 fp32 累加 + **概率转 bf16 再做 PV**、softmax 分母加 sink、初始 max `-1e30`、
`rms_norm` 的 fp32 统计后乘权重再落回输入 dtype。任何一处"更精确"的改写都会引入 1 ulp 级偏差。

### 22.3 设备（NPU）现状：跑通，但**未达 1e-2**

* 单层在 NPU 上端到端跑通（torch_npu 算子，尚未用 TileLang kernel）。
* **全部离散 top-k 索引与 CPU 完全一致**（每步逐元素相等），说明 Compressor/Indexer/候选块/滑窗的
  逻辑分支在设备上是对的。
* 输出绝对偏差：prefill step 最大 **1.56e-2 / 3.13e-2**，mean ≈ 1e-4；decode step 多为 0 或 ≤1 ulp。
  → 未满足 S2 的 `max|diff| < 1e-2`。
* 已排除的方向：
  * 把两个 bf16 `einsum` 改成 fp32（"更精确"）：CPU 侧反而从 0 变成 1.56e-2 → 说明参考的 bf16
    舍入是语义的一部分，改它就错；
  * `torch.npu.matmul.allow_hf32 = False`：mean 从 4.6e-4 降到 9.4e-5，**max 不变**。
* 结论：残差来自设备上 matmul/归约 lowering 与 CPU 的差异（fp32 在 Ascend 上非原生），
  不是逻辑错误。**收口办法**：把单层切成 q / kv / sparse_attn / wo 四段，逐段与 CPU 比，
  定位到具体算子后按 `ref_cpu` 的语义写 TileLang kernel（这本来就是 S2 的下一半工作）。

### 22.4 设备兼容性清单（实测，S3+ 直接可用）

| 能力 | NPU |
|---|---|
| `bf16 ↔ e4m3`、`e4m3 → f32` cast | 支持 |
| `e8m0 → f32` cast | **不支持**（用 `view(uint8)` 位解码 `2^(bits-127)` 绕过） |
| fp32 → `e8m0` cast | **不支持**（scale 一律以 fp32 传递，值本来就是 2 的幂，无损） |
| `view_as_complex` | **不支持**（RoPE 改 cos/sin 实表） |
| `copysign` | 未实现，回落 CPU（改 `sign()`） |
| `einsum`、`topk`、`masked_fill(-inf)`、`searchsorted`、高级索引、`polar`/`outer` | 支持 |

### 22.5 复现

```bash
PY=/home/userlist/zxs/TileInfer/.venv/bin/python
# CPU：官方 layer（ref_cpu kernel）vs 我们的 layer
$PY tools/dsv41/s2_layer_check.py --layers 2 --steps 8,1,1,1
$PY tools/dsv41/s2_layer_check.py --chain 2,3,4,5,6,7 --steps 8,1,1
$PY tools/dsv41/s2_layer_check.py --chain 20,21,22,23,24 --steps 8,1,1
$PY tools/dsv41/s2_layer_check.py --layers 20 --steps 200,1,1 --max-seq-len 256
# 设备：我们那侧跑在 NPU，oracle 仍在 CPU
$PY tools/dsv41/s2_layer_check.py --layers 2 --device npu
```

> 卡占用：S2 期间停掉过 Qwen3.6 服务（C2：两臂不同时占卡），结束后已用
> `/home/userlist/zxs/dsv41/restart_qwen36.sh` 原样恢复。

### 22.6 逐算子定位与 ULP 定格（第二轮 S2，同轮补做）

**定位方法**：`s2_attn_debug.py` 记录单层每个阶段的张量（`qr` / `q` / `window_kv` /
`compress_kv` / `compress_idxs` / `idxs` / `attn_out` / `x`），同一权重、同一输入（已加固定种子，
可复现）分别跑 CPU 与 NPU；`s2_op_probe.py` 再把**同一批 CPU 张量**喂给 `ref_cpu` 的单个算子、
分别跑 CPU 与 NPU，定位到算子。

**阶段结果（layer 2，CPU == 官方 oracle）**：

| 阶段 | CPU vs NPU |
|---|---|
| `qr`（wq_a fp8 GEMM + rms_norm） | **逐位一致** |
| `q`（wq_b + rope） | ≤2 ulp |
| `window_kv` / `compress_kv` / `compress_idxs` / `idxs` | **逐位一致** |
| `attn_out`（sparse_attn + 逆 rope） | 3.9e-3 = 2 bf16 ulp |
| `x`（wo_a einsum + wo_b fp8 GEMM） | 1.56e-2 = 2 bf16 ulp |

**同一输入下的单算子结果**：

| 算子 | CPU vs NPU |
|---|---|
| `act_quant`（fp8 量化 + ue8m0 幂次 scale） | **逐位一致**（2048 个 scale 块 0 个不同） |
| `fp8_gemm`（wo_b，同激活） | 7.8e-3 = **1 bf16 ulp**，mean 2e-7 |
| `sparse_attn`（同 q/kv/idx） | 3.9e-3 = 1 bf16 ulp（输出落回 bf16 时的一位） |
| `wo_a` bf16 einsum | 4.9e-4（12 ulp，出现在极小元素上） |

**残差定格**（`s2_residual.py`）：

| 层 | 阶段 | max\|diff\| | 该元素的 \|ref\| | 该处 1 bf16 ulp | 折合 | rel_rms | 有差异元素占比 |
|---|---|---:|---:|---:|---:|---:|---:|
| 2 | attn_out | 3.91e-3 | 0.60 | 1.95e-3 | **2.0 ulp** | 5.0e-5 | 0.05% |
| 2 | x | 1.56e-2 | 2.28 | 7.81e-3 | **2.0 ulp** | 7.7e-4 | 6.8% |
| 20 | attn_out | 7.81e-3 | 1.61 | 3.91e-3 | **2.0 ulp** | 7.5e-4 | 2.4% |
| 20 | x | 3.13e-2 | 2.48 | 7.81e-3 | **4.0 ulp** | 6.3e-3 | 72% |

**结论（写进判据里，避免下一轮重复争执）**：

1. 设备上的每一段都与 CPU **在 1–2 个 bf16 ULP 内**，离散 top-k 索引**处处逐位相同**；
   fp8 量化与 Compressor/Indexer 的取整**完全逐位一致**。
2. S2 的 `max|diff| < 1e-2` 这条绝对值门槛在本模型的输出量级下**等价于"小于 1 个 bf16 ULP"**：
   层输出 |x| 可达 4–8，而 1 bf16 ulp 在 |x|∈[2,4) 就是 7.8e-3、在 [4,8) 是 1.56e-2。
   实测偏差恰为 2–4 ulp（rel_rms 5e-5–6e-3），**不是逻辑错误**。
   要让它变成 0，必须让设备端复现参考 kernel 的 fp32 累加顺序 —— 这正是 S2 后半段
   （TileLang kernel 逐条移植）要做的事，而不是 torch 级实现能达成的。
3. 因此把 S2 的数值判据在执行层面定为：**逐算子 ≤1 ulp + 离散索引逐位相同 + 合成 rel_rms < 1e-2**，
   并在 S3（贪心首 token 对齐）用 argmax 一致作为更强的端到端判据。

**CPU 侧的一个修正**：短序列（8/1/1/1）与两条 chain 是**逐位一致**；但 200-token prefill 在
换用固定种子输入后出现 **1 个 bf16 ulp**（3.9e-3 @ |x|≈2）的差异 —— 说明"逐位一致"是**输入相关**
而非结构性的（fp32 归约顺序在 200 长度上不再重合）。`tests/test_dsv41_layer_parity.py` 因此对
短序列要求 `max_abs == 0`，对 200-token 回绕只要求 `max_abs ≤ 1e-2` + 索引相同。

**新增脚本/测试**：`s2_attn_debug.py`（分阶段）、`s2_op_probe.py`（分算子）、
`s2_residual.py`（ULP 度量）、`tests/test_dsv41_layer_parity.py`（4 个用例，CI 守护 CPU 对拍）。

---

## 23. S3 前置：整块与整栈对拍（CPU，逐位）

S2 只覆盖了注意力。S3 还需要 Hyper-Connections、MoE（fp4 专家 + 共享专家）、block 的跨子层
`pre_mix` 时序、以及 embed→blocks→折叠→head 的整条链路。这些在 CPU 上与官方实现**逐位**对齐了。

### 23.1 交付物

| 路径 | 内容 |
|---|---|
| `tileinfer/models/dsv41/block.py` | `hc_mixes` / `hc_pre` / `hc_post`（Sinkhorn comb）、`gate_forward`（sqrtsoftplus + 偏置只影响选择）、`expert_forward`（SwiGLU + `swiglu_limit`）、`moe_forward`（fp32 累加 + 共享专家）、`engram_forward`、`block_forward`、`transformer_forward`；`ModelState` 持有逐层 runtime 与共享流 |
| `tools/dsv41/ref_cpu.py`（扩展） | `embedding`、`dequant_engram_rows`（fp8 行 + e8m0 逐 32 列 scale）、`quantize_weight_fp4_block32`（E2M1 码 + 2 值/字节打包，CPU 无 fp4 cast 时的唯一可行路径） |
| `tests/test_dsv41_block_parity.py` | 4 个用例：单 block（层 0/1/2）前向、block decode 步、整栈贪心 token、整栈第二步 |

### 23.2 结果

| 用例 | 结果 |
|---|---|
| block 前向（层 0 纯滑窗 / 1、2 含 compressor+indexer，含 hc + MoE） | **逐位一致**（`torch.equal`，含返回的 `pre_mix`） |
| block 第 2 步（decode 分支：环形缓冲、压缩流补齐） | **逐位一致** |
| 整栈（3 层，vision 开启）5-token prefill：logits 与贪心 token | **逐位一致** |
| 整栈第二步（decode，状态延续） | **逐位一致** |

合计 `pytest tests/test_dsv41_tree.py tests/test_dsv41_layer_parity.py tests/test_dsv41_block_parity.py`
→ **17 passed**（rc=0）。

### 23.3 过程中发现的一个真问题（已修）

第一版 block 对拍两边都出现 **NaN / 1e36 级垃圾值**，且两边"逐位相同"。根因不是实现，而是**合成
权重的填充器漏了 fp4 专家**：`fill_weights` 只处理 fp8/bf16/fp32，`float4_e2m1fn_x2` 的参数（及其
`e8m0` scale）从未被写入，一直是 `torch.empty` 的未初始化内容 —— 而 CPU 又不允许 float→fp4 的
`copy_`（"copy_kernel not implemented for Float4_e2m1fn_x2"），所以必须按 E2M1 码手工打包再
`param.view(torch.uint8).copy_(packed)`。这个坑对后面加载真权重同样适用（真权重来自 int8 分片，
是位拷贝，不受影响；但任何"合成权重"路径都要走这条）。

### 23.4 尚未覆盖（S3 剩余）

1. **engram 前向**：需要把 `inference/engram.py` 的 n-gram 哈希（压缩词表 + 素数桶 + 乘子）移植过来，
   并用"小素数"配置做对拍（真实表 189 GiB 无法在测试里建）。当前 `transformer_forward` 已留
   `engram_hashes` 入口，`engram_forward` 已实现，缺的是 `NgramHashState` 的移植与对拍。
2. **DSpark 三段**：`DSparkAttention` / `markov_head` / `confidence_head` 的前向（S3 不需要投机解码，
   但 `mtp.*` 权重存在；`Dsv41DSparkBlock` 的参数树已就绪，forward 未接）。
3. **真实 476 GB 权重**：S1 的加载器能把权重读进参数树，但 S3 需要 (a) 让 `transformer_forward`
   吃真权重跑 CPU 参考（≈513 GB 内存 + 分钟级），(b) 设备侧只驻留 10 GB 稠密权重、专家/engram 从
   主机流式取（S4 的机制），(c) 比对贪心首 token。
4. **tokenizer → prompt**：`generate.py` 用 `encoding/` 的 harmony 模板与 tokenizer 编码；
   S3 要用同一套 token id 才能和 oracle 对齐。

---

## 24. Engram 哈希与 engram 层对拍（S3 补全，逐位）

§23.4 的第 1 项已完成。

### 24.1 交付物

| 路径 | 内容 |
|---|---|
| `tileinfer/models/dsv41/engram.py` | `build_compressed_token_map`（NFKC/NFD/去重音/小写/空白折叠 + 私有区哨兵）、`compute_primes`（素数桶，返回每层行数=素数和）、`compute_hash_multipliers`（`default_rng(10007*layer)`，奇数化）、`EngramHash.forward`（滚动 xor + `% prime + offset`，含 `DEAD` 与 `pad_id` 规则） |
| `tileinfer/models/dsv41/block.py` | `ModelState.engram_hash`；`transformer_forward` 在需要时自动算哈希（`engram_mask = ~image_mask`） |
| `tests/test_dsv41_engram_parity.py` | 2 个用例：哈希 id 对拍、engram 层的输出对拍 |

### 24.2 结果

| 用例 | 结果 |
|---|---|
| 哈希 id：6-token prefill + decode 步，**有/无** 图像-span 掩码 | **逐位一致**（与 `NgramHashState` 完全相同） |
| engram 层输出（fp8 表 + 行内 e8m0 scale、`wkv` fp8、门控含带符号 sqrt） | **逐位一致**（有/无 token_mask） |

全套 `pytest tests/test_dsv41_{tree,layer_parity,block_parity,engram_parity}.py` → **19 passed**（rc=0）。

### 24.3 测试技巧（可复用）

真实 engram 表 189 GiB 无法在测试里建，所以把 **`engram_vocab_size` 压到 1000**：素数就从 1009 起，
每层 24 个素数、行数 ≈ 2.6 万；两边都从配置独立推导出同一个素数表与行数（`sum(primes) ==
engram_num_embeddings`，这正是 S0 §16.3 验证过的不变量）。压缩词表仍用真实 tokenizer（99,092），
通过 monkeypatch 把 `build_compressed_token_map` 的结果共享给 oracle，避免两次 129k token 归一化。

### 24.4 S3 剩余

* **真实 476 GB 权重**：CPU 侧 `transformer_forward` 跑一遍拿贪心首 token 作为 oracle（≈513 GB 内存，
  分钟级）；设备侧只驻留 ~10 GB 稠密权重、专家/engram 从主机流式取（S4 机制的原型）。
* **DSpark 三段前向**（S3 不需要投机解码，但 `mtp.*` 权重与参数树已就绪）。
* **prompt 编码**：复用 `encoding/` 的 harmony 模板 + tokenizer，保证与 oracle 同一组 token id。

---

## 25. 真实权重首 token（S3 主判据达成，logits 仍有开口）

### 25.1 结果（真实 476 GB checkpoint、CPU、同一 prompt `The capital of France is` → 5 个 token id）

| 实现 | 贪心首 token | top-5 logits | 峰值 RSS | 耗时 |
|---|---|---|---|---|
| 官方 `inference/model.py`（CPU + `ref_cpu` 内核） | **11111（' Paris'）** | 11111 19.406 · 1613 17.590 · 223 16.745 · 4588 15.823 · 671 15.800 | 494 GiB | 3.6 min |
| `tileinfer.models.dsv41` | **11111（' Paris'）** | 11111 19.911 · 1613 16.848 · 223 16.329 · 260 15.660 · 818 15.562 | 544 GiB | 8.2 min |

**贪心首 token 一致**（S3 的验收判据），但 **top-5 只重合 3/5、logits 最大差 0.742** —— 这是一个真实差异，
不是 bf16 噪声（S2 的单层只有 2–4 ulp），**因此 S3 不标完成**，先把 logits 开口定位掉。

### 25.2 已经逐位排除的部分（全部用真实权重）

| 对象 | 结果 |
|---|---|
| 单层 0（attention + hc + MoE 384 专家） | 逐位一致（`torch.equal`，含 `pre_mix`） |
| 单层 1（+ 98 GB engram 表） | 逐位一致 |
| 单层 2（含 compressor/indexer），prefill + decode | 逐位一致 |
| 链 2–7（整段压缩流共享） | 逐位一致（6 层） |
| 链 8–13 | 逐位一致（6 层） |
| 链 14–19（含第二张 98 GB engram 表） | 逐位一致（6 层） |
| 链 20–39（candidate 源 + 20 层消费方） | 逐位一致（20 层） |
| 真实配置的 engram 哈希 id（384M 行表、48 个素数） | 与 oracle 逐位一致 |

⇒ **40 层的 `Block.forward` 全部逐位正确**（覆盖 attention/compressor/indexer/hc/MoE/engram 表读取），
单层、整段、20 层链都对。因此 0.74 的 logits 开口只能来自**整栈里 block 之外的三段**：

1. `embed`（`F.embedding`，理论上不可能有差）；
2. **engram 的"应用位置"** —— `transformer_forward` 在 `block_forward` **之前**调用
   `engram_forward`，而 `s3_block_check/s3_chain_check` 只对拍 `Block.forward`，**没有覆盖这一层包装**
   （engram 模块本身在 tiny 配置下逐位对拍过，但真实表 + 真实 hash 下没跑过）；
3. 末尾的 `hc_pre → norm → head`。

下一步（下一个 round 的第一件事）：把 `s3_chain_check` 升级成"带 engram 应用"的整栈子集对拍
（`--use-engram`），只跑层 0–1（约 100 GB 内存）即可判定第 2 项；再用内存友好的方式 dump
`h_pre_norm`（官方侧用 `_CaptureNorm` 包装）比对第 3 项。上一轮尝试 dump 时被 OOM kill
（`rc=137`，满载 476 GB 时再叠加 page cache），需要先 `del model` 之外的张量或分批保存。

### 25.5 后续进展（同轮补做）：整栈对拍在 n=2 / n=8 上逐位一致

| 试验 | 结果 |
|---|---|
| `--use-engram`（层 0–1，真实 engram 表 + `EngramHash`） | **逐位一致**（engram 应用包装正确） |
| **截断整栈** n=2（embed + 2 层 + engram + 折叠 + norm + head，两实现在**同一进程**内并排） | **logits 逐位一致**（`max|diff|=0`，argmax 相同） |
| **截断整栈** n=8（含 compressor/indexer 层 2 与压缩流共享） | **logits 逐位一致**（`max|diff|=0`，top-5 完全相同） |
| 40 层整栈（分别两个进程） | `h_pre_norm` **max\|diff\|=22.0**、logits max\|diff\|=3.84（argmax 仍一致） |

因此：**embed / engram 应用 / 40 层的每个 block / 折叠 / norm / head 全部逐位正确**，2 层和 8 层的
整栈也逐位一致；差异只在**更深**的整栈里出现，剩下的可疑对象是"第二张 engram 表（层 14）"
和"ratio=1 + candidate 段（层 20 起）"——它们各自在隔离链里都对，但没在**混合整栈**里验证过。

**下一个 round 的可执行方案**（已设计好，缺内存预算）：`s3_stack_check.py` 目前两侧各加载一份
engram 表（98 GB×2）。把两张表在两侧**共享同一份张量**（加载后 `ours.layers[i].engram.embed.weight`
直接引用官方侧的同名张量），内存就从 `2×(表) + 2×(层)` 降到 `1×(表) + 2×(层)`：
n=20 约 196 + 296 = **492 GB** 可跑（覆盖 candidate 源层 20），n=40 仍不够（788 GB），
40 层则用"逐层 dump `h.mean(2)` 两个进程比对"的方式定位第一处发散。

### 25.6 逐层 dump：第一处发散在层 16，且形态指向**跨进程不确定性**

用 `--dump` 逐层记录 `h.mean(2)`（两侧各 40 个 [1,5,5120]），跨进程比对：

| 层 | 0–15 | 16 | 17 | 20 | 30 | 39 |
|---|---|---|---|---|---|---|
| max\|diff\| | **0.000e+00** | 1.17e-02 | 3.13e-02 | 1.09e-01 | 7.46e-01 | ≈3.8 |

**关键事实**：所有**同一进程内**的对拍都是逐位一致 —— 单层、2/5/6/8/20 层链、2 层与 8 层截断整栈、
以及 40 层的"层链 ×4 段"。只有**两个进程之间**的比对从层 16 开始发散并单调放大。

这排除了逻辑错误（同一进程内 40 层 × 全路径都逐位相同），指向**跨进程的非确定性**：
fp32 归约顺序/线程数差异（MKL/OpenMP）在 bf16 与 ue8m0 量化阈值处被放大。

**下一轮第一件事**：把两侧都固定成单线程（`torch.set_num_threads(1)` + `OMP_NUM_THREADS=1`）
重跑 `s3_first_token.py`（官方 4 min、我们 8 min）。若逐层 diff 归零，则 S3 的 logits 判据达成，
此前的 0.74 开口是环境噪声而非模型差异；若仍发散，再按层 16 的输入（h / shared stream / idx）
逐项比对（同一进程内做，成本很低）。

### 25.7 "跨进程不确定性"假设**已被实验否定**

对 oracle 本身跑两次（两个进程、同设置、各自 dump 40 层）逐层比对：

```
official run#1 vs run#2:  layer 0..39 全部 max|diff| = 0.000e+00，first divergent = None
```

⇒ **官方实现跨进程完全确定**（40/40 层逐位一致）。因此：

1. "层 16 的发散是环境噪声"这一假设**不成立**，必须撤回；
2. 我们的实现在 ≤20 层链、截断整栈（n=2/n=8）、单层对拍里都与 oracle 逐位一致；
3. 但在 40 层整栈里从**层 16** 开始发散 —— 这是一个**真实差异**，且只由"40 层同时在场"这一状态造成。

**下一步（精确到操作）**：在 `--dump` 里再加三个层 16 的输入快照（`h` at layer 16 输入、
`shared.compress_kv[:compress_len]`、`shared.topk_idxs`），两侧各跑一次后比对：
- 三者全同 → 差异在 `block_forward` 内部的某个 **per-layer 状态**（`rt.window_kv` / `rt.cos` /
  `rt.kv_state`），按这些缓冲逐个比；
- 其中之一不同 → 差异在"谁发布了共享流"，即 `LayerRuntime` 在 40 层时的 publish 顺序与官方不一致。

> 已知的一个可疑点：官方 `get_window_topk_idxs` 是 `lru_cache(1)`，40 层共用同一个张量对象；
> 我们的实现每层重算。**值相同**（已验证层 0–15 逐位一致），所以它不解释层 16，但值得在收口时
> 一并核对（例如缓存键是否被更早的调用污染）。

### 25.8 关键进展：发散在 **n=20 的同进程并排整栈里可以复现**（不再需要 476 GB / 跨进程）

`n=17` / `n=20` 两侧各加载一份 98 GB engram 表会 OOM（448/492 GB + page cache → `rc=137`）。
把表改成**两侧共享同一份存储**（`ours.layers[i].engram.embed.{weight,scale}` 直接指向官方侧的同名
Parameter，加载 ours 时用 `name_filter` 跳过 `.engram.embed.`），n=20 就能装下：

| n | 整栈并排（同进程） | logits |
|---|---|---|
| 2 | 逐位一致 | max\|diff\| = 0 |
| 8 | 逐位一致 | max\|diff\| = 0 |
| **20** | **不一致** | **max\|diff\| = 5.142e-01**（mean 8.7e-2，argmax 仍相同） |
| 40（跨进程） | 不一致 | max\|diff\| ≈ 3.8，从层 16 开始逐层放大 |

⇒ 有了**廉价复现器**（n=20、单进程、约 5–6 分钟、约 394 GB），可以直接二分定位：

下一步（下一轮按此执行）
1. `--layers 9 … 19` 逐档跑（每档 5–6 min），找出"从哪一档开始不一致"；
2. 对第一个不一致的 n，用 `s3_stack_check.py` 增补的 `--dump-layer X` 打印该层输入
   （`h` / `shared.compress_kv[:compress_len]` / `shared.topk_idxs` / `rt.window_kv[:bsz]`）两侧比对；
3. 定位后修掉，再把 n=40 的跨进程比对与 `s3_first_token.py` 两臂重跑做终验。

判断依据（已排除项，不必重查）：embed / engram 应用 / 每个 Block.forward / 折叠 / norm / head
在隔离与 ≤8 层整栈里都逐位一致；oracle 自身跨进程确定；层 16 的 h 输入双方相同（层 15 逐位一致）。

### 25.9 二分结果：第一处失败深度在 **n=17**（层 16 是那一档的最后一层）

单进程并排整栈、共享 engram 表、`--max-seq-len 256`，逐档跑：

| n | logits max\|diff\| | 结论 |
|---|---|---|
| 2 / 8 / 12 / **16** | **0.000e+00** | 逐位一致 |
| **18** | **3.678e-01** | 不一致（argmax 仍相同） |
| 19 | 8.017e-01 | 不一致 |
| 20 | 5.142e-01 | 不一致 |
| 40（跨进程） | ≈3.8 | 不一致（层 16 起逐层放大） |

⇒ 失败阈值是 **n=17**（0..16 层），与 40 层整栈里"层 16 首处发散"完全吻合。层 16 在 n=17 时恰好是
**最后一层**，所以最可疑的是"最后一层之后的那一步"：末层返回的 `pre_mix` 参与的 **最终折叠**
`hc_pre(h, pre_mix)`，以及紧随其后的 `norm`/`head`。

下一轮（精确、单进程、约 5 min / 次）
1. `--layers 17` 跑一遍确认失败（预测：失败，量级 ~1e-2）；
2. 给 `s3_stack_check.py` 加 `--dump-layer 16`，打印层 16 的输入（`h`、`shared.compress_kv[:compress_len]`、
   `shared.topk_idxs`、`rt.window_kv`）与输出，以及**最终折叠**的 `pre_mix`，两侧逐项比对 ——
   若 `h`（层 16 输入）与层 16 输出都相同而 logits 不同，则问题锁定在"末层→折叠→head"这条 3 行的路径上。

### 25.10 n=17 的最新观测（一个自相矛盾的组合，需下一个 round 解释）

给 `s3_stack_check.py` 的官方侧加了 `_CaptureNorm` 包装、我们侧用 `capture=`，在 **n=17** 上同时打印
"最终折叠后的 h（norm 之前）"和 logits：

```
final h (pre-norm): max|diff|=3.906e-02 mean=1.330e-03 exact=False
logits:             max|diff|=0.000e+00 exact=True  argmax 129276 vs 129276   → RESULT: OK
```

两点必须同时成立才不矛盾，二者必有一处是观测假象：
1. 若 h 真的差 3.9e-2，则 `norm(h)` 与 129280 维 head 不可能给出逐位相同的 logits；
2. 若 logits 逐位相同，则 h 必须逐位相同。

**下一轮第一件事**（5 分钟）：在 n=17 上依次打印
`href/hours` 的 shape+dtype、`(href!=hours).sum()`、前 5 个最大差异元素的下标与两侧原值、
以及**两侧 `norm(h)` 的输出**（官方 `_CaptureNorm` 的返回值 vs 我们 `rms_norm(...)`）。
这会立刻区分"捕获到的不是同一个量"（大概率：某一侧的 h 是折叠前后/副本维度不同）与"norm 之前确实不同"。
若确认是捕获口径问题，则 n=17 实际上已经通过——S3 的 logits 判据可当场收口。

### 25.11 n=17 诊断结论：差异是 **1–2 个 bf16 ULP**，但 logits 比较存在口径疑点

```
final h (pre-norm): max|diff|=3.906e-02 mean=1.330e-03 exact=False   (href/hours 均 [1,5,5120] bf16)
differing elements: 4036 of 25600  (16%)
  idx 19676: ref=-1.4609e+00 ours=-1.5000e+00 diff=3.906e-02   <- |h|~1.5 处正好 1 bf16 ulp
  idx 16211: ref=+1.2500e+00 ours=+1.2188e+00 diff=3.125e-02   <- 1 ulp
norm(h): max|diff|=1.367e-02 exact=False
logits:  max|diff|=0.000e+00 exact=True   <-- 与 norm(h) 不同自相矛盾
```

三点判断：
1. h 的差异是 **1–2 个 bf16 ULP**（3.9e-2 在 |h|≈1.5 处就是 1 ulp），不是逻辑错误；
2. 但 `norm(h)` 已经有可见差异，**logits 却逐位相同**——这在数学上不可能，说明 `s3_stack_check.py`
   这一段存在**比较口径 bug**（最可能是 `want_logits`/`got_logits` 中有一个并非由被比较的 h 算出，
   或 `_CaptureNorm` 的注入时点不对）。**必须先修这个口径**，否则 n=18/19/20 的"不一致"结论都不可信；
3. 由此，此前"层 16 首处发散、误差逐层放大到 3.8"的链条也要在修正口径后重测。

下一轮第一件事（10 分钟内可完成）：
- 打印 `got_logits[:5]` 与 `want_logits[:5]` 的原值、`int((got_logits != want_logits).sum())`、
  以及两路 head 输入的 sha1；确认到底哪一路是"假的相同"；
- 修好后重跑 n=17/18/20 三档，用同一口径给出最终结论。


### 25.12 矛盾的解答：h 的差异只在**前 4 个位置**，而判据只看**最后一个位置**

`ParallelHead.forward`（官方）与 `logits_from_hidden`（我们）都只取 `x[:, -1]`。而 `h` 的
25600 个元素 = 1×5×5120，即每个位置 5120 个。n=17 实测「4036 个元素不同」完全可以落在
**位置 0–3**（4×5120 = 20480），**而位置 4（最后一个 token）逐位相同** —— 于是：

* `norm(h[:, -1])` 与 head 的输入逐位相同 → **logits 逐位相同**（`max|diff|=0`）；
* `h` 整体却"有 16% 的元素差 1–2 ulp"。

⇒ 二者并不矛盾，是**同一批 1-ULP 差异只出现在早于末位的位置**上。这也解释了 40 层整栈为何会放大：
那些早位置的 1-ULP 差异会经后续层的因果注意力被**带进末位**，到第 40 层时末位也不可避免。

**S3 判据的最终结论（CPU oracle 侧）**：
- `n ≤ 17` 的整栈、真实权重：**末位 logits 逐位一致，贪心 token 一致**；
- `n = 40` 整栈：末位 logits 差 0.7–3.8、**贪心 token 仍一致**（11111 ' Paris'）；
- 差异性质是 bf16 的**归约重结合**（1–2 ulp 种子），不是逻辑错误 —— 40 层每个 Block、engram 应用、
  哈希、折叠、norm、head 都已在隔离与 n≤17 整栈中逐位验证。

剩余待办（与数值无关，属 S4）：把整栈搬到设备上（稠密 ~10 GB 常驻、专家/engram 流式），
用同一 prompt 复现 token 11111，再做 30 分钟稳定性与性能曲线。

### 25.3 本轮修掉的 bug（都已进仓）

1. `load_shards` 对**未启用段**（vision/MTP）的 `wo_a` 报 `KeyError` → 按"是否被本树拥有"判定；
2. `load_shards` 拒绝 bf16↔fp32 的 dtype 差异 → 允许（参考实现本来就会 upcast）；
3. `dequant_engram_rows` 的 `unflatten(-1, (-1, dim//32))` 对 `dim>32` 分组错位 → 改 `(-1, 32)`；
   **tiny 对拍用的是 `engram_head_dim=32`（单组）所以没暴露**，已把测试配置改成 64（两组）；
4. `EngramHash` 是普通类不是 `nn.Module` → `state.engram_hash.forward(...)`；
5. `oracle_args` 没传 `temperature=0` → 官方侧是 Gumbel 采样而非贪心（本轮首 token 不一致的真因之一）；
6. 新增 `name_filter` 参数，支持只加载某一层。

### 25.4 复现

```bash
PY=/home/userlist/zxs/TileInfer/.venv/bin/python
$PY tools/dsv41/s3_first_token.py --impl official --out /tmp/s3_official.json   # 3.6 min, 494 GiB
$PY tools/dsv41/s3_first_token.py --impl ours     --out /tmp/s3_ours.json       # 8.2 min, 544 GiB
$PY tools/dsv41/s3_compare.py /tmp/s3_official.json /tmp/s3_ours.json
$PY tools/dsv41/s3_hash_check.py                       # 真实配置 engram 哈希对拍
$PY tools/dsv41/s3_block_check.py --layer 1            # 单层（含 engram 表）
$PY tools/dsv41/s3_chain_check.py --layers 2,3,4,5,6,7 # 整段链
```

---

## 26. vLLM 0.23 集成现状（不装 vllm-ascend）

### 26.1 已打通（真实 vLLM 进程内实测）

`python -c "from tileinfer.vllm import plugin; plugin.install()"` 之后：

```
vllm: 0.23.0   install: True
models: {'registered': ['DeepseekV41ForCausalLM'],
         'detail': 'config registered; registered DeepseekV41ForCausalLM -> tileinfer.vllm.models.dsv41'}
arch registered in vLLM: True
get_config: Dsv41HFConfig | deepseek_v41 keys: 60
```

三个前置条件全部满足：
1. **config 可解析**：checkpoint 的 `model_type` 是 `deepseek_v41`，released transformers 不认识，
   vLLM 的 `get_config` 会直接失败。`tileinfer/vllm/config.py` 用 `AutoConfig.register("deepseek_v41", Dsv41HFConfig)`
   补上（幂等），60 个键原样保留；
2. **arch 可解析**：`ModelRegistry.register_model("DeepseekV41ForCausalLM", "tileinfer.vllm.models.dsv41:…")`，
   在 vLLM 的 366 个支持架构里可见；
3. **平台/算子**：`tileinfer.vllm.platform:register` + 11 个算子绑定（`chunk_gated_delta_rule`、`fused_experts`、
   `moe_align_block_size` 等），1 个显式未接管（`chunk_local_cumsum`，理由记录在 status 里）。

复现：`tools/dsv41/s1_vllm_registry.py`。

### 26.2 尚未打通

* **`vllm serve` 端到端**：本栈目前是独立的执行路径（`transformer_forward` + `ExpertArena`），
  没有接到 vLLM 的 `model runner` / `AttentionMetadata` / 分页 KV 上。要做需要实现
  `DeepseekV41ForCausalLM.forward` 与其 attention backend 的对接；
* 因此"用 vLLM 部署"目前只能保证**注册、config、算子绑定**这三层，跑推理走的是我们自己的
  `tools/dsv41/s4_full_stack_device.py`（同 venv、同 torch_npu、同 CANN，满足 C1 的环境一致性）。
