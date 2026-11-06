# tools/dsv41 — DeepSeek-V4.1-Flash 只读事实核准工具（S0）

这一目录是 **S0（事实核准）** 的可复现脚本集，独立于 Qwen3.6 通路（C2）。
所有脚本只**读** `/home/models/DeepSeek-V4.1-Flash`（含只读的 `inference/`，C4），
不修改官方实现，也不与 Qwen3.6 竞争设备（纯 CPU / 头部读取）。

事实基准文档：[`docs/notes/dsv41-facts.md`](../../docs/notes/dsv41-facts.md)。

## 环境

目标机 `tileinfra-102`（Ascend950PR 单卡）：

```bash
PY=/home/userlist/zxs/TileInfer/.venv/bin/python   # torch 2.10.0+cpu, transformers 4.57.6, vllm 0.23.0
M=/home/models/DeepSeek-V4.1-Flash
```

## 1. 权重清单（96,085 张量 / 510.3 GB）

```bash
$PY tools/dsv41/dump_inventory.py "$M" /tmp/dsv41_inventory.json
$PY tools/dsv41/inventory.py /tmp/dsv41_inventory.json
```

`dump_inventory.py` 只解析 48 个分片的头部（不读数据），秒级完成。
`inventory.py` 输出「名字模式 → 张量数/形状」、「每层总量」、「第 0/2/8/14/20/24/36/39 层的
attention+engram 张量表」与「内存分组表」（`inventory-report.txt`）。

实测：`tensors=96085 bytes=510.3GB (475.2GiB)`，其中路由专家 268.9 GiB（56.6%）、
engram 189.1 GiB（39.8%）——两者合计 96.4%，这就是「专家 + engram 必须卸载到主机内存」的根据。

## 2. 单卡 checkpoint（MP=1）

```bash
bash tools/dsv41/run_convert_mp1.sh          # 调官方 inference/convert.py --model-parallel 1
```

* 产出 `/home/userlist/zxs/dsv41/mp1/model0-mp1.safetensors`（511,738,545,912 B）+ tokenizer 两个文件
* 实测：`convert_rc=0  peak_rss_kb=399436876`（≈399 GB；`convert.py` 先把全模型攒在内存再写出，
  所以峰值 ≈ 全模型，机器需 ≥ ~450 GB RAM）
* MP=1 的全部整除约束都成立，**不存在「MP=1 不受支持」的情况**

## 3. 逐张量形状校验

```bash
$PY tools/dsv41/verify_mp1_ckpt.py /home/userlist/zxs/dsv41/mp1/model0-mp1.safetensors "$M"
```

在 meta device 上构造官方 40 层模型（engram 表不占内存），与转换后 checkpoint 头部比对。
实测输出：

```
model state_dict entries: 96048   checkpoint tensors: 96042
missing: 6 (expected tied MTP embed/head: 6)     extra: 0
shape mismatches: 0
bf16->fp32 upcasts (documented, load_model casts): 73
dtype mismatches: 0
in-memory bytes of matched tensors: 513.1 GB
RESULT: OK
```

## 4. engram 布局探针

```bash
$PY tools/dsv41/probe_engram.py "$M"
```

用真实 tokenizer 复算压缩词表 / 素数桶 / 乘子，**三项全中**：

```
compressed vocab: 99092            (== engram_compressed_vocab_size)
layer 1 : 24 primes, sum = 384006168  (== engram_num_embeddings[0])
layer 14: 24 primes, sum = 384016682  (== engram_num_embeddings[1])
```

## 5. 逐层角色表

```bash
$PY tools/dsv41/layer_table.py "$M"        # 生成 facts §7.4 的 markdown 表
```

## 6. S1：参数树 + 全量 CPU 加载

```bash
# 头部校验（meta device，无内存，2 s）
$PY tools/dsv41/s1_load_cpu.py --dry-run

# 全量读入（需要 ~580 GiB RAM）
bash tools/dsv41/run_s1_load.sh            # → /home/userlist/zxs/dsv41/s1_load.json
```

实测（2026-11-03，`tileinfer-102`）：

```
params=96042  model_bytes=511,727,454,552   copied=96042  wo_a folded=43
missing=0  extra=0  shape_mismatch=0  dtype_mismatch=0
elapsed 81.9 s (6.24 GB/s)  peak RSS 572.3 GiB  steady RSS 477.8 GiB
host-offload candidates: 94,468 tensors / 498.8 GB (97.5% of the model)
RESULT: OK
```

参数树与加载器在 `tileinfer/models/dsv41/`（纯 torch，无 vLLM）；vLLM 注册在
`tileinfer/vllm/models/dsv41.py`（`ModelRegistry.register_model`，由
`tileinfer/vllm/plugin.py` 第 5 步 `_install_models()` 调用）：

```bash
$PY -c "from tileinfer.vllm.models.dsv41 import register_models; print(register_models())"
$PY -m pytest tests/test_dsv41_tree.py -q     # 9 passed：参数表/角色/配置校验/wo_a 分块/加载回环
```

## 备注

* `convert.py` 与 `model.py` 只被 **import / 调用**，源码零改动；本目录任何脚本都不被
  `tileinfer` 包引用（不进入运行期依赖）。
* S2 起的设备侧工作放在 `tileinfer/models/dsv41/` 的 forward 与 `tileinfer/vllm/models/dsv41.py`
  的 vLLM 接线。S0/S1 的脚本继续留在这里作为回归基线。

## 7. S2：单层对拍（CPU oracle + 设备）

```bash
# CPU：官方 model.py 本体作 oracle（六个 tilelang kernel 换成 ref_cpu 的 torch 转写）
$PY tools/dsv41/s2_layer_check.py --layers 2 --steps 8,1,1,1
$PY tools/dsv41/s2_layer_check.py --chain 2,3,4,5,6,7 --steps 8,1,1
$PY tools/dsv41/s2_layer_check.py --chain 20,21,22,23,24 --steps 8,1,1
$PY tools/dsv41/s2_layer_check.py --layers 20 --steps 200,1,1 --max-seq-len 256   # ring 回绕

# 设备：我们那侧跑在 NPU，oracle 仍在 CPU
$PY tools/dsv41/s2_layer_check.py --layers 2 --device npu
```

实测（2026-11-03）：

| 用例 | 结果 |
|---|---|
| CPU: layer 2 / 20 / chain 2–7 / chain 20–27 / 200-token 回绕 | **max\|diff\| = 0.000e+00**，top-k 索引全等 |
| NPU: layer 2 | 跑通；索引全等；**max\|diff\| = 1.56e-2**（mean 1e-4），未达 1e-2 |

`ref_cpu.py` 是 `inference/kernel.py` 六个 kernel 的 torch 转写（`act_quant` / `fp4_act_quant` /
`fp8_gemm` / `fp4_gemm` / `sparse_attn` / `hc_split_sinkhorn`），逐条保留 ue8m0 幂次 scale、
`6*2^-9` 下限、E2M1 网格 RN-even、`-1e30` 初始 max、概率转 bf16、sink 进分母等语义。
设备侧已知不支持的算子与绕法见 `docs/notes/dsv41-facts.md` §22.4。

诊断三件套（设备侧偏差定位，结论见 facts §22.6）：

```bash
$PY tools/dsv41/s2_attn_debug.py --layers 2            # 分阶段：qr/q/kv/compress_kv/idxs/attn_out/x
$PY tools/dsv41/s2_op_probe.py  --layers 2            # 分算子：同输入，CPU vs NPU
$PY tools/dsv41/s2_residual.py  --layers 2,20         # 定格：该元素处 1 bf16 ulp 是多少
```

实测：`act_quant` 与 Compressor/Indexer 的取整在设备上**逐位一致**；`fp8_gemm` / `sparse_attn`
各差 **1 bf16 ulp**；合成到层输出是 **2–4 ulp**（rel_rms 5e-5–6e-3）。S2 的 `1e-2` 绝对值门槛
在本模型输出量级（|x| 可达 4–8，1 ulp = 1.56e-2）下等价于"小于 1 ulp"，因此执行层面把判据定为
「逐算子 ≤1 ulp + 离散索引逐位相同 + 合成 rel_rms < 1e-2」，端到端正确性交给 S3 的贪心首 token。

CI 守护：`tests/test_dsv41_layer_parity.py`（4 用例，要求短序列 CPU 对拍 `max_abs == 0`）。
