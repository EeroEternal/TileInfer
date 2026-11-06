# deep_gemm fp8×fp4 GEMM：调用成功但结果不正确（第 5 轮结论）

## 1. 已确定正确的调用契约

| 参数 | 正确形式 | 依据 |
|---|---|---|
| a | `(fp8_e4m3fn [M,K], sf fp32 [M,K/32])` | `csrc/apis/gemm.hpp:83` `"mk"=a` |
| **b** | **`(int8 [N,K/2], sf fp32 [N,K/32])`** | `csrc/utils/dtype.hpp:9` `kPackedFP4 = torch::kInt8`；`"nk"=b` |
| d | `bf16 [M,N]` | `"mn"=d`，且断言 `kBFloat16 or kFloat` |
| recipe | `(1,1,32)` | 本模型 act 是 block-32 |

环境（缺一不可）：
```bash
export LD_PRELOAD=/home/userlist/zxs/TileInfer/.work/gcc13/lib64/libstdc++.so.6
bash tools/dsv41/patch_deepgemm_headers.sh     # CANN 9.3 无 ascendc_assert
```

## 2. 当前症状（`tools/dsv41/perf_deepgemm_shapes.py` 实测）

用哨兵值 7.0 填 `d`，调用后检查是否被覆写，并与"反量化 + fp32 matmul"参考值比较：

```
M=512  N=2048  K=5120  wrote=True  rel=1.265e+00  per-iter=0.0238 ms
M=64   N=2304  K=5120  wrote=True  rel=1.336e+00  per-iter=0.0288 ms
M=256  N=2304  K=5120  wrote=True  rel=1.338e+00  per-iter=0.0392 ms
M=1    N=2304  K=5120  wrote=True  rel=1.315e+00  per-iter=0.0225 ms
M=6    N=2304  K=5120  wrote=True  rel=1.391e+00  per-iter=0.0323 ms
```

两条判据：
1. **`wrote=True`** ⇒ 不是"静默跳过"（上一轮基于全零输入的猜测已推翻）；
2. **耗时与规模无关**（FLOPs 差 1000 倍，时间都在 0.02–0.04 ms）⇒ 生成的 kernel **没有真正做 GEMM**，
   属于退化/空转内核；`rel≈1.3` 也说明输出与乘积无关。

## 3. 已排除与待查

**已排除**
- 形状契约（见上表）；
- dtype 分派（`fp8_e4m3fn` × `kPackedFP4` 已被 `dispatch_fp8_fp4_gemm` 接受，否则会 panic
  "Unsupported operand dtypes"）；
- `gemm_early_return(...)` 导致的 nullopt（若为 nullopt 则 `d` 不会被覆写，实测被覆写）；
- sf 的尾数问题（已把 sf 取成 2 的整数次幂，`e8m0` 语义；结果仍是错值）。

**待查（下一轮）**
1. `recipe` 的 `gran_m`/`gran_n`：文档说 `(gran_m, gran_n, gran_k)`，我们按 `(1,1,32)` 传；
   DeepInfer 实际传的是 `recipe=(1, 1, get_fp8_casting_granularity())`，需确认它返回多少
   （很可能是 128 而非 32）。若真实 gran_k=128，我们的 sf 列数（K/32=160）就是错的。
2. sf 的**存储类型**：文档允许 `fp32` 或 `short (packed ue8m0)`；Ascend 实现可能只走 `short` 路径，
   传 fp32 时会走一条"只写不计算"的退化分支。
3. `compiled_dims`（默认 `"nk"`）与 `disable_ue8m0_cast` 的组合。

## 4. 判断与建议

deep_gemm 这条路已经花了 4 轮：契约、环境、头文件补丁都解决，但**数值仍不对**，
且症状（时间不随规模变化）不像参数微调问题，更像 Ascend 后端的某条路径尚未支持
"fp32 sf + fp8×fp4" 组合。

建议：**下一轮再给 1 次机会**（查 `get_fp8_casting_granularity()` 的真实取值 + 试 `short` 型 sf）；
若仍不通，按原计划切到 **选项 (c)**：移植 `DeepInfer/tilelang_ops/asc/*` 的 TileLang 实现
（Python 侧编译，不依赖这套 C++ JIT 与 CANN 头版本）。

**不能把 0.02 ms 当作收益记录**——它测的是空转内核。

## 5. int16 (packed ue8m0x2) 的 sf 也无效 —— 决定性否定

DeepInfer 的 `platform.py:138` 显示 Ascend 侧 `fp8_sf_type = torch.int16`（fp32 是另一平台的取值），
于是按 e8m0 语义（字节 = 指数+127，两字节打包进 int16，K 轴折半）重做 sf，再测：

```
M=512 N=2048 K=5120  sfA(512,80)  sfB(2048,80)  wrote=True rel=1.265e+00 per=0.0367 ms
M=6   N=2304 K=5120  sfA(6,80)    sfB(2304,80)  wrote=True rel=1.391e+00 per=0.0227 ms
M=1   N=2304 K=5120  sfA(1,80)    sfB(2304,80)  wrote=True rel=1.315e+00 per=0.0235 ms
```

**关键：rel 与 fp32 版本逐位相同（1.265 / 1.391 / 1.315）**。sf 的 dtype 与布局完全不影响输出，
说明这条内核路径的输出**与输入基本无关**，属于退化执行；耗时同样与规模无关。

## 6. 结论：deep_gemm 的 fp8xfp4 在本机不可用（4 轮、6 组实验后）

已排除：形状契约、dtype 分派、early-return 静默跳过、sf 尾数、sf 存储类型(int16/fp32)、recipe 粒度(32 与
DeepInfer 的 get_fp8_casting_granularity() 一致)。

不建议继续在 deep_gemm 上投入。**按原计划切到选项 (c)**：移植
`DeepInfer/deepinfer/model_executor/layers/tilelang_ops/asc/*`（TileLang，Python 侧编译，
不依赖这套 C++ JIT 与 CANN 头文件版本）。

## 7. 建议的重排（把"能做成的"排在前面）

| 杠杆 | 预估收益 | 可行性 |
|---|---|---|
| P2 流式 + TTFT 测量 | 无加速但是**验收要求**，且能给出 prefill/decode 拆解 | 极高（vLLM 原生支持 stream） |
| H2D 预取双缓冲（88 ms） | 大 | 高（纯 Python 调度，无新 kernel） |
| MTP 投机解码 | 大 | 中（我们主动没加载 MTP，需先验证数值） |
| attention per-kernel（150 ms） | 大 | 中（移植 DeepInfer 的 asc TileLang） |
| 融合反量化 GEMM（170 ms） | 大 | **低**（deep_gemm 路径已证不通） |
