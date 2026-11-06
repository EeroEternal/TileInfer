# deep_gemm JIT 编译受阻：`ascendc_assert` 未定义（CANN 版本不匹配）

> 性能优化目标第 3 轮记录。前置：`dsv41-perf-deepgemm-note.md`（LD_PRELOAD 方案 + 形状契约）。

## 1. 形状契约已解决

`deep_gemm/csrc/apis/gemm.hpp:253` 定义：

```cpp
get_shape_by_spec<"mnk">("mk"_sp = a, "nk"_sp = b, "mn"_sp = d);
```

实测结论（枚举了 6 种组合）：

| B 的传法 | 结果 |
|---|---|
| `uint8 [N, K/2]` | ❌ `Conflicting shape dimension 'k'` |
| `float4_e2m1fn_x2` `[N, K/2]` | ❌ 同上（K 轴被当成 128，未 ×2） |
| **`float4_e2m1fn_x2` `[N, K]`（逻辑形状）** | ✅ **形状检查通过**，进入 JIT 编译 |

即 packed-FP4 的 B 要按**逻辑 [N, K]** 传，dtype 用 `float4_e2m1fn_x2`（该 dtype 元素即含 2 个 fp4 值），
物理字节数自然是 `N*K/2`。sfA/sfB 按 `[M, K/gran]`、`[N, K/gran]` fp32 传即可，recipe `(1,1,32)`。

## 2. 新阻塞：JIT 编译失败（环境/版本问题，非内核逻辑）

```
deep_gemm/include/deep_gemm/transform_sf.hpp:19:5: error: use of undeclared identifier 'ascendc_assert'
```

- deep_gemm 的 CUDA-free Ascend 头文件里用了 `ascendc_assert(cond, fmt, args...)`；
- 当前 `ASCEND_HOME_PATH=/home/dlb/Ascend/install/cann-9.3.0`（**别人的 CANN 9.3.0**，我们的 torch_npu 2.10 就是按它装的），
  其 `asc/include/c_api/*.h` 里**没有**这个宏（grep 为空）；
- 换成我们自己的 `/usr/local/Ascend/ascend-toolkit/latest` 则 `torch_npu` 直接起不来
  （`LazySetDevice ... error code 507033`），所以不能整体换 CANN。

⇒ deep_gemm 0.1.1 的头文件是面向另一个（更新的）CANN 版本写的，与本机 9.3.0 不匹配。

## 3. 三条可选出路（下一轮按 a → b → c 尝试）

**(a) 给 JIT 注入宏定义。** 若 deep_jit 支持额外编译 flag（检查 `deep_gemm/envs.py` 与
`deep_jit` 的 flag 环境变量），加

```
-D'ascendc_assert(...)='
```

直接把该断言编译掉（它只用于校验 sf 的 fp32 尾数，我们传的 sf 是自己生成的，可控）。

**(b) 在头文件里打补丁。** 在 `deep_gemm/include/deep_gemm/transform_sf.hpp` 顶部插入
`#ifndef ascendc_assert` 回退定义（映射到 `assert` 或空实现）。属于改第三方包，需注意：
两臂都带上同一补丁才满足 C1；且补丁要写进文档与脚本，避免"手工改动未留痕"。

**(c) 放弃 deep_gemm，改用 DeepInfer 的 TileLang 实现。**
`DeepInfer/deepinfer/model_executor/layers/tilelang_ops/asc/`（约 7418 行，含 `compress.py`、
`get_kv_index.py`、`candidate_blocks.py`、`kv_quant.py` 等）。TileLang 走 Python 侧编译，
不依赖 C++ JIT 与 CANN 头文件版本，**风险最低**，但需要更多移植工作。

## 4. 本轮判定

- deep_gemm 的 fp8×fp4 融合 GEMM **在能力上存在且形状契约已解**，目前仅被 CANN 头文件版本卡住；
- 它是 P1 里收益最大的一项（现状 unpack 0.80 + scale 0.42 + bmm 0.18 ms/层/GEMM），值得再花 1–2 轮；
- 若 (a)/(b) 都不通，立刻切到 (c)，不要在环境上继续纠缠。
