# deep_gemm（DeepGEMM-Ascend）可用性核查

> 记录于性能优化目标的第 2 轮。相关代码：`tools/dsv41/perf_deepgemm_probe.py`。

## 1. 发现

DeepInfer 的 `deepinfer/model_executor/layers/linear.py` 里**无条件**调用：

```python
deep_gemm.fp8_fp4_gemm_nt(x, weight, out, out_acc, recipe=(1, 1, get_fp8_casting_granularity()))
```

而 `deep_gemm 0.1.1`（`deep_gemm 0.1.1+3a35326`）**已经装在我们的 venv 里**。它实际上是 Ascend 移植版
（报错栈中的路径为 `/tmp/opencode/DeepGEMM-Ascend/csrc/...`），因此这正是 P1 里"融合反量化 GEMM"的现成实现。

## 2. 环境坑（已解决）

直接 `import deep_gemm` 失败：

```
ImportError: /usr/lib64/libstdc++.so.6: version GLIBCXX_3.4.31 not found
```

系统 libstdc++ 是 6.0.28（对应 GLIBCXX_3.4.28），太旧。机上有更新的一份：`.work/gcc13/lib64/libstdc++.so.6`
（GCC 13，含 3.4.31/3.4.32）。实测：

```bash
export LD_PRELOAD=/home/userlist/zxs/TileInfer/.work/gcc13/lib64/libstdc++.so.6
.venv/bin/python -c "import torch, torch_npu; import deep_gemm; print(deep_gemm.__version__, hasattr(deep_gemm,'fp8_fp4_gemm_nt'))"
# -> torch ok 2.10.0+cpu
# -> deep_gemm ok 0.1.1   fp8_fp4_gemm_nt: True
```

**注意**：不能用 `LD_LIBRARY_PATH` 指向该目录——实测会把 `torch_npu` 的 backend 扩展加载搞坏
（`RuntimeError: Failed to load the backend extension: torch_npu`）。必须只 `LD_PRELOAD` 单个 libstdc++。

## 3. 尚未解决：参数布局契约

`tools/dsv41/perf_deepgemm_probe.py` 在 M=8 / N=256 / K=256（act block-32）的用例上调用内核。
sfA 的 `[M,K32]` / `[K32,M]` 与 sfB 的 `[N,K32]` / `[K32,N]` 四种组合，配合 recipe `(1,1,32)` /
`(1,1,16)`，**八种组合全部被拒**：

```
Panic error (csrc/utils/layout.hpp:196): Conflicting shape dimension 'k'
```

从 `csrc/utils/layout.hpp` 读到关键规则：

```cpp
if (tensor.scalar_type() == kPackedFP4 && axis == major_axes[tensor_idx])
    value *= 2;      // packed fp4 只在“主序轴”上把逻辑维度 ×2
```

即 packed-FP4 的 K 只有在**主轴**上才被视为两倍。我们传 `[N, K/2]` 的 uint8 再
`.view(torch.float4_e2m1fn_x2)` 后形状变成 3 维（实测打印 `(256, 8, 16)`，因为 `.view` 对小 dtype 会拆分末轴），
主序轴判定因此失败。

## 4. 下一轮做法（纯读源码，不占卡）

读 `deep_gemm/csrc/apis/` 中 `fp8_fp4_gemm_nt` 的绑定实现，取出它对 A / B / sfA / sfB 的精确
annotation（`mk` / `nk` / `kn` 等）与打包要求，据此修正 probe 的张量构造；通过后即可做
"融合 kernel vs 我们现有 unpack+scale+bmm"的数值与耗时对比（当前那条路径是 0.80 + 0.42 + 0.18 ms/层/GEMM）。

## 5. 对本目标的意义

- P1 的"融合反量化 GEMM"**不需要从零写 kernel**：DeepSeek 的 Ascend 实现已经在环境里，只差调用契约；
- 代价是运行期必须带 `LD_PRELOAD`（可写进启动脚本，不影响 C1 的"两臂同环境"——两臂都带即可）；
- 若最终该 kernel 仍不可用，退路是读 `DeepInfer/tilelang_ops/asc/*.py`（约 7418 行）自行移植。
