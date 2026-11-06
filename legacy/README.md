# legacy/ — 冻结快照

这里是从主树搬出来的 vLLM-Ascend 专用代码：旧适配器 `tileinfer/integrations/vllm_ascend.py`、
建在它内部结构上的 `install_*` 补丁、被它调用的内核模块（GDN small-ops / MoE weight-prefetch /
greedy 采样）、对应的一批测试，以及若干脚本 / `benchmarks/probes` / 示例。目录结构按仓库相对路径
保留（`legacy/tests/...` 等），搬过来之后**不再改动**。

## 为什么留着

1. **生产还在跑它。** 现网（vLLM 0.23 + vllm-ascend 0.23 (950)）通过发布分支上的这套代码提供服务，
   主树的新后端（`tileinfer/vllm/` + `tileinfer/serving/`）在硬件验证通过前不是交付路径。
2. **它是 A/B 参考。** 性能与数值回归要拿现网实现做对照，删掉就失去基线。

## 边界

* **不打包**：不在 `pyproject.toml` 的包列表里，`pip install tileinfer` 不会带上它。
* **不测试**：不参与 `pytest tests`，CI 也不跑；里面的用例只在需要复现现网行为时手动执行。
* **不回收**：新代码不得 `import legacy.*`，也不得再往这里加东西——它只会缩小，不会长大。
  主树的守卫（`tests/test_no_vllm_ascend.py`、`tests/test_layer_deps.py`）保证 `legacy/` 之外
  没有任何 vllm_ascend 引用。

## 什么时候删

硬件验证完成、新后端在目标机型上达到或超过现网（吞吐 / 数值 / 稳定性），并且发布分支切到新后端
之后，整个目录可以直接删除——它没有任何主树依赖，删掉不需要改代码。
