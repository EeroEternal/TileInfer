# legacy/ — 冻结快照

这里存放**不在主树、不参与构建**的历史实现，按版本分目录：

| 目录 | 内容 | 状态 |
|---|---|---|
| `legacy/`（本层） | v1：旧 vLLM-Ascend 适配器、补丁、内核模块、测试、脚本 | 冻结，不再改动 |
| `legacy/v2/` | v2：`tileinfer/vllm/` 整棵后端树 + 只测 v2 的测试/脚本/benchmark + 两个 import v2 的 DSV41 脚本 | 2026-11-16 归档，源码级参考 |

v2 的说明见 [`v2/README.md`](v2/README.md)。v3 的规则见
[`docs/development-rules.md`](../docs/development-rules.md)。

## 为什么留着

1. **A/B 参考**：v1/v2 都能跑，用来回答“厂商在这条形状上是真不行，还是只是慢”。
2. **历史对照**：性能与数值回归需要旧实现做基线，删掉就失去对照。

## 边界

* **不打包**：`pyproject.toml` 只包含 `tileinfer*`，`pip install tileinfer` 不会带上 `legacy/`。
* **不测试**：不参与 `pytest tests`，CI 也不跑。
* **不回收**：新代码不得 `import legacy.*`，也不得再往这里加东西——它只会缩小，不会长大。
  主树的守卫（`tests/test_layer_deps.py`、`tests/test_v3_*`）保证 v3 live 路径不引用 v2。

## 什么时候删

确认不再需要 A/B 对照、且没有外部脚本引用 `legacy/` 之后，整个目录可以直接删除——它没有
主树依赖，删掉不需要改 v3 代码。
