# M1 收尾：v2 已归档到 `legacy/v2/`

**状态（2026-11-16）：已完成。** `tileinfer/vllm/` 整体移动到
[`legacy/v2/tileinfer/vllm/`](../../legacy/v2/tileinfer/vllm/)，v2 不再属于安装包，也不再是
可 import 的路径。

## 1. 移动了什么

| 原路径 | 新路径 | 说明 |
|---|---|---|
| `tileinfer/vllm/` | `legacy/v2/tileinfer/vllm/` | v2 平台/worker/attn/ops/profiles/plugin 等 52 个文件 |
| `tests/test_*.py`（34 个 + `fake_vllm.py` + `test_native_prefill_oracle.py`） | `legacy/v2/tests/` | 只测 v2 的契约/内核/DSV41 用例 |
| `scripts/serve_tileinfer.py`、`serve_qwen36.py`、`install-native-backend.sh`、两个 `.example` | `legacy/v2/scripts/` | v2 启动器与安装脚本 |
| `tools/dsv41/{perf_spec_parity,s1_vllm_registry}.py` | `legacy/v2/tools/dsv41/` | 两个 import v2 的脚本；`tools/dsv41` 其余部分留在主树，因为 v3 的 DSV41 parity 测试要 import 它 |
| `benchmarks/` 中 17 个 import v2 的脚本 | `legacy/v2/benchmarks/` | 其余 v3/通用 benchmark 留在原处 |

## 2. 现在哪些事成立

- 安装包只包含 `tileinfer*`（`pyproject.toml`），所以 `legacy/` 不会被打包；
- `import tileinfer.vllm` 现在失败（`ModuleNotFoundError`），这是预期行为；
- `pytest tests` 不再收集 v2 用例；v2 用例只在 `legacy/v2/tests/` 里，作为历史参考；
- v3 live 路径（`tileinfer/{cli,bridge,resolve,contracts,manifests,providers}`）不 import
  `tileinfer.vllm`，并且有测试守着：
  - `tests/test_v3_impl_table.py::test_bridge_does_not_import_the_v2_operator_installer`
  - `tests/test_v3_impl_table.py::test_v3_provider_sources_do_not_import_the_v2_tree`
  - `tests/test_v3_contracts.py::test_v2_package_is_not_importable`
  - `tests/test_v3_contracts.py::test_pyproject_registers_only_the_general_plugin`

## 3. 为什么是“归档”而不是“删除”

- v2 是 A/B 参照物：它能跑，用来回答“厂商在这条形状上是真不行，还是只是慢”；
- 旧部署仍可能需要读它；删除会让历史对照彻底消失；
- 归档而不是原地冻结，是因为原地保留会让 `tileinfer.vllm` 继续可 import，容易被误用
  （之前 Muse 就是靠手工恢复 v2 入口跑起来的）。

归档是**源码级**的：`legacy/v2` 不进入 `sys.path`，也不应该被加进去。要看某个 v2 文件，
直接读 `legacy/v2/...`；要跑旧的 v2 行为，用 move 之前的提交 checkout 一个独立工作区。

## 4. v3 里还保留什么

v2 的**能力**已经迁到 v3（不是 v2 目录）：

- 模型实现：`tileinfer/bridge/models/`（MuseGlimmer + ATEM parser）；
- 算子实现：`tileinfer/providers/`（Mamba2、MoE、OOT MoE method）；
- 安装逻辑：`tileinfer/bridge/impl_table.py` + `tileinfer/bridge/seams.py`；
- 声明：`tileinfer/manifests/`（Qwen3.6、MuseGlimmer、Nemotron draft）。

## 5. 入口点账本

`pyproject.toml` 只注册一个入口：

```toml
[project.entry-points."vllm.general_plugins"]
tileinfer = "tileinfer.bridge.install:install"
```

`vllm.platform_plugins` 下没有 TileInfer 条目；`.venv_v3` 里该组只剩
`ascend -> vllm_ascend:register`。平台槽只有一个主人，这是 v3 与 v2 最硬的一条分界。

## 6. 后续

- 不再在 `legacy/v2/` 里加功能或修 bug；
- 如果某个 v2 能力被证实厂商确实不可用，把它作为 provider 补进 `tileinfer/providers/`
  并留证据，而不是复活 v2；
- 如果确认不再需要 A/B 参照，可以整目录删除 `legacy/v2/`，但先确认没有外部脚本引用它。
