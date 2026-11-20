# 开发规范（v3 模型适配与 Provider）

> 本文件是**规范性文档**，描述新增模型、provider、bridge 代码、证据和文档必须遵守的规则。
> [`extending.md`](extending.md) 是操作指南；两者冲突时，以本文件为准。

## 1. 分层与 import 边界

| 层 | 目录 | 允许 import | 禁止 |
|---|---|---|---|
| L1 声明 | `tileinfer/manifests/` | `contracts` | vLLM / vllm-ascend |
| L1 engine-free 辅助 | `tileinfer/models/` | `contracts`、标准库、torch（纯数学） | vLLM / vllm-ascend |
| L2 决策 | `tileinfer/resolve/` | `contracts`、`manifests`、`providers` 的声明 | 引擎、provider 实现 |
| L3 契约 | `tileinfer/contracts/` | 标准库 | 引擎、模型实现 |
| L4 Provider 声明/实现 | `tileinfer/providers/` | `contracts`、torch | vLLM / vllm-ascend |
| L5 Bridge / 模型 payload | `tileinfer/bridge/` | vLLM、vllm-ascend、providers、models | 被 L1–L4 import |

硬规则：

- **只有 `tileinfer/bridge/` 可以 import vLLM / vllm-ascend。**
- 会 import vLLM 的模型类放在 `tileinfer/bridge/models/`，不要放回 `tileinfer/models/`。
- `tileinfer/models/` 只放 engine-free 的配置、布局、兼容和纯数学代码。
- `legacy/v2/` 是历史归档，不新增功能、不新增测试、不 import。
- 新代码不得引入 `tileinfer.vllm` 路径。

## 2. 模型适配的唯一流程

```text
manifest 声明
  → resolve.decide 生成 Plan
  → bridge 读取 Plan
  → ModelAdapters 统一注册
  → 对应 seam 安装 provider / 模型 / parser
  → doctor / C1–C4 验证
```

模型事实只写在 manifest，不要写在 bridge、provider 或测试代码里。

### 2.1 Manifest 必须表达的事实

- `name` / `manifest_id` / `arch` / `model_type`；
- `num_layers` / `layer_types`；
- attention 几何、sliding window、RoPE 参数；
- `linear` / `state` / `mamba` / `moe` 等 mixer；
- 必需 capability；
- benchmark 样本；
- `model_impl` / `config_impl` / parser；
- `support`（SupportSpec）；
- `parallel`（已验证的并行轴）；
- `hooks`（需要的 CompatHook 名称）。

新增模型必须提供 `verify_manifest()` 能通过的 checkpoint 校验；没有真实 config 的模型只能算草案。

### 2.2 `model_impl`

- 模块必须提供 `register_config()` 和 `register_models()`；
- 只做注册，不写业务逻辑；
- 模型类放在 `bridge/models/<model>_model.py` 或 `bridge/models/<model>/model.py`；
- `model_impl` 为空表示模型由 vendor 提供。

### 2.3 `config_impl`

- 指向 `module:actor`，默认 actor 为 `register`；
- 在 `ModelConfig` 创建之前执行；
- 用于未知 `model_type`、嵌套 text/vision config、layer type / rope 规范化；
- 失败必须 fail closed，不允许带着未初始化 config 继续启动。

### 2.4 Parser

- `reasoning_parser` 是 CLI 名称；
- `reasoning_parser_impl` 是 `module:Class`；
- parser 注册属于 ModelAdapters，不要散落在模型文件里。

## 3. ModelAdapters：唯一模型注册流水线

文件：`tileinfer/bridge/model_adapters.py`

固定顺序：

```text
config adapter
  → model.register_config
  → model.register_models
  → parser.register
```

规则：

- 每一步返回 `InstallResult`；
- 任何一步失败，整个 adapters 结果为 `ok=False`；
- 结果缓存，`install()` 和 `model_table` seam 共用，不允许重复注册；
- actor 返回 `{"registered": False}` / `{"ok": False}` 必须视为失败；
- 新模型只实现 adapter 需要的 actor，不要自己写一套 import/异常处理循环。

## 4. Provider 统一注册表

文件：`tileinfer/providers/registry.py`

所有 provider 查询必须通过：

```text
find_provider(name)
providers_for(capability)
records_for(capability)
global_provider(capability)
route_providers(capability)
candidate_providers(capability)
```

禁止：

- 在 `decide`、`doctor`、`impl_table` 里各自拼表；
- 在 seam 里按 provider 名字写 `if name == "tileinfer.short_conv"`；
- 新增 provider 只加到某一张表而忘记 registry / plan / doctor。

四类来源仍然存在，但只由 registry 解释：

```text
vendor    resolve/vendor.py（兼容重导出 providers/vendor.py）
global    providers/global_.py
route_o   providers/catalog.py
candidate providers/nemotron/
```

## 5. Provider 声明规范

`ProviderSpec` 必须表达：

| 字段 | 规则 |
|---|---|
| `capability` | 来自封闭的 `Capability` 枚举 |
| `name` | 全局唯一，`tileinfer.*` |
| `impl` | 实际实现路径 |
| `covers` | `mixer` / `model_type` / shape 等匹配条件 |
| `status` | `default` 或 `candidate` |
| `scope` | `global` 或 `model` |
| `install_actions` | 一个或多个安装动作 |
| `parallel` | 已验证的并行轴 |
| `graphable` | 声明“形状上可图” |
| `graph_safe` | 实测图捕获/replay/无 host 同步 |
| `evidence_level` | `declared/candidate/cpu/device-c1..c4/verified` |
| `evidence` / `delete_when` | 必须有；没有删除条件的临时实现不允许合入 |

### 5.1 安装动作

新 provider 必须使用 `install_actions`：

```python
InstallAction("impl_bind", actor="module:function")
InstallAction("vendor_patch", hook="compat_hook_name")
InstallAction("model_call")
```

- `impl_bind`：把实现绑定到 vLLM 算子/方法表；
- `vendor_patch`：安装一个已在 `compat_hooks.py` 注册的 CompatHook；
- `model_call`：provider 由我们的模型实现内部消费，不安装到算子表。

`install` / `install_actor` 是旧兼容字段；新代码不要使用，也不要再增加 `PROVIDER_BINDERS` 之类的第二张表。

### 5.2 status / scope

- `status=candidate` 只能通过显式 bring-up 进入计划，计划必须保持 `servable=false`；
- `status=default` 才能在正常计划中被选中；
- `scope=model` 只对声明了 `model_impl` 的模型生效；
- 不允许把模型作用域 provider 放进全局默认表。

## 6. 安装结果与失败语义

所有安装动作统一返回 `tileinfer/bridge/results.py::InstallResult`：

```text
ok / installed / skipped / failed / detail
```

规则：

- 任何 failed 非空 → `ok=False`；
- 不允许“捕获异常后返回 registered=False，但外层仍报 OK”；
- `install()`、`model_table`、`impl_table`、config adapter、compat hook 使用同一结果类型；
- provider 依赖的 hook 被禁用时，必须 fail closed；
- 安装是幂等的；同一进程内重复执行不能重复注册或重复报错。

## 7. Plan 是唯一安装依据

`Plan` 必须记录：

- `selections`：每个 capability 的实现；
- `access_points`：粗粒度接入点；
- `mechanisms`：实际注入类别（config_adapter / model_register / parser_register / impl_bind / model_call / vendor_patch 等）；
- `hooks`：本次计划涉及的 CompatHook 名称；
- `parallel`：构建计划时的并行轴；
- `servable`：是否可部署。

规则：

- `install()` 只能按 Plan 安装，不能“按模型名猜”；
- `install()` 和 `doctor` 必须共用 `bridge/runtime_gate.py` 的版本/并行/provider 适用性检查；
- 单卡计划不得注入 TP/EP/DP/PP 运行；
- `support`、`mechanisms`、`hooks` 是审计元数据，不进入 plan id；`parallel` 和 `config_impl` 等部署事实进入 plan id。

## 8. 支持范围（SupportSpec）

`manifest.support` 必须是 `SupportSpec`，状态只允许：

```text
verified / partial / failed / not_applicable / not_implemented / not_verified
```

- `text` / `vision` / `audio` / `mtp` 用上述状态；
- `quant`、`tp`、`max_context_verified` 必须写实际验证值；
- 不允许把 C1 成功写成 `verified`；
- C2 未做、C3 未测、C4 未测必须分别标明。

## 9. 共享工具

### 9.1 WeightMapSpec

`tileinfer/bridge/weight_map.py`

新模型优先声明：

- `renames`；
- `stacked`（qkv / w13 / gate_up）；
- `skip_prefixes` / `skip_substrings`；
- `expert_remap`；
- `fp32_prefixes`。

不要在模型 `load_weights()` 里重复写遍历循环。

### 9.2 StateLayout

`tileinfer/models/state_layout.py`

- 逻辑 state shape、dtype、Ascend 页对齐、padded shape 必须从这里计算；
- 不允许每个模型自己复制 `_padded_state_len`；
- 图捕获安全更新由模型/provider 的 `graph_safe` 证据负责，不能只靠声明。

### 9.3 CompatHook

`tileinfer/bridge/compat_hooks.py`

每个兼容补丁必须声明：

```text
name / kind / target / installer
disable_env / delete_when / evidence
```

- `kind`：`rope_patch` / `platform_guard` / `monkeypatch` / `model_side_numeric_fix`；
- `disable_env` 必须可关闭；
- `delete_when` 必须写上游修复条件；
- 不允许无登记地 monkeypatch 全局对象。

## 10. 目录与命名约定

新模型建议使用：

```text
tileinfer/manifests/<model>.py
tileinfer/bridge/models/<model>.py          # register_config / register_models
tileinfer/bridge/models/<model>_model.py    # vLLM 模型，按需
tileinfer/models/<model>/config.py          # engine-free config/compat
tileinfer/models/<model>/state.py           # 可选纯数学
```

当前 Muse/LFM3/Nemotron 的布局尚未完全统一；新代码必须按上述目标布局，不再新增单文件混合 config/model/parser 的模式。

## 11. 新模型接入检查清单

- [ ] manifest 通过 `validate()`；
- [ ] `verify_manifest()` 与真实 `config.json` 一致；
- [ ] `model_impl` 的 actors 存在且幂等；
- [ ] `config_impl`（如需要）存在且失败 fail closed；
- [ ] provider 使用 registry + install_actions 声明；
- [ ] `Plan.mechanisms` / `Plan.hooks` 能解释所有实际注入；
- [ ] `SupportSpec` 写清 text/vision/mtp/quant/tp/context；
- [ ] C1 真权重通过；
- [ ] C2 有参考/top-5/容差；
- [ ] C3 图捕获和 fallback 计数；
- [ ] C4 有 TTFT/TPOT/吞吐和门禁；
- [ ] doctor PASS；
- [ ] 证据文档、plan id、README/status 更新。

## 12. 禁止事项

- 禁止在 bridge 里按模型名/provider 名写特例；
- 禁止 provider 没有 `evidence` 或 `delete_when`；
- 禁止把 vendor 已能正确提供的实现写成 provider；
- 禁止不同模块各自定义“安装成功”；
- 禁止 plan 外的注入（config adapter、vendor patch 必须可审计）；
- 禁止开发者本机路径进入跟踪文件；
- 禁止在 C2/C4 未闭合时宣称“支持”。

## 13. 验证命令

CPU 契约测试：

```bash
python -m pytest tests/test_p0_adaptation.py tests/test_p1_adaptation.py \
  tests/test_p2_architecture.py tests/test_v3_contracts.py \
  tests/test_v3_impl_table.py tests/test_plan.py tests/test_parallel.py \
  tests/test_manifest_checkpoint_verifier.py -q -m 'not npu and not tilelang'
```

设备测试：按 `verification-matrix.md` 跑 C1–C4；Muse/LFM3/Nemotron 的 C1 必须使用真实权重和对应 transformers/CANN 环境。
