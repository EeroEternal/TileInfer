# 结构清理计划

> 状态：建议稿，未开始大规模迁移。
> 目标：把 v3 适配层与早期 kernel/服务研究线的边界整理清楚，让目录和命名自己说明职责。
> 当前已完成的局部清理：删除 `resolve/vendor.py`、`providers/mamba2.py` 两个兼容重导出；provider registry 增加 `vendor_provider()` 和 status/selectable 过滤。

## 1. 结论

L1–L5 的分层逻辑是成立的：

```text
manifests -> resolve -> contracts -> providers -> bridge -> engine
```

现在的“乱”主要不是决策逻辑乱，而是三个历史问题：

1. **两个产品共用一个包**：v3 适配层和早期 kernel/服务线都在 `tileinfer/` 下，`tileinfer/__init__.py` 还导出早期 kernel API。
2. **route-O 模型布局不统一**：MuseGlimmer 单文件、LFM3 注册/实现分离、Nemotron 的 config/moe 直接放在 bridge 根目录。
3. **provider 声明和实现混放**：`providers/` 下同时有数据表、纯 torch 实现、vLLM AOT 适配器和统一 registry。

这些问题不影响当前 LFM3 运行，但会增加新人理解成本，并让后续模型接入继续堆积历史风格。

---

## 2. 当前结构中的具体问题

### P0-1：包入口把两个产品混在一起

`tileinfer/__init__.py` 当前导出：

```python
from .attention import BatchAttention
from .metadata import AttentionMode, PageTable, RaggedMetadata
from .plan import AttentionPlan, TileSchedule, WorkspaceManager
```

这会导致：

```python
import tileinfer.bridge
```

先加载 `attention`、`plan`、`metadata` 和 torch 相关代码。v3 bridge 不需要这些，而且：

- `tileinfer.plan` 与 `tileinfer.contracts.plan` 同名；
- 新人 `import tileinfer` 看到的是早期 kernel API，不是 v3；
- v3 runtime 的 import 链路被 kernel 线污染。

### P0-2：route-O 模型布局不统一

当前：

```text
bridge/models/lfm3.py
bridge/models/lfm3_model.py
bridge/models/muse_glimmer.py
bridge/models/atem.py
bridge/nemotron_config.py
bridge/nemotron_moe.py
```

三种风格：

- LFM3：注册模块 + 模型实现分离；
- MuseGlimmer：注册、config shim、模型实现全在一个文件；
- Nemotron：config/moe 直接放在 bridge 根目录。

`atem` 是 chat/reasoning parser，不是模型，却放在 `bridge/models/`。

### P0-3：providers 声明和实现混放

当前：

```text
providers/vendor.py           # 声明
providers/global_.py          # 声明
providers/catalog.py          # 声明
providers/nemotron/__init__   # 声明
providers/state/short_conv.py # 实现
providers/moe.py              # 实现
providers/fused_moe_oot.py    # vLLM AOT 适配器
providers/registry.py         # 统一查询
```

目录层级无法区分“数据表”和“实现代码”；`global_.py` 的命名也不直观。

### P1-1：L4 反向依赖 L5/服务层

`providers/moe.py` 内部：

```python
from tileinfer.serving import flags
```

L4 是纯 provider，`serving` 是上层服务逻辑。即使是函数内 import，也属于反向依赖。

### P1-2：`models/` 与 `bridge/models/` 边界模糊

当前：

```text
models/dsv41/           # 纯 torch 研究模型
models/lfm3/compat.py   # transformers shim
models/state_layout.py  # 状态页布局工具
bridge/models/          # vLLM 模型实现
```

`models/lfm3/compat.py` 不是模型结构；`state_layout.py` 也不是模型，而是状态布局工具。

### P1-3：`fused_moe_oot.py` 目前没有接线

`providers/fused_moe_oot.py` 是完整的 OOT MoE 适配器，但当前没有 `ProviderSpec` 或 bridge 引用，属于悬空实现。需要决定：

- 接入 `NEMOTRON_MOE` 的安装动作；
- 或归档到 `legacy/`。

### P1-4：空目录/占位目录

```text
tileinfer/aot/       # 只有 __init__.py
tileinfer/weights/   # 只有 __init__.py
tileinfer/observ/    # 只有 opaudit 等少量内容
```

会增加“目录很多但不知道干什么”的观感。

---

## 3. 推荐目标结构

```text
tileinfer/
  __init__.py              # 只保留 __version__，旧 API 走 kernel_api
  contracts/               # L3
  manifests/               # L1
  resolve/                 # L2
  providers/               # L4
    declarations/
      vendor.py
      defaults.py          # 原 global_.py
      route_o.py           # 原 catalog.py
      candidates.py
    implementations/
      state/
        short_conv.py
        layout.py          # 原 models/state_layout.py
      nemotron/
        mamba2.py
      moe/
      fused_moe_oot.py
    registry.py
  bridge/                  # L5
    install/
      install.py
      seams.py
      impl_table.py
      results.py
      runtime_gate.py
      model_adapters.py
    compat/
      compat_hooks.py
      config_adapter.py
    route_o/
      lfm3/
        __init__.py        # register_config / register_models
        model.py
        compat.py
      muse_glimmer/
        __init__.py
        model.py
        config.py
      nemotron/
        __init__.py
        config.py
        moe.py
    parsers/
      atem.py
    metrics.py
  kernels/                 # L6
  kernel_api/              # 原 attention/plan/metadata
    attention/
    plan.py
    metadata.py
  serving/                 # 早期服务线
  testing/
  legacy/
```

`models/dsv41` 如果是独立研究线，建议：

```text
tileinfer/experimental/dsv41/
```

不要继续和 v3 的 `models` 混放。

---

## 4. 迁移顺序

不要一次性大搬家。按依赖顺序分五步。

### 第一步：冻结包入口

- `tileinfer/__init__.py` 只保留 `__version__`；
- 旧 API 移到 `tileinfer.kernel_api`；
- 暂时保留兼容 alias，并标记 deprecated；
- 验证 `import tileinfer.bridge` 不再加载 `attention/plan/metadata`。

### 第二步：抽 flags

- `tileinfer/serving/flags.py` -> `tileinfer/common/flags.py`；
- `providers/moe.py` 改为依赖 common flags；
- 去掉 L4 -> serving 的反向依赖。

### 第三步：统一 route-O

- 建 `bridge/route_o/`；
- 先迁 LFM3：

```text
bridge/route_o/lfm3/__init__.py
bridge/route_o/lfm3/model.py
bridge/route_o/lfm3/compat.py
```

- 再迁 MuseGlimmer；
- 再迁 Nemotron 的 config/moe；
- 更新 manifest 的 `model_impl`、`reasoning_parser_impl`；
- 重新生成 plan，更新 evidence 中的 plan id。

### 第四步：整理 providers

- `declarations/` + `implementations/` 分目录；
- `global_.py` 改名 `defaults.py`；
- `catalog.py` 改名 `route_o.py`；
- `fused_moe_oot.py` 决定接线或归档；
- 保持 `registry.py` 是唯一查询入口。

### 第五步：清理空目录和旧服务线

- 删除 `aot/`、`weights/`；
- `observ/opaudit.py` 如果只服务 kernel 线，移到 `testing/` 或 `tools/`；
- `integrations/` 如果没有 v3 入口，归到 kernel/serving 线；
- `models/dsv41` 如需保留，移到 `experimental/`。

---

## 5. 合理、不建议乱动的部分

以下结构是合理的，不要为了“整齐”重写：

- `contracts/` 的数据契约；
- `manifests/` 的声明与 registry；
- `resolve/` 的纯函数决策；
- `providers/registry.py` 的统一查询方向；
- `bridge/install`、`seams`、`runtime_gate`、`model_adapters`；
- `kernels/` 作为 L6；
- `tests/test_layer_deps.py` 的边界检查。

当前问题主要是**目录边界和命名**，不是核心逻辑。

---

## 6. 验收检查

完成上述迁移后，至少满足：

1. `import tileinfer.bridge` 不加载 `tileinfer.attention` / `tileinfer.plan` / `tileinfer.metadata`。
2. `tileinfer/__init__.py` 不再导出 `BatchAttention` 等 kernel API。
3. 每个 route-O 模型都在 `bridge/route_o/<model>/` 下，布局一致。
4. `providers/` 的 declarations 与 implementations 物理分开。
5. `providers/` 不再 import `serving`。
6. `fused_moe_oot.py` 有明确的 ProviderSpec / bridge 入口，或已归档。
7. `aot/`、`weights/` 等空占位目录删除。
8. `layer_deps`、`v3_contracts`、LFM3 计划与 doctor 仍然通过。
9. plan id 变化时同步更新 evidence 和状态文档。

---

## 7. 一句话

> 现在的“乱”主要是历史叠加：v3 适配层和早期 kernel/服务线共用一个包，route-O 模型布局不统一，providers 的声明与实现混放。先冻结包入口、抽 flags、统一 route-O，再拆 providers 和清理空目录，不要重写核心分层。
