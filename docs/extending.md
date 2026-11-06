# 扩展指南：加一个模型、一个算子族，或一个更快的实现

> **读者**：读完 [`status-native-backend.md`](status-native-backend.md) 之后，要往这个后端里**加东西**的人。
> 不是来修 bug 的，是来加一个模型 / 一族算子 / 一个 TileLang 内核的。
> **用途**：三个扩展点的**文件位置、最小骨架、注册表、证明它的测试**，以及**哪些代码不能动、违约后是什么症状**。
> **前提**：本文的「不能改」是契约，不是风格建议。核心流水线只有一条，扩展点只有三处；
> 违反第三条（拿不到设备实测比值就默认开）的代价，仓库里已经付过一次（见 §3.6）。
> **交叉阅读**：[`hardware-checklist.md`](hardware-checklist.md)（上机 11 步，设备实测的记录方式）、
> [`plan-vllm-native-backend.md`](plan-vllm-native-backend.md)（v0 范围与取舍）、
> [`notes/fla-chunk-contract.md`](notes/fla-chunk-contract.md)（GDN chunk 链的张量/dtype 契约）。
> **约定**：`file:line` 默认指本仓库；`runtime/vllm-api/...` 的行号指本地转录的 vLLM 0.23 源码。

---

## 0. 三个扩展点一览

| 扩展点 | 加哪个文件 | 一句话职责 | 证明 seam 的测试 |
|---|---|---|---|
| **模型** `ModelProfile` | `tileinfer/vllm/profiles/<model>.py` | 模型事实的**唯一**归属地：arch、head_size、GQA 底线、页策略、KV 形状、是否 hybrid、op_family、拒绝清单 | `tests/test_model_profiles.py`（含 dummy profile） |
| **算子族** `OpFamily` | `tileinfer/vllm/ops/families/<family>.py` | 「本模型族需要接管哪些 vLLM 名字、各自换成什么」的**数据表**：`required_names()` + `bind()` | `tests/test_ops_install.py`、`tests/test_model_profiles.py::test_ops_family_iteration_follows_the_profile` |
| **更快的实现** `Candidate` | 实现模块可新建；**注册**只加在 family 模块的 `CandidateSet.add(...)` 一行 | 同一个 ABI 名字的第二个实现 + 它能运行的**门**（predicate / flag / 设备实测比值） | `tests/test_op_candidates.py`、`tests/test_doctor.py`（候选门状态出现在 `ops.binding` 行） |

三者的关系是**串联**的：

```text
ModelProfile.op_family ──► family_for("gdn-moe") ──► families/gdn_moe.py::bind()
                                                        │  每个 OpBinding.impl = torch baseline
                                                        ▼
                                            CandidateSet(baseline=torch)
                                                        │  add(Candidate(kernel, predicate, ratio))
                                                        ▼
                                        install.py 绑定的 = CandidateSet.dispatched()
                                                        │  select(): 每调用一次，形状/门决定
                                                        ▼
                                            vLLM 的调用名 → 我们的实现 → (或) baseline
```

所以：**profile 决定「用哪张表」，family 决定「表里有什么」，candidate 决定「同一行还有没有第二个实现」。**
三者都注册进进程级注册表，因为 vLLM 的 hook 不带 config 来问问题
（例如 runner 只调用 `get_kv_cache_shape(blocks, page, heads, dim)`）。

---

## 1. 三条共同规则（三个扩展点都适用）

1. **注册表 + 查找函数，不写 `if model == ...`。**
   三个注册表分别是 `profiles.base.PROFILES`、`ops.families.FAMILIES`、
   `CandidateSet` 内部的候选元组。核心代码只调用查找函数
   （`active_profile()` / `family_for()` / `select()`），从不判断具体是哪个模型/哪个族/哪个实现。
2. **可插拔文件不得在模块级 import `vllm` / `torch_npu` / `tilelang`。**
   `tileinfer doctor`、`scripts/serve_qwen36.py --plan-only` 都要能在**没有引擎、没有 NPU** 的
   CPU 机器上 import 整个注册表；`ops/families/gdn_moe.py` 因此把 torch 实现放在 `bind()` **内部**
   import。`tileinfer/kernels/**` 是反例（模块级 `import tilelang`），所以 candidate 的实现函数必须在
   **函数体内** import 内核模块（见 §3.5）。
3. **每个扩展点都靠一个「dummy」证明 seam**：`tests/test_model_profiles.py` 的 dummy profile
   （不同 head 宽、不同 GQA 底线、不同页策略、不同 KV 布局、不同 op_family）在不改
   `platform.py` / `attn/backend.py` / `worker.py` / `install.py` 的前提下驱动了全部四者。
   你加扩展时**照抄这个写法**：先加一个只用一次的 dummy，证明核心真的按注册表走，再接真模型。

判断某段逻辑该放核心还是插件，只问一句：

> **「它随模型变、随实现变，还是随并行度变？」**
> 随模型变 → profile；随实现变 → family / candidate；随并行度变 → **现在无处可放**（§5），
> 先别塞进上面任何一个，否则你会得到一个「只在单卡正确」的字段。

---

## 2. 扩展点 A：模型 —— `ModelProfile`

### 2.1 它是什么

`ModelProfile`（`tileinfer/vllm/profiles/base.py`）是一个 frozen dataclass，字段就是
「这个模型让后端必须不同」的全部事实：

| 字段 | 含义 | 谁消费 |
|---|---|---|
| `name` | 注册名（`"qwen3.6"`） | `profile_by_name`、日志 |
| `archs` | vLLM 架构字符串；`Foo*` 是家族前缀，但优先写 `_VLLM_MODELS` 里的全名 | `profile_for` |
| `head_size` | 内核接受的 head 宽度元组 | `get_supported_head_sizes`、impl 的 head 校验 |
| `group_min` | `num_heads // num_kv_heads` 的下限（paged kernel 的 GQA 底线） | `check_head_layout` |
| `block_size_policy` | `BlockSizePolicy(kernel_page_size, hybrid_align)`：引擎页必须是内核页的整数倍；hybrid 模型还要是 state 页的整数倍 | `validate_block_size`、`get_supported_kernel_block_sizes` |
| `kv_shape` | `KvShape(shape_fn, stride_order, block_dim)`：融合 KV 的**逻辑形状** + **物理 stride 顺序** | `get_kv_cache_shape` / `get_kv_cache_stride_order` |
| `hybrid` | `HybridKind`（mamba 类型、SSU 拒绝类型、state 页字节数函数、state update 算子名），或 `None` | `support_hybrid_kv_cache`、`align_mamba_state_pages`、`worker.mamba_state_path` |
| `op_family` | 算子族名（`"gdn-moe"`；也接受 `a,b` / `a+b` 组合） | `ops/install.py::_profile_family_names` |
| `unsupported_features` | 拒绝清单：`selector` / `plain_selector` / `attn_type` / `config` 四种 `UnsupportedFeature`，各带**两套**原因文案 + 证据 | `platform.check_and_update_config`、`attn/backend.py::validate_configuration` |
| `notes` | 出处/范围说明，**永不解析** | 人 |

**不在 profile 里的**（这也是它存在的原因）：内核页 `KERNEL_PAGE_SIZE=128` 是编译期内核属性；
bf16/fp16 的 dtype 集合是后端属性；`worker_cls`、ABC 拒绝策略形状、eager 门是本后端线一次定死的机制。
profile **引用**内核页（`BlockSizePolicy` 里填 `KERNEL_PAGE_SIZE`），不重新定义它。

### 2.2 加哪个文件

`tileinfer/vllm/profiles/<model>.py`（一模型一文件）。然后**一行**把它加入发货列表：
`tileinfer/vllm/profiles/__init__.py` 末尾的 `from . import qwen3_6` 旁边再加一行。
这一行属于「插件包的注册列表」，不属于固定流水线（§2.6）；`platform.py` / `attn/` / `worker.py` / `ops/install.py`
一个字都不用改。

### 2.3 最小骨架

```python
# tileinfer/vllm/profiles/my_model.py
"""<模型名> 的 profile：只写模型事实，不写硬件事实。"""

from __future__ import annotations

from typing import Tuple

from tileinfer.serving.buckets import KERNEL_PAGE_SIZE
from tileinfer.vllm.profiles.base import (
    BlockSizePolicy,
    HybridKind,
    KvShape,
    ModelProfile,
    UnsupportedFeature,
    register_profile,
)

NAME = "my-model"


def fused_kv_shape(
    num_blocks: int, block_size: int, num_kv_heads: int, head_size: int
) -> Tuple[int, ...]:
    """逻辑形状；`block_dim` 必须真的指向 num_blocks 那一维（KvShape 会 assert）。"""
    return (int(num_blocks), 2, int(block_size), int(num_kv_heads), int(head_size))


MY_MODEL = register_profile(
    ModelProfile(
        name=NAME,
        archs=("MyModelForConditionalGeneration",),      # 优先写 vLLM 注册表里的全名
        head_size=(128, 256),
        group_min=2,
        block_size_policy=BlockSizePolicy(
            kernel_page_size=int(KERNEL_PAGE_SIZE),
            hybrid_align=None,                            # 纯 attention 模型：没有 state 页
        ),
        kv_shape=KvShape(
            shape_fn=fused_kv_shape,
            stride_order=(1, 0, 2, 3, 4),                 # 物理 [2][blocks][page][heads][dim]
            block_dim=0,
        ),
        hybrid=None,                                      # 或 HybridKind(...)，见 qwen3_6.py
        op_family="my-family",                            # 必须和 ops/families/<name>.py 对得上
        unsupported_features=(
            # 每一种拒绝都要有「给平台」和「给 backend」两句原因（历史上它们从来不一样），
            # 以及证据。抄 qwen3_6.py 的格式，别自己发明。
            UnsupportedFeature(
                field="use_mla",
                reason="the paged kernels are GQA/MHA with an explicit head dim",
                backend_reason="MLA is not supported by the TileInfer attention backend",
            ),
            UnsupportedFeature(
                field="attn_type",
                kind="attn_type",
                reason="decoder self-attention only",
            ),
        ),
        notes="provenance: config.json / layer_types / 与上游文件的行号",
    )
)
```

### 2.4 注册表与查找

* 注册：`register_profile(profile, *, replace=False)`；名字重复会抛（`unregister_profile` 给测试/重载用）。
* 按架构：`profile_for(archs, config_hints=None)`。`config_hints` 可以带 `{"profile": <name>}` 直接指定，
  或用 `head_size` / `group_min` / `is_hybrid` 在**多候选**间收窄；hints **永远不会凭空造出匹配**，
  未知架构照样抛，错误信息里列全部已知 profile 并**指出该往哪个目录加文件**。
* 进程内激活：`active_profile()`。`platform.check_and_update_config` 里
  `profile_for_config()` → `set_active_profile(profile)`，之后 backend/worker 这些**没有 config 参数**的
  hook 都回答「当前服务的模型」而不是默认 profile。测试用 `reset_active_profile()` 还原。
* 默认值：`default_profile()` = **第一个注册的 profile**。所以新增的 `from . import my_model` 要写在
  `qwen3_6` **之后**，否则「手工 config、没有 architecture」的路径会落到新 profile 上
  （`tests/test_model_profiles.py::test_registry_resolves_qwen36_by_arch` 钉住了 default 是 qwen3.6）。
* profile **没有** lazy import：它是按 arch 匹配，不是按名字 import。所以「加了文件但忘了那行 import」
  的症状是 `profile_for` 抛 `no TileInfer model profile matches ... known profiles: qwen3.6 ...`，
  **这是设计好的失败**，不是 bug（错误信息就是修复指引）。

### 2.5 证明它的测试

`tests/test_model_profiles.py`（8 个用例）里，能被你复用的四类：

| 用例 | 证明什么 |
|---|---|
| `test_registry_resolves_qwen36_by_arch` | 架构名 → profile；`default_profile()` 是发货 profile |
| `test_unknown_arch_raises_with_the_known_profile_list` | 未知架构的报错含已知列表 + **该加文件的目录** |
| `test_platform_enforces_the_profile_page_and_mamba_geometry` / `test_backend_enforces_the_profile_shape_heads_and_kernel_page` / `test_worker_mamba_gate_uses_the_profile_hybrid_kind` | 平台/backend/worker 真的在用 profile 的**值**（不是读同名模块常量） |
| **`test_a_dummy_profile_drives_platform_and_backend`** | **seam 本身**：dummy 的 64 页策略、`(2, blocks, ...)` 形状、`(0,1,2,3,4)` 顺序、GQA 底线 4、自己的拒绝文案，在**不改核心**的情况下全部生效 |

加模型时你要补的测试：一个 dummy（照抄）+ 一个「真 config 值被强制」的用例。不要再写一套「读模块常量」的断言。

### 2.6 不能改的（固定流水线）与原因

| 不能改 | 为什么 | 违约后的症状 |
|---|---|---|
| `platform.check_and_update_config` 的**顺序**：解析 profile → `set_active_profile` → 拒绝 feature → `validate_block_size` → 对齐 mamba 页 | backend/worker 的 hook 无 config 参数，只能读「已激活」的 profile；顺序颠倒时它们读到默认 profile | 第二个模型静默按 Qwen3.6 的 2048 页/128 head 跑 → 启动过、数值错 |
| `validate_block_size` / `align_mamba_state_pages` 的规则（「引擎页 = 内核页整数倍」「hybrid 还要对齐 state 页」「padded mamba 页 == 一个 attention 页」） | 这三条是 vLLM 0.23 的 `update_block_size_for_backend` 与 `unify_kv_cache_spec_page_size` 的前提 | 池子分配成功但 `pool_sub_view` 拿不到平坦 `[pages,128,heads,dim]`，或 hybrid KV manager 直接拒绝混编 |
| `attn/backend.py` 的 `get_kv_cache_shape` / `get_kv_cache_stride_order` / `get_supported_*` 全部走 `active_profile()` | runner 用它们决定**分配多少字节、按什么顺序摆**；两处各留一份常量就一定会漂 | 池形状对、物理顺序错 → 内核把 K 当 V 读，无异常、全错 |
| `worker.py` 四个 hook 的名字/顺序、`mamba_state_path` 的判定、`_init_kv_zero_meta` 的拒绝 | 由本地 `gpu_worker.py` / `gpu_model_runner.py` 的调用点转录而来（`worker.py:562` / `:752` / `:931` / `:999`） | 换名字 = vLLM 找不到 hook，退回 CUDA worker |
| `plugin.install()` 的四步顺序、`patches` 注册表 | 安装必须在模型层构造**之前**完成（`CustomOp` 在 `__init__` 里重钉 `_forward_method`） | 接管晚一步 → 层里还是上游 Triton 实现 |
| `platform.HYBRID_BLOCK_SIZE` 这类模块别名 | 它只描述**启动 profile**（`BOOT_PROFILE`），是给日志/测试读的镜像，不是第二个真值 | 有人改成「按当前 profile 动态算」→ 启动期写入的页被后来的调用改掉 |

一句话：**profile 只提供数据，判定与写入顺序留在核心。** 你在 profile 里能做的是「换值」，不是「换流程」。

### 2.7 这个 port 学到的坑

* **`head_size` 是元组不是标量。** GDN 的 state head 是 128、full-attention 是 256；写成单个 256 会让
  `get_supported_head_sizes()` 少一项，`validate_configuration` 拒绝合法配置。
* **`group_min` 只能更严，不能更松。** 它编码的是 paged kernel 的 GQA 下限；为了「先跑起来」把它调成 1，
  等于让内核读越界 head 而不报错。
* **`stride_order` 与 `shape_fn` 是一对，必须同源。** 只改形状不改顺序（或反之）会得到
  「形状正确的错误指针」——这是 §1 第 21 条那个静默 bug 的同款。
* **`hybrid_align` 必须是 `kernel_page_size` 的正整数倍**（`BlockSizePolicy.__post_init__` 会拒绝）；
  state 页字节数必须**恰好等于一个 attention 页**，否则 block manager 无法互换两种页。
* **拒绝清单要写两套文案。** `reason` 给平台（`(...)` 里的父级原因），`backend_reason` 给
  `validate_configuration` 的返回列表；两处历史上不同，测试同时钉住两边。写一份会被测试抓。
* **profile 里不要 `import torch`。** 本包要在 CPU 机器上可 import（`--plan-only`、契约测试）；
  `tileinfer.serving.buckets` 已经带 torch，除此之外别再拉重依赖。

---

## 3. 扩展点 B：算子族 —— `OpFamily`

### 3.1 它是什么

一个 **family 模块**只提供两个函数，是「本模型族要接管哪些 vLLM 名字」的**数据表**：

* `required_names() -> tuple[str, ...]`：本族承诺的名字，顺序即表序。
  `install.py` 会用它校验 `bind()` 的表（少一项就 WARNING），`doctor` 也会打印。
* `bind() -> list[OpBinding]`：每行一个 vLLM 符号。
  `OpBinding` 的字段就是「怎么接管 + 为什么 + 什么时候能删」：

```text
op            本仓库的算子名（用于报告/日志/测试，不是 vLLM 名）
family        ★ TILEINFER_OPS 的开关族：只能是 "gdn" / "moe" / "dense"（见 3.7 的坑）
targets       同一个入口的备选 vLLM 名字（上游改名 = 第一个解析不到，不是崩溃）
aliases       额外要重绑的**消费方命名空间**（上游 by-value import 的副本）
impl          本仓库 torch 实现（= candidate 的 baseline）
kind          "attr"（重绑模块属性）或 "custom_op"（注入 forward_native/forward_oot）
abi_ok / abi_note  False = 故意只报告不接管（不是 drop-in 时）
why / remove_when / upstream_ref   证据：为什么必须接管、什么条件下能删、本地行号
candidates    CandidateSet（可选；§3 的内容）
```

`install.py` 拥有**机制**（对活模块树做 plan、包 switch、写 `Patch`、出报告），family 拥有**数据**。
这就是「加一族 = 加一个文件」的全部含义。

### 3.2 加哪个文件

`tileinfer/vllm/ops/families/<family>.py`。**不需要**改 `families/__init__.py`：
`family_for()` 会把 `my-family` 里的 `-` 换成 `_` 后 `importlib.import_module` 该模块，
模块若没有显式注册就按它的 `NAME` 自动注册（并给请求的拼写留一个别名）。
`families/__init__.py` 里那行显式 `from . import gdn_moe` 只是为了「发货面在文件里可见」，不是必须动作。

### 3.3 最小骨架

```python
# tileinfer/vllm/ops/families/my_family.py
"""<模型族> 的 vLLM 接管表：数据，不是机制。"""

from __future__ import annotations

from typing import Any, List, Tuple

NAME = "my-family"                      # profile.op_family 里写的那个字符串

REQUIRED_NAMES: Tuple[str, ...] = ("my_op", "l2norm_fwd")


def required_names() -> Tuple[str, ...]:
    return REQUIRED_NAMES


def bind() -> List[Any]:
    # 延迟 import：本模块必须能在没有 torch 的机器上 import（doctor / --plan-only）。
    # 也因为 install 会 import 本模块，OpBinding 只能从已加载的 install 里取（顶层 import 会循环）。
    from tileinfer.vllm.ops import gdn
    from tileinfer.vllm.ops.install import OpBinding

    return [
        OpBinding(
            op="my_op",
            family="gdn",                               # ★ 开关族，不是本 family 的名字
            targets=("vllm.model_executor.layers.fla.ops.my_op",),
            aliases=("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn.my_op",),
            impl=gdn.my_op,
            why="调用点 + 上游是 Triton/CUDA kernel，Ascend 上没有 fallback",
            remove_when="上游出现 NPU/torch 分支时",
            upstream_ref="runtime/vllm-api/...py:123-145",
        ),
        # 只报告、不接管（ABI 不是 drop-in）——让「缺席」出现在日志和 doctor 里：
        OpBinding(
            op="declined_op",
            family="gdn",
            targets=("vllm.model_executor.layers.fla.ops.declined_op",),
            abi_ok=False,
            abi_note="declined - not a drop-in: 上游多 chunk_indices/head_first 两个参数",
            why="Reported, not patched: see abi_note.",
            remove_when="本仓库端口补齐该 ABI 时",
            upstream_ref="runtime/vllm-api/...py:243-252",
        ),
    ]
```

需要 `custom_op` 形态（vLLM 的 `CustomOp` 子类在 `__init__` 里重钉 `_forward_method`）时，
只改 `kind="custom_op"` 和 `targets`；`_custom_op_takeover` 会同时写 `forward_native` 与 `forward_oot`。
**不要**自己写 takeover 函数——那是机制。

### 3.4 注册表与查找

* `FAMILIES[name] -> module`，`register_family(name, module)` 会校验 `required_names`/`bind` 可调用。
* `family_for(name)`：先查表；查不到就 lazy import `tileinfer.vllm.ops.families.<name 把 - 换成 _>`；
  模块里有 `NAME` 就按 `NAME` 注册，否则按请求的拼写注册。找不到时抛 `KeyError` 并列出已知族。
* `install.py` 的入口：`_profile_family_names(profile)` 读 `profile.op_family`
  （支持 `a,b` / `a+b` 组合）→ `family_for` → `bind()` → 逐行 plan。表按族缓存，`_reset()` 清缓存。
* 一个族一个 `bind()`，`required_names()` 必须与表里的 `op` 集合一致（否则 install 只 WARNING，
  但 doctor/测试会露出来）。

### 3.5 证明它的测试

| 测试 | 证明什么 |
|---|---|
| `tests/test_model_profiles.py::test_ops_family_iteration_follows_the_profile` | 注册一个 dummy family 并把 dummy profile 激活后，`install.bindings()` **只剩** dummy 的表；换回 qwen profile 又恢复发货表——**没有改 `install.py`** |
| `tests/test_ops_install.py`（19 个用例，`EXPECTED_BOUND` / `EXPECTED_UNBOUND` / `EXPECTED_DISABLED` 精确断言） | 每个 vLLM 目标都真的解析；`chunk_local_cumsum` 因 `abi_ok=False` 故意 unbound；`silu_and_mul` 因 `dense` 默认关而 disabled |
| `tests/test_op_candidates.py::test_the_family_still_promises_exactly_the_same_names` | `required_names()` 与表序一致（漂了就是失败） |

README 级别的判据：**`python -m pytest tests/test_ops_install.py -q` 必须精确通过**，不要改成「至少有一个」。那张精确列表就是「上游改名/我们漏移植」的探测器。

### 3.6 不能改的（固定流水线）与原因

| 不能改 | 为什么 |
|---|---|
| `install.py` 的 `_plan` 五态（disabled / abi_ok=False / impl 缺失 / 目标解析失败 / planned）与 `OpsReport` 三态 | doctor、启动日志、`tests/test_ops_install.py` 都以它为契约；改一态等于改所有运维面的语义 |
| `_attribute_takeover` + `_wrap`：幂等 + **每次调用**读 `TILEINFER_OPS` + 带 `__tileinfer_*` 标记 | 每个进程都会装一遍插件；不幂等就会套娃。开关必须 per-call，否则翻转要重启 |
| **双写**：定义模块 + 消费方命名空间（`aliases`） | 上游 `qwen_gdn_linear_attn.py:24-49` 是 **by-value import** 五个 FLA 算子；只绑定义模块时调用点仍指向旧对象（import 顺序决定成败，这是最难查的一类） |
| `patches.Patch` 记录（why / since / remove_when / upstream_ref） | 这是「我们改了 vLLM 哪个符号」的唯一审计线索；绕过它 = 下一代人不知道这是我们的 |
| `CandidateSet.baseline.impl` 必须仍是 `binding.impl`（`_installed_impl` 绑 `dispatched()`） | 报告/测试/`__tileinfer_impl__` 都按 baseline 命名实现；把 dispatcher 当 impl 报告，等于所有「实现是谁」的断言失真 |
| 一个 ABI 名字一行；不要合并两个算子到一个 `OpBinding` | 报告、开关、候选门都是 per-op 的粒度 |

### 3.7 这个 port 学到的坑

* **两个「family」不是一回事。** `ModelProfile.op_family` 是**模块名**（`"gdn-moe"`）；
  `OpBinding.family` 是 `TILEINFER_OPS` 的**开关族**（`"gdn"` / `"moe"` / `"dense"`，见 `ops/switch.py`）。
  把 `"gdn-moe"` 填进 `OpBinding.family`，`install._plan` 的 `report.families.get("gdn-moe", False)` 恒为 False
  → 这一行永远 `disabled`（doctor 会显示 "off by switch"，看起来像你忘了开开关）。
* **`aliases` 不是可选的美化。** 少写一条 alias，接管只在「定义模块先 import」的顺序下生效；
  换个 import 顺序就静默退回上游实现。GDN 那五个 FLA 名字全部需要双写。
* **`abi_ok=False` 的条目必须留下。** 它是「我们知道这里差一个 ABI、故意没接」的证据；
  删掉它就变成「没人知道为什么缺」。
* **不要模块级 import torch。** `bind()` 是唯一允许 import 实现的地方；否则 doctor 在纯 CPU 机器上就挂。
* **掩码必须用 `torch.where` 选择，并且和单边钳位 `safe_exp` 配对。**
  `safe_exp(x) = exp(where(x <= 0, x, -inf))` 是**单边截断**：正的指数参数一律变 0，不是 `exp(x)`。
  写成 `a * mask` 会得到 `inf * 0 = nan`；`nan` 只在 `g` 衰减足够大时出现——小规模单测全绿，真模型才炸。
  六个位置的已核实分工：`exp` 只用于「绝对的 `g`」（必 ≤ 0），`safe_exp` 只用于「`g` 的差」（可正）。
  参考实现：`ops/fla/kkt.py:148-154`、`ops/fla/chunk_o.py:291-299`（`torch.where` + `_safe_exp`）。
  另注：`kkt` 用**严格**下三角 `>`，`chunk_o` 用**含对角** `>=`，两者都不是笔误。
* **dtype 契约照抄上游，不要「优化成 fp32」。** `A_inv` 是 `k.dtype`、`wy_fast` 两个 dot 前先降回 bf16、
  `h` 每个 chunk 边界落盘截断一次——这三处是**契约**：全程 fp32 更准但不逐位一致。
  另一个具体例子：`fused_post_conv_prep` 上游把 `g`/`beta` 装进 **fp32** 容器
  （`runtime/vllm-api/...fused_gdn_prefill_post_conv.py:202-203`），本仓库端口把 `beta` 落回 `b.dtype`
  （bf16）——这是**已知差异**，GDN prefill 数值不对时第一个查它，不要顺手「修」回去。
* **`upstream_ref` 要写行号。** 「函数签名」的最终依据是调用点实参（例如 `fla/ops/chunk.py:41-86`），
  不是 sglang 的 vendored 副本；B 级数学的地方要标注来源（见 `notes/fla-chunk-contract.md`）。

---

## 4. 扩展点 C：更快的实现 —— `Candidate`

### 4.1 它是什么

一个 vLLM ABI 名字今天只有一份实现（torch baseline）。`ops/candidates.py` 把它变成
「baseline + 若干候选」，**按调用选择**，并且**默认永不启用未实测的候选**：

* `CallSpec`：本次调用的实参/关键字实参，作为数据。predicate 只能看
  `.shape` / `.dtype` / `.is_contiguous()`——**不能** `.item()` / `.max()` / 任何张量运算（那是 host sync）。
* `Candidate`：`impl` + `predicate` + `enabled_by_default` + `measured_ratio` + `note`。
* `CandidateSet`：一个 op 的集合，`baseline` 必存在；`add()` 追加，**注册顺序即优先级顺序**。
* 门（顺序固定）：① `TILEINFER_OPS_CANDIDATES=off` 全局熔断 → ② `enabled_by_default` →
  ③ `measured_ratio >= threshold` → ④ predicate。
  `ENV_CANDIDATES` **只能关，不能开**；拼错的值 fail-safe 成 off 并只警告一次。
* `install._installed_impl` 绑的是 `set.dispatched()`（一个薄封装：`select()` 后原样转发实参）；
  没有任何候选 eligible 时它调用 baseline —— **与接入候选之前逐字节相同**。

### 4.2 加哪个文件

* **实现**可以是一个新模块（`tileinfer/vllm/ops/<op>_kernel.py`，或 `tileinfer/kernels/**` 里的 TileLang 内核）。
* **注册只有一处**：该 op 所属 family 模块里的 `CandidateSet.add(Candidate(...))` 一行
  （`families/gdn_moe.py::_with_candidates` 就是模板和唯一实例）。

不需要新注册表，也不需要改 `install.py` / vLLM 侧任何 ABI。

### 4.3 最小骨架

```python
# 在 family 模块里（数据面）——注册一行：
from dataclasses import replace
from tileinfer.vllm.ops.candidates import Candidate, CandidateSet, always


def _fp32_rect(call_spec):                      # 只读形状/dtype/连续性：predicate 必须是免费的
    x = call_spec.arg(0, "hidden_states")       # 位置实参 或 同名关键字，两种调用都要能匹配
    shape = getattr(x, "shape", ())
    return len(shape) == 2 and str(getattr(x, "dtype", None)) == "torch.float32" and bool(x.is_contiguous())


def _my_kernel(hidden_states, *args, **kwargs):
    # 同 ABI；内核模块必须**在这里** import（tileinfer/kernels 模块级 import tilelang，
    # 顶层 import 会让 CPU 机器上的 doctor 直接挂）。
    from tileinfer.kernels.my_op import my_op_kernel

    return my_op_kernel(hidden_states, *args, **kwargs)


def _with_candidates(binding):
    # baseline 永远在：它就是接入候选之前的 torch 实现，名字和 impl 都别动。
    chosen = CandidateSet(
        op=binding.op,
        baseline=Candidate(
            name="torch", impl=binding.impl, predicate=always, enabled_by_default=True,
            note="the pre-candidate behaviour, byte for byte",
        ),
    )
    chosen.add(
        Candidate(
            name="my_kernel",
            impl=_my_kernel,
            predicate=_fp32_rect,
            enabled_by_default=True,
            measured_ratio=1.12,        # ★ 设备实测比值（1.0 = 与 baseline 持平）；见 4.6/4.7
            note=(
                "device A/B on <card>, <date>: 1.12x vs torch baseline (benchmarks/<file>.py), "
                "shapes covered: ...; not covered: ..."
            ),
        )
    )
    return replace(binding, candidates=chosen)
```

### 4.4 注册表与查找

* 集合在 `bind()` 里构建（`_with_candidates` 的写法：表仍是「一行一个 vLLM 名字」，候选是最后附加的）。
* 查找发生在**每次调用**：`dispatched()` → `CandidateSet.select(CallSpec(...))` → 第一个 eligible 且
  predicate 命中的候选，否则 baseline。`select` 不许分配、不许碰张量值；predicate 抛异常只记一次警告
  并按「不匹配」处理（一个坏 predicate 不能带崩一个 forward）。
* 报告：`install.candidate_report()` → `{op: CandidateSet.describe()}`，每行带 `status`
  （`baseline` / `eligible` / `disabled: <原因; 原因>`）和 `note`。
  `tileinfer doctor` 的 `ops.binding` 行现在会把非 baseline 候选的门状态打出来（`tests/test_doctor.py`），
  所以「为什么没用那个内核」永远有记录可查。

### 4.5 证明它的测试

`tests/test_op_candidates.py`（25 个用例）。加候选时至少要让下面四类继续成立：

| 用例 | 证明什么 |
|---|---|
| `test_a_baseline_only_set_is_the_pre_candidate_behaviour` | 只有 baseline 时 `select` 返回**实现对象本身**，`dispatched()` 与直接调用逐字节相同 |
| `test_select_never_touches_tensor_values` | predicate 只能读形状/dtype/连续性（用一个访问值就抛的假张量钉住） |
| `test_a_recorded_ratio_below_the_threshold_stays_disabled` | 记录 0.86x（本仓库真实测量）且 flag=on，仍然不 eligible；`describe()` 里能看到原因 |
| `test_the_kill_switch_disarms_every_non_baseline_candidate` | 熔断 per-call，不用重装；typo fail-safe |
| `test_a_shape_predicate_selects_the_candidate` / `test_the_wrapper_dispatches_by_shape_in_either_calling_style` | 位置实参与关键字实参两种调用风格都能被 predicate 看到（`CustomOp` 走关键字） |
| `test_install_binds_the_dispatcher_and_reports_the_baseline_impl` | `install._wrap` 的 `__tileinfer_impl__` 仍指 baseline，`__tileinfer_candidates__` 指集合 |

### 4.6 不能改的（固定流水线）与原因

| 不能改 | 为什么 |
|---|---|
| 门的四个步骤与顺序 | 熔断必须在最前（运维要能一键关掉所有候选）；flag 与比值是**静态**过滤，必须在 predicate 之前（否则每次调用都要跑一个未启用候选的 predicate）；predicate 最后（它是唯一要看实参的一步） |
| `DEFAULT_THRESHOLD = 1.0` 与「比值低于阈值即不 eligible，**即使 flag=on**」 | 这是「未实测/未达标的内核永远不会因为有人翻 flag 而变成默认」的唯一保证。本仓库的教训：MoE grouped-GEMM 内核在设备上三轮优化后只有 **0.86x**（`docs/status.md` E19/H7），已在 `families/gdn_moe.py` 的 `moe_reference_oracle.note` 里记档 |
| `binding.impl` 仍是 baseline；dispatcher 只做转发 | 报告、`__tileinfer_impl__`、旧测试全部按 baseline 命名实现 |
| `select()` 的「零分配、无张量值、无实现调用」契约 | 它在每个 forward 的每个 op 调用上跑一遍 |
| 一个 op 一个 `CandidateSet`、候选名唯一、baseline 名不可复用 | 名字是 `describe()`/日志/报告里识别实现的唯一键 |
| `ENV_CANDIDATES` 的方向性（只能 disable） | 环境变量是熔断器，不是启用开关；能开的话，一次 typo 就能让没测过的内核上线 |

### 4.7 这个 port 学到的坑

* **`plan_key` 必须包含「改变编译内核或其 staging buffer」的每一个维度。**
  `serving/planner.py::plan_key` 现在是
  `(config_key, split_pages, bucket, width, block_size, pool_cap)`：
  `config_key` 里带 backend 名、phase、heads/kv_heads/head_size、dtype、device；
  `split_pages`（每个 split tile 的 KV 页数）曾经**不在** key 里，于是两个 split 因子不同的 bucket
  共享了同一份 plan —— 调度与 workspace 大小不同，**静默算错**（`planner.py:75-80` 的 docstring 就是这条）。
  另外 `pool_cap` 用的是**声明容量**（向上取 2 的幂），不是精确页数：精确池会让 9425/9899 的页数漂移
  击穿缓存，把 70–100 s 的编译推回引擎内部（`docs/status.md` §5）。
  **给新配置加维度时，先问「这条配置会不会改变 kernel 或它的 staging buffer」**：会，就必须进 key，
  哪怕多一次缓存未命中。
* **bf16 往返可以是契约。** 见 §3.7：`A_inv`/`wy_fast`/`h` 三处截断、`beta` 的 bf16 vs 上游 fp32。
  一个「更准」的实现如果不复现这些截断，就不是 bit-comparable，A/B 结论会被数值噪声吃掉。
* **只有拿到设备实测比值才能默认启用。** `measured_ratio` 写的是**人测出来并记下方法**的数字
  （`note` 里写卡型/日期/benchmark 文件/覆盖的 shape），不是「应该会快」；没有测量就写 `None` +
  `enabled_by_default=False`，并像 `moe_reference_oracle` 那样把原因留在 note 里。
  比值 `< 1.0` 的候选可以登记（作为记录），但永远不会 eligible。
* **不要为了绑定内核而在模块级 import 内核。** `tileinfer/kernels/**` 模块级 `import tilelang`；
  candidate 的 `impl` 必须在函数体里 import，否则 `tileinfer doctor` 在 CPU 机器上直接失去这个检查。
* **predicate 不要吃调用约定。** 位置调用（`l2norm_fwd(x, eps)`）和关键字调用
  （`forward_native(q=..., k=...)`）都要覆盖，用 `spec.arg(index, name)` 而不是 `spec.args[index]`。
* **优先级 = 注册顺序。** 两个同样 eligible 的候选，先 `add()` 的赢；想「先降级到某实现再降级到 baseline」，
  顺序就是语义。

---

## 5. 固定 vs 可插拔：一条判断规则

> **「它随模型变、随实现变，还是随并行度变？」**

| 内容 | 归属 | 判据 / 为什么 |
|---|---|---|
| arch 字符串、head 宽度、GQA 底线、页策略、KV 形状/顺序、hybrid 几何、op_family、拒绝清单 | **可插拔**：`profiles/<model>.py` | 每个模型都不同；核心只读不判 |
| `KERNEL_PAGE_SIZE = 128`、内核 tile 算术、bf16/fp16 dtype 集合 | **固定**：`serving/buckets.py`、backend | 编译期内核/硬件属性；profile 只**引用** |
| `worker_cls` 选择、ABC 拒绝策略形状、eager 门、hook 名字/顺序、plugin 四步 | **固定**：`platform.py` / `worker.py` / `plugin.py` | 本后端线一次定死；由 vLLM 0.23 调用点决定 |
| vLLM 名字 → 实现、aliases、why/remove_when、abi_ok | **可插拔**：`ops/families/<family>.py` | 每个模型族不同 |
| plan / wrap / switch / Patch 注册 / 报告三态 / 双写 | **固定**：`ops/install.py` | 每个族都一样；换族不该换机制 |
| 第二个实现、谓词、默认开关、实测比值、note | **可插拔**：family 里的 `Candidate`（实现可新建模块） | 每个实现不同 |
| 门顺序、`DEFAULT_THRESHOLD=1.0`、熔断方向、`select` 的零开销契约 | **固定**：`ops/candidates.py` | 全局契约，不能 per-op 重定义 |
| `plan_key` 的维度集合 | **固定**：`serving/planner.py`（新增维度要改核心） | 配置组合维度；见 §4.7 的教训 |
| 进程级注册表（`PROFILES` / `FAMILIES` / `_BINDINGS`）与激活语义 | **固定** | vLLM 的 hook 不带 config；激活点只能在 `check_and_update_config` |
| **TP / EP / PP / DP** | **尚不可插拔** | 见 §6 |

反例判据：如果一段代码需要写 `if profile.name == "qwen3.6":` 或 `if nproc > 1:`，它就不是扩展点，
而是核心逻辑还没被设计出来。

---

## 6. 明确尚不可扩展的地方：并行（TP / EP）

**结论：现在加不了 TP/EP，不要「顺手加一个字段」。** 三条硬证据：

1. **`ModelProfile` 没有 `tp` / `ep` 维度。** head 数、GQA 底线、KV 形状都是「整卡全量」的口径；
   加入 `tp` 意味着 head 要在 rank 间切分，`head_size`/`group_min`/`kv_shape` 全部变成
   「每 rank 的」量，而 `kv_shape(shape_fn)` 的签名里没有 rank/world_size。
2. **`plan_key` 没有 `tp` 维度。** `plan_key(config_key, split_pages, bucket, width, block_size, pool_cap)`
   的 `config_key` 只带 backend/phase/heads/kv_heads/head_size/dtype/device
   （`attn/impl.py::_plan_key`）。多卡下同一个 key 会在不同 rank 上代表不同的本地形状 → 计划共享即错。
   这正是 §4.7 那条「缺一个维度 = 静默共享」的放大版。
3. **算子是「全张量」假设。** `fused_experts` 直接吃完整的 `w1/w2` 与全局 `topk_ids`；
   `moe_align_block_size` 明确**拒绝** `expert_map`（「its kernel body is not local and v0 has no EP」，
   `families/gdn_moe.py` 里该行的 `why` 写明了这一点）；attention 内核按 `[pages, 128, heads, dim]` 读整份
   KV，没有 head 分片；KV 池是**每进程一份**融合池（`supports_kv_connector() == False`）。
   多卡要有 collective 边界（all-reduce / all-gather / dispatch-combine），而现在的接管点全是
   「单算子替换」，没有放通信的位置。

**建议（按顺序）：**

1. **先做完第一次单卡上机**：`docs/hardware-checklist.md` 步骤 6–11（权重 + KV 池 → 首个 decode step 对
   logprobs → GDN state 有限 → 256 专家路由对 oracle → eager/图 → TTFT/TPOT）。
   TP 的设计会被这次跑出来的数字与失败模式决定（尤其是 KV 池分配与 4 MiB state 页的真实行为）。
2. **然后先写设计，不写代码**：至少在 `docs/plan-vllm-native-backend.md` 里回答
   ① head 切分规则由 profile 声明还是由 parallel config 推导；② `tp` 进 `config_key` 还是进
   `ModelProfile`（二选一，不要两边各存一份）；③ 通信算子放在 op 边界（可插拔的 `all_reduce` hook）
   还是内核内部（不可插拔，不建议）；④ `expert_map`（EP）在 `moe_align_block_size` 里补齐后，
   哪些 candidate 的 predicate 要随之失效。
3. **最后才加 `plan_key` 的 `tp` 维度**，并在同一次改动里补一个「两个 rank 形状不同不得共享 plan」的
   测试——参照 `tests/test_plan_cache.py` 现有的 key 断言写法。

在那之前，任何「先加个 `tp=1` 字段占位」的做法都会让 profile 看起来已经支持多卡，
而实际行为是每个 rank 都按全量计算——**形状正确、数值全错**，正是本仓库反复踩的那一类 bug。

---

## 7. 落地清单（加任何东西都走这一遍）

1. **抄 dummy**：先照 `tests/test_model_profiles.py` 的 `_dummy_profile` / `_DummyFamily` 写一个
   只用一次的扩展，证明核心确实按注册表走（不改核心）。
2. **接真东西**：
   * 新模型 → `profiles/<model>.py` + `profiles/__init__.py` 一行 import（写在 `qwen3_6` 之后）；
   * 新族 → `ops/families/<family>.py`（profile 的 `op_family` 指过来）；
   * 新实现 → family 里 `CandidateSet.add(Candidate(...))` 一行，`measured_ratio` 只填实测值。
3. **跑测试**（Windows 开发机口径；`runtime/pylibs` 里只有 torch，没有 vLLM/NPU）：

   ```powershell
   $env:PYTHONPATH="E:\project\TileInfer\runtime\pylibs"
   $env:PYTHONDONTWRITEBYTECODE="1"
   python -m pytest tests/test_model_profiles.py tests/test_op_candidates.py tests/test_ops_install.py tests/test_doctor.py -q -p no:cacheprovider -p no:warnings
   ```

   当前基线：**92 passed**（`test_doctor` 40 / `test_model_profiles` 8 / `test_op_candidates` 25 /
   `test_ops_install` 19，其中 doctor 的候选行用例在内）。数字变少 = 你删掉了别人的契约。
   注意 `pyproject.toml` 已有 `addopts = "-q"`，**不要再手写第二个 `-q`**（会吞掉汇总行）。
4. **看 doctor 的那一行**：`python -m tileinfer.vllm.doctor`。`ops.binding` 行必须能回答
   「哪些候选存在、哪些被关、为什么被关」；本机（无 vLLM）预期是 SKIP 但**仍带**
   `| candidates: fused_experts[moe_reference_oracle=disabled: off by default]`。
   exit code 只由 hard check 决定：本机无 `torch_npu` / `tilelang`，仍然 **exit 1**。
5. **补台账**：`docs/status-native-backend.md` §2 的模块清单要加新文件（否则下一个读者会以为它不存在），
   §0 一句话状态要能看出这次扩展；拒绝/取舍写进 profile 的 `unsupported_features` 和
   `docs/plan-vllm-native-backend.md`，不要只活在 commit message 里。
