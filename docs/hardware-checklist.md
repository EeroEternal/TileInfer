# TileInfer 昇腾 950 硬件点亮清单（bring-up checklist）

> **范围说明（2026-11-12）**：本文是**操作清单**，不是架构文档。
> 文中出现的「原生后端」「平台层」「`tileinfer/vllm/*`」等表述属于 **v2 形态**
> （自带平台层、不装 vllm-ascend）；项目定位已改为 **v3（vllm-ascend 之上的兼容适配层）**，
> 见 [`architecture.md`](architecture.md)。各步骤的判据与失败归因仍然有效。


> 适用对象：第一次在这台 Ascend 950 机器上把 **TileInfer 作为 vLLM 0.23 原生后端**跑起来的人。
> 读者**不需要**本项目上下文：每一步给出「命令 → 期望输出 → 失败归因」，顺序按「后一步依赖前一步的结论」排。
> 单卡、单模型（Qwen3.6-35B-A3B，40 层 = 10 层 full-attention + 30 层 GDN，256 专家 MoE）、
> vLLM 0.23.x、CANN 9.3.0、TileLang Ascend 0.1.15。
>
> **当前进度前提（先读这段，否则会误判失败）**：平台层（`tileinfer/vllm/platform.py`）、attention
> 后端（`tileinfer/vllm/attn/`）、插件入口（`tileinfer/vllm/plugin.py`）、patch 注册表
> （`tileinfer/vllm/compat/patches.py`）都已就位；worker 的四个 hook
> （`init_device` / `determine_available_memory` / `initialize_from_config` /
> `compile_or_warm_up_model`）在 `tileinfer/vllm/worker.py` 里**已经实现**
> （`:848` / `:1533` / `:1712` / `:1819`），它们依据的 `vllm/v1/worker/gpu_model_runner.py` 也**已经入库**
> （`reference/vllm-0.23/vllm__v1__worker__gpu_model_runner.py`，334,736 B，含 `_dummy_run:5599`、
> `profile_run:6168`、`capture_model:6525`、`initialize_kv_cache:7284`）。
> 所以：**步骤 1–5 今天就能过**；步骤 4 的「缺 runner 文件」这条 blocker 已经关闭——6–11 的阻塞
> 不再是缺文件或骨架，而是这些硬件路径**还没在真机上跑过**（未验证 ≠ 已坏）：每步列出的
> `HARDWARE:` 待核实项（数据形式的清单见 `worker.py` 的 `HARDWARE_CHECKLIST`）就是上机时要逐条
> 确认的东西。失败时按各步「失败归因」表定位，**不要**再假设会看到
> `NotImplementedError: TileInfer NPUWorker.<hook> is not implemented`——那四个 hook 已经不会抛它；
> 唯一仍会主动拒绝的是 V2 model runner（`--use-v2-model-runner` 被 `worker.py:689-696` 明确挡下）。
> 文末附了一张「每步今天能不能过」的表，先看一眼再逐条执行。
> **动手前先跑一次 `doctor`（见 §0）**：它是步骤 1–5 的自动版，红在哪一项就从本清单的哪一步开始修。

---

## 0. 约定（读命令前先看一眼）

```bash
# 四个变量按机器实际情况改一次，后面所有命令都用它们
export MODEL=/models/Qwen3.6-35B-A3B          # 权重目录（含 config.json）
export REPO=/data/TileInfer                    # 本仓库根目录
export PORT=8321                               # 服务端口
export TILELANG_CACHE_DIR=/data/tl_cache_ti    # kernel 缓存目录，必须是可写本地盘
cd "$REPO"
```

**第 0 步：先跑 `doctor`（步骤 1–5 的自动版，一条命令）。**

```bash
python -m tileinfer.vllm.doctor           # 推荐：不依赖 console script
tileinfer                                 # 装了 console script 时的同一入口（tileinfer.vllm.doctor:main）
tileinfer doctor                          # 同一命令；check / status 是它的别名
python -m tileinfer.vllm.doctor --json    # 机器可读版本，贴进报告/CI 日志
```

它把本清单步骤 1–5 里能在**单进程**内回答的问题一次问完：CANN 环境、`torch`/`torch_npu`、
设备可见性、TileLang **Ascend** wheel 与可写的 `TILELANG_CACHE_DIR`、vLLM 版本、插件的四段安装状态
（patch 报告 / attention backend 槽位 / 图桥 / 算子接线）与 ops 绑定、`reference/vllm-0.23/` 里的
runner，以及步骤 10 的图判定 `graphs.capture_path_ready`（这一项是 soft）。逐项打印
`PASS`/`FAIL`/`SKIP` 加一行 `fix:` 提示，末行是
`Start at docs/hardware-checklist.md step N: …`——**它指的那一步就是先要修的**；有 hard check 失败时
退出码非 0。两点边界：**步骤 3 的全量 pytest 故意不在 doctor 里启动**（几分钟的活会把报告淹掉），
doctor 全绿之后仍要手跑步骤 3；步骤 6 之后的硬件路径（权重/KV/图捕获）也要按下面的步骤做。
另注意：子命令词是**可选**的——`tileinfer`、`tileinfer doctor`（以及别名 `check` / `status`）完全等价，
都能和 `--json` / `--help` 组合；除此之外的未知参数仍按 usage error 处理（退出码 2）。

三条通用判读规则：

1. **一眼看插件装没装**：`grep "TileInfer vLLM plugin" /tmp/ti-serve.log`。vLLM 会在**每个进程**
   （API server / EngineCore / worker）加载一次 general plugin，每次一行 INFO。**没有这行 = 插件没跑**，
   后面所有 TileInfer 现象都不用解释。
2. **失败归因优先看「who」而不是「what」**：同一条报错在「平台没装上」和「内核写错」下都可能出现，
   所以每一步都先定位到进程/模块，再看数值。
3. **记下每一步的实际输出**（`>> /tmp/bringup.log`）。这份日志是后面跟基线对比的唯一凭据。

插件有三个可选环境变量（默认值在括号里，都是「坏了大不了降级」，不会让引擎起不来）：
`TILEINFER_VLLM_STRICT`（默认 `0`，把 patch skip 记成失败，doctor/CI 用）、
`TILEINFER_VLLM_REGISTER_BACKEND`（默认 `1`，是否注册 `AttentionBackendEnum.CUSTOM`）、
`TILEINFER_VLLM_GRAPH_BRIDGE`（默认 `1`，是否**委托** `tileinfer.vllm.graphs.install()` 把
`graphs.TORCH_CUDA_ALIASES` 里的 4 个**必需**别名（`CUDAGraph`、`graph`、`mem_get_info`、`Stream`）
加 5 个 **best-effort** 别名（`current_stream`、`stream`、`Event`、`OutOfMemoryError`、`set_stream`）桥到 `torch.npu`；
表在 `graphs.py` 里，`plugin.py` 只调用、不维护第二张表，见步骤 10）。

---

## 1. 环境：CANN、torch_npu、TileLang 缓存目录

```bash
# 1.1 CANN 环境（每个新 shell 都要 source，写进 ~/.bashrc 更省事）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info

# 1.2 设备可见性
echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"
python -c "import torch, torch_npu; \
print('torch', torch.__version__, '| torch_npu', torch_npu.__version__); \
print('available', torch.npu.is_available(), '| count', torch.npu.device_count()); \
print('name', torch.npu.get_device_name(0)); \
print('hbm_bytes', torch.npu.get_device_properties(0).total_memory)"

# 1.3 TileLang 缓存目录（预烧与在线必须是同一个目录）
: "${TILELANG_CACHE_DIR:?请先 export TILELANG_CACHE_DIR（见 §0）}"
echo "TILELANG_CACHE_DIR=$TILELANG_CACHE_DIR"
mkdir -p "$TILELANG_CACHE_DIR" && touch "$TILELANG_CACHE_DIR/.w" && rm "$TILELANG_CACHE_DIR/.w" \
  && echo "cache dir writable"
```

**期望输出**

| 检查 | 期望 |
|---|---|
| `npu-smi info` | 卡片信息表（950、HBM 容量、温度、功耗），无 `No devices found` |
| 1.2 | `available True`、`count 1`、`name` 含 `Ascend950`；`hbm_bytes` ≈ 卡容量 |
| 1.3 | `cache dir writable` |

**失败归因**

| 现象 | 指向 | 下一步 |
|---|---|---|
| `npu-smi: command not found` | CANN 没装或没 source | 重新 `source`；`ls /usr/local/Ascend/` |
| `available False` | 驱动/CANN 不匹配、卡被别人占用、或 `ASCEND_RT_VISIBLE_DEVICES` 设成了空 | `npu-smi info` 看进程列表；`echo $ASCEND_RT_VISIBLE_DEVICES` |
| `torch.npu` 不存在（AttributeError） | `torch_npu` 与 `torch` 版本不配对 | 按 CANN 版本的配套表重装 torch_npu |
| `get_device_name` 抛错（ABC 的 `get_device_name` 在 platform 里就是走这个 API） | torch_npu 变体缺 API | 记下 traceback，这就是 `platform.py` 里 `UNVERIFIED_NOTES` 要回答的问题之一 |
| `TILELANG_CACHE_DIR` unset/不可写 | 预烧白做：服务进程内首次编译 70–100 s 且**持有 GIL**（EngineCore 表现为「卡住」） | 设成一个本地可写目录，预烧与 serve 用同一个 |
| `total_memory` 报错或为 0 | `platform.get_device_total_memory` 的兜底（`mem_get_info`）会接手；若两者都失败，KV 池没法算 | 直接记 traceback，这是步骤 6 的前置 |

---

## 2. `import vllm`、`import tileinfer.vllm.plugin`、入口点可解析

```bash
python -c "import vllm; print('vLLM', vllm.__version__)"
python -c "import tileinfer.vllm.plugin as p; print('plugin import OK;', 'callable:', callable(p.install))"
python - <<'PY'
from importlib.metadata import entry_points
expected = {"vllm.platform_plugins": "tileinfer.vllm.platform:register",
            "vllm.general_plugins":  "tileinfer.vllm.plugin:install"}
for group, want in expected.items():
    found = [e for e in entry_points(group=group) if e.name == "tileinfer"]
    if not found:
        print(f"{group:24s} MISSING (expected tileinfer = {want})")
        continue
    for e in found:
        print(f"{group:24s} tileinfer = {e.value} | loadable: {callable(e.load())} "
              f"| matches pyproject: {e.value == want}")
PY
```

**期望输出**

```text
vLLM 0.23.x
plugin import OK; callable: True
vllm.platform_plugins    tileinfer = tileinfer.vllm.platform:register | loadable: True | matches pyproject: True
vllm.general_plugins     tileinfer = tileinfer.vllm.plugin:install   | loadable: True | matches pyproject: True
```

**失败归因**

| 现象 | 指向 | 下一步 |
|---|---|---|
| `import vllm` 失败 | 装的是别的版本 / 不在这个 venv | `pip show vllm`；确认 `>=0.23,<0.24` |
| `MISSING`（两组都没列） | 装的是**旧版** TileInfer：editable 安装不会自动刷新 entry points | `pip install -e .` 再跑一次；`pip show -f tileinfer \| head` |
| 只缺 `vllm.platform_plugins` | `pyproject.toml` 的该表没落地（或被别的发行版盖住） | 看 `pyproject.toml` 的两个 `[project.entry-points.…]` 表 |
| `loadable: False` | 入口点指向的模块**导入即失败**（典型：`tileinfer.vllm.platform` 在无 vLLM 环境下抛带版本说明的 ImportError） | 直接 `python -c "import tileinfer.vllm.platform"` 看 ImportError 原文 |
| `matches pyproject: False` | 机器上还有第二份 `tileinfer`（`PYTHONPATH` 或另一个 site-packages） | `python -c "import tileinfer; print(tileinfer.__file__)"`，只留一份 |

---

## 3. CPU 套件：全绿（含 9 个**既存**的收集错误）

```bash
# 3.1 全量（CPU box 上的标准入口）
python -m pytest tests -q -p no:cacheprovider 2>&1 | tail -20
```

**期望输出**

```text
ERROR tests/test_fused_sample.py
ERROR tests/test_fused_sample_integration.py
ERROR tests/test_greedy_fast_sampler.py
ERROR tests/test_moe_align_packer.py
ERROR tests/test_moe_prenz_pack.py
ERROR tests/test_moe_slot_layout.py
ERROR tests/test_moe_weight_prefetch.py
ERROR tests/test_prep_fastpath.py
ERROR tests/test_sw_pipeline_switch.py
!!!!!!! Interrupted: 9 errors during collection !!!!!!!
```

**这 9 个收集错误是既存的、与本次接线无关**：其中 8 个在模块级 `import tilelang`（CPU box 没有
TileLang Ascend 工具链），第 9 个 `test_prep_fastpath.py` 在模块级 `import numpy`（缺 numpy 的机器才会出现）。
项目文档把它们统称「9 个既存的 tilelang 收集错误」，记录在 `docs/notes/dev-env-windows.md`
「Pre-existing collection failures」。判读口径：**这 9 个以外的任何错误都是新的**。
想一次拿到完整计数就加 `--continue-on-collection-errors`（参考开发机：`979 tests collected, 9 errors`）。

```bash
# 3.2 本次接线的守门测试（必须 263 passed）
python -m pytest tests/test_no_vllm_ascend.py tests/test_layer_deps.py \
                 tests/test_platform_contract.py tests/test_attn_contract.py \
                 -q -p no:cacheprovider
```

**期望输出**：`263 passed`（39 + 43 + 122 + 59），无 failed/skipped 新增。旧的 `222 passed`
（57 + 43 + 39 + 83）已作废：`test_platform_contract.py` 加入 worker_cls / 架构接受 / 源契约后
从 83 涨到 122，`test_attn_contract.py` 从 57 涨到 59。**顺序有意义**：这里刻意让
`test_platform_contract` 先跑，fake `vllm` 必须在该顺序下也完整（`get_kv_cache_block_dim` 就是这么补上的）。

**失败归因**

| 现象 | 指向 | 下一步 |
|---|---|---|
| `test_no_vllm_ascend` 失败 | 新代码 `import vllm_ascend`（迁移目标是不再依赖它） | 报错里直接点名文件；把依赖搬进 `tileinfer/vllm/` |
| `test_layer_deps` 失败 | `serving/`、`attention/` 等引擎无关层 `import vllm` / 模块级 `import torch_npu` | 报错给出 `文件:行号`；把 import 挪进函数或放到 `tileinfer/vllm/` |
| `test_platform_contract` / `test_attn_contract` 失败 | vLLM 版本漂移（ABC 方法增减、字段改名） | 先看断言里的行号引用，再决定是改代码还是钉版本 |
| 收集阶段 `ImportError: No module named 'vllm'` | 在没装 vLLM 的 venv 里跑了守门测试（它们会注入 fake vllm，正常情况下不需要真 vLLM） | 确认 `PYTHONPATH`/venv 一致 |

---

## 4. 核对目标版本的 `gpu_model_runner.py`（**已入库**，只需核对版本与四个 `def`）

引擎要的四个 worker hook（`init_device` / `determine_available_memory` /
`initialize_from_config` / `compile_or_warm_up_model`）全部落在 model runner 上，而这份
`vllm/v1/worker/gpu_model_runner.py`（本项目转录的 v0.23 线）**已经在仓库里**：
`reference/vllm-0.23/vllm__v1__worker__gpu_model_runner.py`（334,736 B），其中
`_dummy_run:5599`、`profile_run:6168`、`capture_model:6525`、`initialize_kv_cache:7284` 就是四个
hook 的调用点。`tileinfer/vllm/worker.py` 的四个 hook 已按这些签名实现（该文件开头的 file:line
表即出处），所以这一步不再是「先取回文件、再靠猜签名」。

下面两条命令保留作**核对 / 重取**用：换机器、或怀疑 `reference/vllm-0.23/` 被别的版本覆盖时，用它们
确认文件与仓库记录的 v0.23 线一致（不需要网络）。

```bash
# 4.1 从**已安装的 vLLM wheel** 重取一份（不需要网络，版本天然正确；仓库里已有，重取前先比较）
python - <<'PY'
import pathlib, shutil, vllm
src = pathlib.Path(vllm.__file__).parent / "v1" / "worker" / "gpu_model_runner.py"
dst = pathlib.Path("reference/vllm-0.23/vllm__v1__worker__gpu_model_runner.py")
assert src.is_file(), f"not found: {src} (installed vllm is not 0.23.x?)"
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(src, dst)
print(f"{src} -> {dst} ({dst.stat().st_size} bytes)")
PY

# 4.2 校验：文件里必须有这四个定义（worker 的四个 hook 就是它们的调用方）
grep -n "def capture_model\|def profile_run\|def initialize_kv_cache\|def _dummy_run" \
     reference/vllm-0.23/vllm__v1__worker__gpu_model_runner.py
```

**期望输出**（当前仓库里的实际状态）：`334736` bytes，随后 4 行 `def …`，行号与上面那句一致
（`5599` / `6168` / `6525` / `7284`；换机器重取时行号相同即同一版本）。

**失败归因**

| 现象 | 指向 | 下一步 |
|---|---|---|
| `AssertionError: not found` | 装的 vLLM 不是 0.23.x（路径变了） | `pip show vllm`；`python -c "import vllm,pathlib;print(pathlib.Path(vllm.__file__).parent)"` 手工找 |
| `AttributeError: 'NoneType'… __file__` | vLLM 以 namespace/奇怪的可编辑方式安装 | `python -c "import vllm;print(vllm.__path__)"`，在 `__path__[0]` 下找 |
| 只有 1–2 个 `def`，或行号与本清单差很多 | `reference/vllm-0.23/` 里是旧/新大版本的 runner，签名与本项目转录不符 | 用 4.1 重取 `v0.23.0` 的版本；**不要**用别的版本来「先跑起来」 |
| 有网络时的替代 | — | `curl -sSL -o reference/vllm-0.23/vllm__v1__worker__gpu_model_runner.py https://raw.githubusercontent.com/vllm-project/vllm/v0.23.0/vllm/v1/worker/gpu_model_runner.py` |

> 核对通过后：`tileinfer/vllm/worker.py` 的四个 hook **已经**按这份文件的签名实现；`HARDWARE_CHECKLIST`
> 里剩的是上机待核实项，不再是「等文件」。步骤 6 之后若失败，第一处通常**不是** `NotImplementedError`：
> 没有 `torch_npu` 时 `NPUWorker.init_device` 会抛
> `RuntimeError: tileinfer.vllm.worker.init_device requires torch_npu …`（回步骤 1），有设备时才轮到
> 显存口径 / `dist_backend` / 图捕获这些项——按下面各步的「失败归因」走。

---

## 5. 平台被选中 + 配置校验通过 + block_size 对齐 2048

```bash
# 5.1 平台选择（必须在装了 vLLM、且能看到 NPU 的进程里）
python -c "from vllm.platforms import current_platform as p; \
print('platform', type(p).__module__ + '.' + type(p).__name__); \
print('device_type', p.device_type, '| enum', p._enum, '| out_of_tree', p.is_out_of_tree()); \
print('device_name', p.get_device_name(0)); print('total_memory', p.get_device_total_memory(0))"
```

```bash
# 5.2 配置校验（不加载权重，只走 config 解析 + platform.check_and_update_config）
python - <<'PY'
from vllm.engine.arg_utils import EngineArgs
cfg = EngineArgs(model="/models/Qwen3.6-35B-A3B", dtype="bfloat16", max_model_len=32768,
                 max_num_seqs=32, max_num_batched_tokens=8192).create_engine_config()
print("block_size      =", cfg.cache_config.block_size)
print("user_specified  =", cfg.cache_config.user_specified_block_size)
print("is_hybrid       =", cfg.model_config.is_hybrid)
print("worker_cls      =", cfg.parallel_config.worker_cls)
print("enforce_eager   =", cfg.model_config.enforce_eager)
PY
```

**期望输出**

```text
platform tileinfer.vllm.platform.TileInferPlatform
device_type npu | enum PlatformEnum.OOT | out_of_tree True
device_name Ascend950…
total_memory <整数，卡容量>
block_size      = 2048
user_specified  = False
is_hybrid       = True
worker_cls      = tileinfer.vllm.worker.NPUWorker
enforce_eager   = True            # 图捕获尚未就绪时平台会强制 eager（见步骤 10）
```

**失败归因**

| 现象 | 指向 | 下一步 |
|---|---|---|
| `platform …cuda.CudaPlatform` / `cpu.CpuPlatform` / `UnspecifiedPlatform` | `tileinfer.vllm.platform.register()` 返回了 `None`：**没有可见 NPU**，vLLM 于是回落到内置插件 | 回步骤 1；`register()` 只在 `torch.npu.device_count() > 0` 时返回类路径 |
| `RuntimeError: Only one platform plugin can be activated` | 同组入口点里有两个包都声称自己提供平台 | v3 只有 vllm-ascend 该注册平台入口；`pip show -f` 查同组入口点，卸掉多出来的那个（**不要卸 vllm-ascend**） |
| `block_size = 1024/…` 且 `user_specified True` | 命令行显式传了非 2048 倍数的 `--block-size`，平台按设计**拒绝**而不是偷偷改 | 去掉 `--block-size`（让平台对齐），或显式传 `2048` |
| `block_size` 非 2048 且 `is_hybrid True`、无报错 | 对齐逻辑没跑到（vLLM 没调 `check_and_update_config`） | 确认 `current_platform` 是 TileInferPlatform（5.1），否则平台根本没参与 |
| `worker_cls` 不是 `tileinfer.vllm.worker.NPUWorker` | 平台没写进 `parallel_config`。字段名现在是**已核实**的：`reference/vllm-0.23/vllm__config__parallel.py:248` 声明 `worker_cls: str = "auto"`，消费点是 `vllm__v1__worker__worker_base.py:249-253` | 回看 5.1 的 `current_platform`；若启动直接抛 `ValueError: parallel_config.worker_cls=… conflicts`，说明命令行显式传了别的 `--worker-cls`——去掉它（平台只在 `"auto"` 哨兵时改写，见 `docs/notes/handover-2026-10-28.md` §7.7） |
| 5.2 抛 `ValueError` 提到 MLA / sparse / sleep mode / speculative | 配置里带了 v0 明确不做的特性（见 `docs/notes/archive/plan-vllm-native-backend-v2.md` §0） | 去掉对应 flag；这类拒绝是**期望行为**，报错文本会直接点名 |
| 5.2 抛 `ValueError: kv-cache dtype …` | `--kv-cache-dtype` 不是 `auto` | 去掉该 flag（v0 只跑未量化） |

---

## 6. 权重加载 + KV 池分配（`npu-smi` 可见占用）+ 混合 spec（attention 池 + mamba state）

```bash
# 6.1 先预烧 kernel（必须在起服务**之前**、用同一个 TILELANG_CACHE_DIR）
TILELANG_CACHE_DIR=$TILELANG_CACHE_DIR python scripts/prewarm-tileinfer-kernels.py \
  --kv-heads 2 --group 8 --dim 256 --block-size 2048 \
  --num-engine-pages 2288 --max-num-reqs 32 \
  --max-num-batched-tokens 8192 --max-model-len 32768 --prefill 2>&1 | tail -5

# 6.2 起服务（前台，另开一个窗口做 6.3）
#     注：TILEINFER_METRICS_PATH 是遗留服务路径的在役探针；原生后端 v0 还没接线（见步骤 10.2）
TILELANG_CACHE_DIR=$TILELANG_CACHE_DIR \
TILEINFER_METRICS_PATH=/tmp/tileinfer_probe.json \
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name Qwen3.6-35B-A3B \
  --dtype bfloat16 --max-model-len 32768 --max-num-seqs 32 \
  --max-num-batched-tokens 8192 --block-size 2048 --port $PORT 2>&1 | tee /tmp/ti-serve.log

# 6.3 另开窗口：占用与日志证据
npu-smi info | sed -n '1,40p'
grep -n "TileInfer platform:" /tmp/ti-serve.log
grep -niE "kv cache|blocks|mamba|hybrid" /tmp/ti-serve.log | head -20
grep -n "TileInfer vLLM plugin" /tmp/ti-serve.log
```

**期望输出**

- 6.1 结尾是预烧计划摘要（`ready_plans == 预期桶数` 的等价物），没有 `bit_cast` 一类编译错误。
- 6.2 服务在 `Application startup complete` 之后才可访问；启动期日志里至少有：

  ```text
  TileInfer vLLM plugin installed | vLLM 0.23.x | strict=False | patches: N applied, M skipped (target gone), 0 failed | backend: registered … | graph: …
  TileInfer platform: device=npu block_size=2048 hybrid=True worker=tileinfer.vllm.worker.NPUWorker enforce_eager=True
  TileInfer hybrid KV: block_size=2048 mamba_block_size=2048 mamba_page_size_padded=4194304 (attention page = 2*2 kv_heads*256 head_dim = 2048 B/token)
  ```

- 6.3 `npu-smi info` 的 HBM 占用从「空闲」变为「权重 + KV 池」，且**起服务前后差值是几十 GB 量级**。
- 混合 spec：40 层里 10 层拿到 NHD attention 池（逻辑形状 `(blocks, 2, 2048, 2, 256)`，物理顺序
  `get_kv_cache_stride_order() == (1, 0, 2, 3, 4)` = `[2][blocks][page][heads][dim]`），30 层拿到 mamba state
  （`conv_state` + `ssm_state`，页对齐 2048）。vLLM 的日志措辞会随版本变，**判据是数量**：10 + 30 = 40，
  不是 40 层都进同一个池。

**失败归因**

| 现象 | 指向 | 下一步 |
|---|---|---|
| `NotImplementedError: … only implements the V1 runner …` | 开了 `--use-v2-model-runner`：TileInfer 只有 V1 runner 的子类，`worker.py:689-696` 明确拒绝 V2 | 去掉该 flag；这是设计边界，不是环境问题 |
| `RuntimeError: tileinfer.vllm.worker.init_device requires torch_npu` | 四个 hook 已实现，缺的是设备栈：`torch_npu` 不在这个 venv，或与 torch 不配对 | 回步骤 1.1 / 1.2：source CANN、装配套的 `torch_npu` |
| `RuntimeError: Not support device type`（来自 `gpu_worker.py`） | 平台没把 `worker_cls` 写进去，vLLM 用了自己的 CUDA worker | 回步骤 5.2 看 `worker_cls` |
| 权重加载到一半 `torch.OutOfMemoryError` | 池子/alive 权重超过卡容量（`--gpu-memory-utilization`、`--max-model-len` 太大） | 降 `--gpu-memory-utilization` 或 `--max-model-len` |
| `npu-smi` 看不到占用增长 | 权重没真正上卡（meta device 加载失败），或 KV 池没分配：`initialize_from_config`（`worker.py:1712`）把 `KVCacheConfig` 转交 runner，失败会从 `model_runner.initialize_kv_cache` 抛出 | 搜日志 traceback；确认步骤 5.2 的 `worker_cls` 是 `NPUWorker`，再看 `TileInfer: KV pool initialised by the runner` 那行 |
| KV 块数为 0 / 极小 | `determine_available_memory`（`worker.py:1533`）量错了口径：`torch.npu.mem_get_info` / `allocated_bytes.all.peak` 的拼写或统计范围不对，或权重记账把池吃光 | 看 `TileInfer: Memory accounting:` 与 `available KV cache` 两行，数值对 `npu-smi info`（`HARDWARE_CHECKLIST` 第 3 条） |
| 只有 attention 层进池、mamba 层缺失 | `support_hybrid_kv_cache()` 现在返回 `True`（`platform.py`）；若 mamba 层仍缺失，是 `check_and_update_config` 没跑到（步骤 5.1）或 mamba spec 构建失败，不是能力位的问题 | 看 `TileInfer hybrid KV:` 那行的 `block_size` / `mamba_block_size` / `mamba_page_size_padded`；判据仍是 10 + 30 = 40 层（分组数由 vLLM 的 page-size 分组决定） |
| 首个 forward 报 `KV pool must be contiguous for the zero-copy sub-page view` | 池不是 block-first 物理序。本后端声明 `get_kv_cache_shape = (blocks, 2, page, heads, dim)` + `get_kv_cache_stride_order = (1, 0, 2, 3, 4)`（`attn/backend.py:168-231`），vLLM 因此**跳过** hybrid 重写（`gpu_model_runner.py:7190-7191`） | 先确认 `TileInferAttentionBackend` 来自本仓库、且没有 `block_dim == 1` 的旧实现；旧的「W7 剩余工作」已关闭，再现即旧 wheel/旧副本 |
| 启动期报 `TileInfer cannot start this model: N mamba layer(s) use vLLM's Mamba SSU dispatch` | 只有 `MambaSpec(mamba_type=MAMBA1/MAMBA2)` 才会进 `initialize_mamba_ssu_backend`（`ssu_dispatch.py:202-208`）；vLLM 0.23 只支持 `triton`/`flashinfer`，且 `_BACKEND_REGISTRY` 是私有 dict、没有注册钩子（`ssu_dispatch.py:185-188`），本仓库无 NPU SSU 实现 | 这是**预期拦截**：Qwen3.6 的 GDN 层是 `GDN_ATTN`，不会走到这里；若真在 MAMBA1/2 模型上看到，二选一——换 GDN/linear-attention 混合模型，或上游加 NPU SSU 后端（非配置项） |
| 预烧 6.1 报 `no member named 'bit_cast'` | TileLang wheel 的 `numeric_limits.h` 缺 `<bit>`（已知缺陷） | `python scripts/patch-tilelang-ascend-bitcast.py` 后重跑 |

---

## 7. 第一个 decode step：logprobs 对齐冻结基线

```bash
# 7.0 先在**现网（vllm-ascend）**服务上冻结基线，存成文件（同一模型、同一 prompt、temperature=0）
PROMPT='{"model":"Qwen3.6-35B-A3B","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":1,"temperature":0,"logprobs":true,"top_logprobs":5}'
curl -s http://127.0.0.1:8321/v1/chat/completions -H 'Content-Type: application/json' \
     -d "$PROMPT" > /tmp/baseline_fia.json   # 现网端口

# 7.1 TileInfer 服务上同一请求
curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
     -d "$PROMPT" > /tmp/baseline_ti.json

# 7.2 逐位比较（4 位小数，与项目既有口径一致）
python - <<'PY'
import json
def top(path):
    d = json.load(open(path))
    items = (d["choices"][0].get("logprobs") or {}).get("content") or []
    return [(t["token"], round(t["logprob"], 4)) for t in (items[0].get("top_logprobs") or [])]
fia, ti = top("/tmp/baseline_fia.json"), top("/tmp/baseline_ti.json")
print("FIA:", fia)
print(" TI:", ti)
print("top-5 token 集合一致:", {t for t, _ in fia} == {t for t, _ in ti})
print("MATCH(4dp):", fia == ti)
PY
```

> 逐 prompt 的机器判据用 `benchmarks/ab_report.py`：`--compare OBS BASELINE --atol 1e-4` 输出每个 prompt 的 `max_abs_delta`，缺失 prompt 直接算错误而不是静默通过。
> 自动驱动两侧服务、抓取 logprobs 的封装 `benchmarks/ab_native_vs_ascend.py` 是后续任务，本步骤仍按 7.0–7.2 手工执行。

**期望输出**：两者 token 集合一致、`MATCH(4dp): True`（历史上 Qwen2.5 的口径就是 4 位小数逐位一致，
见 `docs/status.md`；Qwen3.6 需要你先在 7.0 冻结，**不要**用记忆里的数字）。

**失败归因**

| 现象 | 指向 | 下一步 |
|---|---|---|
| token 完全不同 | 输出通道坏了：attention 选错层、KV 写错位置、block table 错位 | 步骤 6 的 spec 数量 + 步骤 8 的 GDN 状态；再看 `attn/kv_cache.py` 的 `slot_mapping` 写法 |
| token 相同、`logprob` 差在第 3–4 位 | 数值路径差异：head layout（GQA group）、softmax 精度、KV 布局（NHD vs HND） | 确认 `head_size=256`、`group=8`、`get_kv_cache_shape` 是 `(blocks, 2, 2048, 2, 256)`（物理序 `(1,0,2,3,4)`） |
| 出现 `nan` / `-inf` | 数值已经坏了（GDN 或 attention） | 直接做步骤 8，别在采样器上找 |
| 只有第一个 token 对、后面漂 | 状态更新错（mamba state / KV 逐 step 写） | 步骤 8 的「每步有限性」探针 |
| 7.0 的基线服务自己就 `EngineDeadError` | 现网侧不稳定（历史上出现过），基线不可信 | 先修/重启基线服务再冻结；**不要**拿坏基线当判据 |

---

## 8. GDN 路径：30 层线性注意力的 state 必须是有限值

GDN 层的 state 在 `QwenGatedDeltaNetAttention.kv_cache`（`[conv_state, ssm_state]`，
见 `reference/vllm-0.23/vllm__model_executor__layers__mamba__gdn__qwen_gdn_linear_attn.py:1308-1316`）。
必须在**同一个进程**里包一层断言——所以用离线 `LLM` 跑，而不是给 `vllm serve` 外挂探针。

```bash
cat > /tmp/gdn_state_probe.py <<'PY'
"""断言第 1 个 decode step 之后 30 个 GDN 层的 state 全有限。"""
import sys, torch
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention

_orig = QwenGatedDeltaNetAttention.forward
seen, bad = {}, []

def checked(self, *a, **kw):
    out = _orig(self, *a, **kw)
    cache = getattr(self, "kv_cache", None)
    if cache is None:
        return out
    conv, ssm = cache[0], cache[1]
    n_conv = int(torch.isfinite(conv).all()) if conv is not None else 1
    n_ssm = int(torch.isfinite(ssm).all()) if ssm is not None else 1
    seen[id(self)] = (getattr(self, "prefix", "?"), n_conv, n_ssm,
                      float(ssm.abs().max()) if ssm is not None else 0.0)
    if not (n_conv and n_ssm):
        bad.append(seen[id(self)])
    return out

QwenGatedDeltaNetAttention.forward = checked

from vllm import LLM, SamplingParams
llm = LLM(model="/models/Qwen3.6-35B-A3B", dtype="bfloat16", max_model_len=4096,
          enforce_eager=True, gpu_memory_utilization=0.9)
llm.generate(["The capital of France is"], SamplingParams(max_tokens=1, temperature=0))

print(f"checked {len(seen)} GDN layers, non-finite {len(bad)}")
for layer, c, s, mx in seen.values():
    print(f"  {layer:40s} conv_finite={c} ssm_finite={s} |ssm|max={mx:.4g}")
sys.exit(1 if bad or len(seen) != 30 else 0)
PY
TILELANG_CACHE_DIR=$TILELANG_CACHE_DIR python /tmp/gdn_state_probe.py 2>&1 | tail -40
```

**期望输出**

```text
checked 30 GDN layers, non-finite 0
  model.layers.0.linear_attn   conv_finite=1 ssm_finite=1 |ssm|max=…
  …（共 30 行）
```

退出码 0。**只在第 1 步之后断言**：后续 step 的 state 会按设计累积，仍然必须有限，但「第 1 步就 NaN」
是最容易归因的信号。

**失败归因**

| 现象 | 指向 | 下一步 |
|---|---|---|
| `checked 27/29/…`（≠30） | 有的 GDN 层没走 `forward`（被别的实现替换/没进图） | 打印 `prefix` 列表与模型 config 的 `linear_attention` 层号对照 |
| `ssm_finite=0` | 递归/门控数值坏了：`fused_recurrent_gated_delta_rule`、`fused_sigmoid_gating_delta_rule_update` 这一族 | 单算子对拍（`tileinfer/vllm/ops/fla/`），先 CPU oracle 再上卡 |
| `conv_finite=0` | `causal_conv1d_fn` / `causal_conv1d_update` 的布局约定搞错（`[dim, T]` vs token-major） | 看 `docs/notes/vllm-api-0.23.md` 里两条入口的转录 |
| 有限但 `abs(ssm).max()` 巨大（>1e4），或后一步立刻 NaN | gating 的 `softplus`/`beta` 约定错（形状对、语义错，最隐蔽） | 用 fp64 CPU 参考对拍同一组 q/k/v/g/beta |
| 探针根本没打印 | 包错了类（vLLM 版本里类名/模块路径变了） | `python -c "import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as m; print([n for n in dir(m) if 'Delta' in n])"` |

---

## 9. MoE 路径：256 专家路由与 CPU oracle 在同一个 batch 上一致

```bash
# 9.1 CPU oracle（已经在仓库里：256 专家的 topk_softmax + fused_experts 用例）
python -m pytest tests/test_moe_ops.py -q -k "256" -p no:cacheprovider
```

**期望输出**：`3 passed`（`test_topk_softmax_large_256_expert`、`test_fused_experts_256_experts_top8` 等）。

```bash
# 9.2 设备 vs CPU：同一组 router logits，top-k 的 id 必须逐元素相等
cat > /tmp/moe_route_probe.py <<'PY'
import torch, torch_npu
from tileinfer.vllm.ops.moe import topk_softmax

torch.manual_seed(0)
num_tokens, num_experts, top_k = 64, 256, 8      # 一个 decode batch 的几何
logits = torch.randn(num_tokens, num_experts, dtype=torch.float32)

def run(x):
    w = torch.empty(num_tokens, top_k, dtype=torch.float32, device=x.device)
    ids = torch.empty(num_tokens, top_k, dtype=torch.int32, device=x.device)
    tei = torch.empty(num_tokens, top_k, dtype=torch.int32, device=x.device)
    topk_softmax(w, ids, tei, x)
    return w, ids, tei

w_cpu, ids_cpu, _ = run(logits)
w_npu, ids_npu, _ = run(logits.npu())
print("ids equal:", torch.equal(ids_cpu, ids_npu.cpu()))
print("weights max|diff|:", float((w_cpu - w_npu.cpu()).abs().max()))
print("distinct experts used:", int(ids_cpu.unique().numel()), "/ 256")
assert torch.equal(ids_cpu, ids_npu.cpu()), "routing 不一致 -> 先看 tie-break 与 logits 布局"
assert float((w_cpu - w_npu.cpu()).abs().max()) < 1e-6
print("OK")
PY
python /tmp/moe_route_probe.py
```

**期望输出**：`ids equal: True`、`weights max|diff|: <1e-7 量级>`、`distinct experts used` 接近 top_k × 行数
（随机 logits 下 64×8=512 个槽位，去重后 200+）、`OK`。

然后在**引擎里**再确认一次：步骤 7 的 top-1 token 与基线一致，就说明整条 `router → topk_softmax →
moe_align_block_size → fused_experts → 共享专家` 链在真实 batch 上是通的（MoE 错一定改输出）。

**失败归因**

| 现象 | 指向 | 下一步 |
|---|---|---|
| 9.1 失败 | CPU oracle 本身坏了（不是设备问题） | 先修 CPU 侧；设备结论无效 |
| `ids equal: False`，但每行**集合**相同、只有顺序不同 | tie/顺序约定差异（本项目规定「并列取小 index」） | 检查是否是并列构造的场景；随机 logits 不该出现 |
| `ids equal: False`，且每行集合不同 | logits 布局错（`[tokens, experts]` 被当成 `[experts, tokens]`）或 dtype 转换丢精度 | 打印两条路径的 `softmax` 前 3 行对比 |
| `weights` 的 `max abs diff` 在 1e-3 量级 | 权重做了 bf16 往返（可以接受）或 `renormalize` 开关不一致 | 统一 fp32 输入判断；确认引擎侧的 renormalize 语义 |
| 引擎出字但内容乱、路由探针本身 OK | 不是路由，是专家计算 / 共享专家折叠 | 看 `fused_experts` 的 CPU oracle 与引擎内调用点 |

---

## 10. eager vs 图捕获：`capture_calls > 0` 且 `fallbacks == 0`

前提：**先确认插件的图桥真的就绪**。

- **别名表只有一张，在 `tileinfer/vllm/graphs.py`**：`graphs.TORCH_CUDA_ALIASES`，4 个**必需**
  （`CUDAGraph`、`graph` = 捕获；`mem_get_info`、`Stream` = runner 的显存 profile 与 async copy 流）
  加 5 个 **best-effort**（`current_stream`、`stream`、`Event`、`OutOfMemoryError`、`set_stream`）。必需集是
  **全有或全无**：`graphs.install()` 要么把它们全写进 `torch.cuda`、要么一个都不留（写失败会回滚），
  `graphs.capture_path_ready()` 再按**对象 identity** 逐个核对（look-alike 不算）；best-effort 写不
  进去只产生一条 warning 和一句 `best-effort skipped: …`，不影响判定——用到它们的那几条 runner 路径
  可以关着。
- **`plugin.py` 只做委托**：`_install_graph_bridge()` 先问 `graphs.capture_path_ready()`；只有在它说
  「没就绪」且 `TILEINFER_VLLM_GRAPH_BRIDGE` 没被设成 `0` 时才调 `graphs.install()`，然后**再问一次**
  同一判据（日志里对应 `TileInfer graph bridge ready:` 或
  `bridge … but capture_path_ready() is still False`）。`plugin.uninstall()` 也只是调
  `graphs.uninstall()`，后者只还原它自己写过的属性，torch_npu 装的桥不动。因为**表和判据同属一个
  模块**，「桥装了但判定还是 False」只剩「某个必需别名被别的库改回去」或「`torch.npu.graph_pool_handle`
  缺失」两种解释，不会出现「两张表各说各话」。
- 图没就绪时平台按设计强制 `enforce_eager=True`——此时 `capture_calls == 0` 是**决策**，不是故障。

```bash
# 10.1 本进程的插件状态（patch 报告 + 图就绪 + 后端类路径）
#      先显式跑一次 general-plugin 加载，否则 status() 只会如实回答「本进程没装」
python - <<'PY' | sed -n '1,80p'
import json
try:
    import vllm.plugins as vp
    vp.load_general_plugins()          # 与引擎启动时同一个入口
except Exception as exc:               # 这个 vLLM 的 loader 名字变了
    print("load_general_plugins unavailable:", repr(exc))
    from tileinfer.vllm.plugin import install
    install()                          # 直接调入口，效果等同
from tileinfer.vllm.plugin import status
print(json.dumps(status(), indent=2, ensure_ascii=False))
PY

# 10.2 服务起来后：探针里的捕获计数（需要步骤 6 的 TILEINFER_METRICS_PATH）
curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B","messages":[{"role":"user","content":"hi"}],"max_tokens":32,"temperature":0}' > /dev/null
python -c "import json;m=json.load(open('/tmp/tileinfer_probe.json'));\
print({k:m.get(k) for k in ('ready_plans','capture_calls','eager_calls','fallbacks','exceptions')})"

# 10.3 同一 prompt 的图 / eager 输出一致（图模式不能改数值）
#     eager：加 --enforce-eager 重启一次；图：不加。两次都用步骤 7.2 的比较脚本。
```

**期望输出**

- 10.1：`graphs.capture_path_ready: true`、`graphs.bridge_installed: true`、`backend.registered: true`。
  `graphs.active_aliases` 里应能看到那 4 个**必需**别名（`torch.cuda.CUDAGraph`、`torch.cuda.graph`、
  `torch.cuda.mem_get_info`、`torch.cuda.Stream`）；5 个 best-effort 的也在更好，
  缺谁由 `graphs.bridge_detail` 的 `best-effort skipped: …` 点名。
  `unverified_symbols` 是 `graphs.py` 里那份「必须在硬件上核实」的**静态清单**（5 条常量），
  它出现在输出里不代表失败——真正的判据是 `capture_path_ready`（对象级 identity 检查，核对**必需的
  四个**）和 `backend.registered`（`AttentionBackendEnum.CUSTOM.get_class()` 的 identity 检查）。
- 10.2：`capture_calls > 0`、`fallbacks == 0`、`exceptions == 0`、`ready_plans >= 1`。
  ⚠️ 这个 JSON 目前由**遗留服务路径**的计数线程产出（`TILEINFER_METRICS_PATH`）；原生后端 v0 还没接线，
  所以文件不存在是**待办**而不是「没进图」。在没有这个文件之前，用日志里的
  `TileInfer graph bridge ready:` + 步骤 10.3 的图/eager 数值一致性来判图模式，
  并在报告里注明「计数未接线」。
- 10.3：两次 top-5 logprobs 完全一致。

**失败归因**

| 现象 | 指向 | 下一步 |
|---|---|---|
| 10.1 `capture_path_ready: false` | `reason` 直接点名：没有 accelerator；`torch.npu` 缺 4 个**必需**别名里的某一个（`CUDAGraph`/`graph`/`mem_get_info`/`Stream`）；缺 `graph_pool_handle`；或别名都在但没桥（`TILEINFER_VLLM_GRAPH_BRIDGE=0`） | 按 reason 走：缺原语 → 只能自写 wrapper（`graphs.py` 的 branch 3）；缺 `graph_pool_handle` → 先覆盖 `TileInferPlatform.get_global_graph_pool`；只有 best-effort 缺 → 只影响 async copy / OOM 兜底，图判定不受影响（`graphs.bridge_detail` 会点名）；没桥 → 查 `TILEINFER_VLLM_GRAPH_BRIDGE` 是否被设成 0 |
| `bridge_installed: true` 但 `capture_path_ready: false` | 表与判据同属 `graphs.py`，所以只剩两种原因：某个**必需**别名在 install 之后被别的库/插件改了回去，或 `torch.npu.graph_pool_handle` 缺失（它不是别名表的一员，但 `capture_path_ready` 要它） | 看日志里那句 `bridge … but capture_path_ready() is still False`（reason 会点名），别信 capture 数字；再用 10.1 的 `graphs.active_aliases` 对比少了谁 |
| `capture_calls == 0`、`eager_calls > 0` | 引擎仍在 eager：平台的强制（图未就绪）或 `--enforce-eager` / `cudagraph_mode=NONE` | 回 10.1；再 `grep enforce_eager /tmp/ti-serve.log` |
| `fallbacks > 0` | 捕获区内走了降级路径（最常见：该 bucket 的 plan 没预烧） | 用与 serve 相同的 `--num-engine-pages`/`--max-num-reqs` 重跑步骤 6.1 预烧；再对 `graphs.capture_contract_ok` 的 violation code 定位 |
| `exceptions > 0` | 运行期熔断了（内核异常被兜底） | 搜日志里的第一处 exception；这是正确性问题，先别谈性能 |
| 10.2 的探针 JSON 不存在 | 原生后端的计数还没接线（遗留路径才写 `TILEINFER_METRICS_PATH`） | 改用 10.3 的图/eager 数值一致性 + 日志 `graph bridge ready`；报告里注明计数未接线 |
| 图/eager 输出不一致 | 捕获区内做了不该做的事（编译/分配/同步）或 KV 地址在 replay 后失效 | 用 `tileinfer/vllm/graphs.py` 的 `CaptureState` 断言四个计数 |

---

## 11. 性能：TTFT / TPOT 对比现网 vllm-ascend 基线

在同机、同模型、同参数下分别对**两个**服务跑同一套脚本（仓库自带）：

```bash
# 11.1 现网（vllm-ascend）基线
python benchmarks/ab_qwen36_full.py --label fia --port 8321 --model Qwen3.6-35B-A3B \
  --outdir /tmp/ab --runs 3 --warmup 1

# 11.2 TileInfer（--probe 指向步骤 6 的探针 JSON：数字只有在 fallbacks 增量为 0 时才算数；
#      该文件还没接线时直接去掉 --probe，并在报告里注明「探针不可用」）
python benchmarks/ab_qwen36_full.py --label ti --port $PORT --model Qwen3.6-35B-A3B \
  --outdir /tmp/ab --runs 3 --warmup 1 --probe /tmp/tileinfer_probe.json

# 11.3 汇总对比（TTFT/TPOT 单位 ms；TPOT = (latency - ttft) / (tokens - 1)）
python - <<'PY'
import json, glob
for path in sorted(glob.glob("/tmp/ab/*.json")):
    r = json.load(open(path))
    print("==", r.get("label"), "| top5:", r.get("top5_logprobs"))
    for s in r.get("scenarios", []):
        print(f"  {s['scenario']:22s} ttft_p50={s.get('ttft_ms_p50')} tpot_p50={s.get('tpot_ms_p50')}")
PY
```

> 汇总与准入用 `benchmarks/ab_report.py`：`--summarize` 出 median/p90，`--compare` 对基线，`--check report.json` 在任一侧未 ready、`fallbacks > 0` 或 `cold` 时退出 1（该轮数字不得写进报告）。
> 自动驱动两侧服务并落盘的 `benchmarks/ab_native_vs_ascend.py` 是后续任务，目前仍按 11.1/11.2 手工跑。

**期望输出**

- 两侧 `top5_logprobs` 一致（同步骤 7）。
- 短上下文（1 tok / 128 tok / 512 tok）：`TI / FIA` 的 TTFT 与 TPOT 比值 **≈ 1.0（目标 ≤ 1.0）**。
  仓库历史记录：短上下文 0.91–1.09；128 tok 生成 0.94x；并发 4 路 0.70x（调度差异）。
- 长上下文（8k / 16k）：**允许落后**（结构性原因：split-KV 默认关闭，KI-1）。把比值记进报告，
  本阶段不追平。

**失败归因**

| 现象 | 指向 | 下一步 |
|---|---|---|
| TTFT 出现几十秒的离群值 | 测量窗口内有 kernel 编译（预烧没覆盖该 bucket / 缓存目录不一致） | 检查两侧 `TILELANG_CACHE_DIR` 相同、预烧参数与 serve 一致；重跑前清空探针 |
| `TI` 的 TPOT 明显劣化且 `fallbacks > 0` | 没进图 / 每条走了降级路径 | 回步骤 10，别在性能上找原因 |
| `TI` 的 TPOT 劣化、`fallbacks == 0`、`capture_calls > 0` | 真·内核/schedule 性能问题 | 分场景定位（decode d256、GDN、MoE），对照 `docs/performance.md` 的方法论 |
| 并发场景吞吐差距大、单流正常 | 批处理/调度（bucket、`max_num_seqs`），不是内核 | 看探针 `ready_plans` 与桶命中率 |
| 两侧数字都抖得厉害 | 外部环境退化（历史上有过） | 用双侧**比值**口径记录，绝对值标记为不可信 |

---

## 首次失败分类表（first failure triage）

按「症状 → 最可能原因 → 第一条要跑的命令」查；一条跑不通再往下一条。

| # | 症状（原样搜日志） | 最可能原因 | 先跑这条 |
|---|---|---|---|
| 1 | `ImportError: tileinfer.vllm.platform requires vLLM 0.23's platform API` | vLLM 不是 0.23.x，或入口点来自旧安装 | `pip show vllm tileinfer`，再回步骤 2 |
| 2 | `current_platform` 是 `CudaPlatform` / `CpuPlatform` / `UnspecifiedPlatform` | `register()` 返回 `None`：没有可见 NPU | 步骤 1.2 的 `torch.npu.is_available()` |
| 3 | `RuntimeError: Only one platform plugin can be activated` | 同组入口点里有两个包都提供平台 | v3 只有 vllm-ascend 注册平台入口；查同组入口点卸掉多出来的那个（**不要卸 vllm-ascend**） |
| 4 | `RuntimeError: Not support device type` | `parallel_config.worker_cls` 没被平台改写（机制已核实：字段默认 `"auto"`，平台在 `check_and_update_config` 里替换） | 步骤 5.2 打印 `worker_cls`，并搜日志 `TileInfer platform: … worker=` |
| 5 | `RuntimeError: tileinfer.vllm.worker.init_device requires torch_npu` | 设备栈缺失（`torch_npu` 没装/不配对）；四个 hook 已实现，不再抛 `NotImplementedError` | 步骤 1.1 / 1.2 |
| 6 | `block_size … is not a multiple of 2048` | 显式传了非 2048 倍数的 `--block-size` | 去掉 `--block-size`，或传 `2048` |
| 7 | 首个请求卡 70–100 s 后正常 | 服务进程内编译 kernel（缓存未命中，编译持有 GIL） | 步骤 6.1 预烧；确认两侧 `TILELANG_CACHE_DIR` 相同 |
| 8 | 生成内容出现 `nan` / `-inf` / 乱码 | KV 写入或 GDN state 数值坏了 | 步骤 8 的 `/tmp/gdn_state_probe.py`；再步骤 7.2 |
| 9 | `Backend CUSTOM must be registered before use` | 该进程没跑 general plugin（入口点缺失），或 `TILEINFER_VLLM_REGISTER_BACKEND=0` | `grep "TileInfer vLLM plugin"` 日志；步骤 2 |
| 10 | `capture_calls == 0` 且 `eager_calls > 0` | 图未就绪 → 平台强制 eager（设计行为） | 步骤 10.1 `status()["graphs"]` 的 `reason` |
| 11 | `fallbacks > 0` | plan 没为该 bucket 预烧，捕获区内降级 | 步骤 6.1 用与 serve 相同的页数/请求数重烧 |
| 12 | `npu-smi` HBM 无增长 | 权重没上卡 / KV 池没分配（`initialize_from_config` → `model_runner.initialize_kv_cache` 抛错） | 步骤 6.3 的三条 grep + traceback |
| 13 | 路由 id 与 CPU 不一致 | logits 布局或 tie 规则 | 步骤 9.2 探针 |
| 14 | `test_no_vllm_ascend` 变红 | `tileinfer/vllm/` 里出现了 `import vllm_ascend` | 步骤 3.2，报错点名文件 |
| 15 | 所有 TileInfer 现象都不出现 | **插件根本没加载**（最常见、也最容易被当成内核 bug） | `grep "TileInfer vLLM plugin" /tmp/ti-serve.log` |

---

## 附：本清单的「完成」定义

| 步骤 | 完成判据 | 今天能过吗 |
|---|---|---|
| 1–3 | 命令输出与期望一致 | ✅ 能 |
| 4 | `reference/vllm-0.23/` 里有 runner 且 4 个 `def` 命中 | ✅ 已满足（334,736 B；`5599`/`6168`/`6525`/`7284`） |
| 5 | `TileInferPlatform` + `block_size=2048` + `worker_cls` 正确 | ✅ 能 |
| 6 | 权重加载 + KV 池 + `npu-smi` 占用 + 10/30 混合 spec | 🔶 工件就位（runner 在库、四个 hook 已实现），**未上机验证** |
| 7 | top-5 logprobs 与冻结基线 4 位小数一致 | 🔶 未上机验证（依赖 6） |
| 8 | 30 层 GDN state 全有限 | 🔶 未上机验证（依赖 6） |
| 9 | 256 专家路由与 CPU oracle 一致 | 🟡 9.1/9.2 今天能过；引擎内一致性待 6 起来后验证 |
| 10 | `capture_calls > 0 && fallbacks == 0` | 🔶 未上机验证（依赖 6） |
| 11 | TTFT/TPOT 与现网比值落盘 | 🔶 未上机验证（依赖 7–10） |

> 🔶 不是「环境坏了」，也不再是「缺文件 / 骨架」：6–11 的阻塞是这些硬件路径**尚未在真机上跑过**
> （v0 的施工顺序见 `docs/notes/archive/plan-vllm-native-backend-v2.md` §9；`docs/notes/handover-2026-10-28.md` 的 worker
> 行仍写着「骨架」，是更早的状态，以本清单与 `tileinfer/vllm/worker.py` 为准）。
> 每完成一格，把实际输出追加到 `/tmp/bringup.log`，并把本节表格的状态改成 ✅。
