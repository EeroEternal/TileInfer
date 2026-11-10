# 工作区清理记录（2026-11-10）

> 目的：让"打开仓库就知道该看哪、该跑哪"重新成立。项目定位换过几轮，仓库里因此堆了
> 三份互相矛盾的世界观（内核库 / vllm-ascend 插件 / 原生后端）和大量过程日志。
> 本次只做清理与纠错，不改任何算法或性能行为。

---

## 1. 磁盘（未进 git 的产物）

| 删掉 | 大小 | 为什么 |
|---|---|---|
| `.work/native30_mmap/` | 442 GB | DeepSeek-V4.1 的 mmap 权重转储（两个 92 GB engram + 几十个 4.3 GB w13）。能从 checkpoint 重新生成，与 Qwen3.6 无关 |
| `.work/cann920/` | 35 GB | CANN 9.2 安装包，在用的是 `.work/cann930` |
| `.cann910/` | 19 GB | CANN 9.1 展开目录，无人引用 |
| `.agent_loop/` | 17 GB | 旧的 agent-loop 驱动与全部运行产物。**唯一有价值的是 `run/env.sh`**，已抢救为 `scripts/env-ascend102.sh` 并接线到 `tests/conftest.py` 与 `scripts/run_device_ci.sh` |
| `extra-info/` + `export_only_prof_dir/` | 670 MB | NPU data dump 与 profiling 导出 |
| `.dsv41/` | 984 MB | DSV4.1 的隔离 venv + 第三方源码副本；venv 先删（见下），剩下的源码可重新下载 |
| `.venv/`、`.venv_ascend/`、`.venv_ascend30/` | 13.6 GB | 三个 venv：`.venv` 是 DSV4.1 的、`.venv_ascend30` 是它的 0.30 尝试、`.venv_ascend` 与 `/home/dlz/.venv_ascend` 重复 |
| `.pytest_cache/`、`trace_out/`、`.tilelang_cache/`、`.work/` 下 200+ 个日志/脚本/JSON | ~1 GB | 一次性探针与试跑残留 |
| `inventory-report.txt`、`h50_dense_bench.json`、`h79_stride_bench.json`、`opencode.json` | 小 | 旧探针输出；`opencode.json` 里还有两个明文 API key（**建议轮换**） |

**保留**：`.work/cann930`（在用的 CANN）、`glibcxx`、`gcc13`、`wheels`、`triton_cache`、
`DeepGEMM-Ascend`、`tilelang-ascend`；`.venv_native213`（当前后端）、`.venv_baseline`（A/B 基线）。

项目目录：**560 GB → 30 GB**；`/home` 使用率 58% → 41%。

---

## 2. 仓库里删掉的文件

| 删除 | 行数 | 为什么 |
|---|---|---|
| `docs/performance.md` 原内容 | 3016 | append 出来的流水账，混着 Qwen2.5 时代与 Qwen3.6 时代的数字，文首自己写着"绝对值不可混读"。**重写**为口径/基线/当前数字/测量方法 |
| `docs/plan-performance.md` | 4252 | 同上，工作日志形态。内容并入新方案与 `performance.md` |
| `docs/design.md` | 161 | 写着"引擎无关的内核库、不绑定推理引擎"，与当前定位相反 |
| `docs/next-tasks.md`、`docs/roadmap.md` | 379 | 停在 2026-10-13 的 vllm-ascend 阶段划分 |
| `docs/integration-vllm-ascend.md` | 175 | 描述已经被砍掉的 vllm-ascend 集成；历史实现仍在 `legacy/` |
| `docs/notes/tmp-three-areas-plan.md` | 202 | 临时方案稿，已并入 `docs/plan-vllm-native-backend.md` |
| `docs/notes/dsv41-*.md`（14 篇） | 3900 | DSV4.1 的过程日志。**保留** `dsv41-facts.md`（模型事实）、`dsv41-final-report.md`（结论）、`dsv41-native-mmap.md`（技术手法） |

---

## 3. 修的错（不是删，是纠正）

1. **`runtime/vllm-api/` 引用全是空指针。** 代码里 20 多处写着"以 `runtime/vllm-api/xxx.py:行号`
   为准"，但该目录被 `.gitignore` 排除，102 上根本不存在 —— 任何人想核对都核对不了。
   现在把它作为 **`reference/vllm-0.23/`（101 个文件，2.7 MB）入库**，所有引用改指过去，
   目录内附 `README.md` 说明来源（vLLM v0.23.0，Apache-2.0，未修改）与"不许改"的规则。
2. **环境定义只活在临时目录里。** `tests/conftest.py` 和 `scripts/run_device_ci.sh` 都
   `source .agent_loop/run/env.sh`，而 `.agent_loop/` 是未跟踪的。现在环境定义是
   `scripts/env-ascend102.sh`（入库），旧路径保留为后备。
3. **`hardware-checklist.md` 的 worker hook 行号是旧的**（`:562/:752/:931/:999`），
   改成实际的 `:848/:1533/:1712/:1819`。
4. **`status-native-backend.md` 自称"代码从未在 Ascend 上跑过一行"** —— 早已不成立。
   它被改名为 `notes/handover-2026-10-28.md`，文首加了"这是历史快照 + 哪两条结论已作废"的横幅。
5. **`docs/integration-vllm-native.md` 还挂着"待实现清单 N1–N12"和"建议里程碑"**，而迁移
   早就做完了。改成"耦合地图"，删掉作废的 §6/§7，保留仍然有效的 §2/§5/§8。
6. **契约测试在装了真 vLLM 的机器上无法收集**（`test_boot_rehearsal` / `test_worker_contract`
   在模块级读只有 fake 才提供的记录器）。加了 `allow_module_level` 的 skip 并说明原因。
7. **`pyproject.toml` 的 description 还是旧定位**（"engine-agnostic attention kernels"）。
8. **断链**：42 个相对链接里有 8 个指向已删文件，全部改指到仍然存在的文档。
9. **`benchmarks/**` 里 30 个脚本的默认输出路径指向已删除的 `.agent_loop/run/`**，
   改为已被 `.gitignore` 的 `bench_out/`。

---

## 4. 清理后的验证

| 检查 | 结果 |
|---|---|
| `import tileinfer` / `import tileinfer.vllm.platform` | 通过 |
| 文档相对链接（42 条） | 0 条断链 |
| `git grep 'runtime/vllm-api'` | 0 处（除历史说明） |
| CPU 测试套件 vs 清理前纯净树 | **0 项新增失败，9 项被修好**（89 → 80） |

CPU 套件本身仍然红，原因与清理无关（跨文件顺序污染），已在 `docs/status.md` §6 写清，
并作为独立任务保留。

---

## 5. 没做的事（有意为之）

- **DSV4.1 的代码没删**：它是"加一个模型"的第二个样例，服务于第三块（多模型/多卡兼容）。
  但它的运行环境（`.dsv41/`、`.venv`、`.venv_ascend30`）删了 —— 它现在是**参考实现**，
  不是可运行路径。要重新跑起来按 `tools/dsv41/README.md` 重建环境。
- **`legacy/` 没删**：vllm-ascend 时代的算子实现是第二块要用到的素材库。
- **`benchmarks/probes/` 的 44 个探针没删**：每个都对应一条已登记的结论，删掉就没法复现。
  但它们需要一个"哪些还有效"的索引，这件事没做。
- **80 项 CPU 测试失败没修**：属于独立的一轮排查。
