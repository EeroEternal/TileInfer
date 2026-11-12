# 环境版本策略（2026-11-12）

> 回答一个问题：**"vLLM / vllm-ascend 不是最新，是不是应该先升级环境？"**
> 结论先说：**Qwen3.6 验证期间不升级。** 升级是 M5 前的独立 gate，不是环境维护动作。

---

## 1. 当前栈与"最新"的实际差距

| 组件 | 当前 | 索引最新 | 关键事实 |
|---|---|---|---|
| vLLM | 0.23.0 | 0.33.0 | Ascend 侧没有 0.33 的 vllm-ascend；配对规则要求同版本 |
| vllm-ascend | 0.23.0（**950/A5 专用 wheel**） | 0.30.0（PyPI，**A2 build**） | PyPI wheel 的 `_build_info.py` 是 `A2`；本机没找到 0.30-950 wheel |
| torch | 2.10.0+cpu | — | 0.30 仍要求 2.10.0 |
| torch_npu | 2.10.0.post4 | 2.14.0 | 0.30 要求 2.10.0.post7（小版本，不是 2.14） |
| transformers | 4.57.6 | 5.14.1（0.30 的依赖） | 大版本跳跃；模型 config 写的是 4.57.1 |
| CANN | 9.3.0 | 0.30 文档写 9.1.0 | 950 是专用 build，不能按 PyPI 通用包升级 |

来源：`pip index versions`（2026-11-12）、PyPI [vllm-ascend 0.30.0](https://pypi.org/project/vllm-ascend/0.30.0/) 的 metadata、
本机 `/home/userlist/ww/vllm_ascend-0.23.0-950-cp312-cp312-manylinux_2_34_x86_64.whl`。
官方 README 仍把 **v0.23.0 写成 latest stable**，把 0.30 线写成 RC（`v0.30.0rc1`）。

---

## 2. 为什么不能直接 `pip install -U`

1. **设备类型断言会直接拒绝。** 当前 `vllm_ascend/utils.py:845 check_ascend_device_type()` 要求
   wheel 的 `__device_type__` 与 `torch_npu.npu.get_soc_version()` 一致；本机是 A5/950，
   PyPI 0.30 是 A2，启动即 assert。
2. **没有 0.30-950 build。** 本机只找到 `vllm_ascend-0.23.0-950-...whl`。
   升级的硬前提是先拿到厂商/内源的 950 wheel；没有它，升级不可执行。
3. **0.30 不是稳定线。** 官方 README 把 0.23 标为 latest stable，0.30 标为 RC；
   PyPI 上有 0.30.0 包，但"能下载"不等于"能在这张卡上稳定跑"。
4. **依赖面太大。** transformers 4.57 → 5.14、torch_npu post4 → post7、CANN 9.3 vs 文档 9.1；
   任何一项不匹配，症状都是数值错/起不来，而不是一条明确的报错。
5. **接入点钉在 0.23 API 上。** 6 个 seam 依赖 `CustomOp.dispatch_forward`、
   `PluggableLayer.__new__`、`CudagraphDispatcher.dispatch`、`GPUModelRunner.capture_model`、
   attention selector、config gate。跳 7 个版本是重新集成，不是环境维护。
6. **升级会让 M1 的 C1–C4 证据全部失效**，而 Qwen3.6 在当前栈上已经跑通；
   对当前目标没有收益。

补充：`0.23.0-950` wheel 目前只在 `/home/userlist/ww/`，不在仓库里。重建环境需要它，
建议记录来源与校验和；否则"唯一能跑的栈"不可复现。

---

## 3. M5 前的版本 gate（V0）

1. 先确认是否存在 **950/A5 的 vllm-ascend ≥ 0.26 官方 build**。没有就冻结 `0.23.0-950`，
   并在模型声明里写清"pinned stack"。
2. 有则建独立 `.venv_next`，**不碰** `.venv_v3` / `.venv_baseline`；装同版本
   vLLM + torch_npu + transformers + CANN。
3. 跑：seam 盘点 → `tileinfer doctor` → 逐文件 CPU 套件 → Qwen3.6 C1–C4 冒烟 →
   图 / MTP / 8K 子集。
4. 判据：≤2 个接入点需要改 → 可评估升级；≥3 → 按
   [`../long-term-task.md`](../long-term-task.md) §8 红线收缩或不升。
5. 升级前后各留一份 A/B，两臂同环境同参数；不要把新旧栈的数字混着比。

---

## 4. 对 M5 选模的影响（最关键）

"厂商部署不了"必须相对**这台机器能部署的厂商栈**判断，而不是相对 vLLM 最新版。

- 一个模型如果只是"0.23 不支持、0.30 支持"，那是**版本滞后**，不是能力缺口；
  给它写 provider 就是重复劳动（[`../extending.md`](../extending.md) §3.0 红线）。
- 选模顺序：列 2–3 个候选 → 在当前 `0.23.0-950` 上试加载 → 再在（若存在的）新 950 build
  上试 → 只有两边都不行、或两边都只能走降级路径，才是真缺口。
- 若新 950 build 不存在，gap report 必须写明"pinned-stack gap；上游可能已支持"，
  证据等级降一档，不能当成"厂商能力缺口"来宣传。

---

## 5. 决策（2026-11-12：A）

选 **A：冻结 `0.23.0-950`，先跑完 Qwen3.6 矩阵；M5 前做 V0 gate。**

- 在拿到 950/A5 的新 vllm-ascend build 之前，不升级，也不把 0.30 的数字与 0.23 混比。
- 若 M5 前仍无新 build，gap report 必须标"pinned-stack gap；上游可能已支持"。
- 0.30 的 A2 wheel 只作离线 API 差异参考，不装进可运行环境。
- 选项 B（离线 spike）与 C（现在升级）本轮不采纳；C 在拿到 0.30-950 wheel 前不可执行。

矩阵里的新模型章节已经把 V0 作为第 0 步：
[`../verification-matrix.md`](../verification-matrix.md) §2。
