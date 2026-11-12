# M1 删除清单（v2 代码的冻结与退场）

> v3 不删除 v2 代码，但**停止引用它**。这一页是"哪些还会活、哪些等着删、凭什么判断可以删"。
> 判定标准只有一条：**v3 的 live 路径（`tileinfer/{cli,bridge,resolve,contracts,manifests,
> providers}`）不再 import 它，且 M2+ 的对应 provider/bridge 有等价实现与证据。**

## 1. 冻结区（S6 允许保留）

| 文件 | 行数（约） | 为什么留 |
|---|---|---|
| `tileinfer/vllm/platform.py` | 1954 | v2 平台插件的历史实现；v3 不注册平台，但测试与对照实验还要读它 |
| `tileinfer/vllm/worker.py` | 2059 | v2 worker 四个 hook 的实现；v3 由 vllm-ascend 的 worker 接管 |
| `tileinfer/vllm/platform_dsv41.py` | 67 | DeepSeek-V4.1 专用平台变体，同上 |

冻结区的规则：

* 不再新增功能，不再修性能；
* 不再被 `tileinfer/__init__.py`、`tileinfer/cli/`、`tileinfer/bridge/`、`tileinfer/resolve/`
  引用；
* 只有 `tests/` 里的 v2 契约测试可以 import 它，且这些测试的失败不算 v3 回归。

## 2. 等着删（M2–M4 后逐批）

| 目录/文件 | 行数（约） | 删除前提 |
|---|---|---|
| `tileinfer/vllm/attn/` | 数千 | `providers/attention` 有 vendor-compatible 后端，C2/C3/C4 有 A/B 证据 |
| `tileinfer/vllm/ops/` | 数千 | 对应 GDN/MoE/RMSNorm/RoPE provider 进 `providers/` 并有逐位/性能证据 |
| `tileinfer/vllm/compat/patches.py` | 370 | 所有 patch 被 bridge 的 6 个钩子替代，接入点账本更新 |
| `tileinfer/vllm/plugin.py` | 775 | `bridge/install.py` 覆盖其全部职责；入口点已切换（已完成） |
| `tileinfer/vllm/doctor.py` | – | `cli/doctor.py` 覆盖；console script 已切换（已完成） |
| `tileinfer/vllm/models/`、`weights/` 等 | – | 对应声明/provider 进新树；route O 有证据 |

**不要一次删完**：v2 的 GDN/MoE 内核是 M2+ 的候选 provider，删早了会把"可复用的最快实现"
一起删掉。每批删除前先跑 `tests/test_no_vllm_ascend.py`、`tests/test_layer_deps.py` 和
对应 provider 的 A/B。

## 3. 零引用证明（当前）

```bash
cd /home/userlist/zxs/TileInfer
# v3 live 路径不得出现 tileinfer.vllm
grep -rn "tileinfer\.vllm\|from \.\.vllm\|from \.vllm" \
  tileinfer/cli tileinfer/bridge tileinfer/resolve tileinfer/contracts \
  tileinfer/manifests tileinfer/providers 2>/dev/null || echo "no live references"
```

2026-11-12 的结果：**no live references**。

## 4. 入口点账本

`pyproject.toml` 现在只注册：

```toml
[project.entry-points."vllm.general_plugins"]
tileinfer = "tileinfer.bridge.install:install"
```

`vllm.platform_plugins` 下**没有** TileInfer 条目；`.venv_v3` 里该组只剩
`ascend -> vllm_ascend:register`。这是 v3 与 v2 最硬的一条分界：平台槽只有一个主人。
