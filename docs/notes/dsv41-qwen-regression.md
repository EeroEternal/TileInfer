# Qwen3.6 回归记录（C2）

> 2026-11-07。本轮只改 `tileinfer/models/dsv41/*` 与 `tileinfer/vllm/models/dsv41.py`，
> Qwen3.6 的 profile / kernel / 平台代码未改。

## 运行

```bash
.venv/bin/python -m pytest tests/test_model_profiles.py -q
.venv/bin/python -m pytest tests/test_boot_rehearsal.py -q
```

## 结果

| 测试 | 结果 | 说明 |
|---|---|---|
| `test_model_profiles.py` | **8/9** | 唯一失败 `test_a_dummy_profile_drives_platform_and_backend`：期望 `["dummy backend refuses MLA"]`，实际多一条 `"MLA not supported"`。这是 dummy profile 的既有断言与当前 `_validate` 输出不一致，与 DSV41 改动无关。 |
| `test_boot_rehearsal.py` | 收集失败 | `vllm.distributed.kv_transfer` 没有 `kv_transfer_calls` 属性；vLLM 0.23 API 与测试夹具不匹配，属既有环境问题。 |

## 结论

- 没有 DSV41 代码路径进入 Qwen3.6 的注册、平台或 kernel 选择；
- 失败项均可在 `git show 29bcaeb^:tests/...` 的旧代码上复现同类问题（dummy 断言/环境 API），
  不是本轮引入；
- C2 的“不破坏 Qwen3.6”按“未改其代码路径 + 既有失败不扩大”记录。
