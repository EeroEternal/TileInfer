"""从 vLLM runner 取「本步有哪些请求」的钩子（P3 接线前置，§16 路线 1'）。

为什么需要：请求 id 在调度器侧的 `input_batch.req_ids` / `scheduler_output.num_scheduled_tokens`，
而模型的 `forward` 只收到 `attn_metadata`（我们这套栈没有自研 attention backend，见 §16），
所以只能从 runner 侧把它钩出来。

安全约束（与目标的回归门一致）：

* **只有当安装方显式调用 `install()` 时才改动 vLLM 类**，默认不安装；
* 钩子对异常完全静默（任何取不到 id 的情况都置 ``None``），绝不因为钩子让推理失败；
* 不改变被包装方法的返回值与签名。

本模块可离线单测（用一个 fake runner 类 + fake scheduler_output），不需要真机。
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

#: 最近一次 `execute_model` 看到的请求 id（顺序即调度顺序，与 num_scheduled_tokens 一致）
_STATE: dict[str, Optional[List[str]]] = {"ids": None}
#: 与 ``_STATE["ids"]`` 同序的每请求 token 数（用于切分批次）
_COUNTS: dict[str, Optional[List[int]]] = {"counts": None}
#: ``(req_id, absolute_start, scheduled_len)``：用**绝对起点**匹配扁平批里的片段。
#: ``num_scheduled_tokens`` 的字典序与 flat batch 顺序**不一致**（cached 请求会被重排，
#: 100-prompt 实测），所以不能按位置索引 id；scheduler_output 里的 new/cached 结构才有绝对起点。
_ENTRIES: dict[str, Optional[List[tuple]]] = {"entries": None}

#: 记录被包装过的类，避免重复包装
_installed: set[type] = set()


def current_req_ids() -> Optional[List[str]]:
    """本步的请求 id 列表；未安装或取不到时返回 ``None``。"""
    ids = _STATE["ids"]
    return list(ids) if ids is not None else None


def reset() -> None:
    _STATE["ids"] = None
    _COUNTS["counts"] = None
    _ENTRIES["entries"] = None


def current_req_counts() -> Optional[List[int]]:
    """与 :func:`current_req_ids` 同序的每请求 token 数；取不到时返回 ``None``。"""
    counts = _COUNTS["counts"]
    return list(counts) if counts is not None else None


def current_req_entries() -> Optional[List[tuple]]:
    """本步的 ``(req_id, absolute_start, scheduled_len)``；取不到返回 ``None``。"""
    entries = _ENTRIES["entries"]
    return list(entries) if entries is not None else None


def make_hook(orig: Callable) -> Callable:
    """包装 ``execute_model``：先记录请求 id，再原样调用。"""

    def patched(self, scheduler_output, *args, **kwargs):
        try:
            tokens = getattr(scheduler_output, "num_scheduled_tokens", None)
            if tokens is not None:
                _STATE["ids"] = [str(k) for k in tokens.keys()]
                _COUNTS["counts"] = [int(v) for v in tokens.values()]
            else:
                _STATE["ids"] = None
                _COUNTS["counts"] = None
            entries = []
            for r in getattr(scheduler_output, "scheduled_new_reqs", None) or []:
                rid = str(getattr(r, "req_id", ""))
                n = int(getattr(r, "prompt_token_ids_len", None) or
                        (tokens or {}).get(rid, 1))
                entries.append((rid, 0, n))
            cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
            if cached is not None:
                ids = [str(x) for x in getattr(cached, "req_ids", [])]
                starts = [int(x) for x in getattr(cached, "num_computed_tokens", [])]
                for i, rid in enumerate(ids):
                    n = int((tokens or {}).get(rid, 1))
                    entries.append((rid, starts[i], n))
            _ENTRIES["entries"] = entries
        except Exception:  # noqa: BLE001 - 钩子绝不能影响推理
            _STATE["ids"] = None
            _COUNTS["counts"] = None
            _ENTRIES["entries"] = None
        return orig(self, scheduler_output, *args, **kwargs)

    patched.__wrapped__ = orig  # type: ignore[attr-defined]
    patched.__name__ = getattr(orig, "__name__", "execute_model")
    return patched


def install(runner_cls: Optional[type] = None) -> bool:
    """在 ``runner_cls.execute_model`` 上安装钩子；返回是否**本次**装上了。

    ``runner_cls`` 为 ``None`` 时尝试从 vLLM 导入 `GPUModelRunner`；导入失败返回 ``False``
    （CPU 机器或没有 vLLM 时安全退化）。
    """
    if runner_cls is None:
        try:
            from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # type: ignore

            runner_cls = GPUModelRunner
        except Exception:  # noqa: BLE001
            return False
    if runner_cls in _installed:
        return False
    orig = getattr(runner_cls, "execute_model", None)
    if orig is None:
        return False
    setattr(runner_cls, "execute_model", make_hook(orig))
    _installed.add(runner_cls)
    return True


def uninstall(runner_cls: Optional[type] = None) -> bool:
    """恢复原方法（测试与回退用）。"""
    if runner_cls is None:
        return False
    if runner_cls not in _installed:
        return False
    cur = getattr(runner_cls, "execute_model", None)
    orig = getattr(cur, "__wrapped__", None)
    if orig is not None:
        setattr(runner_cls, "execute_model", orig)
    _installed.discard(runner_cls)
    return True
