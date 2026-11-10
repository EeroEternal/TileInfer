"""Shared pytest fixtures.

Everything here is CPU-friendly: the suite must pass on a laptop with no NPU so that the planning
logic and the reference implementation stay testable in CI, and so that a kernel bug can be
separated from a scheduling bug.
"""

from __future__ import annotations

import ctypes
import os
import shlex
import subprocess
import sys

_BOOTSTRAP_MARK = "TILEINFER_ENV_BOOTSTRAPPED"


def _find_env_sh(root: str) -> str | None:
    """定位机器环境脚本:仓库内的 scripts/env-ascend102.sh 优先,旧的 loop 脚本作后备。

    环境脚本必须**随仓库走**:它原来只存在于 .agent_loop/run/env.sh(未跟踪的临时目录),
    删掉那个目录就等于删掉了项目唯一的工具链定义。
    """
    for rel in (os.path.join("scripts", "env-ascend102.sh"),
                os.path.join(".agent_loop", "run", "env.sh")):
        path = os.path.join(root, rel)
        if os.path.exists(path):
            return path
    return None


def _bootstrap_ascend_env() -> None:
    """裸 shell 下自举项目 CANN 工具链环境。

    项目工具链在 .work/cann930,由 scripts/env-ascend102.sh 导出;但 pytest 被直接调用时
    不经过任何 source。此时 TILELANG_CACHE_DIR 缺失导致内核缓存全量未命中,tilelang
    退回系统 cann-9.1.1 编译器并因缺模板 intrinsic 而失败,decode/prefill/moe/gdn
    全族设备用例连坐失败。

    这里把 env 脚本的导出变量导入 os.environ:子进程(tilelang 编译器、vllm 子进程)
    与运行期读取的变量(缓存目录、ASCEND_* 路径)由此与手工 source 完全一致;
    LD_PRELOAD 的 libcust_opapi 对当前进程无效,改为 ctypes 全局预加载。
    仅当调用方未 source(env 脚本的关键变量缺失)时才生效,不改变手工环境行为;
    无 env 脚本的机器(CPU-only CI)直接跳过,失败只告警不抛异常。
    """
    if os.environ.get(_BOOTSTRAP_MARK) == "1":
        return
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_sh = _find_env_sh(root)
    if env_sh is None:
        return
    if os.environ.get("TILELANG_CACHE_DIR"):  # 调用方已 source env 脚本
        os.environ[_BOOTSTRAP_MARK] = "1"
        return
    try:
        # 排除会随子 shell 漂移的进程型变量,其余全量导入
        drop = {"PWD", "OLDPWD", "SHLVL", "_", "BASH_SOURCE"}
        out = subprocess.run(
            ["bash", "-c", f"source {shlex.quote(env_sh)} >/dev/null 2>&1 && env -0"],
            capture_output=True,
            timeout=60,
            check=True,
        )
        preload = ""
        for chunk in out.stdout.split(b"\0"):
            if not chunk:
                continue
            key, _, val = chunk.partition(b"=")
            k = key.decode(errors="replace")
            v = val.decode(errors="replace")
            if k in drop:
                continue
            if k == "LD_PRELOAD":
                preload = v.split(":")[0]  # 记录 libcust 路径,当前进程用 ctypes 预加载
            os.environ[k] = v
        if preload and os.path.exists(preload):
            try:
                # RTLD_GLOBAL:符号入全局命名空间,等效 LD_PRELOAD 对本进程的作用
                ctypes.CDLL(preload, mode=ctypes.RTLD_GLOBAL)
            except OSError as exc:
                print(f"[conftest] libcust 预加载失败(继续,依赖调用方环境): {exc}", file=sys.stderr)
        os.environ[_BOOTSTRAP_MARK] = "1"
    except Exception as exc:  # 环境自举失败不阻塞收集,交由具体用例给出真实错误
        print(f"[conftest] 环境自举失败(继续,依赖调用方环境): {exc}", file=sys.stderr)


_bootstrap_ascend_env()

import pytest
import torch


@pytest.fixture
def cpu() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def seed() -> int:
    torch.manual_seed(0)
    return 0


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "npu: test requires an Ascend NPU")
    config.addinivalue_line("markers", "tilelang: test requires the TileLang Ascend toolchain")
