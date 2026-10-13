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


def _bootstrap_ascend_env() -> None:
    """裸 shell 下自举项目 CANN 工具链环境(见 .agent_loop/state/DECISIONS.md D26/D32)。

    项目真实工具链在 .work/cann930,经 .agent_loop/run/env.sh 导出;但 loop 框架的
    验收调用(verify_task.sh → pytest)不经过任何 source。此时 TILELANG_CACHE_DIR
    缺失导致内核缓存全量未命中,tilelang 退回系统 cann-9.1.1 编译器并因缺模板
    intrinsic 而失败(D26),decode/prefill/moe/gdn 全族设备用例连坐失败(E25)。

    这里把 env.sh 的导出变量导入 os.environ:子进程(tilelang 编译器、vllm 子进程)
    与运行期读取的变量(缓存目录、ASCEND_* 路径)由此与手工 source 完全一致;
    LD_PRELOAD 的 libcust_opapi 对当前进程无效,改为 ctypes 全局预加载。
    仅当调用方未 source(env.sh 的关键变量缺失)时才生效,不改变手工环境行为;
    无 env.sh 的机器(CPU-only CI)直接跳过,失败只告警不抛异常。
    """
    if os.environ.get(_BOOTSTRAP_MARK) == "1":
        return
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_sh = os.path.join(root, ".agent_loop", "run", "env.sh")
    if not os.path.exists(env_sh):
        return
    if os.environ.get("TILELANG_CACHE_DIR"):  # 调用方已 source env.sh
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

#: 测试进程内真实加载过 vllm_ascend_C(TORCH_LIBRARY ``_C_ascend``)的标记。
#: 该 .so 载入后,解释器终结期的 CANN teardown 会 double-free(D14 同源的已知
#: 上游缺陷,serve 进程被 kill 收尾所以从未暴露),把 pytest 的真实退出码变成
#: SIGABRT(134)。pytest_unconfigure(全部输出已写完)时按状态码硬退出跳过
#: 终结期析构;未加载该 .so 的运行(CPU 套件、其余设备套件)完全不受影响。
_CUSTOM_OPS_STATE = {"vllm_ascend_C": False}


def mark_vllm_ascend_C_loaded() -> None:
    """由真实执行 ``import vllm_ascend.vllm_ascend_C`` 的设备用例调用(见 dispatch 测试)。"""
    _CUSTOM_OPS_STATE["vllm_ascend_C"] = True


def pytest_unconfigure(config: pytest.Config) -> None:
    if _CUSTOM_OPS_STATE["vllm_ascend_C"]:
        import atexit
        import os as _os
        import sys as _sys

        def _hard_exit_skip_cann_teardown() -> None:
            # 此时 pytest 汇总已写完;os._exit 跳过解释器终结期的 CANN teardown
            # (double-free,D14 同源上游缺陷),保住真实退出码。
            _sys.stdout.flush()
            _sys.stderr.flush()
            _os._exit(int(getattr(config, "exitstatus", 0)))

        atexit.register(_hard_exit_skip_cann_teardown)


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
