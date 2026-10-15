#!/usr/bin/env bash
# Device CI: one reproducible entrypoint for the Ascend 950 kernel suite (设备套件)。
#
# 为什么需要它(next-tasks.md「Device CI ⏳ none yet」):设备用例此前靠零散手工命令执行,
# 没人记得完整 flags(env 自举/标记筛选/失败口径),回归无从复现。本脚本把执行口径固化:
#   1. 先跑一个 canary(已知最稳的 decode 探针)确认设备/工具链真的可用;
#   2. 再跑设备套件(pytest -m npu);
#   3. 统一汇总 + 退出码。
#
# 前置:项目工具链在 .work/cann930,经 .agent_loop/run/env.sh 导出;此处 source 它,
# 与 tests/conftest.py 的裸 shell 自举等价(两者任一即可)。NPU 被他人占用时设备用例会
# 失败/超时——脚本只报告,不 kill 任何他人进程(与 D48/D56 gate 纪律一致)。
#
# 用法:
#   bash scripts/run_device_ci.sh                 # 全套设备用例
#   bash scripts/run_device_ci.sh -k prefill      # 只跑匹配用例(透传给 pytest)
#   RUN_DEVICE_CI_CANARY=0 bash scripts/run_device_ci.sh   # 跳过 canary
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

if [ -f .agent_loop/run/env.sh ]; then
  # shellcheck disable=SC1090
  source .agent_loop/run/env.sh >/dev/null 2>&1 || true
fi
# 裸 shell 下 conftest 会自举同一套 env;这里显式 source 让 canary 子进程也一致。

PY="${PY:-$ROOT/.venv/bin/python}"
if [ ! -x "$PY" ]; then
  echo "device-ci: python not found at $PY (set PY=...)" >&2
  exit 2
fi

echo "== device-ci environment =="
echo "root=$ROOT"
echo "python=$PY"
echo "TILELANG_CACHE_DIR=${TILELANG_CACHE_DIR:-(unset)}"
if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info -t usages -i 0 2>/dev/null | grep -E "HBM Usage Rate|Aicore Usage Rate" || true
fi

rc=0

# --- 1. canary:最稳的 decode 探针(设备 + 工具链可用性) ---
if [ "${RUN_DEVICE_CI_CANARY:-1}" = "1" ]; then
  echo "== canary: paged decode probe (5 cases) =="
  if "$PY" benchmarks/probes/ascend950_paged_decode.py; then
    echo "canary: PASS"
  else
    echo "canary: FAIL (device/toolchain busy or broken; aborting before the suite)" >&2
    exit 1
  fi
fi

# --- 2. device suite ---
echo "== device suite: pytest -m npu $* =="
"$PY" -m pytest tests -m npu -p no:cacheprovider -ra "$@"
rc=$?
echo "== device suite exit=$rc =="

# --- 3. summary ---
if [ "$rc" = 0 ]; then
  echo "device-ci: PASS"
else
  echo "device-ci: FAIL (rc=$rc) — 常见原因:NPU 被他人占用 / 工具链未初始化 / 真实回归" >&2
fi
exit "$rc"
