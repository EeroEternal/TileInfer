#!/usr/bin/env bash
# TileInfer 原生 vLLM 后端 —— 服务器端安装/自检脚本
#
# 用法（在解包后的目录里）：
#   bash scripts/install-native-backend.sh            # 安装 + 自检
#   bash scripts/install-native-backend.sh --check    # 只自检，不安装
#
# 前置条件（脚本会逐项检查并给出修复提示）：
#   - 已 source CANN 的 set_env.sh
#   - python 3.10+，且当前解释器里 torch 与 torch_npu 版本匹配
#   - vllm >= 0.23,<0.24（本后端钉这个区间）
#   - tilelang（Ascend 版，带 tilelang.ascend），内核编译需要
#
# 退出码：0 全部通过；1 有检查项失败；2 用法错误。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="install"
case "${1:-}" in
  ""|--install) MODE="install" ;;
  --check)      MODE="check" ;;
  -h|--help)    sed -n '2,20p' "$0"; exit 0 ;;
  *)            echo "unknown argument: $1" >&2; exit 2 ;;
esac

fail=0
pass() { printf '  [PASS] %s\n' "$1"; }
skip() { printf '  [SKIP] %s\n' "$1"; }
bad()  { printf '  [FAIL] %s\n' "$1"; fail=$((fail+1)); }
hint() { printf '         fix: %s\n' "$1"; }

echo "== TileInfer native backend installer =="
echo "   repo: $ROOT"
echo

echo "-- 1. python / torch / torch_npu --"
python - <<'PY' || fail=$((fail+1))
import sys
print("  python", sys.version.split()[0], "(>=3.10:", sys.version_info >= (3, 10), ")")
try:
    import torch
    print("  torch", torch.__version__)
except Exception as exc:
    print("  [FAIL] torch:", exc); raise SystemExit(1)
try:
    import torch_npu  # noqa: F401
    print("  torch_npu", getattr(torch_npu, "__version__", "?"))
    print("  device_count", torch.npu.device_count(), "| name", torch.npu.get_device_name(0) if torch.npu.device_count() else "n/a")
except Exception as exc:
    print("  [FAIL] torch_npu:", exc); raise SystemExit(1)
PY

echo "-- 2. CANN environment --"
for v in ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_OPP_PATH; do
  if [ -n "${!v:-}" ]; then pass "$v=${!v}"; else bad "$v unset"; hint "source <CANN>/set_env.sh"; fi
done

echo "-- 3. vllm --"
python - <<'PY' || fail=$((fail+1))
try:
    import vllm
    v = vllm.__version__
    ok = tuple(int(x) for x in v.split(".")[:2]) >= (0, 23) and tuple(int(x) for x in v.split(".")[:2]) < (0, 24)
    print(f"  vllm {v} (supported >=0.23,<0.24: {ok})")
    raise SystemExit(0 if ok else 1)
except ImportError as exc:
    print("  [FAIL] vllm not importable:", exc)
    raise SystemExit(1)
PY

echo "-- 4. tilelang (Ascend build) --"
python - <<'PY'
try:
    import tilelang
    have = hasattr(tilelang, "ascend")
    print(f"  tilelang {getattr(tilelang, '__version__', '?')} (tilelang.ascend: {have})")
    raise SystemExit(0 if have else 1)
except Exception as exc:
    print("  [SKIP] tilelang:", exc, "-- kernels cannot compile until it is installed")
    raise SystemExit(0)
PY

if [ "$MODE" = "install" ]; then
  echo "-- 5. install the package (editable) --"
  if python -m pip install -e "$ROOT" --no-deps; then
    pass "pip install -e $ROOT"
  else
    bad "pip install -e failed"; hint "check pip / network / permissions"
  fi
fi

echo "-- 6. entry points --"
python - <<'PY'
import importlib.metadata as md
want = {"vllm.platform_plugins": "tileinfer.vllm.platform:register",
        "vllm.general_plugins": "tileinfer.vllm.plugin:install"}
try:
    eps = md.entry_points()
    ok = True
    for group, value in want.items():
        found = [e.value for e in eps.select(group=group) if e.name == "tileinfer"]
        good = value in found
        ok = ok and good
        print(f"  {'[PASS]' if good else '[FAIL]'} {group} -> {found or 'MISSING'} (want {value})")
    raise SystemExit(0 if ok else 1)
except Exception as exc:
    print("  [FAIL] entry points:", exc); raise SystemExit(1)
PY
[ $? -ne 0 ] && fail=$((fail+1))

echo "-- 7. tileinfer doctor --"
if python -m tileinfer.vllm.doctor; then pass "doctor"; else bad "doctor reported failures"; hint "read the doctor lines above; they name the fix"; fi

echo
if [ "$fail" -eq 0 ]; then
  echo "RESULT: all checks passed -- next: docs/hardware-checklist.md step 5 onward"
else
  echo "RESULT: $fail check(s) failed -- fix them before step 5 of docs/hardware-checklist.md"
fi
exit $([ "$fail" -eq 0 ] && echo 0 || echo 1)
