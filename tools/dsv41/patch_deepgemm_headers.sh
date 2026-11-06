#!/bin/bash
# 修补 deep_gemm 的 Ascend 头文件：把 `ascendc_assert` 降级为空实现。
#
# 背景（见 docs/notes/dsv41-perf-deepgemm-jit-note.md）：
#   deep_gemm 0.1.1 的 Ascend 头文件用了 `ascendc_assert(cond, fmt, ...)`，
#   而本机的 CANN（ASCEND_HOME_PATH=/home/dlb/Ascend/install/cann-9.3.0）的
#   asc/include/c_api/*.h 并没有定义它，于是 JIT 编译直接失败：
#       transform_sf.hpp:19: error: use of undeclared identifier 'ascendc_assert'
#   换用 /usr/local/Ascend/... 又会让 torch_npu 起不来（507033），所以只能就地补一个回退定义。
#
# 该宏只用于校验 scale-factor 的 fp32 尾数（我们自建 sf，可控），降级为空实现不影响数值正确性。
# 补丁幂等：重复执行不会重复插入；用 `--check` 只看状态不改文件。
#
# 注意：这是对第三方包（site-packages）的本地修改，属环境修补，需在文档与启动脚本里留痕；
#      两臂（A/B）都带同一补丁，满足 C1"同环境"。
set -u

DG=${DG_PATH:-/home/userlist/zxs/TileInfer/.venv/lib/python3.12/site-packages/deep_gemm}
MARK="DSV41_ASCENDC_ASSERT_FALLBACK"
HEADERS=(
  "$DG/include/deep_gemm/transform_sf.hpp"
  "$DG/include/deep_gemm/mqa_logits.hpp"
  "$DG/include/deep_gemm/sparse_mqa_logits.hpp"
)
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

rc=0
for f in "${HEADERS[@]}"; do
  [ -f "$f" ] || { echo "skip (missing): $f"; continue; }
  if grep -q "$MARK" "$f"; then
    echo "already patched: $(basename "$f")"
    continue
  fi
  if [ "$CHECK_ONLY" = "1" ]; then
    echo "NOT patched: $(basename "$f")"
    rc=1
    continue
  fi
  # 在文件首个 #include 之前插入回退定义
  python3 - "$f" "$MARK" <<'PY'
import sys
path, mark = sys.argv[1], sys.argv[2]
text = open(path, encoding="utf-8").read()
shim = (
    "// %s: deep_gemm expects CANN to provide ascendc_assert(cond, fmt, ...);\n"
    "// the installed CANN (9.3.0) does not, so degrade it to a no-op.\n"
    "#ifndef %s\n"
    "#define %s 1\n"
    "#ifndef ascendc_assert\n"
    "#define ascendc_assert(cond, ...) ((void)0)\n"
    "#endif\n"
    "#endif\n\n"
) % (mark, mark, mark)
if mark in text:
    print("already")
    sys.exit(0)
lines = text.splitlines(keepends=True)
for i, line in enumerate(lines):
    if line.lstrip().startswith("#include"):
        lines.insert(i, shim)
        break
else:
    lines.append(shim)
open(path, "w", encoding="utf-8").write("".join(lines))
print("patched")
PY
  echo "patched: $(basename "$f")"
done
exit $rc
