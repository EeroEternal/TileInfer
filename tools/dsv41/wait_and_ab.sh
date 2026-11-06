#!/bin/bash
# 长跑守护：反复等**稳定**的内存窗口，然后执行双臂 TPOT A/B；失败自动退避重试。
#
# 为什么需要"稳定"判定：本会话有 7 次在 available=532–541 GB 时启动、却在加载 3–5 分钟内被
# OOM-killer 杀掉（`EngineCore: -9`，anon-rss 484–491 GB）。同一脚本在 511/528/539/616 GB 下也
# 成功过 —— 说明问题是窗口**瞬时性**（别人恰好同时申请），所以这里要求连续 4 次采样（2 分钟）
# 都 >= NEED 才动手，且整轮失败后退避 5 分钟再试。
#
# 用法： NEED=525 MAX_HOURS=6 bash tools/dsv41/wait_and_ab.sh
# 产出： /tmp/wait_ab.log（守护日志）、/tmp/moe_tpot.result（单次结果）、
#        /tmp/moe_tpot.SUCCESS（成功后复制一份，便于识别）
set -u

NEED="${NEED:-525}"
MAX_HOURS="${MAX_HOURS:-6}"
LOG=/tmp/wait_ab.log
REPO=/home/userlist/zxs/TileInfer

: > "$LOG"
DEADLINE=$(( $(date +%s) + MAX_HOURS * 3600 ))
attempt=0

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  attempt=$((attempt + 1))
  echo "[$(date +%F_%H:%M:%S)] attempt ${attempt}: 检查窗口稳定性（连续 4 次 >= ${NEED}G）" >> "$LOG"
  stable=1
  for k in 1 2 3 4; do
    A=$(free -g | awk '/^Mem:/{print $7}')
    echo "   probe${k} avail=${A}G" >> "$LOG"
    if [ "$A" -lt "$NEED" ]; then stable=0; break; fi
    sleep 30
  done
  if [ "$stable" != "1" ]; then sleep 60; continue; fi

  echo "[$(date +%F_%H:%M:%S)] 窗口稳定，开始双臂 A/B" >> "$LOG"
  pkill -f "vllm.entrypoints" 2>/dev/null
  sleep 5
  bash "$REPO/tools/dsv41/perf_moe_tpot.sh" --need 500 > /tmp/ab_try.log 2>&1
  rc=$?

  if [ "$rc" = "0" ] && grep -qa "TPOT" /tmp/moe_tpot.result 2>/dev/null; then
    echo "[$(date +%F_%H:%M:%S)] SUCCESS（attempt ${attempt}）" >> "$LOG"
    cp /tmp/moe_tpot.result /tmp/moe_tpot.SUCCESS
    exit 0
  fi

  echo "[$(date +%F_%H:%M:%S)] attempt ${attempt} 失败 rc=${rc}，退避 5 分钟" >> "$LOG"
  tail -n 3 /tmp/moe_tpot.result 2>/dev/null >> "$LOG"
  sleep 300
done

echo "[$(date +%F_%H:%M:%S)] 超时 ${MAX_HOURS}h，放弃" >> "$LOG"
exit 1
