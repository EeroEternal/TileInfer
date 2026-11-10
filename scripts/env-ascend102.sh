#!/usr/bin/env bash
# Ascend 950 toolchain environment for this project (the 102 box is the reference machine).
#
# This file is the **single** source of truth for "what environment does this repo need".
# It used to live only in `.agent_loop/run/env.sh`, an untracked scratch directory, which meant
# deleting that directory deleted the project's only environment definition. Source this instead:
#
#   source scripts/env-ascend102.sh
#
# `tests/conftest.py` sources it automatically when pytest is invoked from a bare shell, and
# `scripts/run_device_ci.sh` sources it before the canary run.

set -u

_TI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- CANN ---------------------------------------------------------------------------------------
# The system default is CANN 9.1.1, which does not match this project's dependencies. The tarball
# the kernels are actually built and verified against lives under .work/.
export ASCEND_HOME="${ASCEND_HOME:-${_TI_ROOT}/.work/cann930/install/cann}"
if [ -f "${ASCEND_HOME}/set_env.sh" ]; then
  # shellcheck disable=SC1091
  source "${ASCEND_HOME}/set_env.sh" >/dev/null 2>&1
else
  echo "env-ascend102: ${ASCEND_HOME}/set_env.sh not found (CANN not installed?)" >&2
fi

# --- TileLang kernel cache ----------------------------------------------------------------------
# A cold cache means every kernel is recompiled (70-100 s each, holding the GIL). Keep the cache
# outside the repo so `git status` stays clean and `make clean` cannot throw compiled kernels away.
# The default is a sibling of the checkout, NOT $HOME: the warmed cache on this box lives at
# /home/userlist/zxs/tl_cache_213, and running as a different user must not silently start a cold
# one (a cold cache turns a 90 s run into a 20 min run).
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-$(dirname "${_TI_ROOT}")/tl_cache_213}"
mkdir -p "${TILELANG_CACHE_DIR}" 2>/dev/null || true

# --- runtime flags ------------------------------------------------------------------------------
export ACL_OP_INIT_MODE=1
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1

# glibc symbol versions newer than the system's; CANN needs them on the library path, not preloaded.
export LD_LIBRARY_PATH="${_TI_ROOT}/.work/glibcxx:${LD_LIBRARY_PATH:-}"

# --- opt-in: vllm-ascend custom op library ------------------------------------------------------
# Historically this was always preloaded, because the vllm-ascend-era path needed its short-conv
# custom op. The native vLLM backend must not pull vllm_ascend symbols into the process, so the
# preload is now opt-in and off by default. Set TILEINFER_PRELOAD_CUST_OPAPI=<path to .so> to
# re-enable it for a one-off comparison against the vllm-ascend arm.
if [ -n "${TILEINFER_PRELOAD_CUST_OPAPI:-}" ] && [ -f "${TILEINFER_PRELOAD_CUST_OPAPI}" ]; then
  export LD_PRELOAD="${TILEINFER_PRELOAD_CUST_OPAPI}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

unset _TI_ROOT
