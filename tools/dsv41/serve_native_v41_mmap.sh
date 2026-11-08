#!/bin/bash
# Native vllm-ascend 0.30 DeepSeek-V4.1-Flash launcher (Ascend950PR, single card).
#
# Status: the engine loads all 48 shards, offloads 40 expert layers, builds
# per-expert NZ buffers, but profile_run fails in aclnnHcPre because the
# prebuilt ascend950 kernel only ships tilingKey 1000/1001 while the 0.30 host
# tiling requests 1003.  Kept as the reproduction script for that blocker.
#
# Prereqs:
#   * patches from tools/dsv41/native_*_patch.py applied to the 0.30 venv;
#   * /home/dlb/vllm-ascend vendor tree for ascend950 custom ops.
set -e
cd /home/userlist/zxs/TileInfer

export VLLM_PLUGINS=ascend
export VLLM_USE_V1=1
export VLLM_USE_V2_MODEL_RUNNER=1
# No second pinned host copy: ND weights stay file-backed/reclaimable and only
# the router-selected experts are cast ND->NZ on device.
export VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1
export VLLM_ASCEND_OFFLOAD_HOST_NZ=0
export VLLM_ASCEND_OFFLOAD_PER_EXPERT_NZ=1
export VLLM_ASCEND_OFFLOAD_SELECTIVE_TOKENS=64
export VLLM_ASCEND_ENGRAM_PAGEABLE=1
export VLLM_CPU_WEIGHT_MMAP_DIR=$PWD/.work/native30_mmap
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# Custom transformer ops (HcPre etc.).  The wheel only ships ascend910b
# kernels, so use the checkout that has the ascend950 build.
CUSTOM_VENDOR=/home/dlb/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export ASCEND_CUSTOM_OPP_PATH=$CUSTOM_VENDOR${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=$CUSTOM_VENDOR/op_api/lib:/home/dlb/Ascend/install/cann-9.3.0/x86_64-linux/kirin_dependency/lib64:/usr/local/Ascend/nnal/atb/9.1.1/atb/cxx_abi_1/lib:/usr/local/Ascend/nnal/atb/9.1.1/atb/cxx_abi_0/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=$CUSTOM_VENDOR/op_api/lib/libcust_opapi.so${LD_PRELOAD:+:$LD_PRELOAD}

rm -rf "$VLLM_CPU_WEIGHT_MMAP_DIR"
mkdir -p "$VLLM_CPU_WEIGHT_MMAP_DIR"
exec .venv_ascend30/bin/python tools/dsv41/native_v41_entry.py \
  --model /home/models/DeepSeek-V4.1-Flash \
  --served-model-name dsv41 \
  --trust-remote-code \
  --quantization deepseek_v4_fp8 \
  --enforce-eager \
  --max-model-len 32768 \
  --block-size 64 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.85 \
  --offload-backend prefetch \
  --offload-group-size 1 \
  --offload-num-in-group 1 \
  --offload-prefetch-step 1 \
  --offload-params w13_weight w2_weight \
  --no-enable-chunked-prefill \
  --port 8124
