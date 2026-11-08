"""Entry point for the native vllm-ascend V4.1 engine (isolated 0.30 venv).

vllm-ascend 0.30 needs three things that the stock CLI does not do for this
model: register the model class, route the legacy GPUModelRunner's
create_offloader global to the Ascend factory, and set the offload env before
vllm is imported.
"""

import os

os.environ.setdefault("VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY", "1")
os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")

import vllm_ascend

vllm_ascend.register_model()

from vllm_ascend.model_executor.offloader import create_offloader as _ascend_create_offloader

import vllm.v1.worker.gpu.model_runner as _gpu_mr
import vllm.v1.worker.gpu_model_runner as _gpu_mr1

_gpu_mr.create_offloader = _ascend_create_offloader
_gpu_mr1.create_offloader = _ascend_create_offloader

from vllm.entrypoints.launchers.api_server.entry import main

if __name__ == "__main__":
    main()
