"""Operator implementations that replace the CUDA/Triton ones vLLM would otherwise call.

Name, precisely: ``ops`` vs ``kernels``
---------------------------------------
``tileinfer/vllm/ops/**`` (this package) is the **vLLM operator ABI** layer: the module and function
names are vLLM's, the bodies are torch, and the contract is correctness-first - a bit-comparable
replacement that ``install.py`` binds to the call sites vLLM actually uses, and that the CPU suite
can test against a reference.  ``tileinfer/kernels/**`` is the **TileLang kernel** layer:
device-compiled performance code (it imports ``tilelang``/``torch_npu`` at module level) that some
of these ops dispatch to only where a profile says it pays.  An op here may have no matching kernel
yet; a kernel there may have no vLLM call site yet.  Neither directory is an alias for the other:
seeing ``ops/fla/chunk.py`` does not mean TileLang is involved, and seeing ``kernels/gdn/`` does not
mean the vLLM ABI is implemented.

Every module here exists because the corresponding upstream kernel cannot run on Ascend: vLLM 0.23
routes dense, GDN, MoE and sampling through either ``torch.ops._C`` (compiled CUDA) or its in-tree
Triton FLA port, and neither is available.  The rule for this package is **correctness first**: a
torch implementation that matches the reference exactly, then - only where a profile says it pays -
the matching TileInfer kernel from ``tileinfer/kernels/``.

``gdn``
    ``l2norm_fwd`` and ``chunk_local_cumsum``, the two stateless FLA leaves.
``moe``
    ``topk_softmax``, ``moe_align_block_size``, ``fused_experts`` and ``silu_and_mul`` - the five
    operators a MoE layer needs (see the module docstring for which upstream behaviour is
    [unverified] offline and how each uncertainty is made selectable instead of guessed).
``fused_moe_oot``
    The out-of-tree ``UnquantizedFusedMoEMethod`` vLLM 0.23 documents as the platform's extension
    point (``CustomOp.register_oot``): a class factory whose subclass fills the kernel slot the OOT
    branch leaves empty and routes the expert compute to ``moe.fused_experts``.  It imports **no**
    vLLM - the base class is a parameter - so it belongs to this package rather than to the
    installer; ``install.py``'s third patch kind (``custom_op_oot``) is what registers it.
``fla``
    The GDN operator family: the five chunked-prefill kernels, the composed
    ``chunk_gated_delta_rule`` chain, ``causal_conv1d_fn``/``causal_conv1d_update``, and the decode
    pair ``fused_sigmoid_gating_delta_rule_update`` /
    ``fused_recurrent_gated_delta_rule_packed_decode``.  ``from tileinfer.vllm.ops.fla import ...``
    is the whole public surface.
``switch``
    The ``TILEINFER_OPS`` family switch (``gdn``/``moe``/``dense``).
``install``
    ``install_ops()``: the wiring itself - which vLLM name each implementation is bound to, and the
    ``Patch`` record that documents why.  Imported by ``tileinfer.vllm.plugin`` as its step 4; not
    imported here, because it is the only module in the package that talks to vLLM at all and
    importing it eagerly would make every ``tileinfer.vllm.ops.<x>`` import pay for the probe.
"""

from __future__ import annotations

from .gdn import chunk_local_cumsum, l2norm_fwd
from .moe import (
    fused_experts,
    moe_align_block_size,
    moe_align_block_size_block_first,
    silu_and_mul,
    topk_softmax,
)

__all__ = [
    "chunk_local_cumsum",
    "fused_experts",
    "l2norm_fwd",
    "moe_align_block_size",
    "moe_align_block_size_block_first",
    "silu_and_mul",
    "topk_softmax",
]
