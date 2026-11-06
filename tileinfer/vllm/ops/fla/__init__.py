"""The FLA/GDN operators the GDN layers call, as exact torch ports (no Triton, no CUDA).

Layout of the package
---------------------
``gdn.py`` (one level up)
    The two stateless leaves: ``l2norm_fwd`` and ``chunk_local_cumsum``.
``solve_tril`` / ``kkt`` / ``wu`` / ``delta_h`` / ``chunk_o``
    The five chunked-prefill operators, in the order the chain calls them.
``chunk.py``
    The composed chain (``chunk_gated_delta_rule_fwd`` is vLLM's six-step pipeline) and the public
    entry ``chunk_gated_delta_rule``, which returns ``(o, final_state)`` exactly like upstream.
``causal_conv1d``
    ``causal_conv1d_fn`` (prefill, ``[dim, cu_seqlen]``) and ``causal_conv1d_update`` (decode,
    token-major) - two different layouts, deliberately two functions.
``gating`` / ``recurrent``
    The decode path: gate + state update, and the packed one-token-per-sequence recurrence.

Every public name here is signature-compatible with the vLLM 0.23 function of the same name, so
``tileinfer.vllm.ops.install`` can rebind it there (see that module for which names are actually
bound and which are only reported).
"""

from __future__ import annotations

from .causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from .chunk import (
    FLA_CHUNK_SIZE,
    assert_strict_lower_chunks,
    chunk_gated_delta_rule,
    chunk_gated_delta_rule_fwd,
)
from .chunk_o import chunk_fwd_o
from .delta_h import chunk_gated_delta_rule_fwd_h
from .gating import fused_sigmoid_gating_delta_rule_update
from .kkt import chunk_scaled_dot_kkt_fwd
from .recurrent import (
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_packed_decode,
)
from .solve_tril import prepare_chunk_indices, solve_tril
from .wu import recompute_w_u_fwd

__all__ = [
    "FLA_CHUNK_SIZE",
    "assert_strict_lower_chunks",
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "chunk_fwd_o",
    "chunk_gated_delta_rule",
    "chunk_gated_delta_rule_fwd",
    "chunk_gated_delta_rule_fwd_h",
    "chunk_scaled_dot_kkt_fwd",
    "fused_recurrent_gated_delta_rule",
    "fused_recurrent_gated_delta_rule_packed_decode",
    "fused_sigmoid_gating_delta_rule_update",
    "prepare_chunk_indices",
    "recompute_w_u_fwd",
    "solve_tril",
]
