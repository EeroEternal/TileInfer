"""Short causal convolution state math (engine-free).

The vendor state provider is GDN (``vllm_ascend.ops.gdn``), which cannot serve an LFM3
gated depthwise convolution.  This module is the pure-torch implementation of the fused
unit: one causal depthwise convolution plus its rolling state.  The vLLM layer in
``tileinfer/bridge/models/lfm3_model.py`` owns the metadata and buffer plumbing; the numbers live here so
they can be tested on CPU.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class ShortConvState:
    """Causal depthwise convolution with a ``kernel - 1`` rolling input state."""

    @staticmethod
    def prefill(
        x: torch.Tensor,
        weight: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convolve ``[tokens, dim]`` and return ``(output, new_state)``.

        ``weight`` is ``[dim, 1, kernel]`` and ``state`` is ``[dim, kernel - 1]`` in the
        dim-first layout used by vLLM's state cache.
        """
        dim = x.shape[-1]
        width = weight.shape[-1]
        keep = width - 1
        if state is None:
            state = x.new_zeros((dim, keep))
        stream = torch.cat((state, x.transpose(0, 1)), dim=-1)
        out = F.conv1d(stream.unsqueeze(0), weight, bias, groups=dim)
        return out[0].transpose(0, 1), stream[:, -keep:].contiguous()

    @staticmethod
    def step(
        x: torch.Tensor,
        weight: torch.Tensor,
        state: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One-token decode; ``x`` is ``[1, dim]`` and ``state`` is updated in place."""
        return ShortConvState.prefill(x, weight, state, bias)

    @staticmethod
    def step_batch(
        x: torch.Tensor,
        weight: torch.Tensor,
        state: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode a batch without host-side state-index reads.

        ``x`` is ``[batch, dim]`` and ``state`` is ``[batch, dim, kernel - 1]``.
        """
        stream = torch.cat((state, x.unsqueeze(-1)), dim=-1)
        # Seven fused multiply-adds beat a 2048-group depthwise conv on Ascend for a
        # one-token decode (the grouped convolution costs ~8 ms per layer there).
        out = (stream * weight[:, 0, :].unsqueeze(0)).sum(-1)
        if bias is not None:
            out = out + bias
        return out, stream[..., 1:].contiguous()

    @staticmethod
    def prefill_batch(
        x: torch.Tensor,
        weight: torch.Tensor,
        state: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorised prefill over packed requests.

        ``x`` is ``[tokens, dim]``; ``starts``/``ends`` are ``[requests]``; ``state`` is
        ``[requests, dim, kernel - 1]``.  Every operation is tensor-only so the layer can be
        captured by a graph.
        """
        requests = starts.numel()
        tokens, dim = x.shape
        keep = weight.shape[-1] - 1
        device = x.device
        positions = torch.arange(tokens, device=device)
        request_ids = (
            (positions.unsqueeze(0) >= starts.unsqueeze(1)).sum(0).sub(1).clamp_min(0)
        )
        local = positions - starts[request_ids]

        stream = x.new_zeros((requests, dim, keep + tokens))
        stream[:, :, :keep] = state
        stream[request_ids, :, keep + local] = x

        conv = F.conv1d(stream, weight, bias, groups=dim)
        out = conv[request_ids, :, local]

        offsets = torch.arange(keep, device=device)
        indices = (ends - starts).unsqueeze(1) + offsets.unsqueeze(0)
        new_state = torch.gather(
            stream, 2, indices.unsqueeze(1).expand(requests, dim, keep)
        )
        return out, new_state.contiguous()



__all__ = ["ShortConvState"]
