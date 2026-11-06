"""``tools/dsv41/ref_cpu.py`` — a torch/CPU transcription of the official CUDA kernels.

The oracle (``/home/models/DeepSeek-V4.1-Flash/inference/model.py``) cannot run here: its layers call
tilelang kernels compiled for ``torch.cuda``.  This module re-implements exactly those six kernels
from ``inference/kernel.py`` in plain torch, so the *official layer code* can run on a CPU box once
its module-level kernel names are swapped for these (see ``s2_layer_check.py``).  It is also the
device implementation until S2's TileLang kernels land, which is why it lives in the package and not
in the test.

Fidelity notes (each one is a place where a "cleaner" implementation would differ):

* ``act_quant`` takes ``amax`` per row per 32-element group, floors it at ``1e-4``, and with
  ``scale_fmt`` set rounds the scale **up to a power of two** (``fast_round_scale`` = ``2^ceil(log2(
  amax/448))``) — not to the nearest power of two, which is what an e8m0 cast would do.
* ``fp4_act_quant`` floors ``amax`` at ``6*2^-9`` (e4m3 scales) or ``6*2^-126`` (e8m0 scales); the
  e4m3 branch takes ``amax/6`` **rounded to the e4m3 grid** (this is the compressed-KV latent path).
* fp4 values land on the E2M1 grid ``{0,.5,1,1.5,2,3,4,6}``; torch has no CPU cast for the packed
  dtype, so the grid is applied by hand with round-nearest-even.
* ``fp8_gemm``/``fp4_gemm`` dequantize **per K block** (32 or 128) and accumulate in fp32; the fp4
  path additionally uses one weight scale per 32 elements of K.
* ``sparse_attn`` starts the running max at ``-1e30`` (not ``-inf``), maps ``idx == -1`` to KV 0 and
  score ``-inf``, casts the softmax probabilities to bf16 before the PV product, and adds
  ``exp(sink - max)`` to the denominator — an all-(-1) row therefore yields zeros, not NaN.
* ``hc_split_sinkhorn`` is ``softmax + eps`` → column-normalize → ``iters-1`` × (row, column), all in
  fp32.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "FP4_GRID",
    "act_quant",
    "fp4_act_quant",
    "quantize_weight_fp8_block32",
    "fp8_gemm",
    "fp4_gemm",
    "sparse_attn",
    "hc_split_sinkhorn",
    "unpack_fp4",
    "linear",
    "embedding",
    "dequant_engram_rows",
    "scale_to_float",
]

FP8_MAX = 448.0
FP4_MAX = 6.0

#: The E2M1 value grid, in bit-pattern order (see ``inference/convert.py:FP4_TABLE``).
FP4_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


# =============================================================================================
# Quantization
# =============================================================================================


def _pow2_ceil_scale(amax: torch.Tensor, inv_max: float) -> torch.Tensor:
    """``fast_round_scale``: ``2^ceil(log2(amax / max))`` in fp32 (exact for powers of two)."""
    return torch.pow(2.0, torch.ceil(torch.log2(amax * inv_max)))


def _pow2_round_scale(s: torch.Tensor) -> torch.Tensor:
    """Nearest power of two, i.e. what storing a scale as e8m0 would do."""
    return torch.pow(2.0, torch.round(torch.log2(s)))


def scale_to_float(scale: torch.Tensor) -> torch.Tensor:
    """Read a scale tensor as fp32, including the ``e8m0`` case the NPU cannot cast directly.

    ``float8_e8m0fnu`` stores ``2^(bits - 127)``: on Ascend a dtype cast for it is unsupported, but a
    bit-view to ``uint8`` is not, so the exponent is decoded by hand.  On a device that supports the
    cast this is still exact (the value is a power of two).
    """
    if scale.dtype == torch.float8_e8m0fnu:
        bits = scale.view(torch.uint8).to(torch.int32)
        return torch.pow(2.0, (bits - 127).float())
    return scale.float()


def _quantize_fp4(x: torch.Tensor) -> torch.Tensor:
    """Round to the E2M1 grid (values must already be ``clamp``-ed to +-6)."""
    grid = torch.tensor(FP4_GRID, dtype=torch.float32, device=x.device)
    mid = (grid[:-1] + grid[1:]) / 2.0  # 7 midpoints
    a = x.abs().float()
    idx = torch.searchsorted(mid, a, right=True)  # 0..7, ties go to the larger value
    lo = (idx - 1).clamp_min(0)
    hi = idx.clamp_max(grid.numel() - 1)
    d_lo = (a - grid[lo]).abs()
    d_hi = (grid[hi] - a).abs()
    take_lo = d_lo < d_hi
    tie = d_lo == d_hi
    # round-nearest-even: on a tie pick the candidate with an even LSB in the bit pattern
    take_lo = torch.where(tie, (lo % 2 == 0), take_lo)
    val = torch.where(take_lo, grid[lo], grid[hi])
    # sign() rather than copysign(): Ascend has no copysign and would fall back to the CPU
    return val * torch.sign(x)


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
):
    """Block-wise fp8 quantization (``kernel.py:act_quant``).

    ``scale_fmt`` set (``"ue8m0"``) means power-of-two scales; ``inplace`` writes the
    quantize-dequantize result back as the input dtype (bf16) and returns ``x``.
    """
    assert x.size(-1) % block_size == 0, (x.shape, block_size)
    z = x.contiguous()
    dtype = z.dtype
    f = z.float()
    groups = f.unflatten(-1, (-1, block_size))
    amax = groups.abs().amax(dim=-1).clamp_min(1e-4)
    if scale_fmt is not None:
        s = _pow2_ceil_scale(amax, 1.0 / FP8_MAX)
    else:
        s = amax * (1.0 / FP8_MAX)
        if scale_dtype == torch.float8_e8m0fnu:
            s = _pow2_round_scale(s)  # what storing it as e8m0 would round to
    q = (groups / s.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    deq = q.float() * s.unsqueeze(-1)
    y = deq.flatten(-2)
    if inplace:
        x.copy_(y.to(dtype))
        return x
    # ``scale_dtype`` is honoured by rounding ``s`` (see above) but the tensor is returned in fp32:
    # storing it as e8m0 would be a lossless re-encoding of a power of two, and Ascend cannot cast
    # to that dtype at all (aclnnInplaceCopy 561103).  ``scale_to_float`` reads either form.
    return q.flatten(-2), s


def fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
    scale_dtype: torch.dtype = torch.float8_e8m0fnu,
):
    """Block-wise fp4 quantization with e8m0 or e4m3 scales (``kernel.py:fp4_act_quant``)."""
    assert x.size(-1) % block_size == 0, (x.shape, block_size)
    z = x.contiguous()
    dtype = z.dtype
    groups = z.float().unflatten(-1, (-1, block_size))
    if scale_dtype == torch.float8_e4m3fn:
        amax = groups.abs().amax(dim=-1).clamp_min(FP4_MAX * 2**-9)
        s = (amax / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        amax = groups.abs().amax(dim=-1).clamp_min(FP4_MAX * 2**-126)
        s = _pow2_ceil_scale(amax, 1.0 / FP4_MAX)
    q = _quantize_fp4((groups / s.unsqueeze(-1)).clamp(-FP4_MAX, FP4_MAX))
    deq = (q * s.unsqueeze(-1)).flatten(-2)
    if inplace:
        x.copy_(deq.to(dtype))
        return x
    return deq, s.to(scale_dtype)


# =============================================================================================
# fp4 storage
# =============================================================================================


def unpack_fp4(packed: torch.Tensor, out_features: int, in_features: int) -> torch.Tensor:
    """``float4_e2m1fn_x2 [out, in//2]`` -> fp32 ``[out, in]`` (low nibble first, as convert.py)."""
    raw = packed.view(torch.uint8)
    low = raw & 0x0F
    high = (raw >> 4) & 0x0F
    table = torch.tensor(
        FP4_GRID + (0.0,) + tuple(-v for v in FP4_GRID[1:]),
        dtype=torch.float32,
        device=packed.device,
    )  # 16 entries, indexed by the 4-bit pattern (bit 3 = sign, bits 0..2 = magnitude)
    out = torch.stack([table[low.long()], table[high.long()]], dim=-1).flatten(-2)
    return out.reshape(out_features, in_features)


def unpack_fp4_lut(packed: torch.Tensor, out_features: int, in_features: int, dtype=None) -> torch.Tensor:
    """Same mapping as :func:unpack_fp4 but one byte-indexed gather instead of two nibble gathers.

    The packed byte holds two codes, so a 256-entry lookup table of *pairs* turns the whole expansion
    into a single `index_select`: no `&`/`>>` on 5.9M elements, no `stack` + `flatten` copy.
    The fp4 grid values (0, .5, 1, 1.5, 2, 3, 4, 6 and their negatives) are exact in bf16, so building
    the table in the GEMM dtype is lossless.
    """
    raw = packed.view(torch.uint8)
    grid = FP4_GRID + (0.0,) + tuple(-v for v in FP4_GRID[1:])
    codes = torch.arange(256, device=packed.device)
    targets = dtype if dtype is not None else torch.float32  # fp4 storage dtypes are not table dtypes
    lut16 = torch.tensor(grid, dtype=targets, device=packed.device)
    pair = torch.empty(256, 2, dtype=targets, device=packed.device)
    pair[:, 0] = lut16[codes & 0x0F]
    pair[:, 1] = lut16[(codes >> 4) & 0x0F]
    out = torch.index_select(pair, 0, raw.reshape(-1).to(torch.int64))
    return out.reshape(out_features, in_features)


def quantize_weight_fp8_block32(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a *weight* matrix the way the checkpoint stores it: one e8m0 scale per 32x32 block.

    Note the difference from :func:`act_quant`: a linear weight has a 2-D block scale
    (``[out/32, in/32]``, shared by all 1024 elements of the block), while an activation has a
    per-row scale (``[M, K/32]``).  ``fp8_gemm`` consumes exactly these two layouts.
    """
    w = weight.float()
    out_f, in_f = w.shape
    assert out_f % 32 == 0 and in_f % 32 == 0, (out_f, in_f)
    blocks = w.unflatten(0, (-1, 32)).unflatten(-1, (-1, 32))  # [ob, 32, ib, 32]
    amax = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-4)  # over the 32x32 block -> [ob, ib]
    scale = _pow2_ceil_scale(amax, 1.0 / FP8_MAX)
    q = (blocks / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.flatten(2, 3).flatten(0, 1), scale.to(torch.float8_e8m0fnu)


def _fp4_codes(x: torch.Tensor) -> torch.Tensor:
    """E2M1 bit patterns for ``x`` (already clamped to +-6): ``sign << 3 | magnitude index``."""
    grid = torch.tensor(FP4_GRID, dtype=torch.float32, device=x.device)
    mid = (grid[:-1] + grid[1:]) / 2.0
    a = x.abs().float()
    idx = torch.searchsorted(mid, a, right=True)
    lo = (idx - 1).clamp_min(0)
    hi = idx.clamp_max(grid.numel() - 1)
    d_lo = (a - grid[lo]).abs()
    d_hi = (grid[hi] - a).abs()
    take_lo = d_lo < d_hi
    take_lo = torch.where(d_lo == d_hi, (lo % 2 == 0), take_lo)
    mag = torch.where(take_lo, lo, hi).to(torch.int64)
    sign = (x < 0).to(torch.int64) << 3
    # pattern 8 is "negative zero"; keep +0 for exact zeros so the bytes stay canonical
    code = torch.where(x == 0, torch.zeros_like(mag), mag | sign)
    return code


def quantize_weight_fp4_block32(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight matrix to MXFP4 as the checkpoint stores it.

    Returns ``(packed_uint8 [out, in//2], scale_e8m0 [out, in//32])``: one e8m0 scale per 32 elements
    along K (the *reduce* dim), two 4-bit values per byte, low nibble first.  torch has no CPU cast to
    ``float4_e2m1fn_x2``, so the packing is done by hand (which is also what ``convert.py`` assumes
    when it reads ``I8`` shards as fp4).
    """
    w = weight.float()
    out_f, in_f = w.shape
    assert in_f % 32 == 0 and out_f % 32 == 0, (out_f, in_f)
    groups = w.unflatten(-1, (-1, 32))
    amax = groups.abs().amax(dim=-1).clamp_min(FP4_MAX * 2**-126)
    scale = _pow2_ceil_scale(amax, 1.0 / FP4_MAX)  # [out, in//32]
    q = (groups / scale.unsqueeze(-1)).clamp(-FP4_MAX, FP4_MAX).flatten(-2)
    codes = _fp4_codes(q).to(torch.uint8)
    low = codes[:, 0::2]
    high = codes[:, 1::2]
    packed = (low | (high << 4)).to(torch.uint8)
    return packed, scale.to(torch.float8_e8m0fnu)


# =============================================================================================
# GEMMs
# =============================================================================================


def fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
    block_size: int = 128,
) -> torch.Tensor:
    """``C[M,N] = A[M,K] @ B[N,K]^T`` with one scale per ``block_size`` on both operands."""
    K = a.size(-1)
    N = b.size(0)
    a_f = a.float() * scale_to_float(a_s).repeat_interleave(block_size, dim=-1)
    b_f = b.float() * scale_to_float(b_s).repeat_interleave(block_size, dim=0).repeat_interleave(
        block_size, dim=-1
    )
    b_f = b_f[:N, :K]
    c = a_f.reshape(-1, K) @ b_f.transpose(0, 1)
    return c.to(torch.get_default_dtype()).reshape(*a.shape[:-1], N)


def fp4_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
    act_block_size: int = 128,
) -> torch.Tensor:
    """``C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T`` with per-32 weight scales (``kernel.py:fp4_gemm``)."""
    K = a.size(-1)
    N = b.size(0)
    b_float = unpack_fp4(b, N, K)
    a_f = a.float() * scale_to_float(a_s).repeat_interleave(act_block_size, dim=-1)
    b_f = b_float * scale_to_float(b_s).repeat_interleave(32, dim=-1)[:, :K]
    c = a_f.reshape(-1, K) @ b_f.transpose(0, 1)
    return c.to(torch.get_default_dtype()).reshape(*a.shape[:-1], N)


def linear(x: torch.Tensor, weight: torch.Tensor, scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``inference/model.py:linear`` — pick the GEMM from the weight's storage dtype.

    fp4 and fp8 weights both take an fp8 activation (one e8m0 scale per 32), for fp4 the kernel does
    the mixed-precision product.
    """
    if weight.dtype == torch.float4_e2m1fn_x2:
        xq, s = act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu, False)
        return fp4_gemm(xq, s, weight, scale, torch.float8_e8m0fnu, act_block_size=32)
    if weight.dtype == torch.float8_e4m3fn:
        xq, s = act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu, False)
        return fp8_gemm(xq, s, weight, scale, torch.float8_e8m0fnu, block_size=32)
    return F.linear(x, weight)


def embedding(input_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``ParallelEmbedding`` on one rank: a plain gather (no all-reduce)."""
    return F.embedding(input_ids, weight)


def dequant_engram_rows(table, hash_ids: torch.Tensor) -> torch.Tensor:
    """``ParallelEngramEmbedding.forward``: gather fp8 rows + one e8m0 scale per 32 columns."""
    values = F.embedding(hash_ids, table.weight)
    scales = F.embedding(hash_ids, table.scale)
    out = values.float().unflatten(-1, (-1, 32)) * scale_to_float(scales).unsqueeze(-1)
    return out.flatten(-2).to(torch.bfloat16)


# =============================================================================================
# Sparse attention
# =============================================================================================


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Gathered-index flash attention with an attention sink (``kernel.py:sparse_attn``).

    ``q [b,m,h,d]`` bf16, ``kv [b,n,d]`` bf16, ``attn_sink [h]`` fp32, ``topk_idxs [b,m,topk]`` int32.
    """
    b, m, h, d = q.size()
    topk = topk_idxs.size(-1)
    device = q.device
    block = 64
    acc = torch.zeros(b, m, h, d, dtype=torch.float32, device=device)
    total = torch.zeros(b, m, h, dtype=torch.float32, device=device)
    running_max = torch.full((b, m, h), -1e30, dtype=torch.float32, device=device)

    qf = q.float()
    batch_idx = torch.arange(b, device=device).view(b, 1, 1)
    n_blocks = (topk + block - 1) // block
    for t in range(n_blocks):
        idx = topk_idxs[..., t * block : (t + 1) * block]
        pad = block - idx.size(-1)
        if pad:
            idx = F.pad(idx, (0, pad), value=-1)
        valid = idx != -1
        safe = idx.clamp_min(0)
        kv_blk = kv[batch_idx, safe]  # [b,m,block,d]
        kv_blk = torch.where(valid.unsqueeze(-1), kv_blk, torch.zeros_like(kv_blk)).float()
        s = torch.einsum("bmhd,bmkd->bmhk", qf, kv_blk) * softmax_scale
        s = torch.where(valid.unsqueeze(2), s, torch.full_like(s, -math.inf))

        new_max = torch.maximum(running_max, s.amax(dim=-1))
        corr = torch.exp(running_max - new_max)
        p = torch.exp(s - new_max.unsqueeze(-1))
        total = total * corr + p.sum(dim=-1)
        acc = acc * corr.unsqueeze(-1) + torch.einsum("bmhk,bmkd->bmhd", p.to(torch.bfloat16).float(), kv_blk)
        running_max = new_max

    total = total + torch.exp(attn_sink.float() - running_max)
    acc = acc / total.unsqueeze(-1)
    return acc.to(q.dtype)


# =============================================================================================
# Hyper-Connections
# =============================================================================================


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the mix vector into ``pre``/``post``/``comb`` (``kernel.py:hc_split_sinkhorn``)."""
    hc = int(hc_mult)
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2.0 * torch.sigmoid(m[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = m[..., 2 * hc : 2 * hc + hc * hc] * hc_scale[2] + hc_base[2 * hc : 2 * hc + hc * hc]
    comb = comb.unflatten(-1, (hc, hc))
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(int(sinkhorn_iters) - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb
