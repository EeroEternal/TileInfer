"""The forward math of one DeepSeek-V4.1 attention block.

A transcription of ``inference/model.py``'s ``Attention`` / ``Compressor`` / ``Indexer`` /
``select_candidate_blocks`` / ``get_window_topk_idxs``, expressed against this package's parameter
tree and a kernel set (:mod:`tools.dsv41.ref_cpu` today, TileLang kernels once the device port lands).

Deliberate fidelity notes -- each is observable in a max|diff| check:

* ``Compressor.wkv``/``wgate`` are bf16 here and fp32 in the reference (which upcasts at load).  The
  upcast is exact, so this module casts the bf16 weights to fp32 before the pooling matmul, which is
  bit-identical to the reference's stored-fp32 weights.
* The compressed stream is published through a shared object, not returned: the last kv-source layer
  writes, every layer after it reads, so the four segments are (2-7), (8-13), (14-19), (20-39).
  Same for the index keys and the per-query top-k list (facts section 6.3).
* ``window_kv`` and ``compress_kv`` hold quantize-dequantized values, so "fp8 KV" means "bf16 values
  on the e4m3 grid".
* RoPE is applied with real cos/sin tables instead of the reference's complex ``view_as_complex``
  (Ascend has no such op); the values are identical, the reference's ``polar(1, theta)`` being
  ``(cos theta, sin theta)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .config import Dsv41Config
from .modules import Dsv41Attention, Dsv41Compressor, Dsv41Indexer, Dsv41RMSNorm

__all__ = [
    "SharedAttentionState",
    "LayerRuntime",
    "attention_forward",
    "compressor_forward",
    "indexer_forward",
    "select_candidate_blocks",
    "window_topk_idxs",
    "rotate",
    "rms_norm",
    "precompute_rope_cos_sin",
    "ops",
]

_E8M0 = torch.float8_e8m0fnu
_FP8 = torch.float8_e4m3fn


def ops():
    """The kernel set in force (torch reference until the TileLang port lands)."""
    from tools.dsv41 import ref_cpu

    return ref_cpu


# =============================================================================================
# Shared per-forward state (the model's ``shared_attn`` singleton)
# =============================================================================================


@dataclass
class SharedAttentionState:
    """What attention layers hand down the stack instead of recomputing (facts section 6.3)."""

    compress_kv: Optional[torch.Tensor] = None
    index_k: Optional[torch.Tensor] = None
    topk_idxs: Optional[torch.Tensor] = None
    candidates: Optional[torch.Tensor] = None


# =============================================================================================
# Small numerical helpers (``inference/model.py`` semantics)
# =============================================================================================


def rms_norm(norm: Dsv41RMSNorm, x: torch.Tensor) -> torch.Tensor:
    """``model.py:RMSNorm`` -- fp32 statistics, ``weight * x_hat`` back in the input dtype."""
    dtype = x.dtype
    xf = x.float()
    var = xf.square().mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + norm.eps)
    return (norm.weight.float() * xf).to(dtype)


def precompute_rope_cos_sin(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``model.py:precompute_freqs_cis`` as real ``(cos, sin)`` tables, one row per position."""
    freqs = 1.0 / (
        torch.pow(
            torch.tensor(float(base), dtype=torch.float32, device=device),
            torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim,
        )
    )
    if original_seq_len > 0:

        def corrected_dim(rotations: int) -> float:
            return (
                dim
                * math.log(original_seq_len / (rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = (
            (torch.arange(dim // 2, dtype=torch.float32, device=device) - low)
            / max(high - low, 1e-3)
        ).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    angles = torch.outer(torch.arange(seqlen, device=device).float(), freqs)
    return angles.cos(), angles.sin()


def rotate(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """``model.py:apply_rotary_emb`` on adjacent element pairs, in place on ``x[..., -rd:]``.

    ``cos``/``sin`` are ``[seqlen, rd/2]`` fp32; ``inverse`` conjugates the rotation (negating sin).
    """
    rd = cos.size(-1) * 2
    target = x[..., -rd:]
    pairs = target.float().unflatten(-1, (-1, 2))
    x1, x2 = pairs[..., 0], pairs[..., 1]
    if inverse:
        sin = -sin
    n = target.size(1)
    if target.ndim == 3:
        c, s = cos.view(1, n, -1), sin.view(1, n, -1)
    else:
        c, s = cos.view(1, n, 1, -1), sin.view(1, n, 1, -1)
    out = torch.stack([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).flatten(-2)
    target.copy_(out.to(target.dtype))
    return x


def window_slots_for(pos: int, window_size: int, device="cpu") -> torch.Tensor:
    """绝对位置 ``pos`` 处的 query 可以读的环形槽位（``window_size`` 个，-1 表示还没写）。

    这是**投机解码**（一次验证 k 个 token）需要的基础件：位置 ``pos`` 的 query 只能看到
    ``[pos-window_size+1, pos]`` 这些绝对位置对应的槽位，早于 0 的置 -1。

    现有的 :func:`window_topk_idxs` 在 ``start_pos > 0`` 时对所有 query 返回同一套槽位，
    只对 ``seqlen == 1`` 成立；``seqlen > 1``（分块解码 / 投机验证）必须逐 query 计算，
    本函数就是那条逐 query 的实现，纯索引运算，可离线单测。
    """
    lo = pos - window_size + 1
    abs_pos = torch.arange(lo, pos + 1, device=device)
    slots = abs_pos % window_size
    return torch.where(abs_pos < 0, torch.full_like(slots, -1), slots)


def window_topk_idxs_batched(window_size: int, positions, device="cpu") -> torch.Tensor:
    """批处理版：``positions`` 是 ``[R]`` 个请求各自的绝对位置，返回 ``[R, window_size]`` 槽位。

    这是真批处理前向（`perf-plan.md` §24 的 B 步）所需的重载：现有
    :func:`window_topk_idxs` 只接受**单个** ``start_pos``，批处理时每个请求的窗口不同。

    纯索引运算，可离线与逐请求调用逐一比对（见
    `tests/test_dsv41_batch_windows.py`）。旧签名与旧行为完全不变。
    """
    pos_list = [int(p) for p in positions]
    if not pos_list:
        return torch.empty(0, window_size, dtype=torch.int32, device=device)
    rows = [window_slots_for(p, window_size, device=device) for p in pos_list]
    return torch.stack(rows, dim=0).int()


def window_topk_idxs(
    window_size: int, bsz: int, seqlen: int, start_pos: int, device="cpu"
) -> torch.Tensor:
    """``model.py:get_window_topk_idxs`` -- which ring slots each query may read, -1 = empty."""
    if start_pos == 0:
        end = torch.arange(seqlen, device=device).unsqueeze(1)
        idxs = (end - window_size + 1).clamp(0) + torch.arange(
            min(seqlen, window_size), device=device
        )
        idxs = torch.where(idxs > end, -1, idxs)
    elif seqlen == 1:
        oldest = start_pos % window_size + 1
        idxs = torch.cat(
            [
                torch.arange(oldest, window_size, device=device),
                torch.arange(oldest, device=device),
            ]
        )
        idxs = torch.where(idxs > start_pos, -1, idxs)
    else:
        # 分块解码 / 投机验证：每个 query 看到的窗口不同，必须逐位置计算。
        # （老写法对所有 query 返回同一套槽位，只对 seqlen==1 成立。）
        idxs = torch.stack(
            [window_slots_for(start_pos + i, window_size, device=device) for i in range(seqlen)],
            dim=0,
        )
    return idxs.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


def select_candidate_blocks(
    logits: torch.Tensor, compress_lens, topk_blocks: int, block_size: int
) -> torch.Tensor:
    """``model.py:select_candidate_blocks`` -- level one of the two-level top-k (facts 7.3)."""
    width = logits.size(-1)
    scores = torch.nn.functional.pad(logits, (0, -width % block_size), value=-math.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, math.inf)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -math.inf
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


# =============================================================================================
# Per-layer buffers
# =============================================================================================


@dataclass
class LayerRuntime:
    """The state one attention layer carries across steps, plus the shared sink object."""

    cfg: Dsv41Config
    layer_id: int
    shared: SharedAttentionState
    bsz: int = 1
    max_seq_len: int = 4096
    window_kv: Optional[torch.Tensor] = None
    compress_kv: Optional[torch.Tensor] = None
    index_k: Optional[torch.Tensor] = None
    kv_state: Optional[torch.Tensor] = None
    score_state: Optional[torch.Tensor] = None
    cos: Optional[torch.Tensor] = None
    sin: Optional[torch.Tensor] = None
    device: str = "cpu"

    def build(self) -> "LayerRuntime":
        cfg, b = self.cfg, self.bsz
        self.window_kv = torch.zeros(
            b, cfg.sliding_window, cfg.head_dim, dtype=torch.bfloat16, device=self.device
        )
        ratio = cfg.compress_ratio(self.layer_id)
        self.cos, self.sin = precompute_rope_cos_sin(
            cfg.rope_head_dim,
            self.max_seq_len,
            cfg.original_seq_len if ratio else 0,
            cfg.compress_rope_theta if ratio else cfg.rope_theta,
            cfg.rope_factor,
            cfg.beta_fast,
            cfg.beta_slow,
            device=self.device,
        )
        if cfg.is_kv_source(self.layer_id):
            self.compress_kv = torch.zeros(
                b,
                self.max_seq_len // max(ratio, 1),
                cfg.head_dim,
                dtype=torch.bfloat16,
                device=self.device,
            )
            if ratio > 1:
                self.kv_state = torch.zeros(
                    b, ratio, cfg.head_dim, dtype=torch.float32, device=self.device
                )
                self.score_state = torch.full(
                    (b, ratio, cfg.head_dim),
                    -math.inf,
                    dtype=torch.float32,
                    device=self.device,
                )
        if cfg.owns_index_k(self.layer_id):
            self.index_k = torch.zeros(
                b,
                self.max_seq_len // max(ratio, 1),
                cfg.index_head_dim,
                dtype=torch.bfloat16,
                device=self.device,
            )
        return self


# =============================================================================================
# Compressor
# =============================================================================================


def _linear_fp32(x: torch.Tensor, slot) -> torch.Tensor:
    """``linear`` against a slot the reference holds in fp32; our tree stores the bf16 checkpoint.

    bf16 -> fp32 is exact, so casting here is identical to the reference's load-time upcast.
    """
    return torch.nn.functional.linear(x, slot.weight.float())


def compressor_forward(
    comp: Dsv41Compressor, x: torch.Tensor, start_pos: int, rt: LayerRuntime
) -> Optional[torch.Tensor]:
    """``model.py:Compressor.forward`` -- pool ``ratio`` tokens into one pre-RoPE latent."""
    ratio, dtype = comp.compress_ratio, x.dtype
    bsz, seqlen, _ = x.shape
    if ratio == 1:
        return rms_norm(comp.norm, ops().linear(x, comp.wkv.weight, comp.wkv.scale))

    xf = x.float()
    kv = _linear_fp32(xf, comp.wkv)
    score = _linear_fp32(xf, comp.wgate)
    if start_pos == 0:
        should_compress = seqlen >= ratio
        remainder = seqlen % ratio
        cutoff = seqlen - remainder
        if remainder:
            kv, kv_tail = kv.split([cutoff, remainder], dim=1)
            score, score_tail = score.split([cutoff, remainder], dim=1)
            rt.kv_state[:bsz, :remainder] = kv_tail
            rt.score_state[:bsz, :remainder] = score_tail
        kv = kv.unflatten(1, (-1, ratio))
        score = score.unflatten(1, (-1, ratio))
        kv = (kv * score.softmax(dim=2)).sum(dim=2)
    else:
        if seqlen == 1:
            should_compress = (start_pos + 1) % ratio == 0
            slot = start_pos % ratio
            rt.kv_state[:bsz, slot] = kv.squeeze(1)
            rt.score_state[:bsz, slot] = score.squeeze(1)
            if should_compress:
                kv = (
                    rt.kv_state[:bsz] * rt.score_state[:bsz].softmax(dim=1)
                ).sum(dim=1, keepdim=True)
        else:
            # Chunked prefill: a chunk that starts and ends on a compression boundary produces
            # ``seqlen // ratio`` complete latents, so no partial-block state is carried.
            if start_pos % ratio or seqlen % ratio:
                raise ValueError(
                    "chunked compressor requires alignment to ratio: "
                    f"start_pos={start_pos} seqlen={seqlen} ratio={ratio}"
                )
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
            should_compress = True
    if not should_compress:
        return None
    return rms_norm(comp.norm, kv.to(dtype))


# =============================================================================================
# Indexer
# =============================================================================================


def indexer_forward(
    ix: Dsv41Indexer,
    x: torch.Tensor,
    qr: torch.Tensor,
    latent: Optional[torch.Tensor],
    start_pos: int,
    offset: int,
    rt: LayerRuntime,
    cfg: Dsv41Config,
) -> torch.Tensor:
    """``model.py:Indexer.forward`` -- the second index level (facts section 7)."""
    o = ops()
    bsz, seqlen, _ = x.shape
    ratio, rd = cfg.compress_ratio(rt.layer_id), cfg.rope_head_dim
    end_pos = start_pos + seqlen
    if ratio == 0:
        raise ValueError("indexer on a non-compressing layer")

    if ix.owns_k and latent is not None:
        n_lat = int(latent.size(1))
        if start_pos == 0:
            c = rt.cos[: seqlen - seqlen % ratio : ratio]
            s = rt.sin[: seqlen - seqlen % ratio : ratio]
        elif n_lat == 1:
            c = rt.cos[start_pos + 1 - ratio].unsqueeze(0)
            s = rt.sin[start_pos + 1 - ratio].unsqueeze(0)
        else:
            # Chunked prefill: latents cover consecutive blocks starting at ``start_pos``.
            if start_pos % ratio:
                raise ValueError(
                    "chunked indexer requires start_pos % ratio == 0: "
                    f"start_pos={start_pos} ratio={ratio}"
                )
            pos = torch.arange(n_lat, device=x.device) * ratio + start_pos
            c = rt.cos[pos]
            s = rt.sin[pos]
        k = rms_norm(ix.k_norm, o.linear(latent, ix.wk.weight, ix.wk.scale))
        rotate(k[..., -rd:], c, s)
        o.fp4_act_quant(k, 32, True)
        rt.index_k[:bsz, start_pos // ratio : start_pos // ratio + k.size(1)] = k
        rt.shared.index_k = rt.index_k

    q = o.linear(qr, ix.wq_b.weight, ix.wq_b.scale).unflatten(
        -1, (cfg.index_n_heads, cfg.index_head_dim)
    )
    rotate(q[..., -rd:], rt.cos[start_pos:end_pos], rt.sin[start_pos:end_pos])
    o.fp4_act_quant(q, 32, True)

    index_k = rt.shared.index_k[:bsz, : end_pos // ratio]
    # the reference scales the projected weights in the activation dtype and reduces in it too
    weights = o.linear(x, ix.weights_proj.weight, ix.weights_proj.scale) * (
        cfg.index_head_dim**-0.5 * cfg.index_n_heads**-0.5
    )

    if start_pos == 0:
        compress_lens = (torch.arange(1, seqlen + 1, device=x.device) // ratio).unsqueeze(-1)
    elif seqlen == 1:
        compress_lens = end_pos // ratio
    else:
        # one lens per query: how many compressed blocks each absolute position can see
        compress_lens = ((torch.arange(seqlen, device=x.device) + start_pos + 1) // ratio).unsqueeze(0)
        compress_lens = compress_lens.expand(bsz, -1)

    import os as _os_idx

    _chunk = int(_os_idx.environ.get("DSV41_INDEXER_CHUNK", "0"))
    if _chunk > 0 and start_pos == 0 and seqlen > _chunk:
        # P4: full [S, 32, T] score is 26+ GiB at 14K and OOMs.  Stream over T with an exact
        # running top-k; candidate-source layers do a cheap block-max pass first.
        return _indexer_prefill_streaming(
            ix, q, index_k, weights, compress_lens.squeeze(-1), cfg, rt, offset,
            bsz, seqlen, ratio, _chunk
        )

    score = torch.einsum("bshd,btd->bsht", q, index_k)
    score = (score.relu() * weights.unsqueeze(-1)).sum(dim=2)

    if start_pos == 0:
        score.masked_fill_(
            torch.arange(seqlen // ratio, device=x.device) >= compress_lens, -math.inf
        )
    elif seqlen > 1:
        score.masked_fill_(
            torch.arange(score.size(-1), device=x.device) >= compress_lens.unsqueeze(-1),
            -math.inf,
        )

    if cfg.is_candidate_source(rt.layer_id):
        _cand_lens = compress_lens.unsqueeze(-1) if (start_pos > 0 and seqlen > 1) else compress_lens
        rt.shared.candidates = select_candidate_blocks(
            score, _cand_lens, cfg.candidate_topk_blocks, cfg.candidate_block_size
        )
    elif cfg.uses_candidates(rt.layer_id):
        candidates = rt.shared.candidates
        if candidates is not None and candidates.shape[0] == 1 and bsz != 1:
            candidates = candidates.expand(bsz, -1, -1)
        if candidates is not None:
            score = score.masked_fill(~candidates, -math.inf)

    topk = min(cfg.index_topk, end_pos // ratio)
    idxs = score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
    if seqlen > 1 and start_pos > 0:
        valid = idxs < compress_lens.unsqueeze(-1)
    else:
        valid = idxs < compress_lens
    return torch.where(valid, idxs + offset, -1).int()


def _indexer_block_scores(q, index_k, weights, t0: int, t1: int) -> torch.Tensor:
    """ ``[b, s, t1-t0]`` indexer scores for one T-block (same math as the full einsum)."""
    score = torch.einsum("bshd,btd->bsht", q, index_k[:, t0:t1])
    return (score.relu() * weights.unsqueeze(-1)).sum(dim=2)


def _select_from_block_max(block_max, compress_lens, topk_blocks: int, block_size: int):
    """`select_candidate_blocks` on already-computed per-block maxima (no per-token expand)."""
    scores = block_max
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(scores.size(-1), device=scores.device) == last, math.inf
    )
    top = scores.topk(min(int(topk_blocks), scores.size(-1)), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -math.inf
    )
    return keep.repeat_interleave(int(block_size), dim=-1)


def _indexer_prefill_streaming(
    ix, q, index_k, weights, compress_lens, cfg, rt, offset, bsz, seqlen, ratio, chunk
) -> torch.Tensor:
    """Exact blockwise top-k for long prefills (``DSV41_INDEXER_CHUNK``).

    ``compress_lens`` is ``[b, s]`` here.  Pass 1 builds per-query block maxima for the candidate
    source (needed by ``select_candidate_blocks``); pass 2 streams a running top-k over T.  Peak
    memory is O(b*s*chunk) instead of O(b*s*T).
    """
    device = q.device
    total_t = int(compress_lens.max().item()) if compress_lens.numel() else 0
    total_t = max(total_t, 1)
    cand_width = 0
    candidates = None

    if cfg.is_candidate_source(rt.layer_id):
        bs = int(cfg.candidate_block_size)
        nblocks = (total_t + bs - 1) // bs
        block_max = torch.full((bsz, seqlen, nblocks), -math.inf, dtype=torch.float32, device=device)
        step = max(bs, (chunk // bs) * bs)
        for t0 in range(0, total_t, step):
            t1 = min(total_t, t0 + step)
            s_blk = _indexer_block_scores(q, index_k, weights, t0, t1)
            ts = torch.arange(t0, t1, device=device)
            s_blk = s_blk.masked_fill(ts.unsqueeze(0).unsqueeze(0) >= compress_lens.unsqueeze(-1),
                                      -math.inf)
            pad = (-(t1 - t0)) % bs
            if pad:
                s_blk = torch.nn.functional.pad(s_blk, (0, pad), value=-math.inf)
            block_max[..., t0 // bs : t0 // bs + (t1 - t0 + pad) // bs] = s_blk.unflatten(
                -1, (-1, bs)
            ).amax(dim=-1)
        candidates = _select_from_block_max(
            block_max, compress_lens.unsqueeze(-1), cfg.candidate_topk_blocks, bs
        )
        rt.shared.candidates = candidates
        cand_width = candidates.shape[-1]
    elif cfg.uses_candidates(rt.layer_id):
        candidates = rt.shared.candidates

    topk = min(cfg.index_topk, total_t)
    top_v = torch.full((bsz, seqlen, topk), -math.inf, dtype=torch.float32, device=device)
    top_i = torch.full((bsz, seqlen, topk), -1, dtype=torch.int64, device=device)

    for t0 in range(0, total_t, chunk):
        t1 = min(total_t, t0 + chunk)
        s_blk = _indexer_block_scores(q, index_k, weights, t0, t1)
        ts = torch.arange(t0, t1, device=device)
        mask = ts.unsqueeze(0).unsqueeze(0) >= compress_lens.unsqueeze(-1)
        if candidates is not None:
            cm = candidates[..., t0:min(t1, cand_width)] if cand_width else None
            if cm is not None and cm.shape[-1] == (t1 - t0):
                mask = mask | (~cm)
        s_blk = s_blk.masked_fill(mask, -math.inf)

        cat_v = torch.cat([top_v, s_blk], dim=-1)
        cat_i = torch.cat(
            [top_i, ts.unsqueeze(0).unsqueeze(0).expand(bsz, seqlen, -1)], dim=-1
        )
        vals, sel = cat_v.topk(topk, dim=-1, sorted=False)
        top_v = vals
        top_i = cat_i.gather(-1, sel)

    valid = (top_i >= 0) & (top_i < compress_lens.unsqueeze(-1))
    idx = torch.where(valid, top_i + offset, -1).int()
    return idx.sort(dim=-1).values


# =============================================================================================
# Attention
# =============================================================================================


def attention_forward(
    attn: Dsv41Attention,
    x: torch.Tensor,
    start_pos: int,
    rt: LayerRuntime,
    cfg: Optional[Dsv41Config] = None,
    recorder: Optional[dict] = None,
) -> torch.Tensor:
    """``model.py:Attention.forward`` -- one layer, prefill or decode.

    ``recorder`` (debug only): a dict that receives a detached copy of each stage tensor --
    ``qr``/``q``/``kv``/``compress_kv``/``idxs``/``attn_out``/``x`` -- so a device run can be
    localized stage by stage against the CPU oracle (``tools/dsv41/s2_attn_debug.py``).
    """
    cfg = cfg or rt.cfg
    o = ops()
    bsz, seqlen, _ = x.shape
    cos = rt.cos[start_pos : start_pos + seqlen]
    sin = rt.sin[start_pos : start_pos + seqlen]
    rd = cfg.rope_head_dim

    def rec(name, tensor):
        if recorder is not None:
            recorder[name] = tensor.detach().clone()

    qr = rms_norm(attn.q_norm, o.linear(x, attn.wq_a.weight, attn.wq_a.scale))
    rec("qr", qr)
    q = o.linear(qr, attn.wq_b.weight, attn.wq_b.scale).unflatten(
        -1, (cfg.num_attention_heads, cfg.head_dim)
    )
    rotate(q[..., -rd:], cos, sin)
    rec("q", q)

    kv, topk_idxs = _window_kv(attn, x, (cos, sin), start_pos, rt, cfg, o)
    rec("window_kv", kv)
    ratio = cfg.compress_ratio(rt.layer_id)
    if ratio:
        if cfg.is_kv_source(rt.layer_id):
            # publish this segment's index_k even when the step yields no new key (ratio>1,
            # non-boundary): consumers and restored-prefix states need the buffer, not None.
            rt.shared.index_k = rt.index_k
        compress_kv, compress_idxs = _compress_kv(
            attn, x, qr, start_pos, kv.size(1), rt, cfg, o
        )
        rec("compress_kv", compress_kv)
        rec("compress_idxs", compress_idxs)
        kv = torch.cat([kv, compress_kv], dim=1)
        topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)
    rec("idxs", topk_idxs)

    out = o.sparse_attn(q, kv, attn.attn_sink, topk_idxs, cfg.head_dim**-0.5)
    rec("attn_out", out)
    rotate(out[..., -rd:], cos, sin, True)

    groups = cfg.o_groups
    out = out.view(bsz, seqlen, groups, -1)
    wo_a = attn.wo_a.weight.view(groups, cfg.o_lora_rank, -1)
    out = torch.einsum("bsgd,grd->bsgr", out, wo_a)
    final = o.linear(out.flatten(2), attn.wo_b.weight, attn.wo_b.scale)
    rec("x", final)
    return final


def compress_rope_positions(start_pos: int, ratio: int, n_latent: int) -> List[int]:
    """压缩 KV 的 ``n_latent`` 个新条目各自使用的 rope 位置。

    约定（与已验证的单 token 路径一致）：一个压缩条目代表 ``ratio`` 个原始 token 的块，
    其 rope 位置取**该块起始的绝对位置**。

    单 token 解码（``start_pos + 1`` 恰为块边界）时只有 1 个新条目，位置是
    ``start_pos + 1 - ratio`` —— 本函数在该情形下与老代码完全相同；
    分块解码 / 投机验证（一次 k 个 token）会产出多个条目，按 ``ratio`` 递增。

    纯整数运算，可离线单测（不需要 NPU）。
    """
    if n_latent <= 0:
        return []
    if start_pos == 0:
        # prefill：从 0 开始每 ratio 一个
        return [i * ratio for i in range(n_latent)]
    first = start_pos + 1 - ratio
    return [first + i * ratio for i in range(n_latent)]


def write_ring_chunk(ring: torch.Tensor, start_pos: int, kv: torch.Tensor, window: int) -> None:
    """把 ``kv`` 的 ``seqlen`` 个位置按绝对位置连续写进环形缓冲（`pos % window`）。

    ``ring`` 形状 ``[bsz, window, dim]``，``kv`` 形状 ``[bsz, seqlen, dim]``。
    这是分块解码 / 投机验证的写入原语；``seqlen == 1`` 时等价于老代码的
    ``ring[:bsz, start_pos % window] = kv.squeeze(1)``。

    被拒绝的草稿 token **不需要回滚**：下一次写入按新的 ``start_pos`` 覆盖即可。
    """
    seqlen = int(kv.shape[1])
    if seqlen <= 0:
        return
    dev = kv.device
    positions = (torch.arange(seqlen, device=dev) + int(start_pos)) % int(window)
    ring[: kv.shape[0]].index_copy_(1, positions, kv.to(ring.dtype))


def write_ring_batched(
    batched_kv: torch.Tensor,
    slots,
    positions,
    kv: torch.Tensor,
    window: int,
) -> None:
    """批处理散写：把 R 个请求的当前 KV 写进**共享**的 `[R_max, window, dim]` 缓冲。

    参数：

    * ``batched_kv``：``[R_max, window, dim]``，第 i 行属于槽位 i 的请求；
    * ``slots``：``[R]`` 的长整型槽位号；
    * ``positions``：``[R]`` 的绝对位置（每个请求各自不同）；
    * ``kv``：``[R, dim]``（decode，每请求 1 个 token）。

    这是 `perf-plan.md` §24 的 A 步原语。等价于对每个请求单独执行
    ``state.window_kv[0, pos % window] = kv[i]``，但与逐请求循环不同，它是一次索引写入。

    被拒绝的草稿 token 同样无需回滚：下一次写入按新的 ``positions`` 覆盖。
    """
    dev = batched_kv.device
    slot_idx = torch.as_tensor(list(slots), dtype=torch.long, device=dev)
    pos_idx = (torch.as_tensor(list(positions), dtype=torch.long, device=dev) % int(window))
    batched_kv[slot_idx, pos_idx] = kv.to(batched_kv.dtype)


def _window_kv(attn, x, rope, start_pos, rt, cfg, o):
    cos, sin = rope
    bsz, seqlen, _ = x.shape
    win = cfg.sliding_window
    kv = rms_norm(attn.kv_norm, o.linear(x, attn.wkv.weight, attn.wkv.scale))
    rotate(kv[..., -cfg.rope_head_dim :], cos, sin)
    o.act_quant(kv, 32, "ue8m0", _E8M0, True)
    if start_pos == 0:
        if seqlen <= win:
            rt.window_kv[:bsz, :seqlen] = kv
        else:
            cutoff = seqlen % win
            head, tail = kv[:, -win:].split([win - cutoff, cutoff], dim=1)
            rt.window_kv[:bsz, cutoff:win] = head
            rt.window_kv[:bsz, :cutoff] = tail
        window_kv = kv
    else:
        if seqlen == 1:
            rt.window_kv[:bsz, start_pos % win] = kv.squeeze(1)
        else:
            # 分块解码 / 投机验证：k 个 token 依次写进连续的环形槽位。
            # 被拒绝的草稿 token 无需回滚——下一次写入会按新的 start_pos 覆盖。
            write_ring_chunk(rt.window_kv, start_pos, kv, win)
        window_kv = rt.window_kv[:bsz]
    return window_kv, window_topk_idxs(win, bsz, seqlen, start_pos, device=x.device)


def _compress_kv(attn, x, qr, start_pos, offset, rt, cfg, o):
    bsz, seqlen, _ = x.shape
    ratio = cfg.compress_ratio(rt.layer_id)
    compress_len = (start_pos + seqlen) // ratio
    latent = None
    if cfg.is_kv_source(rt.layer_id):
        latent = compressor_forward(attn.compressor, x, start_pos, rt)
        rt.shared.compress_kv = rt.compress_kv

    idxs = _compress_topk_idxs(attn, x, qr, latent, start_pos, offset, compress_len, rt, cfg, o)

    if latent is not None:
        n_lat = int(latent.size(1))
        if start_pos == 0:
            c = rt.cos[: seqlen - seqlen % ratio : ratio]
            s = rt.sin[: seqlen - seqlen % ratio : ratio]
        elif n_lat == 1:
            # 已验证的单 token 路径：逐位保持
            c = rt.cos[start_pos + 1 - ratio].unsqueeze(0)
            s = rt.sin[start_pos + 1 - ratio].unsqueeze(0)
        else:
            # 分块 prefill：chunk 与 ratio 对齐时，latent 的 rope 位置就是块起点；
            # 非对齐（投机验证）沿用 decode 的“块末尾”约定。
            if start_pos % ratio == 0:
                pos = torch.arange(n_lat, device=rt.cos.device) * ratio + start_pos
            else:
                pos = torch.tensor(
                    compress_rope_positions(start_pos, ratio, n_lat), device=rt.cos.device
                ).clamp_min(0)
            c = rt.cos[pos]
            s = rt.sin[pos]
        rotate(latent[..., -cfg.rope_head_dim :], c, s)
        o.fp4_act_quant(latent, 16, True, scale_dtype=_FP8)
        rt.compress_kv[:bsz, start_pos // ratio : start_pos // ratio + latent.size(1)] = latent
    return rt.shared.compress_kv[:bsz, :compress_len], idxs


def _compress_topk_idxs(attn, x, qr, latent, start_pos, offset, compress_len, rt, cfg, o):
    bsz, seqlen, _ = x.shape
    if not cfg.is_index_source(rt.layer_id):
        return rt.shared.topk_idxs
    if compress_len == 0:
        idxs = torch.empty(bsz, seqlen, 0, dtype=torch.int32, device=x.device)
    else:
        idxs = indexer_forward(attn.indexer, x, qr, latent, start_pos, offset, rt, cfg)
    rt.shared.topk_idxs = idxs
    return idxs
