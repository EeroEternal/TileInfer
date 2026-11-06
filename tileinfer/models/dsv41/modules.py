"""``tileinfer.models.dsv41.modules`` — the parameter tree of DeepSeek-V4.1-Flash.

Design rule: **``named_parameters()`` is the checkpoint's name set.**  Module attribute names spell
exactly what the HF shards contain, so the loader is a name-for-name copy with no mapping table:

    embed.weight                      head.weight                       norm.weight
    layers.{i}.attn_norm.weight       layers.{i}.ffn_norm.weight
    layers.{i}.hc_attn_{fn,base,scale} / hc_ffn_{fn,base,scale}
    layers.{i}.attn.attn_sink         layers.{i}.attn.{wq_a,q_norm,wq_b,wkv,kv_norm,wo_a,wo_b}
    layers.{i}.attn.compressor.{wkv,wgate,norm}          (kv-source layers)
    layers.{i}.attn.indexer.{wq_b,weights_proj,wk,k_norm} (index-source layers)
    layers.{i}.ffn.gate.{weight,bias,bias_vl}
    layers.{i}.ffn.experts.{e}.{w1,w2,w3}.{weight,scale}
    layers.{i}.ffn.shared_experts.{w1,w2,w3}.{weight,scale}
    layers.{i}.engram.{embed,wkv,q_weight,k_weight}      (engram layers)
    mtp.{s}.* (same as a layer) + mtp.0.{main_proj,main_norm} + mtp.2.{norm,markov_head,confidence_head}
    vision.* / aligner.* / image_{start,end,newline}

Every tensor is stored in **the checkpoint's own dtype** (fp8 e4m3 + e8m0 block scale, packed fp4,
bf16, fp32) — no dequantization at load time except ``wo_a``, which the reference converter also
folds to bf16 (``docs/notes/dsv41-facts.md`` §3.4).  Forward passes are deliberately not here yet:
S1 is the load milestone (``docs/notes/dsv41-facts.md`` §18), S2/S3 add the math.
"""

from __future__ import annotations

import math
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch import nn

from .config import FP4_BLOCK, FP8_BLOCK, Dsv41Config

__all__ = [
    "Dsv41Slot",
    "Dsv41RMSNorm",
    "Dsv41Expert",
    "Dsv41MoE",
    "Dsv41Gate",
    "Dsv41Compressor",
    "Dsv41Indexer",
    "Dsv41Attention",
    "Dsv41Engram",
    "Dsv41Block",
    "Dsv41DSparkBlock",
    "Dsv41Model",
    "Dsv41ViT",
    "Dsv41Aligner",
    "storage_dtype",
    "NotImplementedYet",
]

FP8 = torch.float8_e4m3fn
FP4 = torch.float4_e2m1fn_x2
E8M0 = torch.float8_e8m0fnu


def storage_dtype(storage: str) -> torch.dtype:
    if storage == "fp8":
        return FP8
    if storage == "fp4":
        return FP4
    if storage == "bf16":
        return torch.bfloat16
    if storage == "fp32":
        return torch.float32
    raise ValueError(f"unknown storage {storage!r} (expected fp8/fp4/bf16/fp32)")


class NotImplementedYet(RuntimeError):
    """Raised by symbols that belong to a path which is not wired up (today: the vLLM runner)."""


def _unimplemented(what: str, hint: str = "") -> "NotImplementedYet":
    """Build the error for an unwired symbol.

    ``hint`` is required reading: the first version of this helper carried a fixed "S1 does the tree,
    the maths lands in S2/S3" sentence, which was true for two rounds and misleading for eighty after
    that.  Callers now say what is missing *and* where the working path is.
    """
    base = f"{what} is not implemented yet"
    if hint:
        return NotImplementedYet(f"{base}: {hint}")
    return NotImplementedYet(
        f"{base}: the model runs today through tileinfer.models.dsv41.block (see "
        "docs/notes/dsv41-status.md); this symbol belongs to a path that is still open."
    )


class Dsv41Slot(nn.Module):
    """One checkpoint entry group: ``weight`` [+ ``scale``] [+ ``bias``], in the stored dtype.

    ``out_features``/``in_features`` are *logical* (the checkpoint's ``[out, in]`` convention, i.e.
    ``nn.Linear``'s weight orientation).  For ``fp4`` the weight is allocated packed along K
    (``[out, in // 2]`` in ``float4_e2m1fn_x2``, two values per byte) with one e8m0 scale per 32
    elements of K — which is exactly what the shards contain (``I8 [out, in//2]`` viewed as fp4).
    """

    def __init__(
        self,
        out_features: int,
        in_features: int,
        storage: str = "bf16",
        bias: bool = False,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.out_features = int(out_features)
        self.in_features = int(in_features)
        self.storage = storage
        self.device_name = device

        if storage == "fp4":
            packed_in = self.in_features // 2
            if self.in_features % FP4_BLOCK:
                raise ValueError(
                    f"fp4 needs in_features % {FP4_BLOCK} == 0, got {self.in_features}"
                )
            self.weight = nn.Parameter(
                torch.empty(self.out_features, packed_in, dtype=FP4, device=device)
            )
            self.scale = nn.Parameter(
                torch.empty(
                    self.out_features,
                    self.in_features // FP4_BLOCK,
                    dtype=E8M0,
                    device=device,
                )
            )
        elif storage == "fp8":
            self.weight = nn.Parameter(
                torch.empty(self.out_features, self.in_features, dtype=FP8, device=device)
            )
            self.scale = nn.Parameter(
                torch.empty(
                    -(-self.out_features // FP8_BLOCK),
                    -(-self.in_features // FP8_BLOCK),
                    dtype=E8M0,
                    device=device,
                )
            )
        else:
            self.weight = nn.Parameter(
                torch.empty(
                    self.out_features, self.in_features, dtype=storage_dtype(storage), device=device
                )
            )
            self.register_parameter("scale", None)

        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features, dtype=torch.bfloat16, device=device)
            )
        else:
            self.register_parameter("bias", None)

    def extra_repr(self) -> str:
        return (
            f"{self.in_features}->{self.out_features}, storage={self.storage}, "
            f"bias={self.bias is not None}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - S2/S3
        raise _unimplemented(f"{type(self).__name__}.forward")


class Dsv41RMSNorm(nn.Module):
    """RMSNorm with the checkpoint's dtype for ``weight`` (eps is 1e-20 on this model)."""

    def __init__(self, dim: int, eps: float = 1e-20, dtype: torch.dtype = torch.bfloat16,
                 device: str = "cpu") -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.dim, dtype=dtype, device=device))

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - S2/S3
        raise _unimplemented(f"{type(self).__name__}.forward")


class Dsv41Expert(nn.Module):
    """One SwiGLU expert (``w1`` gate, ``w3`` up, ``w2`` down); fp4 unless the checkpoint is fp8."""

    def __init__(self, cfg: Dsv41Config, storage: str = "fp4", device: str = "cpu") -> None:
        super().__init__()
        inter, dim = cfg.moe_intermediate_size, cfg.hidden_size
        self.w1 = Dsv41Slot(inter, dim, storage, device=device)
        self.w2 = Dsv41Slot(dim, inter, storage, device=device)
        self.w3 = Dsv41Slot(inter, dim, storage, device=device)
        self.swiglu_limit = float(cfg.swiglu_limit)


class Dsv41Gate(nn.Module):
    """MoE router: bf16 selection scores, fp32 correction bias (plus the VL variant)."""

    def __init__(self, cfg: Dsv41Config, n_routed: int, device: str = "cpu") -> None:
        super().__init__()
        self.n_routed_experts = int(n_routed)
        self.dim = cfg.hidden_size
        self.topk = None  # the owning MoE sets this (routed vs DSpark counts differ)
        self.score_func = cfg.scoring_func
        self.gate_temp = 1.0
        self.norm_topk_prob = True
        self.route_scale = float(cfg.routed_scaling_factor)
        self.weight = nn.Parameter(
            torch.empty(n_routed, cfg.hidden_size, dtype=torch.bfloat16, device=device)
        )
        self.bias = nn.Parameter(torch.empty(n_routed, dtype=torch.float32, device=device))
        self.bias_vl = nn.Parameter(torch.empty(n_routed, dtype=torch.float32, device=device))


class Dsv41MoE(nn.Module):
    """Routed experts + one always-on shared expert (fp8 e4m3 on this checkpoint)."""

    def __init__(self, cfg: Dsv41Config, layer_id: int, device: str = "cpu") -> None:
        super().__init__()
        n_routed, n_activated = cfg.moe_config(layer_id)
        self.layer_id = int(layer_id)
        self.n_routed_experts = int(n_routed)
        self.n_activated_experts = int(n_activated)
        self.gate = Dsv41Gate(cfg, n_routed, device=device)
        self.gate.topk = int(n_activated)
        expert_storage = "fp4" if cfg.expert_dtype == "fp4" else "fp8"
        self.experts = nn.ModuleList(
            [Dsv41Expert(cfg, expert_storage, device=device) for _ in range(n_routed)]
        )
        self.shared_experts = Dsv41Expert(cfg, "fp8", device=device)


class Dsv41Compressor(nn.Module):
    """Pools ``compress_ratio`` tokens into one 512-d KV latent (pre-RoPE).

    ``ratio == 1`` (layer 20) is a plain bf16 projection with no gate; ``ratio > 1`` pools with a
    learned softmax gate in fp32 (facts §6).  The checkpoint stores both in bf16; the reference
    runtime upcasts the ratio>1 pair to fp32 at load, which is a value-preserving cast, so this tree
    keeps bf16 and casts on use.
    """

    def __init__(self, cfg: Dsv41Config, layer_id: int, device: str = "cpu") -> None:
        super().__init__()
        ratio = cfg.compress_ratio(layer_id)
        self.compress_ratio = int(ratio)
        self.norm = Dsv41RMSNorm(cfg.head_dim, cfg.rms_norm_eps, device=device)
        self.wkv = Dsv41Slot(cfg.head_dim, cfg.hidden_size, "bf16", device=device)
        if ratio > 1:
            self.wgate = Dsv41Slot(cfg.head_dim, cfg.hidden_size, "bf16", device=device)
        else:
            self.wgate = None


class Dsv41Indexer(nn.Module):
    """Side attention that keeps the ``index_topk`` best compressed positions (facts §7)."""

    def __init__(self, cfg: Dsv41Config, layer_id: int, device: str = "cpu") -> None:
        super().__init__()
        self.layer_id = int(layer_id)
        self.owns_k = cfg.owns_index_k(layer_id)
        self.wq_b = Dsv41Slot(
            cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank, "fp8", device=device
        )
        self.weights_proj = Dsv41Slot(cfg.index_n_heads, cfg.hidden_size, "bf16", device=device)
        if self.owns_k:
            self.wk = Dsv41Slot(cfg.index_head_dim, cfg.head_dim, "bf16", device=device)
            self.k_norm = Dsv41RMSNorm(cfg.index_head_dim, cfg.rms_norm_eps, device=device)
        else:
            self.wk = None
            self.k_norm = None


class Dsv41Attention(nn.Module):
    """Latent attention with a sliding window plus an optional compressed stream (facts §5)."""

    def __init__(self, cfg: Dsv41Config, layer_id: int, device: str = "cpu") -> None:
        super().__init__()
        self.layer_id = int(layer_id)
        dim, heads, head_dim = cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim
        self.attn_sink = nn.Parameter(
            torch.empty(heads, dtype=torch.float32, device=device)
        )
        self.wq_a = Dsv41Slot(cfg.q_lora_rank, dim, "fp8", device=device)
        self.q_norm = Dsv41RMSNorm(cfg.q_lora_rank, cfg.rms_norm_eps, device=device)
        self.wq_b = Dsv41Slot(heads * head_dim, cfg.q_lora_rank, "fp8", device=device)
        self.wkv = Dsv41Slot(head_dim, dim, "fp8", device=device)
        self.kv_norm = Dsv41RMSNorm(head_dim, cfg.rms_norm_eps, device=device)
        # grouped output projection: [o_groups * o_lora_rank, heads * head_dim / o_groups], bf16
        self.wo_a = Dsv41Slot(
            cfg.o_groups * cfg.o_lora_rank, heads * head_dim // cfg.o_groups, "bf16", device=device
        )
        self.wo_b = Dsv41Slot(dim, cfg.o_groups * cfg.o_lora_rank, "fp8", device=device)
        if cfg.is_kv_source(layer_id):
            self.compressor: Optional[Dsv41Compressor] = Dsv41Compressor(cfg, layer_id, device=device)
        else:
            self.compressor = None
        if cfg.is_index_source(layer_id):
            self.indexer: Optional[Dsv41Indexer] = Dsv41Indexer(cfg, layer_id, device=device)
        else:
            self.indexer = None


class Dsv41Engram(nn.Module):
    """One engram table (fp8 rows + e8m0 row-block scales) and its key/value projection.

    ``embed.weight`` is ``[rows, 256]`` fp8 and ``embed.scale`` ``[rows, 8]`` (one e8m0 per 32 of the
    256-wide row), so a lookup dequantizes a 256-d row on the fly (facts §10.4).
    """

    def __init__(self, cfg: Dsv41Config, layer_id: int, device: str = "cpu") -> None:
        super().__init__()
        self.layer_id = int(layer_id)
        self.layer_hash_index = tuple(int(x) for x in cfg.engram_layer_ids).index(self.layer_id)
        self.eps = float(cfg.rms_norm_eps)
        self.clamp_value = 1e-6
        rows = cfg.engram_rows(self.layer_id)
        self.embed = _EngramTable(rows, cfg.engram_head_dim, device=device)
        self.wkv = Dsv41Slot(cfg.engram_wkv_out, cfg.engram_wkv_in, "fp8", device=device)
        self.q_weight = nn.Parameter(
            torch.empty(cfg.hc_mult, cfg.hidden_size, dtype=torch.bfloat16, device=device)
        )
        self.k_weight = nn.Parameter(
            torch.empty(cfg.hc_mult, cfg.hidden_size, dtype=torch.bfloat16, device=device)
        )


class _EngramTable(nn.Module):
    """``embed.weight`` (fp8) + ``embed.scale`` (e8m0, one per 32 columns)."""

    def __init__(self, rows: int, dim: int, device: str = "cpu") -> None:
        super().__init__()
        self.rows = int(rows)
        self.dim = int(dim)
        self.weight = nn.Parameter(torch.empty(rows, dim, dtype=FP8, device=device))
        self.scale = nn.Parameter(
            torch.empty(rows, dim // FP8_BLOCK, dtype=E8M0, device=device)
        )


class Dsv41Block(nn.Module):
    """One decoder block: Hyper-Connections mix matrices + attention + MoE (+ optional engram)."""

    def __init__(self, cfg: Dsv41Config, layer_id: int, device: str = "cpu") -> None:
        super().__init__()
        self.layer_id = int(layer_id)
        self.norm_eps = float(cfg.rms_norm_eps)
        self.hc_mult = int(cfg.hc_mult)
        self.hc_sinkhorn_iters = int(cfg.hc_sinkhorn_iters)
        self.hc_eps = float(cfg.hc_eps)
        self.attn = Dsv41Attention(cfg, layer_id, device=device)
        self.ffn = Dsv41MoE(cfg, layer_id, device=device)
        self.attn_norm = Dsv41RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, device=device)
        self.ffn_norm = Dsv41RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, device=device)
        mix, hc_dim = cfg.mix_hc, cfg.hc_dim
        for name in ("hc_attn_fn", "hc_ffn_fn"):
            self.register_parameter(
                name,
                nn.Parameter(torch.empty(mix, hc_dim, dtype=torch.float32, device=device)),
            )
        for name in ("hc_attn_base", "hc_ffn_base"):
            self.register_parameter(
                name, nn.Parameter(torch.empty(mix, dtype=torch.float32, device=device))
            )
        for name in ("hc_attn_scale", "hc_ffn_scale"):
            self.register_parameter(
                name, nn.Parameter(torch.empty(3, dtype=torch.float32, device=device))
            )
        if cfg.has_engram(layer_id):
            self.engram: Optional[Dsv41Engram] = Dsv41Engram(cfg, layer_id, device=device)
        else:
            self.engram = None


class Dsv41MarkovHead(nn.Module):
    """``markov_head.{embed,head}`` — a token-id n-gram bias over the draft block (facts §11.3)."""

    def __init__(self, cfg: Dsv41Config, device: str = "cpu") -> None:
        super().__init__()
        rank = cfg.dspark_markov_rank
        self.embed = Dsv41Slot(cfg.vocab_size, rank, "bf16", device=device)
        self.head = Dsv41Slot(cfg.vocab_size, rank, "bf16", device=device)


class Dsv41ConfidenceHead(nn.Module):
    """``confidence_head.proj``: ``[1, hidden + markov_rank]`` (fp32 in the reference)."""

    def __init__(self, cfg: Dsv41Config, device: str = "cpu") -> None:
        super().__init__()
        self.proj = Dsv41Slot(
            1, cfg.hidden_size + cfg.dspark_markov_rank, "bf16", device=device
        )


class Dsv41DSparkBlock(Dsv41Block):
    """``mtp.{s}``: a DSpark draft stage (facts §11).

    Stage 0 folds the three target layers' hidden states in (``main_proj``); the last stage owns the
    vocab projection (``norm`` + ``markov_head`` + ``confidence_head``).  All stages reuse the
    backbone's ``embed``/``head`` — those parameters are *not* in the checkpoint, so they are not
    created here either (the backbone's are shared by reference at runtime).
    """

    def __init__(self, cfg: Dsv41Config, stage_id: int, device: str = "cpu") -> None:
        super().__init__(cfg, cfg.n_layers + stage_id, device=device)
        self.stage_id = int(stage_id)
        if stage_id == 0:
            self.main_proj = Dsv41Slot(
                cfg.hidden_size, cfg.hidden_size * len(cfg.dspark_target_layer_ids), "fp8",
                device=device,
            )
            self.main_norm = Dsv41RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, device=device)
        else:
            self.main_proj = None
            self.main_norm = None
        if stage_id == cfg.n_mtp_layers - 1:
            self.norm = Dsv41RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, device=device)
            self.markov_head = Dsv41MarkovHead(cfg, device=device)
            self.confidence_head = Dsv41ConfidenceHead(cfg, device=device)
        else:
            self.norm = None
            self.markov_head = None
            self.confidence_head = None


# =============================================================================================
# Vision tower
# =============================================================================================


class _ViTPatchEmbed(nn.Module):
    def __init__(self, cfg: Dsv41Config, device: str = "cpu") -> None:
        super().__init__()
        self.proj = Dsv41Slot(
            cfg.vision_hidden_size, 3 * cfg.vision_patch_size**2, "bf16", bias=True, device=device
        )


class _ViTAttention(nn.Module):
    def __init__(self, cfg: Dsv41Config, device: str = "cpu") -> None:
        super().__init__()
        dim = cfg.vision_hidden_size
        self.wqkv = Dsv41Slot(3 * dim, dim, "bf16", bias=True, device=device)
        self.wo = Dsv41Slot(dim, dim, "bf16", bias=True, device=device)


class _ViTMLP(nn.Module):
    def __init__(self, cfg: Dsv41Config, device: str = "cpu") -> None:
        super().__init__()
        dim, inter = cfg.vision_hidden_size, cfg.vision_intermediate_size
        self.w1 = Dsv41Slot(2 * inter, dim, "bf16", device=device)
        self.w2 = Dsv41Slot(dim, inter, "bf16", device=device)


class _ViTBlock(nn.Module):
    def __init__(self, cfg: Dsv41Config, device: str = "cpu") -> None:
        super().__init__()
        dim = cfg.vision_hidden_size
        self.norm1 = Dsv41RMSNorm(dim, 1e-6, device=device)
        self.attn = _ViTAttention(cfg, device=device)
        self.norm2 = Dsv41RMSNorm(dim, 1e-6, device=device)
        self.mlp = _ViTMLP(cfg, device=device)


class Dsv41ViT(nn.Module):
    """Bidirectional ViT over one image's patch grid with 2-D RoPE (facts §12).

    The reference holds the ViT RMSNorm weights in fp32; this tree keeps the checkpoint's bf16 for
    every parameter (the uniformity the loader relies on) and casts on use — a value-preserving cast,
    since bf16 -> fp32 is exact.
    """

    def __init__(self, cfg: Dsv41Config, device: str = "cpu") -> None:
        super().__init__()
        self.patch_embed = _ViTPatchEmbed(cfg, device=device)
        self.blocks = nn.ModuleList(
            [_ViTBlock(cfg, device=device) for _ in range(cfg.vision_num_hidden_layers)]
        )
        self.norm = Dsv41RMSNorm(cfg.vision_hidden_size, 1e-6, device=device)


class Dsv41Aligner(nn.Module):
    """``downsample_ratio**2 * vision_dim -> hidden`` then ``hidden -> hidden`` (both with bias)."""

    def __init__(self, cfg: Dsv41Config, device: str = "cpu") -> None:
        super().__init__()
        in_dim = cfg.vision_hidden_size * cfg.vision_downsample_ratio**2
        self.w1 = Dsv41Slot(cfg.hidden_size, in_dim, "bf16", bias=True, device=device)
        self.w2 = Dsv41Slot(cfg.hidden_size, cfg.hidden_size, "bf16", bias=True, device=device)


# =============================================================================================
# The whole model
# =============================================================================================


class Dsv41Model(nn.Module):
    """Backbone (40 layers) + DSpark stages (3) + vision tower, with the checkpoint's names.

    ``with_vision`` / ``with_mtp`` let a caller load only what it needs (S3 text-only can skip both,
    saving 8 GiB and 7.4 GiB respectively); the loader reports what it skipped.

    ``offload_host`` is the S4 switch: when true, expert and engram parameters are tagged (via
    :attr:`host_offload_prefixes`) as "resident in host memory", which is where they must live on a
    128 GiB card holding a 476 GB model (facts §17).  Tagging only — the placement policy is S4.
    """

    #: Parameter prefixes that the S4 offload design keeps in host memory.
    HOST_OFFLOAD_PATTERNS: Tuple[str, ...] = (".ffn.experts.", ".engram.embed.")

    def __init__(
        self,
        config: Dsv41Config,
        device: str = "cpu",
        with_vision: bool = True,
        with_mtp: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.device_name = device
        self.embed = Dsv41Slot(config.vocab_size, config.hidden_size, "bf16", device=device)
        self.layers = nn.ModuleList(
            [Dsv41Block(config, i, device=device) for i in range(config.n_layers)]
        )
        self.norm = Dsv41RMSNorm(config.hidden_size, config.rms_norm_eps, device=device)
        self.head = Dsv41Slot(config.vocab_size, config.hidden_size, "bf16", device=device)
        if with_mtp and config.n_mtp_layers:
            self.mtp = nn.ModuleList(
                [Dsv41DSparkBlock(config, s, device=device) for s in range(config.n_mtp_layers)]
            )
        else:
            self.mtp = nn.ModuleList()
        if with_vision and config.vision_num_hidden_layers:
            self.vision: Optional[Dsv41ViT] = Dsv41ViT(config, device=device)
            self.aligner: Optional[Dsv41Aligner] = Dsv41Aligner(config, device=device)
            self.image_start = nn.Parameter(
                torch.empty(config.hidden_size, dtype=torch.bfloat16, device=device)
            )
            self.image_end = nn.Parameter(
                torch.empty(config.hidden_size, dtype=torch.bfloat16, device=device)
            )
            self.image_newline = nn.Parameter(
                torch.empty(config.hidden_size, dtype=torch.bfloat16, device=device)
            )
        else:
            self.vision = None
            self.aligner = None
            self.register_parameter("image_start", None)
            self.register_parameter("image_end", None)
            self.register_parameter("image_newline", None)

        # Inference-only tree: no autograd, which also lets the loader copy straight into the
        # parameters (an in-place op on a leaf that requires grad is an error).
        for param in self.parameters():
            param.requires_grad_(False)

    # -- introspection -----------------------------------------------------------------------

    def host_offload_names(self) -> List[str]:
        return [
            name
            for name, _ in self.named_parameters()
            if any(pat in name for pat in self.HOST_OFFLOAD_PATTERNS)
        ]

    def parameter_report(self) -> List[Dict[str, object]]:
        """``[{name, shape, dtype, nbytes}]`` — the table the loader verifies against."""
        rows: List[Dict[str, object]] = []
        for name, param in self.named_parameters():
            rows.append(
                {
                    "name": name,
                    "shape": tuple(int(n) for n in param.shape),
                    "dtype": str(param.dtype).replace("torch.", ""),
                    "nbytes": int(param.numel() * param.element_size()),
                }
            )
        return rows

    def n_parameters_bytes(self) -> int:
        return sum(int(p.numel() * p.element_size()) for p in self.parameters())

    def forward(self, *args, **kwargs):  # pragma: no cover - S2/S3
        raise _unimplemented(f"{type(self).__name__}.forward")
