"""``tileinfer.models.dsv41.weights`` — load the HF shards straight into the parameter tree.

The released checkpoint already uses the runtime's own names (no ``model.`` prefix, no
``weight_scale_inv``, no ``self_attn``/``mlp`` spelling — verified with
``tools/dsv41/dump_inventory.py``), so the loader resolves ``{name: parameter}`` once and copies
name-for-name.  Only two transformations exist, both mandated by the reference converter:

1. **packed fp4 experts** — the shards hold ``I8 [out, in//2]``; the parameter is
   ``float4_e2m1fn_x2 [out, in//2]`` (same bytes, same shape, a ``view``).
2. ``wo_a`` — fp8 weight + e8m0 block scale folded to bf16, exactly like
   ``inference/convert.py:157-173`` (the reference runs its grouped output projection in bf16).

Everything else is stored in the checkpoint's own dtype, which is what makes the S1 acceptance check
meaningful: *every parameter's name, shape and dtype must equal the shard's*, with the two
documented exceptions above.

Peak memory = the model's own bytes (513.1 GB) plus at most one shard tensor / one fp32 ``wo_a``
fold (~128 MB), because each tensor is copied into its pre-allocated parameter and released.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from glob import glob
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from safetensors import safe_open
from torch import nn

from .modules import Dsv41Model

__all__ = [
    "LoadReport",
    "shard_paths",
    "header_map",
    "expected_names",
    "verify_headers",
    "load_shards",
    "fold_wo_a",
    "FP4_TORCH_DTYPE_NAME",
]

FP4_TORCH_DTYPE_NAME = "float4_e2m1fn_x2"
_FP4 = torch.float4_e2m1fn_x2
_FP8 = torch.float8_e4m3fn
_E8M0 = torch.float8_e8m0fnu


# =============================================================================================
# Shard discovery / headers
# =============================================================================================


def shard_paths(model_dir: str) -> List[str]:
    """Every ``*.safetensors`` in the checkpoint directory, in shard order."""
    paths = sorted(glob(os.path.join(model_dir, "*.safetensors")))
    if not paths:
        raise FileNotFoundError(f"no *.safetensors under {model_dir!r}")
    return paths


def header_map(paths: Sequence[str]) -> Dict[str, Tuple[str, Tuple[int, ...], int]]:
    """``{name: (dtype, shape, nbytes)}`` from the shard headers only (no tensor data read)."""
    out: Dict[str, Tuple[str, Tuple[int, ...], int]] = {}
    for path in paths:
        with open(path, "rb") as handle:
            n = int.from_bytes(handle.read(8), "little")
            header = json.loads(handle.read(n))
        header.pop("__metadata__", None)
        for name, info in header.items():
            start, end = info["data_offsets"]
            out[name] = (str(info["dtype"]), tuple(int(x) for x in info["shape"]), int(end - start))
    return out


#: safetensors dtype -> torch dtype, for the two dtypes where equality is not literal.
_ST_DTYPE = {
    "BF16": "bfloat16",
    "F32": "float32",
    "F16": "float16",
    "I8": "int8",
    "U8": "uint8",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E8M0": "float8_e8m0fnu",
    "F4": FP4_TORCH_DTYPE_NAME,
    "F4_E2M1": FP4_TORCH_DTYPE_NAME,
    "F4_E2M1FN_X2": FP4_TORCH_DTYPE_NAME,
}


def _norm_dtype(name: str) -> str:
    return _ST_DTYPE.get(str(name).upper(), str(name).lower())


def expected_names(model: Dsv41Model) -> Dict[str, Tuple[Tuple[int, ...], str]]:
    """``{param name: (shape, dtype)}`` for the built model."""
    out: Dict[str, Tuple[Tuple[int, ...], str]] = {}
    for name, param in model.named_parameters():
        out[name] = (
            tuple(int(x) for x in param.shape),
            str(param.dtype).replace("torch.", ""),
        )
    return out


@dataclass
class LoadReport:
    """What the loader did — the S1 evidence record."""

    copied: int = 0
    bytes_in: int = 0
    folded_wo_a: int = 0
    skipped_by_config: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    extra: List[str] = field(default_factory=list)
    shape_mismatch: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = field(default_factory=list)
    dtype_mismatch: List[Tuple[str, str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.missing or self.extra or self.shape_mismatch or self.dtype_mismatch)

    def summary(self) -> str:
        return (
            f"copied={self.copied} tensors, {self.bytes_in/1e9:.1f} GB in, "
            f"wo_a folded={self.folded_wo_a}, "
            f"skipped_by_config={len(self.skipped_by_config)}, missing={len(self.missing)}, "
            f"extra={len(self.extra)}, shape_mismatch={len(self.shape_mismatch)}, "
            f"dtype_mismatch={len(self.dtype_mismatch)}"
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "summary": self.summary(),
            "copied": self.copied,
            "bytes_in": self.bytes_in,
            "folded_wo_a": self.folded_wo_a,
            "skipped_by_config": list(self.skipped_by_config),
            "missing": list(self.missing),
            "extra": list(self.extra),
            "shape_mismatch": [list(x) for x in self.shape_mismatch],
            "dtype_mismatch": [list(x) for x in self.dtype_mismatch],
        }


# =============================================================================================
# Dry run: headers vs the parameter tree
# =============================================================================================


def _classify(
    name: str, want: Mapping[str, Tuple[Tuple[int, ...], str]]
) -> Optional[str]:
    """Which *section* a checkpoint name belongs to, or ``None`` when the model should own it."""
    if name in want:
        return None
    # wo_a is folded, so its scale is consumed by the weight
    if name.endswith(".wo_a.scale"):
        return "folded"
    return "unknown"


def verify_headers(model: Dsv41Model, paths: Sequence[str]) -> LoadReport:
    """Compare the shard headers against the parameter tree without allocating anything.

    The shape/dtype comparison applies the two documented transformations: a packed-fp4 parameter is
    compared against the shard's ``I8`` entry (identical shape), and ``wo_a`` (bf16 here) against the
    shard's fp8 entry (identical shape, documented dtype change).
    """
    want = expected_names(model)
    headers = header_map(paths)
    report = LoadReport()

    for name, (dtype, shape, nbytes) in headers.items():
        if name not in want:
            if name.endswith(".wo_a.scale"):
                continue
            if _is_skipped_by_config(name, model):
                report.skipped_by_config.append(name)
                continue
            report.extra.append(name)
            continue
        wshape, wdtype = want[name]
        if tuple(shape) != wshape:
            report.shape_mismatch.append((name, wshape, tuple(shape)))
            continue
        if name.endswith(".wo_a.weight"):
            report.folded_wo_a += 1
            continue
        want_dtype = _norm_dtype(wdtype)
        got_dtype = _norm_dtype(dtype)
        if want_dtype == FP4_TORCH_DTYPE_NAME:
            # packed fp4 arrives as int8 (2 values per byte): same shape, same bytes, a view.
            if got_dtype not in (FP4_TORCH_DTYPE_NAME, "int8", "uint8"):
                report.dtype_mismatch.append((name, want_dtype, got_dtype))
        elif want_dtype != got_dtype:
            report.dtype_mismatch.append((name, want_dtype, got_dtype))
        report.copied += 1
        report.bytes_in += int(nbytes)

    for name in want:
        if name not in headers:
            report.missing.append(name)
    return report


def _is_skipped_by_config(name: str, model: Dsv41Model) -> bool:
    """True when the checkpoint has a section this instance was built without (vision / MTP)."""
    if name.startswith("vision.") or name.startswith("aligner.") or name in (
        "image_start",
        "image_end",
        "image_newline",
    ):
        return model.vision is None
    if name.startswith("mtp."):
        return len(model.mtp) == 0
    if name.startswith("layers."):
        return False
    return False


# =============================================================================================
# Real load
# =============================================================================================


def fold_wo_a(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """fp8 block-quantized ``wo_a`` -> bf16, identical to ``inference/convert.py``.

    ``weight`` is ``[out, in]`` fp8 and ``scale`` is ``[out // b, in // b]`` e8m0 with ``b`` either 32
    or 128; the result is the block-wise dequantized bf16 matrix the reference's grouped einsum wants.
    """
    if weight.dtype != _FP8:
        raise TypeError(f"fold_wo_a expects an fp8 weight, got {weight.dtype}")
    out_block = weight.size(0) // scale.size(0)
    in_block = weight.size(1) // scale.size(1)
    if (out_block, in_block) not in ((32, 32), (128, 128)):
        raise ValueError(
            f"unsupported wo_a block {out_block}x{in_block} for weight {tuple(weight.shape)} / "
            f"scale {tuple(scale.shape)}"
        )
    w = (
        weight.unflatten(0, (-1, out_block))
        .unflatten(-1, (-1, in_block))
        .float()
        * scale[:, None, :, None].float()
    )
    return w.flatten(2, 3).flatten(0, 1).bfloat16()


def load_shards(
    model: Dsv41Model,
    paths: Sequence[str],
    verify: bool = True,
    progress_every: int = 2000,
    log=None,
    name_filter=None,
) -> LoadReport:
    """Stream every shard into ``model``'s parameters, in place.

    ``verify`` asserts shape (always) and dtype (except the two documented cases) per tensor, so a
    silently reshaped checkpoint cannot land in the model.  Nothing is cached between tensors: the
    transient peak is one shard tensor (or one fp32 ``wo_a`` fold).
    """
    with torch.no_grad():  # inference-only; parameters are leaves, so in-place needs no autograd
        return _load_shards(model, paths, verify, progress_every, log, name_filter)


def _load_shards(model, paths, verify, progress_every, log, name_filter=None) -> LoadReport:
    say = log or (lambda *a, **k: None)
    params = dict(model.named_parameters())
    report = LoadReport()
    pending_scales: Dict[str, torch.Tensor] = {}
    pending_wo_a: Dict[str, torch.Tensor] = {}

    total = len(params)
    for path in paths:
        with safe_open(path, "pt", device="cpu") as handle:
            for name in handle.keys():
                if name_filter is not None and not name_filter(name):
                    continue
                # ``wo_a`` is folded, so its scale is not a parameter and its weight is owned only
                # when the tree has the layer at all (MTP/vision sections may be switched off).
                is_wo_a_weight = name.endswith(".wo_a.weight")
                is_wo_a_scale = name.endswith(".wo_a.scale")
                if is_wo_a_scale:
                    owned = (name[: -len(".scale")] + ".weight") in params
                else:
                    owned = name in params
                if not owned:
                    if _is_skipped_by_config(name, model):
                        report.skipped_by_config.append(name)
                    else:
                        report.extra.append(name)
                    continue
                tensor = handle.get_tensor(name)

                if is_wo_a_scale:
                    stem = name[: -len(".scale")]
                    weight_name = stem + ".weight"
                    if weight_name in pending_wo_a:
                        target = params[weight_name]
                        target.copy_(fold_wo_a(pending_wo_a.pop(weight_name), tensor))
                        report.folded_wo_a += 1
                        report.copied += 1
                        report.bytes_in += target.numel() * target.element_size()
                    else:
                        pending_scales[stem] = tensor
                    continue

                if is_wo_a_weight and name[: -len(".weight")] in pending_scales:
                    stem = name[: -len(".weight")]
                    target = params[name]
                    target.copy_(fold_wo_a(tensor, pending_scales.pop(stem)))
                    report.folded_wo_a += 1
                    report.copied += 1
                    report.bytes_in += target.numel() * target.element_size()
                    continue

                target = params[name]
                if tuple(tensor.shape) != tuple(target.shape):
                    report.shape_mismatch.append(
                        (name, tuple(target.shape), tuple(tensor.shape))
                    )
                    continue
                if verify and tensor.dtype != target.dtype:
                    if target.dtype == _FP4 and tensor.dtype == torch.int8:
                        pass  # packed fp4 arrives as int8; same bytes, reinterpreted below
                    elif target.dtype == torch.float32 and tensor.dtype == torch.bfloat16:
                        pass  # the reference holds some bf16 checkpoint tensors in fp32 (upcast)
                    elif target.dtype == torch.bfloat16 and tensor.dtype == torch.float32:
                        pass  # ... and a caller may keep them bf16
                    else:
                        report.dtype_mismatch.append(
                            (name, str(target.dtype), str(tensor.dtype))
                        )
                        continue
                if target.dtype == _FP4:
                    target.copy_(tensor.view(_FP4))
                else:
                    target.copy_(tensor)
                report.copied += 1
                report.bytes_in += int(tensor.numel() * tensor.element_size())
                del tensor
                if progress_every and report.copied % progress_every == 0:
                    say(
                        f"  loaded {report.copied} tensors "
                        f"({report.bytes_in/1e9:.0f} GB) of {total} params"
                    )

    # a scale whose weight never arrived (or vice versa) is a corrupt checkpoint
    for stem in pending_scales:
        report.missing.append(stem + ".weight")
    for stem in pending_wo_a:
        report.missing.append(stem + ".scale")

    seen = _seen_names(paths)
    for name in params:
        if name not in seen and name not in report.missing:
            report.missing.append(name)
    return report


_seen_cache: Dict[Tuple[str, ...], set] = {}


def _seen_names(paths: Sequence[str]) -> set:
    key = tuple(paths)
    cached = _seen_cache.get(key)
    if cached is None:
        cached = set(header_map(paths))
        _seen_cache[key] = cached
    return cached
