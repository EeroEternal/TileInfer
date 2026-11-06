"""Serve any model the TileInfer profiles know, through vLLM's native backend - one command.

    python scripts/serve_tileinfer.py /models/Qwen3.6-35B-A3B --max-model-len 32768 --port 8100
    python scripts/serve_tileinfer.py /models/Qwen3.6-35B-A3B --plan-only
    python scripts/serve_tileinfer.py /models/Qwen3.6-35B-A3B --doctor --max-num-seqs 32

This is the generic launcher: the model is identified from its ``config.json`` architecture, and
every number it prints or checks comes from that model's
:class:`tileinfer.vllm.profiles.ModelProfile` and from :mod:`tileinfer.serving`
(``buckets.KERNEL_PAGE_SIZE`` / ``planner.buckets_up_to`` / ``planner.plan_width``).  There is no
model-named arithmetic here.  ``scripts/serve_qwen36.py`` is a twelve-line alias that pins the
Qwen3.6 profile name and executes this file, so the old command keeps working without a second
copy of anything.

What it does
------------
1. resolves the profile by architecture and validates the model's ``config.json`` against it
   (head widths, the GQA group floor, and - when ``--hybrid`` is claimed - actual mamba/GDN
   layers);
2. derives the decode-plan pre-warm the worker will perform (bucket set, block-table width, plan
   count) through ``tileinfer.serving`` and prints it - ``--plan-only`` stops there;
3. sets the documented environment (``VLLM_NO_USAGE_STATS``, ``VLLM_DO_NOT_TRACK``,
   ``TILELANG_CACHE_DIR``), optionally runs ``tileinfer.vllm.doctor`` first, and **execs** vLLM's
   CLI with ``serve <model> --attention-backend CUSTOM`` plus every argument it did not consume.

Why ``--attention-backend CUSTOM``
----------------------------------
``tileinfer.vllm.plugin`` registers ``TileInferAttentionBackend`` in
``AttentionBackendEnum.CUSTOM`` (the out-of-tree slot, ``backends/registry.py:99-101``), and
``TileInferPlatform.get_attn_backend_cls`` returns its class path directly.  vLLM 0.23 parses
``--attention-backend`` by upper-casing the string into that enum
(``config/attention.py::validate_backend_before``), so ``CUSTOM`` is the spelling both sides agree
on.  Pinning it also means a stray ``--attention-backend flash_attn`` is refused here instead of
dying inside the engine.

Why no async-scheduling
-----------------------
``GPUModelRunner.__init__`` builds a CUDA ``Stream`` inline when
``scheduler_config.async_scheduling`` is on (``gpu_model_runner.py:696``), which no TileInfer
override can replace - see ``tileinfer/vllm/worker.py``'s module docstring.  This launcher never
passes the flag and refuses it when it appears in the arguments.

Exit codes
----------
``0`` success / ``--plan-only``; ``1`` ``--doctor`` found a hard failure; ``2`` refused to start
(no model directory, unsupported model shape, unknown architecture, vLLM outside the supported
range, a foreign ``--attention-backend``, ``--async-scheduling``, or missing entry points).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import shlex
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

# This file may be executed as a script (`python scripts/serve_tileinfer.py ...`), in which case
# sys.path[0] is `scripts/` and the package next door is not importable.  Add the checkout root
# before importing tileinfer; nothing here imports vLLM.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tileinfer.serving.buckets import (  # noqa: E402
    BATCH_BUCKETS,
    KERNEL_PAGE_SIZE,
    fused_halves,
    next_pow2,
    sub_pages,
)
from tileinfer.serving.planner import buckets_up_to, plan_width  # noqa: E402
from tileinfer.vllm.profiles import (  # noqa: E402
    ModelProfile,
    default_profile,
    profile_by_name,
    profile_for,
)

__all__ = ["main", "Refusal"]

#: ``tileinfer.vllm.attn.backend``'s class path.  A literal, because importing the backend needs
#: vLLM and ``--plan-only`` must work on a box without it (the module still has to name the class).
ATTN_BACKEND_CLASS_PATH = "tileinfer.vllm.attn.backend.TileInferAttentionBackend"

#: ``tileinfer/vllm/plugin.SUPPORTED_VLLM_RANGE`` (kept in sync by ``tests/test_boot_rehearsal.py``).
SUPPORTED_VLLM_RANGE = ">=0.23,<0.24"
SUPPORTED_VLLM_MAJOR_MINOR: Tuple[int, int] = (0, 23)

#: ``AttentionBackendEnum.CUSTOM`` - the only value this platform accepts.
ATTENTION_BACKEND = "CUSTOM"

#: ``vllm.entrypoints.cli.main``: run with ``-m`` because that module has a ``__main__`` guard
#: (v0.23 ``entrypoints/cli/main.py``), so the launcher does not depend on the ``vllm`` console
#: script being first on PATH.
VLLM_CLI_MODULE = "vllm.entrypoints.cli.main"

#: The two entry-point groups ``pyproject.toml`` installs; without them vLLM never sees this
#: package (it would pick its own platform) - the failure mode that looks like "TileInfer ignored".
ENTRY_POINT_GROUPS: Tuple[str, ...] = ("vllm.platform_plugins", "vllm.general_plugins")

#: Environment the launcher documents and sets (never a telemetry opt-in).
ENV_FIXED: Dict[str, str] = {"VLLM_NO_USAGE_STATS": "1", "VLLM_DO_NOT_TRACK": "1"}
ENV_CACHE = "TILELANG_CACHE_DIR"
CACHE_DIR_DEFAULT = "~/.cache/tilelang-ascend950"

#: vLLM's defaults when the flags are absent (used **only** for the printed plan; they are not
#: forwarded, so an explicitly chosen engine default still wins at serve time).
DEFAULT_MAX_NUM_SEQS = 256
DEFAULT_MAX_NUM_BATCHED_TOKENS = 8192

#: The launcher's own options that are also vLLM options: the value is used for the plan *and*
#: re-emitted into the vLLM argv, so the printed plan and the engine cannot disagree.
ENGINE_OPTIONS: Tuple[Tuple[str, str], ...] = (
    ("--max-model-len", "max_model_len"),
    ("--max-num-seqs", "max_num_seqs"),
    ("--max-num-batched-tokens", "max_num_batched_tokens"),
    ("--block-size", "block_size"),
)

#: The boot profile supplies the two module-level compatibility aliases older probes/tests read as
#: data; the plan itself always uses the **resolved** profile (see ``_model_shape``).
_BOOT_PROFILE = default_profile()
HYBRID_BLOCK_SIZE = int(_BOOT_PROFILE.block_size_policy.required_multiple(hybrid=True))
SUPPORTED_HEAD_SIZES: Tuple[int, ...] = tuple(_BOOT_PROFILE.head_size)


class Refusal(Exception):
    """A start-up refusal: the message is printed, the exit code is 2, nothing is launched."""


# =============================================================================================
# Profile resolution and arithmetic (all delegated to tileinfer.serving / the profile)
# =============================================================================================


def _forced_profile_name() -> Optional[str]:
    """The profile this entry point pins, if any (``serve_qwen36.py`` sets ``PROFILE_NAME``)."""
    value = globals().get("PROFILE_NAME")
    return str(value) if value else None


def _profile_for_model(
    config: Dict[str, Any], profile_name: Optional[str] = None
) -> ModelProfile:
    """Resolve the model's profile from its architecture, or pin it by name."""
    if profile_name:
        return profile_by_name(profile_name)
    text = config.get("text_config")
    text = text if isinstance(text, dict) else config
    archs = config.get("architectures") or text.get("architectures")
    if archs:
        try:
            return profile_for(archs)
        except ValueError as exc:
            raise Refusal(str(exc)) from None
    # No architecture in the config: only a single-profile registry is unambiguous.
    try:
        return default_profile()
    except RuntimeError as exc:  # pragma: no cover - the package always registers one
        raise Refusal(f"cannot resolve a model profile: {exc}") from None


def _human_int(text: str, what: str) -> int:
    """Parse vLLM-style integer spellings (``8192``, ``8k``, ``8K``; lowercase = decimal).

    ``vllm.utils.argparse_utils.human_readable_int`` uses the same convention: ``k/m/g/t`` are
    decimal (1000^n), ``K/M/G/T`` binary (1024^n).  Parsing it here keeps a user's ``8k`` from
    becoming an argparse error before vLLM ever sees it.
    """
    raw = str(text).strip()
    if not raw:
        raise Refusal(f"{what} needs an integer, got an empty value")
    multiplier = 1
    suffix = raw[-1]
    if suffix.isalpha():
        exponent = "kmgt".find(suffix.lower())
        if exponent < 0 or len(raw) < 2:
            raise Refusal(f"{what} needs an integer (optionally k/m/g/t), got {text!r}")
        multiplier = (1024 if suffix.isupper() else 1000) ** (exponent + 1)
        raw = raw[:-1]
    try:
        return int(raw) * multiplier
    except ValueError:
        raise Refusal(f"{what} needs an integer (optionally k/m/g/t), got {text!r}") from None


def _decode_buckets(max_num_reqs: int) -> Tuple[int, ...]:
    """``tileinfer.serving.planner.buckets_up_to`` - the buckets a deployment can reach."""
    return buckets_up_to(max(1, int(max_num_reqs)))


def _plan_width(max_model_len: int, block_size: int) -> int:
    """``tileinfer.serving.planner.plan_width`` - block-table width in kernel pages."""
    if max_model_len <= 0:
        raise Refusal(f"max_model_len must be positive, got {max_model_len}")
    try:
        return plan_width(max_model_len, block_size)
    except ValueError as exc:
        raise Refusal(str(exc)) from None


def _next_pow2(value: int) -> int:
    """``tileinfer.serving.buckets.next_pow2`` (the declared pool capacity rounds up)."""
    return next_pow2(value)


def _k_half_engine_pages(pool: Any, block_dim: int = 0) -> int:
    """Engine pages in the K half of the fused KV pool - the count the plan key is built from.

    The split is ``tileinfer.serving.buckets.fused_halves``' rule (the same helper the worker
    uses), and the K half is ``pool[:, 0]`` for the block-first layout this backend declares.  A
    plain integer (``--pool-pages``, the engine's own page count) is already that number.
    """
    if isinstance(pool, int) and not isinstance(pool, bool):
        if pool <= 0:
            raise Refusal(f"the KV pool must have at least one page, got {pool}")
        return int(pool)
    shape = tuple(int(n) for n in getattr(pool, "shape", ()))
    if len(shape) != 5:
        raise Refusal(
            "the fused KV pool must be 5-D ([blocks, 2, page, heads, dim] for this backend), "
            f"got shape {shape}"
        )
    if block_dim not in (0, 1):
        raise Refusal(f"block_dim must be 0 or 1, got {block_dim}")
    try:
        k_half, _v_half = fused_halves(pool, int(block_dim))
    except ValueError as exc:
        raise Refusal(str(exc)) from None
    return int(k_half.shape[0])


# =============================================================================================
# Model directory and shape
# =============================================================================================


def _load_model_config(model: str) -> Dict[str, Any]:
    """Read ``<model>/config.json`` or refuse with the reason (this is the "missing dir" gate)."""
    path = pathlib.Path(model).expanduser()
    if not model:
        raise Refusal(
            "MODEL is required: the model directory must be the first argument "
            "(e.g. `serve_tileinfer.py /models/Qwen3.6-35B-A3B --max-model-len 32768`)"
        )
    if not path.is_dir():
        raise Refusal(
            f"the model directory {model!r} does not exist or is not a directory (resolved to "
            f"{path}). MODEL must be a local path - download the weights first: "
            "`huggingface-cli download <repo> --local-dir <dir>`"
        )
    config_path = path / "config.json"
    if not config_path.is_file():
        raise Refusal(
            f"{model!r} is a directory but contains no config.json, so it is not a HuggingFace "
            "model directory. Point MODEL at the directory that holds config.json (and the "
            "safetensors weights)."
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - the file is user data
        raise Refusal(f"cannot read {config_path}: {exc!r}") from None
    if not isinstance(config, dict):
        raise Refusal(f"{config_path} does not contain a JSON object (got {type(config).__name__})")
    return config


def _first_int(mapping: Dict[str, Any], names: Sequence[str]) -> Optional[int]:
    """The first of ``names`` that is a positive integer, else ``None``."""
    for name in names:
        value = mapping.get(name)
        if isinstance(value, bool) or value is None:
            continue
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            return number
    return None


def _model_shape(
    config: Dict[str, Any],
    *,
    claims_hybrid: bool,
    profile_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate the model's shape against its profile; every rejection names the field and the fix.

    The checks are the ones that would otherwise fail *after* the weights are loaded:
    ``TileInferAttentionBackend.check_head_layout`` (head_size, GQA group) and the block-size /
    hybrid-page policy in ``TileInferPlatform.validate_block_size``.  The numbers come from the
    resolved :class:`ModelProfile`, never from a module constant.
    """
    profile = _profile_for_model(config, profile_name)
    text = config.get("text_config")
    text = text if isinstance(text, dict) else config  # multimodal wrappers nest the LM config

    num_heads = _first_int(text, ("num_attention_heads", "num_heads"))
    if num_heads is None:
        raise Refusal(
            "config.json has no usable num_attention_heads/num_heads, so the attention shape "
            "cannot be checked. Point MODEL at the text model (or its wrapper) config."
        )
    num_kv_heads = _first_int(text, ("num_key_value_heads", "num_kv_heads")) or num_heads
    hidden_size = _first_int(text, ("hidden_size", "n_embd", "d_model"))
    head_dim = _first_int(text, ("head_dim", "head_size"))
    if head_dim is None:
        if hidden_size is None:
            raise Refusal(
                "config.json has neither head_dim nor hidden_size, so the attention head width "
                "cannot be derived. Add head_dim to the config, or use a supported model config."
            )
        head_dim = hidden_size // num_heads

    supported_heads = [int(width) for width in profile.head_size]
    if head_dim not in supported_heads:
        raise Refusal(
            f"head_size (head_dim) is {head_dim}, but model profile {profile.name!r} supports "
            f"{supported_heads} only. This model is not the supported shape; do not serve it "
            "through the native backend (add a profile if it should be supported)."
        )
    if num_kv_heads > num_heads or num_heads % num_kv_heads:
        raise Refusal(
            f"num_attention_heads={num_heads} is not a multiple of num_key_value_heads="
            f"{num_kv_heads}; the kernel needs an integral GQA group"
        )
    group = num_heads // num_kv_heads
    if group < int(profile.group_min):
        raise Refusal(
            f"GQA group is {group} ({num_heads} q heads / {num_kv_heads} kv heads); model profile "
            f"{profile.name!r} requires group >= {int(profile.group_min)} (plain MHA is not "
            "implemented), so this model cannot run on this backend"
        )

    layer_types = text.get("layer_types", config.get("layer_types"))
    mamba_layers: Optional[int] = None
    attention_layers: Optional[int] = None
    if isinstance(layer_types, list) and layer_types:
        attention_layers = sum(1 for entry in layer_types if entry == "full_attention")
        mamba_layers = len(layer_types) - attention_layers
        declared_layers = _first_int(
            text, ("num_hidden_layers", "n_layer")
        ) or _first_int(config, ("num_hidden_layers", "n_layer"))
        if declared_layers is not None and declared_layers != len(layer_types):
            raise Refusal(
                f"config.json is inconsistent: num_hidden_layers={declared_layers} but "
                f"layer_types has {len(layer_types)} entries. Fix the config before serving - the "
                "KV-cache groups are built from layer_types, so a mismatch silently changes the "
                "layer count."
            )

    hybrid = (mamba_layers or 0) > 0
    align = profile.block_size_policy.hybrid_align
    if claims_hybrid and not hybrid:
        raise Refusal(
            "--hybrid was given, but config.json has no mamba/GDN layer (layer_types contains no "
            "linear_attention entry; mamba layers found: "
            f"{mamba_layers if mamba_layers is not None else 'unknown - no layer_types'}). A "
            f"hybrid run needs the {align}-token mamba page; drop --hybrid or point MODEL at the "
            "hybrid config."
        )
    if hybrid and profile.hybrid is None:
        raise Refusal(
            f"config.json has mamba/GDN layers but model profile {profile.name!r} declares no "
            "hybrid state-page geometry; register a profile for this model before serving it."
        )

    max_model_len = _first_int(
        text, ("max_position_embeddings", "max_model_len")
    ) or _first_int(config, ("max_position_embeddings", "max_model_len"))
    architecture = config.get("architectures") or text.get("architectures")
    if isinstance(architecture, list):
        architecture = architecture[0] if architecture else None

    return {
        "profile": profile.name,
        "num_attention_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "gqa_group": group,
        "hidden_size": hidden_size,
        "layer_types": layer_types if isinstance(layer_types, list) else None,
        "attention_layers": attention_layers,
        "mamba_layers": mamba_layers,
        "hybrid": hybrid,
        "max_model_len": max_model_len,
        "architecture": architecture,
        "model_type": text.get("model_type") or config.get("model_type"),
        "torch_dtype": text.get("torch_dtype") or config.get("torch_dtype"),
    }


# =============================================================================================
# The derived pre-warm plan
# =============================================================================================


def _build_plan(
    model: str, shape: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    """Derive everything the worker will pre-warm, or refuse if the requested run cannot work."""
    profile = profile_by_name(str(shape["profile"]))
    policy = profile.block_size_policy
    kernel_page = int(policy.kernel_page_size)
    hybrid_align = int(policy.required_multiple(hybrid=bool(shape["hybrid"])))
    if args.block_size is not None:
        block_size = _human_int(args.block_size, "--block-size")
    else:
        block_size = hybrid_align if shape["hybrid"] else kernel_page
    if block_size <= 0 or block_size % kernel_page:
        raise Refusal(
            f"block_size={block_size} is not a positive multiple of {kernel_page}: one "
            f"engine page must be a whole number of {kernel_page}-token kernel pages"
        )
    if shape["hybrid"] and block_size % hybrid_align:
        raise Refusal(
            f"block_size={block_size} is not a multiple of {hybrid_align}: this is a hybrid "
            "attention + mamba/GDN model, and TileInferPlatform refuses an engine page that is "
            "not a whole mamba state page (platform.validate_block_size). Drop --block-size, "
            f"or pass a multiple of {hybrid_align}."
        )

    raw_max_model_len = args.max_model_len
    if raw_max_model_len is not None and str(raw_max_model_len).strip().lower() != "auto":
        max_model_len = _human_int(raw_max_model_len, "--max-model-len")
    else:
        # vLLM's own default/`auto`: the value comes from the model config.
        max_model_len = shape["max_model_len"]
    if not max_model_len:
        raise Refusal(
            "the model config has no max_position_embeddings, so the block-table width (and every "
            "plan's staging buffers) cannot be sized. Pass --max-model-len explicitly."
        )
    if max_model_len <= 0:
        raise Refusal(f"--max-model-len must be positive, got {max_model_len}")

    max_num_seqs = (
        _human_int(args.max_num_seqs, "--max-num-seqs")
        if args.max_num_seqs is not None
        else DEFAULT_MAX_NUM_SEQS
    )
    max_num_batched_tokens = (
        _human_int(args.max_num_batched_tokens, "--max-num-batched-tokens")
        if args.max_num_batched_tokens is not None
        else DEFAULT_MAX_NUM_BATCHED_TOKENS
    )
    if max_num_seqs <= 0:
        raise Refusal(f"--max-num-seqs must be positive, got {max_num_seqs}")
    if max_num_batched_tokens <= 0:
        raise Refusal(
            f"--max-num-batched-tokens must be positive, got {max_num_batched_tokens}"
        )
    max_num_reqs = max(1, min(max_num_seqs, max_num_batched_tokens))
    buckets = _decode_buckets(max_num_reqs)
    width = _plan_width(max_model_len, block_size)
    sub = sub_pages(block_size)
    engine_width = width // sub

    pool_cap: Optional[int] = None
    pool_kernel_pages: Optional[int] = None
    fitting: Optional[Tuple[int, ...]] = None
    pool_pages: Optional[int] = None
    if args.pool_pages is not None:
        pool_pages = _human_int(args.pool_pages, "--pool-pages")
        if pool_pages <= 0:
            raise Refusal(f"--pool-pages must be positive, got {pool_pages}")
        # The engine's log reports *engine* pages; the plan cache keys on the pool viewed as
        # kernel pages over the K half of the block-first fused pool
        # (`declared_pool_cap(pool_sub_view(pool[:, 0], sub).shape[0])`).
        pool_kernel_pages = _k_half_engine_pages(pool_pages) * sub
        pinned = (os.environ.get("TILEINFER_POOL_CAP") or "").strip()
        if pinned:
            pool_cap = _human_int(pinned, "TILEINFER_POOL_CAP")
            if pool_cap < pool_kernel_pages:
                raise Refusal(
                    f"TILEINFER_POOL_CAP={pool_cap} is below the real pool "
                    f"({pool_kernel_pages} kernel pages); the declared capacity must cover every "
                    "live page"
                )
        else:
            pool_cap = _next_pow2(pool_kernel_pages)
        # The impl's staging table needs one trash slot past every destination:
        # `bucket * width < pool_cap - 1` (`_state_for`, attn/impl.py:662-674).
        limit = (pool_cap - 2) // width
        fitting = tuple(bucket for bucket in buckets if bucket <= limit)

    return {
        "model": model,
        "profile": profile.name,
        "block_size": block_size,
        "sub_pages": sub,
        "max_model_len": max_model_len,
        "engine_width": engine_width,
        "width": width,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_reqs": max_num_reqs,
        "buckets": buckets,
        "pool_pages": pool_pages,
        "pool_kernel_pages": pool_kernel_pages,
        "pool_cap": pool_cap,
        "fitting_buckets": fitting,
        "dtype": shape.get("torch_dtype") or "auto",
    }


def _env_updates() -> Dict[str, str]:
    """The documented environment: telemetry off, one shared TileLang cache directory."""
    cache_dir = (os.environ.get(ENV_CACHE) or "").strip() or CACHE_DIR_DEFAULT
    updates = dict(ENV_FIXED)
    updates[ENV_CACHE] = str(pathlib.Path(cache_dir).expanduser())
    return updates


def _format_plan(
    plan: Dict[str, Any],
    shape: Dict[str, Any],
    vllm_status: str,
    env: Dict[str, str],
    args: argparse.Namespace,
) -> str:
    """The pre-flight report.  ASCII only: a Windows console code page must not fail this."""
    profile = profile_by_name(str(shape["profile"]))
    layers = shape["layer_types"]
    if layers is None:
        layer_line = "unknown (config.json has no layer_types)"
    else:
        layer_line = (
            f"{shape['attention_layers']} full-attention / {shape['mamba_layers']} "
            f"linear-attention (GDN/mamba){' - hybrid' if shape['hybrid'] else ''}"
        )
    batch = f"{plan['max_num_reqs']} decode request(s)"
    batch += (
        f" (min(max_num_seqs={plan['max_num_seqs']}, "
        f"max_num_batched_tokens={plan['max_num_batched_tokens']}))"
    )
    if args.max_num_seqs is None or args.max_num_batched_tokens is None:
        batch += " [vLLM default; pin it with --max-num-seqs / --max-num-batched-tokens]"

    lines = [
        "TileInfer launch plan - native vLLM backend (no vllm-ascend)",
        f"  model profile       : {profile.name} (archs: {', '.join(profile.archs)})",
        f"  model dir           : {plan['model']}",
        f"  architecture        : {shape.get('architecture') or '<unknown>'} "
        f"(model_type={shape.get('model_type') or '<unknown>'}, dtype={plan['dtype']})",
        f"  layers              : {layer_line}",
        f"  q / kv heads        : {shape['num_attention_heads']} / {shape['num_kv_heads']} "
        f"(GQA group {shape['gqa_group']}, profile floor {profile.group_min})",
        f"  head_size           : {shape['head_dim']} "
        f"(profile: {sorted(int(n) for n in profile.head_size)})",
        f"  engine block_size   : {plan['block_size']} tokens = {plan['sub_pages']} x "
        f"{int(profile.block_size_policy.kernel_page_size)}-token kernel pages"
        + (" [hybrid mamba page]" if shape["hybrid"] else ""),
        f"  max_model_len       : {plan['max_model_len']} -> "
        f"{plan['width'] // max(1, plan['sub_pages'])} engine page(s) "
        f"= {plan['width']} kernel pages per request",
        f"  batch bound         : {batch}",
        f"  decode buckets      : {', '.join(str(b) for b in plan['buckets'])} "
        f"-> {len(plan['buckets'])} plan(s)",
    ]
    if plan["pool_pages"] is None:
        lines.append(
            "  kv pool             : not given - the engine sizes it from a memory snapshot; pass "
            "--pool-pages <pages> to check which buckets fit"
        )
    else:
        skipped = tuple(b for b in plan["buckets"] if b not in (plan["fitting_buckets"] or ()))
        lines.append(
            f"  kv pool             : {plan['pool_pages']} engine page(s) = "
            f"{plan['pool_kernel_pages']} kernel pages, declared cap {plan['pool_cap']} -> "
            "buckets that fit: "
            f"{', '.join(str(b) for b in (plan['fitting_buckets'] or ())) or '<none>'}"
        )
        if skipped:
            names = ", ".join(str(b) for b in skipped)
            lines.append(
                f"  WARNING             : bucket(s) {names} do not fit the pool "
                "(bucket*width < cap - 1); the worker will not pre-warm them and a batch that "
                "reaches one would raise on its first step"
            )
    lines += [
        "  prewarm             : NPUWorker._prewarm_decode_plans builds every reachable bucket "
        "above (trimmed to the pool when --pool-pages says otherwise), through the impl's own "
        "plan cache, before graph capture",
        "  attention-backend   : "
        f"{ATTENTION_BACKEND} -> {ATTN_BACKEND_CLASS_PATH}",
        f"  env                 : VLLM_NO_USAGE_STATS={env['VLLM_NO_USAGE_STATS']} "
        f"VLLM_DO_NOT_TRACK={env['VLLM_DO_NOT_TRACK']} {ENV_CACHE}={env[ENV_CACHE]}"
        + (" (listed only; --plan-only sets nothing)" if args.plan_only else " (will be exported)"),
        "  async scheduling    : never passed (the native worker cannot run it; "
        "--async-scheduling is refused)",
        f"  vllm                : {vllm_status} (supported {SUPPORTED_VLLM_RANGE})",
    ]
    return "\n".join(lines)


# =============================================================================================
# Serve-time gates
# =============================================================================================


def _distribution_version(name: str) -> str:
    """The installed distribution version, or ``""`` (a source checkout has no metadata)."""
    try:
        from importlib.metadata import version

        return str(version(name))
    except Exception:  # noqa: BLE001 - absence is an answer, not an error
        return ""


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _version_supported(raw: str) -> bool:
    """``0.23.x`` (with a local/``+`` suffix tolerated); anything else is outside the pin."""
    try:
        numbers = raw.split("+")[0].lstrip("vV").split(".")
        return (int(numbers[0]), int(numbers[1])) == SUPPORTED_VLLM_MAJOR_MINOR
    except (IndexError, ValueError):
        return False


def _vllm_gate(required: bool) -> str:
    """The vLLM version gate; ``required=False`` (``--plan-only``) tolerates a missing vLLM."""
    raw = _distribution_version("vllm")
    if raw:
        if not _version_supported(raw):
            raise Refusal(
                f"vLLM {raw} is outside {SUPPORTED_VLLM_RANGE}: the TileInfer platform, worker "
                "and attention backend are written against 0.23 (the platform ABC and the v1 "
                "attention contract were transcribed from it). Install the pinned line: "
                "`pip install 'vllm>=0.23,<0.24'`."
            )
        return raw
    if _module_available("vllm"):
        return (
            "unknown version (source checkout; the supported range is "
            + SUPPORTED_VLLM_RANGE
            + ")"
        )
    if not required:
        return "not installed (--plan-only does not need it)"
    raise Refusal(
        "vLLM is not installed in this interpreter: `import vllm` would fail right after this "
        "launcher execs it. Install the pinned line (`pip install 'vllm>=0.23,<0.24'`) in the "
        "venv that serves, or run with --plan-only to print the plan without an engine."
    )


def _entry_points_ok() -> Tuple[bool, str]:
    """Are the two ``tileinfer`` entry points visible to *this* interpreter?

    vLLM discovers ``platform_plugins``/``general_plugins`` through distribution metadata, so a
    checkout on ``PYTHONPATH`` without ``pip install -e .`` is silently ignored - the engine then
    runs on whatever platform it would have picked anyway.
    """
    try:
        from importlib.metadata import entry_points

        found = entry_points()
    except Exception as exc:  # noqa: BLE001
        return False, f"cannot read the installed entry points ({exc!r})"
    paths: Dict[str, List[str]] = {}
    for group in ENTRY_POINT_GROUPS:
        if hasattr(found, "select"):
            entries = list(found.select(group=group))
        else:  # Python 3.9: a dict of group -> tuple
            entries = list(found.get(group, ()))  # type: ignore[union-attr]
        paths[group] = [str(getattr(entry, "value", entry)) for entry in entries]
    missing = [
        group
        for group, values in paths.items()
        if not any(value.startswith("tileinfer.") for value in values)
    ]
    if missing:
        return False, (
            f"the tileinfer entry points are not installed for this interpreter (missing from "
            f"{missing}); vLLM would never load the TileInfer platform. Install the package into "
            "the serving venv: `pip install -e .` (from the checkout), then check `pip show "
            "tileinfer`."
        )
    return True, "; ".join(f"{group}={paths[group]}" for group in ENTRY_POINT_GROUPS)


def _async_scheduling_requested(extra: Sequence[str]) -> bool:
    """Whether the forwarded args turn async scheduling on (``--no-...`` is the safe spelling)."""
    for token in extra:
        if token == "--async-scheduling":
            return True
        if token.startswith("--async-scheduling="):
            value = token.split("=", 1)[1].strip().lower()
            return value not in ("", "0", "false", "no", "off")
    return False


def _attention_backend_args(extra: Sequence[str]) -> List[str]:
    """Validate a user-pinned ``--attention-backend``; otherwise pin ``CUSTOM`` ourselves."""
    for index, token in enumerate(extra):
        if token == "--attention-backend" and index + 1 < len(extra):
            value = extra[index + 1]
        elif token.startswith("--attention-backend="):
            value = token.split("=", 1)[1]
        else:
            continue
        if value.strip().lower() not in ("custom", "tileinfer"):
            raise Refusal(
                f"--attention-backend {value!r} is not served by the TileInfer platform: this "
                f"backend is registered as {ATTENTION_BACKEND} (AttentionBackendEnum.CUSTOM) and "
                "the platform returns its class path directly. Leave the flag off (the launcher "
                "sets it) or pass CUSTOM."
            )
        return []  # keep the user's own (accepted) spelling in the forwarded args
    return ["--attention-backend", ATTENTION_BACKEND]


def _run_doctor() -> int:
    """``tileinfer.vllm.doctor`` in this process; its own exit code is returned (0 = healthy)."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from tileinfer.vllm.doctor import main as doctor_main
    except Exception as exc:  # noqa: BLE001
        raise Refusal(
            f"--doctor needs `tileinfer` importable from this interpreter, which failed ({exc!r}). "
            "Install it (`pip install -e .`) or drop --doctor."
        ) from None
    return int(doctor_main([]))


# =============================================================================================
# Argument parsing and the exec
# =============================================================================================


def _parse(argv: Optional[Sequence[str]]) -> Tuple[argparse.Namespace, List[str]]:
    """Split the launcher's own options from everything vLLM should receive.

    MODEL must be the first non-option argument (the ``vllm serve MODEL ...`` convention); every
    other unknown argument is forwarded untouched, in order.
    """
    parser = argparse.ArgumentParser(
        prog=pathlib.Path(sys.argv[0]).name or "serve_tileinfer.py",
        description=(
            "Serve a model supported by a TileInfer profile on the native vLLM backend (no "
            "vllm-ascend). Prints the decode-plan pre-warm it derives, then execs vLLM's CLI."
        ),
        epilog=(
            "examples:\n"
            "  serve_tileinfer.py /models/Qwen3.6-35B-A3B --max-model-len 32768 --port 8100\n"
            "  serve_tileinfer.py /models/Qwen3.6-35B-A3B --plan-only\n"
            "  serve_tileinfer.py /models/Qwen3.6-35B-A3B --doctor --max-num-seqs 32\n"
            "notes:\n"
            "  MODEL must be the first argument; unknown options are passed to `vllm serve`.\n"
            "  --plan-only needs the checkout's `tileinfer` package (and torch), but not vLLM.\n"
            "  exit codes: 0 ok, 1 --doctor hard failure, 2 refused to start."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model", nargs="?", help="local model directory (first argument)")
    parser.add_argument("--model", dest="model_option", help="same as the positional MODEL")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="print the derived pre-warm plan and exit 0; launch nothing, set no environment",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="run `tileinfer.vllm.doctor` first and refuse to serve if it reports a hard failure",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="require the model to have mamba/GDN layers (a hybrid profile)",
    )
    parser.add_argument("--max-model-len", help="forwarded to vLLM and used for the plan")
    parser.add_argument("--max-num-seqs", help="forwarded to vLLM and used for the plan")
    parser.add_argument(
        "--max-num-batched-tokens", help="forwarded to vLLM and used for the plan"
    )
    parser.add_argument("--block-size", help="engine page size; forwarded to vLLM")
    parser.add_argument(
        "--pool-pages",
        help="engine KV pages (from the engine's 'GPU KV cache size' log) for the printed plan",
    )
    return parser.parse_known_args(list(sys.argv[1:] if argv is None else argv))


def _forwarded_args(args: argparse.Namespace, extra: List[str]) -> List[str]:
    """Everything vLLM should see: our consumed engine options re-emitted once, then the rest."""
    forwarded = list(extra)
    for flag, attribute in ENGINE_OPTIONS:
        value = getattr(args, attribute)
        if value is not None:
            forwarded += [flag, str(value)]
    return forwarded


def _vllm_argv(model: str, forwarded: Sequence[str], backend_args: Sequence[str]) -> List[str]:
    """``python -m vllm.entrypoints.cli.main serve MODEL --attention-backend CUSTOM ...``."""
    return [
        sys.executable,
        "-m",
        VLLM_CLI_MODULE,
        "serve",
        model,
        *backend_args,
        *forwarded,
    ]


def _apply_env(env: Dict[str, str]) -> None:
    """Export the documented environment and make the TileLang cache directory usable."""
    os.environ.update(env)
    cache_dir = pathlib.Path(env[ENV_CACHE])
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001 - the engine can still use its own default
        print(
            f"serve_tileinfer.py: warning: cannot create {ENV_CACHE}={cache_dir} ({exc!r}); kernel "
            "compiles may fall back to an unwritable cache (see `tileinfer doctor`)",
            file=sys.stderr,
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Validate, print the plan, then exec vLLM.  Returns only for the non-exec paths."""
    try:
        args, extra = _parse(argv)
        if args.model_option and args.model:
            raise Refusal(
                "give MODEL once: either positionally (first argument) or with --model, not both"
            )
        model = args.model_option or args.model
        config = _load_model_config(model or "")
        shape = _model_shape(
            config, claims_hybrid=bool(args.hybrid), profile_name=_forced_profile_name()
        )
        plan = _build_plan(model, shape, args)
        if _async_scheduling_requested(extra):
            raise Refusal(
                "--async-scheduling is not supported by the native TileInfer worker: "
                "GPUModelRunner.__init__ builds a CUDA Stream inline when async scheduling is on "
                "(gpu_model_runner.py:696), which no TileInfer override can replace. Drop the flag "
                "- eager scheduling is the supported mode."
            )
        backend_args = _attention_backend_args(extra)
        vllm_status = _vllm_gate(required=not args.plan_only)
        env = _env_updates()

        if args.plan_only:
            print(_format_plan(plan, shape, vllm_status, env, args))
            return 0

        ok, detail = _entry_points_ok()
        if not ok:
            raise Refusal(detail)
        if args.doctor:
            code = _run_doctor()
            if code != 0:
                print(
                    f"serve_tileinfer.py: --doctor reported a hard failure (exit {code}); fix the "
                    "checklist above, or drop --doctor to serve anyway",
                    file=sys.stderr,
                )
                return 1

        print(_format_plan(plan, shape, vllm_status, env, args))
        _apply_env(env)
        argv_exec = _vllm_argv(model, _forwarded_args(args, extra), backend_args)
        print("serve_tileinfer.py: exec " + shlex.join(argv_exec), flush=True)
        os.execvpe(argv_exec[0], argv_exec, os.environ)
        return 0  # NOTREACHED on success; exec only returns when it failed
    except Refusal as exc:
        print(f"serve_tileinfer.py: refusing to start: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
