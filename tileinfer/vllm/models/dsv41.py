"""``tileinfer.vllm.models.dsv41`` — vLLM adapter for ``DeepseekV41ForCausalLM``.

What this is, and what it is not (yet)
-------------------------------------
vLLM 0.23.0 registers ``DeepseekV4ForCausalLM`` but **not** v4.1 (facts §18): the engram tables, the
DSpark stages, the two-level candidate indexer and the vision tower are all absent upstream at this
version, so the architecture has to be supplied out of tree.  This module supplies it:

* :data:`MODEL_ARCH` / :func:`register_models` — the ``ModelRegistry.register_model`` call, run from
  the general plugin (``tileinfer.vllm.plugin.install`` step 5).
* :class:`DeepseekV41ForCausalLM` — the module vLLM instantiates.  It owns the parameter tree from
  :mod:`tileinfer.models.dsv41.modules` and implements the weight-loading protocol in *vLLM's*
  iterator form (name -> tensor), reusing the same two transformations as the standalone loader.

Stage status: S1 = registration + the parameter tree + loading (CPU).  ``forward`` raises
:class:`~tileinfer.models.dsv41.modules.NotImplementedYet` until S2/S3 land the Compressor/Indexer
attention and the end-to-end first token; the error text says exactly that, so a half-ported model
cannot be mistaken for a working one.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch
from torch import nn

from tileinfer.models.dsv41.config import Dsv41Config
from tileinfer.models.dsv41.modules import Dsv41Model, NotImplementedYet, _unimplemented
from tileinfer.models.dsv41.weights import fold_wo_a

__all__ = [
    "MODEL_ARCH",
    "DeepseekV41ForCausalLM",
    "build_model",
    "register_models",
    "LOADED_ATTR",
]

#: The architecture string in the checkpoint's ``config.json``.
MODEL_ARCH = "DeepseekV41ForCausalLM"

#: Set on the module once ``load_weights`` completed; the S1 evidence path reads it.
LOADED_ATTR = "_dsv41_loaded_tensors"

_FP4 = torch.float4_e2m1fn_x2


def build_model(config: Dsv41Config, device: str = "cpu") -> Dsv41Model:
    """Construct the parameter tree (also used by the CPU loader and the tests).

    Text-only on purpose: ``Dsv41Model`` defaults to ``with_vision=True, with_mtp=True``, while the
    verified S3/S4/S5 path builds with both off (and S1 reports those 2667 tensors as
    ``skipped_by_config``).  Building them here loaded 96042 tensors instead of 93378 and put the
    model in a different configuration than the one whose first token matches the oracle.
    """
    return Dsv41Model(config, device=device, with_vision=False, with_mtp=False)


class DeepseekV41ForCausalLM(nn.Module):
    """vLLM module for DeepSeek-V4.1-Flash.

    ``vllm_config.model_config.hf_config`` is the checkpoint's ``config.json``; transformers has no
    ``deepseek_v41`` class, so :meth:`Dsv41Config.from_hf` accepts the generic config object or a
    plain dict.  Construction places the parameter tree on ``device`` (CPU by default, so a box
    without an NPU can still load and verify the checkpoint).

    The vLLM-visible submodule is ``self.model`` (the transformer) plus ``self.lm_head``, mirroring
    how the upstream DeepSeek models are shaped; the checkpoint's own names are resolved by
    :meth:`_resolve` rather than by a vLLM ``WeightsMapper``, because the released shards already use
    this runtime's spelling (facts §0).
    """

    #: vLLM reads this for pooling/quantisation plumbing; empty means "no fused QKV/MLP packs".
    packed_modules_mapping: Dict[str, List[str]] = {}

    # vLLM's ModelConfig asks the *class* what it is (registry.inspect_model_cls ->
    # ``model_cls.is_text_generation_model``); without these it rejects ``--runner generate``.
    is_text_generation_model = True
    is_pooling_model = False
    supports_multimodal = False

    def __init__(self, *, vllm_config=None, prefix: str = "", config: Optional[Dsv41Config] = None,
                 device: str = "cpu") -> None:
        super().__init__()
        if config is None:
            if vllm_config is None:
                raise TypeError(
                    "DeepseekV41ForCausalLM needs either vllm_config or an explicit Dsv41Config"
                )
            config = Dsv41Config.from_hf(getattr(vllm_config.model_config, "hf_config", {}))
        self.dsv41_config = config
        self.model = build_model(config, device=device)
        self.lm_head = self.model.head
        self.prefix = prefix
        # P3: the multi-request gate reads request ids from a runner hook.  Install it here (only
        # when the flag is on) -- without this the hook is never called and current_req_ids() stays
        # None, which silently falls back to the single-sequence path and crashes on concurrency.
        import os as _os_init

        if _os_init.environ.get("VLLM_DSV41_MULTI_REQ", "") not in ("", "0"):
            try:
                from tileinfer.vllm.models.dsv41_req_ids import install as _install_req_ids

                self._req_ids_hooked = bool(_install_req_ids())
            except Exception:  # noqa: BLE001 - 钩子失败不影响推理
                self._req_ids_hooked = False
        try:
            self._model_dir = str(getattr(vllm_config.model_config, "model", "") or "")
        except Exception:  # noqa: BLE001 - only used to build the engram hash provider
            self._model_dir = ""
        try:
            self._max_model_len = int(getattr(vllm_config.model_config, "max_model_len", 0) or 0)
        except Exception:  # noqa: BLE001
            self._max_model_len = 0

    # -- introspection -----------------------------------------------------------------------

    @property
    def config(self) -> Dsv41Config:  # vLLM code paths expect `.config`
        return self.dsv41_config

    def host_offload_names(self) -> List[str]:
        """Parameters the S4 offload design keeps in host memory (experts + engram tables)."""
        return self.model.host_offload_names()

    def finalize_placement(self, device: str = "npu:0") -> Dict[str, int]:
        """Apply the S4 arrangement after the weights are loaded (integration step C).

        vLLM profiles device memory right after ``load_model()``, so by then the dense weights
        (10.42 GiB measured) must be on the card and the routed experts plus both engram tables
        (457.78 GiB) must still be in host memory -- otherwise vLLM either refuses to start or counts
        476 GB against the card.  Returns the byte totals so a caller can log the split.
        """
        moved = kept = 0
        host_markers = (".ffn.experts.", ".engram.embed.")
        for name, param in self.model.named_parameters():
            nbytes = param.numel() * param.element_size()
            if any(marker in name for marker in host_markers) or name.startswith(
                ("engram.embed.", "ffn.experts.")
            ):
                kept += nbytes
                continue
            param.data = param.data.to(device)
            moved += nbytes
        return {"on_device_bytes": moved, "on_host_bytes": kept}

    # -- weight loading ----------------------------------------------------------------------

    def _resolve(self, name: str) -> Tuple[Optional[nn.Parameter], Optional[str]]:
        """Checkpoint name -> parameter.

        Returns ``(param, kind)`` where ``kind`` is ``None`` for a direct copy, ``"fp4"`` for a packed
        expert weight (arrives as int8), ``"wo_a_weight"``/``"wo_a_scale"`` for the folded projection,
        or ``(None, "skip")`` for a tensor this instance does not own (vision/MTP disabled).
        """
        params = self._named_params
        # ``wo_a`` MUST be resolved before the parameter lookup below: the tree holds a *folded bf16*
        # ``layers.N.attn.wo_a.weight``, so ``params.get(name)`` would hit it and copy the raw fp8
        # checkpoint tensor into bf16 (garbage of magnitude ~1e2 instead of 2e-2).  That single mistake
        # was what made the vLLM attention output explode ~4000x (plan §27-§29).
        if name.endswith(".wo_a.scale"):
            return None, ("wo_a_scale" if (name[: -len(".scale")] + ".weight") in params else "skip")
        if name.endswith(".wo_a.weight") and name in params:
            return None, "wo_a_weight"
        param = params.get(name)
        if param is not None:
            if param.dtype == _FP4:
                return param, "fp4"
            return param, None
        # the wo_a branches only apply when this instance actually has the folded parameter: a
        # text-only build has no ``mtp.N.attn.wo_a.*``, and the folded copy below would KeyError
        if name.endswith(".wo_a.scale"):
            return None, ("wo_a_scale" if (name[: -len(".scale")] + ".weight") in params else "skip")
        if name.endswith(".wo_a.weight"):
            return None, ("wo_a_weight" if name in params else "skip")
        return None, "skip"

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        """Copy vLLM's ``(name, tensor)`` stream into the tree; returns the consumed names.

        Two transformations, exactly as the standalone loader (facts §3.4):
        packed fp4 experts are a dtype ``view``; ``wo_a`` is folded from fp8 + e8m0 scale to bf16.
        A wo_a pair may arrive in either order, so a small pending map is kept (one entry in
        practice — the scale always precedes the weight within a shard).
        """
        self._named_params = dict(self.model.named_parameters())
        params = self._named_params
        consumed: Set[str] = set()
        pending_scale: Optional[Tuple[str, torch.Tensor]] = None
        loaded = 0

        # mmap 权重（默认开启）：host-offload 的专家/engram 用 mmap 视图而非 copy，
        # 把 457.78 GiB anon 变成可回收页缓存。实测 rss 490 GB → 72 GB，
        # 服务在 available=371 GB 时也能正常启动并完成 4 并发（见 perf-plan §25）。
        # 设 DSV41_MMAP_WEIGHTS=0 可退回旧的 copy 路径。
        _mmap_src = None
        _env_mmap = __import__("os").environ.get("DSV41_MMAP_WEIGHTS", "1")
        if _env_mmap not in ("0",):
            try:
                from tileinfer.vllm.models.mmap_weights import MmapWeightSource

                if getattr(self, "_model_dir", ""):
                    _mmap_src = MmapWeightSource(self._model_dir)
                    self._mmap_src = _mmap_src      # 保活：句柄必须在模型生命周期内一直存活
            except Exception:  # noqa: BLE001 - 失败就退回普通 copy 路径
                _mmap_src = None

        def _is_host_offload(tensor_name: str) -> bool:
            return (
                ".ffn.experts." in tensor_name
                or tensor_name.startswith(("ffn.experts.", "engram.embed.", "engram."))
                or ".engram.embed." in tensor_name
            )

        for name, tensor in weights:
            param, kind = self._resolve(name)
            if kind == "skip":
                # MTP/DSpark and vision tensors exist in the checkpoint but not in a text-only build.
                # They still have to be reported as consumed: vLLM checks the leftovers against its own
                # parameter dict and raises ``KeyError: 'mtp.0.attn.wo_a.weight'`` when a name is missing
                # from both sides.
                consumed.add(name)
                continue
            if kind == "wo_a_scale":
                pending_scale = (name, tensor)
                consumed.add(name)
                continue
            if kind == "wo_a_weight":
                if pending_scale is None:
                    raise RuntimeError(
                        f"{name} arrived before its scale; the folded wo_a needs both in one pass"
                    )
                scale_name, scale = pending_scale
                if scale_name[: -len(".scale")] != name[: -len(".weight")]:
                    raise RuntimeError(f"{name} does not match pending scale {scale_name}")
                target = params[name]
                target.copy_(fold_wo_a(tensor, scale))
                consumed.add(name)
                loaded += 1
                pending_scale = None
                continue
            assert param is not None
            if tuple(tensor.shape) != tuple(param.shape):
                raise ValueError(
                    f"shape mismatch for {name}: checkpoint {tuple(tensor.shape)} vs model "
                    f"{tuple(param.shape)}"
                )
            if kind == "fp4":
                if _mmap_src is not None and _is_host_offload(name):
                    # mmap 视图：不复制，页缓存可回收（anon 峰值大幅下降，见 perf-plan §25）
                    param.data = _mmap_src.get(name).view(_FP4)
                else:
                    param.copy_(tensor.view(_FP4))
            else:
                if _mmap_src is not None and _is_host_offload(name):
                    param.data = _mmap_src.get(name)
                else:
                    param.copy_(tensor)
            consumed.add(name)
            loaded += 1

        import os

        if os.environ.get("VLLM_DSV41_DEBUG"):
            # dtype-only: an earlier version also printed float().sum(), which allocates GBs on
            # 129280x5120 tensors and stalled the load
            probe = {
                "embed": self.model.embed.weight,
                "expert_w1": self.model.layers[0].ffn.experts[0].w1.weight,
                "expert_w1_scale": self.model.layers[0].ffn.experts[0].w1.scale,
                "gate": self.model.layers[0].ffn.gate.weight,
            }
            for nm, prm in probe.items():
                print(
                    "DSV41_DEBUG param",
                    nm,
                    tuple(prm.shape),
                    str(prm.dtype),
                    flush=True,
                )
        setattr(self, LOADED_ATTR, loaded)
        # vLLM profiles device memory right after load_model(); by then the S4 arrangement has to be in
        # force (dense on the card, experts + engram tables in host memory), see finalize_placement.
        try:
            self.finalize_placement(getattr(self, "_placement_device", "npu:0"))
        except Exception:  # noqa: BLE001 - a CPU-only box (CI, S1) has no npu to place onto
            pass
        return consumed

    # -- forward -----------------------------------------------------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Part of vLLM's ``VllmModelForTextGeneration`` protocol.

        ``isinstance(cls, VllmModelForTextGeneration)`` is a structural check over
        ``{forward, compute_logits, embed_input_ids}``; without this method the class fails it and
        ``ModelConfig`` rejects ``--runner generate`` ("This model does not support `--runner
        generate`"), regardless of the ``is_text_generation_model`` attribute.
        """
        weight = self.model.embed.weight
        return torch.nn.functional.embedding(input_ids.to(weight.device), weight)

    def forward(self, input_ids, positions=None, kv_caches=None, attn_metadata=None, **kwargs):
        """Single-sequence forward through the standalone stack (integration step A).

        vLLM's runner passes ``input_ids`` (``[num_tokens]``), ``positions``, the paged ``kv_caches``
        and ``attn_metadata``.  This implementation **ignores the paged KV entirely**: the stack keeps
        its own window / compressed-KV / index buffers in :class:`ModelState`, exactly as the verified
        S3/S4/S5 path does, so the only things taken from the runner are the token ids and the absolute
        positions.  A new request is recognised by ``positions[0] == 0`` and gets a fresh state.

        Returns the final hidden states, ``[num_tokens, hidden]``, which is what vLLM's sampler wants
        through :meth:`compute_logits`.
        """
        from tileinfer.models.dsv41.block import ModelState, transformer_forward

        ids = input_ids.reshape(1, -1)
        pos = int(positions.reshape(-1)[0].item()) if positions is not None else 0
        import os

        # P3 (concurrency): with --max-num-seqs >= 2 vLLM flattens several requests into one
        # [num_tokens] batch, so the single-sequence path above would treat them as one long
        # sequence.  Gated and OFF by default; when on we split by request (request_slices) and
        # only take the multi path if there really is more than one request.
        if os.environ.get("VLLM_DSV41_MULTI_REQ", "") not in ("", "0"):
            n_tokens = int(input_ids.reshape(-1).shape[0])
            from tileinfer.vllm.models.dsv41_batching import (
                is_single,
                request_slices,
                slices_from_counts,
            )
            from tileinfer.vllm.models.dsv41_req_ids import (
                current_req_counts,
                current_req_ids,
            )

            # 主判据是 runner 钩子给的请求数：attn_metadata 在我们的栈里可能是 None（§18），
            # 只靠它会误判成单序列、把多请求拼成一条序列而崩溃。
            _ids = current_req_ids()
            _slices = None
            if _ids and len(_ids) > 1:
                _slices = slices_from_counts(current_req_counts(), n_tokens)
            if _slices is None:
                _slices = request_slices(attn_metadata, n_tokens)
            if _slices is None and n_tokens > 0:
                # 钩子/attn_metadata 都拿不到时也走 slot 路径（整批当一条）：MULTI_REQ 打开后
                # **所有** state 都必须进 StateSlots，否则"单独 prefill → 并发 decode"会丢上下文。
                _slices = [(0, n_tokens)]
            # 单请求也走 _forward_multi：它的 state 必须和并发路径放在同一个 StateSlots 里，
            # 否则"先单独 prefill、再和其它请求一起 decode"时 state 只存在 `_stream_state`，
            # 多请求路径找不到它（100-prompt 质量门实测）。
            if _slices:
                return self._forward_multi(input_ids, positions, _slices, _ids)

        if os.environ.get("VLLM_DSV41_DEBUG"):
            print(
                "DSV41_DEBUG forward: ids",
                tuple(ids.shape),
                ids[0][:6].tolist(),
                "positions",
                positions.reshape(-1)[:6].tolist() if positions is not None else None,
                flush=True,
            )
        state = getattr(self, "_stream_state", None)
        import time as _time

        _t0 = _time.perf_counter()
        _cache_on = os.environ.get("VLLM_DSV41_PROMPT_CACHE", "") not in ("", "0")
        _block_on = os.environ.get("VLLM_DSV41_BLOCK_CACHE", "") not in ("", "0")
        _block_size = int(os.environ.get("VLLM_DSV41_BLOCK_SIZE", "512") or 512)
        _pkey = None
        _ptoks = None
        if _cache_on and pos == 0:
            _ptoks = tuple(int(x) for x in ids.flatten().tolist())
            _pkey = hash(_ptoks)
            if _block_on and len(_ptoks) > _block_size:
                _bsnap = self._block_lookup(_ptoks)
                if _bsnap is not None:
                    state = self._restore_block(_bsnap, len(_ptoks))
                    self._stream_state = state
                    _moe = self._runtime_moe()
                    _suffix = len(_ptoks) - len(_bsnap["tokens"])
                    _chunk = int(os.environ.get("VLLM_DSV41_PREFIX_CHUNK", "0") or 0)
                    if _suffix > 0 and _chunk > 0:
                        _lg, out = self._prefill_suffix_chunked(
                            ids, state, len(_bsnap["tokens"]), len(_ptoks), _moe, _chunk
                        )
                        self._last_logits = _lg
                        if os.environ.get("VLLM_DSV41_DEBUG"):
                            print(
                                "DSV41_BLOCK_CACHE hit block=%d suffix=%d"
                                % (len(_bsnap["tokens"]), _suffix),
                                flush=True,
                            )
                        return out
            _best = None
            for _row in getattr(self, "_prompt_cache", {}).values():
                _t = _row.get("tokens")
                if _t and len(_t) <= len(_ptoks) and _ptoks[: len(_t)] == _t:
                    if _best is None or len(_t) > len(_best["tokens"]):
                        _best = _row
            if _best is not None:
                import copy

                _p = len(_best["tokens"])
                if _p == len(_ptoks):
                    self._stream_state = copy.deepcopy(_best["state"])
                    self._last_logits = _best["logits"].clone()
                    out = _best["hidden"].clone()
                    if os.environ.get("VLLM_DSV41_DEBUG"):
                        print("DSV41_PROMPT_CACHE exact hit len=%d" % _p, flush=True)
                    return out
                # partial prefix: restore, then either chunked prefill (new path, default off) or the
                # verified token-by-token suffix.
                state = self._restore_prefix_state(_best["state"], len(_ptoks))
                self._stream_state = state
                _moe = self._runtime_moe()
                _suffix = len(_ptoks) - _p
                _chunk = int(os.environ.get("VLLM_DSV41_PREFIX_CHUNK", "0") or 0)
                if _chunk > 0 and _suffix > 1:
                    _lg, out = self._prefill_suffix_chunked(ids, state, _p, len(_ptoks), _moe, _chunk)
                    self._last_logits = _lg
                    if os.environ.get("VLLM_DSV41_DEBUG"):
                        print(
                            "DSV41_PROMPT_CACHE prefix hit prefix=%d suffix=%d chunk=%d"
                            % (_p, _suffix, _chunk),
                            flush=True,
                        )
                    return out
                _cap = {}
                _lg = None
                for _i in range(_p, len(_ptoks)):
                    _last = _i == len(_ptoks) - 1
                    _lg, _ = transformer_forward(
                        self.model, ids[:, _i : _i + 1], state, _i,
                        capture=_cap if _last else None, moe_fn=_moe,
                        skip_logits=not _last,
                    )
                self._last_logits = _lg
                out = _cap["h"][0] if _cap["h"].dim() == 3 else _cap["h"]
                if os.environ.get("VLLM_DSV41_DEBUG"):
                    print(
                        "DSV41_PROMPT_CACHE prefix hit prefix=%d suffix=%d"
                        % (_p, len(_ptoks) - _p),
                        flush=True,
                    )
                return out
        if pos == 0 or state is None:
            # allocate for what this request needs, not for the model's 1M ceiling: the standalone
            # stack (S3/S4/S5) runs with a small max_seq_len, and any state that reads this value
            # (rope tables, compressed-stream capacity) must match that configuration
            # capacity must also cover the first decode position (pos == prompt length)
            needed = pos + int(ids.shape[1]) + 64
            state = ModelState(
                cfg=self.dsv41_config,
                max_seq_len=max(1024, needed),
                bsz=1,
                device=str(self.model.embed.weight.device),
                engram_hash=self._engram_hash(),
            )
            self._stream_state = state
        _t1 = _time.perf_counter()
        capture: Dict[str, object] = {}
        _dbg = os.environ.get("VLLM_DSV41_DEBUG")
        cap_layers = {} if _dbg else None
        _moe_fn = self._runtime_moe()
        _t2 = _time.perf_counter()
        _timing_on = os.environ.get("VLLM_DSV41_DECODE_TIMING", "") not in ("", "0")
        _timings = {} if _timing_on else None
        if _cache_on and _block_on and _ptoks is not None and len(_ptoks) > _block_size:
            logits, out = self._prefill_blocks(ids, state, len(_ptoks), _moe_fn, _block_size)
            self._last_logits = logits
            import copy as _copy

            if getattr(self, "_prompt_cache", None) is None:
                self._prompt_cache = {}
            if len(self._prompt_cache) >= 2:
                self._prompt_cache.pop(next(iter(self._prompt_cache)))
            self._prompt_cache[_pkey] = {
                "tokens": _ptoks,
                "state": _copy.deepcopy(state),
                "hidden": out.detach().clone(),
                "logits": logits.detach().clone(),
            }
            return out
        logits, _extras = transformer_forward(
            self.model, ids, state, pos, capture=capture, moe_fn=_moe_fn,
            capture_layers=cap_layers, timings=_timings,
        )
        _t3 = _time.perf_counter()
        if _timing_on:
            print(
                "DSV41_DECODE_TIMING pos=%d fwd=%.0fms moe=%.0fms"
                % (pos, (_t3 - _t2) * 1e3, (_timings or {}).get("moe", 0.0) * 1e3),
                flush=True,
            )
        if _dbg:
            # P2 的 4.3 s 固定开销定位（docs/notes/dsv41-perf-p2-ttft.md §2）：把一次请求拆成
            # state 构造 / arena 与 MoE 闭包准备 / 前向三段
            print(
                "DSV41_TIMING state=%.0fms moe_setup=%.0fms fwd=%.0fms ids=%d pos=%d"
                % ((_t1 - _t0) * 1e3, (_t2 - _t1) * 1e3, (_t3 - _t2) * 1e3,
                   int(ids.shape[1]), pos),
                flush=True,
            )
        # ``transformer_forward`` already applies the final norm and the head (block.py:471-474), which
        # is exactly the path S3 verified against the oracle -- keep its logits instead of recomputing
        # them from the captured hidden state.
        self._last_logits = logits
        hidden = capture["h"]
        out = hidden[0] if hidden.dim() == 3 else hidden
        if _cache_on and _pkey is not None:
            # exact-prompt state cache (P4 prefix-cache first step): snapshot the post-prefill state.
            # Not a partial-prefix cache yet; only identical prompts hit.
            import copy as _copy

            if getattr(self, "_prompt_cache", None) is None:
                self._prompt_cache = {}
            if len(self._prompt_cache) >= 2:
                self._prompt_cache.pop(next(iter(self._prompt_cache)))
            self._prompt_cache[_pkey] = {
                "tokens": _ptoks,
                "state": _copy.deepcopy(state),
                "hidden": out.detach().clone(),
                "logits": logits.detach().clone(),
            }
        import os as _os

        if _os.environ.get("VLLM_DSV41_DEBUG"):
            # what we hand the runner, versus what compute_logits later receives: if these differ,
            # vLLM is transforming in between (that would explain the 1e4 vs 0.1 magnitude gap)
            print(
                "DSV41_DEBUG forward out:",
                tuple(out.shape),
                [round(float(v), 5) for v in out.flatten()[:4].tolist()],
                "norms",
                {
                    nm: round(float(prm.float().abs().mean()), 6)
                    for nm, prm in (
                        ("final", self.model.norm.weight),
                        ("l0_attn", self.model.layers[0].attn_norm.weight),
                        ("l0_ffn", self.model.layers[0].ffn_norm.weight),
                    )
                },
                flush=True,
            )
            # per-layer values to compare against the standalone reference
            # (tools/dsv41/s4_full_stack_device.py prints the same three lines)
            for _lid in sorted(cap_layers or {})[:3]:
                print(
                    "DSV41_DEBUG layer[%d][:4]" % _lid,
                    [round(float(v), 5) for v in cap_layers[_lid].flatten()[:4].tolist()],
                    flush=True,
                )
        return out

    def _snapshot_state(self, state, length: int):
        """Compact CPU snapshot of one prefix (block boundary), only the used slices."""
        layers = {}
        for lid, rt in state.layers.items():
            ratio = int(state.cfg.compress_ratio(lid))
            n = int(length) // ratio if ratio else 0
            d = {"window_kv": rt.window_kv.detach().cpu()}
            if getattr(rt, "compress_kv", None) is not None:
                d["compress_kv"] = rt.compress_kv[:, :n].detach().cpu()
            if getattr(rt, "index_k", None) is not None:
                d["index_k"] = rt.index_k[:, :n].detach().cpu()
            if getattr(rt, "kv_state", None) is not None:
                d["kv_state"] = rt.kv_state.detach().cpu()
                d["score_state"] = rt.score_state.detach().cpu()
            layers[lid] = d
        eh = getattr(state, "engram_hash", None)
        engram = None if eh is None else eh.cache[:, : int(length)].detach().cpu()
        return {"len": int(length), "layers": layers, "engram": engram}

    def _restore_block(self, snap, total_len: int):
        from tileinfer.models.dsv41.block import ModelState

        state = getattr(self, "_stream_state", None)
        if state is None or int(getattr(state, "max_seq_len", 0)) < int(total_len) + 64:
            state = ModelState(
                cfg=self.dsv41_config,
                max_seq_len=max(1024, int(total_len) + 64),
                bsz=1,
                device=str(self.model.embed.weight.device),
                engram_hash=self._engram_hash(),
            )
        for lid, d in snap["layers"].items():
            new = state.runtime(lid, self.dsv41_config.n_layers)
            new.window_kv.copy_(d["window_kv"].to(new.window_kv.device, non_blocking=True))
            if "compress_kv" in d and new.compress_kv is not None:
                n = min(d["compress_kv"].shape[1], new.compress_kv.shape[1])
                new.compress_kv[:, :n].copy_(d["compress_kv"][:, :n].to(new.compress_kv.device, non_blocking=True))
            if "index_k" in d and new.index_k is not None:
                n = min(d["index_k"].shape[1], new.index_k.shape[1])
                new.index_k[:, :n].copy_(d["index_k"][:, :n].to(new.index_k.device, non_blocking=True))
            if "kv_state" in d and new.kv_state is not None:
                new.kv_state.copy_(d["kv_state"].to(new.kv_state.device, non_blocking=True))
                new.score_state.copy_(d["score_state"].to(new.score_state.device, non_blocking=True))
        if snap.get("engram") is not None and getattr(state, "engram_hash", None) is not None:
            n = min(snap["engram"].shape[1], state.engram_hash.cache.shape[1])
            state.engram_hash.cache[:, :n].copy_(
                snap["engram"][:, :n].to(state.engram_hash.cache.device, non_blocking=True)
            )
        return state

    def _block_lookup(self, tokens):
        best = None
        for snap in getattr(self, "_block_cache", {}).values():
            t = snap["tokens"]
            if len(t) <= len(tokens) and tokens[: len(t)] == t:
                if best is None or len(t) > len(best["tokens"]):
                    best = snap
        return best

    def _prefill_blocks(self, ids, state, total: int, moe_fn, block: int):
        """Prefill in block-sized chunks, snapshotting a compact state after each block."""
        from tileinfer.models.dsv41.block import transformer_forward

        p = 0
        logits = None
        out = None
        while p < total:
            n = min(int(block), total - p)
            if n > 1 and n % 2:
                n -= 1
            if n <= 0:
                n = 1
            last = p + n >= total
            cap = {}
            logits, _ = transformer_forward(
                self.model,
                ids[:, p : p + n],
                state,
                p,
                capture=cap if last else None,
                moe_fn=moe_fn,
                skip_logits=not last,
            )
            p += n
            if last:
                h = cap["h"]
                out = h[0] if h.dim() == 3 else h
                break
            if p % int(block) == 0:
                snap = self._snapshot_state(state, p)
                snap["tokens"] = tuple(int(x) for x in ids[0, :p].tolist())
                if getattr(self, "_block_cache", None) is None:
                    self._block_cache = {}
                self._block_cache[(p, hash(snap["tokens"]))] = snap
                if len(self._block_cache) > 64:
                    self._block_cache.pop(next(iter(self._block_cache)))
        return logits, out

    def _prefill_suffix_chunked(self, ids, state, start: int, total: int, moe_fn, chunk: int):
        """Run the suffix after a restored prefix with ratio-aligned chunks.

        All compression ratios in this model divide 2, so aligning the first chunk to an even
        ``start_pos`` lets ``compressor_forward``/``indexer_forward`` take their ``seqlen > 1``
        branches.  Chunks are even-sized; a trailing single token falls back to the normal decode
        shape.
        """
        import time as _time
        import os as _os_env

        from tileinfer.models.dsv41.block import transformer_forward

        _t0 = _time.perf_counter()
        logits = None
        out = None
        if total - start <= 32:
            # A short suffix is cheaper as decode-shaped steps: prefill MoE would fan out to
            # topk * suffix distinct experts per layer, while decode touches only topk.
            cap = {}
            for i in range(int(start), int(total)):
                last = i == total - 1
                logits, _ = transformer_forward(
                    self.model,
                    ids[:, i : i + 1],
                    state,
                    i,
                    capture=cap if last else None,
                    moe_fn=moe_fn,
                    skip_logits=not last,
                )
            h = cap["h"]
            out = h[0] if h.dim() == 3 else h
            return logits, out
        cap = {}
        p = int(start)
        if p % 2 and p < total:
            logits, _ = transformer_forward(
                self.model, ids[:, p : p + 1], state, p, moe_fn=moe_fn, skip_logits=True
            )
            p += 1
        while p < total:
            n = min(int(chunk), total - p)
            if n > 1 and n % 2:
                n -= 1
            if n <= 0:
                n = 1
            last = p + n >= total
            cap = {}
            logits, _ = transformer_forward(
                self.model,
                ids[:, p : p + n],
                state,
                p,
                capture=cap if last else None,
                moe_fn=moe_fn,
                skip_logits=not last,
            )
            if last:
                h = cap["h"]
                out = h[0] if h.dim() == 3 else h
            p += n
        if _os_env.environ.get("VLLM_DSV41_DEBUG"):
            print(
                "DSV41_PREFIX_CHUNK start=%d total=%d elapsed=%.0fms"
                % (start, total, (_time.perf_counter() - _t0) * 1e3),
                flush=True,
            )
        return logits, out

    def _restore_prefix_state(self, cached, total_len: int):
        """Restore a cached post-prefill state, growing buffers for a longer prompt if needed."""
        import copy

        from tileinfer.models.dsv41.block import ModelState

        if int(getattr(cached, "max_seq_len", 0)) >= int(total_len) + 64:
            return copy.deepcopy(cached)
        state = ModelState(
            cfg=self.dsv41_config,
            max_seq_len=max(1024, int(total_len) + 64),
            bsz=1,
            device=str(self.model.embed.weight.device),
            engram_hash=self._engram_hash(),
        )
        for lid, old in cached.layers.items():
            new = state.runtime(lid, self.dsv41_config.n_layers)
            new.window_kv.copy_(old.window_kv)
            for attr in ("compress_kv", "index_k"):
                ov, nv = getattr(old, attr, None), getattr(new, attr, None)
                if ov is not None and nv is not None:
                    n = min(ov.shape[1], nv.shape[1])
                    nv[:, :n].copy_(ov[:, :n])
            if getattr(old, "kv_state", None) is not None:
                new.kv_state.copy_(old.kv_state)
                new.score_state.copy_(old.score_state)
        old_eh, new_eh = getattr(cached, "engram_hash", None), getattr(state, "engram_hash", None)
        if old_eh is not None and new_eh is not None:
            n = min(old_eh.cache.shape[1], new_eh.cache.shape[1])
            new_eh.cache[:, :n].copy_(old_eh.cache[:, :n])
        return state

    def _engram_hash(self):
        """The n-gram hash state the stack needs; without it layers 1 and 14 skip their engram.

        ``transformer_forward`` only applies the engram contribution when ``state.engram_hash`` is set,
        and the hash ids come from the tokenizer (the tables are looked up by hash, not by weight
        arithmetic).  Leaving it ``None`` silently drops those two layers -- which is exactly the kind
        of difference that shows up as a wrong first token.
        """
        if getattr(self, "_engram_hash_state", None) is not None:
            return self._engram_hash_state
        model_dir = getattr(self, "_model_dir", "") or ""
        if not model_dir:
            return None
        try:
            from transformers import AutoTokenizer

            from tileinfer.models.dsv41.engram import EngramHash

            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            # the engram rolling cache is indexed by absolute position.  It must cover not only
            # --max-model-len but also the compile/profiling dummy batch (vLLM warms up with 8192
            # tokens when graph capture is enabled); 4096 was too small and crashed startup.
            _seq = max(
                int(self.dsv41_config.max_seq_len),
                int(getattr(self, "_max_model_len", 0) or 0),
                int(__import__("os").environ.get("VLLM_DSV41_ENGRAM_CACHE", "8192")),
            )
            self._engram_hash_state = EngramHash(
                self.dsv41_config,
                tokenizer=tokenizer,
                max_seq_len=_seq,
                device="cpu",
            )
        except Exception:  # noqa: BLE001 - the stack tolerates a missing hash provider
            self._engram_hash_state = None
        return self._engram_hash_state

    def _forward_multi(self, input_ids, positions, slices, req_ids=None):
        """并发路径：按请求逐段前向，返回 ``[num_reqs, hidden]``（每个请求的最后一个 token）。

        只在 ``VLLM_DSV41_MULTI_REQ=1`` 且批次里确实有 >1 个请求时走到这里，
        所以单序列（已验证）路径完全不受影响。

        state 用 :class:`StateSlots` 按请求 id 复用；id 由 ``dsv41_req_ids`` 的 runner 钩子提供，
        取不到时退化为"按片段序号"，那会让 state 每步重建（慢但正确）。
        """
        from tileinfer.models.dsv41.block import ModelState, transformer_forward
        from tileinfer.vllm.models.dsv41_batching import run_batched
        from tileinfer.vllm.models.dsv41_req_ids import current_req_ids
        from tileinfer.vllm.models.dsv41_state_slots import StateSlots

        ids_flat = input_ids.reshape(-1)
        pos_flat = positions.reshape(-1) if positions is not None else None
        req_ids = req_ids or current_req_ids() or []
        # 诊断（§19）：只打印最初 3 次，用于判断钩子是否真的生效。
        # hooked=False ⇒ 补丁没装到实际调用方；hooked=True 且 req_ids=[] ⇒ 装上了但没被调用。
        _dbg_n = getattr(self, "_multi_dbg_n", 0)
        if _dbg_n < 3:
            self._multi_dbg_n = _dbg_n + 1
            print(
                "DSV41_MULTI slices=%s req_ids=%s hooked=%s"
                % (slices, req_ids, getattr(self, "_req_ids_hooked", None)),
                flush=True,
            )

        # 每个新槽位按"该请求真实需要的长度"分配（**不要**用模型的 1M 上限）。
        # 注意 _want 必须在 if 之外定义：槽位已存在时这个函数仍要走 acquire，需要它可见。
        # 必须放在 self 上：factory 只创建一次，闭包若捕获**第一次调用**的局部 _want，后续
        # 4K/32K 请求仍会拿到 2048 的 RoPE 表（P4 实测 state_max=2048 而 want=4221）。
        if getattr(self, "_want_seq", None) is None:
            self._want_seq = {"n": 2048}
        _want = self._want_seq

        if getattr(self, "_slots", None) is None:
            device = str(self.model.embed.weight.device)

            def _factory(rid):
                return ModelState(
                    cfg=self.dsv41_config,
                    max_seq_len=int(self._want_seq["n"]),
                    bsz=1,
                    device=device,
                    engram_hash=self._engram_hash(),
                )

            # 槽位数与并发度无关地留足：请求在 prefill 后可能**暂时不被调度**（例如另一个请求
            # 正在做 prefill），若按"本步 batch"裁剪就会把它的 state 删掉，下一步 pos>0 重建
            # 出来的 state 没有 layer-20 的 index_k，直接 TypeError（100-prompt 质量门实测）。
            # 256 槽 ≈ 数 GiB（短上下文），100 prompt 的整轮不会触发淘汰。
            self._slots = StateSlots(_factory, max_slots=max(256, 4 * len(slices)))

        # 状态认领：优先用 runner 钩子的 req id（与 num_scheduled_tokens 同序，计数可与片段
        # 长度互相校验）；只有校验不过时才退回"绝对位置链"。
        # 为什么不能只用位置链：warm-up/已结束请求的 state 会留在池里，它们的 `_next_pos` 与
        # 新请求的 pos 可能相同，于是第 4 个 token 开始会读到**另一个序列的上下文**
        # （100-prompt 质量门实测：前 3 个 token 的 top-5 逐位相同，第 4 个开始完全发散）。
        self._step = getattr(self, "_step", 0) + 1
        _used = set()
        # 已结束的请求不会再出现在调度里；保留超过 64 步未使用的 state 只会吃 HBM/主机内存
        # （100-prompt 单请求长跑在 ~67 个请求后 engine 原生退出，见 §28）。每 16 步清一次，
        # 避免每步 free/realloc 造成的分配器抖动。
        _prune_every = int(__import__("os").environ.get("VLLM_DSV41_PRUNE_EVERY", "8"))
        if _prune_every > 0 and len(self._slots) > 32 and self._step % _prune_every == 0:
            keep = [
                k
                for k in self._slots.ids()
                if getattr(self._slots.get(k), "_last_step", 0) >= self._step - 32
            ]
            self._slots.prune(keep)

        from tileinfer.vllm.models.dsv41_req_ids import current_req_counts, current_req_entries

        _entries = list(current_req_entries() or [])
        _counts = current_req_counts()
        _lens = [int(e) - int(s) for s, e in slices]
        _ord_of_start = {int(s): i for i, (s, _e) in enumerate(slices)}
        _keys: List[str] = []
        for _i, (_s, _e) in enumerate(slices):
            _abs = int(pos_flat[int(_s)]) if pos_flat is not None else int(_s)
            _n = int(_e) - int(_s)
            _hit = None
            for _j, (_rid, _start, _len) in enumerate(_entries):
                if int(_start) == _abs and int(_len) == _n:
                    _hit = _j
                    break
            if _hit is None:
                _keys.append(f"seq{self._step}_{_i}")
            else:
                _rid, _start, _len = _entries.pop(_hit)
                _keys.append(str(_rid))
        if __import__("os").environ.get("VLLM_DSV41_MULTI_REQ_DEBUG"):
            print(
                "DSV41_SLOTS keys=%s entries_used=%s counts=%s lens=%s slices=%s"
                % (_keys, _keys, _counts, _lens, slices),
                flush=True,
            )

        def acquire(start, end):
            # ``start``/``end`` are offsets into vLLM's **flat token batch**, not absolute
            # positions: a cached decode request can sit at flat offset 0 while its absolute
            # position is 5.
            start, end = int(start), int(end)
            abs_start = int(pos_flat[start]) if pos_flat is not None else start
            try:
                _pos_end = int(pos_flat[end - 1]) + 1
            except Exception:  # noqa: BLE001 - some runners pass positions=[0] for prefill
                _pos_end = abs_start + (end - start)
            # a prefill's `positions` can be all zeros in vLLM; never allocate a state shorter than
            # the slice it must hold (P4: 4221-token prompt got a 2048 RoPE table and crashed).
            abs_end = max(_pos_end, abs_start + (end - start))
            i = _ord_of_start.get(start, -1)
            key = _keys[i] if 0 <= i < len(_keys) else f"seq{self._step}_x{start}"
            if abs_start == 0:
                # 完整 prefill（chunked prefill 关闭）；新序列必须拿到全新 state。
                # 容量要覆盖**后续 decode 位置**：prompt N 个 token 之后第一步 decode 的 pos=N，
                # 因此不能只分配到 abs_end（P4 4K：cos 在 pos=4221 处为空）。
                _cap = abs_end + 64
                if self._max_model_len:
                    _cap = min(int(self._max_model_len), _cap)
                _want["n"] = max(2048, _cap)
                if __import__("os").environ.get("VLLM_DSV41_MULTI_REQ_DEBUG"):
                    print(
                        f"DSV41_ACQ abs_start=0 abs_end={abs_end} want={_want['n']} "
                        f"pos_len={0 if pos_flat is None else int(pos_flat.numel())}",
                        flush=True,
                    )
                if key in self._slots:
                    self._slots.release(key)
                st = self._slots.acquire(key)
                if st is None:
                    raise RuntimeError("StateSlots 已满：请降低 --max-num-seqs 或调大 max_slots")
                st._next_pos = abs_end
                st._last_step = self._step
                _used.add(key)
                return st
            st = self._slots.get(key)
            if st is not None and getattr(st, "_next_pos", -1) == abs_start:
                st._next_pos = abs_end
                st._last_step = self._step
                _used.add(key)
                return st
            if __import__("os").environ.get("VLLM_DSV41_MULTI_REQ_DEBUG"):
                print(
                    "DSV41_SLOT_MISS key=%s next=%s abs_start=%s slots=%s"
                    % (
                        key,
                        None if st is None else getattr(st, "_next_pos", None),
                        abs_start,
                        [
                            (k, getattr(self._slots.get(k), "_next_pos", None))
                            for k in self._slots.ids()
                        ],
                    ),
                    flush=True,
                )
            # 钩子不可用/异步覆盖时的兜底：位置链必须**唯一**
            cands = [
                k
                for k in self._slots.ids()
                if k not in _used
                and getattr(self._slots.get(k), "_next_pos", -1) == abs_start
            ]
            if len(cands) == 1:
                key = cands[0]
                st = self._slots.get(key)
                st._next_pos = abs_end
                st._last_step = self._step
                _used.add(key)
                return st
            raise RuntimeError(
                f"DSV41 state missing/ambiguous for absolute position {abs_start} "
                f"(flat {start}..{end}, candidates={cands[:4]})"
            )

        # 真批处理（P3 达标路径，§24）：所有片段都是"每请求 1 个 token"时，一次前向吃 bsz=R。
        # 默认关闭（N5）；开启后只有 decode 步走这里，prefill / 混合批仍走下面的逐请求路径，
        # 因此已验证的 S3/S4/S5 路径不受影响。
        import os as _os_batch

        if (
            _os_batch.environ.get("VLLM_DSV41_BATCHED_FWD", "") not in ("", "0")
            and len(slices) > 1
            and all(int(e) - int(s) == 1 for s, e in slices)
        ):
            starts = [int(s) for s, _e in slices]
            states = []
            for start, end in slices:
                st = acquire(start, end)
                if st is None:
                    raise RuntimeError(
                        "StateSlots 已满：无法为请求分配 state，请降低 --max-num-seqs 或调大 max_slots"
                    )
                states.append(st)
            from tileinfer.models.dsv41.batched_decode import transformer_forward_decode

            rows = torch.as_tensor(starts, dtype=torch.long, device=ids_flat.device)
            batch_pos = (
                pos_flat.index_select(0, rows)
                if pos_flat is not None
                else torch.zeros(len(slices), dtype=torch.long, device=ids_flat.device)
            )
            logits, hidden = transformer_forward_decode(
                self.model,
                ids_flat.index_select(0, rows),
                batch_pos.tolist(),
                states,
                cfg=self.dsv41_config,
                moe_fn=self._runtime_moe_batched(),
            )
            self._last_logits = None
            self._multi_logits = logits.reshape(len(slices), -1)
            self._multi_rows = len(slices)
            if getattr(self, "_multi_dbg_n", 0) < 3:
                self._multi_dbg_n = getattr(self, "_multi_dbg_n", 0) + 1
                print(
                    "DSV41_BATCHED_FWD R=%d pos=%s"
                    % (len(slices), [int(p) for p in batch_pos.tolist()]),
                    flush=True,
                )
            return hidden.reshape(len(slices), -1)

        _req_logits = []

        def forward_one(state, ids_slice, pos_slice):
            start = int(pos_slice[0]) if len(pos_slice) else 0
            if __import__("os").environ.get("VLLM_DSV41_MULTI_REQ_DEBUG"):
                _r0 = state.runtime(0, self.dsv41_config.n_layers)
                print(
                    f"DSV41_FWD state_max={state.max_seq_len} cos={tuple(_r0.cos.shape)} "
                    f"len={int(ids_slice.numel())} start={start}",
                    flush=True,
                )
            cap = {}
            # 进度标记：并发路径曾在第 3 步后**原生崩溃**（无 Python traceback），
            # 靠这对标记可以判断死在哪个请求、哪一步（见 perf-plan §25）。
            import os as _os3

            _mark = _os3.environ.get("VLLM_DSV41_MULTI_TRACE")
            if _mark:
                print(f"DSV41_MULTI_REQ start start={start} len={int(ids_slice.numel())}",
                      flush=True)
            # transformer_forward 的第一个返回值是 **logits**（单请求路径就是这么用的：
            # `logits, _extras = transformer_forward(...)`），hidden 在 capture["h"] 里。
            # 之前把返回值当 hidden 传给 run_batched，导致 logits（[1, vocab]）被当成 hidden 使用，
            # 真机表现就是 "5120 vs 129280" 形状不匹配。
            logits, _extra = transformer_forward(
                self.model, ids_slice.reshape(1, -1), state, start, capture=cap,
                moe_fn=self._runtime_moe(),
            )
            _req_logits.append(logits.reshape(-1))
            h = cap["h"]
            return h[0] if h.dim() == 3 else h

        _per_req, last_tok = run_batched(
            slices, ids_flat, pos_flat if pos_flat is not None else ids_flat, acquire, forward_one
        )

        # 每请求的最后一个 hidden 拼成 [num_reqs, hidden]，logits 由 compute_logits 统一取用
        hidden = torch.stack([t.reshape(-1) for t in last_tok], dim=0)
        self._last_logits = None          # 多请求下这个捷径不适用（§14）
        self._multi_rows = int(hidden.shape[0])
        # logits 直接来自 transformer_forward（见 forward_one），与单请求路径同源
        self._multi_logits = (
            torch.stack(_req_logits, dim=0) if _req_logits else None
        )
        return hidden

    def _runtime_moe(self):
        """The streaming/batched MoE bound to this instance's arena (S4 in a vLLM process).

        Without this, ``transformer_forward`` falls back to ``moe_forward``, which assumes the expert
        weights live on the same device as the activations -- false here by design: they stay in host
        memory and are paged in (that is what keeps the card at 12.22 GiB while the checkpoint is
        475.25 GiB).  The first forward vLLM runs is the KV-init warm-up, so this path is hit during
        start-up, not only at request time.
        """
        fn = getattr(self, "_moe_fn", None)
        if fn is not None:
            return fn
        from tileinfer.models.dsv41.block import (
            moe_forward_batched,
            moe_forward_streamed,
            moe_forward_streamed_grouped,
        )
        from tileinfer.models.dsv41.expert_stream import ExpertArena, expert_specs

        device = str(self.model.embed.weight.device)
        # Arena capacity decides how many experts survive between steps.  With 384 slots and ~12
        # distinct experts per layer the previous step's working set is already evicted by the time
        # the next step asks for it (measured 40% hit rate); a larger arena trades HBM for H2D.
        _slots = int(
            __import__("os").environ.get(
                "VLLM_DSV41_ARENA_SLOTS", max(int(self.dsv41_config.n_routed_experts), 8)
            )
        )
        arena = ExpertArena(
            specs=expert_specs(self.model.layers[0].ffn.experts[0]),
            n_slots=_slots,
            pin=True,
        )
        for index, block in enumerate(self.model.layers):
            block.ffn._layer_id_for_stream = index

        def dispatch(moe, x, image_mask=None):
            # Multi-token (prefill): the streamed path with the device-accumulator + merged w1/w3
            # variants is 41-43% faster than the host-accumulator default (measured A/B in
            # docs/notes/dsv41-perf-moe-ab.md).
            #
            # Single-token (decode): `moe_forward_batched` folds the six routed experts into one
            # batched GEMM.  Measured A/B on the vLLM path, differencing with ignore_eos and the
            # actual completion_tokens (docs/notes/dsv41-perf-moe-ab.md §8-§9):
            #     default 697.7 ms/token  vs  batched 592.4 ms/token  ->  -15.1%
            # with both arms passing the `Paris` gate.  Default ON; set DSV41_DECODE_BATCHED=0 to
            # fall back to the streamed path.
            layer_id = getattr(moe, "_layer_id_for_stream", 0)
            rows = x.numel() // x.shape[-1]
            import os as _osmod   # 闭包里看不到 forward 的局部 import，必须自己导

            if rows == 1 and _osmod.environ.get("DSV41_DECODE_MOE", "") == "grouped":
                # decode 也用 grouped（w1/w3 合并成一次 batched GEMM）；A/B 用开关隔离
                return moe_forward_streamed_grouped(
                    moe, x, layer_id, arena, device=device, image_mask=image_mask
                )
            if rows == 1 and _osmod.environ.get("DSV41_DECODE_BATCHED", "1") not in ("", "0"):
                return moe_forward_batched(
                    moe, x, layer_id, arena, device=device, image_mask=image_mask
                )
            return moe_forward_streamed(
                moe, x, layer_id, arena, device=device, image_mask=image_mask
            )

        self._arena = arena
        self._moe_fn = dispatch
        return dispatch

    def _runtime_moe_batched(self):
        """MoE bound to the arena for the **batched** decode path (rows = requests).

        ``moe_forward_batched`` is the one-token fast path and rejects ``rows != 1``; the grouped
        streamed variant is the one that takes R rows through a single expert-grouped GEMM, which is
        what makes true batching pay off on the MoE side (measured prefill -24.5% for the same
        implementation, see docs/notes/dsv41-perf-moe-ab.md).
        """
        self._runtime_moe()
        from tileinfer.models.dsv41.block import (
            moe_forward_batched,
            moe_forward_streamed,
            moe_forward_streamed_grouped,
        )

        arena = self._arena
        device = str(self.model.embed.weight.device)

        if __import__("os").environ.get("VLLM_DSV41_BATCHED_MOE", "") == "per_row":
            # 质量门诊断：按行调用单请求 MoE，使批量臂与串行臂的 MoE 数学完全一致，
            # 从而把"批量 attention"的数值影响与"grouped MoE 形状"分开（慢，仅诊断）。
            def dispatch(moe, x, image_mask=None):
                layer_id = getattr(moe, "_layer_id_for_stream", 0)
                outs = [
                    moe_forward_batched(
                        moe, x[r : r + 1], layer_id, arena, device=device, image_mask=image_mask
                    )
                    for r in range(int(x.shape[0]))
                ]
                return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]

            return dispatch

        def dispatch(moe, x, image_mask=None):
            return moe_forward_streamed_grouped(
                moe, x, getattr(moe, "_layer_id_for_stream", 0), arena,
                device=device, image_mask=image_mask,
            )

        return dispatch

    def compute_logits(self, hidden_states, **kwargs):
        """``logits_from_hidden`` on the runner's hidden states, keeping the batch dim.

        vLLM's sampler does ``logits.size(1)`` (``gpu_model_runner._dummy_sampler_run``), so the return
        value must be ``[num_tokens, vocab]`` -- a squeezed ``[vocab]`` fails with
        "Dimension out of range (expected to be in range of [-1, 0], but got 1)".
        """
        from tileinfer.models.dsv41.block import logits_from_hidden

        # 多请求路径（§14/§18）：_forward_multi 已经逐请求算好 logits，直接用。
        _multi = getattr(self, "_multi_logits", None)
        if _multi is not None:
            self._multi_logits = None
            out = _multi

        last = None if _multi is not None else getattr(self, "_last_logits", None)
        if _multi is not None:
            pass
        elif last is not None:
            # the same tensor the standalone stack samples from (block.py:474)
            out = last.reshape(1, -1)
        else:
            # logits_from_hidden takes the block layout [b, s, hidden] and keeps the last token.
            # Multi-request path (§14): every row is already a distinct request's last token, so the
            # batch dimension must be preserved instead of being forced to 1.
            rows = int(hidden_states.shape[0])
            if rows > 1 and getattr(self, "_multi_rows", 0) == rows:
                x = hidden_states.reshape(1, rows, hidden_states.shape[-1])
                out = logits_from_hidden(self.model, x).reshape(rows, -1)
            else:
                x = hidden_states.reshape(1, -1, hidden_states.shape[-1])
                out = logits_from_hidden(self.model, x)
        import os

        if os.environ.get("VLLM_DSV41_DEBUG"):
            import torch as _t

            vals, idx = _t.topk(out.flatten().float(), 5)
            print(
                "DSV41_DEBUG compute_logits: hidden",
                tuple(hidden_states.shape),
                "h4",
                [round(float(v), 5) for v in hidden_states.flatten()[:4].tolist()],
                "top5",
                list(zip(idx.tolist(), [round(float(v), 3) for v in vals.tolist()])),
                flush=True,
            )
        return out


def register_models(registry=None) -> Dict[str, object]:
    """Register the architecture with vLLM's ``ModelRegistry``; returns a status dict.

    Imported lazily so this module stays importable on a CPU box with no vLLM installed (CI and the
    S1 loader take that path).
    """
    info: Dict[str, object] = {
        "arch": MODEL_ARCH,
        "class_path": "tileinfer.vllm.models.dsv41:DeepseekV41ForCausalLM",
        "registered": False,
        "detail": "",
    }
    try:
        if registry is None:
            from vllm.model_executor.models.registry import ModelRegistry

            registry = ModelRegistry
        registry.register_model(
            MODEL_ARCH, "tileinfer.vllm.models.dsv41:DeepseekV41ForCausalLM"
        )
        info["registered"] = True
        info["detail"] = f"registered {MODEL_ARCH} -> tileinfer.vllm.models.dsv41"
    except Exception as exc:  # noqa: BLE001 - a plugin step must never abort the boot
        info["detail"] = f"registration failed: {exc!r}"
    return info
