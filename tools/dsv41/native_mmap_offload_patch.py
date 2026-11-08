#!/usr/bin/env python3
"""Apply the reclaimable-mmap + selective-expert patches to a native install.

The target environment is the isolated vLLM 0.30 + vllm-ascend 0.30 venv used
for the DeepSeek-V4.1-Flash native comparison.  Every patch is idempotent and
asserts on its anchor, so a partially patched tree fails loudly instead of
silently running stale code.

Usage:
    <venv>/bin/python tools/dsv41/native_mmap_offload_patch.py [--site-packages DIR]
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent

MMAP_HEADER = "# --- reclaimable mmap CPU weights (TileInfer patch) ---"
HOOK_HEADER = "# --- selective expert load hook (TileInfer patch) ---"


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    if new in text:
        print(f"[skip] {what}: already patched")
        return text
    if old not in text:
        raise SystemExit(f"[fail] {what}: anchor not found")
    print(f"[patch] {what}")
    return text.replace(old, new, 1)


def patch_vllm_prefetch(sp: pathlib.Path) -> None:
    path = sp / "vllm/model_executor/offloader/prefetch.py"
    text = path.read_text()

    old_init = """        self._param_deleted: bool = False

        # Offload to CPU immediately to free GPU memory during model loading
        self._offload_to_cpu_internal()
"""
    new_init = f"""        self._param_deleted: bool = False
        {MMAP_HEADER}
        self._cpu_store = None

        # Offload to CPU immediately to free GPU memory during model loading
        self._offload_to_cpu_internal()
"""
    text = _replace_once(text, old_init, new_init, "vllm _CpuParamOffloader._cpu_store")

    old_offload = """        param = self._param
        pin_memory = should_pin_memory()

        # Create pinned CPU storage and copy current GPU data
        self._cpu_storage = torch.empty_strided(
            size=param.data.size(),
            stride=param.data.stride(),
            dtype=param.data.dtype,
            layout=param.data.layout,
            device="cpu",
            pin_memory=pin_memory,
        )
        self._cpu_storage.copy_(param.data)
"""
    new_offload = f"""        param = self._param
        {MMAP_HEADER}
        from vllm.model_executor.offloader.mmap_store import MmapParamStore

        self._cpu_store = MmapParamStore.create(self._module, self._param_name, param.data)
        if self._cpu_store is not None:
            self._cpu_storage = self._cpu_store.tensor
            self._cpu_storage.copy_(param.data)
            self._cpu_store.flush()
        else:
            pin_memory = should_pin_memory()
            # Create pinned CPU storage and copy current GPU data
            self._cpu_storage = torch.empty_strided(
                size=param.data.size(),
                stride=param.data.stride(),
                dtype=param.data.dtype,
                layout=param.data.layout,
                device="cpu",
                pin_memory=pin_memory,
            )
            self._cpu_storage.copy_(param.data)
"""
    text = _replace_once(text, old_offload, new_offload, "vllm _offload_to_cpu_internal")

    old_update = """        param = self._param

        if param.data.device.type == "cpu":
"""
    new_update = f"""        param = self._param

        {MMAP_HEADER}
        if self._cpu_store is not None:
            src = param.data
            if src.data_ptr() == self._cpu_storage.data_ptr():
                return
            if src.numel() != self._cpu_storage.numel() or src.dtype != self._cpu_storage.dtype:
                from vllm.model_executor.offloader.mmap_store import MmapParamStore

                replacement = MmapParamStore.create(self._module, self._param_name, src)
                if replacement is not None:
                    self._cpu_store.release()
                    self._cpu_store = replacement
                    self._cpu_storage = replacement.tensor
                else:
                    self._cpu_store = None
                    self._cpu_storage = None
            if self._cpu_store is not None:
                self._cpu_storage.copy_(src)
                self._cpu_store.flush()
                return

        if param.data.device.type == "cpu":
"""
    text = _replace_once(text, old_update, new_update, "vllm _update_cpu_storage_from_param")
    path.write_text(text)


def patch_vllm_moe_runner(sp: pathlib.Path) -> None:
    path = sp / "vllm/model_executor/layers/fused_moe/runner/moe_runner.py"
    text = path.read_text()
    old = """            topk_weights, topk_ids = self.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_indices_dtype=self._quant_method.topk_indices_dtype,
                input_ids=input_ids,
            )

            fused_out = self.routed_experts.forward_modular(
"""
    new = f"""            topk_weights, topk_ids = self.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_indices_dtype=self._quant_method.topk_indices_dtype,
                input_ids=input_ids,
            )
            {HOOK_HEADER}
            from vllm.model_executor.offloader.base import get_offloader

            _maybe_load_selected = getattr(get_offloader(), "maybe_load_selected_experts", None)
            if _maybe_load_selected is not None:
                _maybe_load_selected(topk_ids)

            fused_out = self.routed_experts.forward_modular(
"""
    text = _replace_once(text, old, new, "vllm MoE selective hook")
    path.write_text(text)


def patch_ascend_prefetch(sp: pathlib.Path) -> None:
    path = sp / "vllm_ascend/model_executor/offloader/prefetch.py"
    text = path.read_text()

    old_alloc = "            host_nz = torch.empty(nbytes, dtype=torch.uint8, pin_memory=should_pin_memory())\n"
    new_alloc = f"""            {MMAP_HEADER}
            try:
                host_nz = torch.empty(nbytes, dtype=torch.uint8, pin_memory=should_pin_memory())
            except RuntimeError as exc:
                logger.warning(
                    "[NPUPrefetchOffloader] pinned host-NZ allocation failed (%s); "
                    "falling back to pageable memory for %s",
                    exc,
                    name,
                )
                host_nz = torch.empty(nbytes, dtype=torch.uint8)
"""
    text = _replace_once(text, old_alloc, new_alloc, "ascend pinned host-NZ fallback")

    old_drop = """    def drop_nd_storage(self) -> None:
        \"\"\"host-NZ 就绪后丢弃 ND 侧存储（host ND 与 device ND buffer 引用）。\"\"\"
        for offloader in self._param_offloaders.values():
            offloader._cpu_storage = None
            offloader._gpu_buffer = None
"""
    new_drop = f"""    def drop_nd_storage(self) -> None:
        \"\"\"host-NZ 就绪后丢弃 ND 侧存储（host ND 与 device ND buffer 引用）。\"\"\"
        {MMAP_HEADER}
        for offloader in self._param_offloaders.values():
            offloader._cpu_storage = None
            offloader._gpu_buffer = None
            store = getattr(offloader, "_cpu_store", None)
            if store is not None:
                store.release()
                offloader._cpu_store = None
"""
    text = _replace_once(text, old_drop, new_drop, "ascend drop ND mmap store")
    path.write_text(text)


def patch_ascend_factory(sp: pathlib.Path) -> None:
    path = sp / "vllm_ascend/model_executor/offloader/base.py"
    text = path.read_text()
    if "VLLM_ASCEND_OFFLOAD_SELECTIVE_TOKENS" not in text:
        if not text.startswith("import os\n"):
            text = "import os\n\n" + text
            print("[patch] ascend factory: import os")
        old = """            offload_params=prefetch.offload_params,
            mode="cpu",
        )
"""
        new = """            offload_params=prefetch.offload_params,
            mode="cpu",
            selective_token_threshold=int(os.getenv("VLLM_ASCEND_OFFLOAD_SELECTIVE_TOKENS", "0")),
        )
"""
        text = _replace_once(text, old, new, "ascend factory selective threshold")
    else:
        print("[skip] ascend factory: already patched")
    path.write_text(text)


def install_store(sp: pathlib.Path) -> None:
    src = HERE / "native_mmap_store.py"
    dst = sp / "vllm/model_executor/offloader/mmap_store.py"
    shutil.copyfile(src, dst)
    print(f"[copy] {src.name} -> {dst}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-packages", default="")
    args = ap.parse_args()
    if args.site_packages:
        sp = pathlib.Path(args.site_packages).resolve()
    else:
        import vllm

        sp = pathlib.Path(vllm.__file__).resolve().parent.parent
    print(f"[info] site-packages = {sp}")
    install_store(sp)
    patch_vllm_prefetch(sp)
    patch_vllm_moe_runner(sp)
    patch_ascend_prefetch(sp)
    patch_ascend_factory(sp)
    print("[done] native mmap + selective patches applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
