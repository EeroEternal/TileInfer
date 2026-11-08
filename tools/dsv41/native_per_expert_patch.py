#!/usr/bin/env python3
"""Per-expert NZ cast + pageable-mmap Engram for the native vLLM-ascend stack.

The stock host-NZ mode needs a second full copy of the expert weights on the
host (543 GB for V4.1-Flash), which does not fit next to the 196 GB Engram
tables even when both are file-backed.

This patch instead keeps only the reclaimable ND mmap store and casts the
**selected experts** ND->NZ on the device at decode time (verified byte
identical to a slice of the full [E, M, K] cast), copying each converted expert
into the persistent NZ buffer with an ACL D2D copy.  Prefill keeps the existing
full-layer path.

It also makes the pageable Engram tables file-backed so their 196 GB are page
cache rather than anonymous pinned memory.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    if new in text:
        print(f"[skip] {what}: already patched")
        return text
    if old not in text:
        raise SystemExit(f"[fail] {what}: anchor not found")
    print(f"[patch] {what}")
    return text.replace(old, new, 1)


def patch_per_expert(sp: pathlib.Path) -> None:
    path = sp / "vllm_ascend/model_executor/offloader/prefetch.py"
    text = path.read_text()

    old = '''# host 侧直接持有 NZ 字节（跳过每 forward 的整块 ND->NZ cast），默认关闭。
_OFFLOAD_HOST_NZ = int(os.getenv("VLLM_ASCEND_OFFLOAD_HOST_NZ", "0")) > 0
'''
    new = '''# host 侧直接持有 NZ 字节（跳过每 forward 的整块 ND->NZ cast），默认关闭。
_OFFLOAD_HOST_NZ = int(os.getenv("VLLM_ASCEND_OFFLOAD_HOST_NZ", "0")) > 0
# 只转选中专家的 ND->NZ（每个专家 0.2 ms），避免 host NZ 的第二份 543 GB。
_OFFLOAD_PER_EXPERT_NZ = int(os.getenv("VLLM_ASCEND_OFFLOAD_PER_EXPERT_NZ", "0")) > 0
'''
    text = _replace_once(text, old, new, "per-expert env switch")

    old = '''        if _OFFLOAD_HOST_NZ:
            # 分两遍：先把每层权重转成 host 侧 NZ 字节（每次只留一块临时 NZ），
            # 再为每个 (param, slot) 建持久 NZ buffer。避免同时多占两份整块显存。
            prepare_t0 = time.perf_counter()
            for offloader in self.module_offloaders:
                offloader.prepare_host_nz()
            for offloader in self.module_offloaders:
                offloader.create_nz_buffers()
            logger.info_once(
                f"[NPUPrefetchOffloader] host-NZ cache ready in {time.perf_counter() - prepare_t0:.1f}s "
                f"(NZ buffers: {self.buffer_pool.total_nz_bytes / 1e9:.4f} GB)"
            )
'''
    new = '''        if _OFFLOAD_HOST_NZ:
            # 分两遍：先把每层权重转成 host 侧 NZ 字节（每次只留一块临时 NZ），
            # 再为每个 (param, slot) 建持久 NZ buffer。避免同时多占两份整块显存。
            prepare_t0 = time.perf_counter()
            for offloader in self.module_offloaders:
                offloader.prepare_host_nz()
            for offloader in self.module_offloaders:
                offloader.create_nz_buffers()
            logger.info_once(
                f"[NPUPrefetchOffloader] host-NZ cache ready in {time.perf_counter() - prepare_t0:.1f}s "
                f"(NZ buffers: {self.buffer_pool.total_nz_bytes / 1e9:.4f} GB)"
            )
        elif _OFFLOAD_PER_EXPERT_NZ:
            # host 只留可回收 ND mmap；持久 NZ buffer 仍建好，decode 时按专家
            # 就地 cast 后 D2D 写入对应专家偏移。
            prepare_t0 = time.perf_counter()
            for offloader in self.module_offloaders:
                offloader.create_nz_buffers()
            logger.info_once(
                f"[NPUPrefetchOffloader] per-expert NZ buffers ready in {time.perf_counter() - prepare_t0:.1f}s "
                f"(NZ buffers: {self.buffer_pool.total_nz_bytes / 1e9:.4f} GB)"
            )
'''
    text = _replace_once(text, old, new, "post_init per-expert branch")

    old = '''    def create_nz_buffers(self) -> None:
        """为每个 (param, slot) 建立持久 NZ buffer；共享同 slot 的各层复用同一块。"""
        assert self._buffer_pool is not None
        for name, offloader in self._param_offloaders.items():
            cpu_storage = offloader._cpu_storage
            gpu_buffer = offloader._gpu_buffer
            npu_format = self._npu_formats.get(name, ACL_FORMAT_FRACTAL_ND)
            if npu_format not in _NZ_FORMATS or cpu_storage is None or gpu_buffer is None:
                continue
            shared_nz = self._buffer_pool.get_or_create_nz_buffer(
                name=name,
                shape=tuple(cpu_storage.shape),
                dtype=cpu_storage.dtype,
                npu_format=npu_format,
                slot_idx=self._buffer_slot_idx,
                factory=lambda buf=gpu_buffer, fmt=npu_format: _cast_nd_to_nz(buf, fmt),
            )
            offloader._nz_buffer = shared_nz
            offloader._param.data = shared_nz.transpose(1, 2)
        self._host_nz = True
'''
    new = '''    def create_nz_buffers(self) -> None:
        """为每个 (param, slot) 建立持久 NZ buffer；共享同 slot 的各层复用同一块。"""
        assert self._buffer_pool is not None
        for name, offloader in self._param_offloaders.items():
            cpu_storage = offloader._cpu_storage
            gpu_buffer = offloader._gpu_buffer
            npu_format = self._npu_formats.get(name, ACL_FORMAT_FRACTAL_ND)
            if npu_format not in _NZ_FORMATS or cpu_storage is None or gpu_buffer is None:
                continue
            shared_nz = self._buffer_pool.get_or_create_nz_buffer(
                name=name,
                shape=tuple(cpu_storage.shape),
                dtype=cpu_storage.dtype,
                npu_format=npu_format,
                slot_idx=self._buffer_slot_idx,
                factory=lambda buf=gpu_buffer, fmt=npu_format: _cast_nd_to_nz(buf, fmt),
            )
            offloader._nz_buffer = shared_nz
            offloader._param.data = shared_nz.transpose(1, 2)
            offloader._nz_per_expert_bytes = shared_nz.untyped_storage().nbytes() // int(shared_nz.shape[0])
            if not _OFFLOAD_HOST_NZ:
                # 单专家 ND staging（可复用），decode 时按专家 cast。
                offloader._expert_staging = torch.empty(
                    (1, *tuple(cpu_storage.shape[1:])), dtype=cpu_storage.dtype, device=self.device
                )
        self._expert_nz = True
        self._host_nz = _OFFLOAD_HOST_NZ
'''
    text = _replace_once(text, old, new, "create_nz_buffers per-expert staging")

    old = '''        if self._host_nz:
            self._load_selected_experts_from_host_nz(expert_ids)
            return
'''
    new = '''        if self._host_nz:
            self._load_selected_experts_from_host_nz(expert_ids)
            return
        if self._expert_nz:
            self._load_selected_experts_expert_nz(expert_ids)
            return
'''
    text = _replace_once(text, old, new, "load_selected dispatch")

    old = '''    def _load_selected_experts_from_host_nz(self, expert_ids: torch.Tensor) -> None:
'''
    new = '''    def _load_selected_experts_expert_nz(self, expert_ids: torch.Tensor) -> None:
        """逐专家 ND->NZ：copy 专家到单专家 staging，cast，再 D2D 到共享 NZ 偏移。

        单专家 [1, M, K] cast 与整块 [E, M, K] cast 的对应专家块**逐字节相同**
        （probe_nz_layout.py 在 e=0/1/7/383 验证），因此可以只转激活专家。
        """
        profiling = _OFFLOAD_PROFILE_LEVEL > 0
        host_t0 = time.perf_counter() if profiling else 0.0
        ev0 = ev1 = None
        if profiling:
            ev0 = torch.npu.Event(enable_timing=True)
            ev0.record()
        for name, offloader in self._param_offloaders.items():
            cpu_storage = offloader._cpu_storage
            nz = offloader._nz_buffer
            staging = offloader._expert_staging
            fmt = self._npu_formats.get(name, ACL_FORMAT_FRACTAL_ND)
            assert cpu_storage is not None and nz is not None and staging is not None
            per = offloader._nz_per_expert_bytes
            nz_ptr = nz.data_ptr()
            for expert_id in expert_ids.tolist():
                if expert_id >= int(nz.shape[0]):
                    continue
                staging.copy_(cpu_storage[expert_id : expert_id + 1])
                converted = _cast_nd_to_nz(staging, fmt)
                _acl_memcpy_async(
                    nz_ptr + expert_id * per,
                    converted.data_ptr(),
                    per,
                    _ACL_MEMCPY_DEVICE_TO_DEVICE,
                )
            offloader._param.data = nz.transpose(1, 2)
        if profiling:
            host_ms = (time.perf_counter() - host_t0) * 1000.0
            ev1 = torch.npu.Event(enable_timing=True)
            ev1.record()
            ev1.synchronize()
            _OFFLOAD_PROFILER.add(ev0.elapsed_time(ev1), 0.0, host_ms, int(expert_ids.numel()))

    def _load_selected_experts_from_host_nz(self, expert_ids: torch.Tensor) -> None:
'''
    text = _replace_once(text, old, new, "per-expert loader method")

    old = '''        # host-NZ 模式开关（每个 param 各自持有 host NZ 字节与持久 NZ buffer）。
        self._host_nz = False
'''
    new = '''        # host-NZ 模式开关（每个 param 各自持有 host NZ 字节与持久 NZ buffer）。
        self._host_nz = False
        # 只转选中专家的模式（持久 NZ buffer + 单专家 staging）。
        self._expert_nz = False
'''
    text = _replace_once(text, old, new, "module expert_nz flag")
    path.write_text(text)


def patch_engram(sp: pathlib.Path) -> None:
    path = sp / "vllm_ascend/models/deepseek_v41/engram/npu.py"
    text = path.read_text()
    if "\nimport os\n" not in text:
        if "import ctypes\n" not in text:
            raise SystemExit("[fail] engram npu.py: no import ctypes anchor")
        text = text.replace("import ctypes\n", "import ctypes\nimport os\n", 1)
        print("[patch] engram import os")
    old = '''        if _os.environ.get("VLLM_ASCEND_ENGRAM_PAGEABLE", "0") not in ("", "0"):
            # Benchmark fallback: pageable host table, gathered on CPU per lookup.
            # No aclrtHostRegisterV2, so the pages stay reclaimable/swapable.
            self.tensor = torch.empty(shape, dtype=dtype, device="cpu")
            self.ptrs = None
            self.pointer = None
            self.buffer = None
            self._mm = None
            self._regions = []
            return
'''
    new = '''        if _os.environ.get("VLLM_ASCEND_ENGRAM_PAGEABLE", "0") not in ("", "0"):
            # Pageable host table, gathered on CPU per lookup.  File-backed when
            # VLLM_CPU_WEIGHT_MMAP_DIR is set: the 196 GB table then lives in
            # reclaimable page cache instead of anonymous memory.
            rows = int(shape[0])
            row_bytes = int(torch.Size(shape[1:]).numel()) * torch.empty((), dtype=dtype).element_size()
            root = _os.environ.get("VLLM_CPU_WEIGHT_MMAP_DIR", "")
            if root:
                _os.makedirs(root, exist_ok=True)
                path = _os.path.join(root, f"engram_{_os.getpid()}_{id(self):x}_{rows}.bin")
                nbytes = rows * row_bytes
                fd = _os.open(path, _os.O_RDWR | _os.O_CREAT | _os.O_TRUNC, 0o600)
                try:
                    _os.ftruncate(fd, nbytes)
                    mm = mmap.mmap(fd, nbytes, access=mmap.ACCESS_WRITE)
                finally:
                    _os.close(fd)
                raw = torch.frombuffer(mm, dtype=torch.uint8, count=nbytes)
                self.tensor = raw.view(dtype).view(shape)
                self._mm = mm
                self._mm_path = path
            else:
                self.tensor = torch.empty(shape, dtype=dtype, device="cpu")
                self._mm = None
                self._mm_path = None
            self.ptrs = None
            self.pointer = None
            self.buffer = None
            self._regions = []
            return
'''
    text = _replace_once(text, old, new, "engram pageable mmap")

    old = '''        if getattr(self, "_mm", None) is None and not getattr(self, "_regions", []):
            self.tensor = None
            self.ptrs = None
            return
'''
    new = '''        if getattr(self, "_mm", None) is None and not getattr(self, "_regions", []):
            self.tensor = None
            self.ptrs = None
            return
        pageable_path = getattr(self, "_mm_path", None)
'''
    text = _replace_once(text, old, new, "engram close pageable header")

    old = '''        if getattr(self, "_mm", None) is not None:
            self._mm.close()
            self._mm = None
        self.tensor = None
'''
    new = '''        if getattr(self, "_mm", None) is not None:
            self.tensor = None
            try:
                self._mm.close()
            except (BufferError, ValueError):
                pass
            self._mm = None
        self.tensor = None
        if pageable_path:
            try:
                os.unlink(pageable_path)
            except OSError:
                pass
'''
    text = _replace_once(text, old, new, "engram close pageable unlink")
    path.write_text(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-packages", required=True)
    args = ap.parse_args()
    sp = pathlib.Path(args.site_packages).resolve()
    print(f"[info] site-packages = {sp}")
    patch_per_expert(sp)
    patch_engram(sp)
    print("[done] per-expert NZ + pageable engram patches applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
