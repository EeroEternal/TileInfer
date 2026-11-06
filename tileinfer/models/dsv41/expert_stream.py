"""Flat expert arena: pack host-resident experts, page one slot at a time (S4 step 3c groundwork).

`tools/dsv41/s4_batched_h2d.py` measured the shape of the answer: six 17.9 MiB experts copied as 36
small tensors take 10.4 ms, while the same bytes copied as one 107.6 MiB block take 2.11 ms
(53.6 GB/s, i.e. the pinned bandwidth ceiling).  So the paging unit has to be *one flat block per
slot*, and the device side has to re-derive the individual tensors from byte offsets.

This module is that layout, with no device dependency, so the packing/unpacking logic is unit-testable
on a CPU box:

* :class:`ExpertArena` owns one flat buffer of ``n_slots * slot_bytes`` and a
  :class:`~tileinfer.models.dsv41.offload.SlotCache` that decides which expert lives in which slot;
* :meth:`ExpertArena.pack` writes a slot from an expert module (``w1|w1.scale|w2|...``);
* :meth:`ExpertArena.views` gives back the tensors for that slot, reshaped and dtype-restored, ready
  for the fp4/fp8 kernels;
* :meth:`ExpertArena.pin` / :meth:`ExpertArena.to_device` move the flat block in one transfer.

The dtype restoration is the subtle part: the checkpoint stores fp4 weights as 2-values-per-byte
(``float4_e2m1fn_x2``) and scales as ``float8_e8m0fnu``, both of which are byte views of a ``uint8``
buffer, so a slot view must be ``.view(uint8)[a:b].view(dtype).reshape(shape)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from tileinfer.models.dsv41.offload import SlotCache

__all__ = ["PartSpec", "ExpertArena", "ExpertSlotStub", "expert_specs"]

PARTS: Tuple[str, str, str] = ("w1", "w2", "w3")


class ExpertSlotStub(torch.nn.Module):
    """Stand-in for :class:`~tileinfer.models.dsv41.modules.Dsv41Expert` in CPU-only tests.

    Only the attribute shape matters to the arena (``w1/w2/w3`` each with ``.weight`` and ``.scale``),
    so the stub holds plain tensors instead of quantised parameter slots.
    """

    def __init__(self, w1_weight, w1_scale, w2_weight, w2_scale, w3_weight, w3_scale):
        super().__init__()
        for name, weight, scale in (
            ("w1", w1_weight, w1_scale),
            ("w2", w2_weight, w2_scale),
            ("w3", w3_weight, w3_scale),
        ):
            slot = torch.nn.Module()
            slot.weight = torch.nn.Parameter(weight.detach(), requires_grad=False)
            slot.scale = torch.nn.Parameter(scale.detach(), requires_grad=False)
            setattr(self, name, slot)


@dataclass(frozen=True)
class PartSpec:
    """One tensor of one expert as it sits in the flat slot."""

    name: str  # "w1", "w1.scale", ...
    offset: int  # byte offset inside the slot
    nbytes: int
    shape: Tuple[int, ...]
    dtype: torch.dtype  # logical dtype (may be a byte-view dtype such as float4_e2m1fn_x2)
    stored_bytes_per_element: int  # 1 for uint8-backed dtypes

    @property
    def end(self) -> int:
        return self.offset + self.nbytes

    def slice_of(self, flat: torch.Tensor) -> torch.Tensor:
        """Byte slice -> logical tensor of the original shape/dtype."""
        raw = flat[self.offset : self.end].view(torch.uint8)
        numel = 1
        for dim in self.shape:
            numel *= int(dim)
        if raw.numel() != numel * self.stored_bytes_per_element:
            raise ValueError(
                f"{self.name}: slot holds {raw.numel()} B, a {self.shape} {self.dtype} tensor "
                f"needs {numel * self.stored_bytes_per_element} B"
            )
        typed = raw.view(self.dtype) if self.dtype is not torch.uint8 else raw
        return typed.reshape(self.shape)


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def expert_specs(expert: torch.nn.Module, parts: Sequence[str] = PARTS) -> List[PartSpec]:
    """Layout of one expert: each weight followed by its scale, in ``parts`` order."""
    specs: List[PartSpec] = []
    offset = 0
    for name in parts:
        slot = getattr(expert, name)
        for suffix, tensor in (("", slot.weight), (".scale", slot.scale)):
            n = _nbytes(tensor)
            specs.append(
                PartSpec(
                    name=name + suffix,
                    offset=offset,
                    nbytes=n,
                    shape=tuple(tensor.shape),
                    dtype=tensor.dtype,
                    stored_bytes_per_element=tensor.element_size(),
                )
            )
            offset += n
    return specs


@dataclass
class ExpertArena:
    """One flat pinned buffer + a slot cache; ``slot_bytes`` covers one whole expert."""

    specs: List[PartSpec]
    n_slots: int
    watermark_bytes: Optional[int] = None
    pin: bool = True
    _host: Optional[torch.Tensor] = field(default=None, repr=False)
    _device: Optional[torch.Tensor] = field(default=None, repr=False)
    _device_slots: set = field(default_factory=set, repr=False)
    cache: Optional[SlotCache] = field(default=None, repr=False)

    # -- construction -------------------------------------------------------------------------

    def __post_init__(self) -> None:
        if not self.specs:
            raise ValueError("no specs")
        self._host = self._allocate(self.pin)
        self.cache = SlotCache(
            slot_bytes=self.slot_bytes, n_slots=self.n_slots, watermark_bytes=self.watermark_bytes
        )

    @property
    def slot_bytes(self) -> int:
        return max(s.end for s in self.specs)

    def _allocate(self, pin: bool) -> torch.Tensor:
        total = self.slot_bytes * self.n_slots
        buf = torch.empty(total, dtype=torch.uint8)
        if pin:
            try:
                buf = buf.pin_memory()
            except Exception:
                pass  # CPU-only boxes have no pinned allocator; the logic is unaffected
        return buf

    @property
    def host(self) -> torch.Tensor:
        assert self._host is not None
        return self._host

    def to_device(self, device: str = "npu:0") -> None:
        """Allocate the device arena and copy the whole thing (start-up / warm-up)."""
        total = self.slot_bytes * self.n_slots
        self._device = torch.empty(total, dtype=torch.uint8, device=device)
        self._device.copy_(self.host)
        self._device_slots = set(range(self.n_slots))

    def to_device_slots(self, slots: Iterable[int], device: str = "npu:0") -> int:
        """Copy only the given slots to the card; returns the bytes moved.

        Copying the whole arena per layer is what made the first decode run take 1.9 s/token: with
        96 slots x 17.93 MiB, one full copy per layer is 67 GiB per token.  Only the slots that were
        just packed need to travel, so a layer costs six slots = 108 MiB.
        """
        if self._device is None:
            total = self.slot_bytes * self.n_slots
            self._device = torch.empty(total, dtype=torch.uint8, device=device)
            self._device_slots = set()
        moved = 0
        for slot in sorted(set(int(x) for x in slots)):
            base = slot * self.slot_bytes
            self._device[base : base + self.slot_bytes].copy_(self.host[base : base + self.slot_bytes])
            self._device_slots.add(slot)
            moved += self.slot_bytes
        return moved

    # -- the paging protocol ------------------------------------------------------------------

    def bind(self, layer: int, expert_id: int) -> Tuple[int, bool]:
        """Reserve a slot for ``(layer, expert_id)``; returns ``(slot, hit)``."""
        assert self.cache is not None
        return self.cache.acquire((layer, expert_id))

    def release(self, layer: int, expert_id: int) -> None:
        assert self.cache is not None
        self.cache.release((layer, expert_id))

    def pack(self, slot: int, expert: torch.nn.Module) -> int:
        """Copy one expert into its slot; returns the byte count moved."""
        base = slot * self.slot_bytes
        moved = 0
        for spec in self.specs:
            part, _, sub = spec.name.partition(".")
            tensor = getattr(getattr(expert, part), sub or "weight")
            raw = tensor.detach().reshape(-1).view(torch.uint8)
            if raw.numel() != spec.nbytes:
                raise ValueError(f"{spec.name}: {raw.numel()} B on the host, {spec.nbytes} B expected")
            self.host[base + spec.offset : base + spec.end].copy_(raw)
            moved += spec.nbytes
        return moved

    def load_slot(self, slot: int, expert: torch.nn.Module, device: str = "npu:0") -> int:
        """Copy one expert straight from its host tensors into the device slot.

        This skips :meth:pack entirely.  Measured: staging into the pinned arena first costs a
        host-to-host memcpy of 108 MiB per layer (≈11 ms), while six direct copies to the card cost
        ≈2 ms -- the arena's single-block trick only pays off when the source is *already* the arena.
        """
        if self._device is None:
            self._device = torch.empty(self.slot_bytes * self.n_slots, dtype=torch.uint8, device=device)
            self._device_slots = set()
        base = slot * self.slot_bytes
        moved = 0
        for spec in self.specs:
            part, _, sub = spec.name.partition(".")
            src = getattr(getattr(expert, part), sub or "weight")
            dst = self._device[base + spec.offset : base + spec.end]
            if spec.dtype is torch.float4_e2m1fn_x2:
                src_bytes = src.view(torch.uint8).reshape(-1)
            else:
                src_bytes = src.reshape(-1).view(torch.uint8)
            dst.copy_(src_bytes)
            moved += spec.nbytes
        self._device_slots.add(slot)
        return moved

    def views(self, slot: int, on_device: bool = False) -> Dict[str, torch.Tensor]:
        """The tensors of the expert living in ``slot``, ready for the kernels."""
        base = slot * self.slot_bytes
        if on_device:
            if self._device is None:
                raise RuntimeError("call to_device() first")
            flat = self._device[base : base + self.slot_bytes]
        else:
            flat = self.host[base : base + self.slot_bytes]
        return {spec.name: spec.slice_of(flat) for spec in self.specs}

    # -- diagnostics ---------------------------------------------------------------------------

    def snapshot(self) -> Dict[str, float]:
        assert self.cache is not None
        data = self.cache.snapshot()
        data.update(slot_bytes=self.slot_bytes, arena_bytes=self.slot_bytes * self.n_slots)
        return data
