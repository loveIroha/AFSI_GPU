"""Bounded reusable storage; a live stencil always retains exclusive ownership."""
from dataclasses import dataclass
import weakref
import torch


@dataclass
class _Slot:
    base: torch.Tensor
    phi: torch.Tensor
    capacity: int
    stream: int
    owner: object = None


class StencilWorkspace:
    def __init__(self, max_points, *, slots=2, quantum=262144):
        self.max_points, self.limit, self.quantum = max_points, slots, quantum
        self.slots = []
        self.allocations = self.reuses = self.overflow_allocations = 0

    def acquire(self, like, points):
        if not 0 < points <= self.max_points:
            raise ValueError('stencil workspace point budget exceeded')
        stream = torch.cuda.current_stream(like.device).cuda_stream if like.is_cuda else 0
        available = [s for s in self.slots if s.stream == stream and
                     (s.owner is None or s.owner() is None)]
        slot = min(available, key=lambda s: (s.capacity < points, s.capacity)) if available else None
        quantum = min(self.quantum,max(64,1 << (points.bit_length()-1)))
        capacity = min(self.max_points, ((points+quantum-1)//quantum)*quantum)
        if slot is None:
            base = torch.empty(6*capacity, device=like.device, dtype=torch.int64)
            phi = like.new_empty(24*capacity)
            slot = _Slot(base, phi, capacity, stream)
            self.allocations += 1
            if len(self.slots) < self.limit:
                self.slots.append(slot)
            else:
                # External callers may retain arbitrarily many immutable stencils.
                # Their owned allocation is never added to this bounded cache.
                self.overflow_allocations += 1
        elif slot.capacity < points:
            capacity = min(self.max_points, max(capacity, (slot.capacity*5+3)//4))
            # No live owner: discard old storage before allocating the replacement.
            slot.base = slot.phi = None
            slot.base = torch.empty(6*capacity, device=like.device, dtype=torch.int64)
            slot.phi = like.new_empty(24*capacity)
            slot.capacity = capacity
            self.allocations += 1
        else:
            self.reuses += 1
        # Slice the FLAT allocation first. Logical lattice stride is points,
        # not capacity, so every existing CUDA/reference reader stays valid.
        return (slot.base[:6*points].view(2,points,3),
                slot.phi[:24*points].view(2,points,3,4), slot)

    def retain(self, slot, stencil):
        slot.owner = weakref.ref(stencil)

    def summary(self):
        return dict(slots=len(self.slots), capacity_points=[s.capacity for s in self.slots],
                    reserved_bytes=sum(s.capacity*240 for s in self.slots),
                    allocations=self.allocations, reuses=self.reuses,
                    overflow_allocations=self.overflow_allocations)
