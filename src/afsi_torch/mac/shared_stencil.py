"""Shared face/center one-dimensional Peskin tables for a Cartesian MAC grid."""
from dataclasses import dataclass
import torch
from .compact_transfer import CompactStencil


@dataclass(frozen=True)
class SharedStencil(CompactStencil):
    # base=(2,P,3), phi=(2,P,3,4): lattice 0 is face, lattice 1 is center.
    # Component c uses the face lattice along c and centers along other axes.
    layout = 'shared'

    def expanded(self):
        """CPU/reference compatibility; production CUDA kernels read shared tables."""
        base = torch.stack([torch.stack([self.base[int(a!=c),:,a]
            for a in range(3)],-1) for c in range(3)])
        phi = torch.stack([torch.stack([self.phi[int(a!=c),:,a,:]
            for a in range(3)],1) for c in range(3)])
        return CompactStencil(base,phi)
