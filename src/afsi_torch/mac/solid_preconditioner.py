"""Local inertia/stiffness approximation for the unchanged coupled GMRES.

The true Jacobian still includes both consistent mass inverses, IB transfer,
viscous solve and pressure projection. Lumped P1 reference volume is used ONLY
to estimate local fluid mobility in this optional preconditioner.
"""
from math import isfinite
import torch


class SolidBlockPreconditioner:
    def __init__(self, assembler, *, dt, rho):
        if any(not isfinite(v) or v <= 0 for v in (dt,rho)):
            raise ValueError('positive preconditioner time step/density required')
        self.assembler = assembler
        m = assembler.model
        self.mass = m.volumes.new_zeros(len(m.mesh.X)).index_add(0,
            m.mesh.cells.reshape(-1),(m.volumes[:,None].expand(-1,4)/4).reshape(-1))
        if not torch.isfinite(self.mass).all() or (self.mass <= 0).any():
            raise ValueError('positive reference nodal volume required')
        self.factor = dt*dt/rho

    @torch.no_grad()
    def build(self, tangent):
        K = self.assembler.nodal_diagonal_blocks(tangent)
        identity = torch.eye(3,device=K.device,dtype=K.dtype)
        B = identity-self.factor*(K+K.transpose(-1,-2))/(2*self.mass[:,None,None])
        eigenvalues,Q = torch.linalg.eigh(B)
        # Retain the identity/inertial lower bound when the approximate local
        # stiffness has destabilizing curvature; the true operator is untouched.
        inverse = (Q/eigenvalues.clamp_min(1.)[:,None,:])@Q.transpose(-1,-2)
        if not torch.isfinite(inverse).all():
            raise FloatingPointError('nonfinite solid block preconditioner')
        def apply(v):
            return torch.bmm(inverse,v.reshape(-1,3,1)).reshape_as(v)
        return apply
