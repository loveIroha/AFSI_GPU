"""Fused same-operator FGMRES vector work; no stopping-policy changes."""
import torch
from math import prod
from .execution import tensor_kernel
from .grid import gradient,divergence,velocity_laplacian,zero_normal
from ..nonlinear import GMRESWorkspace


class StokesExecution:
    def __init__(self,flow):
        self.grid=flow.grid
        self.shapes=tuple(flow.grid.face_shape(c) for c in range(3))
        self.sizes=tuple(prod(s) for s in self.shapes)
        self.pressure_size=flow.pressure_solver.diagonals[0].numel()
        device=flow.pressure_solver.diagonals[0].device
        self.action=tensor_kernel(self._action,device)
        self.precondition_rhs=tensor_kernel(self._precondition_rhs,device)
        self.precondition_result=tensor_kernel(self._precondition_result,device)
        self.krylov=GMRESWorkspace()

    def unpack(self,v):
        parts=v.split((*self.sizes,self.pressure_size))
        return tuple(z.reshape(s) for z,s in zip(parts,self.shapes)),parts[-1].reshape(self.grid.shape)

    def pack(self,u,p):
        return torch.cat(tuple(v.reshape(-1) for v in u)+(p.reshape(-1),))

    def _action(self,v,alpha,h):
        u,p=self.unpack(v)
        gp=gradient(p,self.grid.spacing)
        hu=tuple(a-alpha*velocity_laplacian(a,c,self.grid.spacing)+h*g for c,(a,g) in enumerate(zip(u,gp)))
        return self.pack(zero_normal(hu),-h*divergence(u,self.grid.spacing))

    def _precondition_rhs(self,tentative,s,h):
        rhs=-s/h-divergence(tentative,self.grid.spacing)
        return rhs-rhs.mean()

    def _precondition_result(self,tentative,phi,rhs,alpha,h):
        grad=gradient(phi,self.grid.spacing)
        u=tuple(a-g for a,g in zip(tentative,grad))
        q=(phi+alpha*rhs)/h
        return self.pack(u,q-q.mean())
