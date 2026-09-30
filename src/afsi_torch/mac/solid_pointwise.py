"""Experimental scalar 3x3 constitutive fusion, with the original FE assembly.

Explicit three-term contractions let Inductor fuse the small-matrix algebra
instead of dispatching a sequence of batched matrix products. FP64, Guccione
stress, quadrature and assembled nodal force are unchanged.
"""
import torch
from .solid_execution import SolidExecution


def product3(a,b):
    # No broadcast dimension of length 3x3x3, and no library BMM boundary.
    return torch.stack(tuple(torch.stack(tuple(
        a[...,i,0]*b[...,0,j]+a[...,i,1]*b[...,1,j]+a[...,i,2]*b[...,2,j]
        for j in range(3)),-1) for i in range(3)),-2)


class PointwiseSolidExecution(SolidExecution):
    def _stress(self,F,loads):
        m=self.model
        E=.5*(product3(F.transpose(-1,-2),F)-self.identity)
        local=product3(product3(self.axes.transpose(-1,-2),E),self.axes)
        weighted=self.strain_weights*local
        Q=(self.strain_weights*local.square()).sum((-2,-1))
        S=(m.parameters.C*torch.exp(Q))[...,None,None]*product3(
            product3(self.axes,weighted),self.axes.transpose(-1,-2))
        cof=torch.stack((torch.linalg.cross(F[...,1,:],F[...,2,:]),
                         torch.linalg.cross(F[...,2,:],F[...,0,:]),
                         torch.linalg.cross(F[...,0,:],F[...,1,:])),-2)
        J=(F[...,0,:]*cof[...,0,:]).sum(-1)
        P=product3(F,S)+2*m.parameters.kappa*(J-1)[...,None,None]*cof
        fiber=m.fields.fiber
        Ff=(F[..., :,0]*fiber[...,0,None]+F[..., :,1]*fiber[...,1,None]+
            F[..., :,2]*fiber[...,2,None])
        return P+loads[1]*Ff[..., :,None]*fiber[...,None,:]
