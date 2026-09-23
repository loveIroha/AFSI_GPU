"""Batched quadrature-point constitutive laws; no mesh/global DOFs here."""
import torch
from .mechanics import determinant3


def neo_hookean_energy(F, mu, lam):
    logJ = torch.log(determinant3(F))
    return 0.5*mu*(F.square().sum(dim=(-2, -1))-3) - mu*logJ + 0.5*lam*logJ.square()


def neo_hookean_pk1(F, mu, lam):
    invT = torch.linalg.inv(F).transpose(-1, -2)
    return mu*(F-invT) + lam*torch.log(determinant3(F))[..., None, None]*invT
