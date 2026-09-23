"""Batched quadrature-point constitutive laws; no mesh/global DOFs here."""
import torch
from dataclasses import dataclass
from math import isfinite
from .mechanics import determinant3


def neo_hookean_energy(F, mu, lam):
    logJ = torch.log(determinant3(F))
    return 0.5*mu*(F.square().sum(dim=(-2, -1))-3) - mu*logJ + 0.5*lam*logJ.square()


def neo_hookean_pk1(F, mu, lam):
    invT = torch.linalg.inv(F).transpose(-1, -2)
    return mu*(F-invT) + lam*torch.log(determinant3(F))[..., None, None]*invT


@dataclass(frozen=True)
class GuccioneParameters:
    """Parameters for the non-deviatoric model in afsi demo_337.

    C and kappa have stress units; bf, bt, bfs are dimensionless.
    No isochoric variant is implied by this implementation.
    """
    C: float = 20000.0
    bf: float = 8.0
    bt: float = 2.0
    bfs: float = 4.0
    kappa: float = 500000.0

    def __post_init__(self):
        values = (self.C, self.bf, self.bt, self.bfs, self.kappa)
        if not all(isfinite(v) and v >= 0 for v in values) or self.C == 0:
            raise ValueError("Guccione parameters must be finite, C>0 and others>=0")


def _guccione_strain(F, fiber, sheet, normal, parameters):
    E = 0.5 * (F.transpose(-1, -2) @ F - torch.eye(3, dtype=F.dtype, device=F.device))
    axes = torch.stack((fiber, sheet, normal), dim=-1)
    local = axes.transpose(-1, -2) @ E @ axes
    p = parameters
    weights = F.new_tensor([[p.bf, p.bfs, p.bfs], [p.bfs, p.bt, p.bt], [p.bfs, p.bt, p.bt]])
    Q = (weights * local.square()).sum(dim=(-2, -1))
    return axes, local, weights, Q


def guccione_energy(F, fiber, sheet, normal, parameters):
    """Passive density W=C/2*(exp(Q)-1)+kappa*(J-1)^2, det(F)>0."""
    _, _, _, Q = _guccione_strain(F, fiber, sheet, normal, parameters)
    return 0.5*parameters.C*torch.expm1(Q) + parameters.kappa*(determinant3(F)-1).square()


def guccione_pk1(F, fiber, sheet, normal, parameters):
    """Analytic PK1, independent of differentiating guccione_energy."""
    axes, local, weights, Q = _guccione_strain(F, fiber, sheet, normal, parameters)
    S = parameters.C*torch.exp(Q)[..., None, None] * (axes @ (weights*local) @ axes.transpose(-1, -2))
    J = determinant3(F)
    bulk = 2*parameters.kappa*(J-1)*J
    return F @ S + bulk[..., None, None]*torch.linalg.inv(F).transpose(-1, -2)


def active_potential(F, fiber, tension):
    """Instantaneous potential ONLY for prescribed, deformation-independent Ta.

    This is not a stored-energy law for a calcium/length-dependent active model.
    Subtract the reference constant so the value at F=I is zero even if |f|!=1.
    """
    Ff = (F @ fiber[..., None]).squeeze(-1)
    return 0.5*tension*(Ff.square().sum(-1)-fiber.square().sum(-1))


def active_pk1(F, fiber, tension):
    """Ta*(F f0) outer f0, with fixed reference fiber and prescribed Ta."""
    Ff = (F @ fiber[..., None]).squeeze(-1)
    tension = torch.as_tensor(tension, dtype=F.dtype, device=F.device)
    return tension[..., None, None]*Ff[..., :, None]*fiber[..., None, :]
