"""P1 tetrahedral verification kernel; current coordinates x, reference X.

This is a first test, not the P2/Guccione solid model used by afsi.
All cells must be nondegenerate, and all deformations must have det(F)>0.
Material parameters are the shear modulus mu and first Lame parameter lam.
"""

import torch
from torch import Tensor


def determinant3(F: Tensor) -> Tensor:
    """Scalar triple product, smooth even at repeated singular values.

    Avoids decomposition-based determinant higher-derivative paths at F=I.
    """
    return (F[..., 0, :] * torch.linalg.cross(F[..., 1, :], F[..., 2, :])).sum(-1)


def reference_geometry(X: Tensor, cells: Tensor) -> tuple[Tensor, Tensor]:
    """Return physical shape gradients (E,4,3) and volumes (E,)."""
    nodes = X[cells]
    D = (nodes[:, 1:] - nodes[:, :1]).transpose(-1, -2)
    det = torch.linalg.det(D)
    if not torch.isfinite(D).all() or (det == 0).any():
        raise ValueError("Reference tetrahedra must be finite and nondegenerate")
    grad_hat = X.new_tensor([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return grad_hat @ torch.linalg.inv(D), det.abs() / 6


def deformation_gradient(x: Tensor, cells: Tensor, gradients: Tensor) -> Tensor:
    return torch.einsum("eai,eaJ->eiJ", x[cells], gradients)


def validate_deformation(x: Tensor, cells: Tensor, gradients: Tensor) -> None:
    """Call outside torch.func transforms, e.g. when accepting a time step."""
    J = determinant3(deformation_gradient(x, cells, gradients))
    if not torch.isfinite(J).all() or (J <= 0).any():
        raise ValueError("Deformation must have finite det(F)>0 in every cell")


def energy(x: Tensor, cells: Tensor, gradients: Tensor, volumes: Tensor,
           mu: float, lam: float) -> Tensor:
    """Compressible Neo-Hookean total energy, exactly integrated for P1.

    Pure differentiable kernel: the caller validates det(F)>0 separately.
    In this coordinate formulation F=grad_X(x), not I+grad_X(x).
    """
    F = deformation_gradient(x, cells, gradients)
    logJ = torch.log(determinant3(F))
    W = 0.5 * mu * ((F * F).sum(dim=(-2, -1)) - 3) - mu * logJ + 0.5 * lam * logJ**2
    return (volumes * W).sum()


def stress_force(x: Tensor, cells: Tensor, gradients: Tensor, volumes: Tensor,
                 mu: float, lam: float) -> Tensor:
    """Independent analytic PK1/scatter route to nodal force g=-dE/dx."""
    F = deformation_gradient(x, cells, gradients)
    invT = torch.linalg.inv(F).transpose(-1, -2)
    P = mu * (F - invT) + lam * torch.log(determinant3(F))[:, None, None] * invT
    local = -volumes[:, None, None] * torch.einsum("eiJ,eaJ->eai", P, gradients)
    return torch.zeros_like(x).index_add(0, cells.reshape(-1), local.reshape(-1, 3))
