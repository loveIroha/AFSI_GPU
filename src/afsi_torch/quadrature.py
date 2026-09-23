"""Positive-weight tetrahedron quadrature, generated once during setup."""
import numpy as np
import torch


def tetrahedron_rule(degree=4, *, dtype=torch.float64, device="cpu"):
    """Integrate polynomials of total degree <= degree on the unit tetrahedron.

    Duffy map: (u,v,w) -> (u,(1-u)v,(1-u)(1-v)w).
    Its Jacobian is (1-u)^2(1-v). Gauss order n=ceil((degree+3)/2)
    suffices in every direction. degree=4 uses 64 points, not a minimal rule.
    A nonlinear constitutive energy is generally NOT integrated exactly.
    """
    if not isinstance(degree, int) or isinstance(degree, bool) or degree < 0:
        raise ValueError("degree must be a nonnegative integer")
    z, weights = np.polynomial.legendre.leggauss((degree + 4) // 2)
    z = torch.as_tensor((z + 1) / 2, dtype=dtype, device=device)
    weights = torch.as_tensor(weights / 2, dtype=dtype, device=device)
    u, v, w = torch.meshgrid(z, z, z, indexing="ij")
    a, b, c = torch.meshgrid(weights, weights, weights, indexing="ij")
    points = torch.stack((u, (1-u)*v, (1-u)*(1-v)*w), dim=-1).reshape(-1, 3)
    qweights = (a*b*c*(1-u)**2*(1-v)).reshape(-1)
    return points, qweights
