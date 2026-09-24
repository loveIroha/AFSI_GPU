"""Tensor-product equispaced Q1/Q2 basis on [0,1]^3, x-fast local order."""
from itertools import product
import numpy as np
import torch


def reference_nodes(degree, *, device='cpu', dtype=torch.float64):
    if degree not in (1, 2):
        raise ValueError('only Q1 and Q2 are supported')
    return torch.tensor([(i/degree, j/degree, k/degree)
                         for k, j, i in product(range(degree+1), repeat=3)],
                        device=device, dtype=dtype)


def basis(points, degree):
    """Return values (Q,A) and reference gradients (Q,A,3)."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must have shape (Q,3)')
    if degree == 1:
        L = torch.stack((1-points, points), -1)
        dL = torch.stack((-torch.ones_like(points), torch.ones_like(points)), -1)
    elif degree == 2:
        L = torch.stack((2*points**2-3*points+1, 4*points*(1-points), 2*points**2-points), -1)
        dL = torch.stack((4*points-3, 4-8*points, 4*points-1), -1)
    else:
        raise ValueError('only Q1 and Q2 are supported')
    indices = reference_nodes(degree, device=points.device)*degree
    i, j, k = indices.to(torch.int64).unbind(-1)
    lx, ly, lz = L[:, 0, i], L[:, 1, j], L[:, 2, k]
    return lx*ly*lz, torch.stack((dL[:, 0, i]*ly*lz,
                                lx*dL[:, 1, j]*lz, lx*ly*dL[:, 2, k]), -1)


def quadrature(order=4, *, device='cpu', dtype=torch.float64):
    """CPU Gauss setup, then device transfer. Four points integrate Q2 advection."""
    if not isinstance(order, int) or isinstance(order, bool) or order < 4:
        raise ValueError('at least four Gauss points per axis are required')
    q, w = np.polynomial.legendre.leggauss(order)
    q, w = (q+1)/2, w/2
    indices = np.array(list(product(range(order), repeat=3)))[:, ::-1].copy()
    return (torch.as_tensor(q[indices], device=device, dtype=dtype),
            torch.as_tensor(w[indices].prod(1), device=device, dtype=dtype))
