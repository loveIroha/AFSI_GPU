"""P2 triangle trace: vertices 0,1,2 followed by edges 01,02,12."""
import numpy as np
import torch


def tabulate(points):
    if points.ndim != 2 or points.shape[1] != 2 or not points.is_floating_point():
        raise ValueError("triangle points must have floating shape (Q,2)")
    L = torch.cat((1-points.sum(-1, keepdim=True), points), dim=-1)
    dL = points.new_tensor([[-1., -1.], [1., 0.], [0., 1.]])
    edges = torch.tensor([[0, 1], [0, 2], [1, 2]], device=points.device)
    i, j = edges[:, 0], edges[:, 1]
    N = torch.cat((L*(2*L-1), 4*L[:, i]*L[:, j]), dim=1)
    dN = torch.cat(((4*L-1)[..., None]*dL,
                    4*(L[:, i, None]*dL[j]+L[:, j, None]*dL[i])), dim=1)
    return N, dN


def quadrature(degree=4, *, dtype=torch.float64, device="cpu"):
    """Positive Duffy/Gauss rule, exact through total degree; degree=4: 9 points."""
    if not isinstance(degree, int) or isinstance(degree, bool) or degree < 0:
        raise ValueError("degree must be a nonnegative integer")
    z, w = np.polynomial.legendre.leggauss((degree+3)//2)
    z = torch.as_tensor((z+1)/2, dtype=dtype, device=device)
    w = torch.as_tensor(w/2, dtype=dtype, device=device)
    u, v = torch.meshgrid(z, z, indexing="ij")
    a, b = torch.meshgrid(w, w, indexing="ij")
    return torch.stack((u, (1-u)*v), dim=-1).reshape(-1, 2), (a*b*(1-u)).reshape(-1)
