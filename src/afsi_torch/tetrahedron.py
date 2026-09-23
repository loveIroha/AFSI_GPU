"""P2 Lagrange tetrahedron with explicit vertex/edge DOF ordering.

Local nodes: v0,v1,v2,v3,e01,e02,e03,e12,e13,e23.
Never assume this equals a mesh reader's or DOLFINx's local ordering.
"""
import torch
from torch import Tensor

EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def reference_nodes(*, dtype=torch.float64, device="cpu") -> Tensor:
    vertices = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            dtype=dtype, device=device)
    edges = torch.tensor(EDGES, device=device)
    return torch.cat((vertices, vertices[edges].mean(dim=1)))


def tabulate(points: Tensor) -> tuple[Tensor, Tensor]:
    """N[q,a], dN_dxi[q,a,j] at points[q,3] of the unit tetrahedron."""
    if points.ndim != 2 or points.shape[1] != 3 or not points.is_floating_point():
        raise ValueError("points must be floating point with shape (Q,3)")
    L = torch.cat((1 - points.sum(-1, keepdim=True), points), dim=-1)
    dL = points.new_tensor([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    edges = torch.tensor(EDGES, device=points.device)
    i, j = edges[:, 0], edges[:, 1]
    N = torch.cat((L * (2 * L - 1), 4 * L[:, i] * L[:, j]), dim=-1)
    dN = torch.cat(((4 * L - 1)[..., None] * dL,
                    4 * (L[:, i, None] * dL[j] + L[:, j, None] * dL[i])), dim=1)
    return N, dN


def validate_mesh(X: Tensor, cells: Tensor, nodes_per_cell: int) -> None:
    """Setup-time checks, outside automatic differentiation transforms."""
    if X.ndim != 2 or X.shape[1] != 3 or X.dtype not in (torch.float32, torch.float64):
        raise ValueError("X must have shape (N,3), float32 or float64")
    if cells.ndim != 2 or cells.shape[1] != nodes_per_cell or cells.shape[0] == 0:
        raise ValueError(f"cells must have nonempty shape (E,{nodes_per_cell})")
    if cells.dtype != torch.int64 or cells.device != X.device:
        raise ValueError("cells must be int64 on the same device as X")
    if not torch.isfinite(X).all() or (cells < 0).any() or (cells >= X.shape[0]).any():
        raise ValueError("nonfinite coordinates or out-of-range connectivity")


def promote_p1(X: Tensor, cells: Tensor) -> tuple[Tensor, Tensor]:
    """Append ONE global midpoint per unique edge, sharing DOFs across cells.

    Mesh preprocessing only. No Python per-element loops or CPU index lists.
    """
    validate_mesh(X, cells, 4)
    edges = torch.tensor(EDGES, device=cells.device)
    pairs = cells[:, edges].reshape(-1, 2).sort(dim=1).values
    unique, inverse = torch.unique(pairs, dim=0, return_inverse=True)
    midpoints = X[unique].mean(dim=1)
    p2_cells = torch.cat((cells, X.shape[0] + inverse.reshape(-1, 6)), dim=1)
    return torch.cat((X, midpoints)), p2_cells
