"""Reference material fields: interpolate first, then form sheet x fiber.

No normalization or orthogonalization is applied: both would change the
discrete material represented by afsi's interpolated P2 Functions.
"""
from dataclasses import dataclass
import torch
from torch import Tensor
from .solid import P2Geometry


@dataclass(frozen=True)
class ReferenceFields:
    fiber: Tensor       # E,Q,3
    sheet: Tensor       # E,Q,3
    normal: Tensor      # E,Q,3; cross(sheet,fiber), evaluated at quadrature
    tension: Tensor     # E,Q; prescribed independently of current coordinates


def interpolate_p2(nodal: Tensor, geometry: P2Geometry) -> Tensor:
    """Interpolate scalar (N,) or vector (N,k) coefficients to (E,Q[,k])."""
    if nodal.ndim not in (1, 2) or nodal.shape[0] != geometry.node_count:
        raise ValueError("nodal field must have shape (N,) or (N,k)")
    if nodal.device != geometry.values.device or nodal.dtype != geometry.values.dtype:
        raise ValueError("nodal field must match geometry device and dtype")
    local = nodal[geometry.cells]
    if nodal.ndim == 1:
        return torch.einsum("qa,ea->eq", geometry.values, local)
    return torch.einsum("qa,eak->eqk", geometry.values, local)


def prepare_reference_fields(geometry: P2Geometry, fiber, sheet, tension=0.0) -> ReferenceFields:
    """Directions: constant (3,) or nodal (N,3); tension: scalar or nodal (N,).

    Call again when prescribed tension or reference fields change. Inputs are
    converted to geometry's device/dtype only during this preprocessing step.
    Degenerate directions are rejected; non-unit/non-orthogonal ones are kept.
    """
    like = geometry.values
    E, Q = geometry.weights.shape

    def direction(value):
        data = torch.as_tensor(value, dtype=like.dtype, device=like.device)
        if data.shape == (3,):
            return data.expand(E, Q, 3)
        if data.shape == (geometry.node_count, 3):
            return interpolate_p2(data, geometry)
        raise ValueError("direction must be constant (3,) or nodal (N,3)")

    f, s = direction(fiber), direction(sheet)
    n = torch.linalg.cross(s, f)
    t = torch.as_tensor(tension, dtype=like.dtype, device=like.device)
    if t.ndim == 0:
        t = t.expand(E, Q)
    elif t.shape == (geometry.node_count,):
        t = interpolate_p2(t, geometry)
    else:
        raise ValueError("tension must be a scalar or a nodal field (N,)")
    if not all(torch.isfinite(v).all() for v in (f, s, n, t)):
        raise ValueError("reference fields must be finite")
    lengths = torch.linalg.vector_norm(f, dim=-1)*torch.linalg.vector_norm(s, dim=-1)
    if (torch.linalg.vector_norm(n, dim=-1) <= 100*torch.finfo(like.dtype).eps*lengths).any():
        raise ValueError("fiber and sheet must be nonzero and nonparallel at quadrature points")
    return ReferenceFields(f, s, n, t)
