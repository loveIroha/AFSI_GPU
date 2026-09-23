"""P2 follower pressure and reference-area foundation springs.

Outward means outward from SOLID material (into the cavity on an inner wall).
Positive pressure applies -p*n. Spring beta acts in all three directions.
Kernels return integrated global nodal forces, not force densities.
"""
from dataclasses import dataclass
import torch
from torch import Tensor
from . import triangle
from .tetrahedron import EDGES, validate_mesh


def extract_boundary(X: Tensor, cells: Tensor) -> Tensor:
    """Conforming P2 tetra mesh -> outward oriented exterior facets (B,6).

    Includes cavity walls. Nonmanifold connectivity, duplicate cells and
    different midpoint DOFs on the same global edge are rejected at setup.
    Local tetra orientation may be either sign; input must be a valid mesh.
    """
    validate_mesh(X, cells, 10)
    vertices = cells[:, :4]
    if torch.unique(vertices.sort(1).values, dim=0).shape[0] != cells.shape[0]:
        raise ValueError("duplicate tetrahedra")
    edge_ids = torch.tensor(EDGES, device=X.device)
    pairs = vertices[:, edge_ids].reshape(-1, 2).sort(1).values
    unique, inverse = torch.unique(pairs, dim=0, return_inverse=True)
    mids = cells[:, 4:].reshape(-1)
    low = torch.full((unique.shape[0],), X.shape[0], device=X.device, dtype=torch.int64)
    high = torch.full_like(low, -1)
    low.scatter_reduce_(0, inverse, mids, reduce="amin", include_self=True)
    high.scatter_reduce_(0, inverse, mids, reduce="amax", include_self=True)
    if (low != high).any():
        raise ValueError("nonconforming P2 mesh: shared edge has different midpoint DOFs")
    # Face opposite vertex 0,1,2,3; middle nodes follow triangle edge order.
    local_faces = torch.tensor([[1, 2, 3, 7, 8, 9], [0, 2, 3, 5, 6, 9],
                                [0, 1, 3, 4, 6, 8], [0, 1, 2, 4, 5, 7]], device=X.device)
    faces = cells[:, local_faces].reshape(-1, 6)
    keys = faces[:, :3].sort(1).values
    _, inverse, counts = torch.unique(keys, dim=0, return_inverse=True, return_counts=True)
    if (counts > 2).any():
        raise ValueError("nonmanifold mesh: more than two cells share a face")
    keep = counts[inverse] == 1
    faces, opposite = faces[keep], vertices.reshape(-1)[keep]
    a, b, c = X[faces[:, 0]], X[faces[:, 1]], X[faces[:, 2]]
    area = torch.linalg.cross(b-a, c-a)
    toward_inside = (area*(X[opposite]-a)).sum(-1)
    if not torch.isfinite(toward_inside).all() or (toward_inside == 0).any():
        raise ValueError("degenerate boundary owner tetrahedron")
    flip = torch.tensor([0, 2, 1, 4, 3, 5], device=X.device)
    return torch.where((toward_inside > 0)[:, None], faces[:, flip], faces)


@dataclass(frozen=True)
class SurfaceGeometry:
    faces: Tensor                # B,6; outward ordered vertices then edges 01,02,12
    values: Tensor               # Q,6
    derivatives: Tensor          # Q,6,2
    quadrature_weights: Tensor   # Q; reference triangle area is 1/2
    reference_weights: Tensor    # B,Q; reference physical surface measure
    reference_positions: Tensor # B,Q,3
    reference_area_vectors: Tensor # B,3; cross of reference edge tangents
    node_count: int


def prepare_surface(X: Tensor, faces: Tensor, degree=4) -> SurfaceGeometry:
    """Prepare selected outward faces from extract_boundary; no empty region.

    Caller retains facet tags by selecting rows. Only straight reference
    triangles are supported; their CURRENT P2 shape may be curved.
    """
    validate_mesh(X, faces, 6)
    if torch.unique(faces[:, :3].sort(1).values, dim=0).shape[0] != faces.shape[0]:
        raise ValueError("duplicate surface faces")
    nodes = X[faces]
    edges = nodes[:, 1:3]-nodes[:, :1]
    singular = torch.linalg.svdvals(edges.transpose(-1, -2))
    tol = 100*torch.finfo(X.dtype).eps
    if (singular[:, -1] <= tol*singular[:, 0]).any():
        raise ValueError("reference triangles must be nondegenerate")
    edge_ids = torch.tensor([[0, 1], [0, 2], [1, 2]], device=X.device)
    expected = nodes[:, edge_ids].mean(2)
    if ((nodes[:, 3:]-expected).abs() > tol*singular[:, 0, None, None]).any():
        raise ValueError("reference triangle edge nodes must be midpoints in documented order")
    q, w = triangle.quadrature(degree, dtype=X.dtype, device=X.device)
    N, dN = triangle.tabulate(q)
    area = torch.linalg.cross(edges[:, 0], edges[:, 1])
    physical_weights = torch.linalg.vector_norm(area, dim=-1)[:, None]*w
    return SurfaceGeometry(faces.clone(), N, dN, w, physical_weights,
                           torch.einsum("qa,bai->bqi", N, nodes), area, X.shape[0])


def interpolate(nodal: Tensor, surface: SurfaceGeometry) -> Tensor:
    """Scalar (N,) or vector (N,k) field to (B,Q[,k]); same device/dtype."""
    if nodal.ndim not in (1, 2) or nodal.shape[0] != surface.node_count:
        raise ValueError("surface nodal field must have shape (N,) or (N,k)")
    if nodal.dtype != surface.values.dtype or nodal.device != surface.values.device:
        raise ValueError("surface field must match geometry dtype/device")
    if nodal.ndim == 1:
        return torch.einsum("qa,ba->bq", surface.values, nodal[surface.faces])
    return torch.einsum("qa,bai->bqi", surface.values, nodal[surface.faces])


def prepare_coefficient(surface, value, *, nonnegative=False):
    """Setup: scalar, nodal (N,) or quadrature (B,Q) coefficient; fixed in x.

    For a spring use nonnegative=True, including the interpolated values.
    P2 interpolation can overshoot: invalid stiffness is rejected, not clipped.
    """
    c = torch.as_tensor(value, dtype=surface.values.dtype, device=surface.values.device)
    if c.shape == (surface.node_count,):
        c = interpolate(c, surface)
    elif c.ndim == 0:
        c = c.expand_as(surface.reference_weights)
    elif c.shape != surface.reference_weights.shape:
        raise ValueError("coefficient must be scalar, nodal (N,) or quadrature (B,Q)")
    if not torch.isfinite(c).all() or (nonnegative and (c < 0).any()):
        raise ValueError("coefficient must be finite and stiffness must be nonnegative")
    return c


def area_vectors(x: Tensor, surface: SurfaceGeometry) -> Tensor:
    """a(x)=dx/dr cross dx/ds; do NOT normalize or multiply by area again."""
    tangents = torch.einsum("bai,qaj->bqij", x[surface.faces], surface.derivatives)
    return torch.linalg.cross(tangents[..., 0], tangents[..., 1])


def validate_surface(x, surface):
    """Check current surface regularity at integration points outside transforms.

    Does not establish volume det(F)>0, global orientation or injectivity.
    Call solid.validate_deformation separately on the owning volume mesh.
    """
    if x.shape != (surface.node_count, 3) or x.dtype != surface.values.dtype or x.device != surface.values.device:
        raise ValueError("x must match surface node count, dtype and device")
    area = area_vectors(x, surface)
    scale = torch.linalg.vector_norm(surface.reference_area_vectors, dim=-1)[:, None]
    if not torch.isfinite(area).all() or (torch.linalg.vector_norm(area, dim=-1) <= 100*torch.finfo(x.dtype).eps*scale).any():
        raise ValueError("current surface is nonfinite or degenerate at quadrature points")


def _coefficient(value, surface):
    c = torch.as_tensor(value, dtype=surface.values.dtype, device=surface.values.device)
    if c.ndim != 0 and c.shape != surface.reference_weights.shape:
        raise ValueError("kernel coefficient must be scalar or prepared (B,Q)")
    return c


def _scatter(local, surface):
    return local.new_zeros((surface.node_count, 3)).index_add(0, surface.faces.reshape(-1), local.reshape(-1, 3))


def pressure_force(x, surface, pressure):
    """Follower force -integral(N p cof(F) N0 dA0), positive p compressive.

    area_vectors supplies cof(F) N0 dA0 / (dr ds) via the P2 trace exactly.
    On an open loaded patch this force generally has no scalar potential.
    """
    p = _coefficient(pressure, surface)
    traction = -p[..., None]*area_vectors(x, surface)
    return _scatter(torch.einsum("q,qa,bqi->bai", surface.quadrature_weights, surface.values, traction), surface)


def spring_energy(x, surface, beta):
    """1/2 integral_ref beta |x-X|^2 dA0; fixed reference target in all axes."""
    u = interpolate(x, surface)-surface.reference_positions
    return 0.5*(surface.reference_weights*_coefficient(beta, surface)*u.square().sum(-1)).sum()


def spring_force(x, surface, beta):
    """Reference-area foundation spring force -integral_ref N beta (x-X)."""
    traction = -_coefficient(beta, surface)[..., None]*(interpolate(x, surface)-surface.reference_positions)
    return _scatter(torch.einsum("bq,qa,bqi->bai", surface.reference_weights, surface.values, traction), surface)
