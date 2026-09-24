"""Quadrature-based P2 mechanics on straight-sided reference tetrahedra.

Current coordinates may vary quadratically. Curved reference geometry is
explicitly rejected until isoparametric geometry validation is implemented.
"""
from dataclasses import dataclass
import torch
from torch import Tensor

from .materials import (neo_hookean_energy, neo_hookean_pk1,
                        guccione_energy as guccione_density, guccione_pk1,
                        active_potential, active_pk1)
from .mechanics import determinant3
from .quadrature import tetrahedron_rule, prepare_rule
from .tetrahedron import EDGES, tabulate, validate_mesh


@dataclass(frozen=True)
class P2Geometry:
    cells: Tensor           # E,10
    values: Tensor          # Q,10
    gradients: Tensor       # E,Q,10,3 (reference physical derivatives)
    weights: Tensor         # E,Q (reference physical integration weights)
    node_count: int


def prepare_p2(X: Tensor, cells: Tensor, degree=4, *, quadrature=None) -> P2Geometry:
    """Precompute reference data; optional (points, weights) overrides degree."""
    validate_mesh(X, cells, 10)
    nodes = X[cells]
    vertices = nodes[:, :4]
    D = (vertices[:, 1:] - vertices[:, :1]).transpose(-1, -2)
    # Reject near-singular geometry using a dimensionless singular-value ratio.
    singular = torch.linalg.svdvals(D)
    tol = 100 * torch.finfo(X.dtype).eps
    if (singular[:, -1] <= tol * singular[:, 0]).any():
        raise ValueError("Reference tetrahedra must be nondegenerate and well conditioned")
    edges = torch.tensor(EDGES, device=X.device)
    expected = vertices[:, edges].mean(dim=2)
    scale = singular[:, 0, None, None]
    if ((nodes[:, 4:] - expected).abs() > tol * scale).any():
        raise ValueError("P2 reference edge nodes must be midpoints in the documented ordering")
    points, weights = (tetrahedron_rule(degree, dtype=X.dtype, device=X.device)
                       if quadrature is None else prepare_rule(quadrature, 3, X))
    N, dN = tabulate(points)
    gradients = torch.einsum("qaj,ejk->eqak", dN, torch.linalg.inv(D))
    physical_weights = determinant3(D).abs()[:, None] * weights
    return P2Geometry(cells.clone(), N, gradients, physical_weights, X.shape[0])


def deformation_gradient(x: Tensor, geometry: P2Geometry) -> Tensor:
    """F[e,q,i,J]=sum_a x[e,a,i] grad_X(N)[e,q,a,J]."""
    return torch.einsum("eai,eqaJ->eqiJ", x[geometry.cells], geometry.gradients)


def validate_deformation(x: Tensor, geometry: P2Geometry) -> None:
    """Sampled positivity at quadrature points, NOT a global injectivity proof."""
    if x.shape != (geometry.node_count, 3) or x.device != geometry.gradients.device or x.dtype != geometry.gradients.dtype:
        raise ValueError("x must match the prepared mesh shape, device and dtype")
    J = determinant3(deformation_gradient(x, geometry))
    if not torch.isfinite(J).all() or (J <= 0).any():
        raise ValueError("det(F) must be finite and positive at every quadrature point")


def energy(x: Tensor, geometry: P2Geometry, mu: float, lam: float) -> Tensor:
    """Total strain energy; call validate_deformation outside torch.func."""
    F = deformation_gradient(x, geometry)
    return (geometry.weights * neo_hookean_energy(F, mu, lam)).sum()


def stress_force(x: Tensor, geometry: P2Geometry, mu: float, lam: float) -> Tensor:
    """Integrated nodal force, g=-dE/dx; no division by mass/volume."""
    P = neo_hookean_pk1(deformation_gradient(x, geometry), mu, lam)
    return assemble_pk1(P, geometry)


def assemble_pk1(P: Tensor, geometry: P2Geometry) -> Tensor:
    """Assemble -integral(P grad(N)) from batched (E,Q,3,3) PK1 stresses."""
    local = -torch.einsum("eq,eqiJ,eqaJ->eai", geometry.weights, P, geometry.gradients)
    return P.new_zeros((geometry.node_count, 3)).index_add(0, geometry.cells.reshape(-1), local.reshape(-1, 3))


def guccione_energy(x, geometry, fields, parameters):
    """Passive energy plus fixed-Ta active potential, not cycle stored energy.

    fields must be prepared on the SAME geometry and held fixed in x derivatives.
    Validate det(F)>0 separately, as for the Neo-Hookean kernel.
    """
    F = deformation_gradient(x, geometry)
    W = guccione_density(F, fields.fiber, fields.sheet, fields.normal, parameters)
    W = W + active_potential(F, fields.fiber, fields.tension)
    return (geometry.weights*W).sum()


def guccione_force(x, geometry, fields, parameters):
    """Integrated passive+active force with afsi's negative weak-form sign."""
    F = deformation_gradient(x, geometry)
    P = guccione_pk1(F, fields.fiber, fields.sheet, fields.normal, parameters)
    P = P + active_pk1(F, fields.fiber, fields.tension)
    return assemble_pk1(P, geometry)
