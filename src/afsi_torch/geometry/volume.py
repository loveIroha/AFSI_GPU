"""Signed wall volume and P2 cavity volume with a measurement-only virtual cap."""
from dataclasses import dataclass
import torch
from .. import boundary as bd, triangle
from ..mechanics import determinant3


def signed_cell_volumes(X, cells):
    corners = X[cells[:, :4]]
    return determinant3((corners[:, 1:]-corners[:, :1]).transpose(-1, -2))/6


@dataclass(frozen=True)
class CavityGeometry:
    endo: bd.SurfaceGeometry
    rim: torch.Tensor          # directed ENDO boundary edges: a,b,mid
    values: torch.Tensor      # reference triangle P2 tables for virtual cap
    derivatives: torch.Tensor
    weights: torch.Tensor


def prepare_cavity(X, endo_faces):
    """Validate a single manifold base loop once, outside differentiation.

    ENDO normals point into the cavity (outward from solid). Cap boundary
    orientation follows ENDO, so cap + reversed ENDO form a closed cavity.
    """
    surface = bd.prepare_surface(X, endo_faces, degree=4)
    edges = {}
    for face in endo_faces.detach().cpu().tolist():
        for a, b, m in ((face[0], face[1], face[3]), (face[1], face[2], face[5]),
                        (face[2], face[0], face[4])):
            edges.setdefault(tuple(sorted((a, b))), []).append((a, b, m))
    rim = []
    for incidences in edges.values():
        if len(incidences) == 1:
            rim.append(incidences[0])
        elif (len(incidences) != 2 or incidences[0] !=
              (incidences[1][1], incidences[1][0], incidences[1][2])):
            raise ValueError('ENDO must be an oriented conforming manifold')
    following = {a: b for a, b, _ in rim}
    if (len(rim) < 3 or len(following) != len(rim) or
            set(following) != set(following.values())):
        raise ValueError('ENDO must have one simple open base loop')
    start = rim[0][0]
    visited, current = set(), start
    while current not in visited:
        visited.add(current)
        current = following[current]
    if current != start or len(visited) != len(rim):
        raise ValueError('multiple base loops are not supported')
    q, w = triangle.quadrature(4, dtype=X.dtype, device=X.device)
    N, dN = triangle.tabulate(q)
    return CavityGeometry(surface, torch.tensor(rim, dtype=torch.int64, device=X.device), N, dN, w)


def cavity_volume(x, geometry):
    """Signed enclosed volume, differentiable for fixed topology.

    Use a P2 triangle fan from the mean rim-vertex center, with actual P2 rim
    edges and straight radial edges. The fan is for measurement ONLY, never
    added to the mechanical pressure surface. Moving nonplanar rims use this
    explicitly defined fan, not a unique anatomical valve plane.
    """
    surface = geometry.endo
    positions = bd.interpolate(x, surface)
    area = bd.area_vectors(x, surface)
    wall_flux = -(surface.quadrature_weights*(positions*area).sum(-1)).sum()/3
    a, b, mid = (x[geometry.rim[:, i]] for i in range(3))
    center = a.mean(0).expand_as(a)
    cap_nodes = torch.stack((center, a, b, .5*(center+a), .5*(center+b), mid), 1)
    cap_x = torch.einsum('qa,bai->bqi', geometry.values, cap_nodes)
    tangents = torch.einsum('bai,qaj->bqij', cap_nodes, geometry.derivatives)
    cap_area = torch.linalg.cross(tangents[..., 0], tangents[..., 1])
    cap_flux = (geometry.weights*(cap_x*cap_area).sum(-1)).sum()/3
    return wall_flux+cap_flux
