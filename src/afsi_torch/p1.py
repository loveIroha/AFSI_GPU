"""P1 tetrahedra/triangular traces compatible with the existing FE/MAC API."""
from itertools import permutations
import torch
from .solid import P2Geometry
from .boundary import SurfaceGeometry
from .tetrahedron import validate_mesh
from .quadrature import tetrahedron_rule
from . import triangle
from .mechanics import determinant3


def tetra_rule(degree, like):
    """Positive degree-2 (4 points) or degree-5 (15 points) simplex rule."""
    if degree == 2:
        b = (5-5**.5)/20
        orbits = [((1-3*b, b, b, b), 1/24)]
    elif degree == 5:
        # Keast's positive degree-5 rule; barycentric permutations.
        orbits = [((.25,)*4, .030283678097089182),
                  ((0., 1/3, 1/3, 1/3), .006026785714285717),
                  ((8/11, 1/11, 1/11, 1/11), .011645249086028967),
                  ((.4334498464263357, .4334498464263357,
                    .0665501535736643, .0665501535736643), .010949141561386449)]
    else:
        return tetrahedron_rule(degree, dtype=like.dtype, device=like.device)
    points, weights = [], []
    for orbit, weight in orbits:
        for bary in sorted(set(permutations(orbit))):
            points.append(bary[1:]); weights.append(weight)
    return like.new_tensor(points), like.new_tensor(weights)


def prepare_p1(X, cells, degree=5):
    validate_mesh(X, cells, 4)
    vertices = X[cells]
    D = (vertices[:, 1:]-vertices[:, :1]).transpose(-1, -2)
    singular = torch.linalg.svdvals(D)
    if (singular[:, -1] <= 100*torch.finfo(X.dtype).eps*singular[:, 0]).any():
        raise ValueError('degenerate reference tetrahedron')
    q, w = tetra_rule(degree, X)
    N = torch.cat((1-q.sum(-1, keepdim=True), q), -1)
    dN = X.new_tensor([[-1., -1., -1.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    grad = dN @ torch.linalg.inv(D)
    # Expanded view: no duplicated gradients for a constant P1 cell gradient.
    return P2Geometry(cells.clone(), N, grad[:, None].expand(-1, len(q), -1, -1),
                      determinant3(D).abs()[:, None]*w, len(X))


def prepare_surface(X, faces, degree=5):
    validate_mesh(X, faces, 3)
    nodes = X[faces]
    edges = nodes[:, 1:]-nodes[:, :1]
    area = torch.linalg.cross(edges[:, 0], edges[:, 1])
    if (torch.linalg.vector_norm(area, dim=-1) <= 0).any():
        raise ValueError('degenerate reference triangle')
    q, w = triangle.quadrature(degree, dtype=X.dtype, device=X.device)
    N = torch.cat((1-q.sum(-1, keepdim=True), q), -1)
    dN = X.new_tensor([[-1., -1.], [1., 0.], [0., 1.]]).expand(len(q), -1, -1)
    return SurfaceGeometry(faces.clone(), N, dN, w,
                           torch.linalg.vector_norm(area, dim=-1)[:, None]*w,
                           torch.einsum('qa,bai->bqi', N, nodes), area, len(X))


def cavity_rim(faces):
    incidences = {}
    for a, b, c in faces.detach().cpu().tolist():
        for edge in ((a, b), (b, c), (c, a)):
            incidences.setdefault(tuple(sorted(edge)), []).append(edge)
    rim = []
    for owners in incidences.values():
        if len(owners) == 1:
            rim.append(owners[0])
        elif len(owners) != 2 or owners[0] != owners[1][::-1]:
            raise ValueError('endocardium must be an oriented manifold')
    following = dict(rim)
    if len(rim) < 3 or len(following) != len(rim) or set(following) != set(following.values()):
        raise ValueError('endocardium must have one open base loop')
    current, visited = rim[0][0], set()
    while current not in visited:
        visited.add(current); current = following[current]
    if current != rim[0][0] or len(visited) != len(rim):
        raise ValueError('multiple endocardial base loops are unsupported')
    return torch.tensor(rim, dtype=torch.int64, device=faces.device)


def cavity_volume(x, faces, rim):
    """Signed P1 cavity volume with a measurement-only mean-rim triangle fan."""
    # Recenter for translation invariance and better cancellation behavior.
    center = x[rim[:, 0]].mean(0)
    nodes = x[faces]-center
    area = torch.linalg.cross(nodes[:, 1]-nodes[:, 0], nodes[:, 2]-nodes[:, 0])
    # The recentered planar triangles of the virtual fan have zero flux.
    return -(nodes.mean(1)*area).sum()/6
