"""P1 harmonic transmural coordinate and ellipsoidal fiber/sheet construction.

Matches the mathematical recipe of cardiac-geometries' lv_ellipsoid fibers
with long_axis=0. Does not reproduce unknown external projected fiber files.
"""
from dataclasses import asdict
from math import pi
import torch
from .ellipsoid import ENDO, EPI
from .fibers import FiberField
from ..tetrahedron import EDGES
from ..fluid.solvers import pcg, SolverOptions


def ellipsoidal_frame(X, config, transmural):
    if config.long_axis != 'x':
        raise ValueError('AFSI ellipsoid frame requires long_axis=x')
    t = transmural
    rs = config.inner_axes[0]+t*(config.outer_axes[0]-config.inner_axes[0])
    rl = config.inner_axes[2]+t*(config.outer_axes[2]-config.inner_axes[2])
    xyz = X-X.new_tensor(config.center)
    mu = torch.atan2(torch.linalg.vector_norm(xyz[:,1:], dim=-1)/rs, xyz[:,0]/rl)
    theta = pi-torch.atan2(xyz[:,2], -xyz[:,1])
    # A tangent frame has an unavoidable polar gauge singularity. Select the
    # theta=0 limiting frame exactly at the apex; no radial apical escape.
    theta = torch.where(torch.linalg.vector_norm(xyz[:,1:], dim=-1) < 1e-14, 0., theta)
    meridian = torch.stack((-rl*mu.sin(), rs*mu.cos()*theta.cos(), rs*mu.cos()*theta.sin()), -1)
    circum = torch.stack((torch.zeros_like(t), -theta.sin(), theta.cos()), -1)
    unit = lambda v: v/torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    meridian = unit(meridian)
    angle = (1-2*t)*pi/2
    fiber = unit(angle.sin()[:,None]*meridian+angle.cos()[:,None]*circum)
    normal = unit(torch.linalg.cross(meridian, circum))
    sheet = unit(torch.linalg.cross(fiber, normal))
    return FiberField(fiber, sheet, t, angle, torch.zeros_like(t))


@torch.no_grad()
def laplace_ellipsoid_fibers(mesh):
    """Assemble P1 Laplace CSR once; solve t=0 endo, t=1 epi, natural base.

    Generated meshes have leading P1 vertices and shared appended P2 edges.
    The P1 scalar is interpolated to P2 before evaluating the analytic frame.
    """
    X, cells, n = mesh.X, mesh.cells[:,:4], mesh.vertex_count
    if (cells >= n).any():
        raise ValueError('expected generated P1 vertex numbering')
    d = (X[cells[:,1:]]-X[cells[:,:1]]).transpose(-1,-2)
    grad = X.new_tensor([[-1,-1,-1],[1,0,0],[0,1,0],[0,0,1]])@torch.linalg.inv(d)
    local = (grad@grad.transpose(-1,-2))*(torch.linalg.det(d).abs()/6)[:,None,None]
    rows = cells[:,:,None].expand(-1,4,4).reshape(-1)
    cols = cells[:,None,:].expand(-1,4,4).reshape(-1)
    matrix = torch.sparse_coo_tensor(torch.stack((rows,cols)), local.reshape(-1),
                                     (n,n),check_invariants=True).coalesce().to_sparse_csr()
    diagonal = X.new_zeros(n).index_add_(0,cells.reshape(-1),local.diagonal(dim1=-2,dim2=-1).reshape(-1))
    endo = mesh.surface(ENDO)[:,:3].unique()
    epi = mesh.surface(EPI)[:,:3].unique()
    if torch.isin(endo, epi).any():
        raise ValueError('endo/epi Dirichlet nodes overlap')
    fixed = torch.zeros(n, dtype=torch.bool, device=X.device)
    fixed[endo], fixed[epi] = True, True
    values = X.new_zeros(n)
    values[epi] = 1
    action = lambda a: torch.sparse.mm(matrix, a[:,None])[:,0]
    scalar, info = pcg(action, X.new_zeros(n), diagonal, fixed=fixed, values=values,
                      options=SolverOptions(rtol=1e-12,atol=1e-13,max_iterations=5000,check_every=8))
    # Do not clamp a discrete harmonic solution: obtuse cells can violate a
    # maximum principle, and silent clipping changes the reference recipe.
    mids = scalar[cells[:,torch.tensor(EDGES,device=X.device)]].mean(-1)
    t = X.new_zeros(len(X))
    count = X.new_zeros(len(X))
    t.index_add_(0,mesh.cells[:,4:].reshape(-1),mids.reshape(-1))
    count.index_add_(0,mesh.cells[:,4:].reshape(-1),torch.ones_like(mids).reshape(-1))
    t[n:] /= count[n:]
    t[:n] = scalar
    return ellipsoidal_frame(X, mesh.config, t), dict(asdict(info),
        minimum=float(t.min()), maximum=float(t.max()))

