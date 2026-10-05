"""MAC finite-volume operators for the Ma et al. real-LV open box.

Velocity diffusion uses homogeneous Neumann data; pressure is zero on the
physical box faces, half a cell from pressure centres. Normal boundary faces
have half control volumes. D and G are negative weighted adjoints, so DG is
the Dirichlet Poisson stencil, including its boundary factor of two.
These operators must not be mixed with closed-wall/mean-zero pressure code.
"""
import torch
import torch.nn.functional as nnf
from .grid import slab, divergence


def gradient(p, spacing):
    result = []
    for axis, h in enumerate(spacing):
        low = 2*p[slab(axis, 0, 1)]/h
        high = -2*p[slab(axis, -1, None)]/h
        result.append(torch.cat((low, torch.diff(p, dim=axis)/h, high), dim=axis))
    return tuple(result)


def negative_laplacian(p, spacing):
    return -divergence(gradient(p, spacing), spacing)


def face_weights(grid, component, like):
    w = torch.ones(grid.face_shape(component), device=like.device, dtype=like.dtype)
    w[slab(component, 0, 1)] = .5
    w[slab(component, -1, None)] = .5
    return w*grid.volume


def velocity_laplacian(u, component, spacing):
    result = torch.zeros_like(u)
    for axis, h in enumerate(spacing):
        # Cell-centred tangential fields: constant ghost extension. A normal
        # field lies on the wall itself: reflection across its boundary node.
        first = u[slab(axis, 1, 2)] if axis == component else u[slab(axis, 0, 1)]
        last = u[slab(axis, -2, -1)] if axis == component else u[slab(axis, -1, None)]
        before = torch.cat((first, u[slab(axis, None, -1)]), axis)
        after = torch.cat((u[slab(axis, 1, None)], last), axis)
        result = result+(before-2*u+after)/h**2
    return result


def pressure_diagonal(shape, spacing, like):
    d = torch.full(shape, 2*sum(h**-2 for h in spacing), device=like.device, dtype=like.dtype)
    for axis, h in enumerate(spacing):
        d[slab(axis, 0, 1)] += h**-2
        d[slab(axis, -1, None)] += h**-2
    return d


def velocity_diagonal(shape, component, spacing, like):
    d = torch.full(shape, 2*sum(h**-2 for h in spacing), device=like.device, dtype=like.dtype)
    for axis, h in enumerate(spacing):
        if axis != component:
            d[slab(axis, 0, 1)] -= h**-2
            d[slab(axis, -1, None)] -= h**-2
    return d


def sample_face(u, points, grid, component):
    """Trilinear interpolation of one staggered component, constant extension.

    Tensor axes are x,y,z; grid_sample expects coordinates in W,H,D order.
    align_corners=False maps lattice node i to 2*(i+.5)/N-1.
    """
    index = (points-points.new_tensor(grid.face_origin(component)))/points.new_tensor(grid.spacing)
    normalized = 2*(index+.5)/points.new_tensor(grid.face_shape(component))-1
    query = normalized.flip(-1).unsqueeze(0)
    return nnf.grid_sample(u[None, None], query, mode='bilinear',
                          padding_mode='border', align_corners=False)[0, 0]


class SemiLagrangian:
    """First-order characteristic tracing, trilinear MAC interpolation.

    The paper identifies semi-Lagrangian convection but does not publish a
    tracing/interpolation prescription. This is an explicit reconstruction
    choice, not a claim of reproducing the authors' unpublished implementation.
    """
    def __init__(self, grid, like, fused=False):
        self.grid = grid
        self.coordinates = tuple(grid.coordinates(c, device=like.device, dtype=like.dtype)
                                 for c in range(3))
        from .execution import tensor_kernel
        self.kernel = tensor_kernel(self._advect, like.device) if fused else self._advect

    def _advect(self, velocity, dt):
        advected = []
        for c, points in enumerate(self.coordinates):
            speed = torch.stack(tuple(velocity[c] if k == c else sample_face(velocity[k], points, self.grid, k)
                                      for k in range(3)), -1)
            departure = points-dt*speed
            advected.append(sample_face(velocity[c], departure, self.grid, c))
        return tuple(advected)

    def __call__(self, velocity, dt):
        return self.kernel(velocity, dt)
