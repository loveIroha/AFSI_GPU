"""Three-dimensional MAC fields, ordered (x, y, z), in a closed box."""
from dataclasses import dataclass
from math import isfinite, prod
import torch


def slab(axis, start=None, stop=None):
    key = [slice(None)]*3
    key[axis] = slice(start, stop)
    return tuple(key)


@dataclass(frozen=True)
class MACGrid:
    shape: tuple[int, int, int]
    lengths: tuple[float, float, float]
    origin: tuple[float, float, float] = (0., 0., 0.)

    def __post_init__(self):
        if len(self.shape) != 3 or any(type(n) is not int or n < 4 for n in self.shape):
            raise ValueError('MAC requires three integer cell counts >= 4')
        if (len(self.lengths) != 3 or len(self.origin) != 3 or
                any(not isfinite(v) or v <= 0 for v in self.lengths) or
                any(not isfinite(v) for v in self.origin)):
            raise ValueError('finite origin and positive box lengths required')

    @property
    def spacing(self):
        return tuple(L/n for L, n in zip(self.lengths, self.shape))

    @property
    def volume(self):
        return prod(self.spacing)

    def face_shape(self, component):
        return tuple(n+(axis == component) for axis, n in enumerate(self.shape))

    def face_origin(self, component):
        return tuple(o+(0. if axis == component else .5*h)
                     for axis, (o, h) in enumerate(zip(self.origin, self.spacing)))

    def zeros(self, *, device='cpu', dtype=torch.float64):
        return tuple(torch.zeros(self.face_shape(c), device=device, dtype=dtype) for c in range(3))

    def coordinates(self, component=None, *, device='cpu', dtype=torch.float64):
        shape = self.shape if component is None else self.face_shape(component)
        origin = (tuple(o+.5*h for o, h in zip(self.origin, self.spacing))
                  if component is None else self.face_origin(component))
        axes = [o+h*torch.arange(n, device=device, dtype=dtype)
                for n, h, o in zip(shape, self.spacing, origin)]
        return torch.stack(torch.meshgrid(*axes, indexing='ij'), -1)

    def check_velocity(self, velocity):
        if len(velocity) != 3:
            raise ValueError('three MAC velocity components required')
        for c, u in enumerate(velocity):
            if (u.shape != self.face_shape(c) or u.dtype not in (torch.float32, torch.float64)
                    or u.device != velocity[0].device or u.dtype != velocity[0].dtype):
                raise ValueError('MAC field shape/device/dtype mismatch')


def zero_normal(velocity):
    result = tuple(u.clone() for u in velocity)
    for c, u in enumerate(result):
        u[slab(c, 0, 1)] = 0.
        u[slab(c, -1, None)] = 0.
    return result


def divergence(velocity, spacing):
    return sum(torch.diff(u, dim=c)/spacing[c] for c, u in enumerate(velocity))


def gradient(pressure, spacing):
    """Homogeneous Neumann gradient: boundary-normal faces are exactly zero."""
    result = []
    for c, h in enumerate(spacing):
        shape = list(pressure.shape)
        shape[c] += 1
        g = pressure.new_zeros(shape)
        g[slab(c, 1, -1)] = torch.diff(pressure, dim=c)/h
        result.append(g)
    return tuple(result)


def negative_laplacian(p, spacing):
    """A=-D G, including the one-sided Neumann boundary rows."""
    out = torch.zeros_like(p)
    for c, h in enumerate(spacing):
        d = torch.diff(p, dim=c)/(h*h)
        out[slab(c, None, -1)] -= d
        out[slab(c, 1, None)] += d
    return out


def velocity_laplacian(u, component, spacing):
    """Normal wall values zero; odd tangential ghosts enforce no slip."""
    out = torch.zeros_like(u)
    for axis, h in enumerate(spacing):
        left = torch.cat((-u[slab(axis, 0, 1)], u[slab(axis, None, -1)]), axis)
        right = torch.cat((u[slab(axis, 1, None)], -u[slab(axis, -1, None)]), axis)
        out += (left-2*u+right)/(h*h)
    out[slab(component, 0, 1)] = 0.
    out[slab(component, -1, None)] = 0.
    return out


def convection(velocity, spacing):
    """Centered conservative MAC momentum fluxes, including closed-wall fluxes.

    Extends the standard MAC-taichi predictor stencil to three dimensions.
    This operator is paired with explicit viscosity and a checked time step.
    """
    result = []
    for c, u in enumerate(velocity):
        out = torch.zeros_like(u)
        for axis, h in enumerate(spacing):
            if axis == c:
                flux = (.5*(u[slab(c, 1, None)]+u[slab(c, None, -1)])).square()
                out[slab(c, 1, -1)] += torch.diff(flux, dim=c)/h
            else:
                other = velocity[axis]
                # Cross-component velocity at edges of the c-momentum volume.
                padded = torch.cat((-other[slab(c, 0, 1)], other,
                                    -other[slab(c, -1, None)]), c)
                advector = .5*(padded[slab(c, 1, None)]+padded[slab(c, None, -1)])
                padded_u = torch.cat((-u[slab(axis, 0, 1)], u,
                                      -u[slab(axis, -1, None)]), axis)
                carried = .5*(padded_u[slab(axis, 1, None)]+padded_u[slab(axis, None, -1)])
                out += torch.diff(advector*carried, dim=axis)/h
        out[slab(c, 0, 1)] = 0.
        out[slab(c, -1, None)] = 0.
        result.append(out)
    return tuple(result)
