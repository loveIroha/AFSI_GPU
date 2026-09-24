"""Four-point Peskin transfer on a uniform 3D velocity-node lattice.

All vector fields are (number_of_nodes, 3), x-fast ordering. Inputs to spread
are INTEGRATED solid nodal forces, with no second solid quadrature weight.
The prepared stencil must be rebuilt whenever solid coordinates change.
"""
from dataclasses import dataclass
from math import isfinite, prod
from numbers import Integral
import torch
from torch import Tensor


@dataclass(frozen=True)
class UniformGrid:
    """shape=(nx,ny,nz) counts velocity nodes, not fluid finite elements.

    For a Q2 velocity mesh with n cells along an axis: shape=2*n+1 and
    spacing=L/(2*n). No fluid discretization is implied by this container.
    """
    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float] = (0., 0., 0.)

    def __post_init__(self):
        if (len(self.shape) != 3 or any(not isinstance(n, Integral) or isinstance(n, bool)
                                      or n < 4 for n in self.shape)):
            raise ValueError("grid shape must contain three integer node counts >= 4")
        if len(self.spacing) != 3 or any(not isfinite(h) or h <= 0 for h in self.spacing):
            raise ValueError("grid spacing must contain three finite positive lengths")
        if not isfinite(prod(self.spacing)) or prod(self.spacing) <= 0:
            raise ValueError("grid lattice volume must be finite and positive")
        if len(self.origin) != 3 or any(not isfinite(o) for o in self.origin):
            raise ValueError("grid origin must contain three finite coordinates")
        object.__setattr__(self, 'shape', tuple(int(n) for n in self.shape))
        object.__setattr__(self, 'spacing', tuple(float(h) for h in self.spacing))
        object.__setattr__(self, 'origin', tuple(float(o) for o in self.origin))

    @property
    def node_count(self):
        return prod(self.shape)

    @property
    def cell_volume(self):
        """Velocity-lattice volume h_x*h_y*h_z, not a Q2 cell volume."""
        return prod(self.spacing)

    def coordinates(self, *, dtype=torch.float64, device='cpu'):
        nx, ny, nz = self.shape
        k, j, i = torch.meshgrid(torch.arange(nz, device=device),
                                 torch.arange(ny, device=device),
                                 torch.arange(nx, device=device), indexing='ij')
        ijk = torch.stack((i, j, k), -1).reshape(-1, 3).to(dtype)
        return ijk*ijk.new_tensor(self.spacing)+ijk.new_tensor(self.origin)


def peskin4(distance: Tensor) -> Tensor:
    """Dimensionless even kernel phi(r); support |r|<2, phi(0)=1/2.

    Clamp arguments in BOTH inactive branches before sqrt: torch.where alone
    does not prevent invalid branch derivatives from poisoning autograd.
    """
    r = distance.abs()
    a, b = r.clamp(max=1), r.clamp(min=1, max=2)
    inner = (3-2*a+torch.sqrt(1+4*a-4*a.square()))/8
    outer = (5-2*b-torch.sqrt(-7+12*b-4*b.square()))/8
    return torch.where(r < 1, inner, torch.where(r < 2, outer, torch.zeros_like(r)))


@dataclass(frozen=True)
class IBStencil:
    grid: UniformGrid
    indices: Tensor   # Ns,64, int64
    weights: Tensor   # Ns,64, dimensionless product of three phi factors


def prepare_stencil(x: Tensor, grid: UniformGrid) -> IBStencil:
    """Build 64 neighbors per solid node on x.device; no dense Ns*Nf matrix.

    Require the entire four-node stencil in each axis inside the grid. No
    clipping, silent loss of weights, periodic wrapping or renormalization.
    Validation performs host synchronization; keep outside compiled kernels.
    """
    if (x.ndim != 2 or x.shape[1] != 3 or x.shape[0] == 0 or
            x.dtype not in (torch.float32, torch.float64) or not torch.isfinite(x).all()):
        raise ValueError("positions must be finite nonempty (N,3) float32/float64")
    scaled = (x-x.new_tensor(grid.origin))/x.new_tensor(grid.spacing)
    # Validate before float -> integer conversion, including extreme inputs.
    limits = x.new_tensor(grid.shape)
    if not torch.isfinite(scaled).all() or (scaled < 1).any() or (scaled >= limits-2).any():
        raise ValueError("four-point IB support leaves grid; enlarge the fluid box")
    base = torch.floor(scaled-1).to(torch.int64)
    axis = torch.arange(4, device=x.device)
    offsets = torch.cartesian_prod(axis, axis, axis)
    nodes = base[:, None, :]+offsets[None, :, :]
    weights = peskin4(scaled[:, None, :]-nodes.to(x.dtype)).prod(-1)
    nx, ny, _ = grid.shape
    indices = nodes[..., 0]+nx*(nodes[..., 1]+ny*nodes[..., 2])
    return IBStencil(grid, indices, weights)


def _check_field(field, count, stencil):
    if (field.shape != (count, 3) or field.dtype != stencil.weights.dtype or
            field.device != stencil.weights.device):
        raise ValueError("field must be (N,3) on stencil device with matching dtype")


def interpolate(velocity: Tensor, stencil: IBStencil) -> Tensor:
    """U_s=H u_f, a smoothed velocity interpolation, not P2 point evaluation."""
    _check_field(velocity, stencil.grid.node_count, stencil)
    return (velocity[stencil.indices]*stencil.weights[..., None]).sum(1)


def spread_load(nodal_force: Tensor, stencil: IBStencil) -> Tensor:
    """Return dual fluid-node load b=H.T g; u.T b == (H u).T g.

    If b is inserted directly into a finite-element weak RHS, do NOT multiply
    it by a mass matrix. This is distinct from an interpolated force density.
    """
    _check_field(nodal_force, stencil.indices.shape[0], stencil)
    local = stencil.weights[..., None]*nodal_force[:, None, :]
    return nodal_force.new_zeros((stencil.grid.node_count, 3)).index_add(
        0, stencil.indices.reshape(-1), local.reshape(-1, 3))


def spread_density(nodal_force: Tensor, stencil: IBStencil) -> Tensor:
    """Return f=H.T g/(hx*hy*hz), matching afsi's nodal density scaling.

    Adjoint under the uniform lattice volume inner product. This does not
    claim adjointness under a consistent fluid FE mass matrix M_f.
    """
    return spread_load(nodal_force, stencil)/stencil.grid.cell_volume
