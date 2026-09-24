"""Shared velocity/pressure DOF lattices on an axis-aligned box, lengths in cm."""
from dataclasses import dataclass
from math import isfinite
from numbers import Integral
import torch
from ..ib import UniformGrid


def _indices(shape, device):
    k, j, i = torch.meshgrid(*(torch.arange(n, device=device) for n in shape[::-1]), indexing='ij')
    return torch.stack((i, j, k), -1).reshape(-1, 3)


@dataclass(frozen=True)
class BoxMesh:
    counts: tuple[int, int, int]
    lengths: tuple[float, float, float]
    origin: tuple[float, float, float]
    velocity_coordinates: torch.Tensor
    pressure_coordinates: torch.Tensor
    velocity_cells: torch.Tensor
    pressure_cells: torch.Tensor
    velocity_boundary: torch.Tensor
    pressure_boundary: torch.Tensor

    @property
    def cell_sizes(self):
        return tuple(L/n for L, n in zip(self.lengths, self.counts))

    @property
    def velocity_grid(self):
        """IB lattice; requires >=2 cells on EVERY axis for four-point support."""
        return UniformGrid(tuple(2*n+1 for n in self.counts),
                           tuple(h/2 for h in self.cell_sizes), self.origin)


def create_box(counts=(2, 2, 2), lengths=(1., 1., 1.), origin=(0., 0., 0.),
               *, device='cpu', dtype=torch.float64):
    if (len(counts) != 3 or any(not isinstance(n, Integral) or isinstance(n, bool)
                              or n < 1 for n in counts)):
        raise ValueError('counts must be three positive integers')
    if len(lengths) != 3 or any(not isfinite(L) or L <= 0 for L in lengths):
        raise ValueError('lengths must be three finite positive values')
    if len(origin) != 3 or any(not isfinite(x) for x in origin):
        raise ValueError('origin must be three finite values')
    if dtype not in (torch.float32, torch.float64):
        raise ValueError('floating dtype required')
    counts, lengths, origin = tuple(map(int, counts)), tuple(map(float, lengths)), tuple(map(float, origin))
    element_indices = _indices(counts, device)
    coords, cells, boundaries = [], [], []
    for degree in (2, 1):
        shape = tuple(degree*n+1 for n in counts)
        indices = _indices(shape, device)
        coordinates = indices.to(dtype)*torch.tensor(
            [L/(degree*n) for L, n in zip(lengths, counts)], device=device, dtype=dtype)
        coordinates = coordinates+coordinates.new_tensor(origin)
        local = degree*element_indices[:, None]+_indices((degree+1,)*3, device)[None]
        connectivity = local[..., 0]+shape[0]*(local[..., 1]+shape[1]*local[..., 2])
        boundary = ((indices == 0) | (indices == indices.new_tensor(shape)-1)).any(-1)
        coords.append(coordinates)
        cells.append(connectivity)
        boundaries.append(boundary)
    return BoxMesh(counts, lengths, origin, *coords, *cells, *boundaries)
