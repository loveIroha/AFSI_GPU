"""Quadrature/gather/scatter operator actions without assembled global matrices.

Positive stiffness, divergence D=(q,div u), gradient G=(v,grad p).
Before boundary elimination G != -D.T. Coefficients rho and mu are left to
the caller; viscosity here is componentwise grad-grad, as in the afsi Chorin
example, not the symmetric-strain form. No BCs or pressure gauge are imposed.
"""
from dataclasses import dataclass
from math import prod
import torch
from .mesh import BoxMesh
from .elements import basis, quadrature


@dataclass(frozen=True)
class FluidOperators:
    mesh: BoxMesh
    weights: torch.Tensor
    N: torch.Tensor
    dN: torch.Tensor
    Q: torch.Tensor
    dQ: torch.Tensor

    def _field(self, x, pressure=False, vector=False):
        coordinates = self.mesh.pressure_coordinates if pressure else self.mesh.velocity_coordinates
        shape = (len(coordinates), 3) if vector else (len(coordinates),)
        if x.shape != shape or x.dtype != coordinates.dtype or x.device != coordinates.device:
            raise ValueError(f'expected field {shape} on mesh device with matching dtype')

    def _scatter(self, local, pressure=False):
        cells = self.mesh.pressure_cells if pressure else self.mesh.velocity_cells
        count = len(self.mesh.pressure_coordinates if pressure else self.mesh.velocity_coordinates)
        tail = local.shape[2:]
        return local.new_zeros((count,)+tail).index_add(0, cells.reshape(-1), local.reshape((-1,)+tail))

    def velocity_mass(self, u):
        """Consistent M_v u, also the weak load of a Q2 nodal force density."""
        self._field(u, vector=True)
        uq = torch.einsum('qa,eac->eqc', self.N, u[self.mesh.velocity_cells])
        return self._scatter(torch.einsum('qa,eqc,q->eac', self.N, uq, self.weights))

    def velocity_stiffness(self, u):
        self._field(u, vector=True)
        du = torch.einsum('qaj,eac->eqcj', self.dN, u[self.mesh.velocity_cells])
        return self._scatter(torch.einsum('qaj,eqcj,q->eac', self.dN, du, self.weights))

    def pressure_mass(self, p):
        self._field(p, pressure=True)
        pq = torch.einsum('qa,ea->eq', self.Q, p[self.mesh.pressure_cells])
        return self._scatter(torch.einsum('qa,eq,q->ea', self.Q, pq, self.weights), True)

    def pressure_stiffness(self, p):
        self._field(p, pressure=True)
        dp = torch.einsum('qaj,ea->eqj', self.dQ, p[self.mesh.pressure_cells])
        return self._scatter(torch.einsum('qaj,eqj,q->ea', self.dQ, dp, self.weights), True)

    def divergence(self, u):
        """Q1 dual load D u = integral q div(u), not nodal divergence values."""
        self._field(u, vector=True)
        div = torch.einsum('qaj,eaj->eq', self.dN, u[self.mesh.velocity_cells])
        return self._scatter(torch.einsum('qa,eq,q->ea', self.Q, div, self.weights), True)

    def divergence_transpose(self, p):
        """D.T p = integral p div(v); sign and boundary differ from G p."""
        self._field(p, pressure=True)
        pq = torch.einsum('qa,ea->eq', self.Q, p[self.mesh.pressure_cells])
        return self._scatter(torch.einsum('qaj,eq,q->eaj', self.dN, pq, self.weights))

    def gradient(self, p):
        """G p = integral v dot grad(p), the Chorin velocity-correction load."""
        self._field(p, pressure=True)
        dp = torch.einsum('qaj,ea->eqj', self.dQ, p[self.mesh.pressure_cells])
        return self._scatter(torch.einsum('qa,eqj,q->eaj', self.N, dp, self.weights))

    def convection(self, u, advector=None):
        """C(a,u) = integral v dot ((a dot grad) u); advector defaults to u.

        Advective form, not skew-symmetric. Four Gauss points/axis integrate
        the Q2 triple products exactly on these affine cells.
        """
        advector = u if advector is None else advector
        self._field(u, vector=True)
        self._field(advector, vector=True)
        aq = torch.einsum('qa,eaj->eqj', self.N, advector[self.mesh.velocity_cells])
        du = torch.einsum('qaj,eac->eqcj', self.dN, u[self.mesh.velocity_cells])
        transport = torch.einsum('eqj,eqcj->eqc', aq, du)
        return self._scatter(torch.einsum('qa,eqc,q->eac', self.N, transport, self.weights))

    def density_load(self, density):
        """M_v f for a nodal Q2 force density. Never apply this to H.T g."""
        return self.velocity_mass(density)


def prepare_operators(mesh, *, quadrature_order=4):
    """Small reference tables; same cell sizes reused throughout a uniform box."""
    X = mesh.velocity_coordinates
    q, w = quadrature(quadrature_order, device=X.device, dtype=X.dtype)
    N, dN = basis(q, 2)
    Q, dQ = basis(q, 1)
    h = X.new_tensor(mesh.cell_sizes)
    return FluidOperators(mesh, w*prod(mesh.cell_sizes), N, dN/h, Q, dQ/h)
