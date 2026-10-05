"""BE viscous solve + Dirichlet-pressure Chorin projection (paper eq. 43).

With convection enabled, the old velocity is traced once per coupled step.
The resulting departure field is fixed during the nonlinear solid solve.
Pressure and Helmholtz true residuals are checked; no centred-Euler A screen
is used for this different transport discretization.
"""
from dataclasses import dataclass
from math import isfinite
import torch
from .grid import divergence
from .open_boundary import gradient, velocity_laplacian, velocity_diagonal, SemiLagrangian
from .dirichlet_mg import DirichletMultigrid
from .execution import tensor_kernel


@dataclass(frozen=True)
class BEFlowOptions:
    convection: bool = True
    helmholtz_rtol: float = 1e-12
    helmholtz_atol: float = 1e-13
    max_sweeps: int = 500
    check_every: int = 4

    def __post_init__(self):
        if type(self.convection) is not bool or any(not isfinite(t) or t < 0 for t in
                (self.helmholtz_rtol, self.helmholtz_atol)) or self.helmholtz_rtol+self.helmholtz_atol == 0:
            raise ValueError('invalid BE flow options')
        if any(type(n) is not int or n < 1 for n in (self.max_sweeps, self.check_every)):
            raise ValueError('positive Helmholtz iteration controls required')


class BackwardEulerFlow:
    def __init__(self, grid, *, dt, rho=1., mu=.04, device='cpu', dtype=torch.float64,
                 pressure_options=None, options=None, backend='torch', pressure_backend='torch'):
        if any(not isfinite(v) or v <= 0 for v in (dt, rho, mu)):
            raise ValueError('positive finite dt, density and viscosity required')
        self.grid, self.dt, self.rho, self.mu = grid, dt, rho, mu
        self.options = options or BEFlowOptions()
        self.pressure_solver = DirichletMultigrid(grid, device=device, dtype=dtype,
            options=pressure_options, backend=pressure_backend)
        like = torch.empty((), device=device, dtype=dtype)
        self.transport = SemiLagrangian(grid, like, fused=backend == 'fused') if self.options.convection else None
        self.diagonals = tuple(1+dt*mu/rho*velocity_diagonal(grid.face_shape(c), c, grid.spacing, like)
                               for c in range(3))
        self._smooth = self._smooth_block
        self._metrics = self._measure
        self._project = self._projection
        if backend == 'fused':
            self._smooth = tensor_kernel(self._smooth, device)
            self._metrics = tensor_kernel(self._metrics, device)
            self._project = tensor_kernel(self._project, device)
        self.calls = 0

    def advect(self, velocity):
        self.grid.check_velocity(velocity)
        return tuple(v.clone() for v in velocity) if self.transport is None else self.transport(velocity, self.dt)

    def _smooth_block(self, velocity, rhs):
        for _ in range(self.options.check_every):
            velocity = tuple(u+(b-u+self.dt*self.mu/self.rho*velocity_laplacian(u, c, self.grid.spacing))/d
                             for c, (u, b, d) in enumerate(zip(velocity, rhs, self.diagonals)))
        return velocity

    def _measure(self, velocity, rhs):
        norms = []
        for c, (u, b) in enumerate(zip(velocity, rhs)):
            r = b-u+self.dt*self.mu/self.rho*velocity_laplacian(u, c, self.grid.spacing)
            norms += [torch.linalg.vector_norm(r), torch.linalg.vector_norm(b)]
        return torch.stack(norms)

    def _projection(self, velocity, pressure):
        return tuple(u-self.dt/self.rho*g for u, g in zip(velocity, gradient(pressure, self.grid.spacing)))

    @torch.no_grad()
    def solve_rhs(self, rhs, pressure_initial=None):
        self.grid.check_velocity(rhs)
        u = tuple(b.clone() for b in rhs)
        count = 0
        while True:
            numbers = self._metrics(u, rhs).tolist()
            if not all(isfinite(v) for v in numbers):
                raise FloatingPointError('nonfinite BE Helmholtz solve')
            targets = [max(self.options.helmholtz_atol, self.options.helmholtz_rtol*n) for n in numbers[1::2]]
            if all(r <= t for r, t in zip(numbers[::2], targets)):
                break
            if count >= self.options.max_sweeps:
                raise RuntimeError('BE Neumann velocity Helmholtz failed to converge')
            u = self._smooth(u, rhs)
            count += self.options.check_every
        # A=-DG; physical pressure, not the dt-scaled projection potential.
        pressure, info = self.pressure_solver.solve(-self.rho/self.dt*divergence(u, self.grid.spacing), pressure_initial)
        velocity = self._project(u, pressure)
        self.calls += 1
        return velocity, pressure, dict(pressure=info, helmholtz_sweeps=count,
            helmholtz_residuals=numbers[::2], helmholtz_tolerances=targets)

    def response(self, density):
        # Zero advection/time lift, used only in an assembled Jacobian action.
        return self.solve_rhs(tuple(self.dt/self.rho*f for f in density))[0]

    def advance(self, departure_velocity, density, pressure_initial=None):
        return self.solve_rhs(tuple(a+self.dt/self.rho*f for a, f in zip(departure_velocity, density)), pressure_initial)
