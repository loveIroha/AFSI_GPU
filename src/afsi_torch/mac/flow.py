"""Experimental first-order explicit-viscous MAC projection on a closed box."""
from dataclasses import dataclass
from math import isfinite
import torch
from .grid import divergence, gradient, zero_normal, velocity_laplacian, convection, slab
from .multigrid import GeometricMultigrid


@dataclass(frozen=True)
class MACFlowResult:
    velocity: tuple
    pressure: torch.Tensor
    diagnostics: dict


class MACFlow:
    def __init__(self, grid, *, dt, rho=1., mu=1., device='cpu', dtype=torch.float64, options=None):
        if any(not isfinite(v) or v <= 0 for v in (dt, rho, mu)):
            raise ValueError('this explicit-viscous MAC scheme requires positive dt/rho/mu')
        self.grid, self.dt, self.rho, self.mu = grid, float(dt), float(rho), float(mu)
        self.viscous_number = dt*mu/rho*sum(1/h**2 for h in grid.spacing)
        # Boundary-adjacent tangential unknowns have a larger diagonal.
        if self.viscous_number > .25:
            raise ValueError('explicit viscous time step too large; reduce dt')
        self.pressure_solver = GeometricMultigrid(grid, device=device, dtype=dtype, options=options)

    @torch.no_grad()
    def project(self, tentative, initial=None):
        self.grid.check_velocity(tentative)
        for c, u in enumerate(tentative):
            if not torch.isfinite(u).all():
                raise ValueError('nonfinite tentative velocity')
            if u[slab(c,0,1)].count_nonzero() or u[slab(c,-1,None)].count_nonzero():
                raise ValueError('closed-box normal velocity must be zero before projection')
        p, info = self.pressure_solver.solve(-self.rho/self.dt*divergence(tentative, self.grid.spacing), initial)
        grad = gradient(p, self.grid.spacing)
        velocity = tuple(u-self.dt/self.rho*g for u, g in zip(tentative, grad))
        return MACFlowResult(velocity, p, dict(pressure=info))

    @torch.no_grad()
    def step(self, velocity, density, *, pressure_initial=None):
        self.grid.check_velocity(velocity)
        self.grid.check_velocity(density)
        if any(not torch.isfinite(u).all() for u in (*velocity, *density)):
            raise ValueError('finite MAC velocity and force density required')
        # Conservative guard for this initial centered-advection implementation.
        # High cell Reynolds numbers need a different transport scheme, not an
        # unchecked continuation with a small pressure residual.
        speeds = torch.stack([u.abs().max() for u in velocity])
        h = speeds.new_tensor(self.grid.spacing)
        courant = (self.dt*speeds/h).sum().item()
        if courant > .25 or (speeds*h/(self.mu/self.rho)).max().item() > 1.:
            raise ValueError('MAC centered-advection stability guard exceeded; refine grid or change transport scheme')
        vel = zero_normal(velocity)
        adv = convection(vel, self.grid.spacing)
        star = zero_normal(tuple(u+self.dt*(-a+self.mu/self.rho*
            velocity_laplacian(u,c,self.grid.spacing)+f/self.rho)
            for c,(u,a,f) in enumerate(zip(vel,adv,density))))
        result = self.project(star, pressure_initial)
        return MACFlowResult(result.velocity, result.pressure,
                             dict(result.diagnostics, courant=courant, viscous_number=self.viscous_number))
