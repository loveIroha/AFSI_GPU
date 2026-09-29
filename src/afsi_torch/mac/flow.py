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
    def __init__(self, grid, *, dt, rho=1., mu=1., device='cpu', dtype=torch.float64,
                 options=None, pressure_backend='torch', execution_backend='torch'):
        if execution_backend not in ('torch','fused'):
            raise ValueError('execution backend must be torch or fused')
        if any(not isfinite(v) or v <= 0 for v in (dt, rho, mu)):
            raise ValueError('this explicit-viscous MAC scheme requires positive dt/rho/mu')
        self.grid, self.dt, self.rho, self.mu = grid, float(dt), float(rho), float(mu)
        self.viscous_number = dt*mu/rho*sum(1/h**2 for h in grid.spacing)
        # Boundary-adjacent tangential unknowns have a larger diagonal.
        if self.viscous_number > .25:
            raise ValueError('explicit viscous time step too large; reduce dt')
        self.pressure_solver = GeometricMultigrid(grid, device=device, dtype=dtype,
                                                  options=options, backend=pressure_backend)
        self.execution_backend=execution_backend
        if execution_backend=='fused':
            from .execution import tensor_kernel
            self.pressure_solver.enable_tensor_fusion()
            self._predict=tensor_kernel(self._predict,device)
            self._correct=tensor_kernel(self._correct,device)
            self._rhs=tensor_kernel(self._rhs,device)
            self._step_checks=tensor_kernel(self._step_checks,device)
            self._project_checks=tensor_kernel(self._project_checks,device)

    def _predict(self,velocity,density):
        vel=zero_normal(velocity)
        adv=convection(vel,self.grid.spacing)
        return zero_normal(tuple(u+self.dt*(-a+self.mu/self.rho*
            velocity_laplacian(u,c,self.grid.spacing)+f/self.rho)
            for c,(u,a,f) in enumerate(zip(vel,adv,density))))

    def _correct(self,tentative,p):
        grad=gradient(p,self.grid.spacing)
        return tuple(u-self.dt/self.rho*g for u,g in zip(tentative,grad))

    def _rhs(self,tentative):
        return -self.rho/self.dt*divergence(tentative,self.grid.spacing)

    def _step_checks(self,velocity,density):
        finite=torch.stack([torch.isfinite(u).all() for u in (*velocity,*density)]).all()
        speeds=torch.stack([u.abs().max() for u in velocity])
        h=speeds.new_tensor(self.grid.spacing)
        return torch.stack((finite.to(speeds.dtype),(self.dt*speeds/h).sum(),
                            (speeds*h/(self.mu/self.rho)).max()))

    def _project_checks(self,tentative):
        finite=torch.stack([torch.isfinite(u).all() for u in tentative]).all()
        normal=torch.stack([(u[slab(c,0,1)]==0).all() & (u[slab(c,-1,None)]==0).all()
                            for c,u in enumerate(tentative)]).all()
        return torch.stack((finite,normal))

    @torch.no_grad()
    def project(self, tentative, initial=None):
        self.grid.check_velocity(tentative)
        if self.execution_backend=='fused':
            finite,normal=self._project_checks(tentative).tolist()
            if not finite:
                raise ValueError('nonfinite tentative velocity')
            if not normal:
                raise ValueError('closed-box normal velocity must be zero before projection')
            p,info=self.pressure_solver.solve(self._rhs(tentative),initial)
            return MACFlowResult(self._correct(tentative,p),p,dict(pressure=info))
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
        if self.execution_backend=='fused':
            finite,courant,reynolds=self._step_checks(velocity,density).tolist()
            if not finite:
                raise ValueError('finite MAC velocity and force density required')
            if courant>.25 or reynolds>1.:
                raise ValueError('MAC centered-advection stability guard exceeded; refine grid or change transport scheme')
            result=self.project(self._predict(velocity,density),pressure_initial)
            return MACFlowResult(result.velocity,result.pressure,
                                 dict(result.diagnostics,courant=courant,viscous_number=self.viscous_number))
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
