"""Explicit-viscous centered MAC predictor and open-channel projection."""
from dataclasses import dataclass
from math import sin,pi,isfinite
import torch
from ..mac.execution import tensor_kernel
from .grid import boundary,convection,laplacian,divergence,gradient
from .multigrid import ChannelMultigrid


@dataclass(frozen=True)
class ChannelResult:
    velocity: tuple
    pressure: torch.Tensor
    diagnostics: dict


class ChannelFlow:
    def __init__(self,grid,*,dt=1/16000,rho=1.,mu=.1,device='cpu',fused=True,options=None):
        if any(not isfinite(v) or v<=0 for v in (dt,rho,mu)):
            raise ValueError('positive finite dt/rho/mu required')
        self.grid,self.dt,self.rho,self.mu=grid,dt,rho,mu
        self.viscous_number=dt*mu/rho*sum(1/h**2 for h in grid.spacing)
        if self.viscous_number>.25:
            raise ValueError('explicit viscous step too large')
        self.pressure_solver=ChannelMultigrid(grid,device=device,options=options,fused=fused)
        y=grid.coordinates(0,device=device)[0,:,1]
        self.profile=y*(grid.lengths[1]-y)
        self.scale=self.profile.new_empty(())
        self._predict=tensor_kernel(self._predict,device) if fused else self._predict
        self._correct=tensor_kernel(self._correct,device) if fused else self._correct

    def inlet(self,time):
        self.scale.fill_(5*(sin(2*pi*time)+1.1))
        return self.scale*self.profile

    def _predict(self,velocity,density,inlet):
        vel=boundary(velocity,inlet)
        adv=convection(vel,self.grid.spacing)
        diffusion=laplacian(vel,self.grid.spacing)
        return boundary(tuple(u+self.dt*(-a+self.mu/self.rho*l+f/self.rho)
                        for u,a,l,f in zip(vel,adv,diffusion,density)),inlet)

    def _correct(self,tentative,p):
        return tuple(u-self.dt/self.rho*g for u,g in zip(tentative,gradient(p,self.grid.spacing)))

    @torch.no_grad()
    def project(self,tentative,initial=None):
        self.grid.check_velocity(tentative)
        if any(not torch.isfinite(v).all() for v in tentative):
            raise ValueError('finite tentative velocity required')
        if tentative[1][:,0].count_nonzero() or tentative[1][:,-1].count_nonzero():
            raise ValueError('normal wall velocities must be zero')
        p,info=self.pressure_solver.solve(-self.rho/self.dt*divergence(tentative,self.grid.spacing),initial)
        return ChannelResult(self._correct(tentative,p),p,dict(pressure=info))

    @torch.no_grad()
    def step(self,velocity,density,*,time,pressure_initial=None):
        self.grid.check_velocity(velocity)
        self.grid.check_velocity(density)
        if not isfinite(time) or any(not torch.isfinite(v).all() for v in (*velocity,*density)):
            raise ValueError('finite time/velocity/force required')
        inlet=self.inlet(time)
        speeds=torch.stack((torch.maximum(velocity[0].abs().max(),inlet.abs().max()),velocity[1].abs().max()))
        cfl=(self.dt*speeds/speeds.new_tensor(self.grid.spacing)).sum().item()
        centered_number=(self.dt*speeds.square().sum()/(self.mu/self.rho)).item()
        if cfl>.25 or centered_number>1:
            raise ValueError('centered explicit advection stability guard exceeded; reduce dt')
        result=self.project(self._predict(velocity,density,inlet),pressure_initial)
        return ChannelResult(result.velocity,result.pressure,dict(result.diagnostics,
            courant=cfl,centered_number=centered_number,viscous_number=self.viscous_number,inlet_time_s=time))
