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
    def __init__(self,grid,*,dt=1/16000,rho=1.,mu=.1,device='cpu',fused=True,options=None,
                 optimized=False,pressure_backend=None,inlet_config=None):
        if any(not isfinite(v) or v<=0 for v in (dt,rho,mu)):
            raise ValueError('positive finite dt/rho/mu required')
        self.grid,self.dt,self.rho,self.mu=grid,dt,rho,mu
        from ..config import InletConfig
        self.inlet_config=InletConfig(**(inlet_config or {}))
        self.optimized=optimized
        self.viscous_number=dt*mu/rho*sum(1/h**2 for h in grid.spacing)
        if self.viscous_number>.25:
            raise ValueError('explicit viscous step too large')
        if pressure_backend is None:
            pressure_backend=('graph' if torch.device(device).type=='cuda' else 'workspace') if optimized else 'reference'
        self.pressure_solver=ChannelMultigrid(grid,device=device,options=options,fused=fused,backend=pressure_backend)
        y=grid.coordinates(0,device=device)[0,:,1]
        self.profile=y*(grid.lengths[1]-y)
        self.scale=self.profile.new_empty(())
        self._predict=tensor_kernel(self._predict,device) if fused else self._predict
        self._correct=tensor_kernel(self._correct,device) if fused else self._correct
        if optimized and fused:
            self._input_metrics=tensor_kernel(self._input_metrics,device)
            self._project_metrics=tensor_kernel(self._project_metrics,device)

    def _input_metrics(self,velocity,density,inlet):
        valid=torch.ones((),device=inlet.device,dtype=torch.bool)
        for v in (*velocity,*density):
            valid=valid & torch.isfinite(v).all()
        speeds=torch.stack((torch.maximum(velocity[0].abs().max(),inlet.abs().max()),velocity[1].abs().max()))
        cfl=(self.dt*speeds/speeds.new_tensor(self.grid.spacing)).sum()
        centered=self.dt*speeds.square().sum()/(self.mu/self.rho)
        return torch.stack((valid.to(inlet.dtype),cfl,centered))

    def _project_metrics(self,tentative):
        u,v=tentative
        finite=torch.isfinite(u).all() & torch.isfinite(v).all()
        walls=(v[:,0]==0).all() & (v[:,-1]==0).all()
        return torch.stack((finite.to(u.dtype),walls.to(u.dtype)))

    def inlet(self,time):
        config=self.inlet_config
        self.scale.fill_(config.amplitude*(sin(2*pi*time/config.period)+config.offset))
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
        if self.optimized:
            finite,walls=self._project_metrics(tentative).tolist()
            if not finite:
                raise ValueError('finite tentative velocity required')
            if not walls:
                raise ValueError('normal wall velocities must be zero')
        else:
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
        if not isfinite(time):
            raise ValueError('finite time/velocity/force required')
        inlet=self.inlet(time)
        if self.optimized:
            valid,cfl,centered_number=self._input_metrics(velocity,density,inlet).tolist()
            if not valid:
                raise ValueError('finite time/velocity/force required')
        else:
            if any(not torch.isfinite(v).all() for v in (*velocity,*density)):
                raise ValueError('finite time/velocity/force required')
            speeds=torch.stack((torch.maximum(velocity[0].abs().max(),inlet.abs().max()),velocity[1].abs().max()))
            cfl=(self.dt*speeds/speeds.new_tensor(self.grid.spacing)).sum().item()
            centered_number=(self.dt*speeds.square().sum()/(self.mu/self.rho)).item()
        if cfl>.25 or centered_number>1:
            raise ValueError('centered explicit advection stability guard exceeded; reduce dt')
        result=self.project(self._predict(velocity,density,inlet),pressure_initial)
        return ChannelResult(result.velocity,result.pressure,dict(result.diagnostics,
            courant=cfl,centered_number=centered_number,viscous_number=self.viscous_number,inlet_time_s=time))
