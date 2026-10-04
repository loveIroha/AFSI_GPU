"""Explicit IB/FE-MAC stepping with AFSI's lagged force/load sampling order."""
from dataclasses import dataclass, asdict
from math import isfinite
import torch
from .grid import divergence


@dataclass(frozen=True)
class MACState:
    step: int
    time: float
    x: torch.Tensor
    velocity: tuple
    pressure: torch.Tensor
    force: torch.Tensor
    force_time: float | None
    previous_advection: tuple | None = None


class MACIBStepper:
    def __init__(self, flow, transfer, force, validate, *, max_displacement=.25,
                 optimized=False,solid_execution=None):
        if flow.grid != transfer.grid:
            raise ValueError('flow/transfer grids differ')
        if not isfinite(max_displacement) or max_displacement <= 0:
            raise ValueError('positive displacement bound required')
        self.flow, self.transfer, self.force, self.validate = flow, transfer, force, validate
        self.max_displacement = max_displacement
        self.optimized,self.solid_execution=optimized,solid_execution
        if optimized and (solid_execution is None or not hasattr(transfer,'from_points')):
            raise ValueError('optimized coupling requires compact transfer and solid execution')
        self._cached_state,self._cached_version,self._stencil=None,None,None
        self.stencil_builds,self.point_evaluations=0,0
        if optimized:
            from .execution import tensor_kernel
            self._accept_metrics=tensor_kernel(self._accept_metrics,solid_execution.model.mesh.X.device)

    @staticmethod
    def _version(x):
        return None if torch.is_inference(x) else x._version

    def _remember(self,state,stencil):
        self._cached_state,self._cached_version,self._stencil=state,self._version(state.x),stencil

    def _prepare(self,state):
        if (self.optimized and state is self._cached_state and self._cached_version is not None and
                self._version(state.x)==self._cached_version):
            return self._stencil
        self.stencil_builds+=1; self.point_evaluations+=1
        return self.transfer.prepare(state.x)

    def _accept_metrics(self,increment,flags,force,points):
        fraction=(increment.abs()/increment.new_tensor(self.flow.grid.spacing)).max()
        return torch.stack((fraction,flags.all().to(force.dtype),
            self.transfer._support_flags(points).to(force.dtype),torch.isfinite(force).all().to(force.dtype)))

    @torch.no_grad()
    def initialize(self, x):
        self.validate(x)
        if self.optimized:
            stencil=self.transfer.prepare(x)
            self.stencil_builds+=1; self.point_evaluations+=1
        else:
            self.transfer.check_support(self.transfer.interaction_points(x))
            self.point_evaluations+=1
        velocity = self.flow.grid.zeros(device=x.device,dtype=x.dtype)
        state=MACState(0,0.,x.clone(),velocity,x.new_zeros(self.flow.grid.shape),torch.zeros_like(x),None)
        if self.optimized:
            self._remember(state,stencil)
        return state

    @torch.no_grad()
    def step(self, state, *, diagnostics=True):
        dt = self.flow.dt
        if type(state.step) is not int or state.step < 0 or not isfinite(state.time) or abs(state.time-state.step*dt)>1e-12:
            raise ValueError('inconsistent MAC state clock')
        self.validate(state.x)
        stencil = self._prepare(state)
        density, spread_info = self.transfer.spread(state.force,stencil)
        flow = self.flow.step(state.velocity,density,pressure_initial=state.pressure)
        velocity, interpolation_info = self.transfer.interpolate(flow.velocity,stencil)
        increment = dt*velocity
        x_new = state.x+increment
        if self.optimized:
            force,geometry=self.solid_execution.force_with_geometry(x_new,state.time)
            points=self.transfer.evaluate(x_new).reshape(-1,3)
            self.point_evaluations+=1
            fraction,valid,support,finite_force=self._accept_metrics(increment,geometry[-1],force,points).tolist()
            if not isfinite(fraction) or fraction>self.max_displacement:
                raise ValueError('MAC solid displacement exceeds the grid-based limit; reduce dt')
            if not valid:
                raise ValueError('invalid LV deformation: det(F), surface or cavity volume')
            if not support:
                raise ValueError('MAC IB support reaches a wall or is nonfinite; enlarge/refine the fluid box')
            if not finite_force:
                raise ValueError('invalid FE nodal field')
            next_stencil=self.transfer.from_points(points)
            self.stencil_builds+=1
            self.solid_execution.remember_geometry(x_new,geometry)
        else:
            fraction = (increment.abs()/increment.new_tensor(self.flow.grid.spacing)).max().item()
            if not isfinite(fraction) or fraction > self.max_displacement:
                raise ValueError('MAC solid displacement exceeds the grid-based limit; reduce dt')
            self.validate(x_new)
            self.transfer.check_support(self.transfer.interaction_points(x_new))
            self.point_evaluations+=1
            force = self.force(x_new,state.time)
            self.transfer._nodal(force)
        new = MACState(state.step+1,(state.step+1)*dt,x_new,flow.velocity,flow.pressure,force,state.time)
        if self.optimized:
            self._remember(new,next_stencil)
        info = dict(flow=flow.diagnostics,force_mass=asdict(spread_info),
                    velocity_mass=asdict(interpolation_info),max_grid_displacement=fraction,
                    used_force_time_s=state.force_time,next_force_time_s=state.time)
        if diagnostics:
            solid_power = (velocity*state.force).sum()
            fluid_power = self.flow.grid.volume*sum((u*f).sum() for u,f in zip(flow.velocity,density))
            info.update(solid_power=solid_power.item(),fluid_power=fluid_power.item(),
                        power_error=abs((solid_power-fluid_power).item()),
                        divergence_l2=(self.flow.grid.volume*divergence(flow.velocity,self.flow.grid.spacing).square().sum()).sqrt().item())
        return new,info
