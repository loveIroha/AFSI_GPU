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


class MACIBStepper:
    def __init__(self, flow, transfer, force, validate, *, max_displacement=.25):
        if flow.grid != transfer.grid:
            raise ValueError('flow/transfer grids differ')
        if not isfinite(max_displacement) or max_displacement <= 0:
            raise ValueError('positive displacement bound required')
        self.flow, self.transfer, self.force, self.validate = flow, transfer, force, validate
        self.max_displacement = max_displacement

    @torch.no_grad()
    def initialize(self, x):
        self.validate(x)
        self.transfer.check_support(self.transfer.interaction_points(x))
        velocity = self.flow.grid.zeros(device=x.device,dtype=x.dtype)
        return MACState(0,0.,x.clone(),velocity,x.new_zeros(self.flow.grid.shape),torch.zeros_like(x),None)

    @torch.no_grad()
    def step(self, state, *, diagnostics=True):
        dt = self.flow.dt
        if type(state.step) is not int or state.step < 0 or not isfinite(state.time) or abs(state.time-state.step*dt)>1e-12:
            raise ValueError('inconsistent MAC state clock')
        self.validate(state.x)
        stencil = self.transfer.prepare(state.x)
        density, spread_info = self.transfer.spread(state.force,stencil)
        flow = self.flow.step(state.velocity,density,pressure_initial=state.pressure)
        velocity, interpolation_info = self.transfer.interpolate(flow.velocity,stencil)
        increment = dt*velocity
        fraction = (increment.abs()/increment.new_tensor(self.flow.grid.spacing)).max().item()
        if not isfinite(fraction) or fraction > self.max_displacement:
            raise ValueError('MAC solid displacement exceeds the grid-based limit; reduce dt')
        x_new = state.x+increment
        self.validate(x_new)
        self.transfer.check_support(self.transfer.interaction_points(x_new))
        force = self.force(x_new,state.time)
        self.transfer._nodal(force)
        new = MACState(state.step+1,(state.step+1)*dt,x_new,flow.velocity,flow.pressure,force,state.time)
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
