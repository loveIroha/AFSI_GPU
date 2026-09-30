"""Explicit fluid -> solid -> updated force order of AFSI demo_340."""
from dataclasses import dataclass,asdict
from math import isfinite
import torch
from .grid import divergence


@dataclass(frozen=True)
class ValveState:
    step: int
    time: float
    x: torch.Tensor
    velocity: tuple
    pressure: torch.Tensor
    force: torch.Tensor


class ValveStepper:
    def __init__(self,flow,transfer,solid):
        self.flow,self.transfer,self.solid=flow,transfer,solid

    def initialize(self):
        X=self.solid.mesh.X
        self.solid.validate(X)
        self.transfer.prepare(X)
        return ValveState(0,0.,X.clone(),self.flow.grid.zeros(device=X.device,dtype=X.dtype),
                          X.new_zeros(self.flow.grid.shape),torch.zeros_like(X))

    @torch.no_grad()
    def step(self,state,*,diagnostics=False):
        if type(state.step) is not int or state.step<0 or not isfinite(state.time) or abs(state.time-state.step*self.flow.dt)>1e-12:
            raise ValueError('inconsistent valve state clock')
        stencil=self.transfer.prepare(state.x)
        density,fi=self.transfer.spread(state.force,stencil)
        fluid=self.flow.step(state.velocity,density,time=state.time,pressure_initial=state.pressure)
        velocity,vi=self.transfer.interpolate(fluid.velocity,stencil)
        increment=self.flow.dt*velocity
        fraction=(increment.abs()/increment.new_tensor(self.flow.grid.spacing)).max().item()
        if not isfinite(fraction) or fraction>.25:
            raise ValueError('valve motion exceeds the grid displacement limit; reduce dt')
        x=state.x+increment
        self.solid.validate(x)
        self.transfer.prepare(x)  # support acceptance before publishing a state
        force=self.solid.force(x)
        if not torch.isfinite(force).all():
            raise ValueError('nonfinite valve force')
        new=ValveState(state.step+1,(state.step+1)*self.flow.dt,x,fluid.velocity,fluid.pressure,force)
        info=dict(flow=fluid.diagnostics,force_mass=asdict(fi),velocity_mass=asdict(vi),max_grid_displacement=fraction)
        if diagnostics:
            ps=(velocity*state.force).sum().item()
            pf=(self.flow.grid.volume*sum((u*f).sum() for u,f in zip(fluid.velocity,density))).item()
            info.update(solid_power=ps,fluid_power=pf,power_abs_error=abs(ps-pf),
                power_relative_error=abs(ps-pf)/max(abs(ps),abs(pf),1e-30),
                divergence_l2=(self.flow.grid.volume*divergence(fluid.velocity,self.flow.grid.spacing).square().sum()).sqrt().item())
        return new,info
