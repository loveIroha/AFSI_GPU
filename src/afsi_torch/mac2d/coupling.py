"""Explicit fluid -> solid -> updated force order of AFSI demo_340."""
from dataclasses import dataclass,asdict
from math import isfinite
import torch
from .grid import divergence
from ..mac.execution import tensor_kernel


@dataclass(frozen=True)
class ValveState:
    step: int
    time: float
    x: torch.Tensor
    velocity: tuple
    pressure: torch.Tensor
    force: torch.Tensor


class ValveStepper:
    def __init__(self,flow,transfer,solid,*,optimized=False,fused=True):
        self.flow,self.transfer,self.solid=flow,transfer,solid
        self.optimized=optimized
        self._cached_state,self._cached_version,self._stencil=None,None,None
        self.stencil_builds=0
        self._accept_metrics=tensor_kernel(self._accept_metrics,solid.mesh.X.device) if optimized and fused else self._accept_metrics

    @staticmethod
    def _version(x):
        try:
            return x._version
        except RuntimeError:
            return None  # Inference-mode tensors have no counter; do not cache them.

    def _remember(self,state,stencil):
        self._cached_state,self._cached_version,self._stencil=state,self._version(state.x),stencil

    def _prepare(self,state):
        if (self.optimized and state is self._cached_state and self._cached_version is not None and
            self._version(state.x)==self._cached_version):
            return self._stencil
        self.stencil_builds+=1
        return self.transfer.prepare(state.x)

    def _accept_metrics(self,increment,J,force,points):
        fraction=(increment.abs()/increment.new_tensor(self.flow.grid.spacing)).max()
        jacobian=torch.isfinite(J).all() & (J>0).all()
        # Inline the support test so all acceptance flags share one host read.
        hx,hy=self.flow.grid.spacing
        support=(torch.isfinite(points).all() & (points[:,0]>=2*hx).all() &
            (points[:,0]<=self.flow.grid.lengths[0]-2*hx).all() &
            (points[:,1]>=-hy).all() & (points[:,1]<=self.flow.grid.lengths[1]+hy).all())
        return torch.stack((fraction,jacobian.to(force.dtype),support.to(force.dtype),torch.isfinite(force).all().to(force.dtype)))

    def initialize(self):
        X=self.solid.mesh.X
        self.solid.validate(X)
        stencil=self.transfer.prepare(X)
        self.stencil_builds+=1
        state=ValveState(0,0.,X.clone(),self.flow.grid.zeros(device=X.device,dtype=X.dtype),
                          X.new_zeros(self.flow.grid.shape),torch.zeros_like(X))
        if self.optimized:
            self._remember(state,stencil)
        return state

    @torch.no_grad()
    def step(self,state,*,diagnostics=False):
        if type(state.step) is not int or state.step<0 or not isfinite(state.time) or abs(state.time-state.step*self.flow.dt)>1e-12:
            raise ValueError('inconsistent valve state clock')
        stencil=self._prepare(state)
        density,fi=self.transfer.spread(state.force,stencil)
        fluid=self.flow.step(state.velocity,density,time=state.time,pressure_initial=state.pressure)
        velocity,vi=self.transfer.interpolate(fluid.velocity,stencil)
        increment=self.flow.dt*velocity
        x=state.x+increment
        if self.optimized:
            force,J=self.solid.force_and_det(x)
            points=self.transfer.evaluate(x).reshape(-1,2)
            fraction,jacobian,support,finite_force=self._accept_metrics(increment,J,force,points).tolist()
            if not isfinite(fraction) or fraction>.25:
                raise ValueError('valve motion exceeds the grid displacement limit; reduce dt')
            if not jacobian:
                raise ValueError('nonpositive or nonfinite valve det(F)')
            if not support:
                raise ValueError('valve IB points leave the channel support region')
            if not finite_force:
                raise ValueError('nonfinite valve force')
            next_stencil=self.transfer.from_points(points)
            self.stencil_builds+=1
        else:
            fraction=(increment.abs()/increment.new_tensor(self.flow.grid.spacing)).max().item()
            if not isfinite(fraction) or fraction>.25:
                raise ValueError('valve motion exceeds the grid displacement limit; reduce dt')
            self.solid.validate(x)
            self.transfer.prepare(x)  # support acceptance before publishing a state
            self.stencil_builds+=1
            force=self.solid.force(x)
            if not torch.isfinite(force).all():
                raise ValueError('nonfinite valve force')
        new=ValveState(state.step+1,(state.step+1)*self.flow.dt,x,fluid.velocity,fluid.pressure,force)
        if self.optimized:
            self._remember(new,next_stencil)
        info=dict(flow=fluid.diagnostics,force_mass=asdict(fi),velocity_mass=asdict(vi),max_grid_displacement=fraction)
        if diagnostics:
            ps=(velocity*state.force).sum().item()
            pf=(self.flow.grid.volume*sum((u*f).sum() for u,f in zip(fluid.velocity,density))).item()
            info.update(solid_power=ps,fluid_power=pf,power_abs_error=abs(ps-pf),
                power_relative_error=abs(ps-pf)/max(abs(ps),abs(pf),1e-30),
                divergence_l2=(self.flow.grid.volume*divergence(fluid.velocity,self.flow.grid.spacing).square().sum()).sqrt().item())
        return new,info
