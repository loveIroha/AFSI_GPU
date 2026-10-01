"""Projected SSPRK(3,3) fluid stepping with one frozen IB density per step.

Only the autonomous fluid subproblem is RK3. The surrounding FE/IB update
retains its lagged force and first-order kinematics. Centered space stencils,
physical viscosity, consistent FE mass and adjoint transfer are unchanged.
"""
import torch
from .flow import MACFlow, MACFlowResult, MACTransportGuardError
from .grid import zero_normal
from ..transport import rk3_policy, rk3_violations, COURANT_LIMIT, VISCOUS_LIMIT


class MACRK3TransportGuardError(MACTransportGuardError):
    def __init__(self,diagnostics):
        self.diagnostics = diagnostics
        ValueError.__init__(self,
            f'MAC SSPRK3 stage {diagnostics["stage"]} transport guard exceeded: '
            f'CFL={diagnostics["courant"]:.9g} (limit {COURANT_LIMIT}), '
            f'D={diagnostics["viscous_number"]:.9g} (limit {VISCOUS_LIMIT}); '
            f'triggered={",".join(diagnostics["triggered"])}. Reduce dt. '
            'This screen does not establish explicit elastic/IB stability.')


def _blend(a,b,weight_a,weight_b):
    return tuple(weight_a*x+weight_b*y for x,y in zip(a,b))


def _pressure_average(a,b,c):
    return a/6+b/6+(2/3)*c


class MACRK3Flow(MACFlow):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.implicit_transport:
            raise ValueError('SSPRK3 requires explicit transport')
        from .execution import tensor_kernel
        self._blend = tensor_kernel(_blend,self.pressure_solver.diagonals[0].device) if self.execution_backend=='fused' else _blend
        self._pressure_average = tensor_kernel(_pressure_average,self.pressure_solver.diagonals[0].device) if self.execution_backend=='fused' else _pressure_average

    def _stage(self,velocity,density,pressure,stage):
        finite,courant,reynolds,A = self._step_checks(velocity,density).tolist()
        if not finite:
            raise ValueError('finite MAC velocity and force density required')
        numbers = dict(courant=courant,cell_reynolds=reynolds,
                       advection_diffusion_number=A,viscous_number=self.viscous_number)
        triggered = rk3_violations(courant,self.viscous_number)
        if triggered:
            raise MACRK3TransportGuardError(dict(numbers,**rk3_policy(),triggered=triggered,stage=stage,
                component_max_abs_velocity=[u.abs().max().item() for u in velocity],
                spacing=list(self.grid.spacing),dt=self.dt,rho=self.rho,mu=self.mu))
        result = self.project(self._predict(velocity,density),initial=pressure)
        return result,numbers

    @torch.no_grad()
    def step(self,velocity,density,*,pressure_initial=None):
        self.grid.check_velocity(velocity)
        self.grid.check_velocity(density)
        velocity = zero_normal(velocity)
        first,n1 = self._stage(velocity,density,pressure_initial,1)
        second,n2 = self._stage(first.velocity,density,first.pressure,2)
        middle = self._blend(velocity,second.velocity,.75,.25)
        third,n3 = self._stage(middle,density,second.pressure,3)
        result = self._blend(velocity,third.velocity,1/3,2/3)
        # Butcher weights: the returned pressure is the step-average
        # multiplier, not a separately solved endpoint pressure.
        pressure = self._pressure_average(first.pressure,second.pressure,third.pressure)
        stages = [s.diagnostics['pressure'] for s in (first,second,third)]
        weights = (1/6,1/6,2/3)
        info = dict(cycles=sum(s['cycles'] for s in stages),backend=self.pressure_solver.backend,
                    stages=stages,meaning='RK-weighted step-average pressure; each stage true residual checked',
                    residual_norm_bound=sum(w*s['residual_norm'] for w,s in zip(weights,stages)),
                    tolerance_bound=sum(w*s['tolerance'] for w,s in zip(weights,stages)))
        numbers = {key:max(n[key] for n in (n1,n2,n3)) for key in n1}
        return MACFlowResult(result,pressure,dict(pressure=info,**numbers,
            time_integrator='SSPRK(3,3)',stage_transport=[n1,n2,n3]))
