"""Backward-Euler MAC/FE Newton solve with step-frozen adjoint IB operators.

Velocity is the reduced unknown: x=x_n+dt*J_n*u, pressure is eliminated
by the existing Poisson projection. Convection, viscosity and FE force
are evaluated at the new state. The exact reduced Newton action uses an
assembled CSR solid-force tangent; no differentiation through linear solves.
The IB stencil is frozen at x_n, a deliberate geometric semi-implicit choice.
"""
from dataclasses import dataclass, field, asdict, replace
from math import isfinite, sqrt
import torch
from .coupling import MACIBStepper, MACState
from .grid import zero_normal, convection, velocity_laplacian, divergence
from ..nonlinear import (NewtonOptions, GMRESOptions, NonlinearFailure,
                         newton, normalized_linear_action)
from .cnab import CNABOptions
from .midpoint_solver import AndersonOptions
from ..transport import transport_numbers, implicit_policy


@dataclass(frozen=True)
class MACCouplingOptions:
    scheme: str = 'explicit-lagged'
    newton: NewtonOptions = field(default_factory=lambda: NewtonOptions(
        rtol=1e-8, atol=1e-9, max_iterations=12, max_backtracks=16,
        linear_tolerance_fraction=.2,
        linear=GMRESOptions(rtol=1e-2, atol=1e-11, restart=12, max_iterations=120, check_every=3)))
    tangent_chunk_size: int = 2048
    cnab: CNABOptions = field(default_factory=CNABOptions)
    semiimplicit_solver: str = 'newton'
    anderson: AndersonOptions = field(default_factory=AndersonOptions)
    reuse_final_evaluation: bool = True
    stokes_warm_start: bool = False
    reuse_validation: bool = False
    adaptive_substeps: bool = False
    substep_courant_target: float = .20
    max_substep_levels: int = 4

    def __post_init__(self):
        if type(self.adaptive_substeps) is not bool:
            raise ValueError('adaptive_substeps must be a bool')
        if self.adaptive_substeps and self.scheme!='cnab-semiimplicit':
            raise ValueError('adaptive coupled substeps require cnab-semiimplicit')
        if not isfinite(self.substep_courant_target) or not 0 < self.substep_courant_target < .25:
            raise ValueError('substep Courant target must be in (0,0.25)')
        if type(self.max_substep_levels) is not int or not 1 <= self.max_substep_levels <= 4:
            raise ValueError('max_substep_levels must be in [1,4]')
        if type(self.reuse_validation) is not bool:
            raise ValueError('reuse_validation must be a bool')
        if type(self.reuse_final_evaluation) is not bool:
            raise ValueError('reuse_final_evaluation must be a bool')
        if type(self.stokes_warm_start) is not bool:
            raise ValueError('stokes_warm_start must be a bool')
        if self.scheme not in ('explicit-lagged', 'explicit-rk3', 'implicit-newton', 'cnab-midpoint', 'cnab-semiimplicit'):
            raise ValueError('unsupported MAC coupling scheme')
        if not isinstance(self.cnab,CNABOptions):
            raise ValueError('cnab must be CNABOptions')
        if self.semiimplicit_solver not in ('newton','anderson-newton'):
            raise ValueError('semiimplicit_solver must be newton or anderson-newton')
        if not isinstance(self.anderson,AndersonOptions):
            raise ValueError('anderson must be AndersonOptions')
        if not isinstance(self.newton, NewtonOptions):
            raise ValueError('coupling newton must be NewtonOptions')
        if type(self.tangent_chunk_size) is not int or self.tangent_chunk_size < 1:
            raise ValueError('positive tangent_chunk_size required')


class ImplicitMACIBStepper(MACIBStepper):
    def __init__(self, flow, transfer, model, options, solid_execution=None):
        super().__init__(flow, transfer, model.force, model.validate)
        from ..solids.contracts import make_tangent
        from .execution import tensor_kernel
        self.model, self.options, self.solid_execution = model, options, solid_execution
        self.tangent = make_tangent(model, options.tangent_chunk_size)
        self._sizes = tuple(u.numel() for u in flow.grid.zeros(device='meta'))
        self._shapes = tuple(flow.grid.face_shape(c) for c in range(3))
        self._right = tensor_kernel(self._right, model.mesh.X.device)
        self._linear_right = tensor_kernel(self._linear_right, model.mesh.X.device)

    def pack(self, velocity):
        return torch.cat([u.reshape(-1) for u in velocity])

    def unpack(self, value):
        return tuple(u.reshape(shape) for u, shape in zip(value.split(self._sizes), self._shapes))

    def _right(self, old, velocity, density):
        flow = self.flow
        adv = convection(velocity, flow.grid.spacing)
        return zero_normal(tuple(a+flow.dt*(-b+flow.mu/flow.rho*
            velocity_laplacian(u,c,flow.grid.spacing)+f/flow.rho)
            for c,(a,u,b,f) in enumerate(zip(old,velocity,adv,density))))

    def _linear_right(self, velocity, direction, density):
        flow = self.flow
        adv = torch.func.jvp(lambda v: convection(v, flow.grid.spacing), (velocity,), (direction,))[1]
        return zero_normal(tuple(flow.dt*(-a+flow.mu/flow.rho*
            velocity_laplacian(v,c,flow.grid.spacing)+f/flow.rho)
            for c,(v,a,f) in enumerate(zip(direction,adv,density))))

    @torch.no_grad()
    def step(self, state, *, diagnostics=True):
        dt, grid = self.flow.dt, self.flow.grid
        if type(state.step) is not int or state.step < 0 or not isfinite(state.time) or abs(state.time-state.step*dt) > 1e-12:
            raise ValueError('inconsistent MAC state clock')
        self.validate(state.x)
        grid.check_velocity(state.velocity)
        if any(not torch.isfinite(u).all() for u in state.velocity):
            raise ValueError('nonfinite initial MAC velocity')
        stencil = self.transfer.prepare(state.x)
        self.stencil_builds += 1
        self.point_evaluations += 1
        target_time = (state.step+1)*dt
        cache = {}
        def kinematics(value):
            if cache.get('value') is value and cache.get('version') == value._version:
                return cache['kinematics']
            u = self.unpack(value)
            U, info = self.transfer.interpolate(u, stencil)
            x = state.x+dt*U
            cache.update(value=value, version=value._version, kinematics=(u,x,info))
            return u,x,info
        def validate(value):
            _, x, _ = kinematics(value)
            self.validate(x)
            self.transfer.check_support(self.transfer.interaction_points(x))
            fraction = ((x-state.x).abs()/x.new_tensor(grid.spacing)).max().item()
            if fraction > self.max_displacement:
                raise ValueError('implicit MAC solid displacement exceeds grid-based limit; reduce dt')
        def evaluate(value):
            u,x,interpolation = kinematics(value)
            force = self.force(x,target_time) if self.solid_execution is None else self.solid_execution.force(x,target_time)
            density, spread = self.transfer.spread(force,stencil)
            result = self.flow.project(self._right(state.velocity,u,density))
            return value-self.pack(result.velocity), result, force, density, spread, interpolation
        def residual(value):
            return evaluate(value)[0]
        def linearization(value):
            velocity,x,_ = kinematics(value)
            K = self.tangent.assemble(x,target_time)
            def action(direction):
                v = self.unpack(direction)
                U,_ = self.transfer.interpolate(v,stencil)
                dforce = torch.sparse.mm(K,(dt*U).reshape(-1,1)).reshape_as(x)
                density,_ = self.transfer.spread(dforce,stencil)
                projected = self.flow.project(self._linear_right(velocity,v,density))
                return direction-self.pack(projected.velocity)
            # Mass and pressure solves use absolute as well as relative
            # tolerances. Evaluate every Krylov direction at the same norm;
            # especially important when Newton corrections approach zero.
            return normalized_linear_action(action)
        initial = self.pack(zero_normal(state.velocity))
        fixed = self.pack(tuple(torch.ones_like(u) for u in state.velocity))-self.pack(
            zero_normal(tuple(torch.ones_like(u) for u in state.velocity)))
        result = newton(residual,initial,validate=validate,fixed=fixed.bool(), values=torch.zeros_like(initial),
                        linearization_factory=linearization,options=self.options.newton)
        # Keep the Newton unknown. Replacing u by G(u)=P[right(u)] here is
        # an unguarded Picard iteration, not a projection of u. For a stiff
        # problem it can amplify an already acceptable residual arbitrarily.
        # Recover pressure/force at the SAME u and independently recheck it.
        value = result.x
        validate(value)
        r, flow, force, density, spread, interpolation = evaluate(value)
        true_norm = torch.linalg.vector_norm(r).item()
        velocity,x,_ = kinematics(value)
        div_norm = torch.linalg.vector_norm(divergence(velocity,grid.spacing)).item()
        projected_div_norm = torch.linalg.vector_norm(divergence(flow.velocity,grid.spacing)).item()
        # u=G(u)+R(u): ||D u|| <= ||D G(u)|| + ||D||*tol. The MAC
        # difference operator has ||D||_2 <= 2*sqrt(sum(h_c**-2)). Allow
        # floating-point evaluation error, while checking the actual D u.
        div_operator_bound = 2*sqrt(sum(h**-2 for h in grid.spacing))
        roundoff = 64*torch.finfo(value.dtype).eps*div_operator_bound*max(
            torch.linalg.vector_norm(value).item(),1.)
        div_tolerance = projected_div_norm+div_operator_bound*result.tolerance+roundoff
        acceptance = dict(stage='final-coupled-state',step=state.step+1,time_s=target_time,
            newton_residual_norm=result.residual_norm,residual_norm=true_norm,tolerance=result.tolerance,
            divergence_norm=div_norm,divergence_tolerance=div_tolerance,
            projected_divergence_norm=projected_div_norm,pressure=flow.diagnostics['pressure'],
            force_mass=asdict(spread),velocity_mass=asdict(interpolation))
        if not isfinite(true_norm) or true_norm > result.tolerance or not isfinite(div_norm) or div_norm > div_tolerance:
            failure = NonlinearFailure(
                f'implicit MAC final coupled-state check failed at step {state.step+1}: '
                f'residual={true_norm:.9g}, tolerance={result.tolerance:.9g}, '
                f'Newton residual={result.residual_norm:.9g}; '
                f'divergence={div_norm:.9g}, divergence tolerance={div_tolerance:.9g}',
                replace(result,converged=False,residual_norm=true_norm))
            failure.coupled_diagnostics = acceptance
            raise failure
        fraction = ((x-state.x).abs()/x.new_tensor(grid.spacing)).max().item()
        new = MACState(state.step+1,target_time,x,velocity,flow.pressure,force,target_time)
        numbers = transport_numbers([u.abs().max().item() for u in velocity], grid.spacing, dt, self.flow.mu/self.flow.rho)
        info = dict(flow=dict(pressure=flow.diagnostics['pressure'], **numbers),
                    force_mass=asdict(spread), velocity_mass=asdict(interpolation),
                    max_grid_displacement=fraction, used_force_time_s=target_time,next_force_time_s=target_time,
                    nonlinear=dict(iterations=result.iterations,residual_norm=true_norm,
                                   tolerance=result.tolerance,history=result.history,
                                   acceptance=acceptance,
                                   solid_tangent='assembled CSR',ib_geometry='frozen at preceding accepted position'))
        if diagnostics:
            U,_ = self.transfer.interpolate(velocity,stencil)
            solid_power = (U*force).sum()
            fluid_power = grid.volume*sum((u*f).sum() for u,f in zip(velocity,density))
            info.update(solid_power=solid_power.item(),fluid_power=fluid_power.item(),
                        power_error=abs((solid_power-fluid_power).item()),
                        divergence_l2=sqrt(grid.volume)*div_norm)
        return new,info
