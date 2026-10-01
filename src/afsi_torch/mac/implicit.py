"""Backward-Euler MAC/FE Newton solve with step-frozen adjoint IB operators.

Velocity is the reduced unknown: x=x_n+dt*J_n*u, pressure is eliminated
by the existing Poisson projection. Convection, viscosity and FE force
are evaluated at the new state. The exact reduced Newton action uses an
assembled CSR solid-force tangent; no differentiation through linear solves.
The IB stencil is frozen at x_n, a deliberate geometric semi-implicit choice.
"""
from dataclasses import dataclass, field, asdict
from math import isfinite
import torch
from .coupling import MACIBStepper, MACState
from .grid import zero_normal, convection, velocity_laplacian, divergence
from ..nonlinear import NewtonOptions, GMRESOptions, newton, normalized_linear_action
from ..transport import transport_numbers, implicit_policy


@dataclass(frozen=True)
class MACCouplingOptions:
    scheme: str = 'explicit-lagged'
    newton: NewtonOptions = field(default_factory=lambda: NewtonOptions(
        rtol=1e-8, atol=1e-9, max_iterations=12, max_backtracks=16,
        linear_tolerance_fraction=.2,
        linear=GMRESOptions(rtol=1e-2, atol=1e-11, restart=12, max_iterations=120, check_every=3)))
    tangent_chunk_size: int = 2048

    def __post_init__(self):
        if self.scheme not in ('explicit-lagged', 'implicit-newton'):
            raise ValueError('coupling scheme must be explicit-lagged or implicit-newton')
        if not isinstance(self.newton, NewtonOptions):
            raise ValueError('coupling newton must be NewtonOptions')
        if type(self.tangent_chunk_size) is not int or self.tangent_chunk_size < 1:
            raise ValueError('positive tangent_chunk_size required')


class ImplicitMACIBStepper(MACIBStepper):
    def __init__(self, flow, transfer, model, options, solid_execution=None):
        super().__init__(flow, transfer, model.force, model.validate)
        from ..ho_tangent import HOTangentAssembler
        from .execution import tensor_kernel
        self.model, self.options, self.solid_execution = model, options, solid_execution
        self.tangent = HOTangentAssembler(model, options.tangent_chunk_size)
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
        # Project final velocity, then re-evaluate the complete coupled residual:
        # accepted x is derived from the accepted u, and force is at (x,t_new).
        value = self.pack(evaluate(result.x)[1].velocity)
        validate(value)
        r, flow, force, density, spread, interpolation = evaluate(value)
        true_norm = torch.linalg.vector_norm(r).item()
        if true_norm > result.tolerance:
            raise RuntimeError('implicit MAC final projected residual exceeds Newton tolerance')
        velocity,x,_ = kinematics(value)
        fraction = ((x-state.x).abs()/x.new_tensor(grid.spacing)).max().item()
        new = MACState(state.step+1,target_time,x,velocity,flow.pressure,force,target_time)
        numbers = transport_numbers([u.abs().max().item() for u in velocity], grid.spacing, dt, self.flow.mu/self.flow.rho)
        info = dict(flow=dict(pressure=flow.diagnostics['pressure'], **numbers),
                    force_mass=asdict(spread), velocity_mass=asdict(interpolation),
                    max_grid_displacement=fraction, used_force_time_s=target_time,next_force_time_s=target_time,
                    nonlinear=dict(iterations=result.iterations,residual_norm=true_norm,
                                   tolerance=result.tolerance,history=result.history,
                                   solid_tangent='assembled CSR',ib_geometry='frozen at preceding accepted position'))
        if diagnostics:
            U,_ = self.transfer.interpolate(velocity,stencil)
            solid_power = (U*force).sum()
            fluid_power = grid.volume*sum((u*f).sum() for u,f in zip(velocity,density))
            info.update(solid_power=solid_power.item(),fluid_power=fluid_power.item(),
                        power_error=abs((solid_power-fluid_power).item()),
                        divergence_l2=(grid.volume*divergence(velocity,grid.spacing).square().sum()).sqrt().item())
        return new,info
