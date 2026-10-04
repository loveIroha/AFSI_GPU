"""CN--AB2 with nonlinear midpoint elasticity, reduced to FE node unknowns.

IB geometry is frozen at the predicted midpoint. The actual force is evaluated
at the solved midpoint; its assembled CSR derivative includes H-O, follower
pressure and basal traction. Fluid pressure/velocity are eliminated with the
same no-slip CN Stokes solver, not a commuting or single-projection substitute.
"""
from contextlib import contextmanager
from dataclasses import asdict, replace
from math import isfinite
import torch
from .cnab import MidpointMACIBStepper, blend
from .coupling import MACState
from .grid import zero_normal, divergence
from ..nonlinear import newton, normalized_linear_action, NonlinearFailure


class MidpointProblem:
    """R(y)=x_hat+y-x_n-dt/4 J_hat(u_n+u(y)), with y shaped (nodes,3)."""
    def __init__(self, driver, state, predicted, stencil, advection):
        self.driver, self.state, self.predicted, self.stencil = driver, state, predicted, stencil
        self.dt, self.half_time = driver.flow.dt, state.time+.5*driver.flow.dt
        zeros = driver.flow.grid.zeros(device=state.x.device, dtype=state.x.dtype)
        self.base_rhs = driver.flow._right(state.velocity, advection, zeros)
        self.tangent = self.inverse_diagonal = self.last_evaluation = None
        self.evaluations = self.actions = self.assemblies = 0

    def validate(self, y):
        x = self.predicted+y
        self.driver.validate(x)
        self.driver.validate(2*x-self.state.x)
        self.driver.transfer.check_support(self.driver.transfer.interaction_points(2*x-self.state.x))
        fraction = ((2*x-2*self.state.x).abs()/x.new_tensor(self.driver.flow.grid.spacing)).max().item()
        if not isfinite(fraction) or fraction > self.driver.max_displacement:
            raise ValueError('semi-implicit solid displacement exceeds grid-based limit; reduce dt')

    def evaluate(self, y):
        d = self.driver
        x = self.predicted+y
        force = d._force_geometry(x, self.half_time)
        density, spread = d.transfer.spread(force, self.stencil)
        rhs = zero_normal(tuple(b+self.dt/d.flow.rho*f for b,f in zip(self.base_rhs,density)))
        flow = d.flow.stokes(rhs, self.state.pressure)
        average = blend(self.state.velocity, flow.velocity)
        U, interpolation = d.transfer.interpolate(average, self.stencil)
        residual = x-self.state.x-.5*self.dt*U
        self.evaluations += 1
        self.last_midpoint = x.detach()
        self.last_evaluation = (residual, flow, force, density, spread, interpolation, U)
        return self.last_evaluation

    def residual(self, y):
        return self.evaluate(y)[0]

    @contextmanager
    def cold_linear_transfer(self):
        # Arnoldi directions are not production force/velocity guesses. Do not
        # let their very different scales contaminate the trajectory's caches.
        t = self.driver.transfer
        warm = t.warm_start
        t.warm_start = False
        try:
            yield
        finally:
            t.warm_start = warm

    def linearization(self, y):
        d = self.driver
        self.tangent = d.tangent.assemble(self.predicted+y, self.half_time)
        self.assemblies += 1
        T = self.tangent
        # Local diagonal mobility is a RIGHT PRECONDITIONER only. The residual,
        # Jacobian and both IB directions still use the consistent mass solves.
        diagonal = d.tangent.diagonal(T)
        self.inverse_diagonal = (1+self.dt**2/(4*d.flow.rho)*diagonal.abs().reshape_as(y)/d.lumped_mass).reciprocal()
        def action(v):
            with self.cold_linear_transfer():
                force = torch.sparse.mm(T, v.reshape(-1,1)).reshape_as(v)
                density, _ = d.transfer.spread(force, self.stencil)
                response = d.flow.linear_response(tuple(self.dt/d.flow.rho*f for f in density))
                U, _ = d.transfer.interpolate(response, self.stencil)
            self.actions += 1
            return v-.25*self.dt*U
        return normalized_linear_action(action)

    def preconditioner(self, y):
        # newton calls the linearization factory before this factory.
        inverse = self.inverse_diagonal
        return lambda v: inverse*v


class SemiImplicitMACIBStepper(MidpointMACIBStepper):
    def __init__(self, flow, transfer, model, options, solid_execution=None, *, optimized=False):
        solid = model if solid_execution is None else solid_execution
        super().__init__(flow, transfer, solid.force, solid.validate,
                         optimized=optimized, solid_execution=solid_execution)
        from ..ho_tangent import HOTangentAssembler
        self.model, self.options = model, options
        self.tangent = HOTangentAssembler(model, options.tangent_chunk_size)
        self.lumped_mass = transfer.mass_action(torch.ones_like(model.mesh.X))
        if not torch.isfinite(self.lumped_mass).all() or (self.lumped_mass <= 0).any():
            raise ValueError('positive P1 row-sum mass required for the preconditioner')

    @torch.no_grad()
    def step(self, state, *, diagnostics=True):
        dt = self.flow.dt
        if type(state.step) is not int or state.step < 0 or not isfinite(state.time) or abs(state.time-state.step*dt)>1e-12:
            raise ValueError('inconsistent semi-implicit CNAB state clock')
        self.validate(state.x)
        stencil_n = self.transfer.prepare(state.x)
        U_n, predictor_mass = self.transfer.interpolate(state.velocity, stencil_n)
        predicted = state.x+.5*dt*U_n
        current_adv = self.flow.advection(state.velocity)
        startup = state.previous_advection is None
        startup_info = None
        if startup:
            force = self._force_geometry(state.x, state.time)
            density, _ = self.transfer.spread(force, stencil_n)
            provisional = self.flow.advance(state.velocity, density, current_adv, state.pressure)
            half_adv = self.flow.advection(blend(state.velocity, provisional.velocity))
            startup_info = provisional.diagnostics
        else:
            self.flow.grid.check_velocity(state.previous_advection)
            half_adv = blend(current_adv, state.previous_advection, 1.5, -.5)
        stencil = self.transfer.prepare(predicted)
        problem = MidpointProblem(self, state, predicted, stencil, half_adv)
        transport_sample = 'accepted input'
        try:
            # Convection remains explicitly extrapolated and retains its CFL
            # screen; viscosity and structural force are coupled implicitly.
            self.flow.check_transport(state.velocity, self.flow.grid.zeros(device=state.x.device,dtype=state.x.dtype))
            result = newton(problem.residual, torch.zeros_like(state.x), validate=problem.validate,
                linearization_factory=problem.linearization, preconditioner_factory=problem.preconditioner,
                options=self.options.newton)
            problem.validate(result.x)
            r, flow, half_force, density, spread, interpolation, U = problem.evaluate(result.x)
            norm = torch.linalg.vector_norm(r).item()
            acceptance = dict(stage='solid-midpoint-coupled-state', residual_norm=norm, tolerance=result.tolerance,
                step=state.step+1, time_s=(state.step+1)*dt, unknown_dofs=state.x.numel(),
                fluid_momentum=flow.diagnostics['stokes'], pressure=flow.diagnostics['pressure'])
            if not isfinite(norm) or norm > result.tolerance:
                error = NonlinearFailure('semi-implicit final midpoint kinematic residual exceeds tolerance',
                                         replace(result, converged=False, residual_norm=norm))
                error.coupled_diagnostics = acceptance
                raise error
            # Retain the solved structural unknown. An extra unguarded Picard
            # update can amplify the very stiffness that this solve controls.
            x_new = 2*(predicted+result.x)-state.x
            force_new = self._force_geometry(x_new, (state.step+1)*dt)
            transport_sample = 'candidate endpoint; rejected if the screen fails'
            numbers = self.flow.check_transport(flow.velocity, density)
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            if getattr(exc,'diagnostics',{}).get('triggered')==['courant']:
                from ..transport import semiimplicit_policy
                exc.diagnostics.update(semiimplicit_policy(),sampled_velocity=transport_sample)
            try:
                from ..real_lv_diagnostics import coupled_failure_diagnostics
                exc.coupled_diagnostics = dict(getattr(exc, 'coupled_diagnostics', {}),
                    local=coupled_failure_diagnostics(self.model, self.flow.grid, state, problem))
            except Exception as diagnosis_error:
                exc.coupled_diagnostics = dict(getattr(exc, 'coupled_diagnostics', {}),
                    diagnostic_error=str(diagnosis_error))
            raise
        new = MACState(state.step+1, (state.step+1)*dt, x_new, flow.velocity, flow.pressure,
                       force_new, (state.step+1)*dt, tuple(a.detach().clone() for a in current_adv))
        fraction = ((x_new-state.x).abs()/x_new.new_tensor(self.flow.grid.spacing)).max().item()
        info = dict(flow=dict(flow.diagnostics, **numbers), force_mass=asdict(spread),
            velocity_mass=asdict(interpolation), predictor_mass=asdict(predictor_mass),
            max_grid_displacement=fraction, used_force_time_s=problem.half_time,
            next_force_time_s=new.time, time_integrator='CN-AB2/implicit-midpoint-elasticity',
            startup_predictor_corrector=startup, startup_flow=startup_info,
            ib_geometry='frozen at predicted midpoint', force_sampling='solved midpoint',
            nonlinear=dict(iterations=result.iterations, residual_norm=norm, tolerance=result.tolerance,
                history=result.history, acceptance=acceptance, solid_tangent='assembled CSR nodal-force derivative',
                unknown_dofs=state.x.numel(), residual_evaluations=problem.evaluations,
                tangent_assemblies=problem.assemblies, jacobian_actions=problem.actions,
                preconditioner='diagonal mass/stiffness approximation; consistent mass retained in equations'))
        if diagnostics:
            average = blend(state.velocity, flow.velocity)
            solid_power = (U*half_force).sum()
            fluid_power = self.flow.grid.volume*sum((u*f).sum() for u,f in zip(average,density))
            info.update(solid_power=solid_power.item(), fluid_power=fluid_power.item(),
                power_error=abs((solid_power-fluid_power).item()),
                divergence_l2=(self.flow.grid.volume*divergence(flow.velocity,self.flow.grid.spacing).square().sum()).sqrt().item())
        return new, info
