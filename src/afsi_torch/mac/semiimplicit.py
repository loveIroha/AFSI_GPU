"""CN--AB2 with nonlinear midpoint elasticity, reduced to FE node unknowns.

IB geometry is frozen at the predicted midpoint. The actual force is evaluated
at the solved midpoint; the solid adapter supplies its complete assembled CSR
force derivative, including boundary loads. Fluid pressure/velocity are eliminated with the
same no-slip CN Stokes solver, not a commuting or single-projection substitute.
"""
from contextlib import contextmanager
from dataclasses import asdict, replace, dataclass
from math import isfinite
import torch
from .cnab import MidpointMACIBStepper, blend
from .coupling import MACState
from .grid import zero_normal, divergence
from ..nonlinear import newton, normalized_linear_action, NonlinearFailure


@dataclass(frozen=True)
class _ValidatedMidpoint:
    source: torch.Tensor
    version: int
    anchors: tuple
    snapshot: torch.Tensor
    midpoint: torch.Tensor
    endpoint: torch.Tensor
    mid_geometry: tuple | None
    end_geometry: tuple | None
    midpoint_version: int
    endpoint_version: int


class MidpointProblem:
    """R(y)=x_hat+y-x_n-dt/4 J_hat(u_n+u(y)), with y shaped (nodes,3)."""
    def __init__(self, driver, state, predicted, stencil, advection):
        self.driver, self.state, self.predicted, self.stencil = driver, state, predicted, stencil
        self.dt, self.half_time = driver.flow.dt, state.time+.5*driver.flow.dt
        zeros = driver.flow.grid.zeros(device=state.x.device, dtype=state.x.dtype)
        self.base_rhs = driver.flow._right(state.velocity, advection, zeros)
        self.tangent = self.inverse_diagonal = self.last_evaluation = None
        self.evaluations = self.actions = self.assemblies = 0
        self._last_y = None
        self._last_stokes_call = None
        self._pressure_guess = None
        self.pressure_warm_starts = self.pressure_warm_fallbacks = 0
        self._validated = None
        self.validation_evaluations = self.validation_reuses = 0

    def _anchors(self):
        tensors=(self.predicted,self.state.x)
        if any(torch.is_inference(x) for x in tensors):
            return None
        return tuple((id(x),x._version) for x in tensors)

    def _validation_cache(self,y,*,equal=False):
        cache=self._validated
        if not self.driver.options.reuse_validation or cache is None or torch.is_inference(y):
            return None
        if y.shape!=cache.snapshot.shape or y.dtype!=cache.snapshot.dtype or y.device!=cache.snapshot.device:
            return None
        if (self._anchors()!=cache.anchors or cache.midpoint._version!=cache.midpoint_version or
                cache.endpoint._version!=cache.endpoint_version):
            return None
        if (cache.source is y and cache.version==y._version) or (equal and torch.equal(y,cache.snapshot)):
            return cache
        return None

    def validate(self, y):
        if self._validation_cache(y) is not None:
            self.validation_reuses += 1
            return
        self._validated=None
        self.validation_evaluations += 1
        x = self.predicted+y
        self.driver.validate(x)
        execution=self.driver.solid_execution
        checked=getattr(execution,'checked_geometry',lambda x:None)
        mid_geometry=checked(x)
        endpoint=2*x-self.state.x
        self.driver.validate(endpoint)
        end_geometry=checked(endpoint)
        transfer = self.driver.transfer
        points = transfer.validation_points(endpoint) if hasattr(transfer,'validation_points') else transfer.interaction_points(endpoint)
        transfer.check_support(points)
        fraction = ((2*x-2*self.state.x).abs()/x.new_tensor(self.driver.flow.grid.spacing)).max().item()
        if not isfinite(fraction) or fraction > self.driver.max_displacement:
            raise ValueError('semi-implicit solid displacement exceeds grid-based limit; reduce dt')
        anchors=self._anchors()
        if (self.driver.options.reuse_validation and anchors is not None and
                not any(torch.is_inference(t) for t in (y,x,endpoint))):
            self._validated=_ValidatedMidpoint(y,y._version,anchors,y.detach().clone(),x,endpoint,
                                               mid_geometry,end_geometry,x._version,endpoint._version)

    def validate_final(self,y):
        # Solver results may be owned clones of the last checked iterate. Exact
        # equality and untouched anchors/coordinates permit that final reuse.
        cache=self._validation_cache(y,equal=True)
        if cache is None:
            return self.validate(y)
        self.validation_reuses += 1
        self._validated=replace(cache,source=y,version=y._version)

    def endpoint(self,y):
        cache=self._validation_cache(y)
        if cache is None:
            return 2*(self.predicted+y)-self.state.x
        if cache.end_geometry is not None:
            self.driver.solid_execution.remember_geometry(cache.endpoint,cache.end_geometry)
        return cache.endpoint

    def evaluate(self, y):
        # A failed evaluation must never leave an eligible older response.
        self._last_y = None
        d = self.driver
        cache=self._validation_cache(y)
        x = self.predicted+y if cache is None else cache.midpoint
        if cache is not None and cache.mid_geometry is not None:
            d.solid_execution.remember_geometry(x,cache.mid_geometry)
        force = d._force_geometry(x, self.half_time)
        density, spread = d.transfer.spread(force, self.stencil)
        rhs = zero_normal(tuple(b+self.dt/d.flow.rho*f for b,f in zip(self.base_rhs,density)))
        if d.options.stokes_warm_start and self._pressure_guess is not None:
            self.pressure_warm_starts += 1
            try:
                flow = d.flow.stokes(rhs,self._pressure_guess)
            except RuntimeError:
                # A difficult warm guess cannot weaken acceptance or turn an
                # otherwise solvable residual into a rejected time step.
                self.pressure_warm_fallbacks += 1
                flow = d.flow.stokes(rhs,self.state.pressure)
        else:
            flow = d.flow.stokes(rhs,self.state.pressure)
        average = blend(self.state.velocity, flow.velocity)
        U, interpolation = d.transfer.interpolate(average, self.stencil)
        residual = x-self.state.x-.5*self.dt*U
        self.evaluations += 1
        self.last_midpoint = x.detach()
        self.last_evaluation = (residual, flow, force, density, spread, interpolation, U)
        self._last_y = y.detach().clone()
        self._last_stokes_call = d.flow.stokes_calls
        if d.options.stokes_warm_start:
            # An owned snapshot: linear responses, graph workspaces and later
            # trials must not overwrite the last successful nonlinear guess.
            self._pressure_guess = flow.pressure.detach().clone()
        return self.last_evaluation

    def final_evaluation(self,y,*,reuse=True):
        """Reuse only the exact last nonlinear point with untouched flow state.

        Geometry validation and final residual/momentum/divergence acceptance
        are still performed by the caller. This is not tolerance-based reuse.
        """
        same = (reuse and self._last_y is not None and
                y.shape==self._last_y.shape and y.dtype==self._last_y.dtype and
                y.device==self._last_y.device and
                self.driver.flow.stokes_calls==self._last_stokes_call and
                torch.equal(y,self._last_y))
        if same:
            return self.last_evaluation,True
        return self.evaluate(y),False

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
            # Linear solves may reuse pressure/mass workspaces. Their response
            # cannot be mistaken for an eligible nonlinear acceptance cache.
            self._last_y = None
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
        from ..solids.contracts import make_tangent
        self.model, self.options = model, options
        self.tangent = make_tangent(model, options.tangent_chunk_size)
        self.lumped_mass = transfer.mass_action(torch.ones_like(model.mesh.X))
        if not torch.isfinite(self.lumped_mass).all() or (self.lumped_mass <= 0).any():
            raise ValueError('positive P1 row-sum mass required for the preconditioner')

    @torch.no_grad()
    def step(self, state, *, diagnostics=True):
        dt = self.flow.dt
        start_stokes_calls = self.flow.stokes_calls
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
            if self.options.semiimplicit_solver=='anderson-newton':
                from .midpoint_solver import accelerated_midpoint
                result,solver_info = accelerated_midpoint(problem,torch.zeros_like(state.x),
                    self.options.newton,self.options.anderson,newton_solve=newton)
            else:
                result = newton(problem.residual, torch.zeros_like(state.x), validate=problem.validate,
                    linearization_factory=problem.linearization, preconditioner_factory=problem.preconditioner,
                    options=self.options.newton)
                solver_info = dict(solver='newton',anderson_iterations=0,newton_iterations=result.iterations,
                                   newton_fallback=False,fallback_reason=None)
            problem.validate_final(result.x)
            final,final_reused = problem.final_evaluation(result.x,reuse=self.options.reuse_final_evaluation)
            r, flow, half_force, density, spread, interpolation, U = final
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
            x_new = problem.endpoint(result.x)
            force_new = self._force_geometry(x_new, (state.step+1)*dt)
            transport_sample = 'candidate endpoint; rejected if the screen fails'
            numbers = self.flow.check_transport(flow.velocity, density)
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            if getattr(exc,'diagnostics',{}).get('triggered')==['courant']:
                from ..transport import semiimplicit_policy
                exc.diagnostics.update(semiimplicit_policy(),sampled_velocity=transport_sample)
            try:
                from ..solids.contracts import failure_diagnostics
                exc.coupled_diagnostics = dict(getattr(exc, 'coupled_diagnostics', {}),
                    local=failure_diagnostics(self.model, self.flow.grid, state, problem))
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
                **solver_info,
                history=result.history, acceptance=acceptance, solid_tangent='assembled CSR nodal-force derivative',
                unknown_dofs=state.x.numel(), residual_evaluations=problem.evaluations,
                final_evaluation_reused=final_reused,
                stokes_pressure_warm_starts=problem.pressure_warm_starts,
                stokes_pressure_warm_fallbacks=problem.pressure_warm_fallbacks,
                stokes_solves=self.flow.stokes_calls-start_stokes_calls,
                tangent_assemblies=problem.assemblies, jacobian_actions=problem.actions,
                validation_evaluations=problem.validation_evaluations,validation_reuses=problem.validation_reuses,
                preconditioner='diagonal mass/stiffness approximation; consistent mass retained in equations'))
        if diagnostics:
            average = blend(state.velocity, flow.velocity)
            solid_power = (U*half_force).sum()
            fluid_power = self.flow.grid.volume*sum((u*f).sum() for u,f in zip(average,density))
            info.update(solid_power=solid_power.item(), fluid_power=fluid_power.item(),
                power_error=abs((solid_power-fluid_power).item()),
                divergence_l2=(self.flow.grid.volume*divergence(flow.velocity,self.flow.grid.spacing).square().sum()).sqrt().item())
        return new, info
