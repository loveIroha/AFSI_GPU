"""Old-geometry dual IB + endpoint nonlinear force for the BE-BE scheme.

The unknown y=X[n+1]-X[n] satisfies y-dt*I(X[n],u[n+1](y))=0.
Semi-Lagrangian departure velocity and the identical spread/gather stencil
are prepared ONCE per step and remain fixed for every nonlinear trial.
"""
from dataclasses import dataclass, asdict
from math import isfinite
import torch
from .coupling import MACState
from .backward_euler import BackwardEulerFlow
from .grid import divergence
from .open_boundary import face_weights
from .adaptive_transfer import AdaptiveP1Transfer
from .compact_transfer import CompactFETransfer
from .midpoint_solver import accelerated_midpoint
from ..nonlinear import newton, normalized_linear_action, NewtonResult, NonlinearFailure


@dataclass(frozen=True)
class PaperState(MACState):
    previous_x: torch.Tensor | None = None


@torch.no_grad()
def bicgstab(action, rhs, options):
    """BiCGSTAB with periodic true residuals and mandatory acceptance checks.

    ``action`` is an expensive coupled response. Use recurrence residuals
    between checks, and restart from the true residual if a convergence
    candidate is rejected or recurrence drift becomes significant.
    """
    calls = checks = restarts = reads = 0
    def norm(v):
        nonlocal reads
        reads += 1
        return torch.linalg.vector_norm(v).item()
    def apply(v):
        nonlocal calls
        calls += 1
        return action(v)
    def verify(x):
        nonlocal checks
        checks += 1
        actual = rhs-apply(x)
        size = norm(actual)
        if not isfinite(size):
            raise RuntimeError('nonfinite BiCGSTAB true residual')
        return actual, size
    def report(k, size):
        return dict(iterations=k, residual_norm=size, tolerance=tolerance,
            jacobian_actions=calls, true_residual_checks=checks,
            residual_restarts=restarts, scalar_reads=reads)
    rhs_norm = norm(rhs)
    if not isfinite(rhs_norm):
        raise ValueError('nonfinite BiCGSTAB right-hand side')
    tolerance = max(options.atol, options.rtol*rhs_norm)
    x = torch.zeros_like(rhs)
    r, r_norm = verify(x)
    if r_norm <= tolerance:
        return x, report(0, r_norm)
    shadow = r.clone()
    p, v = torch.zeros_like(r), torch.zeros_like(r)
    rho_old, alpha, omega = 1., 1., 1.
    tiny = torch.finfo(r.dtype).tiny
    for k in range(1, options.max_iterations+1):
        reads += 1
        rho = (shadow*r).sum().item()
        if not isfinite(rho) or abs(rho) <= tiny or abs(omega) <= tiny:
            raise RuntimeError('BiCGSTAB scalar breakdown before true convergence')
        beta = (rho/rho_old)*(alpha/omega)
        p = r+beta*(p-omega*v)
        v = apply(p)
        reads += 1
        denominator = (shadow*v).sum().item()
        if not isfinite(denominator) or abs(denominator) <= tiny:
            raise RuntimeError('BiCGSTAB alpha breakdown')
        alpha = rho/denominator
        s = r-alpha*v
        trial = x+alpha*p
        if norm(s) <= tolerance:
            actual, actual_norm = verify(trial)
            if actual_norm <= tolerance:
                return trial, report(k, actual_norm)
            # A rejected candidate must not feed an almost-zero recursive
            # residual into the next omega solve (notably for FD JVPs).
            x, r = trial, actual
            shadow = r.clone(); p.zero_(); v.zero_()
            rho_old, alpha, omega = 1., 1., 1.
            restarts += 1
            continue
        t = apply(s)
        # One host transfer for both dot products, rather than two syncs.
        reads += 1
        tt, ts = torch.stack(((t*t).sum(), (t*s).sum())).tolist()
        if not isfinite(tt) or tt <= tiny:
            raise RuntimeError('BiCGSTAB omega breakdown')
        omega = ts/tt
        if not isfinite(omega):
            raise RuntimeError('BiCGSTAB nonfinite omega')
        x = trial+omega*s
        r = s-omega*t
        rho_old = rho
        r_norm = norm(r)
        if not isfinite(r_norm):
            raise RuntimeError('nonfinite BiCGSTAB recurrence residual')
        if r_norm <= tolerance or k % options.check_every == 0 or k == options.max_iterations:
            actual, actual_norm = verify(x)
            if actual_norm <= tolerance:
                return x, report(k, actual_norm)
            # A norm of the difference protects against directional drift,
            # including residuals whose norms happen to remain similar.
            drift = norm(actual-r)
            if r_norm <= tolerance or drift > .1*max(actual_norm, tolerance):
                r = actual; shadow = r.clone(); p.zero_(); v.zero_()
                rho_old, alpha, omega = 1., 1., 1.
                restarts += 1
    # The last iteration already verified its current x, including the
    # rejected s-candidate branch. Do not repeat that expensive action.
    raise RuntimeError(f'BiCGSTAB failed true residual: {actual_norm:g} > {tolerance:g}')


@torch.no_grad()
def jfnk(problem, initial, options):
    """Paper-style finite-difference Newton/BiCGSTAB and Armijo line search.

    Uses displacement-scaled residuals and finite-difference step sizes.
    Inner tolerances are implementation choices, recorded in the report.
    """
    from dataclasses import replace
    norm = lambda v: torch.linalg.vector_norm(v).item()
    y = initial.clone()
    r = problem.residual(y)
    r_norm = norm(r)
    tolerance = max(options.atol, options.rtol*r_norm)
    history = [dict(iteration=0, residual_norm=r_norm)]
    result = lambda ok: NewtonResult(y.clone(), ok, r_norm, tolerance, len(history)-1, list(history))
    for k in range(options.max_iterations):
        if r_norm <= tolerance:
            return result(True)
        base_y, base_r = y.clone(), r.clone()
        scale = torch.finfo(y.dtype).eps**.5*(1+norm(y))
        def action(v):
            n = norm(v)
            if n == 0:
                return torch.zeros_like(v)
            epsilon = scale/n
            return (problem.residual(base_y+epsilon*v)-base_r)/epsilon
        linear = replace(options.linear, atol=max(options.linear.atol, options.linear_tolerance_fraction*tolerance))
        try:
            direction, info = bicgstab(action, -r, linear)
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            raise NonlinearFailure(f'paper JFNK linear solve failed: {exc}', result(False)) from exc
        old_norm = r_norm
        for backtrack in range(options.max_backtracks+1):
            alpha = 2.**(-backtrack)
            trial = y+alpha*direction
            try:
                candidate = problem.residual(trial)
            except (ValueError, FloatingPointError):
                continue
            candidate_norm = norm(candidate)
            if candidate_norm <= (1-options.armijo*alpha)*old_norm:
                y, r = trial, candidate
                r_norm = candidate_norm
                history.append(dict(iteration=k+1, residual_norm=r_norm, linear=info, alpha=alpha))
                break
        else:
            raise NonlinearFailure('paper JFNK line search failed', result(False))
    if r_norm > tolerance:
        raise NonlinearFailure('paper JFNK nonlinear iteration budget exhausted', result(False))
    return result(True)


class BEProblem:
    def __init__(self, driver, state, stencil, departure):
        self.driver, self.state, self.stencil, self.departure = driver, state, stencil, departure
        self.time = state.time+driver.flow.dt
        self.last_y = self.last_residual = None
        self.last_data = None
        self.pressure_initial = state.pressure
        self.evaluations = 0
        self._trial_y = self._trial_version = self._trial_x = None
        self._trial_state_version = self._validated_x_version = None
        self._last_input = self._last_version = self._last_state_version = self._last_x = None
        self._trial_output_version = self._last_x_version = None

    @staticmethod
    def _version(x):
        return None if torch.is_inference(x) else x._version

    def configuration(self, y):
        version, state_version = self._version(y), self._version(self.state.x)
        if (version is not None and state_version is not None and y is self._trial_y
                and version == self._trial_version and state_version == self._trial_state_version
                and self._version(self._trial_x) == self._trial_output_version):
            return self._trial_x
        self._trial_x = self.state.x+y
        self._trial_y, self._trial_version, self._trial_state_version = y, version, state_version
        self._trial_output_version = self._version(self._trial_x)
        self._validated_x_version = None
        return self._trial_x

    def validate(self, y):
        x = self.configuration(y)
        version = self._version(x)
        if version is not None and version == self._validated_x_version:
            return
        self.driver.solid.validate(x)
        self._validated_x_version = version

    def residual(self, y):
        # Identity/version cache avoids a GPU-wide equality reduction on
        # every distinct FD probe. Cloned accepted iterates are checked once
        # in accepted(), not for every residual evaluation.
        version, state_version = self._version(y), self._version(self.state.x)
        if (version is not None and state_version is not None and y is self._last_input
                and version == self._last_version and state_version == self._last_state_version
                and self._version(self._last_x) == self._last_x_version):
            return self.last_residual.clone()
        x = self.configuration(y)
        self.validate(y)
        force = self.driver.solid.force(x, self.time)
        density, spread = self.driver.transfer.spread(force, self.stencil)
        velocity, pressure, flow = self.driver.flow.advance(self.departure, density, self.pressure_initial)
        nodal, interpolation = self.driver.transfer.interpolate(velocity, self.stencil)
        residual = y-self.driver.flow.dt*nodal
        if not torch.isfinite(residual).all():
            raise FloatingPointError('nonfinite BE-BE coupling residual')
        self.pressure_initial = pressure
        self.last_y, self.last_residual = y.clone(), residual.clone()
        self._last_input, self._last_version, self._last_state_version = y, version, state_version
        self._last_x = x
        self._last_x_version = self._version(x)
        self.last_data = force, velocity, pressure, flow, nodal, density, spread, interpolation
        self.evaluations += 1
        return residual

    def accepted(self, y):
        state_version = self._version(self.state.x)
        if (self.last_y is not None and state_version is not None
                and state_version == self._last_state_version
                and self._version(self._last_x) == self._last_x_version and torch.equal(y, self.last_y)):
            return self.last_residual.clone(), self._last_x
        return self.residual(y), self._last_x

    def linearization(self, y):
        d = self.driver
        if d.assembler is None:
            d.assembler = d.model.tangent_factory()
        tangent = d.assembler.assemble(self.state.x+y, self.time)
        def action(v):
            df = torch.sparse.mm(tangent, v.reshape(-1, 1)).reshape_as(v)
            # No warm-start-dependent affine offset in the derivative map.
            d.transfer.reset_warm_start()
            density, _ = d.transfer.spread(df, self.stencil)
            fluid = d.flow.response(density)
            nodal, _ = d.transfer.interpolate(fluid, self.stencil)
            return v-d.flow.dt*nodal
        return normalized_linear_action(action)

    def preconditioner(self, y):
        return None


class BEIBStepper:
    def __init__(self, model, config, device):
        from .grid import MACGrid
        self.model, self.config = model, config
        self.grid = MACGrid(config.fluid.shape, config.fluid.lengths, config.fluid.origin)
        fused = config.execution.execution_backend == 'fused'
        self.flow = BackwardEulerFlow(self.grid, dt=config.time.dt, rho=config.fluid.rho, mu=config.fluid.mu,
            device=device, dtype=model.mesh.X.dtype, pressure_options=config.pressure_solver,
            options=config.flow, backend=config.execution.execution_backend,
            pressure_backend=config.execution.pressure_backend)
        from ..p1 import prepare_p1
        geometry = prepare_p1(model.mesh.X, model.mesh.cells, config.interaction_degree)
        cls = AdaptiveP1Transfer if config.interaction_quadrature.mode == 'adaptive' else CompactFETransfer
        extra = dict(quadrature_options=config.interaction_quadrature, fused=fused) if cls is AdaptiveP1Transfer else {}
        self.transfer = cls(self.grid, geometry, warm_start=config.execution.warm_start,
            mass_backend=config.execution.mass_backend, options=config.mass_solver, **extra)
        self.solid = model.execution_factory() if fused else model
        self.assembler = None

    def initialize(self, x):
        self.solid.validate(x)
        self.check_support(x)
        return PaperState(0, 0., x.clone(), self.grid.zeros(device=x.device, dtype=x.dtype),
            x.new_zeros(self.grid.shape), self.solid.force(x, 0.), 0.)

    def check_support(self, x):
        if self.config.support_backend == 'vertices':
            self.transfer.check_configuration_support(x)
        else:
            self.transfer.check_support(self.transfer.interaction_points(x))

    @torch.no_grad()
    def step(self, state, *, diagnostics=True):
        if state.step < 0 or abs(state.time-state.step*self.flow.dt) > 1e-12:
            raise ValueError('inconsistent paper BE-BE state clock')
        self.solid.validate(state.x)
        # Frozen OLD geometry, not midpoint or new geometry.
        stencil = self.transfer.prepare(state.x)
        departure = self.flow.advect(state.velocity)
        problem = BEProblem(self, state, stencil, departure)
        initial = torch.zeros_like(state.x) if state.previous_x is None else state.x-state.previous_x
        before = self.flow.calls
        solver = self.config.nonlinear_solver
        acceleration = {}
        if solver == 'jfnk':
            result = jfnk(problem, initial, self.config.nonlinear)
        elif solver == 'newton':
            result = newton(problem.residual, initial, validate=problem.validate,
                linearization_factory=problem.linearization, options=self.config.nonlinear)
        else:
            result, acceleration = accelerated_midpoint(problem, initial, self.config.nonlinear,
                self.config.anderson, newton_solve=newton)
        # Reuse only the exact accepted iterate; validate support at endpoint.
        residual, x = problem.accepted(result.x)
        if torch.linalg.vector_norm(residual).item() > result.tolerance:
            raise NonlinearFailure('BE-BE final residual exceeds target', result)
        self.solid.validate(x)
        self.check_support(x)
        force, velocity, pressure, flow, nodal, density, spread, interpolation = problem.last_data
        courant = self.flow.dt*sum(u.abs().max()/h for u, h in zip(velocity, self.grid.spacing)).item()
        if not isfinite(courant) or courant > self.config.max_courant:
            raise ValueError(f'BE/semi-Lagrangian characteristic CFL {courant:g} > {self.config.max_courant:g}; reduce dt')
        next_state = PaperState(state.step+1, (state.step+1)*self.flow.dt, x, velocity, pressure,
            force, problem.time, pressure_time=problem.time, previous_x=state.x.clone())
        info = dict(nonlinear=dict(iterations=result.iterations,
            residual_norm=result.residual_norm, tolerance=result.tolerance, history=result.history,
            residual_evaluations=problem.evaluations, fluid_solves=self.flow.calls-before,
            **dict(acceleration, solver=solver)),
            flow=flow, force_mass=asdict(spread), velocity_mass=asdict(interpolation), courant=courant,
            max_grid_displacement=(result.x.abs()/x.new_tensor(self.grid.spacing)).max().item())
        linear_history = [entry['linear'] for entry in result.history if 'linear' in entry]
        if solver == 'jfnk':
            info['nonlinear'].update({key: sum(entry.get(key, 0) for entry in linear_history)
                for key in ('jacobian_actions', 'true_residual_checks', 'residual_restarts', 'scalar_reads')})
        if diagnostics:
            # Pressure boundary faces have half weights. IB support is wholly
            # interior, so the usual transfer h^3 power remains equivalent.
            fluid_power = sum((u*f*face_weights(self.grid, c, x)).sum()
                              for c, (u, f) in enumerate(zip(velocity, density)))
            solid_power = (nodal*force).sum()
            info.update(divergence_l2=(self.grid.volume*divergence(velocity, self.grid.spacing).square().sum()).sqrt().item(),
                solid_power=solid_power.item(), fluid_power=fluid_power.item(),
                power_error=abs((fluid_power-solid_power).item()))
        return next_state, info
