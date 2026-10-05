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
    """Unpreconditioned BiCGSTAB with a true-residual acceptance test."""
    norm = lambda v: torch.linalg.vector_norm(v).item()
    tolerance = max(options.atol, options.rtol*norm(rhs))
    x = torch.zeros_like(rhs)
    r = rhs.clone(); shadow = r.clone()
    p, v = torch.zeros_like(r), torch.zeros_like(r)
    rho_old, alpha, omega = 1., 1., 1.
    tiny = torch.finfo(r.dtype).tiny
    for k in range(options.max_iterations+1):
        actual = rhs-action(x)
        if norm(actual) <= tolerance:
            return x, dict(iterations=k, residual_norm=norm(actual), tolerance=tolerance)
        if k == options.max_iterations:
            break
        rho = (shadow*r).sum().item()
        if not isfinite(rho) or abs(rho) <= tiny or abs(omega) <= tiny:
            raise RuntimeError('BiCGSTAB scalar breakdown before true convergence')
        beta = (rho/rho_old)*(alpha/omega)
        p = r+beta*(p-omega*v)
        v = action(p)
        denominator = (shadow*v).sum().item()
        if not isfinite(denominator) or abs(denominator) <= tiny:
            raise RuntimeError('BiCGSTAB alpha breakdown')
        alpha = rho/denominator
        s = r-alpha*v
        trial = x+alpha*p
        if norm(s) <= tolerance:
            actual = rhs-action(trial)
            if norm(actual) <= tolerance:
                return trial, dict(iterations=k+1, residual_norm=norm(actual), tolerance=tolerance)
        t = action(s)
        tt = (t*t).sum().item()
        if not isfinite(tt) or tt <= tiny:
            raise RuntimeError('BiCGSTAB omega breakdown')
        omega = (t*s).sum().item()/tt
        x = trial+omega*s
        r = s-omega*t
        rho_old = rho
    raise RuntimeError(f'BiCGSTAB failed true residual: {norm(rhs-action(x)):g} > {tolerance:g}')


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
    tolerance = max(options.atol, options.rtol*norm(r))
    history = [dict(iteration=0, residual_norm=norm(r))]
    result = lambda ok: NewtonResult(y.clone(), ok, norm(r), tolerance, len(history)-1, list(history))
    for k in range(options.max_iterations):
        if norm(r) <= tolerance:
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
        old_norm = norm(r)
        for backtrack in range(options.max_backtracks+1):
            alpha = 2.**(-backtrack)
            trial = y+alpha*direction
            try:
                candidate = problem.residual(trial)
            except (ValueError, FloatingPointError):
                continue
            if norm(candidate) <= (1-options.armijo*alpha)*old_norm:
                y, r = trial, candidate
                history.append(dict(iteration=k+1, residual_norm=norm(r), linear=info, alpha=alpha))
                break
        else:
            raise NonlinearFailure('paper JFNK line search failed', result(False))
    if norm(r) > tolerance:
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

    def validate(self, y):
        self.driver.solid.validate(self.state.x+y)

    def residual(self, y):
        # Exact repeated iterate only. This includes the final Newton clone,
        # avoids repeating a complete fluid solve, and never reuses a trial
        # from another geometry/load time. Each problem is step-local.
        if self.last_y is not None and torch.equal(y, self.last_y):
            return self.last_residual.clone()
        x = self.state.x+y
        self.driver.solid.validate(x)
        force = self.driver.solid.force(x, self.time)
        density, spread = self.driver.transfer.spread(force, self.stencil)
        velocity, pressure, flow = self.driver.flow.advance(self.departure, density, self.pressure_initial)
        nodal, interpolation = self.driver.transfer.interpolate(velocity, self.stencil)
        residual = y-self.driver.flow.dt*nodal
        if not torch.isfinite(residual).all():
            raise FloatingPointError('nonfinite BE-BE coupling residual')
        self.pressure_initial = pressure
        self.last_y, self.last_residual = y.clone(), residual.clone()
        self.last_data = force, velocity, pressure, flow, nodal, density, spread, interpolation
        self.evaluations += 1
        return residual

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
        self.transfer.check_support(self.transfer.interaction_points(x))
        return PaperState(0, 0., x.clone(), self.grid.zeros(device=x.device, dtype=x.dtype),
            x.new_zeros(self.grid.shape), self.solid.force(x, 0.), 0.)

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
        residual = problem.residual(result.x)
        if torch.linalg.vector_norm(residual).item() > result.tolerance:
            raise NonlinearFailure('BE-BE final residual exceeds target', result)
        x = state.x+result.x
        self.solid.validate(x)
        self.transfer.check_support(self.transfer.interaction_points(x))
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
