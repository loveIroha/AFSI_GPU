"""Residual-only Anderson acceleration with a CSR Newton fallback.

Only the nonlinear solver changes. The midpoint residual, physical parameters,
consistent mass systems and final acceptance tolerance are identical to Newton.
History is local to a time step; a rejected trial is never a simulation state.
"""
from dataclasses import dataclass, replace
from math import isfinite
import torch
from ..nonlinear import NewtonResult, NonlinearFailure


@dataclass(frozen=True)
class AndersonOptions:
    max_iterations: int = 6
    history_size: int = 4
    growth_limit: float = 4.
    regularization: float = 1e-12

    def __post_init__(self):
        for n in (self.max_iterations,self.history_size):
            if type(n) is not int or n < 1:
                raise ValueError('positive Anderson iteration/history limits required')
        if not isfinite(self.growth_limit) or self.growth_limit < 1:
            raise ValueError('Anderson growth_limit must be finite and >=1')
        if not isfinite(self.regularization) or not 0 < self.regularization < 1:
            raise ValueError('Anderson regularization must be in (0,1)')


@torch.no_grad()
def accelerated_midpoint(problem, initial, options, acceleration, *, newton_solve):
    """Solve the SAME true residual; acceleration has a bounded trial budget.

    Type-II Anderson: min ||r - dR gamma||, x_next=x-r-(dX-dR)gamma.
    Columns are normalized; a regularized small Gram solve stays on device.
    A limited nonmonotone inner probe supplies secant information even for
    a mildly noncontractive fixed-point map. Actual residual growth/geometry
    checks bound these probes. Failure/slow progress falls back from the best
    iterate to CSR Newton with the ORIGINAL nonlinear target, not a looser one.
    """
    norm = lambda r: torch.linalg.vector_norm(r).item()
    def evaluate(x):
        problem.validate(x)
        r = problem.residual(x)
        if r.shape!=initial.shape or r.dtype!=initial.dtype or r.device!=initial.device:
            raise ValueError('midpoint residual must match the nodal unknown')
        if not torch.isfinite(r).all():
            raise FloatingPointError('nonfinite midpoint residual')
        return r.detach()
    x = initial.detach().clone()
    r = evaluate(x)
    initial_norm = norm(r)
    current_norm = initial_norm
    tolerance = max(options.atol,options.rtol*initial_norm)
    best_x,best_norm = x.clone(),initial_norm
    history = [dict(iteration=0,method='anderson',residual_norm=initial_norm)]
    xs,rs = [x.clone()],[r.clone()]
    completed = 0
    reason = 'acceleration iteration budget reached'
    for _ in range(acceleration.max_iterations):
        if current_norm <= tolerance:
            result = NewtonResult(x.clone(),True,current_norm,tolerance,completed,history)
            return result,dict(solver='anderson-newton',anderson_iterations=completed,
                newton_iterations=0,newton_fallback=False,fallback_reason=None)
        step = -r
        if len(rs)>1:
            dR = torch.stack([(b-a).reshape(-1) for a,b in zip(rs[:-1],rs[1:])],1)
            dX = torch.stack([(b-a).reshape(-1) for a,b in zip(xs[:-1],xs[1:])],1)
            scales = torch.linalg.vector_norm(dR,dim=0)
            valid = scales > 100*torch.finfo(r.dtype).eps*max(initial_norm,torch.finfo(r.dtype).tiny)
            if valid.any():
                A,B = dR[:,valid]/scales[valid],dX[:,valid]/scales[valid]
                gram = A.T@A
                ridge = max(acceleration.regularization,100*torch.finfo(r.dtype).eps)
                gamma = torch.linalg.solve(gram+ridge*torch.eye(gram.shape[0],device=x.device,dtype=x.dtype),A.T@r.reshape(-1))
                step = -r-((B-A)@gamma).reshape_as(r)
        trial = x+step
        try:
            candidate = evaluate(trial)
        except (ValueError,RuntimeError,FloatingPointError) as exc:
            reason = f'acceleration trial failed: {type(exc).__name__}: {exc}'
            break
        candidate_norm = norm(candidate)
        if candidate_norm > acceleration.growth_limit*max(best_norm,tolerance):
            reason = 'acceleration residual growth guard'
            break
        x,r = trial,candidate
        current_norm = candidate_norm
        completed += 1
        history.append(dict(iteration=completed,method='anderson',residual_norm=candidate_norm))
        if candidate_norm < best_norm:
            best_x,best_norm = x.clone(),candidate_norm
        xs.append(x.clone()); rs.append(r.clone())
        xs,rs = xs[-acceleration.history_size-1:],rs[-acceleration.history_size-1:]
    if current_norm <= tolerance:
        return NewtonResult(x.clone(),True,current_norm,tolerance,completed,history),dict(
            solver='anderson-newton',anderson_iterations=completed,newton_iterations=0,
            newton_fallback=False,fallback_reason=None)
    # Preserve the target derived from the ORIGINAL residual. Starting Newton
    # from a better iterate must not change relative-tolerance semantics.
    fallback_options = replace(options,rtol=0.,atol=tolerance)
    try:
        result = newton_solve(problem.residual,best_x,validate=problem.validate,
            linearization_factory=problem.linearization,preconditioner_factory=problem.preconditioner,
            options=fallback_options)
    except NonlinearFailure as exc:
        exc.result = replace(exc.result,iterations=completed+exc.result.iterations,
            history=history+[dict(h,iteration=completed+h['iteration'],method='newton') for h in exc.result.history])
        raise
    newton_iterations = result.iterations
    result = replace(result,iterations=completed+newton_iterations,
        history=history+[dict(h,iteration=completed+h['iteration'],method='newton') for h in result.history])
    return result,dict(solver='anderson-newton',anderson_iterations=completed,
        newton_iterations=newton_iterations,newton_fallback=True,fallback_reason=reason)
