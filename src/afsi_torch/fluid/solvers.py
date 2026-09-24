"""GPU-resident Jacobi-PCG with symmetric Dirichlet elimination.

Only scalar convergence diagnostics synchronize to the host. No SciPy/PETSc,
CPU matrix solve or dense global matrix is used. Iterative solves are currently
forward-only (no autograd through stopping/restarts); operator JVPs remain valid.
"""
from dataclasses import dataclass
from math import isfinite
import torch


@dataclass(frozen=True)
class SolverOptions:
    rtol: float = 1e-10
    atol: float = 1e-12
    max_iterations: int = 1000
    recompute_every: int = 40

    def __post_init__(self):
        if (not isfinite(self.rtol) or not isfinite(self.atol) or self.rtol < 0 or
                self.atol < 0 or self.rtol+self.atol == 0):
            raise ValueError('finite nonnegative tolerances, at least one positive, required')
        for name in ('max_iterations', 'recompute_every'):
            n = getattr(self, name)
            if not isinstance(n, int) or isinstance(n, bool) or n < 1:
                raise ValueError(f'{name} must be a positive integer')


@dataclass(frozen=True)
class SolveInfo:
    iterations: int
    residual_norm: float
    rhs_norm: float
    tolerance: float


@torch.no_grad()
def pcg(action, rhs, diagonal, *, fixed=None, values=None, initial=None, options=None):
    """Solve A x=b on free DOFs, x_fixed=values, requiring SPD on free DOFs.

    The returned residual is the TRUE free residual b-Ax, not just the CG
    recurrence. Failure, nonfinite data or loss of positive curvature raises.
    Tolerance is max(atol,rtol*||P(b-A g)||), where g is the boundary lift.
    """
    options = SolverOptions() if options is None else options
    if rhs.dtype not in (torch.float32, torch.float64) or not rhs.numel():
        raise ValueError('rhs must be nonempty floating data')
    def check(x, name):
        if x.shape != rhs.shape or x.dtype != rhs.dtype or x.device != rhs.device or not torch.isfinite(x).all():
            raise ValueError(f'{name} must be finite with rhs shape, dtype and device')
    check(rhs, 'rhs')
    check(diagonal, 'diagonal')
    if (diagonal <= 0).any():
        raise ValueError('Jacobi diagonal must be positive')
    fixed = torch.zeros_like(rhs, dtype=torch.bool) if fixed is None else fixed
    if fixed.shape != rhs.shape or fixed.dtype != torch.bool or fixed.device != rhs.device:
        raise ValueError('fixed mask must match rhs shape/device and have bool dtype')
    values = torch.zeros_like(rhs) if values is None else values
    check(values, 'boundary values')
    project = lambda x: x.masked_fill(fixed, 0)
    lift = torch.where(fixed, values, 0)
    reduced_rhs = project(rhs-action(lift))
    if not torch.isfinite(reduced_rhs).all():
        raise RuntimeError('nonfinite operator action')
    norm = lambda x: torch.linalg.vector_norm(x).item()
    rhs_norm = norm(reduced_rhs)
    tol = max(options.atol, options.rtol*rhs_norm)
    if initial is not None:
        check(initial, 'initial guess')
    correction = torch.zeros_like(rhs) if initial is None else project(initial-lift)
    residual = project(rhs-action(lift+correction))
    if norm(residual) <= tol:
        return lift+correction, SolveInfo(0, norm(residual), rhs_norm, tol)
    z = residual/diagonal
    direction = z.clone()
    rz = (residual*z).sum()
    for iteration in range(1, options.max_iterations+1):
        Ad = project(action(direction))
        curvature = (direction*Ad).sum()
        if not torch.isfinite(curvature) or curvature <= 0 or not torch.isfinite(rz) or rz <= 0:
            raise RuntimeError('PCG breakdown: operator/preconditioner must be positive definite')
        alpha = rz/curvature
        correction = correction+alpha*direction
        residual = residual-alpha*Ad
        recurrence_norm = norm(residual)
        if not isfinite(recurrence_norm):
            raise RuntimeError('nonfinite PCG residual')
        restart = iteration % options.recompute_every == 0 or recurrence_norm <= tol
        if restart or iteration == options.max_iterations:
            residual = project(rhs-action(lift+correction))
            true_norm = norm(residual)
            if not isfinite(true_norm):
                raise RuntimeError('nonfinite true PCG residual')
            if true_norm <= tol:
                return lift+correction, SolveInfo(iteration, true_norm, rhs_norm, tol)
            restart = True
        z = residual/diagonal
        next_rz = (residual*z).sum()
        direction = z if restart else z+(next_rz/rz)*direction
        rz = next_rz
    raise RuntimeError(f'PCG did not converge after {options.max_iterations} iterations: '
                       f'true residual {true_norm:.6e} > tolerance {tol:.6e}')


def operator_diagonals(op):
    """Exact assembled diagonals, computed once from local quadrature tables."""
    result = {}
    for pressure, N, dN in ((False, op.N, op.dN), (True, op.Q, op.dQ)):
        local_mass = torch.einsum('qa,qa,q->a', N, N, op.weights)
        local_stiffness = torch.einsum('qaj,qaj,q->a', dN, dN, op.weights)
        for name, local in (('mass', local_mass), ('stiffness', local_stiffness)):
            assembled = op._scatter(local.expand(len(op.mesh.velocity_cells), -1), pressure)
            result[('pressure_' if pressure else 'velocity_')+name] = (
                assembled if pressure else assembled[:, None].expand(-1, 3))
    return result
