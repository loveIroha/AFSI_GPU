"""Forward Newton--GMRES on the tensor device, with a true-residual check.

No assembled global Jacobian, CPU linear solve or differentiation through the
solver. JVPs differentiate the complete residual, including follower loads.
Dirichlet data are absolute unknown values, not displacement increments.
"""
from dataclasses import dataclass, field
from math import isfinite
import torch


@dataclass(frozen=True)
class GMRESOptions:
    rtol: float = 1e-3
    atol: float = 1e-12
    restart: int = 50
    max_iterations: int = 1000
    check_every: int = 5

    def __post_init__(self):
        if not all(isfinite(t) and t >= 0 for t in (self.rtol,self.atol)) or self.rtol+self.atol == 0:
            raise ValueError('invalid linear tolerances')
        for n in (self.restart,self.max_iterations,self.check_every):
            if not isinstance(n,int) or isinstance(n,bool) or n < 1:
                raise ValueError('positive integer GMRES controls required')


@dataclass(frozen=True)
class NewtonOptions:
    rtol: float = 1e-8
    atol: float = 1e-8
    max_iterations: int = 25
    max_backtracks: int = 20
    armijo: float = 1e-4
    linear: GMRESOptions = field(default_factory=GMRESOptions)

    def __post_init__(self):
        if not all(isfinite(t) and t >= 0 for t in (self.rtol,self.atol)) or self.rtol+self.atol == 0:
            raise ValueError('invalid nonlinear tolerances')
        for n in (self.max_iterations,self.max_backtracks):
            if not isinstance(n,int) or isinstance(n,bool) or n < 1:
                raise ValueError('positive integer Newton controls required')
        if not isfinite(self.armijo) or not 0 < self.armijo < 1:
            raise ValueError('Armijo constant must be between zero and one')
        if not isinstance(self.linear,GMRESOptions) or self.linear.rtol >= 1-self.armijo:
            raise ValueError('linear relative tolerance must ensure a descent direction')


@dataclass(frozen=True)
class NewtonResult:
    x: torch.Tensor
    converged: bool
    residual_norm: float
    tolerance: float
    iterations: int
    history: list


class NonlinearFailure(RuntimeError):
    """Retains the last accepted iterate; never mutates the caller's input."""
    def __init__(self,message,result):
        super().__init__(message)
        self.result = result


def _check(value,reference,name):
    if (not isinstance(value,torch.Tensor) or value.shape != reference.shape or
        value.dtype != reference.dtype or value.device != reference.device):
        raise ValueError(f'{name} must match unknown shape, dtype and device')
    if not torch.isfinite(value).all():
        raise FloatingPointError(f'nonfinite {name}')


@torch.no_grad()
def gmres(action,rhs,*,precondition=None,options=None):
    """Restarted right-preconditioned GMRES; residuals use the original A.

    Two orthogonalization passes. The small Hessenberg least-squares problem
    uses QR/triangular solve on the same device. Breakdown is never success
    unless the actual residual meets tolerance. Starts from zero.
    """
    opt = GMRESOptions() if options is None else options
    if rhs.dtype not in (torch.float32,torch.float64) or not rhs.numel():
        raise ValueError('nonempty floating rhs required')
    _check(rhs,rhs,'rhs')
    norm = lambda x: torch.linalg.vector_norm(x).item()
    tol = max(opt.atol,opt.rtol*norm(rhs))
    shape = rhs.shape
    def apply(v):
        out = action(v.reshape(shape))
        _check(out,rhs,'operator action')
        return out.reshape(-1)
    def solve_pre(v):
        out = v.reshape(shape) if precondition is None else precondition(v.reshape(shape))
        _check(out,rhs,'preconditioner action')
        return out.reshape(-1)
    x = torch.zeros_like(rhs).reshape(-1)
    b = rhs.reshape(-1)
    residual = b-apply(x)
    count = 0
    while norm(residual) > tol and count < opt.max_iterations:
        beta = torch.linalg.vector_norm(residual)
        width = min(opt.restart,opt.max_iterations-count,b.numel())
        V = b.new_zeros((b.numel(),width+1))
        Z = b.new_zeros((b.numel(),width))
        H = b.new_zeros((width+1,width))
        V[:,0] = residual/beta
        base = x.clone()
        for j in range(width):
            Z[:,j] = solve_pre(V[:,j])
            w = apply(Z[:,j])
            original_norm = norm(w)
            for _ in range(2):
                coefficients = V[:,:j+1].T@w
                H[:j+1,j] += coefficients
                w = w-V[:,:j+1]@coefficients
            H[j+1,j] = torch.linalg.vector_norm(w)
            breakdown = H[j+1,j].item() <= 100*torch.finfo(b.dtype).eps*max(original_norm,torch.finfo(b.dtype).tiny)
            if not breakdown:
                V[:,j+1] = w/H[j+1,j]
            count += 1
            if breakdown or (j+1)%opt.check_every == 0 or j+1 == width:
                Q,R = torch.linalg.qr(H[:j+2,:j+1],mode='reduced')
                if (R.diagonal().abs() <= torch.finfo(b.dtype).eps*R.abs().max()).any():
                    raise RuntimeError('GMRES singular Hessenberg system')
                target = b.new_zeros(j+2)
                target[0] = beta
                y = torch.linalg.solve_triangular(R,(Q.T@target)[:,None],upper=True)[:,0]
                x = base+Z[:,:j+1]@y
                residual = b-apply(x)
                if norm(residual) <= tol:
                    return x.reshape(shape),dict(iterations=count,residual_norm=norm(residual),tolerance=tol)
                if breakdown:
                    raise RuntimeError('GMRES breakdown before true-residual convergence')
    if norm(residual) > tol:
        raise RuntimeError(f'GMRES failed: true residual {norm(residual):.6g} > {tol:.6g} after {count} iterations')
    return x.reshape(shape),dict(iterations=count,residual_norm=norm(residual),tolerance=tol)


@torch.no_grad()
def newton(residual,x0,*,validate,fixed=None,values=None,preconditioner_factory=None,options=None):
    """Solve free residual=0 with exact JVP and Armijo residual-norm backtracking.

    A preconditioner is prepared once per Newton iteration and applied on the
    right in GMRES. Geometry is checked before evaluating any trial residual.
    Singular tangents, nonconvergence and failed line searches raise with the
    last accepted state. Initial invalid input raises before any iteration.
    """
    opt = NewtonOptions() if options is None else options
    if x0.dtype not in (torch.float32,torch.float64) or not x0.numel():
        raise ValueError('nonempty floating unknown required')
    _check(x0,x0,'initial unknown')
    fixed = torch.zeros_like(x0,dtype=torch.bool) if fixed is None else fixed
    if fixed.shape != x0.shape or fixed.dtype != torch.bool or fixed.device != x0.device:
        raise ValueError('fixed mask must match unknown shape/device')
    values = x0 if values is None else values
    _check(values,x0,'Dirichlet values')
    x = torch.where(fixed,values,x0).detach().clone()
    project = lambda y: y.masked_fill(fixed,0)
    def evaluate(y):
        validate(y)
        r = residual(y)
        _check(r,x0,'nonlinear residual')
        return project(r).detach()
    r = evaluate(x)
    norm = lambda a: torch.linalg.vector_norm(a).item()
    initial = norm(r)
    tol = max(opt.atol,opt.rtol*initial)
    history = [dict(iteration=0,residual_norm=initial)]
    def result(ok):
        return NewtonResult(x.detach().clone(),ok,norm(r),tol,len(history)-1,list(history))
    for iteration in range(1,opt.max_iterations+1):
        if norm(r) <= tol:
            return result(True)
        def action(v):
            return project(torch.func.jvp(residual,(x,),(project(v),))[1]).detach()
        try:
            inverse = None if preconditioner_factory is None else preconditioner_factory(x)
            precondition = None if inverse is None else lambda v: project(inverse(project(v)))
            step,linear_info = gmres(action,-r,precondition=precondition,options=opt.linear)
        except (RuntimeError,FloatingPointError) as exc:
            raise NonlinearFailure(f'Newton linear solve failed: {exc}',result(False)) from exc
        old_norm = norm(r)
        accepted = False
        for backtrack in range(opt.max_backtracks+1):
            alpha = 2.**(-backtrack)
            trial = torch.where(fixed,values,x+alpha*step)
            try:
                candidate = evaluate(trial)
            except (ValueError,FloatingPointError):
                continue
            if norm(candidate) <= (1-opt.armijo*alpha)*old_norm:
                x,r = trial.detach(),candidate
                accepted = True
                history.append(dict(iteration=iteration,residual_norm=norm(r),alpha=alpha,
                                    backtracks=backtrack,linear=linear_info))
                break
        if not accepted:
            raise NonlinearFailure('Newton backtracking failed',result(False))
    if norm(r) <= tol:
        return result(True)
    raise NonlinearFailure('Newton iteration limit reached',result(False))
