"""Buffered CSR Jacobi-PCG for the unconstrained, constant IB mass matrix.

Same recurrence, restart/check schedule and true-residual acceptance as pcg().
No mass lumping, precision change, fixed-iteration acceptance or new preconditioner.
"""
from math import isfinite
import torch
from ..fluid.solvers import SolveInfo, SolverOptions
from .execution import tensor_kernel


def _advance(x,r,d,Ad,rz,active,broken,tol):
    curvature = (d*Ad).sum()
    valid = torch.isfinite(curvature) & (curvature>0) & torch.isfinite(rz) & (rz>0)
    broken = broken | (active & ~valid)
    working = active & valid & ~broken
    alpha = torch.where(working,rz,0)/torch.where(working,curvature,1)
    x.add_(alpha*d)
    r.sub_(alpha*Ad)
    rn = torch.linalg.vector_norm(r)
    broken = broken | ~torch.isfinite(rn)
    active = working & (rn>tol)
    return active, broken, rn


def _direction(r,diag,d,rz,active):
    z = r/diag
    next_rz = (r*z).sum()
    beta = next_rz/torch.where(active,rz,1)
    d.copy_(torch.where(active,z+beta*d,0))
    return next_rz


def _restart(r,diag,d):
    d.copy_(r/diag)
    return (r*d).sum()


class MassSolver:
    def __init__(self,mass,diagonal,options=None):
        self.mass, self.diagonal = mass, diagonal
        self.options = options or SolverOptions()
        if not torch.isfinite(diagonal).all() or (diagonal<=0).any():
            raise ValueError('Jacobi diagonal must be finite and positive')
        self.x,self.r,self.d,self.Ad = [torch.empty_like(diagonal) for _ in range(4)]
        self.tol = diagonal.new_empty(())
        self.advance = tensor_kernel(_advance,diagonal.device)
        self.direction = tensor_kernel(_direction,diagonal.device)
        self.restart = tensor_kernel(_restart,diagonal.device)

    def action(self,value,out):
        torch.mm(self.mass,value,out=out)

    @torch.no_grad()
    def solve(self,rhs,initial=None):
        opt = self.options
        for v in (rhs,) if initial is None else (rhs,initial):
            if v.shape!=self.x.shape or v.dtype!=self.x.dtype or v.device!=self.x.device:
                raise ValueError('mass solve shape, dtype or device mismatch')
        finite = torch.isfinite(rhs).all()
        if initial is not None:
            finite = finite & torch.isfinite(initial).all()
        if not finite:
            raise ValueError('finite mass RHS and initial guess required')
        rhs_norm = torch.linalg.vector_norm(rhs).item()
        tolerance = max(opt.atol,opt.rtol*rhs_norm)
        self.tol.fill_(tolerance)
        if initial is None:
            self.x.zero_()
        else:
            self.x.copy_(initial)
        def true_residual():
            self.action(self.x,self.Ad)
            torch.sub(rhs,self.Ad,out=self.r)
            return torch.linalg.vector_norm(self.r).item()
        true_norm = true_residual()
        if true_norm<=tolerance:
            return self.x.clone(),SolveInfo(0,true_norm,rhs_norm,tolerance)
        rz = self.restart(self.r,self.diagonal,self.d)
        active = torch.ones((),device=rhs.device,dtype=torch.bool)
        broken = torch.zeros_like(active)
        for iteration in range(1,opt.max_iterations+1):
            self.action(self.d,self.Ad)
            active,broken,rn = self.advance(self.x,self.r,self.d,self.Ad,rz,active,broken,self.tol)
            restart = iteration%opt.recompute_every==0
            check = restart or iteration%opt.check_every==0 or iteration==opt.max_iterations
            converged = False
            if check:
                failed, value = torch.stack((broken.to(rhs.dtype),rn)).tolist()
                if failed:
                    raise RuntimeError('PCG breakdown: operator/preconditioner must be positive definite and finite')
                converged = value<=tolerance
            if restart or converged or iteration==opt.max_iterations:
                true_norm = true_residual()
                if not isfinite(true_norm):
                    raise RuntimeError('nonfinite true PCG residual')
                if true_norm<=tolerance:
                    return self.x.clone(),SolveInfo(iteration,true_norm,rhs_norm,tolerance)
                restart = True
                active.fill_(True)
            rz = (self.restart(self.r,self.diagonal,self.d) if restart else
                  self.direction(self.r,self.diagonal,self.d,rz,active))
        raise RuntimeError(f'PCG did not converge: {true_norm:g} > {tolerance:g}')
