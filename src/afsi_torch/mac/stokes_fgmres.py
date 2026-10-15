"""CN saddle-point FGMRES with a pressure-free projection preconditioner.

FGMRES stores each preconditioned Arnoldi vector separately (the project's
right-preconditioned gmres does this), allowing inexact multigrid subsolves.
The physical momentum AND continuity residuals decide acceptance.
"""
import torch
from math import isfinite
from .flow import MACFlowResult
from .grid import zero_normal,gradient,divergence
from ..nonlinear import gmres,GMRESOptions


@torch.no_grad()
def solve(flow,b,initial=None):
    flow.stokes_calls+=1
    b=zero_normal(b)
    template=flow.pressure_solver.diagonals[0]
    p=torch.zeros_like(template) if initial is None else initial.clone()
    if p.shape!=template.shape or p.device!=template.device or p.dtype!=template.dtype or not torch.isfinite(p).all():
        raise ValueError('invalid initial CN Stokes pressure')
    p-=p.mean()
    h=min(flow.grid.spacing)
    scale=flow.dt/flow.rho
    sizes=tuple(v.numel() for v in b)
    shapes=tuple(v.shape for v in b)
    execution=None
    if flow.execution_backend=='fused' and flow.cnab_options.stokes_vector_backend=='workspace':
        if not hasattr(flow,'_fgmres_execution'):
            from .stokes_execution import StokesExecution
            flow._fgmres_execution=StokesExecution(flow)
        execution=flow._fgmres_execution
    def pack(u,q):
        return torch.cat(tuple(v.reshape(-1) for v in u)+(q.reshape(-1),))
    def unpack(v):
        parts=v.split((*sizes,template.numel()))
        return tuple(z.reshape(s) for z,s in zip(parts,shapes)),parts[-1].reshape(template.shape)
    def action(v):
        if execution is not None:
            return execution.action(v,flow.alpha,h)
        u,q=unpack(v)
        gp=gradient(q,flow.grid.spacing)
        hu=tuple(a-flow.alpha*l+h*g for a,l,g in zip(u,flow._lap(u),gp))
        return pack(zero_normal(hu),-h*divergence(u,flow.grid.spacing))
    poisson_solves=cycles=0
    def precondition(v):
        nonlocal poisson_solves,cycles
        r,s=unpack(v)
        tentative=flow._solve_helmholtz(r)
        if execution is None:
            rhs=-s/h-divergence(tentative,flow.grid.spacing)
            rhs=rhs-rhs.mean()
        else:
            rhs=execution.precondition_rhs(tentative,s,h)
        phi,info=flow.pressure_solver.solve(rhs,rtol=flow.cnab_options.stokes_preconditioner_rtol)
        poisson_solves+=1;cycles+=info['cycles']
        if execution is not None:
            return execution.precondition_result(tentative,phi,rhs,flow.alpha,h)
        grad=gradient(phi,flow.grid.spacing)
        u=tuple(a-g for a,g in zip(tentative,grad))
        q=(phi+flow.alpha*rhs)/h
        q=q-q.mean()
        return pack(u,q)
    norm_b=torch.sqrt(sum(v.square().sum() for v in b)).item()
    if not isfinite(norm_b):
        raise ValueError('nonfinite CN Stokes RHS')
    opt=flow.pressure_solver.options
    momentum_tol=max(opt.atol,opt.rtol*norm_b)
    # Same physical continuity scaling as the existing exact-Schur solver.
    z=flow._solve_helmholtz(b)
    div_rhs=-divergence(z,flow.grid.spacing)/scale
    div_tol=scale*max(opt.atol,opt.rtol*torch.linalg.vector_norm(div_rhs).item())
    floor=100*torch.finfo(p.dtype).eps*max(norm_b,torch.finfo(p.dtype).tiny)*sum(1/d for d in flow.grid.spacing)
    div_tol=max(div_tol,floor)
    initial_u=flow._solve_helmholtz(b,p)
    guess=pack(initial_u,scale/h*p)
    target=pack(b,torch.zeros_like(p))
    rhs=target-action(guess)
    tolerance=.5*min(momentum_tol,h*div_tol)
    controls=GMRESOptions(rtol=0.,atol=tolerance,restart=flow.cnab_options.stokes_restart,
        max_iterations=flow.cnab_options.max_stokes_iterations,check_policy='estimated',true_check_interval=5)
    correction,info=gmres(action,rhs,precondition=precondition,options=controls,
        workspace=None if execution is None else execution.krylov)
    u,q=unpack(guess+correction)
    p=h/scale*(q-q.mean())
    momentum,div=flow._metrics(u,p,b).tolist()
    if momentum>momentum_tol or div>div_tol:
        raise RuntimeError(f'CN FGMRES true residual rejected: momentum {momentum}/{momentum_tol}, divergence {div}/{div_tol}')
    return MACFlowResult(u,p,dict(pressure=dict(cycles=cycles,backend=flow.pressure_solver.backend,
        poisson_solves=poisson_solves,schur_iterations=info['iterations'],residual_norm=div/scale,
        tolerance=div_tol/scale,meaning='CN half-time pressure; coupled saddle-point solve'),
        stokes=dict(momentum_residual=momentum,momentum_tolerance=momentum_tol,divergence_norm=div,
        divergence_tolerance=div_tol,solver='fgmres',preconditioner='pressure-free projection with geometric MG',
        iterations=info['iterations'],operator_actions=info['operator_actions'],
        preconditioner_rtol=flow.cnab_options.stokes_preconditioner_rtol,
        helmholtz_iterations=flow.helmholtz_iterations,helmholtz_backend=flow.helmholtz_backend)))
