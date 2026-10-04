"""Same-residual acceleration, noncontractive history and safe Newton fallback."""
from dataclasses import replace
import pytest
import torch
from afsi_torch.mac.midpoint_solver import AndersonOptions,accelerated_midpoint
from afsi_torch.nonlinear import NewtonOptions,GMRESOptions,newton,NonlinearFailure


class Problem:
    def __init__(self,gain,limit=float('inf')):
        self.target = torch.full((2,3),.01,dtype=torch.float64)
        self.matrix = (gain*torch.eye(6,dtype=self.target.dtype)).to_sparse_csr()
        self.gain,self.limit,self.assemblies = gain,limit,0
    def validate(self,x):
        if not torch.isfinite(x).all() or x.abs().max()>self.limit:
            raise ValueError('invalid trial')
    def residual(self,x):
        return torch.sparse.mm(self.matrix,(x-self.target).reshape(-1,1)).reshape_as(x)
    def linearization(self,x):
        self.assemblies+=1
        return lambda v:torch.sparse.mm(self.matrix,v.reshape(-1,1)).reshape_as(v)
    def preconditioner(self,x):
        return lambda v:v/self.gain


OPTIONS = NewtonOptions(rtol=1e-9,atol=1e-12,
    linear=GMRESOptions(rtol=1e-3,atol=1e-13,restart=6,check_every=1,max_iterations=12))


def test_anderson_solves_noncontractive_map_without_tangent():
    # Ordinary x<-x-r has amplification -1.5 and diverges; secant acceleration
    # solves this residual, rather than accepting its first overshooting probe.
    problem = Problem(2.5)
    result,info = accelerated_midpoint(problem,torch.zeros_like(problem.target),OPTIONS,
                                      AndersonOptions(),newton_solve=newton)
    torch.testing.assert_close(result.x,problem.target,atol=1e-11,rtol=1e-10)
    assert result.residual_norm<=result.tolerance
    assert info['anderson_iterations']>=2 and not info['newton_fallback']
    assert problem.assemblies==0


@pytest.mark.parametrize('gain,limit',[(100.,float('inf')),(10.,.05)])
def test_growth_or_invalid_probe_falls_back_to_csr_with_original_tolerance(gain,limit):
    problem = Problem(gain,limit)
    initial = torch.zeros_like(problem.target)
    initial_copy = initial.clone()
    target = max(OPTIONS.atol,OPTIONS.rtol*torch.linalg.vector_norm(problem.residual(initial)).item())
    result,info = accelerated_midpoint(problem,initial,OPTIONS,AndersonOptions(),newton_solve=newton)
    assert info['newton_fallback'] and info['newton_iterations']>0 and problem.assemblies>0
    assert result.tolerance==target and result.residual_norm<=target
    torch.testing.assert_close(initial,initial_copy,atol=0,rtol=0)
    torch.testing.assert_close(result.x,problem.target,atol=1e-10,rtol=1e-10)


def test_nonconvergence_cannot_be_reported_as_an_accepted_result():
    problem = Problem(100.)
    def failing(*args,**kwargs):
        result = newton(*args,**kwargs)
        raise NonlinearFailure('injected fallback failure',replace(result,converged=False))
    with pytest.raises(NonlinearFailure,match='injected fallback') as failure:
        accelerated_midpoint(problem,torch.zeros_like(problem.target),OPTIONS,
                             AndersonOptions(),newton_solve=failing)
    assert not failure.value.result.converged


@pytest.mark.parametrize('kwargs',[{'max_iterations':0},{'history_size':True},
    {'growth_limit':.5},{'regularization':0.}])
def test_invalid_acceleration_options(kwargs):
    with pytest.raises(ValueError): AndersonOptions(**kwargs)
