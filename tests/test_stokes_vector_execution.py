"""Workspace reuse and fused vectors retain the actual saddle-point equations."""
from dataclasses import replace
import pytest
import torch
from afsi_torch.nonlinear import gmres,GMRESOptions,GMRESWorkspace
from afsi_torch.mac.cnab import MACCNABFlow,CNABOptions
from afsi_torch.mac.grid import MACGrid,zero_normal,gradient,divergence,velocity_laplacian

DEVICES=['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))]


@pytest.mark.parametrize('device',DEVICES)
def test_workspace_poison_restart_zero_rhs_and_owned_results(device):
    generator=torch.Generator(device=device).manual_seed(51)
    a=torch.randn((19,19),device=device,dtype=torch.float64,generator=generator)
    a=a.T@a+torch.eye(19,device=device,dtype=a.dtype)*15
    b=torch.randn(19,device=device,dtype=a.dtype,generator=generator)
    opt=GMRESOptions(rtol=1e-11,atol=1e-13,restart=5,max_iterations=120,check_policy='estimated')
    work=GMRESWorkspace();owned=[]
    for rhs in (b,b*.03,torch.zeros_like(b),b*-.7):
        expected,ie=gmres(lambda x:a@x,rhs,options=opt)
        v,z=work.vectors(rhs,5);v.fill_(float('nan'));z.fill_(float('nan'))
        actual,ia=gmres(lambda x:a@x,rhs,options=opt,workspace=work)
        torch.testing.assert_close(actual,expected,rtol=2e-10,atol=2e-12)
        assert ia['iterations']==ie['iterations'] and ia['residual_norm']<=ia['tolerance']
        for output,saved in owned:torch.testing.assert_close(output,saved,rtol=0,atol=0)
        owned.append((actual,actual.clone()))
    assert work.allocations==1


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('dt',[.001,.04])
def test_fused_stokes_vector_action_and_final_true_constraints(device,dt):
    from afsi_torch.mac.stokes_execution import StokesExecution
    grid=MACGrid((4,6,8),(1.,1.3,1.7))
    options=CNABOptions(stokes_solver='fgmres',max_stokes_iterations=100)
    reference=MACCNABFlow(grid,dt=dt,rho=1.3,mu=.7,device=device,
        cnab_options=replace(options,stokes_vector_backend='reference'))
    fast=MACCNABFlow(grid,dt=dt,rho=1.3,mu=.7,device=device,
        cnab_options=options,execution_backend='fused')
    generator=torch.Generator(device=device).manual_seed(91)
    u=zero_normal(tuple(torch.randn(v.shape,device=device,dtype=v.dtype,generator=generator) for v in grid.zeros(device=device)))
    q=torch.randn(grid.shape,device=device,dtype=torch.float64,generator=generator)
    vectors=StokesExecution(fast);h=min(grid.spacing)
    gp=gradient(q,grid.spacing)
    want=vectors.pack(zero_normal(tuple(v-fast.alpha*velocity_laplacian(v,c,grid.spacing)+h*g for c,(v,g) in enumerate(zip(u,gp)))),-h*divergence(u,grid.spacing))
    torch.testing.assert_close(vectors.action(vectors.pack(u,q),fast.alpha,h),want,atol=3e-13,rtol=3e-13)
    b=tuple(v*1e-3 for v in u)
    a=reference.stokes(b);actual=fast.stokes(b)
    for v,w in zip(actual.velocity,a.velocity):torch.testing.assert_close(v,w,atol=2e-11,rtol=2e-8)
    torch.testing.assert_close(actual.pressure,a.pressure,atol=3e-8,rtol=3e-7)
    stokes=actual.diagnostics['stokes']
    assert stokes['momentum_residual']<=stokes['momentum_tolerance']
    assert stokes['divergence_norm']<=stokes['divergence_tolerance']
    saved=actual.velocity[0].clone()
    fast.stokes(tuple(-v for v in b))
    torch.testing.assert_close(actual.velocity[0],saved,atol=0,rtol=0)
