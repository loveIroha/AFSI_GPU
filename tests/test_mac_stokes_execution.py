"""Cell-tiled Stokes residuals retain wall equations, norms and owned states."""
from dataclasses import replace
import os
import pytest
import torch
from afsi_torch.mac.grid import MACGrid,zero_normal,divergence
from afsi_torch.mac.cnab import MACCNABFlow
from afsi_torch.mac.stokes_workspace import (StokesWorkspace,_reduce_start,_reduce_norms,
    _pressure_input_reduce,_center_pressure,_pressure_packet)
from test_real_lv import real_case,DEVICES
from test_mac_shared_pressure import warm_config
from test_mac_implicit import settings

INTERPRETER=os.environ.get('TRITON_INTERPRET')=='1'
KERNEL_DEVICES=[pytest.param('cpu',marks=pytest.mark.skipif(not INTERPRETER,reason='requires Triton interpreter')),
    pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available() or INTERPRETER,reason='native CUDA unavailable'))]


@pytest.mark.parametrize('device',KERNEL_DEVICES)
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_actual_pressure_start_reductions_match_original_centering(device,dtype):
    pytest.importorskip('triton')
    from afsi_torch.mac._triton_stokes import pressure_input_partial,pressure_norm_partial
    from afsi_torch.mac.multigrid import GeometricMultigrid
    grid=MACGrid((5,7,9),(1.3,2.7,3.1))
    # Only the fine stencil is used, not a multigrid hierarchy for this odd box.
    rhs=torch.randn(grid.shape,device=device,dtype=dtype)+.15
    p=torch.randn_like(rhs)-.27
    b,q=rhs.clone(),p.clone()
    sample=p.new_empty(((p.numel()+255)//256,5))
    norms=p.new_empty((len(sample),2))
    pressure_input_partial(rhs,p,sample)
    inputs=_pressure_input_reduce(sample,p.numel())
    _center_pressure(rhs,p,inputs)
    pressure_norm_partial(rhs,p,norms,p.new_tensor([h*h for h in grid.spacing]))
    actual=_pressure_packet(inputs,_reduce_norms(norms))
    class Fine:
        spacings=[grid.spacing]
    expected=GeometricMultigrid._start_metrics(Fine(),b,q)
    tol=dict(rtol=3e-6,atol=2e-4) if dtype==torch.float32 else dict(rtol=4e-13,atol=2e-11)
    torch.testing.assert_close(actual,expected,**tol)
    torch.testing.assert_close(rhs,b,**tol);torch.testing.assert_close(p,q,**tol)
    for invalid in (torch.nan,torch.inf,-torch.inf):
        bad=p.clone();bad[-1,-1,-1]=invalid
        pressure_input_partial(rhs,bad,sample)
        assert _pressure_input_reduce(sample,p.numel())[4].item()==0


def fields(grid,device,dtype=torch.float64):
    rng=torch.Generator(device=device).manual_seed(1711)
    values=lambda:zero_normal(tuple(torch.randn(v.shape,device=device,dtype=dtype,generator=rng)
                                   for v in grid.zeros(device=device,dtype=dtype)))
    return values(),values(),torch.randn(grid.shape,device=device,dtype=dtype,generator=rng)


@pytest.mark.parametrize('device',KERNEL_DEVICES)
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_actual_cell_tiled_residuals_bounds_padding_and_coefficients(device,dtype):
    pytest.importorskip('triton')
    from afsi_torch.mac._triton_stokes import start_partial,residual_partial
    grid=MACGrid((5,7,9),(1.3,2.7,3.1))
    flow=MACCNABFlow(grid,dt=.00137,rho=1.7,mu=.81,device=device,dtype=dtype)
    u,b,p=fields(grid,device,dtype)
    workspace=StokesWorkspace(flow)
    tolerance=dict(rtol=3e-6,atol=3e-5) if dtype==torch.float32 else dict(rtol=3e-13,atol=2e-11)
    start_partial(b,p,workspace.start_partial,grid.shape)
    expected=torch.stack((torch.sqrt(sum(v.square().sum() for v in b)),p.mean(),p.new_tensor(1.)))
    torch.testing.assert_close(_reduce_start(workspace.start_partial,p.numel()),expected,**tolerance)
    for dt in (.00137,.000685,.00137):
        flow.set_time_step(dt);workspace.update_coefficients()
        residual_partial(u,p,b,workspace.residual,workspace.partial,grid.shape,workspace.coefficients)
        torch.testing.assert_close(_reduce_norms(workspace.partial),flow._stokes_metrics(u,p,b),**tolerance)
        torch.testing.assert_close(workspace.residual,-flow.rho/dt*divergence(u,grid.spacing),**tolerance)
    for invalid in (torch.nan,torch.inf,-torch.inf):
        bad=p.clone();bad[-1,-1,-1]=invalid
        start_partial(b,bad,workspace.start_partial,grid.shape)
        assert _reduce_start(workspace.start_partial,p.numel())[2].item()==0
    bad=b[0].clone();bad[1,1,1]=torch.nan
    residual_partial(u,p,(bad,*b[1:]),workspace.residual,workspace.partial,grid.shape,workspace.coefficients)
    assert not torch.isfinite(_reduce_norms(workspace.partial)[0])


@pytest.mark.parametrize('device',DEVICES)
def test_stokes_true_residual_ownership_dt_and_pressure_validation(device):
    grid=MACGrid((4,)*3,(1.,1.2,1.7))
    reference=MACCNABFlow(grid,dt=.003,device=device)
    candidate=MACCNABFlow(grid,dt=.003,device=device)
    candidate.set_stokes_backend('workspace')
    _,b,p=fields(grid,device)
    saved=tuple(v.clone() for v in (*b,p));retained=None
    for dt in (.003,.0015,.003):
        reference.set_time_step(dt);candidate.set_time_step(dt)
        a,e=candidate.stokes(b,p),reference.stokes(b,p)
        for x,y in zip(a.velocity,e.velocity):torch.testing.assert_close(x,y,rtol=3e-8,atol=2e-10)
        torch.testing.assert_close(a.pressure,e.pressure,rtol=3e-8,atol=3e-9)
        actual=candidate._stokes_metrics(a.velocity,a.pressure,zero_normal(b))
        assert actual[0].item()<=a.diagnostics['stokes']['momentum_tolerance']*1.001
        assert actual[1].item()<=a.diagnostics['stokes']['divergence_tolerance']*1.001
        if retained:
            for x,y in zip((*retained[0].velocity,retained[0].pressure),retained[1]):
                torch.testing.assert_close(x,y,rtol=0,atol=0)
        retained=a,tuple(v.clone() for v in (*a.velocity,a.pressure))
    for x,y in zip((*b,p),saved):torch.testing.assert_close(x,y,rtol=0,atol=0)
    for value in (torch.nan,torch.inf,-torch.inf):
        bad=p.clone();bad[1,1,1]=value
        with pytest.raises(ValueError,match='initial CN Stokes pressure'):candidate.stokes(b,bad)
    with pytest.raises(ValueError,match='Stokes execution backend'):candidate.set_stokes_backend('bad')


@pytest.mark.parametrize('device',DEVICES)
def test_noncontiguous_stokes_inputs_keep_reference_semantics(device):
    grid=MACGrid((4,)*3,(1.,1.2,1.4))
    a=MACCNABFlow(grid,dt=.002,device=device);a.set_stokes_backend('workspace')
    b=MACCNABFlow(grid,dt=.002,device=device)
    _,rhs,p=fields(grid,device)
    def strided(x):
        target=x.new_empty((*x.shape[:-1],2*x.shape[-1]));target[...,::2]=x
        return target[...,::2]
    actual,expected=a.stokes(tuple(strided(v) for v in rhs),strided(p)),b.stokes(rhs,p)
    for x,y in zip(actual.velocity,expected.velocity):torch.testing.assert_close(x,y,rtol=2e-8,atol=1e-10)


@pytest.mark.parametrize('device',DEVICES)
def test_active_coupled_steps_preserve_csr_residual(real_case,device):
    from afsi_torch.real_lv import imported_model
    from afsi_torch.mac.execution import build_driver
    cfg=warm_config(real_case)
    cfg=replace(cfg,execution=replace(cfg.execution,execution_backend='fused',coupling_backend='optimized'),
        interaction_quadrature=replace(cfg.interaction_quadrature,transfer_backend='fused'))
    model=imported_model(cfg,device)
    a,b=build_driver(model,settings(cfg),device),build_driver(model,settings(cfg),device)
    a.flow.set_stokes_backend('reference')
    initial=a.initialize(model.mesh.X)
    initial=replace(initial,step=6000,time=.6,force_time=.6,force=model.force(initial.x,.6),
        previous_advection=a.flow.advection(initial.velocity),previous_dt=cfg.time.dt)
    left,right=initial,initial
    for _ in range(3):
        left,_=a.step(left,diagnostics=True);right,info=b.step(right,diagnostics=True)
        torch.testing.assert_close(right.x,left.x,rtol=2e-9,atol=2e-10)
        for x,y in zip(right.velocity,left.velocity):torch.testing.assert_close(x,y,rtol=3e-7,atol=3e-9)
        assert info['nonlinear']['residual_norm']<=info['nonlinear']['tolerance']
        assert info['flow']['stokes']['momentum_residual']<=info['flow']['stokes']['momentum_tolerance']


def test_stokes_benchmark_input_readonly_and_same_solver_counts(real_case,tmp_path):
    from afsi_torch.simulation.real_lv_mac import run
    from validation.benchmark_real_lv_schemes import benchmark
    cfg=warm_config(real_case)
    cfg=replace(cfg,execution=replace(cfg.execution,execution_backend='fused',coupling_backend='optimized'))
    run(case_config=cfg,device='cpu',output=tmp_path/'source')
    source=tmp_path/'source'/'checkpoint.npz';before=source.read_bytes()
    report=benchmark(source,device='cpu',schemes=('cnab-semiimplicit',),nonlinear_solvers=('anderson-newton',),
                     stokes_backends=('reference','workspace'),warmup=1,steps=2)
    cases=list(report['cases'].values())
    assert all(c['completed'] for c in cases)
    assert cases[0]['counts']==cases[1]['counts']
    assert next(iter(report['stokes_comparisons'].values()))['final_state_differences']['x_max_abs']<1e-10
    assert source.read_bytes()==before
