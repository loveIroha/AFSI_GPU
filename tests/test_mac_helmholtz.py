"""Wall-stencil equivalence, owned outputs, graph reuse and coupled continuation."""
from dataclasses import replace
import os
import pytest
import torch
from afsi_torch.mac.grid import MACGrid,gradient,zero_normal,velocity_laplacian
from afsi_torch.mac.cnab import MACCNABFlow,CNABOptions
from afsi_torch.mac.helmholtz import HelmholtzWorkspace
from test_real_lv import real_case


CUDA = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
BACKENDS = ['torch',pytest.param('triton',marks=CUDA),pytest.param('graph',marks=CUDA)]


def inputs(grid,device):
    rng = torch.Generator(device=device).manual_seed(619)
    b = tuple(torch.randn(v.shape,device=device,dtype=v.dtype,generator=rng)
              for v in grid.zeros(device=device))  # includes nonzero normal wall RHS
    p = torch.randn(grid.shape,device=device,dtype=b[0].dtype,generator=rng)
    return b,p


def polynomial(flow,b,p=None):
    if p is not None:
        b = tuple(r-flow.dt/flow.rho*g for r,g in zip(b,gradient(p,flow.grid.spacing)))
    u = tuple(r/d for r,d in zip(zero_normal(b),flow._helmholtz_diagonal))
    for _ in range(flow.helmholtz_iterations):
        u = zero_normal(tuple(v+(r-v+flow.alpha*velocity_laplacian(v,c,flow.grid.spacing))/d
            for c,(v,r,d) in enumerate(zip(u,b,flow._helmholtz_diagonal))))
    return u


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('pressure',[False,True])
def test_fixed_polynomial_anisotropic_walls_and_retained_outputs(backend,pressure):
    device = 'cpu' if backend=='torch' else 'cuda'
    grid = MACGrid((4,6,8),(1.,1.7,2.3))
    flow = MACCNABFlow(grid,dt=.003,device=device,cnab_options=CNABOptions(helmholtz_backend='torch'))
    workspace = HelmholtzWorkspace(grid,flow._helmholtz_diagonal,backend=backend)
    b,p = inputs(grid,device)
    saved = tuple(v.clone() for v in b)
    actual = workspace.solve(b,alpha=flow.alpha,count=flow.helmholtz_iterations,
        pressure=p if pressure else None,pressure_scale=flow.dt/flow.rho)
    expected = polynomial(flow,b,p if pressure else None)
    for c,(a,e) in enumerate(zip(actual,expected)):
        torch.testing.assert_close(a,e,rtol=2e-12,atol=2e-13)
        assert a.select(c,0).count_nonzero()==0 and a.select(c,a.shape[c]-1).count_nonzero()==0
    retained = tuple(v.clone() for v in actual)
    for dt in (.0015,.00075,.003):
        flow.set_time_step(dt)
        next_b = tuple(-.7*v for v in b)
        result = workspace.solve(next_b,alpha=flow.alpha,count=flow.helmholtz_iterations,
            pressure=p if pressure else None,pressure_scale=flow.dt/flow.rho)
        for a,e in zip(result,polynomial(flow,next_b,p if pressure else None)):
            torch.testing.assert_close(a,e,rtol=2e-12,atol=2e-13)
    for a,e,s,v in zip(actual,retained,saved,b):
        torch.testing.assert_close(a,e,rtol=0,atol=0)
        torch.testing.assert_close(s,v,rtol=0,atol=0)
    assert workspace.summary()['solves']==4
    if backend=='graph':
        assert 1<=workspace.graph_builds<=3 and len(workspace.graphs)<=4


@pytest.mark.parametrize('backend',BACKENDS)
def test_stokes_response_preserves_true_acceptance_and_ownership(backend):
    device = 'cpu' if backend=='torch' else 'cuda'
    grid = MACGrid((4,)*3,(1.,1.2,1.4))
    reference = MACCNABFlow(grid,dt=.004,device=device,cnab_options=CNABOptions(helmholtz_backend='torch'))
    candidate = MACCNABFlow(grid,dt=.004,device=device,cnab_options=CNABOptions(helmholtz_backend='torch'))
    candidate._helmholtz_workspace = HelmholtzWorkspace(grid,candidate._helmholtz_diagonal,backend=backend)
    candidate.helmholtz_backend = backend
    b,p = inputs(grid,device)
    old = None
    for dt in (.004,.002,.004):
        reference.set_time_step(dt);candidate.set_time_step(dt)
        a,e = candidate.stokes(b,p),reference.stokes(b,p)
        for x,y in zip(a.velocity,e.velocity):
            torch.testing.assert_close(x,y,rtol=1e-8,atol=1e-10)
        torch.testing.assert_close(a.pressure,e.pressure,rtol=1e-8,atol=1e-9)
        check = candidate._stokes_metrics(a.velocity,a.pressure,zero_normal(b))
        assert check[0].item()<=a.diagnostics['stokes']['momentum_tolerance']
        assert check[1].item()<=a.diagnostics['stokes']['divergence_tolerance']
        if old is not None:
            for x,y in zip(old[0].velocity,old[1]):
                torch.testing.assert_close(x,y,rtol=0,atol=0)
        old = a,tuple(v.clone() for v in a.velocity)


@pytest.mark.skipif(os.environ.get('TRITON_INTERPRET')!='1',reason='run separately with TRITON_INTERPRET=1')
@pytest.mark.parametrize('pressure',[False,True])
def test_actual_triton_stencils_with_cpu_interpreter(pressure):
    pytest.importorskip('triton')
    grid = MACGrid((4,6,8),(1.,1.7,2.3))
    flow = MACCNABFlow(grid,dt=.003)
    b,p = inputs(grid,'cpu')
    workspace = HelmholtzWorkspace(grid,flow._helmholtz_diagonal,backend='triton')
    for dt in (.003,.0015):
        flow.set_time_step(dt)
        result = workspace.solve(b,alpha=flow.alpha,count=flow.helmholtz_iterations,
            pressure=p if pressure else None,pressure_scale=dt/flow.rho)
        for a,e in zip(result,polynomial(flow,b,p if pressure else None)):
            torch.testing.assert_close(a,e,rtol=2e-12,atol=2e-13)


def test_backend_defaults_and_reject_unsupported_cpu():
    flow = MACCNABFlow(MACGrid((4,)*3,(1.,)*3),dt=.001)
    assert flow.helmholtz_backend=='torch'
    with pytest.raises(ValueError,match='requires CUDA'):
        MACCNABFlow(flow.grid,dt=.001,cnab_options=CNABOptions(helmholtz_backend='graph'))
    with pytest.raises(ValueError,match='backend'):
        CNABOptions(helmholtz_backend='invalid')


@CUDA
def test_native_graph_cache_is_bounded_and_parameters_update_without_recapture():
    grid = MACGrid((4,)*3,(1.,)*3)
    flow = MACCNABFlow(grid,dt=.001,device='cuda',cnab_options=CNABOptions(helmholtz_backend='graph'))
    b,p = inputs(grid,'cuda')
    workspace = flow._helmholtz_workspace
    for count in range(1,7):
        workspace.solve(b,alpha=flow.alpha,count=count,pressure=p,pressure_scale=flow.dt)
    assert len(workspace.graphs)==4 and workspace.graph_builds==6
    builds = workspace.graph_builds
    # Same count, changed alpha and diagonal: graph arguments are mutable buffers.
    for dt in (.0005,.001):
        flow.set_time_step(dt)
        count = 6
        actual = workspace.solve(b,alpha=flow.alpha,count=count,pressure=p,pressure_scale=dt)
        previous_count = flow.helmholtz_iterations
        flow.helmholtz_iterations = count
        expected = polynomial(flow,b,p)
        flow.helmholtz_iterations = previous_count
        for a,e in zip(actual,expected):
            torch.testing.assert_close(a,e,rtol=2e-12,atol=2e-13)
    assert workspace.graph_builds==builds


@CUDA
def test_native_graph_coupled_trajectory_including_active_load(real_case):
    from afsi_torch.real_lv import imported_model
    from afsi_torch.mac.implicit import MACCouplingOptions
    from afsi_torch.mac.execution import build_driver
    from test_mac_implicit import settings
    cfg = replace(real_case,coupling=MACCouplingOptions(scheme='cnab-semiimplicit',semiimplicit_solver='anderson-newton'))
    model = imported_model(cfg,'cuda')
    def driver(backend):
        selected = replace(cfg,coupling=replace(cfg.coupling,cnab=replace(cfg.coupling.cnab,helmholtz_backend=backend)))
        return build_driver(model,settings(selected),'cuda')
    baseline,candidate = driver('torch'),driver('graph')
    state = baseline.initialize(model.mesh.X)
    state = replace(state,step=6000,time=.6,force_time=.6,force=model.force(state.x,.6),
                    previous_advection=baseline.flow.advection(state.velocity),previous_dt=cfg.time.dt)
    a,b = state,state
    for _ in range(3):
        a,_ = baseline.step(a,diagnostics=False)
        b,info = candidate.step(b,diagnostics=False)
        torch.testing.assert_close(a.x,b.x,rtol=1e-9,atol=1e-10)
        for u,v in zip(a.velocity,b.velocity):
            torch.testing.assert_close(u,v,rtol=2e-6,atol=2e-8)
        torch.testing.assert_close(a.pressure,b.pressure,rtol=2e-6,atol=2e-6)
        assert info['nonlinear']['residual_norm']<=info['nonlinear']['tolerance']


def test_benchmark_backend_comparison_and_phase_registration(real_case,tmp_path):
    from afsi_torch.simulation.real_lv_mac import run
    from afsi_torch.mac.implicit import MACCouplingOptions
    from validation.benchmark_real_lv_schemes import benchmark
    cfg = replace(real_case,coupling=MACCouplingOptions(scheme='cnab-semiimplicit',semiimplicit_solver='anderson-newton'),
                  output=replace(real_case.output,write_vtk=False))
    run(case_config=cfg,device='cpu',output=tmp_path/'case')
    report = benchmark(tmp_path/'case'/'checkpoint.npz',device='cpu',schemes=('cnab-semiimplicit',),
        nonlinear_solvers=('anderson-newton',),helmholtz_backends=('torch',),warmup=1,steps=1,profile=True)
    case = report['cases']['cnab-semiimplicit/anderson-newton/torch']
    assert case['completed'] and case['helmholtz']['backend']=='torch'
    assert case['counts']['helmholtz_solves']>=case['counts']['stokes_solves']
    assert case['counts']['helmholtz_sweeps']>=case['counts']['helmholtz_solves']
    assert case['phases']['velocity_helmholtz']['calls']>0
