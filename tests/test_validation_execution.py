"""Parallel checks preserve strict support, rejected geometries and CSR coupling."""
from dataclasses import replace
import os
import pytest
import torch
from test_real_lv import real_case, DEVICES
from test_mac_shared_pressure import warm_config
from test_mac_implicit import settings
from test_adaptive_p1_transfer import transfer
from afsi_torch.real_lv import imported_model
from afsi_torch.mac.execution import build_driver


INTERPRETER=os.environ.get('TRITON_INTERPRET')=='1'
KERNEL_DEVICES=[pytest.param('cpu',marks=pytest.mark.skipif(not INTERPRETER,reason='requires Triton interpreter')),
                pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available() or INTERPRETER,reason='native CUDA unavailable'))]


@pytest.mark.parametrize('device',KERNEL_DEVICES)
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_actual_blocked_support_strict_bounds_nonfinite_strides_and_padding(device,dtype):
    pytest.importorskip('triton')
    from afsi_torch.mac._triton_support import support_flags,_all_partial
    points=torch.full((1027,6),4.,device=device,dtype=dtype)[:,::2]
    origin=points.new_tensor([0.,0.,0.]);spacing=points.new_tensor([1.,2.,.5]);limits=points.new_tensor([8.,9.,10.])
    reference=lambda p:torch.isfinite(p).all() & (((p-origin)/spacing)>=2).all() & (((p-origin)/spacing)<limits).all()
    assert support_flags(points,origin,spacing,limits).item()==reference(points).item()
    for axis in range(3):
        low=origin[axis]+2*spacing[axis]
        high=origin[axis]+limits[axis]*spacing[axis]
        for value in (low,torch.nextafter(low,low.new_tensor(-torch.inf)),
                      torch.nextafter(high,high.new_tensor(-torch.inf)),high,
                      low.new_tensor(torch.nan),low.new_tensor(torch.inf),low.new_tensor(-torch.inf)):
            changed=points.clone();changed[-1,axis]=value
            assert support_flags(changed,origin,spacing,limits).item()==reference(changed).item()
    assert support_flags(points[:0],origin,spacing,limits).item()
    # Exercise recursive-style padded boolean reduction independently of how
    # many support points a small interpreter test can allocate/run quickly.
    partial=torch.ones(2051,device=device,dtype=torch.int32);partial[-1]=0
    out=torch.empty(3,device=device,dtype=torch.int32)
    _all_partial[(3,)](partial,out,len(partial),1024)
    torch.testing.assert_close(out,torch.tensor([1,1,0],device=device,dtype=torch.int32),rtol=0,atol=0)


@pytest.mark.parametrize('device',DEVICES)
def test_unique_used_p1_vertices_preserve_original_support_set(device):
    X,t=transfer(device)
    from afsi_torch.p1 import prepare_p1
    from afsi_torch.mac.adaptive_transfer import AdaptiveP1Transfer,InteractionQuadratureOptions
    Y=torch.cat((X[:4],X.new_tensor([[3.4,3.4,3.4]])))
    cells=torch.tensor([[0,1,2,3],[1,2,3,4]],device=device)
    g=prepare_p1(Y,cells,degree=2)
    t=AdaptiveP1Transfer(t.grid,g,quadrature_options=InteractionQuadratureOptions(mode='adaptive'))
    assert len(t.validation_points(Y))==5
    t.check_support(t.validation_points(Y))
    t.set_validation_backend('reference')
    assert len(t.validation_points(Y))==8  # shared corners retain old multiplicity
    t.check_support(t.validation_points(Y))
    bad=Y.clone();bad[0,0]=0.
    for mode in ('reference','blocked'):
        t.set_validation_backend(mode)
        with pytest.raises(ValueError,match='support'):t.check_support(t.validation_points(bad))
    with pytest.raises(ValueError,match='validation backend'):t.set_validation_backend('bad')
    # Exercise the subset selector independently: production consistent mass
    # rejects orphan DOFs, but support selection must not add them to its set.
    from types import SimpleNamespace
    subset=SimpleNamespace(validation_backend='blocked',_all_vertices_used=False,
                           _validation_vertices=torch.arange(5,device=device))
    extra=torch.cat((Y,Y.new_tensor([[30.,30.,30.]])))
    t.check_support(AdaptiveP1Transfer.validation_points(subset,extra))


@pytest.mark.parametrize('device',DEVICES)
def test_pointwise_geometry_fields_force_rejections_and_cache(real_case,device):
    model=imported_model(real_case,device)
    fast=model.execution_factory()
    X=model.mesh.X
    x=X+3e-3*torch.sin(1.7*X)
    old=model.geometry_state(x);new=fast._geometry_kernel(x)
    for a,b in zip(old[:-1],new[:-1]):torch.testing.assert_close(b,a,rtol=3e-12,atol=3e-13)
    torch.testing.assert_close(new[-1],old[-1],rtol=0,atol=0)
    for time in (.1,.6):
        torch.testing.assert_close(fast.force(x,time),model.force(x,time),rtol=2e-10,atol=3e-6)
    # Fullgraph tracing forbids silent graph breaks, even on the CPU host.
    compiled=torch.compile(model.execution_geometry_state,backend='eager',fullgraph=True)
    torch.testing.assert_close(compiled(x)[0],old[0],rtol=3e-12,atol=3e-13)
    fast.validate(x);assert fast.checked_geometry(x) is not None
    fast.set_validation_backend('reference');assert fast.checked_geometry(x) is None
    for invalid in (torch.zeros_like(X),2*X.mean(0)-X,torch.full_like(X,torch.nan)):
        for mode in ('reference','blocked'):
            fast.set_validation_backend(mode)
            with pytest.raises(ValueError,match='invalid'):fast.validate(invalid)


@pytest.mark.parametrize('device',DEVICES)
def test_active_coupled_trajectory_and_csr_unchanged(real_case,device):
    from afsi_torch.mac.semiimplicit import MidpointProblem
    cfg=warm_config(real_case)
    cfg=replace(cfg,execution=replace(cfg.execution,execution_backend='fused',coupling_backend='optimized'),
                interaction_quadrature=replace(cfg.interaction_quadrature,transfer_backend='fused'),
                coupling=replace(cfg.coupling,reuse_validation=True))
    model=imported_model(cfg,device)
    a,b=build_driver(model,settings(cfg),device),build_driver(model,settings(cfg),device)
    a.transfer.set_validation_backend('reference');a.solid_execution.set_validation_backend('reference')
    state=a.initialize(model.mesh.X)
    state=replace(state,step=6000,time=.6,force_time=.6,force=model.force(state.x,.6),
        previous_advection=a.flow.advection(state.velocity),previous_dt=cfg.time.dt)
    left,right=state,state
    for _ in range(3):
        left,ia=a.step(left,diagnostics=True);right,ib=b.step(right,diagnostics=True)
        torch.testing.assert_close(right.x,left.x,rtol=2e-10,atol=2e-11)
        torch.testing.assert_close(right.force,left.force,rtol=2e-8,atol=3e-6)
        for x,y in zip(right.velocity,left.velocity):torch.testing.assert_close(x,y,rtol=2e-7,atol=3e-10)
        assert ib['nonlinear']['residual_norm']<=ib['nonlinear']['tolerance']
        assert ib['power_error']/max(abs(ib['solid_power']),abs(ib['fluid_power']),1.)<1e-9
    problem=MidpointProblem(b,right,right.x,b.transfer.prepare(right.x),b.flow.advection(right.velocity))
    y=torch.zeros_like(right.x);v=.01*torch.sin(right.x)
    action=problem.linearization(y)
    finite=(problem.residual(y+1e-4*v)-problem.residual(y-1e-4*v))/2e-4
    assert problem.tangent.layout==torch.sparse_csr
    torch.testing.assert_close(action(v),finite,rtol=3e-5,atol=3e-8)


def test_validation_benchmark_reads_checkpoint_and_separates_modes(real_case,tmp_path):
    from afsi_torch.simulation.real_lv_mac import run
    from validation.benchmark_real_lv_schemes import benchmark
    cfg=warm_config(real_case)
    cfg=replace(cfg,execution=replace(cfg.execution,execution_backend='fused',coupling_backend='optimized'),
                interaction_quadrature=replace(cfg.interaction_quadrature,transfer_backend='fused'))
    run(case_config=cfg,device='cpu',output=tmp_path/'source')
    source=tmp_path/'source'/'checkpoint.npz';before=source.read_bytes()
    r=benchmark(source,device='cpu',schemes=('cnab-semiimplicit',),nonlinear_solvers=('anderson-newton',),
                validation_backends=('reference','blocked'),warmup=1,steps=2,profile=True)
    for case in r['cases'].values():
        assert case['completed'] and case['helmholtz']['backend']=='torch'
        assert 'ib_support_checks' in case['phases']
    assert next(iter(r['validation_comparisons'].values()))['final_state_differences']['x_max_abs']<2e-11
    assert before==source.read_bytes()
