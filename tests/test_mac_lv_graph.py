"""Closed-box graph execution: gauge, partial blocks, ownership and LV guards."""
from dataclasses import replace
from math import pi
import pytest
import torch
import torch.nn.functional as F
from afsi_torch.mac import MACGrid,MGOptions,GeometricMultigrid
from afsi_torch.mac.grid import negative_laplacian
from afsi_torch.mac.execution import build_driver
from afsi_torch.mac.checkpoint import save_mac,load_mac

DEVICES=['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))]


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('smooth',[1,3,4])
@pytest.mark.parametrize('shape',[(4,4,4),(8,12,16)])
def test_closed_workspace_preserves_cycle_gauge_and_storage(device,smooth,shape):
    grid=MACGrid(shape,(2.,3.,4.))
    options=MGOptions(smooth=smooth,max_cycles=200)
    reference=GeometricMultigrid(grid,device=device,options=options,backend='fused')
    candidate=GeometricMultigrid(grid,device=device,options=options,backend='workspace')
    xyz=grid.coordinates(device=device)
    p=torch.cos(pi*xyz[...,0]/2)*torch.cos(pi*xyz[...,1]/3)*torch.cos(pi*xyz[...,2]/4)
    p-=p.mean()
    rhs=negative_laplacian(p,grid.spacing)
    start=.1*p
    reference.workspace.initialize(start,rhs)
    candidate.workspace.initialize(start,rhs)
    torch.testing.assert_close(candidate.workspace.cycle(),reference.workspace.cycle(),rtol=2e-11,atol=2e-12)
    if min(shape)>4:
        out=torch.empty(tuple(n//2 for n in shape),device=device,dtype=p.dtype)
        candidate.workspace.kernels.residual_restrict(start,rhs,out,grid.spacing)
        expected=F.avg_pool3d((rhs-negative_laplacian(start,grid.spacing))[None,None],2,2)[0,0]
        torch.testing.assert_close(out,expected,rtol=2e-11,atol=2e-12)
    result,info=candidate.solve(rhs,start+7)
    saved=result.clone()
    torch.testing.assert_close(result,p,rtol=2e-8,atol=2e-9)
    assert abs(result.mean().item())<1e-12 and info['residual_norm']<=info['tolerance']
    candidate.solve(-rhs,result)
    torch.testing.assert_close(result,saved,rtol=0,atol=0)
    zero,info=candidate.solve(torch.zeros_like(rhs))
    assert info['cycles']==0 and zero.count_nonzero()==0
    with pytest.raises(ValueError,match='incompatible'):
        candidate.solve(torch.ones_like(rhs))
    with pytest.raises(ValueError,match='invalid pressure RHS'):
        candidate.solve(rhs*float('nan'))
    with pytest.raises(ValueError,match='initial pressure'):
        candidate.solve(rhs,torch.full_like(rhs,float('nan')))
    torch.testing.assert_close(rhs,negative_laplacian(p,grid.spacing),rtol=0,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('smooth',[1,3,4])
def test_pressure_graph_partial_block_and_mean_constraint(smooth):
    grid=MACGrid((16,16,16),(2.,3.,4.))
    options=MGOptions(smooth=smooth,max_cycles=3,check_every=2,rtol=0,atol=1e-30)
    reference=GeometricMultigrid(grid,device='cuda',options=options,backend='workspace')
    graph=GeometricMultigrid(grid,device='cuda',options=options,backend='graph')
    p=torch.sin(grid.coordinates(device='cuda').sum(-1)); p-=p.mean()
    rhs=negative_laplacian(p,grid.spacing)
    with pytest.raises(RuntimeError,match='did not converge'):
        reference.solve(rhs)
    with pytest.raises(RuntimeError,match='did not converge'):
        graph.solve(rhs)
    torch.testing.assert_close(graph.workspace.p[0],reference.workspace.p[0],rtol=2e-11,atol=2e-12)
    assert set(graph.workspace.graphs)=={1,2}
    graph=GeometricMultigrid(grid,device='cuda',options=MGOptions(smooth=smooth,max_cycles=200),backend='graph')
    result,info=graph.solve(rhs)
    saved=result.clone()
    graph.solve(-rhs,result)
    torch.testing.assert_close(saved,result,rtol=0,atol=0)
    torch.testing.assert_close(result,p,rtol=2e-8,atol=2e-9)
    assert info['residual_norm']<=info['tolerance'] and abs(result.mean().item())<1e-12


def model_and_settings(device):
    pytest.importorskip('gmsh')
    from afsi_torch.afsi337 import generated_model
    model=generated_model(mesh_size=.4,device=device)
    settings=dict(fluid_cells=16,box_length=5.,dt=5e-5,rho=1.,mu=1.,interaction_degree=None,
        warm_start=True,execution_backend='fused',solid_backend='pointwise',mass_backend='graph')
    return model,settings


def assert_states(a,b):
    assert a.step==b.step and a.time==b.time and a.force_time==b.force_time
    for name in ('x','pressure','force'):
        torch.testing.assert_close(getattr(a,name),getattr(b,name),rtol=1e-7,atol=1e-8)
    for u,v in zip(a.velocity,b.velocity):
        torch.testing.assert_close(u,v,rtol=1e-7,atol=1e-9)


@pytest.mark.parametrize('device',DEVICES)
def test_lv_optimized_checks_cache_invalidation_and_rejected_geometry(device,monkeypatch):
    model,settings=model_and_settings(device)
    reference=build_driver(model,dict(settings,pressure_backend='fused'),device)
    fast=build_driver(model,dict(settings,pressure_backend='graph' if device=='cuda' else 'workspace',coupling_backend='optimized'),device)
    a,b=reference.initialize(model.mesh.X),fast.initialize(model.mesh.X)
    for _ in range(12):
        a,ia=reference.step(a,diagnostics=True)
        b,ib=fast.step(b,diagnostics=True)
    assert_states(a,b)
    assert reference.point_evaluations==25 and fast.point_evaluations==13
    assert ib['used_force_time_s']==10*settings['dt'] and ib['next_force_time_s']==11*settings['dt']
    assert ib['power_error']<1e-7
    old=fast.point_evaluations
    b.x.add_(0.)
    b,_=fast.step(b,diagnostics=False)
    assert fast.point_evaluations==old+2
    remembered=fast.solid_execution._cached_geometry
    method=fast.solid_execution.force_with_geometry
    def invalid(x,time):
        force,geometry=method(x,time)
        return force,(*geometry[:-1],torch.zeros_like(geometry[-1]))
    with monkeypatch.context() as patch:
        patch.setattr(fast.solid_execution,'force_with_geometry',invalid)
        with pytest.raises(ValueError,match='invalid LV deformation'):
            fast.step(b,diagnostics=False)
    assert fast.solid_execution._cached_geometry is remembered
    assert fast._cached_state is b
    with pytest.raises(ValueError,match='invalid LV deformation'):
        fast.step(replace(b,x=torch.zeros_like(b.x)),diagnostics=False)
    flags=torch.ones(4,device=device,dtype=torch.bool)
    metrics=fast._accept_metrics(torch.zeros_like(b.x),flags,b.force,torch.zeros((2,3),device=device,dtype=b.x.dtype)).tolist()
    assert metrics[2]==0  # support guard remains enabled


@pytest.mark.parametrize('device',DEVICES)
def test_old_lv_checkpoint_switches_execution_and_resumes(tmp_path,device):
    model,settings=model_and_settings(device)
    reference=build_driver(model,dict(settings,pressure_backend='fused'),device)
    state=reference.initialize(model.mesh.X)
    for _ in range(4):
        state,_=reference.step(state,diagnostics=False)
    path=tmp_path/'checkpoint.npz'
    settings['pressure_backend']='fused'  # no coupling key in an older checkpoint
    save_mac(path,model,state,settings,dict(elapsed_seconds=0.,segments=[],summary={}))
    ma,a,sa,_=load_mac(path,device)
    mb,b,sb,_=load_mac(path,device)
    ref=build_driver(ma,sa,device)
    opt=build_driver(mb,dict(sb,coupling_backend='optimized',pressure_backend='graph' if device=='cuda' else 'workspace'),device)
    for _ in range(8):
        a,_=ref.step(a,diagnostics=False)
        b,_=opt.step(b,diagnostics=False)
    assert_states(a,b)
    assert opt.point_evaluations==9


def test_combined_lv_checks_are_fullgraph_compilable(monkeypatch):
    import afsi_torch.mac.execution as execution
    import afsi_torch.mac.compact_transfer as transfer
    import afsi_torch.mac.solid_execution as solid
    for module in (execution,transfer,solid):
        monkeypatch.setattr(module,'tensor_kernel',lambda f,d:torch.compile(f,backend='eager',fullgraph=True,dynamic=False))
    model,settings=model_and_settings('cpu')
    driver=build_driver(model,dict(settings,pressure_backend='workspace',coupling_backend='optimized'),'cpu')
    state,_=driver.step(driver.initialize(model.mesh.X),diagnostics=False)
    assert state.step==1
    with pytest.raises(ValueError,match='requires fused'):
        build_driver(None,dict(execution_backend='torch',coupling_backend='optimized'),'cpu')


@pytest.mark.parametrize('device',DEVICES)
def test_lv_demo_graph_options_persist_and_allow_rollback(tmp_path,device):
    pytest.importorskip('gmsh')
    from demo.ideal_lv_fsi.run_mac import main
    output=tmp_path/'demo'
    pressure='graph' if device=='cuda' else 'workspace'
    report=main(['--device',device,'--mesh-size','.4','--fluid-cells','16','--warm-start',
        '--execution-backend','fused','--solid-backend','pointwise','--mass-backend','graph',
        '--coupling-backend','optimized','--pressure-backend',pressure,'--end-time','.0002',
        '--output',str(output),'--log-every','2','--checkpoint-every','2'])
    assert report['coupling_backend']=='optimized'
    assert report['pressure_cuda_graphs']==(2 if device=='cuda' else 0)
    report=main(['--device',device,'--resume',str(output/'checkpoint.npz'),'--end-time','.0004'])
    assert report['settings']['pressure_backend']==pressure and report['coupling_backend']=='optimized'
    report=main(['--device',device,'--resume',str(output/'checkpoint.npz'),'--end-time','.0005',
        '--pressure-backend','fused','--coupling-backend','reference'])
    assert report['coupling_backend']=='reference' and report['pressure_cuda_graphs']==0
    assert report['segments'][-2]['coupling_backend']=='optimized'


def test_benchmark_reads_checkpoint_and_compares_same_state(tmp_path):
    from validation.benchmark_lv_mac_graph import benchmark
    model,settings=model_and_settings('cpu')
    settings['pressure_backend']='fused'
    driver=build_driver(model,settings,'cpu')
    state=driver.initialize(model.mesh.X)
    path=tmp_path/'checkpoint.npz'
    save_mac(path,model,state,settings,dict(elapsed_seconds=0.,segments=[],summary={}))
    original=path.read_bytes()
    report=benchmark(path,device='cpu',steps=3,warmup=2)
    assert path.read_bytes()==original and report['equivalence_passed']
    assert report['end_step']==5 and report['variants']['reference']['point_evaluations']==6
    assert report['variants']['workspace']['point_evaluations']==3
    assert report['variants']['reference']['settings']['solid_backend']==report['variants']['workspace']['settings']['solid_backend']
    assert report['variants']['reference']['settings']['mass_backend']==report['variants']['workspace']['settings']['mass_backend']
