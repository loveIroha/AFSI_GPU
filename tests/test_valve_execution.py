"""Execution equivalence, mixed outlet ghosts, cache invalidation and ownership."""
import pytest
import torch
from afsi_torch.afsi340 import ValveConfig,generate_valve,ValveSolid
from afsi_torch.mac2d import ChannelGrid,negative_laplacian
from afsi_torch.mac2d.multigrid import ChannelMultigrid
from afsi_torch.mac2d.execution import build_driver
from afsi_torch.mac.multigrid import MGOptions

DEVICES=['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))]


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('smooth',[1,3,4])
def test_buffered_channel_cycle_and_manufactured_solution(device,smooth):
    grid=ChannelGrid((32,8))
    options=MGOptions(smooth=smooth,max_cycles=200)
    ref=ChannelMultigrid(grid,device=device,options=options)
    opt=ChannelMultigrid(grid,device=device,options=options,backend='workspace')
    coords=grid.coordinates(device=device)
    p=torch.sin(.7*coords[...,0])+torch.cos(1.1*coords[...,1])+.3
    rhs=negative_laplacian(p,grid.spacing)
    initial=.1*p
    expected=ref._cycle(0,initial.clone(),rhs)
    opt.workspace.initialize(initial,rhs)
    torch.testing.assert_close(opt.workspace.advance(1),expected,rtol=2e-12,atol=2e-12)
    result,info=opt.solve(rhs)
    assert info['residual_norm']<=info['tolerance']
    torch.testing.assert_close(result,p,rtol=2e-8,atol=2e-8)
    saved=result.clone()
    other,_=opt.solve(.7*rhs,initial=result)
    torch.testing.assert_close(result,saved,rtol=0,atol=0)
    torch.testing.assert_close(other,.7*p,rtol=2e-8,atol=2e-8)
    zero,info=opt.solve(torch.zeros_like(rhs))
    assert info['cycles']==0 and zero.count_nonzero()==0
    with pytest.raises(ValueError,match='finite'):
        opt.solve(rhs*float('nan'))
    for solver in (ref,opt):
        with pytest.raises(RuntimeError,match='nonfinite'):
            solver.solve(torch.full_like(rhs,1e308))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_channel_graph_check_schedule_and_result_storage():
    grid=ChannelGrid((32,8))
    options=MGOptions(check_every=3,max_cycles=2,rtol=0,atol=1e-30)
    graph=ChannelMultigrid(grid,device='cuda',options=options,backend='graph')
    p=torch.sin(grid.coordinates(device='cuda').sum(-1))
    rhs=negative_laplacian(p,grid.spacing)
    with pytest.raises(RuntimeError,match='did not converge'):
        graph.solve(rhs)
    assert set(graph.workspace.graphs)=={1,2,3}
    # The graph options/capture schedule are immutable; construct a fresh solver.
    graph=ChannelMultigrid(grid,device='cuda',backend='graph')
    result,info=graph.solve(rhs)
    saved=result.clone()
    graph.solve(2*rhs)
    torch.testing.assert_close(result,saved,rtol=0,atol=0)
    torch.testing.assert_close(result,p,rtol=2e-8,atol=2e-8)
    assert info['residual_norm']<=info['tolerance']


def drivers(device):
    pytest.importorskip('gmsh')
    config=ValveConfig(mesh_size=.03)
    solid=ValveSolid(generate_valve(config,device=device),config)
    settings=dict(dt=1/16000,nx=32,ny=8,rho=1.,mu=.1,mass_backend='graph',warm_start=True,fused=True)
    ref=build_driver(solid,dict(settings,execution_backend='reference'),device)
    opt=build_driver(solid,dict(settings,execution_backend='optimized'),device)
    return ref,opt


@pytest.mark.parametrize('device',DEVICES)
def test_optimized_coupling_matches_reference_and_reuses_stencil(device):
    ref,opt=drivers(device)
    a,b=ref.initialize(),opt.initialize()
    for _ in range(12):
        a,ia=ref.step(a,diagnostics=True)
        b,ib=opt.step(b,diagnostics=True)
    assert ref.stencil_builds==25 and opt.stencil_builds==13
    torch.testing.assert_close(b.x,a.x,rtol=1e-9,atol=1e-10)
    for name in ('pressure','force'):
        torch.testing.assert_close(getattr(b,name),getattr(a,name),rtol=1e-7,atol=1e-7)
    for u,v in zip(b.velocity,a.velocity):
        torch.testing.assert_close(u,v,rtol=1e-7,atol=1e-8)
    assert ib['power_relative_error']<1e-7
    old=opt.stencil_builds
    b.x.add_(0.)  # Mutating even to identical values invalidates the tensor version.
    b,_=opt.step(b)
    assert opt.stencil_builds==old+2
    bad=tuple(torch.full_like(v,float('nan')) for v in b.velocity)
    with pytest.raises(ValueError,match='finite'):
        opt.flow.step(bad,opt.flow.grid.zeros(device=device),time=0.)
    with pytest.raises(ValueError,match='stability'):
        opt.flow.step(tuple(torch.full_like(v,100.) for v in b.velocity),opt.flow.grid.zeros(device=device),time=0.)
    invalid=b.x*0
    J=torch.full_like(opt.solid.geometry.weights,-1.)
    metrics=opt._accept_metrics(torch.zeros_like(invalid),J,torch.zeros_like(invalid),invalid).tolist()
    assert metrics[1]==0 and metrics[2]==0


def test_optimized_fullgraph_trace(monkeypatch):
    import afsi_torch.afsi340 as solid_module
    import afsi_torch.mac2d.flow as flow_module
    import afsi_torch.mac2d.transfer as transfer_module
    import afsi_torch.mac2d.coupling as coupling_module
    import afsi_torch.mac2d.multigrid as mg_module
    for module in (solid_module,flow_module,transfer_module,coupling_module,mg_module):
        monkeypatch.setattr(module,'tensor_kernel',lambda f,d:torch.compile(f,backend='eager',fullgraph=True,dynamic=False))
    _,opt=drivers('cpu')
    state,_=opt.step(opt.initialize())
    assert state.step==1


@pytest.mark.parametrize('device',DEVICES)
def test_optimized_resumes_old_checkpoint_without_changing_discretization(tmp_path,device):
    from afsi_torch.mac2d import checkpoint
    ref,_=drivers(device)
    state=ref.initialize()
    for _ in range(4):
        state,_=ref.step(state)
    # Old checkpoints have no execution/pressure-backend keys or IB caches.
    settings=dict(dt=ref.flow.dt,nx=32,ny=8,rho=1.,mu=.1,mass_backend='graph',warm_start=True,fused=True)
    path=tmp_path/'checkpoint.npz'
    checkpoint.save(path,ref.solid,state,settings,dict(elapsed_seconds=0.,segments=[],frames=[],summary={}))
    loaded,restart,restored,_=checkpoint.load(path,device)
    opt=build_driver(loaded,restored,device)
    # Compare two restarted solvers: both reset PCG warm-start caches at load.
    baseline,state,_,_=checkpoint.load(path,device)
    ref=build_driver(baseline,dict(restored,execution_backend='reference'),device)
    assert opt.optimized
    for _ in range(8):
        state,_=ref.step(state)
        restart,_=opt.step(restart)
    assert restart.step==state.step and restart.time==state.time
    torch.testing.assert_close(restart.x,state.x,rtol=1e-9,atol=1e-10)
    torch.testing.assert_close(restart.pressure,state.pressure,rtol=1e-7,atol=1e-7)
    torch.testing.assert_close(restart.force,state.force,rtol=1e-7,atol=1e-7)
    for a,b in zip(restart.velocity,state.velocity):
        torch.testing.assert_close(a,b,rtol=1e-7,atol=1e-8)
    assert opt.stencil_builds==9  # first resumed input plus one per accepted step
