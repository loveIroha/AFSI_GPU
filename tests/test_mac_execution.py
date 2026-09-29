"""Execution-only changes: reference equations, ownership, guards and resume."""
import pytest
import torch
from afsi_torch.mac import MACGrid,MACFlow,divergence
from afsi_torch.mac.grid import zero_normal
from afsi_torch.mac.transfer import FETransfer
from afsi_torch.mac.compact_transfer import CompactFETransfer
from afsi_torch.mac.mass_solver import MassSolver
from afsi_torch.mac.solid_execution import SolidExecution
from afsi_torch.fluid.solvers import pcg,SolverOptions
from afsi_torch.solid import prepare_p2
from afsi_torch.tetrahedron import promote_p1

DEVICES=['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))]


def setup(device,degree=4):
    X=torch.tensor([[0.,0.,0.],[.6,0.,0.],[0.,.6,0.],[0.,0.,.6],[.6,.6,.6]],device=device,dtype=torch.float64)-.2
    X,cells=promote_p1(X,torch.tensor([[0,1,2,3],[1,2,3,4]],device=device))
    geometry=prepare_p2(X,cells,degree=degree)
    grid=MACGrid((16,20,24),(4.,6.,8.),(-2.,-3.,-4.))
    return X,geometry,grid


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('degree',[4,6])
def test_compact_ib_conservation_and_reference(device,degree):
    X,g,grid=setup(device,degree)
    a,b=FETransfer(grid,g,warm_start=True),CompactFETransfer(grid,g,warm_start=True)
    x=X+.013*torch.sin(3*X)
    sa,sb=a.prepare(x),b.prepare(x)
    assert sb.storage_bytes<sum(t.numel()*t.element_size() for t in (*sa.indices,*sa.weights))/8
    for c in range(3):
        ids,w=b._expanded_component(sb,c,0,g.weights.numel())
        torch.testing.assert_close(ids,sa.indices[c],rtol=0,atol=0)
        torch.testing.assert_close(w,sa.weights[c],rtol=2e-13,atol=2e-15)
    velocity=zero_normal(tuple(torch.sin((c+1)*grid.coordinates(c,device=device).sum(-1)) for c in range(3)))
    for k in range(2):
        force=torch.sin(3*X)+.01*k
        fa,_=a.spread(force,sa)
        fb,fi=b.spread(force,sb)
        ua,_=a.interpolate(velocity,sa)
        ub,ui=b.interpolate(velocity,sb)
        torch.testing.assert_close(ua,ub,rtol=1e-9,atol=1e-10)
        for u,v in zip(fa,fb):
            torch.testing.assert_close(u,v,rtol=1e-9,atol=1e-10)
        assert fi.residual_norm<=fi.tolerance and ui.residual_norm<=ui.tolerance
        torch.testing.assert_close(torch.stack([f.sum()*grid.volume for f in fb]),force.sum(0),rtol=1e-10,atol=2e-10)
        torch.testing.assert_close((ub*force).sum(),grid.volume*sum((u*f).sum() for u,f in zip(velocity,fb)),rtol=1e-10,atol=2e-10)
        torque=torch.zeros(3,device=device,dtype=X.dtype)
        for c,f in enumerate(fb):
            vec=torch.zeros((*f.shape,3),device=device,dtype=X.dtype)
            vec[...,c]=f
            torque+=torch.linalg.cross(grid.coordinates(c,device=device),vec).sum((0,1,2))*grid.volume
        torch.testing.assert_close(torque,torch.linalg.cross(x,force).sum(0),rtol=1e-10,atol=2e-10)
    uniform=tuple(torch.full_like(v,float(c+1)) for c,v in enumerate(velocity))
    u,_=b.interpolate(uniform,sb)
    torch.testing.assert_close(u,X.new_tensor([1.,2.,3.]).expand_as(X),rtol=1e-10,atol=1e-10)
    with pytest.raises(ValueError,match='support'):
        b.prepare(x+10)
    invalid=list(velocity)
    invalid[0]=torch.full_like(invalid[0],float('nan'))
    with pytest.raises(ValueError,match='invalid'):
        b.interpolate(tuple(invalid),sb)


@pytest.mark.parametrize('device',DEVICES)
def test_mass_pcg_restart_true_residual_and_storage(device):
    X,g,grid=setup(device)
    t=FETransfer(grid,g)
    options=SolverOptions(rtol=1e-11,atol=1e-13,recompute_every=12,check_every=4,max_iterations=500)
    solver=MassSolver(t.mass,t.diagonal,options)
    rhs=torch.sin(5*X)
    initial=.1*torch.cos(X)
    result,info=solver.solve(rhs,initial)
    reference,_=pcg(t.mass_action,rhs,t.diagonal,initial=initial,options=options)
    torch.testing.assert_close(result,reference,rtol=1e-8,atol=1e-9)
    assert torch.linalg.vector_norm(rhs-t.mass_action(result)).item()<=info.tolerance
    saved=result.clone()
    ptrs=[v.data_ptr() for v in (solver.x,solver.r,solver.d,solver.Ad)]
    next_result,_=solver.solve(-rhs,result)
    torch.testing.assert_close(next_result,-reference,rtol=1e-8,atol=1e-9)
    torch.testing.assert_close(result,saved,rtol=0,atol=0)
    assert ptrs==[v.data_ptr() for v in (solver.x,solver.r,solver.d,solver.Ad)]
    zero,zinfo=solver.solve(torch.zeros_like(rhs))
    assert zinfo.iterations==0 and zero.count_nonzero()==0
    with pytest.raises(ValueError,match='finite'):
        solver.solve(torch.full_like(rhs,float('nan')))
    bad=MassSolver(-t.mass,t.diagonal,options)
    with pytest.raises(RuntimeError,match='breakdown'):
        bad.solve(rhs)
    short=MassSolver(t.mass,t.diagonal,SolverOptions(max_iterations=1,rtol=1e-14))
    with pytest.raises(RuntimeError,match='converge'):
        short.solve(rhs)


@pytest.mark.parametrize('device',DEVICES)
def test_compiled_fluid_matches_reference_and_keeps_guards(device):
    grid=MACGrid((8,12,16),(2.,3.,4.))
    a=MACFlow(grid,dt=5e-5,device=device)
    b=MACFlow(grid,dt=5e-5,device=device,execution_backend='fused')
    u=zero_normal(tuple(.01*torch.sin(grid.coordinates(c,device=device).sum(-1)) for c in range(3)))
    density=tuple(torch.cos(grid.coordinates(c,device=device).sum(-1)) for c in range(3))
    before=tuple(v.clone() for v in u)
    ra,rb=a.step(u,density),b.step(u,density)
    torch.testing.assert_close(ra.pressure,rb.pressure,rtol=1e-8,atol=1e-9)
    for v,w,old,copy in zip(ra.velocity,rb.velocity,u,before):
        torch.testing.assert_close(v,w,rtol=1e-8,atol=1e-10)
        torch.testing.assert_close(old,copy,rtol=0,atol=0)
    assert divergence(rb.velocity,grid.spacing).abs().max()<1e-8
    with pytest.raises(ValueError,match='stability'):
        b.step(tuple(torch.full_like(v,100.) for v in u),density)
    with pytest.raises(ValueError,match='finite'):
        b.step(u,tuple(torch.full_like(v,float('nan')) for v in u))
    with pytest.raises(ValueError,match='normal'):
        b.project(tuple(torch.ones_like(v) for v in u))


@pytest.mark.parametrize('device',DEVICES)
def test_solid_compiled_force_cache_and_mutation(device):
    pytest.importorskip('gmsh')
    from afsi_torch.afsi337 import generated_model
    m=generated_model(mesh_size=.4,device=device)
    fast=SolidExecution(m)
    x=m.mesh.X+.001*torch.sin(2*m.mesh.X)
    for time in (0.,.3,1.5):
        fast.validate(x)
        expected=m.force(x,time)
        actual=fast.force(x,time)
        torch.testing.assert_close(actual,expected,rtol=2e-9,atol=2e-8)
    old=fast._cached_geometry
    fast.validate(x)
    assert fast._cached_geometry is old
    x.add_(.0001*torch.cos(x))
    fast.validate(x)
    assert fast._cached_geometry is not old
    torch.testing.assert_close(fast.force(x,.4),m.force(x,.4),rtol=2e-9,atol=2e-8)
    x.mul_(0.)
    with pytest.raises(ValueError,match='invalid'):
        fast.validate(x)


@pytest.mark.parametrize('device',DEVICES)
def test_execution_backend_checkpoint_resume(tmp_path,device):
    pytest.importorskip('gmsh')
    from examples.lv_mac import run
    from afsi_torch.mac.checkpoint import load_mac
    common=dict(device=device,mesh_size=.4,fluid_cells=16,warm_start=True,pressure_backend='fused',log_every=2,checkpoint_every=2)
    run(**common,output=tmp_path/'reference',end_time=.0003)
    run(**common,execution_backend='fused',output=tmp_path/'fast',end_time=.00015)
    report=run(device=device,resume=tmp_path/'fast'/'checkpoint.npz',end_time=.0003)
    assert report['settings']['execution_backend']=='fused'
    _,a,_,_=load_mac(tmp_path/'reference'/'checkpoint.npz',device)
    _,b,_,_=load_mac(tmp_path/'fast'/'checkpoint.npz',device)
    for name in ('x','force','pressure'):
        torch.testing.assert_close(getattr(a,name),getattr(b,name),rtol=1e-7,atol=1e-8)
    for u,v in zip(a.velocity,b.velocity):
        torch.testing.assert_close(u,v,rtol=1e-7,atol=1e-9)


def test_full_graph_capture_has_no_time_dependent_recompilation(monkeypatch):
    """Trace the actual optimized path on CPU; CUDA code generation is separate."""
    pytest.importorskip('gmsh')
    import afsi_torch.mac.execution as execution
    import afsi_torch.mac.compact_transfer as transfer
    import afsi_torch.mac.mass_solver as mass
    import afsi_torch.mac.solid_execution as solid_exec
    from afsi_torch.afsi337 import generated_model
    graphs=[]
    def backend(gm,inputs):
        graphs.append(gm)
        return gm.forward
    def compile_graph(function,device):
        return torch.compile(function,backend=backend,fullgraph=True,dynamic=False)
    for module in (execution,transfer,mass,solid_exec):
        monkeypatch.setattr(module,'tensor_kernel',compile_graph)
    model=generated_model(mesh_size=.4)
    settings=dict(fluid_cells=16,box_length=5.,dt=5e-5,rho=1.,mu=1.,
                  interaction_degree=None,warm_start=True,pressure_backend='fused',execution_backend='fused')
    driver=execution.build_driver(model,settings,'cpu')
    state=driver.initialize(model.mesh.X)
    for _ in range(3):
        state,_=driver.step(state,diagnostics=False)
    count=len(graphs)
    state,_=driver.step(state,diagnostics=False)
    assert len(graphs)==count and count>10
