"""Native P2/2D/nodal transfer equivalence and JSON demo selection."""
from dataclasses import replace
from pathlib import Path
import json
import pytest
import torch
from afsi_torch.mac import cuda_ib

GPU=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
ROOT=Path(__file__).resolve().parents[1]

def test_native_transfer_rejects_cpu():
    with pytest.raises(ValueError,match='requires CUDA'):
        cuda_ib.indexed_gather(torch.zeros(4,1),torch.zeros(1,2,dtype=torch.int64),torch.ones(1,2))

@GPU
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('neighbors,components',[(16,1),(64,3),(37,2)])
def test_indexed_signed_duplicate_weights_adjoint(dtype,neighbors,components):
    gen=torch.Generator(device='cuda').manual_seed(31)
    ids=torch.randint(0,53,(19,neighbors),generator=gen,device='cuda')
    weights=torch.randn(ids.shape,generator=gen,device='cuda',dtype=dtype)
    u=torch.randn((53,components),generator=gen,device='cuda',dtype=dtype)
    f=torch.randn((19,components),generator=gen,device='cuda',dtype=dtype)
    v=.017
    expected=(u[ids]*weights[...,None]).sum(1)
    actual=cuda_ib.indexed_gather(u,ids,weights)
    density=cuda_ib.indexed_spread(f,ids,weights,53,v)
    reference=torch.zeros_like(u).index_add(0,ids.flatten(),(f[:,None]*weights[...,None]/v).reshape(-1,components))
    tol=dict(rtol=3e-5,atol=3e-4) if dtype==torch.float32 else dict(rtol=2e-12,atol=2e-11)
    torch.testing.assert_close(actual,expected,**tol)
    torch.testing.assert_close(density,reference,**tol)
    torch.testing.assert_close((u*density).sum()*v,(actual*f).sum(),**tol)

@GPU
def test_compact_p2_mass_transfer_matches_expanded_quadrature():
    from afsi_torch.solid import prepare_p2
    from afsi_torch.mac.grid import MACGrid
    from afsi_torch.mac.transfer import FETransfer
    from afsi_torch.mac.compact_transfer import CompactFETransfer
    vertices=torch.tensor([[1.,1.,1.],[1.3,1.,1.],[1.,1.3,1.],[1.,1.,1.3]],device='cuda',dtype=torch.float64)
    edges=torch.tensor([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],device='cuda')
    x=torch.cat((vertices,vertices[edges].mean(1)))
    g=prepare_p2(x,torch.arange(10,device='cuda')[None])
    grid=MACGrid((12,14,16),(3.,3.5,4.))
    ref=FETransfer(grid,g)
    native=CompactFETransfer(grid,g,ib_backend='cuda')
    f=torch.sin(2*x)
    velocity=tuple(torch.sin(grid.coordinates(c,device='cuda')[...,c]) for c in range(3))
    sr,sn=ref.prepare(x),native.prepare(x)
    a,_=ref.spread(f,sr);b,_=native.spread(f,sn)
    for aa,bb in zip(a,b):torch.testing.assert_close(aa,bb,rtol=2e-9,atol=2e-9)
    a,_=ref.interpolate(velocity,sr);b,_=native.interpolate(velocity,sn)
    torch.testing.assert_close(a,b,rtol=2e-9,atol=2e-9)

@GPU
def test_valve_p2_reflected_wall_transfer_and_loaded_step(tmp_path):
    from test_valve_mac import make_solid
    from afsi_torch.mac2d.grid import ChannelGrid
    from afsi_torch.mac2d.transfer import TriangleTransfer
    from afsi_torch.mac2d.execution import build_driver
    from afsi_torch.mac2d.checkpoint import save,load
    solid=make_solid('cuda',fused=False)
    grid=ChannelGrid((64,16),(8.,1.61))
    transfers=[TriangleTransfer(grid,solid.geometry,fused=False,mass_backend='pcg',ib_backend=b) for b in ('reference','cuda')]
    f=torch.sin(solid.mesh.X)
    velocity=tuple(torch.sin(grid.coordinates(c,device='cuda')[...,0]) for c in range(2))
    results=[]
    for tr in transfers:
        stencil=tr.prepare(solid.mesh.X)
        assert any((w<0).any() for w in stencil.weights)
        spread,_=tr.spread(f,stencil);gather,_=tr.interpolate(velocity,stencil)
        torch.testing.assert_close(sum((u*d).sum()*grid.volume for u,d in zip(velocity,spread)),(f*gather).sum(),rtol=1e-8,atol=1e-8)
        results.append((spread,gather))
    for a,b in zip(results[0][0],results[1][0]):torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-8)
    torch.testing.assert_close(results[0][1],results[1][1],rtol=1e-8,atol=1e-8)
    settings=dict(dt=1/16000,nx=64,ny=16,rho=1.,mu=.1,fused=False,
                  mass_backend='pcg',warm_start=False,execution_backend='reference',pressure_backend='reference')
    states=[]
    for backend in ('reference','cuda'):
        driver=build_driver(solid,dict(settings,ib_backend=backend),'cuda')
        state=driver.initialize()
        state=replace(state,force=f)
        state,_=driver.step(state)
        states.append(state)
    torch.testing.assert_close(states[0].x,states[1].x,rtol=1e-8,atol=1e-10)
    save(tmp_path/'c.npz',solid,states[-1],dict(settings,ib_backend='cuda'),{})
    _,_,restored,_=load(tmp_path/'c.npz','cuda')
    assert restored['ib_backend']=='cuda'

@GPU
def test_fem_nodal_lattice_x_fast_order_and_power():
    from afsi_torch import ib
    grid=ib.UniformGrid((11,13,15),(.2,.3,.4))
    x=torch.tensor([[.6,1.1,1.8],[.83,1.37,2.1]],device='cuda',dtype=torch.float64)
    a=ib.prepare_stencil(x,grid)
    b=ib.prepare_stencil(x,grid,backend='cuda')
    u=torch.sin(grid.coordinates(device='cuda'));f=torch.cos(x)
    torch.testing.assert_close(ib.interpolate(u,a),ib.interpolate(u,b),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(ib.spread_density(f,a),ib.spread_density(f,b),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close((u*ib.spread_load(f,b)).sum(),(ib.interpolate(u,b)*f).sum(),rtol=1e-12,atol=1e-12)

@GPU
def test_native_transfer_nondefault_stream_graph_and_empty():
    cuda_ib.build()
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        ids=torch.tensor([[0,1],[1,2]],device='cuda');w=torch.ones((2,2),device='cuda',dtype=torch.float64)
        f=torch.ones((2,1),device='cuda',dtype=torch.float64)
        cuda_ib.indexed_spread(f,ids,w,3)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            spread=cuda_ib.indexed_spread(f,ids,w,3)
            gathered=cuda_ib.indexed_gather(spread,ids,w)
        f.fill_(2)
        graph.replay()
    stream.synchronize()
    torch.testing.assert_close(gathered,f.new_full((2,1),6))
    empty=cuda_ib.indexed_spread(f[:0],ids[:0],w[:0],3)
    assert empty.count_nonzero()==0

@pytest.mark.parametrize('preset,kind',[
    ('ideal_lv_fsi/configs/mac_gpu.json','ideal-lv-mac'),
    ('ideal_lv_fsi/configs/fem.json','ideal-lv-fem'),
    ('ideal_valve_fsi/configs/mac_gpu.json','ideal-valve-mac'),
    ('real_lv_fsi/configs/active_cuda.json','real-lv'),
    ('real_lv_fsi/configs/diastole_cuda.json','real-lv')])
def test_single_json_runner_dispatches_native_presets(preset,kind,monkeypatch):
    from demo import run
    from afsi_torch.config import load_config,LVSimulationConfig,LVFEMSimulationConfig,ValveSimulationConfig
    from afsi_torch.paper_lv import PaperLVConfig
    classes={'ideal-lv-mac':LVSimulationConfig,'ideal-lv-fem':LVFEMSimulationConfig,'ideal-valve-mac':ValveSimulationConfig,'real-lv':PaperLVConfig}
    path=ROOT/'demo'/preset
    cfg=load_config(path,classes[kind])
    backend=cfg.ib_csr_contraction_backend if kind=='real-lv' else cfg.ib_backend if kind=='ideal-lv-fem' else cfg.execution.ib_backend
    assert backend=='cuda'
    calls=[]
    monkeypatch.setattr(run.runpy,'run_path',lambda entry,**kw:dict(main=lambda args:calls.append((entry,args))))
    run.main([str(path),'--device','cuda','--output','results/custom'])
    assert calls[0][1]==['--config',str(path.resolve()),'--device','cuda','--output','results/custom']
    if kind=='real-lv':assert cfg.time.end_time==1.6 and cfg.material.a==2400

def test_json_demo_kind_cannot_select_wrong_solver(tmp_path):
    from afsi_torch.config import load_config,LVSimulationConfig
    p=tmp_path/'wrong.json';p.write_text(json.dumps({'demo':'real-lv'}))
    with pytest.raises(ValueError,match='does not match'):load_config(p,LVSimulationConfig)

@GPU
@pytest.mark.parametrize('fluid',['mac','fem'])
def test_loaded_ideal_lv_native_coupling_and_checkpoint(tmp_path,fluid):
    pytest.importorskip('gmsh')
    from afsi_torch.afsi337 import generated_model
    from afsi_torch.mac.checkpoint import save_mac,load_mac
    from afsi_torch.cycle_checkpoint import save_cycle,load_cycle
    model=generated_model(mesh_size=.4,device='cuda',basal_constraint='radial',beta=5e6)
    states=[]
    for backend in ('reference','cuda'):
        if fluid=='mac':
            from afsi_torch.mac.execution import build_driver
            settings=dict(fluid_cells=16,box_length=5.,dt=5e-5,rho=1.,mu=1.,interaction_degree=None,
                execution_backend='fused',solid_backend='reference',mass_backend='pcg',
                coupling_backend='reference',ib_backend=backend)
            driver=build_driver(model,settings,'cuda')
            initial=driver.initialize(model.mesh.X)
            initial=replace(initial,force=model.force(initial.x,.001),force_time=.001)
            state,_=driver.step(initial)
        else:
            from functools import partial
            from afsi_torch import ib
            from afsi_torch.fluid import create_box,prepare_operators,ChorinSolver
            from afsi_torch.coupling import ExplicitIBStepper
            mesh=create_box((8,8,8),(5.,5.,5.),device='cuda')
            flow=ChorinSolver(prepare_operators(mesh),dt=5e-5)
            driver=ExplicitIBStepper(flow,model.force,model.validate,
                stencil_factory=partial(ib.prepare_stencil,backend=backend))
            initial=driver.initialize(model.mesh.X)
            initial=replace(initial,force=model.force(initial.x,.001),force_time=.001)
            state=driver.step(initial).state
            settings=dict(dt=5e-5,ib_backend=backend)
        states.append(state)
    for name in ('x','force','pressure'):
        torch.testing.assert_close(getattr(states[0],name),getattr(states[1],name),rtol=2e-7,atol=2e-7)
    velocities=zip(states[0].velocity,states[1].velocity) if fluid=='mac' else [(states[0].velocity,states[1].velocity)]
    for a,b in velocities:
        torch.testing.assert_close(a,b,rtol=2e-7,atol=2e-9)
    path=tmp_path/'checkpoint.npz'
    if fluid=='mac':
        save_mac(path,model,states[-1],settings,{})
        restored_model,restored,settings,_=load_mac(path,'cuda')
        assert build_driver(restored_model,settings,'cuda').transfer.ib_backend=='cuda'
    else:
        save_cycle(path,model,states[-1],model.mesh.X,settings,{})
        _,restored,_,settings,_=load_cycle(path,'cuda')
    assert settings['ib_backend']=='cuda'
    torch.testing.assert_close(restored.x,states[-1].x,rtol=0,atol=0)
