"""Numerical recurrence and ownership checks for the execution experiment."""
import pytest
import torch
from afsi_torch.mac.mass_solver import MassSolver
from afsi_torch.mac.mass_graph import GraphMassSolver
from afsi_torch.mac.solid_pointwise import product3,PointwiseSolidExecution
from afsi_torch.mac.solid_execution import SolidExecution
from afsi_torch.fluid.solvers import SolverOptions
from test_mac_execution import setup,DEVICES
from afsi_torch.mac.transfer import FETransfer


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('recompute',[7,12,40])
def test_graph_mass_recurrence_checks_and_owned_outputs(device,recompute):
    X,g,grid=setup(device)
    t=FETransfer(grid,g)
    opt=SolverOptions(rtol=1e-11,atol=1e-13,recompute_every=recompute,check_every=4,max_iterations=500)
    a,b=MassSolver(t.mass,t.diagonal,opt),GraphMassSolver(t.mass,t.diagonal,opt)
    initial=.1*torch.cos(X)
    owned=[]
    for rhs in (torch.sin(5*X),-torch.sin(5*X),torch.zeros_like(X)):
        expected,ia=a.solve(rhs,initial)
        actual,ib=b.solve(rhs,initial)
        torch.testing.assert_close(actual,expected,rtol=1e-9,atol=1e-10)
        assert ib.iterations==ia.iterations
        assert torch.linalg.vector_norm(rhs-t.mass_action(actual)).item()<=ib.tolerance
        for output,saved in owned:
            torch.testing.assert_close(output,saved,rtol=0,atol=0)
        owned.append((actual,actual.clone()))
    assert bool(b.graphs)==(device=='cuda')
    zero,info=b.solve(torch.zeros_like(X))
    assert info.iterations==0 and zero.count_nonzero()==0
    with pytest.raises(ValueError,match='finite'):
        b.solve(torch.full_like(X,float('nan')))
    bad=GraphMassSolver(-t.mass,t.diagonal,opt)
    with pytest.raises(RuntimeError,match='breakdown'):
        bad.solve(torch.sin(X))
    short=GraphMassSolver(t.mass,t.diagonal,SolverOptions(max_iterations=1,rtol=1e-14))
    with pytest.raises(RuntimeError,match='converge'):
        short.solve(torch.sin(X))


def test_product3_broadcast():
    generator=torch.Generator().manual_seed(817)
    a=torch.randn(2,7,3,3,dtype=torch.float64,generator=generator)
    b=torch.randn(7,3,3,dtype=torch.float64,generator=generator)
    torch.testing.assert_close(product3(a,b),a@b,rtol=2e-14,atol=2e-14)


def test_graph_workspace_fullgraph_trace(monkeypatch):
    import afsi_torch.mac.mass_graph as graph_module
    monkeypatch.setattr(graph_module,'tensor_kernel',lambda function,device:
        torch.compile(function,backend='eager',fullgraph=True,dynamic=False))
    X,g,grid=setup('cpu')
    t=FETransfer(grid,g)
    opt=SolverOptions(rtol=1e-11,atol=1e-13,recompute_every=7,check_every=4)
    a,b=MassSolver(t.mass,t.diagonal,opt),GraphMassSolver(t.mass,t.diagonal,opt)
    for rhs in (torch.sin(5*X),-torch.cos(3*X)):
        expected,ia=a.solve(rhs)
        actual,ib=b.solve(rhs)
        torch.testing.assert_close(actual,expected,rtol=1e-9,atol=1e-10)
        assert ia.iterations==ib.iterations and ib.residual_norm<=ib.tolerance


@pytest.mark.parametrize('device',DEVICES)
def test_pointwise_solid_force_loads_and_mutation(device):
    pytest.importorskip('gmsh')
    from afsi_torch.afsi337 import generated_model
    model=generated_model(mesh_size=.4,device=device)
    a,b=SolidExecution(model),PointwiseSolidExecution(model)
    x=model.mesh.X+.001*torch.sin(2*model.mesh.X)
    for time in (0.,.3,1.5,2.):
        actual=b.force(x,time)
        torch.testing.assert_close(actual,a.force(x,time),rtol=2e-9,atol=2e-8)
        torch.testing.assert_close(actual,model.force(x,time),rtol=2e-9,atol=2e-8)
        x=x+.00001*torch.cos(x)


@pytest.mark.parametrize('device',DEVICES)
def test_four_variants_coupled_replay(tmp_path,monkeypatch,device):
    pytest.importorskip('gmsh')
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'validation'))
    from examples.lv_mac import run
    from validation.benchmark_mac_solid_mass import benchmark
    run(device=device,mesh_size=.4,fluid_cells=16,warm_start=True,
        pressure_backend='fused',output=tmp_path/'checkpoint',end_time=.00015,
        log_every=3,checkpoint_every=3)
    report=benchmark(tmp_path/'checkpoint'/'checkpoint.npz',device=device,steps=3,warmup=3)
    assert report['passed']
    assert set(report['variants'])=={'reference','pointwise','pcg_graph','combined'}


def test_pointwise_fullgraph_trace(monkeypatch):
    pytest.importorskip('gmsh')
    import afsi_torch.mac.solid_execution as execution
    from afsi_torch.afsi337 import generated_model
    graphs=[]
    def backend(gm,inputs):
        graphs.append(gm)
        return gm.forward
    monkeypatch.setattr(execution,'tensor_kernel',lambda function,device:
        torch.compile(function,backend=backend,fullgraph=True,dynamic=False))
    model=generated_model(mesh_size=.4)
    fast=PointwiseSolidExecution(model)
    x=model.mesh.X+.001*torch.sin(model.mesh.X)
    for time in (0.,.3,1.5,2.):
        torch.testing.assert_close(fast.force(x,time),model.force(x,time),rtol=2e-9,atol=2e-8)
    assert len(graphs)==2
