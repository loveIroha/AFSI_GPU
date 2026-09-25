"""Preload persistence, physical load clock, and balanced IB bootstrap."""
from dataclasses import asdict
import json
import numpy as np
import pytest
import torch
from afsi_torch import solid
from afsi_torch.coupling import ExplicitIBStepper
from afsi_torch.fluid import create_box,prepare_operators,ChorinSolver
from afsi_torch.tetrahedron import reference_nodes
from afsi_torch.lv_model import LVSolid,RampLoads,PreloadedLoads
from afsi_torch.preload import save_checkpoint,load_preload
from afsi_torch.units import CGS_UNITS,MMHG_TO_DYN_PER_CM2


@pytest.fixture(params=['cpu',pytest.param('cuda',marks=pytest.mark.skipif(
    not torch.cuda.is_available(),reason='CUDA device unavailable'))])
def device(request):
    return request.param


def test_physical_load_clock_retains_baseline():
    loads=PreloadedLoads(.2,100.,.02,hold_time=.1,ramp_time=.2)
    for time,pressure in [(0.,.2),(.1,.2),(.2,.21),(.3,.22),(1.,.22)]:
        p,t=loads.at(time)
        assert p==pytest.approx(pressure*MMHG_TO_DYN_PER_CM2)
        assert t==100.
    for time in (-1.,float('nan'),float('inf')):
        with pytest.raises(ValueError):
            loads.at(time)


@pytest.mark.parametrize('kwargs',[dict(pressure_mmhg=-1),dict(tension=-1),
    dict(pressure_increment_mmhg=-1),dict(hold_time=-1),dict(ramp_time=0),dict(ramp_time=float('nan'))])
def test_load_validation(kwargs):
    with pytest.raises(ValueError):
        PreloadedLoads(**dict(dict(pressure_mmhg=.2),**kwargs))


def test_prestressed_hold_then_load_increment(device):
    # A deformed P2 solid balanced by a fixed dead nodal load. This tests the
    # transfer/clock, not independent constitutive accuracy (tested elsewhere).
    X=reference_nodes(device=device)*.4+.1
    cells=torch.arange(10,device=device).reshape(1,10)
    geo=solid.prepare_p2(X,cells)
    x0=X*X.new_tensor([1.02,1.,1.])
    internal=lambda x:solid.stress_force(x,geo,3.,5.)
    dead=-internal(x0)
    assert torch.linalg.vector_norm(dead)>1e-5
    # Keep a nonzero residual to detect accidental zeroing or baseline subtraction.
    epsilon=torch.full_like(x0,1e-13)
    force=lambda x,t:internal(x)+dead+epsilon+(1e-3 if t>=.004 else 0.)
    flow=ChorinSolver(prepare_operators(create_box((3,)*3,(3.,)*3,(-1.,)*3,device=device)),dt=.002,mu=.1)
    driver=ExplicitIBStepper(flow,force,lambda x:solid.validate_deformation(x,geo))
    state=driver.initialize_equilibrium(x0,force_tolerance=1e-10)
    torch.testing.assert_close(state.force,epsilon,atol=0,rtol=0)
    assert state.force_time==0. and state.x.data_ptr()!=x0.data_ptr()
    assert state.pressure.count_nonzero()==0 and state.velocity.count_nonzero()==0
    snapshot=X.clone()
    for _ in range(2):
        state=driver.step(state).state
    torch.testing.assert_close(state.x,x0,atol=1e-12,rtol=0)
    # Step 3 evaluates the increment at old time .004; step 4 applies it.
    third=driver.step(state)
    assert third.state.force_time==.004 and third.diagnostics['applied_force_norm_dyn']<1e-10
    fourth=driver.step(third.state)
    assert torch.linalg.vector_norm(fourth.state.x-x0)>1e-10
    torch.testing.assert_close(X,snapshot,atol=0,rtol=0)
    with pytest.raises(ValueError,match='not balanced'):
        ExplicitIBStepper(flow,lambda x,t:internal(x),driver.validate).initialize_equilibrium(x0,force_tolerance=1e-10)
    for tol in (0.,-1.,float('nan')):
        with pytest.raises(ValueError,match='tolerance'):
            driver.initialize_equilibrium(x0,force_tolerance=tol)
    flow.pressure_values.fill_(1.)
    with pytest.raises(ValueError,match='zero pressure gauge'):
        driver.initialize_equilibrium(x0,force_tolerance=1e-10)


@pytest.fixture
def checkpoint(tmp_path):
    pytest.importorskip('gmsh')
    from afsi_torch.geometry import generate_lv,LVConfig
    mesh=generate_lv(LVConfig(mesh_size=1.8))
    model=LVSolid(mesh,loads=RampLoads(pressure_mmhg=0.,tension=0.))
    save_checkpoint(tmp_path/'last_converged.npz',model,mesh.X,1.)
    report=dict(converged=True,history=[dict(load_fraction=1.,residual_norm=0.,tolerance=1e-5)],
        units=CGS_UNITS,solid_config=asdict(mesh.config),material=asdict(model.parameters),beta=model.beta,
        target_pressure_mmhg=0.,target_tension_dyn_per_cm2=0.,nodes=len(mesh.X),cells=len(mesh.cells),
        final=model.diagnostics(mesh.X))
    (tmp_path/'report.json').write_text(json.dumps(report),encoding='utf-8')
    return tmp_path,model,report


def test_checkpoint_roundtrip_without_remeshing(checkpoint,device,monkeypatch):
    path,original,_=checkpoint
    def forbidden(*args,**kwargs):
        raise AssertionError('self-contained checkpoint must not regenerate mesh')
    monkeypatch.setattr('afsi_torch.preload.generate_lv',forbidden)
    model,x,tol,info=load_preload(path,device)
    torch.testing.assert_close(model.mesh.X.cpu(),original.mesh.X,atol=0,rtol=0)
    assert x.device.type==device and model.mesh.X.device.type==device
    assert not info['legacy_tag_reconstruction'] and not info['reference_rebased'] and not info['equilibrium_resolved']
    assert info['recomputed_force_norm_dyn']<=tol
    torch.testing.assert_close(model.fibers.fiber.cpu(),original.fibers.fiber,atol=1e-12,rtol=0)


@pytest.mark.parametrize('damage,match',[
    ('incomplete','fully converged'),('fraction','final load'),('material','different preload models'),
    ('state','fresh force balance'),('fiber','material fields'),('faces','exterior facets'),
    ('legacy_numbering','numbering differs')])
def test_reject_inconsistent_checkpoint(checkpoint,damage,match):
    path,model,report=checkpoint
    with np.load(path/'last_converged.npz',allow_pickle=False) as f:
        arrays={k:f[k] for k in f.files}
    if damage=='incomplete':
        report['converged']=False
    elif damage=='fraction':
        arrays['load_fraction']=.5
    elif damage=='material':
        report['beta']*=2
    elif damage=='state':
        arrays['x']=arrays['x']*1.001
    elif damage=='fiber':
        arrays['fiber']=-arrays['fiber']
    elif damage=='faces':
        arrays['faces']=arrays['faces'][::-1].copy()
    elif damage=='legacy_numbering':
        arrays.pop('metadata')
        arrays['X']=arrays['X']+.001
    np.savez_compressed(path/'last_converged.npz',**arrays)
    (path/'report.json').write_text(json.dumps(report),encoding='utf-8')
    with pytest.raises(ValueError,match=match):
        load_preload(path)


def test_legacy_tag_reconstruction(checkpoint):
    path,original,_=checkpoint
    np.savez_compressed(path/'last_converged.npz',X=original.mesh.X.numpy(),x=original.mesh.X.numpy(),
        cells=original.mesh.cells.numpy(),load_fraction=1.)
    model,x,tol,info=load_preload(path)
    assert info['legacy_tag_reconstruction'] and not info['equilibrium_resolved']
    torch.testing.assert_close(model.mesh.facet_tags,original.mesh.facet_tags,atol=0,rtol=0)
