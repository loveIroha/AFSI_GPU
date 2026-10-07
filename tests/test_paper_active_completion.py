"""Loaded endpoint, cycle rollover, legacy budget recovery and sample diagnostics."""
from dataclasses import replace
import csv
import pytest
import torch
from test_paper_quadrature_budget import small_model, DEVICES
from afsi_torch.mac.paper_coupling import BEIBStepper
from afsi_torch.paper_lv import PaperLVSolid
from afsi_torch.paper_lv_checkpoint import save, load
from afsi_torch.simulation.paper_lv_mac import run


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('step',[6399,7999,15999])
def test_active_peak_and_cycle_endpoint_equations(device,step):
    # Isolated loaded steps exercise the equations/clock at contraction and
    # pressure reset. They are not a complete simulated deformation history.
    model,cfg = small_model()
    if device=='cuda':
        model = PaperLVSolid(model.mesh.to(device),cfg)
    driver = BEIBStepper(model,cfg,device)
    initial = driver.initialize(model.mesh.X)
    x = initial.x+1e-4*(initial.x-initial.x.mean(0))
    time = step*cfg.time.dt
    old = replace(initial,step=step,time=time,x=x,force_time=time,pressure_time=time,
        previous_x=x.clone(),force=model.force(x,time))
    before = x.clone()
    state,info = driver.step(old)
    assert state.time==pytest.approx((step+1)*cfg.time.dt)
    assert state.step==step+1
    nodal,_ = driver.transfer.interpolate(state.velocity,driver.transfer.prepare(old.x))
    residual = state.x-old.x-cfg.time.dt*nodal
    assert torch.linalg.vector_norm(residual).item()<=1.03*info['nonlinear']['tolerance']
    assert info['nonlinear']['residual_norm']<=info['nonlinear']['tolerance']
    torch.testing.assert_close(state.force,model.force(state.x,state.time),rtol=1e-9,atol=1e-8)
    torch.testing.assert_close(old.x,before,rtol=0,atol=0)
    if step==6399:
        assert model.loads.at(state.time)[1]>8e5
    else:
        assert model.loads.at(state.time)==(0.,0.)
    assert model.diagnostics(state.x)['minimum_detF']>0


@pytest.mark.parametrize('protocol,explicit,expected',[
    ('active-cycle',None,22),('active-cycle',8,8),('inflation',None,8)])
def test_resume_default_upgrade_is_active_only_and_preserves_accepted_equations(tmp_path,protocol,explicit,expected):
    model,cfg = small_model()
    cfg = replace(cfg,load_protocol=protocol)
    model = PaperLVSolid(model.mesh,cfg)
    initial = BEIBStepper(model,cfg,'cpu').initialize(model.mesh.X)
    direct,info = BEIBStepper(model,cfg,'cpu').step(initial)
    path = tmp_path/'run'/'checkpoint.npz'
    save(path,model,initial,cfg,dict(elapsed_seconds=0.))
    report = run(resume=path,device='cpu',ib_max_order=explicit)
    _,state,restored,progress = load(path)
    assert restored.interaction_quadrature.max_order==expected
    assert replace(restored,interaction_quadrature=cfg.interaction_quadrature)==cfg
    assert restored.interaction_quadrature.max_points==cfg.interaction_quadrature.max_points
    torch.testing.assert_close(state.x,direct.x,rtol=1e-11,atol=1e-12)
    assert report['last']['nonlinear_residual']<=report['last']['nonlinear_tolerance']
    assert report['last']['ib_prepared_point_count']>0
    rows = list(csv.DictReader((path.parent/'history.csv').open()))
    assert float(rows[-1]['wall_volume_ratio'])>0
    if expected==22:
        assert progress['quadrature_budget_changes'][0]['reason']=='active-resume-supported-order-ceiling'
    else:
        assert 'quadrature_budget_changes' not in progress


def test_sample_geometry_reports_local_fiber_compression_without_altering_pk1():
    model,cfg = small_model()
    X = model.mesh.X
    x = X.clone(); x[:,0] = X[:,0].mean()+.75*(X[:,0]-X[:,0].mean())
    result = model.diagnostics(x)
    assert result['minimum_fiber_stretch']==pytest.approx(.75)
    assert result['maximum_fiber_stretch']==pytest.approx(.75)
    assert result['minimum_active_stretch_multiplier']==pytest.approx(1-4.9*.25)
    assert result['negative_active_multiplier_cell_count']==len(model.mesh.cells)
    assert result['wall_volume_ratio']==pytest.approx(.75)


def test_prepared_quadrature_statistics_do_not_select_a_new_endpoint_rule():
    from test_adaptive_p1_transfer import transfer
    X,t = transfer('cpu',max_order=22)
    assert t.prepared_quadrature_statistics()['point_count']==0
    stencil = t.prepare(X)
    builds = t.quadrature_builds
    stretched = X.clone(); stretched[5,0] += 3
    t.check_configuration_support(stretched)
    stats = t.prepared_quadrature_statistics()
    assert stats['point_count']==stencil.rule.point_count
    assert stats['maximum_order']==3 and stats['high_order_cells']==0
    assert t.quadrature_builds==builds+1
    t.prepared_quadrature_statistics()
    assert t.quadrature_builds==builds+1
