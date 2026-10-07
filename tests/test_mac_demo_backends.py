"""Demo wiring, checkpoint persistence and rollback of solid/mass execution."""
import pytest
import torch
from afsi_torch.mac.checkpoint import load_mac,save_mac
from afsi_torch.mac.execution import build_driver
from test_mac_execution import DEVICES


def assert_states(a,b):
    for name in ('x','force','pressure'):
        torch.testing.assert_close(getattr(a,name),getattr(b,name),rtol=1e-7,atol=1e-8)
    for u,v in zip(a.velocity,b.velocity):
        torch.testing.assert_close(u,v,rtol=1e-7,atol=1e-9)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('solid_backend,mass_backend',[
    ('pointwise','pcg'),('reference','graph'),('pointwise','graph')])
def test_demo_backends_match_existing_fused(tmp_path,device,solid_backend,mass_backend):
    pytest.importorskip('gmsh')
    from examples.lv_mac import run
    common=dict(device=device,mesh_size=.4,fluid_cells=16,warm_start=True,
        pressure_backend='fused',execution_backend='fused',end_time=.0004,
        log_every=4,checkpoint_every=4)
    run(**common,output=tmp_path/'reference')
    report=run(**common,output=tmp_path/'candidate',solid_backend=solid_backend,mass_backend=mass_backend)
    _,a,_,_=load_mac(tmp_path/'reference'/'checkpoint.npz',device)
    _,b,settings,_=load_mac(tmp_path/'candidate'/'checkpoint.npz',device)
    assert_states(a,b)
    assert settings['solid_backend']==solid_backend
    assert settings['mass_backend']==mass_backend
    assert report['mass_cuda_graphs']==(4 if mass_backend=='graph' and device=='cuda' else 0)


@pytest.mark.parametrize('device',DEVICES)
def test_demo_cli_resume_and_explicit_rollback(tmp_path,device):
    pytest.importorskip('gmsh')
    from demo.ideal_lv_fsi.run_mac import main
    from examples.lv_mac import run
    output=tmp_path/'demo'
    report=main(['--device',device,'--ib-backend','reference','--mesh-size','.4','--fluid-cells','16',
        '--warm-start','--pressure-backend','fused','--execution-backend','fused',
        '--solid-backend','pointwise','--mass-backend','graph','--end-time','.0002',
        '--log-every','2','--checkpoint-every','2','--output',str(output)])
    assert report['completed']
    report=main(['--device',device,'--resume',str(output/'checkpoint.npz'),
        '--end-time','.0004','--log-every','2','--checkpoint-every','2'])
    assert report['solid_backend']=='pointwise' and report['mass_backend']=='graph'
    report=main(['--device',device,'--resume',str(output/'checkpoint.npz'),
        '--solid-backend','reference','--mass-backend','pcg','--end-time','.0005'])
    assert report['solid_backend']=='reference' and report['mass_backend']=='pcg'
    assert report['segments'][-2]['mass_backend']=='graph'
    assert report['segments'][-1]['mass_backend']=='pcg'
    run(device=device,mesh_size=.4,fluid_cells=16,warm_start=True,
        pressure_backend='fused',execution_backend='fused',end_time=.0005,
        output=tmp_path/'uninterrupted',log_every=2,checkpoint_every=2)
    _,a,_,_=load_mac(tmp_path/'uninterrupted'/'checkpoint.npz',device)
    _,b,_,_=load_mac(output/'checkpoint.npz',device)
    assert_states(a,b)


def test_old_checkpoint_restores_existing_defaults(tmp_path):
    pytest.importorskip('gmsh')
    from examples.lv_mac import run
    output=tmp_path/'old'
    run(device='cpu',mesh_size=.4,fluid_cells=16,end_time=.0001,
        output=output,log_every=2,checkpoint_every=2)
    model,state,settings,progress=load_mac(output/'checkpoint.npz')
    settings.pop('solid_backend')
    settings.pop('mass_backend')
    save_mac(output/'checkpoint.npz',model,state,settings,progress)
    report=run(device='cpu',resume=output/'checkpoint.npz',end_time=.0002)
    assert report['solid_backend']=='reference' and report['mass_backend']=='pcg'


@pytest.mark.parametrize('settings',[
    dict(execution_backend='torch',solid_backend='pointwise'),
    dict(execution_backend='torch',mass_backend='graph'),
    dict(execution_backend='fused',solid_backend='invalid'),
    dict(execution_backend='fused',mass_backend='invalid')])
def test_backend_validation_precedes_model_access(settings):
    with pytest.raises(ValueError,match='backend'):
        build_driver(None,settings,'cpu')
