"""The actual stable demo wires JSON, CLI, native input and restart correctly."""
from dataclasses import replace
import json
import numpy as np
import pytest
import torch
from afsi_torch.config import LVFEMSimulationConfig,FEMFluidConfig,TimeConfig,OutputConfig,save_config,load_config
from afsi_torch.cycle_checkpoint import load_cycle
from demo.AFSI337_stable.run import main, CONFIG

DEVICES=['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))]


def test_portable_config_matches_demo_and_zero_default(tmp_path):
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'demo/AFSI337_stable/config.json'
    config=load_config(path,CONFIG)
    assert config.fluid.grad_div_alpha==1. and config.solid_input is None
    assert LVFEMSimulationConfig().fluid.grad_div_alpha==0.
    save_config(tmp_path/'roundtrip.json',config)
    assert load_config(tmp_path/'roundtrip.json',LVFEMSimulationConfig)==config


@pytest.mark.parametrize('device',DEVICES)
def test_demo_accepted_state_resume_and_cli_stabilization_override(tmp_path,device):
    pytest.importorskip('gmsh')
    config=replace(CONFIG,time=TimeConfig(5e-5,2e-4),fluid=FEMFluidConfig((8,8,8),mu=.8,grad_div_alpha=.1),
        geometry=replace(CONFIG.geometry,mesh_size=.4),ib_backend='cuda' if device=='cuda' else 'reference',
        history_every=1,output=OutputConfig(4,4,4,False))
    path=tmp_path/'config.json';save_config(path,config)
    args=['--config',str(path),'--device',device,'--grad-div-alpha','.7']
    report=main(args+['--end-time','.00015','--output',str(tmp_path/'resumed')])
    assert report['demo']=='AFSI337_stable'
    assert report['settings']['grad_div_alpha']==.7
    assert report['settings']['grad_div_gamma']==pytest.approx(.56)
    assert report['configuration']['fluid']['grad_div_alpha']==.7
    assert report['stabilization']['placement']=='implicit tentative velocity'
    assert any('grad-div' in s for s in report['source_alignment']['differences'])
    resumed=main(['--resume',str(tmp_path/'resumed/checkpoint.npz'),'--device',device,'--end-time','.0002','--no-vtk'])
    main(args+['--output',str(tmp_path/'full')])
    _,a,_,settings,_=load_cycle(tmp_path/'resumed/checkpoint.npz',device)
    _,b,*_=load_cycle(tmp_path/'full/checkpoint.npz',device)
    for name in ('x','velocity','pressure','force'):
        torch.testing.assert_close(getattr(a,name),getattr(b,name),rtol=1e-8,atol=2e-10)
    assert resumed['configuration']['fluid']['grad_div_alpha']==.7
    assert settings['grad_div_alpha']==.7
    with pytest.raises(SystemExit):
        main(['--resume',str(tmp_path/'full/checkpoint.npz'),'--grad-div-alpha','1','--device',device])


@pytest.mark.parametrize('device',DEVICES)
def test_stable_entry_resumes_legacy_checkpoint_without_enabling_stabilization(tmp_path,device):
    pytest.importorskip('gmsh')
    from demo.ideal_lv_fsi.run_fem import main as baseline
    config=replace(CONFIG,time=TimeConfig(5e-5,1e-4),fluid=FEMFluidConfig((8,8,8)),
        geometry=replace(CONFIG.geometry,mesh_size=.4),ib_backend='reference',
        history_every=1,output=OutputConfig(2,2,2,False))
    path=tmp_path/'case.json';save_config(path,config)
    baseline(['--config',str(path),'--device',device,'--output',str(tmp_path/'legacy')])
    # Produce a real old-schema checkpoint and recompute its integrity hash.
    import hashlib
    checkpoint=tmp_path/'legacy/checkpoint.npz'
    with np.load(checkpoint,allow_pickle=False) as archive:
        arrays={key:archive[key] for key in archive.files}
    metadata=json.loads(str(arrays.pop('metadata')))
    metadata.pop('sha256')
    metadata['settings'].pop('grad_div_alpha');metadata['settings'].pop('grad_div_gamma')
    digest=hashlib.sha256(json.dumps(metadata,sort_keys=True,allow_nan=False).encode())
    for name in sorted(arrays):
        digest.update(name.encode());digest.update(arrays[name].tobytes())
    metadata['sha256']=digest.hexdigest()
    np.savez_compressed(checkpoint,metadata=json.dumps(metadata),**arrays)
    report=main(['--resume',str(checkpoint),'--device',device,'--end-time','.00015','--no-vtk'])
    assert report['settings']['grad_div_alpha']==0. and report['stabilization']['type']=='none'
