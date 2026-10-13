"""Native data must reach the actual FEM demo and survive resume unchanged."""
from dataclasses import replace
import json
import numpy as np
import pytest
import torch
from afsi_torch.afsi337 import generated_model
from afsi_torch.config import LVFEMSimulationConfig,FEMFluidConfig,TimeConfig,OutputConfig,save_config
from afsi_torch.cycle_checkpoint import load_cycle
from demo.ideal_lv_fsi.run import main

DEVICES=['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))]


@pytest.mark.parametrize('device',DEVICES)
def test_native_config_and_cli_preserve_mesh_fibers_loaded_step_and_resume(tmp_path,monkeypatch,device):
    pytest.importorskip('gmsh')
    model=generated_model(mesh_size=.4)
    mesh=model.mesh
    source=tmp_path/'solid.npz'
    np.savez_compressed(source,metadata=json.dumps(dict(schema=1,producer='afsi337-solid-export',units='cm')),
        X=mesh.X.numpy(),cells=mesh.cells.numpy(),tagged_vertices=mesh.faces[:,:3].numpy(),
        facet_tags=mesh.facet_tags.numpy(),fiber=model.fibers.fiber.numpy()*1.01,
        sheet=model.fibers.sheet.numpy()*.99,volume_points=model.volume_quadrature[0].numpy(),
        volume_weights=model.volume_quadrature[1].numpy(),surface_points=model.surface_quadrature[0].numpy(),
        surface_weights=model.surface_quadrature[1].numpy())
    def no_regeneration(*a,**k):raise AssertionError('native import must not regenerate geometry')
    monkeypatch.setattr('afsi_torch.afsi337.generated_model',no_regeneration)
    cfg=replace(LVFEMSimulationConfig(),solid_input=str(source),time=TimeConfig(5e-5,15e-5),
        fluid=FEMFluidConfig((8,8,8)),ib_backend='cuda' if device=='cuda' else 'reference',
        output=OutputConfig(3,3,3,False),history_every=1)
    config=tmp_path/'case.json';save_config(config,cfg)
    report=main(['--config',str(config),'--device',device,'--output',str(tmp_path/'run')])
    restored,state,*_=load_cycle(tmp_path/'run/checkpoint.npz',device)
    torch.testing.assert_close(restored.mesh.X.cpu(),mesh.X,rtol=0,atol=0)
    torch.testing.assert_close(restored.fibers.fiber.cpu(),model.fibers.fiber*1.01,rtol=0,atol=0)
    torch.testing.assert_close(restored.fibers.sheet.cpu(),model.fibers.sheet*.99,rtol=0,atol=0)
    assert torch.linalg.vector_norm(state.force)>0 and torch.linalg.vector_norm(state.velocity)>0
    assert report['source_alignment']['solid_mesh_and_fibers_matched']
    assert report['configuration']['solid_input']==str(source)
    assert report['discretization']['viscosity']=='implicit backward Euler'
    source.unlink()  # A checkpoint is self contained; resume must not read the original source.
    resumed=main(['--resume',str(tmp_path/'run/checkpoint.npz'),'--device',device,'--end-time','.0002'])
    assert resumed['accepted_steps']==4 and resumed['configuration']['solid_input']==str(source)


def test_native_config_rejects_invalid_path_type():
    for path in ('',False,[],3):
        with pytest.raises(ValueError,match='solid_input'):
            LVFEMSimulationConfig(solid_input=path)
