"""Radial-only P2 base: UFL projector, GPU execution and restart physics."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from afsi_torch import boundary as bd
from afsi_torch.afsi337 import generated_model
from afsi_torch.lv_model import LVSolid
from afsi_torch.mac.solid_execution import SolidExecution
from afsi_torch.mac.solid_pointwise import PointwiseSolidExecution

DEVICES=['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))]

@pytest.fixture(scope='module',params=DEVICES)
def model(request):
    pytest.importorskip('gmsh')
    return generated_model(mesh_size=.4,device=request.param,beta=5e6,basal_constraint='radial')

def test_radial_affine_motion_is_free_and_axial_tangential_motion_is_penalized(model):
    X=model.mesh.X
    R=X-X.new_tensor(model.mesh.config.center)
    R=R*X.new_tensor([0.,1.,1.])
    radial_x=X+.012*R
    torch.testing.assert_close(model.basal_force(radial_x),torch.zeros_like(X),rtol=0,atol=5e-9)
    assert model.basal_energy(radial_x).item()<1e-20
    u=X.new_tensor([.01,0.,0.]).expand_as(X)
    axial=model.basal_force(X+u)
    expected=-model.beta*model.base.reference_weights.sum()*X.new_tensor([.01,0.,0.])
    torch.testing.assert_close(axial.sum(0),expected,rtol=1e-12,atol=1e-8)
    assert ((X+u-X)*axial).sum()<0
    tangent=.01*torch.linalg.cross(X.new_tensor([1.,0.,0.]).expand_as(X),R)
    torch.testing.assert_close(model.basal_constraint(X+tangent),-bd.interpolate(tangent,model.base),rtol=1e-11,atol=2e-14)
    assert model.basal_energy(X+tangent)>0

def test_constraint_matches_real_lv_ufl_after_axis_rotation_and_has_energy_gradient(model):
    X=model.mesh.X
    x=X+.002*torch.sin(1.7*X)
    u=bd.interpolate(x,model.base)-model.base.reference_positions
    R=model.base.reference_positions[...,1:]-X.new_tensor(model.mesh.config.center[1:])
    direction=(R*u[...,1:]).sum(-1,keepdim=True)/R.square().sum(-1,keepdim=True)
    expected=torch.cat((-u[...,:1],direction*R-u[...,1:]),-1)
    torch.testing.assert_close(model.basal_constraint(x),expected,rtol=2e-12,atol=2e-14)
    torch.testing.assert_close(model.basal_force(x),-torch.func.grad(model.basal_energy)(x),rtol=2e-11,atol=2e-8)
    torch.testing.assert_close(((x-X)*model.basal_force(x)).sum(),-2*model.basal_energy(x),rtol=2e-11,atol=2e-8)
    # The generated model also supports its unrotated z-long-axis geometry.
    perm=torch.tensor([1,2,0],device=X.device)
    config=replace(model.mesh.config,long_axis='z',center=tuple(model.mesh.config.center[i] for i in [1,2,0]))
    rotated=LVSolid(replace(model.mesh,X=X[:,perm],config=config),beta=model.beta,basal_constraint='radial',
                    surface_quadrature=model.surface_quadrature)
    torch.testing.assert_close(rotated.basal_force(x[:,perm]),model.basal_force(x)[:,perm],rtol=2e-11,atol=2e-8)

@pytest.mark.parametrize('execution',[SolidExecution,PointwiseSolidExecution])
def test_compiled_and_pointwise_force_include_the_same_basal_term(model,execution):
    x=model.mesh.X+.001*torch.sin(model.mesh.X)
    reference=model.force(x,.03)
    optimized=execution(model)
    torch.testing.assert_close(optimized.force(x,.03),reference,rtol=5e-10,atol=3e-7)
    force,geometry=optimized.force_with_geometry(x,.03)
    assert geometry[-1].all()
    torch.testing.assert_close(force,reference,rtol=5e-10,atol=3e-7)

@pytest.mark.parametrize('kind',['mac','fem'])
def test_checkpoint_preserves_radial_and_legacy_checkpoint_keeps_full_spring(model,tmp_path,kind):
    from afsi_torch.mac.checkpoint import save_mac,load_mac,digest
    from afsi_torch.mac.coupling import MACState
    from afsi_torch.cycle_checkpoint import save_cycle,load_cycle
    from afsi_torch.coupling import CoupledState
    from afsi_torch.mac.grid import MACGrid
    X=model.mesh.X
    x=X+.0003*torch.sin(X)
    path=tmp_path/'radial.npz'
    settings=dict(dt=5e-5)
    def save(m,destination):
        if kind=='mac':
            grid=MACGrid((4,4,4),(5.,)*3)
            state=MACState(1,5e-5,x,grid.zeros(device=X.device,dtype=X.dtype),X.new_zeros(grid.shape),m.force(x,0.),0.)
            save_mac(destination,m,state,settings,{})
        else:
            state=CoupledState(1,5e-5,x,X.new_zeros((7,3)),X.new_zeros(3),m.force(x,0.),0.)
            save_cycle(destination,m,state,X,settings,{})
    load=lambda p:(load_mac(p,X.device) if kind=='mac' else load_cycle(p,X.device))
    save(model,path)
    restored,state,*_=load(path)
    assert restored.basal_constraint_mode=='radial' and restored.beta==5e6
    torch.testing.assert_close(restored.basal_force(state.x),model.basal_force(x),rtol=0,atol=1e-8)
    legacy=LVSolid(model.mesh,loads=model.loads,parameters=model.parameters,beta=5e5,
        fibers=model.fibers,volume_quadrature=model.volume_quadrature,surface_quadrature=model.surface_quadrature,
        fiber_metadata=model.fiber_metadata)
    legacy_path=tmp_path/'legacy.npz'
    save(legacy,legacy_path)
    with np.load(legacy_path,allow_pickle=False) as archive:
        arrays={k:archive[k] for k in archive.files}
    metadata=json.loads(str(arrays.pop('metadata')))
    metadata.pop('sha256');metadata.pop('basal_constraint')
    metadata['sha256']=digest(metadata,arrays)
    np.savez_compressed(legacy_path,metadata=json.dumps(metadata),**arrays)
    restored,*_=load(legacy_path)
    assert restored.basal_constraint_mode=='spring' and restored.beta==5e5
    torch.testing.assert_close(restored.basal_force(x),bd.spring_force(x,restored.base,5e5),rtol=0,atol=0)

@pytest.mark.parametrize('name',['mac_gpu.json','fem.json'])
def test_public_ideal_lv_presets_select_radial_base(name):
    from afsi_torch.config import load_config,LVSimulationConfig,LVFEMSimulationConfig
    path=Path(__file__).resolve().parents[1]/'demo/ideal_lv_fsi/configs'/name
    cfg=load_config(path,LVFEMSimulationConfig if name=='fem.json' else LVSimulationConfig)
    assert cfg.basal_constraint=='radial' and cfg.beta==5e6

@pytest.mark.parametrize('kind',['mac','fem'])
def test_radial_demo_run_and_resume_retain_configuration(tmp_path,kind):
    from afsi_torch.config import LVSimulationConfig,LVFEMSimulationConfig,TimeConfig,OutputConfig,FluidConfig,FEMFluidConfig
    from afsi_torch.simulation.lv_mac import run as run_mac
    from afsi_torch.simulation.lv_fem import run as run_fem
    cls=LVSimulationConfig if kind=='mac' else LVFEMSimulationConfig
    fluid=FluidConfig((16,)*3,(5.,)*3) if kind=='mac' else FEMFluidConfig((8,)*3,(5.,)*3)
    config=replace(cls(),basal_constraint='radial',beta=5e6,time=TimeConfig(5e-5,1e-4),fluid=fluid,
        geometry=replace(cls().geometry,mesh_size=.4),output=OutputConfig(2,2,2,False))
    run=run_mac if kind=='mac' else run_fem
    extra={} if kind=='mac' else dict(profile='afsi337',history_every=1)
    report=run(device='cpu',output=tmp_path/'run',case_config=config,**extra)
    assert report['configuration']['basal_constraint']=='radial'
    assert report['configuration']['beta']==5e6
    resumed=run(device='cpu',resume=tmp_path/'run/checkpoint.npz',end_time=1.5e-4,**extra)
    assert resumed['configuration']['basal_constraint']=='radial' and resumed['accepted_steps']==3
