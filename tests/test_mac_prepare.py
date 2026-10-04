"""Direct P1/shared Peskin preparation: precision, support and paired geometry."""
from dataclasses import replace
import json
import os
import numpy as np
import pytest
import torch
from test_adaptive_p1_transfer import transfer
from test_real_lv import real_case,DEVICES
from test_mac_shared_pressure import warm_config
from test_mac_implicit import settings
from afsi_torch.mac.adaptive_transfer import AdaptiveP1Transfer,InteractionQuadratureOptions
from afsi_torch.mac.adaptive_transfer import InteractionRule,QuadratureGroup
from afsi_torch.mac.shared_stencil import SharedStencil
from afsi_torch.mac.grid import MACGrid
from afsi_torch.mac.execution import build_driver
from afsi_torch.real_lv import imported_model


def direct_tables(t,x,rule):
    """Actual kernel interpreter or native CUDA, with group offsets and tails."""
    triton=pytest.importorskip('triton')
    from afsi_torch.mac._triton_prepare import _prepare_group
    base=torch.empty((2,rule.point_count,3),device=x.device,dtype=torch.int64)
    phi=x.new_empty((2,rule.point_count,3,4))
    invalid=torch.zeros((),device=x.device,dtype=torch.int32)
    offset=0
    for group in rule.groups:
        ne,nq=group.cells.shape[0],group.values.shape[0]
        _prepare_group[(triton.cdiv(ne*nq*3,128),)](x,group.cells,group.values,
            t.origin,t.spacing,t.limits,base,phi,invalid,ne,nq,rule.point_count,offset,128,
            num_warps=4,enable_fp_fusion=False)
        offset+=ne*nq
    return SharedStencil(base,phi),invalid.item()


@pytest.mark.parametrize('family',['conical','xiao-gimbutas'])
@pytest.mark.parametrize('device',DEVICES)
def test_direct_kernel_tables_fields_support_and_immutable_rules(family,device):
    if device=='cpu' and os.environ.get('TRITON_INTERPRET')!='1':
        pytest.skip('actual kernel CPU test requires TRITON_INTERPRET=1')
    if family=='xiao-gimbutas':
        pytest.importorskip('basix')
    X,old=transfer(device,rule_family=family)
    grid=MACGrid((16,)*3,(16.,20.,24.),(-1.3,-2.1,-.7))
    t=AdaptiveP1Transfer(grid,old.geometry,
        quadrature_options=InteractionQuadratureOptions(mode='adaptive',rule_family=family,stencil_backend='shared'),fused=False)
    x=X+X.new_tensor([.137,.219,-.073])
    ref=t.prepare(x)
    fast,invalid=direct_tables(t,x,ref.rule)
    assert invalid==0
    torch.testing.assert_close(fast.base,ref.base,rtol=0,atol=0)
    torch.testing.assert_close(fast.phi,ref.phi,rtol=2e-12,atol=2e-14)
    torch.testing.assert_close(fast.phi.sum(-1),torch.ones_like(fast.phi[...,0]),rtol=0,atol=5e-16)
    assert fast.storage_bytes==ref.storage_bytes
    force=torch.sin(1.7*t._points(x,ref.rule))
    a,b=t.spread_grid(force,ref),t.spread_grid(force,fast)
    velocity=tuple(torch.cos(grid.coordinates(c,device=device)[...,c]) for c in range(3))
    for u,v in zip(a,b):
        torch.testing.assert_close(u,v,rtol=2e-11,atol=2e-12)
    torch.testing.assert_close(t.gather_grid(velocity,ref),t.gather_grid(velocity,fast),rtol=2e-11,atol=2e-12)
    cached=fast.phi.clone()
    changed=x.clone(); changed[5,0]+=.6
    other=t.prepare(changed)
    direct_tables(t,changed,other.rule)
    torch.testing.assert_close(fast.phi,cached,rtol=0,atol=0)
    assert direct_tables(t,x-X.new_tensor([10.,0.,0.]),ref.rule)[1]==1
    nonfinite=x.clone(); nonfinite[0,0]=float('nan')
    assert direct_tables(t,nonfinite,ref.rule)[1]==1


@pytest.mark.parametrize('device',DEVICES)
def test_direct_kernel_near_integer_and_half_lattice_edges(device):
    if device=='cpu' and os.environ.get('TRITON_INTERPRET')!='1':
        pytest.skip('actual kernel CPU test requires TRITON_INTERPRET=1')
    X,t=transfer(device,stencil_backend='shared')
    for shift in (0.,.5,-2e-15,2e-15):
        x=X.clone()
        x[:4]=X.new_tensor([[3.,4.,5.],[3.5,4.5,5.5],[6.,7.,8.],[7.5,8.5,9.5]])+shift
        group=QuadratureGroup(t.geometry.cells[:1],torch.eye(4,device=device,dtype=X.dtype),X.new_ones((1,4)))
        rule=InteractionRule((group,),torch.zeros(1,device=device,dtype=torch.int64),4)
        fast,invalid=direct_tables(t,x,rule)
        assert invalid==0
        ref=t.from_points(x[:4])
        # Near an integer, equally valid shifted zero-weight links may differ;
        # compare the resulting fields instead of requiring identical indices.
        velocity=tuple(torch.sin(t.grid.coordinates(c,device=device)[...,c]) for c in range(3))
        reference=SharedStencil(ref.base,ref.phi)
        torch.testing.assert_close(t.gather_grid(velocity,fast),t.gather_grid(velocity,reference),rtol=2e-12,atol=2e-13)
        a,b=t.spread_grid(torch.sin(x[:4]),fast),t.spread_grid(torch.sin(x[:4]),reference)
        for u,v in zip(a,b):
            torch.testing.assert_close(u,v,rtol=2e-12,atol=2e-13)


@pytest.mark.parametrize('device',DEVICES)
def test_prepare_backend_active_coupled_step_work_and_saved_config(real_case,device):
    pytest.importorskip('basix')
    cfg=warm_config(real_case)
    fast_cfg=replace(cfg,interaction_quadrature=replace(cfg.interaction_quadrature,prepare_backend='triton'))
    model=imported_model(cfg,device)
    base,fast=build_driver(model,settings(cfg),device),build_driver(model,settings(fast_cfg),device)
    state=base.initialize(model.mesh.X)
    state=replace(state,step=6000,time=.6,force_time=.6,force=model.force(state.x,.6),
        previous_advection=base.flow.advection(state.velocity))
    expected,_=base.step(state)
    result,info=fast.step(state)
    assert info['nonlinear']['residual_norm']<=info['nonlinear']['tolerance']
    torch.testing.assert_close(result.x,expected.x,rtol=2e-9,atol=2e-11)
    for a,b in zip(result.velocity,expected.velocity):
        torch.testing.assert_close(a,b,rtol=2e-6,atol=2e-9)
    assert info['power_error']/max(abs(info['solid_power']),abs(info['fluid_power']),1.)<1e-8
    summary=fast.transfer.quadrature_summary()
    assert summary['prepare_execution']==('triton' if device=='cuda' else 'torch')


def test_prepare_cli_checkpoint_benchmark_and_old_metadata(real_case,tmp_path):
    pytest.importorskip('basix')
    from demo.real_lv_fsi.run_mac import main
    from afsi_torch.simulation.real_lv_mac import run
    from afsi_torch.real_lv_checkpoint import load_real_lv
    from validation.benchmark_real_lv_schemes import benchmark
    from afsi_torch.config import _decode
    cfg=warm_config(real_case)
    cfg=replace(cfg,interaction_quadrature=replace(cfg.interaction_quadrature,prepare_backend='triton'))
    output=tmp_path/'direct'
    run(case_config=cfg,device='cpu',output=output)
    checkpoint=output/'checkpoint.npz'
    _,_,_,_,restored=load_real_lv(checkpoint)
    assert restored.interaction_quadrature==cfg.interaction_quadrature
    assert run(resume=checkpoint,device='cpu',end_time=3e-4)['completed']
    original=checkpoint.read_bytes()
    report=benchmark(checkpoint,device='cpu',schemes=('cnab-semiimplicit',),
        nonlinear_solvers=('anderson-newton',),execution_variants=('shared-warm','prepare-warm'),
        warmup=0,steps=1,profile=True)
    prefix='cnab-semiimplicit/anderson-newton/'
    for case in report['cases'].values():
        assert case['completed'] and 'profile_failure' not in case
        assert 'ib_rule_selection' in case['phases']
        assert 'ib_point_coordinates' in case['phases']  # explicit CPU fallback
    assert report['cases'][prefix+'prepare-warm']['interaction_quadrature']['prepare_execution']=='torch'
    assert report['execution_comparisons'][prefix+'prepare-warm']['reference']==prefix+'shared-warm'
    assert checkpoint.read_bytes()==original
    from afsi_torch.mac.checkpoint import digest
    with np.load(checkpoint,allow_pickle=False) as archive:
        data={name:archive[name] for name in archive.files}
    metadata=json.loads(str(data.pop('metadata')))
    metadata.pop('sha256')
    for group in (metadata['config'],metadata['settings']):
        group['interaction_quadrature'].pop('prepare_backend')
    metadata['sha256']=digest(metadata,data)
    legacy=tmp_path/'legacy.npz'
    np.savez_compressed(legacy,metadata=json.dumps(metadata),**data)
    _,_,legacy_settings,_,legacy_config=load_real_lv(legacy)
    assert legacy_config.interaction_quadrature.prepare_backend=='torch'
    assert 'prepare_backend' not in legacy_settings['interaction_quadrature']
    path=tmp_path/'config.json'
    main(['--ib-stencil-backend','shared','--ib-prepare-backend','triton','--stokes-warm-start','--write-config',str(path)])
    config=json.loads(path.read_text())
    assert config['interaction_quadrature']['prepare_backend']=='triton'
    assert _decode(InteractionQuadratureOptions,dict(mode='adaptive',stencil_backend='shared')).prepare_backend=='torch'
    with pytest.raises(ValueError):
        InteractionQuadratureOptions(mode='adaptive',prepare_backend='triton')
