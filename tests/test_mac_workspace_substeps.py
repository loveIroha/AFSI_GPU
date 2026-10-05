"""Owned template storage, unequal AB2 intervals, and coupled retry/restart."""
from dataclasses import replace
from contextlib import nullcontext
import gc
import os
import json
import csv
import pytest
import torch
from test_real_lv import real_case,DEVICES
from test_mac_implicit import settings
from test_mac_shared_pressure import warm_config
from afsi_torch.real_lv import imported_model
from afsi_torch.real_lv_checkpoint import save_real_lv,load_real_lv,refine_checkpoint_dt
from afsi_torch.mac.execution import build_driver
from afsi_torch.mac.cnab import extrapolated_advection,CNABTransportGuardError
from afsi_torch.mac.shared_stencil import SharedStencil
from afsi_torch.mac.stencil_workspace import StencilWorkspace
from afsi_torch.mac.memory import allocator_sample


@pytest.mark.parametrize('device',DEVICES)
def test_workspace_live_stencils_never_overwritten_and_free_capacity_reused(device):
    like=torch.zeros((),device=device,dtype=torch.float64)
    pool=StencilWorkspace(100,slots=2,quantum=8)
    def acquire(n,value):
        b,p,s=pool.acquire(like,n)
        b.fill_(value);p.fill_(value)
        stencil=SharedStencil(b,p)
        pool.retain(s,stencil)
        return stencil
    a=acquire(9,1)
    b=acquire(10,2)
    c=acquire(11,3)
    assert pool.summary()['slots']==2 and pool.summary()['overflow_allocations']==1
    assert a.phi.unique().item()==1 and b.phi.unique().item()==2
    pointer=a.phi.data_ptr()
    del a
    gc.collect()
    d=acquire(12,4)
    assert d.phi.data_ptr()==pointer
    assert b.phi.unique().item()==2 and c.phi.unique().item()==3
    del d
    large=acquire(31,5)
    assert large.phi.shape==(2,31,3,4) and large.phi.is_contiguous()
    assert pool.summary()['allocations']==4
    with pytest.raises(ValueError,match='budget'):
        pool.acquire(like,101)


def test_unequal_ab2_is_exact_for_linear_advection_in_time():
    current=(torch.tensor([7.],dtype=torch.float64),)
    previous=(torch.tensor([5.],dtype=torch.float64),)
    # Slope=2/.2=10, average extrapolated to t+h/2 gives 7.5.
    actual=extrapolated_advection(current,previous,.1,.2)
    torch.testing.assert_close(actual[0],current[0]+.5)
    torch.testing.assert_close(extrapolated_advection(current,previous,.1)[0],torch.tensor([8.],dtype=torch.float64))


@pytest.mark.parametrize('device',DEVICES)
def test_coupled_two_substeps_match_independent_half_dt_trajectory_and_restart(real_case,tmp_path,device,monkeypatch):
    cfg=warm_config(real_case)
    cfg=replace(cfg,coupling=replace(cfg.coupling,adaptive_substeps=True,max_substep_levels=2),
        interaction_quadrature=replace(cfg.interaction_quadrature,reuse_stencil_buffers=True,prepare_backend='triton'))
    model=imported_model(cfg,device)
    driver=build_driver(model,settings(cfg),device)
    monkeypatch.setattr(driver,'_select_level',lambda state,dt:1)
    state=driver.initialize(model.mesh.X)
    state=replace(state,step=6000,time=.6,force_time=.6,force=model.force(state.x,.6),
        previous_advection=driver.flow.advection(state.velocity),previous_dt=cfg.time.dt)
    before=state.x.clone()
    actual,info=driver.step(state)
    half_cfg=replace(cfg,time=replace(cfg.time,dt=cfg.time.dt/2),
        coupling=replace(cfg.coupling,adaptive_substeps=False))
    half=build_driver(model,settings(half_cfg),device)
    expected=replace(state,step=12000)
    for _ in range(2):
        expected,_=half.step(expected)
    torch.testing.assert_close(actual.x,expected.x,rtol=2e-9,atol=2e-11)
    for a,b in zip(actual.velocity,expected.velocity):
        torch.testing.assert_close(a,b,rtol=2e-6,atol=2e-9)
    torch.testing.assert_close(state.x,before,rtol=0,atol=0)
    assert actual.step==6001 and actual.time==pytest.approx(.6001)
    assert actual.previous_dt==cfg.time.dt/2
    assert actual.pressure_time==pytest.approx(.600075)
    assert info['substeps']['count']==2 and info['substeps']['max_residual_ratio']<=1
    assert driver.flow.dt==cfg.time.dt and driver._stencil is None
    save_real_lv(tmp_path/'state.npz',model,actual,settings(cfg),{},cfg)
    _,restored,_,_,_=load_real_lv(tmp_path/'state.npz',device)
    assert restored.previous_dt==actual.previous_dt and restored.pressure_time==actual.pressure_time
    resumed,_=driver.step(restored)
    continued,_=driver.step(actual)
    torch.testing.assert_close(resumed.x,continued.x,rtol=2e-8,atol=2e-10)
    refined,_,_,_=refine_checkpoint_dt(model,restored,settings(cfg),cfg,cfg.time.dt/2,end_time=.8)
    assert refined.previous_advection is None and refined.previous_dt is None


def test_cfl_retry_rolls_back_whole_interval_and_does_not_swallow_other_errors(real_case,monkeypatch):
    cfg=warm_config(real_case)
    cfg=replace(cfg,coupling=replace(cfg.coupling,adaptive_substeps=True,max_substep_levels=2))
    model=imported_model(cfg)
    driver=build_driver(model,settings(cfg),'cpu')
    state=driver.initialize(model.mesh.X)
    real_step=driver._step
    calls=[]
    def reject_candidate(s,*,diagnostics=True):
        calls.append((driver.flow.dt,s.time))
        result=real_step(s,diagnostics=diagnostics)
        if driver.flow.dt==cfg.time.dt:
            raise CNABTransportGuardError(dict(courant=.26))
        return result
    monkeypatch.setattr(driver,'_step',reject_candidate)
    accepted,info=driver.step(state)
    assert calls==[(cfg.time.dt,0.),(cfg.time.dt/2,0.),(cfg.time.dt/2,cfg.time.dt/2)]
    assert info['substeps']['rejected_attempts']==1 and accepted.time==cfg.time.dt
    assert driver.flow.dt==cfg.time.dt and state.time==0
    def fail(s,*,diagnostics=True):
        raise RuntimeError('nonlinear failure')
    monkeypatch.setattr(driver,'_step',fail)
    with pytest.raises(RuntimeError,match='nonlinear failure'):
        driver.step(accepted)
    assert driver.flow.dt==cfg.time.dt
    assert allocator_sample('cpu')=={}


def test_substep_limit_rejects_without_changing_the_input_and_coarsening_is_bounded(real_case,monkeypatch):
    cfg=warm_config(real_case)
    cfg=replace(cfg,coupling=replace(cfg.coupling,adaptive_substeps=True,max_substep_levels=2))
    model=imported_model(cfg)
    driver=build_driver(model,settings(cfg),'cpu')
    state=driver.initialize(model.mesh.X)
    x=state.x.clone()
    def fail(s,*,diagnostics=True):
        raise CNABTransportGuardError(dict(courant=.30))
    monkeypatch.setattr(driver,'_step',fail)
    with pytest.raises(CNABTransportGuardError) as caught:
        driver.step(state)
    assert caught.value.diagnostics['minimum_substep_dt_s']==cfg.time.dt/4
    assert state.step==0 and state.time==0 and driver.flow.dt==cfg.time.dt
    torch.testing.assert_close(state.x,x,rtol=0,atol=0)
    assert driver._select_level(replace(state,previous_dt=cfg.time.dt/4),cfg.time.dt)==1


@pytest.mark.parametrize('device',DEVICES)
def test_workspace_paired_quadrature_preserves_old_stencil_and_matches_owned_reference(device):
    from test_adaptive_p1_transfer import transfer
    from afsi_torch.mac.adaptive_transfer import AdaptiveP1Transfer
    X,base=transfer(device,stencil_backend='shared')
    options=replace(base.quadrature_options,reuse_stencil_buffers=True,prepare_backend='triton')
    t=AdaptiveP1Transfer(base.grid,base.geometry,quadrature_options=options,fused=False)
    old=t.prepare(X)
    old_phi=old.phi.clone()
    old_rule=old.rule
    changed=X.clone();changed[5,0]+=.3
    current=t.prepare(changed)
    torch.testing.assert_close(old.phi,old_phi,rtol=0,atol=0)
    assert old.rule is old_rule
    ref=base.prepare(changed)
    velocity=tuple(torch.sin(base.grid.coordinates(c,device=device)[...,c]) for c in range(3))
    torch.testing.assert_close(t.gather_grid(velocity,current),base.gather_grid(velocity,ref),rtol=2e-12,atol=2e-13)
    del old,current
    again=t.prepare(X)
    assert t.stencil_workspace.reuses>=1
    assert again.rule.point_count==old_rule.point_count


@pytest.mark.parametrize('device',DEVICES)
def test_actual_prepare_kernel_accepts_capacity_buffers(device,monkeypatch):
    if device=='cpu' and os.environ.get('TRITON_INTERPRET')!='1':
        pytest.skip('actual kernel CPU test requires TRITON_INTERPRET=1')
    pytest.importorskip('triton')
    from test_adaptive_p1_transfer import transfer
    from afsi_torch.mac._triton_prepare import prepare
    X,t=transfer(device,stencil_backend='shared')
    reference=t.prepare(X)
    pool=StencilWorkspace(1000,quantum=64)
    b,p,slot=pool.acquire(X,reference.rule.point_count)
    if device=='cpu':
        monkeypatch.setattr(torch.cuda,'device',lambda device:nullcontext())
    actual_b,actual_p=prepare(t,X,reference.rule,buffers=(b,p))
    assert actual_b.data_ptr()==b.data_ptr() and actual_p.data_ptr()==p.data_ptr()
    torch.testing.assert_close(actual_b,reference.base,rtol=0,atol=0)
    torch.testing.assert_close(actual_p,reference.phi,rtol=2e-12,atol=2e-14)


def test_demo_records_macro_and_internal_clocks_and_old_options_remain_opt_in(real_case,tmp_path):
    from afsi_torch.simulation.real_lv_mac import run
    from afsi_torch.config import _decode
    from afsi_torch.mac.implicit import MACCouplingOptions
    from afsi_torch.mac.adaptive_transfer import InteractionQuadratureOptions
    cfg=warm_config(real_case)
    cfg=replace(cfg,coupling=replace(cfg.coupling,adaptive_substeps=True,max_substep_levels=2),
        interaction_quadrature=replace(cfg.interaction_quadrature,prepare_backend='triton',reuse_stencil_buffers=True),
        output=replace(cfg.output,write_vtk=False))
    folder=tmp_path/'run'
    report=run(case_config=cfg,device='cpu',output=folder)
    assert report['completed'] and report['accepted_steps']==2
    with (folder/'history.csv').open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    assert float(rows[-1]['max_substep_residual_ratio'])<=1
    assert float(rows[-1]['accepted_substep_dt_s'])==cfg.time.dt
    assert report['interaction_quadrature']['stencil_workspace']['reuses']>0
    saved=json.loads((folder/'configuration.json').read_text())
    assert saved['coupling']['adaptive_substeps']
    assert not _decode(MACCouplingOptions,{}).adaptive_substeps
    assert not _decode(InteractionQuadratureOptions,{}).reuse_stencil_buffers
