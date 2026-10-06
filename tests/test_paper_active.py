"""User active PK1 and cyclic loading on the existing paper BE-BE equations."""
from dataclasses import replace
import csv
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from test_real_lv import real_case
from test_paper_lv import config_for, DEVICES
from afsi_torch.paper_lv import PaperLVConfig, PaperHOParameters, paper_pk1, imported_model
from afsi_torch.holzapfel_ogden import RealLVLoads, active_energy
from afsi_torch.config import TimeConfig, load_config
from afsi_torch.mac.paper_coupling import BEIBStepper
from afsi_torch.paper_lv_checkpoint import save, load
from afsi_torch.simulation.paper_lv_mac import run


USER_MATERIAL=PaperHOParameters(a=2400.,b=5.08,a_f=14600.,b_f=4.15,
    a_s=8700.,b_s=1.6,a_fs=3000.,b_fs=1.3,kappa=5e6)


@pytest.mark.parametrize('device', DEVICES)
def test_active_pk1_matches_supplied_ufl_potential_and_objectivity(device):
    p=USER_MATERIAL
    f=torch.tensor([1.,0.,0.],device=device,dtype=torch.float64)
    s=f.new_tensor([0.,1.,0.])
    Q=f.new_tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    for stretch in (.75,.9,1.,1.12):
        F=torch.diag(f.new_tensor([stretch,1.02,.99])).requires_grad_()
        T=83000.
        total=paper_pk1(F,f,s,p,T)
        passive=paper_pk1(F,f,s,p)
        expected=T*(1+4.9*(torch.sqrt(f@(F.T@F)@f)-1))*(F@torch.outer(f,f))
        torch.testing.assert_close(total-passive,expected,rtol=2e-12,atol=2e-9)
        potential=active_energy(F,f,T,p)
        torch.testing.assert_close(total-passive,torch.autograd.grad(potential,F)[0],rtol=2e-12,atol=2e-9)
        torch.testing.assert_close(paper_pk1(Q@F,f,s,p,T),Q@total,rtol=2e-12,atol=2e-9)
        # The user's coefficient is not an already-normalized Cauchy stress.
        sigma=expected@F.T/torch.linalg.det(F)
        torch.testing.assert_close(sigma,sigma.T)


def test_active_protocol_preserves_cpp_waveform_and_pressure_reset():
    config=replace(PaperLVConfig(),load_protocol='active-cycle')
    for cycle in range(3):
        for t in (0.,.1,.2,.499,.5,.6,.65,.7,.7999):
            p,T=config.applied_loads.at(cycle*.8+t)
            d=t-.5 if t<.65 else .8-t
            expected_p=(1.067*t/.2 if t<.2 else 1.067 if t<.5
                else 1.067+13.46*(1-np.exp(-d*d/.004)))
            expected_T=0. if t<.5 else 84.26*(1-np.exp(-d*d/.005))
            assert p==pytest.approx(expected_p*1e4,rel=1e-11,abs=1e-8)
            assert T==pytest.approx(expected_T*1e4,rel=1e-11,abs=1e-8)
    assert config.applied_loads.at(2.4)==(0.,0.)
    assert config.applied_loads.at(.8-1e-7)[0]>10669.
    assert PaperLVConfig().applied_loads.at(.8)==(8*1333.22387415,0.)


@pytest.mark.parametrize('device', DEVICES)
def test_active_weak_force_compiled_execution_and_csr_tangent(real_case,device):
    cfg=replace(config_for(real_case),load_protocol='active-cycle',material=USER_MATERIAL)
    model=imported_model(cfg,device)
    X=model.mesh.X
    A=X.new_tensor([[1.02,.002,.001],[.001,1.015,.001],[0.,.001,1.01]])
    x=((X-X.mean(0))@A.T+X.mean(0)).requires_grad_()
    F,endo,_=model.geometry_state(x)
    pressure,T=model.loads.at(.6)
    total=model.force_from_geometry(x,F,endo,X.new_tensor([pressure,T]))
    passive=model.force_from_geometry(x,F,endo,X.new_tensor([pressure,0.]))
    W=(active_energy(F,model.mesh.fiber,T,model.parameters)*model.volumes).sum()
    torch.testing.assert_close(total-passive,-torch.autograd.grad(W,x)[0],rtol=2e-10,atol=1e-7)
    x=x.detach()
    execution=model.execution_factory()
    for time in (.49,.6,.7,.8):
        torch.testing.assert_close(execution.force(x,time),model.force(x,time),rtol=2e-10,atol=1e-7)
    tangent=model.tangent_factory().assemble(x,.6)
    assert tangent.layout==torch.sparse_csr and tangent.device==x.device
    direction=.01*torch.sin(1.3*X)
    eps=1e-4
    expected=(model.force(x+eps*direction,.6)-model.force(x-eps*direction,.6))/(2*eps)
    actual=torch.sparse.mm(tangent,direction.reshape(-1,1)).reshape_as(x)
    torch.testing.assert_close(actual,expected,rtol=5e-6,atol=5e-4)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('solver',['jfnk','newton','anderson-newton'])
def test_active_be_endpoint_coupling_and_frozen_old_stencil(real_case,device,solver):
    cfg=replace(config_for(real_case,solver),load_protocol='active-cycle',material=USER_MATERIAL)
    if device=='cuda':
        cfg=replace(cfg,execution=PaperLVConfig().execution)
    model=imported_model(cfg,device)
    driver=BEIBStepper(model,cfg,device)
    initial=driver.initialize(model.mesh.X)
    # An isolated loaded step at a nonzero activation time, not a simulated
    # diastolic history. Test the actual fully coupled endpoint equations.
    old=replace(initial,step=5999,time=.5999,force_time=.5999,pressure_time=.5999,
        force=model.force(initial.x,.5999))
    new,info=driver.step(old)
    assert new.time==pytest.approx(.6)
    assert new.force_time==new.pressure_time==new.time==new.step*cfg.time.dt
    assert model.loads.at(new.time)[1]>7e5
    torch.testing.assert_close(new.force,model.force(new.x,new.time),rtol=1e-10,atol=1e-7)
    nodal,_=driver.transfer.interpolate(new.velocity,driver.transfer.prepare(old.x))
    residual=new.x-old.x-cfg.time.dt*nodal
    assert torch.linalg.vector_norm(residual).item()<=1.03*info['nonlinear']['tolerance']
    assert info['nonlinear']['residual_norm']<=info['nonlinear']['tolerance']
    torch.testing.assert_close(old.x,model.mesh.X,rtol=0,atol=0)


def test_active_checkpoint_resume_records_actual_tension_and_keeps_paper_legacy(real_case,tmp_path):
    cfg=replace(config_for(real_case),load_protocol='active-cycle',material=USER_MATERIAL,time=TimeConfig(1e-4,.6001))
    model=imported_model(cfg)
    driver=BEIBStepper(model,cfg,'cpu')
    state=driver.initialize(model.mesh.X)
    state=replace(state,step=5999,time=.5999,force_time=.5999,pressure_time=.5999,
        previous_x=state.x.clone(),force=model.force(state.x,.5999))
    path=tmp_path/'active'/'checkpoint.npz'
    save(path,model,state,cfg,dict(elapsed_seconds=0.))
    restored_model,restored,restored_config,_=load(path)
    assert restored_config==cfg and restored_model.loads.at(.6)==RealLVLoads().at(.6)
    direct=BEIBStepper(restored_model,cfg,'cpu')
    expected,info=direct.step(restored)
    expected,_=direct.step(expected)
    report=run(resume=path,device='cpu')
    _,actual,_,_=load(path)
    torch.testing.assert_close(actual.x,expected.x,rtol=1e-11,atol=1e-12)
    assert report['case_purpose']=='user active-cycle extension'
    assert any('user material coefficients' in item for item in report['reproduction_differences'])
    assert not report['published_results_reproduced'] and not report['full_horizon_validated']
    assert report['active_stress']['enabled'] and report['completed_active_cycles']==0
    rows=list(csv.DictReader((path.parent/'history.csv').open()))
    for row in rows:
        p,T=cfg.applied_loads.at(float(row['time_s']))
        assert float(row['active_tension_dyn_per_cm2'])==pytest.approx(T)
        assert float(row['endocardial_pressure_dyn_per_cm2'])==pytest.approx(p)

    # Pre-extension passive checkpoints have neither protocol nor active slope.
    cfg=config_for(real_case)
    model=imported_model(cfg)
    state=BEIBStepper(model,cfg,'cpu').initialize(model.mesh.X)
    legacy=tmp_path/'legacy.npz'
    save(legacy,model,state,cfg,{})
    from afsi_torch.mac.checkpoint import digest
    with np.load(legacy,allow_pickle=False) as archive:
        arrays={k:archive[k] for k in archive.files}
    meta=json.loads(str(arrays.pop('metadata')))
    meta.pop('sha256')
    meta['config'].pop('load_protocol');meta['config'].pop('cyclic_loads')
    meta['config']['material'].pop('active_stretch_slope')
    meta['sha256']=digest(meta,arrays)
    np.savez_compressed(legacy,metadata=json.dumps(meta),**arrays)
    restored,_,old_config,_=load(legacy)
    assert old_config.load_protocol=='inflation' and restored.loads.at(.65)[1]==0.


def test_active_preset_cli_cycle_count_and_resume_restrictions(tmp_path):
    from demo.real_lv_fsi.run_mac import main
    preset=Path(__file__).resolve().parents[1]/'demo/real_lv_fsi/active_cycle.json'
    cfg=load_config(preset,PaperLVConfig)
    assert cfg.load_protocol=='active-cycle' and cfg.time.end_time==2.4
    assert cfg.nonlinear_solver=='anderson-newton' and cfg.flow.helmholtz_backend=='graph'
    assert cfg.material==USER_MATERIAL and cfg.fluid==PaperLVConfig().fluid
    target=tmp_path/'config.json'
    main(['--config',str(preset),'--cycles','1','--write-config',str(target)])
    assert load_config(target,PaperLVConfig).time.end_time==.8
    for args in (['--cycles','1'],['--config',str(preset),'--cycles','0'],
                 ['--config',str(preset),'--cycles','1','--end-time','.5'],
                 ['--resume','checkpoint.npz','--load-protocol','active-cycle']):
        with pytest.raises(SystemExit):
            main(args+['--write-config',str(target)])
