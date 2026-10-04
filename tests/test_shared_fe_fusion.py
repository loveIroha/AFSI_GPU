"""Shared FE/IB fusion retains adjoint work, quadrature and accepted equations."""
from dataclasses import replace,asdict
import json
import os
import numpy as np
import pytest
import torch
from test_adaptive_p1_transfer import transfer
from test_real_lv import real_case,DEVICES
from test_mac_implicit import settings
from test_mac_shared_pressure import warm_config
from afsi_torch.mac.adaptive_transfer import InteractionQuadratureOptions
from afsi_torch.mac.execution import build_driver
from afsi_torch.mac.semiimplicit import MidpointProblem
from afsi_torch.real_lv import imported_model


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('family',['conical','xiao-gimbutas'])
def test_shared_fusion_force_torque_work_affine_and_old_rule(device,family,monkeypatch):
    if family=='xiao-gimbutas': pytest.importorskip('basix')
    X,a=transfer(device,rule_family=family,stencil_backend='shared',prepare_backend='triton')
    _,b=transfer(device,rule_family=family,stencil_backend='shared',prepare_backend='triton',transfer_backend='fused')
    x=X+X.new_tensor([.1,.2,-.1])
    sa,sb=a.prepare(x),b.prepare(x)
    other=x.clone();other[5,0]+=.6
    b.prepare(other)  # Frozen stencil must retain paired membership and offsets.
    A=X.new_tensor([[.02,.01,0.],[0.,-.03,.01],[.01,0.,.04]])
    velocity=tuple((a.grid.coordinates(c,device=device)@A.T+.3)[...,c] for c in range(3))
    force=torch.sin(1.7*X)
    ua,_=a.interpolate(velocity,sa);fa,_=a.spread(force,sa)
    if device=='cuda':
        def forbidden(*args,**kwargs): raise AssertionError('point intermediates must not be materialized')
        monkeypatch.setattr(b,'_weighted_kernel',forbidden)
        monkeypatch.setattr(b,'gather_grid',forbidden)
        monkeypatch.setattr(b,'_assemble_kernel',forbidden)
    ub,_=b.interpolate(velocity,sb);fb,_=b.spread(force,sb)
    torch.testing.assert_close(ub,ua,rtol=3e-11,atol=3e-12)
    torch.testing.assert_close(ub,x@A.T+.3,rtol=3e-11,atol=3e-12)
    for u,v in zip(fa,fb): torch.testing.assert_close(v,u,rtol=3e-11,atol=3e-12)
    torch.testing.assert_close((force*ub).sum(),b.grid.volume*sum((u*f).sum() for u,f in zip(velocity,fb)),rtol=3e-11,atol=3e-12)
    torch.testing.assert_close(b.grid.volume*torch.stack([f.sum() for f in fb]),force.sum(0),rtol=3e-11,atol=3e-12)
    torque=torch.zeros(3,device=device,dtype=X.dtype)
    for c,f in enumerate(fb):
        vector=torch.zeros((*f.shape,3),device=device,dtype=X.dtype);vector[...,c]=f
        torque+=b.grid.volume*torch.linalg.cross(b.grid.coordinates(c,device=device),vector).sum((0,1,2))
    torch.testing.assert_close(torque,torch.linalg.cross(x,force).sum(0),rtol=3e-11,atol=3e-11)
    torch.testing.assert_close(b.mass.values(),a.mass.values(),rtol=0,atol=0)


@pytest.mark.skipif(os.environ.get('TRITON_INTERPRET')!='1',reason='requires Triton interpreter')
@pytest.mark.parametrize('family',['conical','xiao-gimbutas'])
def test_actual_shared_cuda_kernels_interpreter_padded_groups_and_offsets(family):
    pytest.importorskip('triton')
    if family=='xiao-gimbutas': pytest.importorskip('basix')
    import triton
    from afsi_torch.mac._triton_shared_fe import _spread_shared,_gather_shared
    X,t=transfer('cpu',rule_family=family,stencil_backend='shared')
    s=t.prepare(X+.1)
    coefficient=torch.sin(X)
    pointforce=torch.cat([t._weighted_kernel(coefficient,g.values,g.cells,g.weights) for g in s.rule.groups])
    expected=t.spread_grid(pointforce,s)
    result=t.grid.zeros()
    velocity=tuple(torch.sin(t.grid.coordinates(c)[...,c])+.2 for c in range(3))
    gathered=t.gather_grid(velocity,s)
    expected_rhs=torch.zeros_like(t.diagonal);rhs=torch.zeros_like(t.diagonal)
    offset=0
    for g in s.rule.groups:
        ne,nq=g.weights.shape;count=ne*nq
        expected_rhs+=t._assemble_kernel(gathered[offset:offset+count],g.values,g.cells,g.weights,t.diagonal)
        for c,out in enumerate(result):
            _spread_shared[(triton.cdiv(count,4),)](s.base,s.phi,g.cells,g.values,g.weights,coefficient,out,
                ne,s.rule.point_count,offset,nq,*t.grid.face_shape(c)[1:],c,t.grid.volume,4)
            tile=min(16,triton.next_power_of_2(nq))
            _gather_shared[(ne,triton.cdiv(nq,tile))](s.base,s.phi,g.cells,g.values,g.weights,velocity[c],rhs,
                s.rule.point_count,offset,nq,*t.grid.face_shape(c)[1:],c,tile)
        offset+=count
    for a,b in zip(expected,result):torch.testing.assert_close(b,a,rtol=3e-12,atol=3e-13)
    torch.testing.assert_close(rhs,expected_rhs,rtol=3e-12,atol=3e-13)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('solver',['newton','anderson-newton'])
def test_active_accepted_trajectory_csr_and_checks_preserved(real_case,device,solver):
    cfg=warm_config(real_case)
    cfg=replace(cfg,execution=replace(cfg.execution,execution_backend='fused',coupling_backend='optimized'))
    cfg=replace(cfg,coupling=replace(cfg.coupling,semiimplicit_solver=solver),
                interaction_quadrature=replace(cfg.interaction_quadrature,prepare_backend='triton'))
    fast=replace(cfg,coupling=replace(cfg.coupling,reuse_validation=True),
                 interaction_quadrature=replace(cfg.interaction_quadrature,transfer_backend='fused'))
    m=imported_model(cfg,device)
    a,b=build_driver(m,settings(cfg),device),build_driver(m,settings(fast),device)
    sa=a.initialize(m.mesh.X)
    sa=replace(sa,step=6000,time=.6,force_time=.6,force=m.force(sa.x,.6),previous_advection=a.flow.advection(sa.velocity))
    sb=sa
    for _ in range(3):
        sa,ia=a.step(sa);sb,ib=b.step(sb)
        torch.testing.assert_close(sb.x,sa.x,rtol=2e-9,atol=2e-11)
        for u,v in zip(sa.velocity,sb.velocity):torch.testing.assert_close(v,u,rtol=2e-6,atol=2e-9)
        assert ib['nonlinear']['residual_norm']<=ib['nonlinear']['tolerance']
        assert ib['flow']['stokes']['momentum_residual']<=ib['flow']['stokes']['momentum_tolerance']
        assert ib['flow']['stokes']['divergence_norm']<=ib['flow']['stokes']['divergence_tolerance']
        assert ib['nonlinear']['validation_reuses']>0
        assert ib['nonlinear']['validation_evaluations']<ia['nonlinear']['validation_evaluations']
        assert ib['nonlinear']['tangent_assemblies']==ia['nonlinear']['tangent_assemblies']
        assert ib['power_error']/max(abs(ib['solid_power']),abs(ib['fluid_power']),1.)<1e-8
    p=MidpointProblem(b,sb,sb.x,b.transfer.prepare(sb.x),b.flow.advection(sb.velocity))
    y=torch.zeros_like(sb.x);v=.01*torch.sin(sb.x)
    action=p.linearization(y)
    finite=(p.residual(y+1e-4*v)-p.residual(y-1e-4*v))/2e-4
    torch.testing.assert_close(action(v),finite,rtol=3e-5,atol=3e-8)


def test_versioned_validation_invalidates_and_never_accepts_failed_probe(real_case,monkeypatch):
    cfg=warm_config(real_case);cfg=replace(cfg,coupling=replace(cfg.coupling,reuse_validation=True))
    cfg=replace(cfg,execution=replace(cfg.execution,execution_backend='fused',coupling_backend='optimized'))
    m=imported_model(cfg);d=build_driver(m,settings(cfg),'cpu');state=d.initialize(m.mesh.X)
    p=MidpointProblem(d,state,state.x.clone(),d.transfer.prepare(state.x),d.flow.advection(state.velocity))
    y=torch.zeros_like(state.x)
    p.validate(y);p.validate(y)
    assert p.validation_evaluations==1 and p.validation_reuses==1
    p.validate_final(y.clone());assert p.validation_evaluations==1
    p.validate(y.to(torch.float32));assert p.validation_evaluations==2  # promotion causes a full recheck
    y.add_(1e-7);p.validate(y);assert p.validation_evaluations==3
    p.predicted.add_(1e-7);p.validate(y);assert p.validation_evaluations==4
    clone=p.predicted.clone();p.predicted=clone;p.validate(y);assert p.validation_evaluations==5
    # A failed new probe clears the prior eligibility, even when returning to y.
    bad=y+100
    with pytest.raises(ValueError):p.validate(bad)
    assert p._validated is None
    p.validate(y);assert p.validation_evaluations==7
    calls=[];kernel=d.solid_execution._geometry_kernel
    def counted(x):calls.append(1);return kernel(x)
    monkeypatch.setattr(d.solid_execution,'_geometry_kernel',counted)
    p.evaluate(y)  # midpoint geometry from validation, no second geometry kernel
    assert not calls
    endpoint=p.endpoint(y);d._force_geometry(endpoint,.001)
    assert not calls
    endpoint.add_(1e-7);d._force_geometry(endpoint,.001)
    assert len(calls)==1
    n=p.validation_evaluations
    p.validate_final(y.clone());assert p.validation_evaluations==n+1
    with torch.inference_mode():
        infer=torch.zeros_like(y)
        p.validate(infer);p.validate(infer)
        assert p._validated is None


def test_benchmark_cli_and_checkpoint_compatibility(real_case,tmp_path):
    from afsi_torch.simulation.real_lv_mac import run
    from afsi_torch.real_lv_checkpoint import load_real_lv
    from validation.benchmark_real_lv_schemes import benchmark
    from demo.real_lv_fsi.run_mac import main
    from afsi_torch.config import _decode
    from afsi_torch.mac.implicit import MACCouplingOptions
    cfg=warm_config(real_case)
    folder=tmp_path/'case';run(case_config=cfg,device='cpu',output=folder)
    path=folder/'checkpoint.npz';original=path.read_bytes()
    report=benchmark(path,device='cpu',schemes=('cnab-semiimplicit',),nonlinear_solvers=('anderson-newton',),
        execution_variants=('prepare-warm','shared-fused-warm','shared-fused-checked'),warmup=0,steps=2,profile=True)
    prefix='cnab-semiimplicit/anderson-newton/'
    for name in ('prepare-warm','shared-fused-warm','shared-fused-checked'):
        case=report['cases'][prefix+name]
        assert case['completed'] and 'profile_failure' not in case
        assert case['counts']==report['cases'][prefix+'prepare-warm']['counts']
        if name=='shared-fused-checked':assert case['validation_reuses']>0
        if name!='prepare-warm':assert report['execution_comparisons'][prefix+name]['final_state_differences']['x_max_abs']<2e-11
    assert path.read_bytes()==original
    # Real older checkpoint metadata omits the new execution flag entirely.
    from afsi_torch.mac.checkpoint import digest
    with np.load(path,allow_pickle=False) as archive:
        arrays={name:archive[name] for name in archive.files}
    metadata=json.loads(str(arrays.pop('metadata')));metadata.pop('sha256')
    for group in (metadata['config'],metadata['settings']):
        group['coupling'].pop('reuse_validation')
    metadata['sha256']=digest(metadata,arrays)
    legacy=tmp_path/'legacy.npz';np.savez_compressed(legacy,metadata=json.dumps(metadata),**arrays)
    assert not load_real_lv(legacy)[-1].coupling.reuse_validation
    f=tmp_path/'config.json'
    main(['--ib-transfer-backend','fused','--ib-stencil-backend','shared','--ib-prepare-backend','triton',
          '--stokes-warm-start','--reuse-validation','--write-config',str(f)])
    data=json.loads(f.read_text());assert data['coupling']['reuse_validation']
    assert data['interaction_quadrature']['transfer_backend']=='fused'
    assert not _decode(MACCouplingOptions,{}).reuse_validation
    fast=replace(cfg,coupling=replace(cfg.coupling,reuse_validation=True))
    out=tmp_path/'cached';run(case_config=fast,device='cpu',output=out)
    assert load_real_lv(out/'checkpoint.npz')[-1].coupling.reuse_validation
