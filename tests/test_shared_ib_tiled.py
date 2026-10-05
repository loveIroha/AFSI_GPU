"""Local IB reduction preserves all links, affine fields and adjoint power."""
from dataclasses import replace
import os
import pytest
import torch
from test_adaptive_p1_transfer import transfer
from test_real_lv import real_case,DEVICES
from test_mac_shared_pressure import warm_config
from test_mac_implicit import settings
from afsi_torch.mac.adaptive_transfer import InteractionQuadratureOptions
from afsi_torch.mac.grid import MACGrid


INTERPRETER = os.environ.get('TRITON_INTERPRET')=='1'
KERNEL_DEVICES = [pytest.param('cpu',marks=pytest.mark.skipif(not INTERPRETER,reason='requires Triton interpreter')),
                 pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available() or INTERPRETER,reason='native CUDA unavailable'))]


@pytest.mark.parametrize('device',KERNEL_DEVICES)
@pytest.mark.parametrize('family',['conical','xiao-gimbutas'])
def test_actual_kernels_dense_padded_rules_nonzero_offsets_and_adjoint_work(device,family):
    pytest.importorskip('triton')
    if family=='xiao-gimbutas':pytest.importorskip('basix')
    import triton
    from afsi_torch.mac._triton_shared_tiled import _spread_reduced,_gather_cell
    X,t = transfer(device,rule_family=family,stencil_backend='shared')
    x = X+.1
    x[5,0]+=1.;x[6,1]+=1.;x[7,2]+=1.
    s = t.prepare(x)  # one sparse and one dense group; cell order is preserved
    coefficient = torch.sin(1.7*X)+.13*torch.cos(.8*X)
    velocity = tuple(torch.sin(.7*t.grid.coordinates(c,device=device).sum(-1))+.2*c for c in range(3))
    pointforce = torch.cat([t._weighted_kernel(coefficient,g.values,g.cells,g.weights) for g in s.rule.groups])
    expected = t.spread_grid(pointforce,s)
    gathered = t.gather_grid(velocity,s)
    expected_rhs = torch.zeros_like(t.diagonal)
    result,rhs = t.grid.zeros(device=device),torch.zeros_like(t.diagonal)
    offset = 0
    for g in s.rule.groups:
        ne,nq = g.weights.shape
        assert ne>0
        expected_rhs += t._assemble_kernel(gathered[offset:offset+ne*nq],g.values,g.cells,g.weights,t.diagonal)
        for c,out in enumerate(result):
            _spread_reduced[(ne,triton.cdiv(nq,4))](s.base,s.phi,g.cells,g.values,g.weights,coefficient,out,
                s.rule.point_count,offset,nq,*out.shape[1:],c,t.grid.volume,out.numel(),4,
                enable_fp_fusion=False)
        _gather_cell[(ne,)](s.base,s.phi,g.cells,g.values,g.weights,*velocity,rhs,
            s.rule.point_count,offset,nq,*t.grid.shape[1:],min(16,triton.next_power_of_2(nq)),
            enable_fp_fusion=False)
        offset += ne*nq
    for a,b in zip(result,expected):torch.testing.assert_close(a,b,rtol=3e-11,atol=3e-12)
    torch.testing.assert_close(rhs,expected_rhs,rtol=3e-11,atol=3e-12)
    torch.testing.assert_close((coefficient*rhs).sum(),t.grid.volume*sum((u*f).sum() for u,f in zip(velocity,result)),
                               rtol=3e-11,atol=3e-12)
    force = t.mass@coefficient
    torch.testing.assert_close(t.grid.volume*torch.stack([f.sum() for f in result]),force.sum(0),rtol=3e-11,atol=3e-12)
    torque = torch.zeros(3,device=device,dtype=X.dtype)
    for c,f in enumerate(result):
        vector = torch.zeros((*f.shape,3),device=device,dtype=X.dtype);vector[...,c]=f
        torque += t.grid.volume*torch.linalg.cross(t.grid.coordinates(c,device=device),vector).sum((0,1,2))
    torch.testing.assert_close(torque,torch.linalg.cross(x,force).sum(0),rtol=3e-11,atol=3e-11)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('mode',['vector','reduced'])
def test_full_transfer_affine_field_frozen_stencil_and_consistent_mass(device,mode):
    X,a = transfer(device,rule_family='xiao-gimbutas',stencil_backend='shared',transfer_backend='fused')
    _,b = transfer(device,rule_family='xiao-gimbutas',stencil_backend='shared',transfer_backend='fused',shared_execution=mode)
    x = X+.1
    sa,sb = a.prepare(x),b.prepare(x)
    changed = x.clone();changed[5,0]+=1.
    b.prepare(changed)
    A = X.new_tensor([[.02,.01,0.],[0.,-.03,.01],[.01,0.,.04]])
    velocity = tuple((a.grid.coordinates(c,device=device)@A.T+.3)[...,c] for c in range(3))
    force = torch.sin(1.7*X)
    ua,_ = a.interpolate(velocity,sa);ub,_ = b.interpolate(velocity,sb)
    fa,_ = a.spread(force,sa);fb,_ = b.spread(force,sb)
    torch.testing.assert_close(ub,ua,rtol=3e-11,atol=3e-12)
    torch.testing.assert_close(ub,x@A.T+.3,rtol=3e-11,atol=3e-12)
    for u,v in zip(fa,fb):torch.testing.assert_close(v,u,rtol=3e-11,atol=3e-12)
    torch.testing.assert_close((force*ub).sum(),b.grid.volume*sum((u*f).sum() for u,f in zip(velocity,fb)),rtol=3e-11,atol=3e-12)
    torch.testing.assert_close(b.mass.values(),a.mass.values(),rtol=0,atol=0)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('mode',['vector','reduced'])
def test_active_coupled_trajectory_and_newton_tangent_action(real_case,device,mode):
    from afsi_torch.real_lv import imported_model
    from afsi_torch.mac.execution import build_driver
    from afsi_torch.mac.semiimplicit import MidpointProblem
    cfg = warm_config(real_case)
    cfg = replace(cfg,execution=replace(cfg.execution,execution_backend='fused',coupling_backend='optimized'),
                  coupling=replace(cfg.coupling,reuse_validation=True),
                  interaction_quadrature=replace(cfg.interaction_quadrature,transfer_backend='fused'))
    other = replace(cfg,interaction_quadrature=replace(cfg.interaction_quadrature,shared_execution=mode))
    model = imported_model(cfg,device)
    baseline,candidate = build_driver(model,settings(cfg),device),build_driver(model,settings(other),device)
    assert candidate.flow.helmholtz_backend=='torch'
    state = baseline.initialize(model.mesh.X)
    state = replace(state,step=6000,time=.6,force_time=.6,force=model.force(state.x,.6),
        previous_advection=baseline.flow.advection(state.velocity),previous_dt=cfg.time.dt)
    a,b = state,state
    for _ in range(2):
        a,_ = baseline.step(a,diagnostics=False);b,info = candidate.step(b,diagnostics=False)
        torch.testing.assert_close(a.x,b.x,rtol=2e-9,atol=2e-11)
        for u,v in zip(a.velocity,b.velocity):torch.testing.assert_close(u,v,rtol=2e-6,atol=2e-9)
        assert info['nonlinear']['residual_norm']<=info['nonlinear']['tolerance']
    p = MidpointProblem(candidate,b,b.x,candidate.transfer.prepare(b.x),candidate.flow.advection(b.velocity))
    y = torch.zeros_like(b.x);v = .01*torch.sin(b.x)
    action = p.linearization(y)
    finite = (p.residual(y+1e-4*v)-p.residual(y-1e-4*v))/2e-4
    torch.testing.assert_close(action(v),finite,rtol=3e-5,atol=3e-8)


def test_options_cli_checkpoint_and_read_only_comparison(real_case,tmp_path):
    import json
    from afsi_torch.config import _decode
    from afsi_torch.simulation.real_lv_mac import run
    from afsi_torch.real_lv_checkpoint import load_real_lv
    from validation.benchmark_real_lv_schemes import benchmark
    from demo.real_lv_fsi.run_mac import main
    from afsi_torch.mac.cnab import MACCNABFlow
    assert _decode(InteractionQuadratureOptions,{}).shared_execution=='reference'
    assert MACCNABFlow(MACGrid((4,)*3,(1.,)*3),dt=.001).helmholtz_backend=='torch'
    for kwargs in (dict(shared_execution='bad'),dict(shared_execution='reduced')):
        with pytest.raises(ValueError,match='shared'):InteractionQuadratureOptions(**kwargs)
    config = tmp_path/'options.json'
    main(['--ib-transfer-backend','fused','--ib-stencil-backend','shared','--ib-shared-execution','reduced',
          '--write-config',str(config)])
    assert json.loads(config.read_text())['interaction_quadrature']['shared_execution']=='reduced'
    cfg = warm_config(real_case)
    cfg = replace(cfg,interaction_quadrature=replace(cfg.interaction_quadrature,transfer_backend='fused',shared_execution='reduced'))
    run(case_config=cfg,device='cpu',output=tmp_path/'case')
    path = tmp_path/'case'/'checkpoint.npz';original = path.read_bytes()
    assert load_real_lv(path)[-1].interaction_quadrature.shared_execution=='reduced'
    report = benchmark(path,device='cpu',schemes=('cnab-semiimplicit',),nonlinear_solvers=('anderson-newton',),
        ib_shared_executions=('reference','vector','reduced'),warmup=0,steps=2)
    assert path.read_bytes()==original
    for label,case in report['cases'].items():
        assert case['completed'] and case['helmholtz']['backend']=='torch'
    for comparison in report['ib_shared_comparisons'].values():
        assert comparison['final_state_differences']['x_max_abs']<2e-11


def test_execution_only_restart_retains_state_and_ab2_history(real_case,tmp_path):
    from afsi_torch.simulation.real_lv_mac import run
    from afsi_torch.real_lv_checkpoint import load_real_lv
    from demo.real_lv_fsi.run_mac import main
    cfg = warm_config(real_case)
    cfg = replace(cfg,interaction_quadrature=replace(cfg.interaction_quadrature,transfer_backend='fused'))
    run(case_config=cfg,device='cpu',output=tmp_path/'source')
    source = tmp_path/'source'/'checkpoint.npz';before = source.read_bytes()
    _,a,_,_,_ = load_real_lv(source)
    with pytest.raises(ValueError,match='new output directory'):
        run(resume=source,device='cpu',shared_execution='vector')
    main(['--resume',str(source),'--device','cpu','--ib-shared-execution','vector',
          '--helmholtz-backend','torch','--end-time',str(a.time),'--output',str(tmp_path/'branch')])
    _,b,settings,progress,config = load_real_lv(tmp_path/'branch'/'checkpoint.npz')
    assert source.read_bytes()==before and b.time==a.time and b.step==a.step
    for field in ('x','force','pressure'):
        torch.testing.assert_close(getattr(b,field),getattr(a,field),rtol=0,atol=0)
    for x,y in zip(b.velocity+b.previous_advection,a.velocity+a.previous_advection):
        torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert b.previous_dt==a.previous_dt and b.pressure_time==a.pressure_time
    assert config.interaction_quadrature.shared_execution=='vector'
    assert settings['coupling']['cnab']['helmholtz_backend']=='torch'
    assert progress['restart_from']['ab2_history_retained']
    assert progress['restart_from']['execution_only_override']
