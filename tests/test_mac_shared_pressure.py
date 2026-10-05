"""Shared MAC tables and same-step pressure warm starts retain FE/IB equations."""
from dataclasses import replace,asdict
import json
import os
import numpy as np
import pytest
import torch
from test_adaptive_p1_transfer import transfer
from test_real_lv import real_case,DEVICES
from test_mac_implicit import settings
from test_mac_compact_quadrature import adaptive_config
from afsi_torch.mac.adaptive_transfer import AdaptiveP1Transfer,InteractionQuadratureOptions
from afsi_torch.mac.compact_transfer import _prepare
from afsi_torch.mac.shared_stencil import SharedStencil
from afsi_torch.mac.grid import MACGrid,zero_normal
from afsi_torch.mac.execution import build_driver
from afsi_torch.mac.semiimplicit import MidpointProblem
from afsi_torch.real_lv import imported_model


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('family',['conical','xiao-gimbutas'])
@pytest.mark.parametrize('nonuniform',[False,True])
def test_shared_tables_same_fields_affine_power_torque_and_storage(device,family,nonuniform):
    if family=='xiao-gimbutas':
        pytest.importorskip('basix')
    X,original = transfer(device,rule_family=family)
    grid = MACGrid((16,16,16),(16.,20.,24.),(-1.3,-2.1,-.7)) if nonuniform else original.grid
    options = InteractionQuadratureOptions(mode='adaptive',rule_family=family)
    a = AdaptiveP1Transfer(grid,original.geometry,quadrature_options=options,fused=False)
    b = AdaptiveP1Transfer(grid,original.geometry,quadrature_options=replace(options,stencil_backend='shared'),fused=False)
    x = X+X.new_tensor([.1,.2,-.1])
    old,new = a.prepare(x),b.prepare(x)
    expanded = new.expanded()
    torch.testing.assert_close(expanded.base,old.base,rtol=0,atol=0)
    torch.testing.assert_close(expanded.phi,old.phi,rtol=1e-13,atol=1e-14)
    assert new.storage_bytes*3==old.storage_bytes*2
    assert b.quadrature_summary()['stencil_storage_bytes']==new.storage_bytes
    other = x.clone(); other[5,0] += .6
    b.prepare(other)  # A later prepare must not overwrite the old shared rule.
    force = torch.sin(1.7*X)
    A = X.new_tensor([[.02,.01,0.],[0.,-.03,.01],[.01,0.,.04]])
    constant = X.new_tensor([.3,-.2,.5])
    velocity = tuple((grid.coordinates(c,device=device)@A.T+constant)[...,c] for c in range(3))
    ua,_ = a.interpolate(velocity,old)
    ub,_ = b.interpolate(velocity,new)
    fa,_ = a.spread(force,old)
    fb,_ = b.spread(force,new)
    torch.testing.assert_close(ub,x@A.T+constant,rtol=3e-11,atol=3e-12)
    torch.testing.assert_close(ub,ua,rtol=3e-11,atol=3e-12)
    for u,v in zip(fa,fb):
        torch.testing.assert_close(v,u,rtol=3e-11,atol=3e-12)
    torch.testing.assert_close(grid.volume*torch.stack([f.sum() for f in fb]),force.sum(0),rtol=3e-11,atol=3e-12)
    torch.testing.assert_close((force*ub).sum(),grid.volume*sum((u*f).sum() for u,f in zip(velocity,fb)),rtol=3e-11,atol=3e-12)
    torque = torch.zeros(3,device=device,dtype=X.dtype)
    for c,f in enumerate(fb):
        field = torch.zeros((*f.shape,3),device=device,dtype=X.dtype); field[...,c] = f
        torque += grid.volume*torch.linalg.cross(grid.coordinates(c,device=device),field).sum((0,1,2))
    torch.testing.assert_close(torque,torch.linalg.cross(x,force).sum(0),rtol=3e-11,atol=3e-11)
    torch.testing.assert_close(b.mass.values(),a.mass.values(),rtol=0,atol=0)


def warm_config(real_case,shared=True,warm=True):
    cfg = adaptive_config(real_case)
    return replace(cfg,
        interaction_quadrature=replace(cfg.interaction_quadrature,rule_family='xiao-gimbutas',
            stencil_backend='shared' if shared else 'component'),
        coupling=replace(cfg.coupling,stokes_warm_start=warm))


def make_problem(real_case,*,warm=True):
    cfg = warm_config(real_case,warm=warm)
    model = imported_model(cfg)
    driver = build_driver(model,settings(cfg),'cpu')
    state = driver.initialize(model.mesh.X)
    state = replace(state,step=6000,time=.6,force_time=.6,force=model.force(state.x,.6))
    problem = MidpointProblem(driver,state,state.x,driver.transfer.prepare(state.x),driver.flow.advection(state.velocity))
    return driver,state,problem


def test_same_step_pressure_reuses_successful_guess_and_reduces_corrections(real_case,monkeypatch):
    pytest.importorskip('basix')
    driver,state,problem = make_problem(real_case)
    solve = driver.flow.pressure_solver.solve
    calls = []
    def counted(*args,**kwargs):
        calls.append(1)
        return solve(*args,**kwargs)
    monkeypatch.setattr(driver.flow.pressure_solver,'solve',counted)
    y = torch.zeros_like(state.x)
    first = problem.evaluate(y)
    cold = len(calls)
    second = problem.evaluate(y)
    warm = len(calls)-cold
    assert cold>0 and warm<cold and problem.pressure_warm_starts==1
    assert problem.pressure_warm_fallbacks==0
    torch.testing.assert_close(second[0],first[0],rtol=2e-7,atol=2e-11)
    info = second[1].diagnostics['stokes']
    assert info['momentum_residual']<=info['momentum_tolerance']
    assert info['divergence_norm']<=info['divergence_tolerance']
    cached = problem._pressure_guess.clone()
    first[1].pressure.add_(7.)
    torch.testing.assert_close(problem._pressure_guess,cached,rtol=0,atol=0)
    problem.linearization(y)(.01*torch.sin(state.x))
    torch.testing.assert_close(problem._pressure_guess,cached,rtol=0,atol=0)


def test_warm_failure_retries_original_pressure_without_looser_acceptance(real_case,monkeypatch):
    pytest.importorskip('basix')
    driver,state,problem = make_problem(real_case)
    y = torch.zeros_like(state.x)
    expected = problem.evaluate(y)
    stokes = driver.flow.stokes
    guesses = []
    def injected(rhs,initial=None):
        guesses.append(initial)
        if len(guesses)==1:
            raise RuntimeError('injected warm-guess failure')
        return stokes(rhs,initial)
    monkeypatch.setattr(driver.flow,'stokes',injected)
    result = problem.evaluate(y)
    assert guesses[0] is not state.pressure and guesses[1] is state.pressure
    assert problem.pressure_warm_starts==1 and problem.pressure_warm_fallbacks==1
    torch.testing.assert_close(result[0],expected[0],rtol=2e-7,atol=2e-11)
    assert result[1].diagnostics['stokes']['momentum_residual']<=result[1].diagnostics['stokes']['momentum_tolerance']


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('solver',['newton','anderson-newton'])
def test_shared_warm_accepted_active_step_and_csr_action(real_case,device,solver):
    pytest.importorskip('basix')
    cfg = warm_config(real_case)
    cfg = replace(cfg,coupling=replace(cfg.coupling,semiimplicit_solver=solver))
    base = warm_config(real_case,shared=False,warm=False)
    base = replace(base,coupling=replace(base.coupling,semiimplicit_solver=solver))
    model = imported_model(cfg,device)
    candidate = build_driver(model,settings(cfg),device)
    reference = build_driver(model,settings(base),device)
    s = reference.initialize(model.mesh.X)
    # Active phase checks the H-O/follower/basal tangent, rather than only t=0.
    velocity = reference.flow.project(zero_normal(tuple(.01*torch.sin(reference.flow.grid.coordinates(c,device=device)[...,(c+1)%3]) for c in range(3)))).velocity
    s = replace(s,step=6000,time=.6,velocity=velocity,force_time=.6,force=model.force(s.x,.6),
        previous_advection=reference.flow.advection(velocity))
    a,ia = reference.step(s)
    b,ib = candidate.step(s)
    assert ib['nonlinear']['residual_norm']<=ib['nonlinear']['tolerance']
    assert ib['nonlinear']['stokes_pressure_warm_starts']>0
    torch.testing.assert_close(b.x,a.x,rtol=2e-9,atol=2e-11)
    for u,v in zip(a.velocity,b.velocity):
        torch.testing.assert_close(v,u,rtol=2e-6,atol=2e-9)
    assert ib['power_error']/max(abs(ib['solid_power']),abs(ib['fluid_power']),1.)<1e-8
    problem = MidpointProblem(candidate,s,s.x,candidate.transfer.prepare(s.x),candidate.flow.advection(s.velocity))
    y,v = torch.zeros_like(s.x),.01*torch.sin(s.x)
    action = problem.linearization(y)
    finite = (problem.residual(y+1e-4*v)-problem.residual(y-1e-4*v))/2e-4
    torch.testing.assert_close(action(v),finite,rtol=3e-5,atol=3e-8)


def test_shared_warm_checkpoint_and_four_way_profile_benchmark(real_case,tmp_path):
    pytest.importorskip('basix')
    from afsi_torch.simulation.real_lv_mac import run
    from afsi_torch.real_lv_checkpoint import load_real_lv
    from validation.benchmark_real_lv_schemes import benchmark
    cfg = warm_config(real_case)
    folder = tmp_path/'shared'
    run(case_config=cfg,device='cpu',output=folder)
    path = folder/'checkpoint.npz'
    _,_,opts,_,restored = load_real_lv(path)
    assert opts['coupling']['stokes_warm_start'] and restored.coupling==cfg.coupling
    assert restored.interaction_quadrature==cfg.interaction_quadrature
    resumed = run(resume=path,device='cpu',end_time=3e-4)
    assert resumed['completed'] and resumed['interaction_quadrature']['stencil_backend']=='shared'
    original = path.read_bytes()
    report = benchmark(path,device='cpu',schemes=('cnab-semiimplicit',),
        nonlinear_solvers=('anderson-newton',),execution_variants=('compact','shared','pressure-warm','shared-warm'),
        warmup=0,steps=2,profile=True)
    prefix = 'cnab-semiimplicit/anderson-newton/'
    cases = report['cases']
    for name in ('compact','shared','pressure-warm','shared-warm'):
        case = cases[prefix+name]
        assert case['completed'] and 'profile_failure' not in case
        assert case['interaction_quadrature']['transfer_backend']=='reference'
        assert case['stokes_warm_start']==(name in ('pressure-warm','shared-warm'))
        if name!='compact':
            assert report['execution_comparisons'][prefix+name]['reference']==prefix+'compact'
            assert report['execution_comparisons'][prefix+name]['final_state_differences']['x_max_abs']<2e-11
    assert cases[prefix+'shared']['interaction_quadrature']['stencil_storage_bytes']*3==cases[prefix+'compact']['interaction_quadrature']['stencil_storage_bytes']*2
    assert cases[prefix+'shared-warm']['stokes_pressure_warm_starts']>0
    assert path.read_bytes()==original
    # A real numeric checkpoint without the new metadata must keep legacy
    # execution, rather than silently opting into a new pressure initial guess.
    from afsi_torch.mac.checkpoint import digest
    with np.load(path,allow_pickle=False) as archive:
        data = {name:archive[name] for name in archive.files}
    metadata = json.loads(str(data.pop('metadata')))
    metadata.pop('sha256')
    for group in (metadata['settings'],metadata['config']):
        group['coupling'].pop('stokes_warm_start')
        group['interaction_quadrature'].pop('stencil_backend')
    metadata['sha256'] = digest(metadata,data)
    legacy = tmp_path/'legacy.npz'
    np.savez_compressed(legacy,metadata=json.dumps(metadata),**data)
    legacy_model,_,legacy_settings,_,legacy_config = load_real_lv(legacy)
    assert legacy_config.interaction_quadrature.stencil_backend=='component'
    assert not legacy_config.coupling.stokes_warm_start
    legacy_driver = build_driver(legacy_model,legacy_settings,'cpu')
    assert legacy_driver.transfer.quadrature_options.stencil_backend=='component'
    assert not legacy_driver.options.stokes_warm_start


def test_demo_cli_records_options_and_rejects_unused_pressure_flag(tmp_path):
    from demo.real_lv_fsi.run_mac import main
    path = tmp_path/'config.json'
    main(['--ib-stencil-backend','shared','--write-config',str(path)])
    config = json.loads(path.read_text())
    assert config['interaction_quadrature']['stencil_backend']=='shared'
    assert config['nonlinear_solver']=='jfnk'  # BE-BE; no midpoint Stokes option
    with pytest.raises(SystemExit) as error:
        main(['--coupling','cnab-midpoint','--stokes-warm-start','--write-config',str(path)])
    assert error.value.code==2


@pytest.mark.skipif(os.environ.get('TRITON_INTERPRET')!='1',reason='optional actual-kernel CPU interpreter')
def test_actual_shared_cuda_kernels_with_tail_mask_and_all_components():
    triton = pytest.importorskip('triton')
    from afsi_torch.mac._triton_ib import _gather,_spread
    grid = MACGrid((16,16,16),(16.,20.,24.),(-1.3,-2.1,-.7))
    # Thirteen points also test padded programs and each axis's lattice mapping.
    p = torch.arange(13,dtype=torch.float64)
    points = torch.stack((3.1+.07*p,4.3+.09*p,5.5+.06*p),-1)
    origins = points.new_tensor([grid.origin,tuple(o+.5*h for o,h in zip(grid.origin,grid.spacing))])
    base,phi = _prepare(points,origins,points.new_tensor(grid.spacing),torch.arange(4))
    stencil = SharedStencil(base.contiguous(),phi.contiguous())
    _,reference = transfer('cpu')
    reference.grid = grid
    coefficient = torch.sin(1.7*points)
    velocity = tuple(torch.cos(grid.coordinates(c)[...,c]) for c in range(3))
    expected = reference.spread_grid(coefficient,stencil.expanded())
    gathered = reference.gather_grid(velocity,stencil.expanded())
    result,out = grid.zeros(),torch.empty_like(points)
    for c,u in enumerate(velocity):
        _spread[(triton.cdiv(13,4),)](base,phi,coefficient,result[c],13,*grid.face_shape(c)[1:],c,grid.volume,4,SHARED=True)
        _gather[(triton.cdiv(13,4),)](base,phi,u,out,13,*grid.face_shape(c)[1:],c,4,SHARED=True)
        torch.testing.assert_close(result[c],expected[c],rtol=3e-12,atol=3e-13)
    torch.testing.assert_close(out,gathered,rtol=3e-12,atol=3e-13)


def test_new_options_reject_unsupported_combinations():
    from afsi_torch.mac.implicit import MACCouplingOptions
    for opts in (dict(mode='adaptive',stencil_backend='bad'),
                 dict(stencil_backend='shared'),dict(mode='adaptive',stencil_backend='shared',transfer_backend='cell')):
        with pytest.raises(ValueError):
            InteractionQuadratureOptions(**opts)
    with pytest.raises(ValueError):
        MACCouplingOptions(stokes_warm_start=1)
