"""CN Stokes wall equations, temporal accuracy, midpoint coupling and restart."""
from dataclasses import replace
import math
import pytest
import torch
from afsi_torch.mac.grid import MACGrid, zero_normal, gradient, divergence, velocity_laplacian
from afsi_torch.mac.cnab import MACCNABFlow, CNABOptions
from afsi_torch.mac.ppm import parabolic_states, convection_ppm
from afsi_torch.mac.implicit import MACCouplingOptions
from afsi_torch.mac.execution import build_driver
from afsi_torch.transport import cnab_policy
from test_real_lv import DEVICES, real_case
from test_mac_implicit import settings


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('dt',[1e-4,.04])
def test_cn_stokes_solves_wall_momentum_and_divergence(device,dt):
    grid = MACGrid((4,)*3,(1.,1.2,1.4))
    flow = MACCNABFlow(grid,dt=dt,device=device)
    generator = torch.Generator(device=device).manual_seed(24)
    raw = zero_normal(tuple(torch.randn(v.shape,device=device,dtype=v.dtype,generator=generator)
                            for v in grid.zeros(device=device)))
    expected = flow.project(raw).velocity
    p = torch.randn(grid.shape,device=device,dtype=torch.float64,generator=generator); p -= p.mean()
    gp = gradient(p,grid.spacing)
    b = tuple(u-flow.alpha*velocity_laplacian(u,c,grid.spacing)+dt*g/flow.rho
              for c,(u,g) in enumerate(zip(expected,gp)))
    result = flow.stokes(b)
    for c,(u,v,g,r) in enumerate(zip(result.velocity,expected,gradient(result.pressure,grid.spacing),b)):
        torch.testing.assert_close(u,v,atol=1e-8,rtol=1e-8)
        residual = u-flow.alpha*velocity_laplacian(u,c,grid.spacing)+dt*g/flow.rho-r
        assert torch.linalg.vector_norm(residual).item() < 2e-8
    torch.testing.assert_close(result.pressure,p,atol=1e-6,rtol=1e-6)
    assert torch.linalg.vector_norm(divergence(result.velocity,grid.spacing)).item() < 1e-8
    s = result.diagnostics['stokes']
    assert s['momentum_residual'] <= s['momentum_tolerance']
    assert s['divergence_norm'] <= s['divergence_tolerance']
    # A single projection of a Helmholtz predictor is not this Stokes solve.
    if dt == .04:
        naive = flow.project(flow.helmholtz(b))
        assert flow._stokes_metrics(naive.velocity,naive.pressure,b)[0].item() > 1e-3
        assert flow.viscous_number > .25


def test_ppm_cell_monotonicity_and_no_slip_fluxes():
    q = torch.tensor([0.,0.,1.,1.,1.,0.,0.,0.],dtype=torch.float64).reshape(8,1,1)
    ql,qr = parabolic_states(q,0)
    assert ql.min() >= 0 and ql.max() <= 1
    assert qr.min() >= 0 and qr.max() <= 1
    grid = MACGrid((8,)*3,(8.,)*3)
    vel = grid.zeros()
    assert all(a.count_nonzero()==0 for a in convection_ppm(vel,grid.spacing))
    rng = torch.Generator().manual_seed(7)
    vel = zero_normal(tuple(torch.randn(u.shape,dtype=u.dtype,generator=rng) for u in vel))
    adv = convection_ppm(vel,grid.spacing)
    for c,a in enumerate(adv):
        assert torch.isfinite(a).all()
        assert a.select(c,0).count_nonzero()==0 and a.select(c,a.shape[c]-1).count_nonzero()==0


def test_cn_ab2_second_order_time_accuracy_with_exact_spatial_forcing():
    # Manufactured semi-discrete solution on the actual no-slip MAC grid.
    # Frozen template times exp(t); source cancels N and supplies u'-nu L u.
    grid = MACGrid((4,)*3,(4.,)*3)
    base = MACCNABFlow(grid,dt=.01)
    raw = zero_normal(tuple(torch.sin(grid.coordinates(c)[...,(c+1)%3]) for c in range(3)))
    template = base.project(raw).velocity
    lap = tuple(velocity_laplacian(u,c,grid.spacing) for c,u in enumerate(template))
    errors = []
    final_time = .5
    for dt in (.01,.005,.0025):
        flow = MACCNABFlow(grid,dt=dt,cnab_options=CNABOptions(advection='centered'))
        # Linear transport N(u)=u, exercising genuine AB2 history independently
        # of errors in a nonlinear source evaluation.
        velocity = tuple(u.clone() for u in template)
        previous = None
        for n in range(round(final_time/dt)):
            half = (n+.5)*dt
            density = tuple(math.exp(half)*(2*u-l) for u,l in zip(template,lap))
            if previous is None:
                # Midpoint startup transport value, as in predictor/corrector.
                old_density = tuple(2*u-l for u,l in zip(template,lap))
                provisional = flow.advance(velocity,old_density,velocity).velocity
                adv = tuple(.5*(u+v) for u,v in zip(velocity,provisional))
            else:
                adv = tuple(1.5*u-.5*v for u,v in zip(velocity,previous))
            previous = velocity
            velocity = flow.advance(velocity,density,adv).velocity
        errors.append(math.sqrt(sum((u-math.exp(final_time)*v).square().sum().item() for u,v in zip(velocity,template))))
    assert errors[0]/errors[1] > 3.5 and errors[1]/errors[2] > 3.5


@pytest.mark.parametrize('device',DEVICES)
def test_midpoint_force_times_adjoint_power_and_reference_execution(real_case,device):
    from afsi_torch.real_lv import imported_model, RealLVConfig
    cfg = replace(real_case,coupling=MACCouplingOptions(scheme='cnab-midpoint'))
    model = imported_model(cfg,device)
    reference = build_driver(model,settings(cfg),device)
    execution = RealLVConfig().execution
    if device=='cpu':
        execution = replace(execution,pressure_backend='workspace')
    optimized = build_driver(model,settings(replace(cfg,execution=execution)),device)
    a,b = reference.initialize(model.mesh.X),optimized.initialize(model.mesh.X)
    old_x = a.x.clone()
    for n in range(3):
        a,ia = reference.step(a)
        b,ib = optimized.step(b)
        for x,y in zip(a.velocity,b.velocity):
            torch.testing.assert_close(x,y,atol=1e-10,rtol=1e-8)
        torch.testing.assert_close(a.x,b.x,atol=1e-11,rtol=1e-9)
        torch.testing.assert_close(a.pressure,b.pressure,atol=1e-8,rtol=1e-8)
        assert a.force_time == pytest.approx(a.time)
        assert ia['used_force_time_s'] == pytest.approx((n+.5)*cfg.time.dt)
        assert ia['startup_predictor_corrector'] == (n==0)
        assert a.previous_advection is not None
        assert ia['power_error']/max(abs(ia['solid_power']),abs(ia['fluid_power']),1.) < 1e-9
        assert ib['flow']['stokes']['momentum_residual'] <= ib['flow']['stokes']['momentum_tolerance']
    assert not torch.equal(a.x,old_x)


def test_cnab_checkpoint_preserves_history_and_equivalent_continuation(real_case,tmp_path):
    from afsi_torch.simulation.real_lv_mac import run
    from afsi_torch.real_lv_checkpoint import load_real_lv, refine_checkpoint_dt
    from validation.diagnose_real_lv_guard import diagnose
    cfg = replace(real_case,coupling=MACCouplingOptions(scheme='cnab-midpoint'),
                  output=replace(real_case.output,write_vtk=False))
    whole,split = tmp_path/'whole',tmp_path/'split'
    run(case_config=replace(cfg,time=replace(cfg.time,end_time=5e-4)),device='cpu',output=whole)
    run(case_config=cfg,device='cpu',output=split)
    _,s,_,_,_ = load_real_lv(split/'checkpoint.npz')
    assert s.previous_advection is not None
    report = run(resume=split/'checkpoint.npz',device='cpu',end_time=5e-4)
    assert report['completed'] and not report['last_solver_info']['startup_predictor_corrector']
    ma,a,_,_,ca = load_real_lv(whole/'checkpoint.npz')
    _,b,_,_,_ = load_real_lv(split/'checkpoint.npz')
    torch.testing.assert_close(a.x,b.x,atol=1e-13,rtol=1e-12)
    for u,v in zip(a.velocity,b.velocity):
        # Consistent mass CG warm guesses are execution caches, not solution
        # history. Restart differences are bounded by the same true tolerances.
        torch.testing.assert_close(u,v,atol=1e-12,rtol=1e-9)
    assert diagnose(split/'checkpoint.npz')['transport_policy']==cnab_policy()
    branch,_,_,info = refine_checkpoint_dt(ma,a,settings(ca),ca,5e-5)
    assert branch.previous_advection is None and info['multistep_history_reset']
    assert not info['lagged_force_resampled']


def test_failed_midpoint_does_not_commit_state_or_ab2_history(real_case):
    from afsi_torch.real_lv import imported_model
    cfg = replace(real_case,coupling=MACCouplingOptions(scheme='cnab-midpoint'))
    model = imported_model(cfg)
    driver = build_driver(model,settings(cfg),'cpu')
    state,_ = driver.step(driver.initialize(model.mesh.X))
    saved = tuple(v.clone() for v in (state.x,state.pressure,state.force,*state.velocity,*state.previous_advection))
    original_force = driver.force
    def failure(x,t):
        if t > state.time:
            raise ValueError('injected midpoint force failure')
        return original_force(x,t)
    driver.force = failure
    with pytest.raises(ValueError,match='injected midpoint'):
        driver.step(state)
    for old,current in zip(saved,(state.x,state.pressure,state.force,*state.velocity,*state.previous_advection)):
        torch.testing.assert_close(old,current,atol=0,rtol=0)
    driver.force = original_force
    accepted,info = driver.step(state)
    assert accepted.step==2 and not info['startup_predictor_corrector']


def test_cnab_benchmark_counts_nested_solves(real_case,tmp_path):
    from afsi_torch.simulation.real_lv_mac import run
    from validation.benchmark_real_lv_schemes import benchmark
    cfg = replace(real_case,coupling=MACCouplingOptions(scheme='cnab-midpoint'),
                  output=replace(real_case.output,write_vtk=False))
    output = tmp_path/'case'
    run(case_config=cfg,device='cpu',output=output)
    report = benchmark(output/'checkpoint.npz',device='cpu',schemes=('cnab-midpoint',),warmup=1,steps=2)
    case = report['cases']['cnab-midpoint']
    assert case['completed'] and case['measured_steps']==2
    assert case['counts']['mass_solves']==6  # predictor, spread, average-velocity interpolation
    assert case['counts']['pressure_solves'] > 0
    assert case['stokes_iterations']==case['counts']['pressure_solves']
