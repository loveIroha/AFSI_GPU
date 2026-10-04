"""Reduced midpoint CSR Jacobian, nonlinear acceptance, stiffness and restart."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from test_real_lv import DEVICES, real_case
from test_mac_implicit import settings
from afsi_torch.real_lv import imported_model, RealLVConfig
from afsi_torch.mac.implicit import MACCouplingOptions
from afsi_torch.mac.execution import build_driver
from afsi_torch.mac.cnab import MACCNABFlow, blend
from afsi_torch.mac.semiimplicit import MidpointProblem, SemiImplicitMACIBStepper
from afsi_torch.mac.grid import MACGrid, zero_normal, gradient, velocity_laplacian
from afsi_torch.nonlinear import NonlinearFailure


def config(real_case):
    return replace(real_case, coupling=MACCouplingOptions(scheme='cnab-semiimplicit'),
                   output=replace(real_case.output,write_vtk=False))


@pytest.mark.parametrize('device', DEVICES)
def test_reduced_csr_action_and_actual_coupled_midpoint_equations(real_case,device):
    cfg = config(real_case)
    model = imported_model(cfg,device)
    driver = build_driver(model,settings(cfg),device)
    s = driver.initialize(model.mesh.X)
    center = s.x.new_tensor([7.5,7.5,7.5])
    x = center+(s.x-center)*1.005
    raw = zero_normal(tuple(.1*torch.sin(driver.flow.grid.coordinates(c,device=device)[...,(c+1)%3]) for c in range(3)))
    velocity = driver.flow.project(raw).velocity
    s = replace(s,step=6000,time=.6,x=x,velocity=velocity,force_time=.6,force=model.force(x,.6),
                previous_advection=driver.flow.advection(velocity))
    U,_ = driver.transfer.interpolate(s.velocity,driver.transfer.prepare(x))
    predicted = x+.5*driver.flow.dt*U
    stencil = driver.transfer.prepare(predicted)
    problem = MidpointProblem(driver,s,predicted,stencil,driver.flow.advection(s.velocity))
    y = torch.zeros_like(x)
    v = .01*torch.sin(1.73*x)
    action = problem.linearization(y)
    exact = action(v)
    assert problem.tangent.layout == torch.sparse_csr
    assert exact.shape==x.shape and exact.numel()==3*len(x)
    finite = (problem.residual(y+1e-4*v)-problem.residual(y-1e-4*v))/2e-4
    torch.testing.assert_close(exact,finite,rtol=2e-5,atol=2e-8)
    torch.testing.assert_close(action(1e-9*v),1e-9*exact,rtol=2e-9,atol=2e-18)
    a,info = driver.step(s)
    xm = .5*(s.x+a.x)
    force = model.force(xm,s.time+.5*driver.flow.dt)
    density,_ = driver.transfer.spread(force,stencil)
    half_adv = driver.flow.advection(s.velocity)
    rhs = driver.flow._right(s.velocity,half_adv,density)
    metrics = driver.flow._stokes_metrics(a.velocity,a.pressure,rhs)
    assert metrics[0] <= 1.01*info['flow']['stokes']['momentum_tolerance']
    assert metrics[1] <= 1.01*info['flow']['stokes']['divergence_tolerance']
    U,_ = driver.transfer.interpolate(blend(s.velocity,a.velocity),stencil)
    assert torch.linalg.vector_norm(a.x-s.x-driver.flow.dt*U) <= 2.01*info['nonlinear']['tolerance']
    assert info['nonlinear']['iterations']>0 and info['nonlinear']['unknown_dofs']==x.numel()
    assert info['nonlinear']['residual_norm']<=info['nonlinear']['tolerance']
    assert info['power_error']/max(abs(info['solid_power']),abs(info['fluid_power']),1.) < 1e-9
    assert a.force_time==pytest.approx(a.time)


@pytest.mark.parametrize('device', DEVICES)
def test_reference_and_optimized_semiimplicit_same_equations(real_case,device):
    cfg = config(real_case)
    model = imported_model(cfg,device)
    a_driver = build_driver(model,settings(cfg),device)
    execution = RealLVConfig().execution
    if device=='cpu':
        execution = replace(execution,pressure_backend='workspace')
    b_driver = build_driver(model,settings(replace(cfg,execution=execution)),device)
    a,b = a_driver.initialize(model.mesh.X),b_driver.initialize(model.mesh.X)
    for _ in range(3):
        a,ia = a_driver.step(a)
        b,ib = b_driver.step(b)
        torch.testing.assert_close(a.x,b.x,rtol=1e-9,atol=1e-11)
        for u,v in zip(a.velocity,b.velocity):
            torch.testing.assert_close(u,v,rtol=2e-8,atol=1e-10)
        assert ib['nonlinear']['residual_norm']<=ib['nonlinear']['tolerance']


@pytest.mark.parametrize('solver',['newton','anderson-newton'])
def test_checkpoint_continuation_reset_and_recovery_snapshot(real_case,tmp_path,solver):
    from afsi_torch.simulation.real_lv_mac import run
    from afsi_torch.real_lv_checkpoint import load_real_lv,refine_checkpoint_dt
    from validation.diagnose_real_lv_guard import diagnose
    from afsi_torch.transport import semiimplicit_policy
    cfg = config(real_case)
    cfg = replace(cfg,coupling=replace(cfg.coupling,semiimplicit_solver=solver))
    whole,split = tmp_path/'whole',tmp_path/'split'
    run(case_config=replace(cfg,time=replace(cfg.time,end_time=5e-4)),device='cpu',output=whole)
    run(case_config=cfg,device='cpu',output=split)
    report = run(resume=split/'checkpoint.npz',device='cpu',end_time=5e-4)
    model,a,sa,_,ca = load_real_lv(whole/'checkpoint.npz')
    _,b,_,_,_ = load_real_lv(split/'checkpoint.npz')
    assert report['completed'] and not report['last_solver_info']['startup_predictor_corrector']
    assert report['coupled_unknown_dofs']==3*len(a.x)
    torch.testing.assert_close(a.x,b.x,rtol=1e-12,atol=1e-13)
    for u,v in zip(a.velocity,b.velocity):
        torch.testing.assert_close(u,v,rtol=1e-9,atol=1e-12)
    _,recovery,_,_,_ = load_real_lv(split/'recovery_checkpoint.npz')
    assert recovery.step==4 and b.step==5
    branch,_,_,details = refine_checkpoint_dt(model,a,sa,ca,5e-5)
    assert branch.previous_advection is None and details['multistep_history_reset']
    assert not details['lagged_force_resampled']
    assert diagnose(split/'checkpoint.npz')['transport_policy']==semiimplicit_policy()


@pytest.mark.parametrize('solver',['newton','anderson-newton'])
def test_failed_final_check_is_transactional_and_retains_local_diagnostics(real_case,monkeypatch,solver):
    import afsi_torch.mac.semiimplicit as module
    cfg = config(real_case)
    cfg = replace(cfg,coupling=replace(cfg.coupling,semiimplicit_solver=solver))
    model = imported_model(cfg)
    driver = build_driver(model,settings(cfg),'cpu')
    state,_ = driver.step(driver.initialize(model.mesh.X))
    state = replace(state,step=6000,time=.6,force_time=.6,force=model.force(state.x,.6))
    saved = tuple(v.clone() for v in (state.x,state.pressure,state.force,*state.velocity,*state.previous_advection))
    original = module.newton
    def incorrect(*args,**kwargs):
        result = original(*args,**kwargs)
        return replace(result,x=result.x+1e-6)
    if solver=='newton':
        monkeypatch.setattr(module,'newton',incorrect)
    else:
        import afsi_torch.mac.midpoint_solver as acceleration
        original_accelerated = acceleration.accelerated_midpoint
        def incorrect_acceleration(*args,**kwargs):
            result,info = original_accelerated(*args,**kwargs)
            return replace(result,x=result.x+1e-6),info
        monkeypatch.setattr(acceleration,'accelerated_midpoint',incorrect_acceleration)
    with pytest.raises(NonlinearFailure,match='final midpoint') as failure:
        driver.step(state)
    local = failure.value.coupled_diagnostics['local']
    assert local['accepted_step']==6000 and len(local['trial_velocity_peaks'])==3
    assert set(local['force_components'])=={'total','passive_without_volume','volume','active','follower','basal'}
    assert local['force_decomposition_max_abs_dyn']<1e-6
    assert local['force_components']['active']['norm_dyn']>0
    for old,current in zip(saved,(state.x,state.pressure,state.force,*state.velocity,*state.previous_advection)):
        torch.testing.assert_close(old,current,atol=0,rtol=0)


def test_benchmark_counts_reduced_unknown_and_nested_costs(real_case,tmp_path):
    from afsi_torch.simulation.real_lv_mac import run
    from validation.benchmark_real_lv_schemes import benchmark
    folder = tmp_path/'case'
    run(case_config=config(real_case),device='cpu',output=folder)
    report = benchmark(folder/'checkpoint.npz',device='cpu',schemes=('cnab-semiimplicit',),warmup=1,steps=2)
    case = report['cases']['cnab-semiimplicit']
    assert case['completed'] and case['measured_steps']==2
    assert case['counts']['mass_solves']>6 and case['counts']['pressure_solves']>0
    from afsi_torch.real_lv_checkpoint import load_real_lv
    _,state,_,_,_ = load_real_lv(folder/'checkpoint.npz')
    assert case['outer_unknown_dofs']==state.x.numel()
    assert case['tangent_assemblies']==case['newton_iterations']>0
    assert case['jacobian_actions']>=case['gmres_iterations']


def test_benchmark_reference_restarts_high_cfl_checkpoint_without_modifying_it(real_case,tmp_path):
    from afsi_torch.real_lv_checkpoint import save_real_lv
    from validation.benchmark_real_lv_schemes import benchmark
    cfg = config(real_case)
    model = imported_model(cfg)
    opts = settings(cfg)
    driver = build_driver(model,opts,'cpu')
    initial = driver.initialize(model.mesh.X)
    velocity = zero_normal(tuple(torch.full_like(v,10000.) for v in initial.velocity))
    source = replace(initial,step=1,time=cfg.time.dt,velocity=velocity,
                     force_time=cfg.time.dt,force=model.force(initial.x,cfg.time.dt),
                     previous_advection=driver.flow.advection(velocity))
    checkpoint = tmp_path/'high_cfl.npz'
    save_real_lv(checkpoint,model,source,opts,{},cfg)
    original = checkpoint.read_bytes()
    kwargs = dict(device='cpu',schemes=('cnab-semiimplicit',),
                  nonlinear_solvers=('newton','anderson-newton'),warmup=0,steps=2)
    saved = benchmark(checkpoint,**kwargs)
    assert saved['initial_state']=='checkpoint' and saved['start_time_s']==source.time
    for case in saved['cases'].values():
        assert not case['completed'] and 'CFL' in case['failure']['message']
        assert case['counts']['stokes_solves']==0
    reference = benchmark(checkpoint,initial_state='reference',**kwargs)
    assert reference['initial_state']=='reference' and reference['start_time_s']==0.
    assert reference['source_checkpoint_step']==source.step
    assert reference['source_checkpoint_time_s']==source.time
    assert reference['dt_s']==cfg.time.dt
    for case in reference['cases'].values():
        assert case['completed'] and case['measured_steps']==2
        assert case['end_time_s']==pytest.approx(2*cfg.time.dt)
        assert case['outer_unknown_dofs']==source.x.numel()
        assert case['final_solid']['max_total_displacement_cm']<1e-4
    assert checkpoint.read_bytes()==original


@pytest.mark.parametrize('device',DEVICES)
def test_acceleration_solves_same_active_ho_equations_with_fewer_stokes(real_case,device):
    cfg = config(real_case)
    model = imported_model(cfg,device)
    slow = build_driver(model,settings(cfg),device)
    fast = build_driver(model,settings(replace(cfg,coupling=replace(cfg.coupling,
                        semiimplicit_solver='anderson-newton'))),device)
    state = slow.initialize(model.mesh.X)
    center = state.x.new_tensor([7.5,7.5,7.5])
    x = center+(state.x-center)*1.002
    state = replace(state,step=6000,time=.6,x=x,force_time=.6,force=model.force(x,.6),
                    previous_advection=slow.flow.advection(state.velocity))
    a,ia = slow.step(state)
    b,ib = fast.step(state)
    torch.testing.assert_close(a.x,b.x,atol=3e-9,rtol=1e-10)
    for u,v in zip(a.velocity,b.velocity):
        torch.testing.assert_close(u,v,atol=1e-7,rtol=5e-6)
    assert ib['nonlinear']['residual_norm']<=ib['nonlinear']['tolerance']
    assert ib['nonlinear']['stokes_solves']<ia['nonlinear']['stokes_solves']
    assert ib['nonlinear']['tangent_assemblies']==0 and not ib['nonlinear']['newton_fallback']
    assert ib['power_error']/max(abs(ib['solid_power']),abs(ib['fluid_power']),1.) < 1e-9


def test_solver_only_resume_preserves_ab2_and_benchmark_compares_same_scheme(real_case,tmp_path):
    from afsi_torch.simulation.real_lv_mac import run
    from afsi_torch.real_lv_checkpoint import load_real_lv
    from validation.benchmark_real_lv_schemes import benchmark
    folder,branch = tmp_path/'case',tmp_path/'branch'
    run(case_config=config(real_case),device='cpu',output=folder)
    _,source,_,_,_ = load_real_lv(folder/'checkpoint.npz')
    report = run(resume=folder/'checkpoint.npz',device='cpu',output=branch,
        end_time=3e-4,nonlinear_solver='anderson-newton')
    assert not report['last_solver_info']['startup_predictor_corrector']
    assert report['configuration']['coupling']['semiimplicit_solver']=='anderson-newton'
    _,unchanged,_,_,_ = load_real_lv(folder/'checkpoint.npz')
    torch.testing.assert_close(source.x,unchanged.x,atol=0,rtol=0)
    b = benchmark(folder/'checkpoint.npz',device='cpu',schemes=('cnab-semiimplicit',),
        nonlinear_solvers=('newton','anderson-newton'),warmup=1,steps=2)
    old,new = (b['cases'][f'cnab-semiimplicit/{s}'] for s in ('newton','anderson-newton'))
    assert old['completed'] and new['completed']
    assert new['counts']['stokes_solves']<old['counts']['stokes_solves']
    assert new['counts']['mass_solves']<old['counts']['mass_solves']
    assert new['outer_unknown_dofs']==old['outer_unknown_dofs']


def test_failure_save_preserves_earlier_scheduled_recovery(real_case,tmp_path,monkeypatch):
    from afsi_torch.simulation.real_lv_mac import run
    from afsi_torch.real_lv_checkpoint import load_real_lv
    import json
    folder = tmp_path/'case'
    run(case_config=config(real_case),device='cpu',output=folder)
    original = SemiImplicitMACIBStepper.step
    def fail_after_one(self,state,**kwargs):
        if state.step==3:
            raise RuntimeError('injected step-four failure')
        return original(self,state,**kwargs)
    monkeypatch.setattr(SemiImplicitMACIBStepper,'step',fail_after_one)
    with pytest.raises(RuntimeError,match='step-four'):
        run(resume=folder/'checkpoint.npz',device='cpu',end_time=5e-4)
    _,last,_,_,_ = load_real_lv(folder/'checkpoint.npz')
    _,earlier,_,_,_ = load_real_lv(folder/'recovery_checkpoint.npz')
    report = json.loads((folder/'report.json').read_text())
    assert report['status']=='failed' and last.step==3 and earlier.step==2


@pytest.mark.parametrize('solver',['newton','anderson-newton'])
def test_stiff_adjoint_modal_spring_midpoint_energy_while_explicit_force_grows(solver):
    # Manufactured rank-one ADJOINT transfer on the actual wall Stokes grid.
    # This isolates temporal stiffness; it is not a spatial IB accuracy test.
    from afsi_torch.mac.cnab import MidpointMACIBStepper
    from afsi_torch.fluid.solvers import SolveInfo
    grid = MACGrid((4,)*3,(4.,)*3)
    flow = MACCNABFlow(grid,dt=.001)
    raw = zero_normal(tuple(torch.sin(grid.coordinates(c)[...,(c+1)%3]) for c in range(3)))
    template = flow.project(raw).velocity
    norm = torch.sqrt(grid.volume*sum(u.square().sum() for u in template))
    template = tuple(u/norm for u in template)
    flow.advection = lambda u: grid.zeros()
    X = torch.tensor([[2.,2.,2.]],dtype=torch.float64)
    stiffness = 1e8  # dt*sqrt(k)=10: explicit elastic midpoint is unstable.
    def force(x,t):
        out = torch.zeros_like(x); out[:,0] = -stiffness*(x[:,0]-X[:,0]); return out
    def validate(x):
        if not torch.isfinite(x).all():
            raise ValueError('nonfinite manufactured spring')
    class Transfer:
        warm_start = True
        def __init__(self): self.grid = grid
        def prepare(self,x): return None
        def interaction_points(self,x): return x
        def check_support(self,x): pass
        def interpolate(self,u,stencil):
            out = torch.zeros_like(X); out[0,0] = grid.volume*sum((a*b).sum() for a,b in zip(u,template))
            return out,SolveInfo(0,0.,0.,0.)
        def spread(self,f,stencil):
            return tuple(f[0,0]*u for u in template),SolveInfo(0,0.,0.,0.)
    transfer = Transfer()
    explicit = MidpointMACIBStepper(flow,transfer,force,validate)
    reduced = object.__new__(SemiImplicitMACIBStepper)
    MidpointMACIBStepper.__init__(reduced,flow,transfer,force,validate)
    reduced.model = SimpleNamespace(mesh=SimpleNamespace(X=X))
    reduced.options = MACCouplingOptions(scheme='cnab-semiimplicit',semiimplicit_solver=solver)
    T = torch.diag(X.new_tensor([-stiffness,0.,0.])).to_sparse_csr()
    reduced.tangent = SimpleNamespace(assemble=lambda x,t:T,diagonal=lambda T:X.new_tensor([-stiffness,0.,0.]))
    reduced.lumped_mass = torch.ones_like(X)
    state = replace(reduced.initialize(X),x=X+X.new_tensor([[1e-5,0.,0.]]),
                    previous_advection=grid.zeros())
    energy = lambda s: .5*grid.volume*sum(u.square().sum() for u in s.velocity)+.5*stiffness*(s.x[0,0]-X[0,0])**2
    e0 = energy(state)
    bad,_ = explicit.step(state)
    assert energy(bad) > 100*e0
    for _ in range(12):
        state,info = reduced.step(state)
        assert energy(state) <= e0*(1+1e-6)
        assert info['nonlinear']['residual_norm']<=info['nonlinear']['tolerance']
