"""Compact positive cubature and exact-point acceptance response reuse."""
from dataclasses import replace, asdict
from math import factorial
import pytest
import torch
from test_real_lv import real_case, DEVICES
from test_mac_implicit import settings
from test_adaptive_p1_transfer import transfer
from afsi_torch.mac.adaptive_transfer import gaussian_tetra_rule, InteractionQuadratureOptions
from afsi_torch.mac.implicit import MACCouplingOptions
from afsi_torch.real_lv import imported_model
from afsi_torch.mac.execution import build_driver
from afsi_torch.mac.semiimplicit import MidpointProblem


@pytest.mark.parametrize('order,count',[(2,6),(3,14),(4,31),(5,57),(6,95),(7,146),(8,214)])
def test_compact_positive_rules_integrate_requested_degree_and_mass(order,count):
    pytest.importorskip('basix')
    q,w = gaussian_tetra_rule(order,torch.zeros((),dtype=torch.float64),'xiao-gimbutas')
    assert len(w)==count and (w>0).all() and (q>=0).all() and (q.sum(-1)<=1+1e-14).all()
    degree = 2*order-1
    for a in range(degree+1):
        for b in range(degree+1-a):
            for c in range(degree+1-a-b):
                exact = factorial(a)*factorial(b)*factorial(c)/factorial(a+b+c+3)
                value = (w*q[:,0]**a*q[:,1]**b*q[:,2]**c).sum().item()
                assert value==pytest.approx(exact,rel=4e-12,abs=2e-16)
    N = torch.cat((1-q.sum(-1,keepdim=True),q),-1)
    expected = (torch.ones((4,4),dtype=q.dtype)+torch.eye(4,dtype=q.dtype))/120
    torch.testing.assert_close(N.T@(w[:,None]*N),expected,rtol=5e-13,atol=2e-15)


def test_compact_count_for_supplied_real_mesh_histogram_and_explicit_high_order_fallback():
    pytest.importorskip('basix')
    like = torch.zeros((),dtype=torch.float64)
    counts = {3:1634,4:123506,5:10268,6:22}
    assert sum(c*len(gaussian_tetra_rule(n,like,'xiao-gimbutas')[1]) for n,c in counts.items())==4438928
    q,w = gaussian_tetra_rule(9,like,'xiao-gimbutas')
    a,b = gaussian_tetra_rule(9,like,'conical')
    torch.testing.assert_close(q,a,atol=0,rtol=0)
    torch.testing.assert_close(w,b,atol=0,rtol=0)


@pytest.mark.parametrize('device',DEVICES)
def test_compact_transfer_affine_field_work_and_same_reference_csr(device):
    pytest.importorskip('basix')
    X,old = transfer(device)
    _,new = transfer(device,rule_family='xiao-gimbutas')
    x = X+X.new_tensor([.1,.2,-.1])
    stencil = new.prepare(x)
    assert stencil.rule.point_count==20
    torch.testing.assert_close(new.mass.values(),old.mass.values(),atol=0,rtol=0)
    A = X.new_tensor([[.01,.02,0.],[0.,.01,-.02],[.03,0.,.04]])
    b = X.new_tensor([.1,-.2,.3])
    field = tuple((new.grid.coordinates(c,device=device)@A.T+b)[...,c] for c in range(3))
    U,_ = new.interpolate(field,stencil)
    torch.testing.assert_close(U,x@A.T+b,rtol=2e-11,atol=2e-12)
    force = torch.sin(1.7*X)
    density,_ = new.spread(force,stencil)
    torch.testing.assert_close(new.grid.volume*torch.stack([v.sum() for v in density]),force.sum(0),rtol=2e-11,atol=2e-12)
    torch.testing.assert_close((force*U).sum(),new.grid.volume*sum((u*f).sum() for u,f in zip(field,density)),rtol=2e-11,atol=2e-12)


def adaptive_config(real_case,reuse=True):
    return replace(real_case,
        coupling=MACCouplingOptions(scheme='cnab-semiimplicit',semiimplicit_solver='anderson-newton',reuse_final_evaluation=reuse),
        interaction_quadrature=InteractionQuadratureOptions(mode='adaptive'),
        output=replace(real_case.output,write_vtk=False))


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('solver',['newton','anderson-newton'])
def test_final_response_reuse_preserves_equations_and_saves_one_stokes(real_case,device,solver):
    cfg = adaptive_config(real_case,False)
    cfg = replace(cfg,coupling=replace(cfg.coupling,semiimplicit_solver=solver))
    model = imported_model(cfg,device)
    slow = build_driver(model,settings(cfg),device)
    fast = build_driver(model,settings(replace(cfg,coupling=replace(cfg.coupling,reuse_final_evaluation=True))),device)
    state,_ = slow.step(slow.initialize(model.mesh.X))
    a,ia = slow.step(state)
    b,ib = fast.step(state)
    assert not ia['nonlinear']['final_evaluation_reused'] and ib['nonlinear']['final_evaluation_reused']
    assert ib['nonlinear']['stokes_solves']==ia['nonlinear']['stokes_solves']-1
    assert ib['nonlinear']['residual_evaluations']==ia['nonlinear']['residual_evaluations']-1
    assert ib['nonlinear']['residual_norm']<=ib['nonlinear']['tolerance']
    torch.testing.assert_close(a.x,b.x,atol=2e-12,rtol=1e-12)
    for u,v in zip(a.velocity,b.velocity):
        torch.testing.assert_close(u,v,atol=2e-10,rtol=2e-8)
    assert ib['power_error']/max(abs(ib['solid_power']),abs(ib['fluid_power']),1.)<1e-9


def test_cache_rejects_changed_point_in_place_and_intervening_linear_action(real_case):
    cfg = adaptive_config(real_case)
    model = imported_model(cfg)
    driver = build_driver(model,settings(cfg),'cpu')
    state = driver.initialize(model.mesh.X)
    problem = MidpointProblem(driver,state,state.x,driver.transfer.prepare(state.x),driver.flow.advection(state.velocity))
    y = torch.zeros_like(state.x)
    first = problem.evaluate(y)
    result,reused = problem.final_evaluation(y.clone())
    assert reused and result is first and problem.evaluations==1
    y.add_(1e-6)
    _,reused = problem.final_evaluation(y)
    assert not reused and problem.evaluations==2
    action = problem.linearization(y)
    action(torch.sin(state.x))
    _,reused = problem.final_evaluation(y)
    assert not reused and problem.evaluations==3
    driver.flow.stokes_calls += 1
    _,reused = problem.final_evaluation(y)
    assert not reused and problem.evaluations==4


def test_benchmark_variants_keep_checkpoint_read_only_and_report_changes(real_case,tmp_path):
    pytest.importorskip('basix')
    from afsi_torch.simulation.real_lv_mac import run
    from validation.benchmark_real_lv_schemes import benchmark
    path = tmp_path/'source'
    run(case_config=adaptive_config(real_case),device='cpu',output=path)
    checkpoint = path/'checkpoint.npz'
    original = checkpoint.read_bytes()
    report = benchmark(checkpoint,device='cpu',schemes=('cnab-semiimplicit',),
        nonlinear_solvers=('anderson-newton',),execution_variants=('baseline','reuse','compact'),warmup=0,steps=2)
    base,reuse,compact = [report['cases'][f'cnab-semiimplicit/anderson-newton/{v}'] for v in ('baseline','reuse','compact')]
    assert all(v['completed'] for v in (base,reuse,compact))
    assert reuse['counts']['stokes_solves']==base['counts']['stokes_solves']-2
    assert reuse['counts']['mass_solves']==base['counts']['mass_solves']-4
    assert reuse['final_evaluation_reuses']==2
    assert compact['interaction_quadrature']['point_count']<base['interaction_quadrature']['point_count']
    comparison = report['execution_comparisons']['cnab-semiimplicit/anderson-newton/reuse']
    assert comparison['final_state_differences']['x_max_abs']<2e-12
    assert checkpoint.read_bytes()==original


@pytest.mark.parametrize('device',DEVICES)
def test_compact_coupled_tangent_at_active_phase(real_case,device):
    pytest.importorskip('basix')
    cfg = adaptive_config(real_case)
    cfg = replace(cfg,interaction_quadrature=replace(cfg.interaction_quadrature,rule_family='xiao-gimbutas'))
    model = imported_model(cfg,device)
    driver = build_driver(model,settings(cfg),device)
    state = driver.initialize(model.mesh.X)
    state = replace(state,step=6000,time=.6,force_time=.6,force=model.force(state.x,.6))
    problem = MidpointProblem(driver,state,state.x,driver.transfer.prepare(state.x),driver.flow.advection(state.velocity))
    y,v = torch.zeros_like(state.x),.01*torch.sin(1.7*state.x)
    action = problem.linearization(y)
    finite = (problem.residual(y+1e-4*v)-problem.residual(y-1e-4*v))/2e-4
    torch.testing.assert_close(action(v),finite,rtol=2e-5,atol=2e-8)
    assert problem.tangent.layout==torch.sparse_csr


def test_compact_demo_checkpoint_retains_rule_family_and_cached_acceptance(real_case,tmp_path):
    pytest.importorskip('basix')
    from afsi_torch.simulation.real_lv_mac import run
    from afsi_torch.real_lv_checkpoint import load_real_lv
    cfg = adaptive_config(real_case)
    cfg = replace(cfg,interaction_quadrature=replace(cfg.interaction_quadrature,rule_family='xiao-gimbutas'))
    folder = tmp_path/'compact'
    run(case_config=cfg,device='cpu',output=folder)
    _,_,opts,_,restored = load_real_lv(folder/'checkpoint.npz')
    assert restored.interaction_quadrature==cfg.interaction_quadrature
    assert opts['interaction_quadrature']==asdict(cfg.interaction_quadrature)
    report = run(resume=folder/'checkpoint.npz',device='cpu',end_time=3e-4)
    assert report['completed'] and report['interaction_quadrature']['rule_family']=='xiao-gimbutas'
    assert report['last']['final_evaluation_reused']
