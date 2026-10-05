"""Paper material, open MAC projection and actual endpoint BE-BE coupling."""
from dataclasses import replace
import json
import pytest
import torch
from afsi_torch.paper_lv import PaperHOParameters, InflationLoads, paper_energy, paper_pk1, PaperLVConfig, imported_model
from afsi_torch.mac.grid import MACGrid, divergence
from afsi_torch.mac.open_boundary import gradient, negative_laplacian, velocity_laplacian, face_weights, SemiLagrangian
from afsi_torch.mac.dirichlet_mg import DirichletMultigrid
from afsi_torch.mac.backward_euler import BackwardEulerFlow, BEFlowOptions
from afsi_torch.mac.paper_coupling import BEIBStepper, BEProblem, bicgstab
from afsi_torch.mac.adaptive_transfer import InteractionQuadratureOptions
from afsi_torch.config import TimeConfig, FluidConfig, LVExecutionConfig, OutputConfig, save_config, load_config
from afsi_torch.paper_lv_checkpoint import save, load
from afsi_torch.simulation.paper_lv_mac import run
from test_real_lv import real_case

DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))]


@pytest.mark.parametrize('device', DEVICES)
def test_paper_stress_is_eq82_not_previous_isochoric_law(device):
    p = PaperHOParameters()
    F = torch.tensor([[1.01, .035, .01], [.01, 1.02, .02], [0., .01, .99]], device=device, dtype=torch.float64, requires_grad=True)
    f, s = F.new_tensor([1., .001, 0.]), F.new_tensor([.002, 1., 0.])
    # Independent C-based transcription of eq81; add the printed correction.
    C = F.T@F
    I1, I3 = torch.trace(C), torch.linalg.det(C)
    ef, es = torch.clamp(f@C@f-1, min=0), torch.clamp(s@C@s-1, min=0)
    I8 = f@C@s
    W = (p.a/(2*p.b)*torch.exp(p.b*(I1-3))
        + p.a_f/(2*p.b_f)*(torch.exp(p.b_f*ef**2)-1)
        + p.a_s/(2*p.b_s)*(torch.exp(p.b_s*es**2)-1)
        + p.a_fs/(2*p.b_fs)*(torch.exp(p.b_fs*I8**2)-1))
    expected = torch.autograd.grad(W, F)[0]+(-p.a*torch.exp(p.b*(I1-3))+p.kappa*torch.log(I3))*torch.linalg.inv(F).T
    torch.testing.assert_close(paper_pk1(F, f, s, p), expected, rtol=2e-12, atol=1e-8)
    torch.testing.assert_close(paper_energy(F, f, s, p), W)
    identity = torch.eye(3, device=device, dtype=F.dtype)
    torch.testing.assert_close(paper_pk1(identity, identity[0], identity[1], p), torch.zeros_like(identity), atol=1e-10, rtol=0)


def test_passive_ramp_and_paper_defaults_roundtrip(tmp_path):
    config = PaperLVConfig()
    assert config.fluid.lengths == (13.,)*3 and config.fluid.mu == .04
    assert config.time.end_time == 1.5 and config.loads.at(1.5) == config.loads.at(.8)
    assert config.loads.at(.4)[0] == pytest.approx(4*1333.22387415)
    assert all(config.loads.at(t)[1] == 0 for t in (0, .5, .8, 1.5))
    save_config(tmp_path/'config.json', config)
    assert load_config(tmp_path/'config.json', PaperLVConfig) == config


@pytest.mark.parametrize('device', DEVICES)
def test_open_mac_weighted_adjoint_and_neumann_diffusion(device):
    torch.manual_seed(51)
    grid = MACGrid((8, 8, 8), (3., 4., 5.))
    p = torch.randn(grid.shape, device=device, dtype=torch.float64)
    velocity = tuple(torch.randn_like(u) for u in grid.zeros(device=device))
    gp = gradient(p, grid.spacing)
    left = sum((v*g*face_weights(grid, c, p)).sum() for c, (v, g) in enumerate(zip(velocity, gp)))
    right = -grid.volume*(p*divergence(velocity, grid.spacing)).sum()
    torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-10)
    assert (p*negative_laplacian(p, grid.spacing)).sum() > 0
    assert negative_laplacian(torch.ones_like(p), grid.spacing).abs().sum() > 0 # no constant nullspace
    for c, v in enumerate(velocity):
        weights = face_weights(grid, c, p)
        lap = velocity_laplacian(v, c, grid.spacing)
        torch.testing.assert_close(velocity_laplacian(torch.ones_like(v), c, grid.spacing), torch.zeros_like(v))
        assert (weights*v*lap).sum() < 0
        w = torch.sin(v)
        torch.testing.assert_close((weights*w*lap).sum(),
            (weights*v*velocity_laplacian(w, c, grid.spacing)).sum(), rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize('device', DEVICES)
def test_dirichlet_multigrid_matches_discrete_manufactured_field_and_owns_results(device):
    grid = MACGrid((8, 16, 8), (3., 5., 4.))
    xyz = grid.coordinates(device=device)
    exact = torch.sin(.7*xyz[..., 0])*torch.cos(.9*xyz[..., 1])+.4
    solver = DirichletMultigrid(grid, device=device, backend='graph' if device == 'cuda' else 'torch')
    rhs = negative_laplacian(exact, grid.spacing)
    solution, info = solver.solve(rhs)
    torch.testing.assert_close(solution, exact, rtol=1e-8, atol=1e-8)
    saved = solution.clone()
    zero, _ = solver.solve(torch.zeros_like(rhs))
    torch.testing.assert_close(solution, saved, rtol=0, atol=0)
    torch.testing.assert_close(zero, torch.zeros_like(rhs), rtol=0, atol=0)
    assert info['residual_norm'] <= info['tolerance']


@pytest.mark.parametrize('device', DEVICES)
def test_semi_lagrangian_constant_and_affine_characteristic(device):
    grid = MACGrid((8,)*3, (4.,)*3)
    like = torch.empty((), device=device, dtype=torch.float64)
    transport = SemiLagrangian(grid, like)
    constant = tuple(torch.full(grid.face_shape(c), c+.25, device=device, dtype=like.dtype) for c in range(3))
    for actual, expected in zip(transport(constant, .01), constant):
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-14)
    velocity = tuple(grid.coordinates(c, device=device)[..., c]*.1+.02 for c in range(3))
    for c, actual in enumerate(transport(velocity, .02)):
        expected = velocity[c]*(1-.1*.02)
        torch.testing.assert_close(actual[1:-1, 1:-1, 1:-1], expected[1:-1, 1:-1, 1:-1], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize('device', DEVICES)
def test_be_projection_divergence_and_weighted_energy(device):
    grid = MACGrid((8,)*3, (2., 3., 4.))
    flow = BackwardEulerFlow(grid, dt=.01, mu=.04, device=device, options=BEFlowOptions(convection=False),
        backend='fused' if device == 'cuda' else 'torch', pressure_backend='graph' if device == 'cuda' else 'torch')
    torch.manual_seed(84)
    rhs = tuple(torch.randn_like(u) for u in grid.zeros(device=device))
    velocity, pressure, info = flow.solve_rhs(rhs)
    assert divergence(velocity, grid.spacing).abs().max().item() < 1e-8
    before = sum((u*u*face_weights(grid, c, pressure)).sum() for c, u in enumerate(rhs))
    after = sum((u*u*face_weights(grid, c, pressure)).sum() for c, u in enumerate(velocity))
    assert after < before
    assert all(r <= t for r, t in zip(info['helmholtz_residuals'], info['helmholtz_tolerances']))


def config_for(real_case, solver='anderson-newton'):
    return PaperLVConfig(source_dir=real_case.source_dir, time=TimeConfig(1e-4, 2e-4),
        fluid=FluidConfig((8,)*3, (13.,)*3, rho=1., mu=.04),
        loads=InflationLoads(8., .0002), execution=LVExecutionConfig(warm_start=True),
        interaction_quadrature=InteractionQuadratureOptions(mode='fixed'),
        nonlinear_solver=solver, output=OutputConfig(1, 1, 1, True))


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('solver', ['jfnk', 'newton', 'anderson-newton'])
def test_actual_be_endpoint_equations_and_frozen_old_geometry(real_case, solver, device):
    config = config_for(real_case, solver)
    if device == 'cuda':
        config = replace(config, execution=PaperLVConfig().execution)
    model = imported_model(config, device)
    driver = BEIBStepper(model, config, device)
    old = driver.initialize(model.mesh.X)
    saved = old.x.clone()
    old_velocity = tuple(v.clone() for v in old.velocity)
    new, info = driver.step(old)
    torch.testing.assert_close(old.x, saved, rtol=0, atol=0)
    assert new.force_time == new.time and new.pressure_time == new.time
    stencil = driver.transfer.prepare(old.x)
    nodal, _ = driver.transfer.interpolate(new.velocity, stencil)
    r = new.x-old.x-config.time.dt*nodal
    assert torch.linalg.vector_norm(r).item() <= info['nonlinear']['tolerance']*1.02
    torch.testing.assert_close(new.force, model.force(new.x, new.time), rtol=1e-10, atol=1e-7)
    for u, v in zip(old.velocity, old_velocity):
        torch.testing.assert_close(u, v, rtol=0, atol=0)
    assert new.x.sub(old.x).abs().max() > 0
    assert info['power_error'] < 1e-6
    assert info['divergence_l2'] < 1e-8


def test_paper_csr_tangent_and_coupled_action(real_case):
    config = config_for(real_case)
    model = imported_model(config)
    driver = BEIBStepper(model, config, 'cpu')
    state = driver.initialize(model.mesh.X)
    problem = BEProblem(driver, state, driver.transfer.prepare(state.x), driver.flow.advect(state.velocity))
    # Away from I4=1, where the tension-only stress has a kink and a centred
    # finite difference averages its two one-sided tangents.
    y = .01*(state.x-state.x.new_tensor([7.5, 7.5, 7.5]))
    torch.manual_seed(14)
    v = torch.randn_like(y); v /= torch.linalg.vector_norm(v)
    J = model.tangent_factory().assemble(state.x+y, config.time.dt)
    eps = 2e-6
    finite = (model.force(state.x+y+eps*v, config.time.dt)-model.force(state.x+y-eps*v, config.time.dt))/(2*eps)
    torch.testing.assert_close(torch.sparse.mm(J, v.reshape(-1, 1)).reshape_as(v), finite, rtol=2e-6, atol=3e-4)
    exact = problem.linearization(y)(v)
    finite = (problem.residual(y+eps*v)-problem.residual(y-eps*v))/(2*eps)
    torch.testing.assert_close(exact, finite, rtol=2e-7, atol=2e-9)


def test_paper_run_checkpoint_resume_and_vtk_metadata(real_case, tmp_path):
    config = config_for(real_case)
    output = tmp_path/'run'
    report = run(case_config=replace(config, time=TimeConfig(1e-4, 1e-4)), device='cpu', output=output)
    assert report['status'] == 'completed' and not report['published_results_reproduced']
    model, state, restored, progress = load(output/'checkpoint.npz')
    assert restored.material == config.material and state.previous_x is not None
    assert json.loads((output/'vtk/fields.json').read_text())['pressure_dyn_per_cm2'].endswith('endpoint time')
    resumed = run(device='cpu', resume=output/'checkpoint.npz', end_time=.0002)
    direct = run(case_config=config, device='cpu', output=tmp_path/'direct')
    _, a, _, _ = load(output/'checkpoint.npz')
    _, b, _, _ = load(tmp_path/'direct/checkpoint.npz')
    torch.testing.assert_close(a.x, b.x, rtol=1e-12, atol=1e-12)
    assert resumed['accepted_steps'] == direct['accepted_steps'] == 2
    assert resumed['visualization']['frames'] == 3


def test_bicgstab_checks_true_residual():
    A = torch.tensor([[4., 2., -1.], [-1., 5., 2.], [1., -.5, 3.]], dtype=torch.float64)
    rhs = torch.tensor([2., -1., 4.], dtype=A.dtype)
    from afsi_torch.nonlinear import GMRESOptions
    x, info = bicgstab(lambda v:A@v, rhs, GMRESOptions(rtol=1e-10, atol=1e-12))
    torch.testing.assert_close(x, torch.linalg.solve(A, rhs), rtol=1e-10, atol=1e-10)
    assert info['residual_norm'] <= info['tolerance']


@pytest.mark.parametrize('device', DEVICES)
def test_bicgstab_periodic_checks_reduce_actions_at_same_true_tolerance(device):
    from afsi_torch.nonlinear import GMRESOptions
    generator = torch.Generator().manual_seed(619)
    A = torch.diag(torch.linspace(1., 9., 24, dtype=torch.float64))
    A += .08*torch.randn((24, 24), generator=generator, dtype=A.dtype)
    rhs = torch.randn(24, generator=generator, dtype=A.dtype).to(device)
    A = A.to(device)
    counts = []
    for interval in (1, 5):
        calls = []
        def action(v):
            calls.append(None)
            return A@v
        options = GMRESOptions(rtol=1e-10, atol=1e-12, check_every=interval)
        x, info = bicgstab(action, rhs, options)
        true = torch.linalg.vector_norm(rhs-A@x).item()
        assert true <= info['tolerance']
        assert info['residual_norm'] == pytest.approx(true)
        assert info['jacobian_actions'] == len(calls)
        torch.testing.assert_close(x, torch.linalg.solve(A, rhs), rtol=1e-8, atol=1e-9)
        counts.append(info)
    assert counts[1]['true_residual_checks'] < counts[0]['true_residual_checks']
    assert counts[1]['jacobian_actions'] < counts[0]['jacobian_actions']


def test_bicgstab_restarts_rejected_candidate_and_never_accepts_stale_residual():
    from afsi_torch.nonlinear import GMRESOptions
    rhs = torch.tensor([1., 2., 3.], dtype=torch.float64)
    # Simulate one inaccurate inner response on the first convergence check.
    # Recurrence says zero, but the true-residual action disagrees.
    rejected = []
    def action(v):
        if not rejected and torch.equal(v, rhs):
            rejected.append(True)
            return v+1e-4
        return v.clone()
    x, info = bicgstab(action, rhs, GMRESOptions(rtol=1e-10, atol=1e-12))
    assert rejected and info['residual_restarts'] >= 1
    assert torch.linalg.vector_norm(rhs-action(x)).item() <= info['tolerance']
    torch.testing.assert_close(x, rhs, rtol=1e-10, atol=1e-10)
    with pytest.raises(RuntimeError, match='true residual'):
        # Exhaust the budget exactly at the rejected candidate.
        rejected.clear()
        bicgstab(action, rhs, GMRESOptions(rtol=1e-10, atol=1e-12, max_iterations=1))


def test_bicgstab_zero_rhs_and_nonfinite_inputs():
    from afsi_torch.nonlinear import GMRESOptions
    rhs = torch.zeros(3, dtype=torch.float64)
    x, info = bicgstab(lambda v: v.clone(), rhs, GMRESOptions())
    torch.testing.assert_close(x, rhs, rtol=0, atol=0)
    assert info['iterations'] == 0
    with pytest.raises(ValueError, match='nonfinite'):
        bicgstab(lambda v: v, torch.full_like(rhs, float('nan')), GMRESOptions())


@pytest.mark.parametrize('device', DEVICES)
def test_be_helmholtz_rhs_norms_are_measured_once(device):
    grid = MACGrid((8,)*3, (2., 3., 4.))
    flow = BackwardEulerFlow(grid, dt=.01, mu=.04, device=device)
    calls = []
    measure = flow._metrics
    def counted(u, b):
        calls.append(None)
        return measure(u, b)
    flow._metrics = counted
    rhs = tuple(torch.randn_like(v) for v in grid.zeros(device=device))
    _, _, info = flow.solve_rhs(rhs)
    assert info['helmholtz_sweeps'] > 0
    assert len(calls) == 1
    assert all(r <= t for r, t in zip(info['helmholtz_residuals'], info['helmholtz_tolerances']))


@pytest.mark.parametrize('device', DEVICES)
def test_default_adaptive_fused_paper_path_matches_reference_and_prepares_once(real_case, device):
    pytest.importorskip('basix')
    cfg = config_for(real_case)
    rule = PaperLVConfig().interaction_quadrature
    reference_cfg = replace(cfg, interaction_quadrature=replace(rule, transfer_backend='reference',
        prepare_backend='torch', reuse_stencil_buffers=False))
    fused_cfg = replace(cfg, interaction_quadrature=rule, execution=PaperLVConfig().execution)
    model = imported_model(cfg, device)
    a, b = BEIBStepper(model, reference_cfg, device), BEIBStepper(model, fused_cfg, device)
    left = a.initialize(model.mesh.X); right = b.initialize(model.mesh.X)
    prepare = b.transfer.prepare
    calls = []
    def counted(x):
        calls.append(x.detach().clone())
        return prepare(x)
    b.transfer.prepare = counted
    for _ in range(2):
        old = right.x.clone()
        left, _ = a.step(left); right, info = b.step(right)
        torch.testing.assert_close(calls[-1], old, rtol=0, atol=0)
        torch.testing.assert_close(right.x, left.x, rtol=2e-10, atol=2e-11)
        for u, v in zip(right.velocity, left.velocity):
            torch.testing.assert_close(u, v, rtol=2e-6, atol=2e-9)
        assert info['nonlinear']['residual_norm'] <= info['nonlinear']['tolerance']
    assert len(calls) == 2


def test_invalid_trial_or_old_checkpoint_does_not_replace_accepted_state(real_case, tmp_path):
    cfg = config_for(real_case)
    model = imported_model(cfg)
    driver = BEIBStepper(model, cfg, 'cpu')
    state = driver.initialize(model.mesh.X)
    problem = BEProblem(driver, state, driver.transfer.prepare(state.x), driver.flow.advect(state.velocity))
    original = state.x.clone()
    with pytest.raises(ValueError, match='invalid real LV deformation'):
        problem.residual(-2*state.x)
    torch.testing.assert_close(state.x, original, rtol=0, atol=0)
    from afsi_torch.simulation.real_lv_mac import run as old_run
    old_run(case_config=real_case, device='cpu', output=tmp_path/'old')
    with pytest.raises(ValueError, match='old CNAB'):
        load(tmp_path/'old/checkpoint.npz')


def test_warmed_paper_benchmark_preserves_checkpoint_and_checks_tolerance(real_case, tmp_path):
    from validation.benchmark_paper_lv import benchmark
    config = config_for(real_case, 'jfnk')
    model = imported_model(config)
    driver = BEIBStepper(model, config, 'cpu')
    state = driver.initialize(model.mesh.X)
    path = tmp_path/'checkpoint.npz'
    save(path, model, state, config, dict(elapsed_seconds=0.))
    original = path.read_bytes()
    report = benchmark(path, device='cpu', warmup=1, steps=1, intervals=(1, 5))
    assert path.read_bytes() == original
    assert len(report['variants']) == 2
    for variant in report['variants']:
        assert variant['max_accepted_residual_to_tolerance'] <= 1.
        assert variant['per_step']['fluid_solves'] > 0
        assert variant['per_step']['true_residual_checks'] > 0
        assert variant['milliseconds_per_step'] > 0
    assert report['variants'][1]['end_state_max_abs_vs_first']['x_cm'] < 1e-8
