"""Same BE equations with cheaper checks, owned scratch and solver comparisons."""
from dataclasses import replace
from math import ceil, sqrt
import pytest
import torch
from test_real_lv import real_case, DEVICES
from test_paper_lv import config_for
from test_adaptive_p1_transfer import transfer
from afsi_torch.paper_lv import imported_model
from afsi_torch.mac.paper_coupling import BEIBStepper, BEProblem
from afsi_torch.mac.backward_euler import BackwardEulerFlow, BEFlowOptions
from afsi_torch.mac.grid import MACGrid
from afsi_torch.mac.adaptive_transfer import AdaptiveP1Transfer, InteractionQuadratureOptions
from afsi_torch.mac.compact_transfer import CompactFETransfer
from afsi_torch.p1 import prepare_p1
from afsi_torch.paper_lv_checkpoint import save


def test_vertex_support_compiler_reuses_graph_across_sizes_instances_and_subclasses(monkeypatch):
    # Exercise Dynamo on CPU too: more than eight static sizes/subclass guards
    # exhausted the former bound-method cache in the CUDA suite.
    import afsi_torch.mac.compact_transfer as module
    compiled_graphs=[]
    def backend(graph,inputs):
        compiled_graphs.append(graph)
        return graph.forward
    compiled=torch.compile(module._vertex_support_flags,backend=backend,fullgraph=True,dynamic=True)
    monkeypatch.setattr(module,'_vertex_support_tensor_kernel',lambda device_type:compiled)
    X=torch.tensor([[3.,3.,3.],[3.4,3.,3.],[3.,3.4,3.],[3.,3.,3.4]],dtype=torch.float64)
    geometry=prepare_p1(X,torch.arange(4).reshape(1,4),degree=2)
    for i in range(12):
        grid=MACGrid((16+i,)*3,(16.+i,)*3,origin=(float(i),)*3)
        t=(AdaptiveP1Transfer(grid,geometry,fused=bool(i%3)) if i%2
           else CompactFETransfer(grid,geometry))
        points=(t.origin+3*t.spacing).repeat(i+2,1)
        assert t._vertex_support_kernel(points)
        points[0,0]=t.origin[0]+2*t.spacing[0]
        assert not t._vertex_support_kernel(points)
        points[0,0]=float('nan')
        assert not t._vertex_support_kernel(points)
    assert len(compiled_graphs)==1


def test_adaptive_order_compiler_reuses_device_scalar_parameters_across_grids():
    from afsi_torch.mac.adaptive_transfer import _orders
    graphs=[]
    def backend(graph,inputs):
        graphs.append(graph)
        # Device coefficients must stay tensor operations, without scalar reads.
        assert not any(node.op=='call_method' and node.target=='item' for node in graph.graph.nodes)
        return graph.forward
    compiled=torch.compile(_orders,backend=backend,fullgraph=True,dynamic=True)
    X=torch.tensor([[3.,3.,3.],[3.4,3.,3.],[3.,3.4,3.],[3.,3.,3.4]],dtype=torch.float64)
    for i in range(12):
        count=1+i%2
        x=torch.cat([X+j for j in range(count)])
        cells=torch.arange(4*count).reshape(count,4)
        geometry=prepare_p1(x,cells,degree=2)
        dx=.4+.031*i
        density=2.+.13*i
        grid=MACGrid((16,)*3,(16*dx,)*3)
        t=AdaptiveP1Transfer(grid,geometry,
            quadrature_options=InteractionQuadratureOptions(mode='adaptive',point_density=density),fused=False)
        t._order_kernel=compiled
        assert t._dx.device==x.device and t._density.device==x.device
        expected=max(2,ceil(density*sqrt(2)*.4/dx))
        assert t._rule(x).orders.tolist()==[expected]*count
    # Size one is specialized by Dynamo; larger cell groups share one graph.
    assert len(graphs)<=2


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float32,torch.float64])
def test_adaptive_order_single_cell_coefficients_are_on_tensor_device(device,dtype):
    X=torch.tensor([[3.,3.,3.],[3.4,3.,3.],[3.,3.4,3.],[3.,3.,3.4]],device=device,dtype=dtype)
    geometry=prepare_p1(X,torch.arange(4,device=device).reshape(1,4),degree=2)
    t=AdaptiveP1Transfer(MACGrid((16,)*3,(16.,)*3),geometry)
    original=t._order_kernel
    calls=[]
    def checked(x,cells,edges,dx,density):
        for value in (dx,density):
            assert isinstance(value,torch.Tensor)
            assert value.device==x.device and value.dtype==x.dtype and value.ndim==0
        calls.append(None)
        return original(x,cells,edges,dx,density)
    t._order_kernel=checked
    assert t._rule(X).orders.tolist()==[2]
    t.quadrature_options=replace(t.quadrature_options,point_density=6.)
    assert t._rule(X).orders.tolist()==[4]
    assert len(calls)==2


@pytest.mark.parametrize('device', DEVICES)
def test_interior_support_shortcut_does_not_materialize_points(device):
    X, t = transfer(device)
    calls = []
    points = t._points
    def counted(x, rule):
        calls.append(None)
        return points(x, rule)
    t._points = counted
    t.check_configuration_support(X)
    assert not calls
    t.check_support(t.interaction_points(X))
    assert len(calls) == 1
    X, limited = transfer(device, max_points=20)
    with pytest.raises(ValueError, match='no density clipping'):
        limited.check_configuration_support(X)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('adaptive', [False, True])
def test_support_hull_rejection_falls_back_to_original_point_acceptance(device, adaptive):
    X = torch.tensor([[1.95, 3., 3.], [2.35, 3., 3.],
        [2.35, 3.4, 3.], [2.35, 3., 3.4]], device=device, dtype=torch.float64)
    geometry = prepare_p1(X, torch.arange(4, device=device).reshape(1, 4), degree=2)
    grid = MACGrid((16,)*3, (16.,)*3)
    t = (AdaptiveP1Transfer(grid, geometry, quadrature_options=InteractionQuadratureOptions(mode='adaptive'))
         if adaptive else CompactFETransfer(grid, geometry))
    assert not t._vertex_support_kernel(X)
    # A vertex is outside, but all points of this rule have complete support.
    t.check_support(t.interaction_points(X))
    t.check_configuration_support(X)
    bad = X-X.new_tensor([1., 0., 0.])
    with pytest.raises(ValueError, match='support reaches a wall'):
        t.check_configuration_support(bad)
    with pytest.raises(ValueError, match='support reaches a wall'):
        t.check_support(t.interaction_points(bad))


@pytest.mark.parametrize('device', DEVICES)
def test_support_backends_preserve_accepted_be_step(real_case, device):
    config = config_for(real_case, 'jfnk')
    model = imported_model(config, device)
    a = BEIBStepper(model, replace(config, support_backend='points'), device)
    b = BEIBStepper(model, replace(config, support_backend='vertices'), device)
    x, _ = a.step(a.initialize(model.mesh.X))
    y, info = b.step(b.initialize(model.mesh.X))
    torch.testing.assert_close(y.x, x.x, rtol=2e-10, atol=2e-11)
    for u, v in zip(y.velocity, x.velocity):
        torch.testing.assert_close(u, v, rtol=2e-8, atol=2e-10)
    assert info['nonlinear']['residual_norm'] <= info['nonlinear']['tolerance']


def test_trial_cache_reuses_validation_without_equality_sync_and_tracks_mutations(real_case, monkeypatch):
    cfg = config_for(real_case)
    model = imported_model(cfg)
    driver = BEIBStepper(model, cfg, 'cpu')
    state = driver.initialize(model.mesh.X)
    problem = BEProblem(driver, state, driver.transfer.prepare(state.x), driver.flow.advect(state.velocity))
    y = torch.zeros_like(state.x)
    original_validate = driver.solid.validate
    validated = []
    def validate(x):
        validated.append(x)
        return original_validate(x)
    monkeypatch.setattr(driver.solid, 'validate', validate)
    equality = torch.equal
    def forbidden_equal(*args):
        raise AssertionError('a residual evaluation must not synchronize for tensor equality')
    monkeypatch.setattr(torch, 'equal', forbidden_equal)
    problem.validate(y)
    first = problem.residual(y)
    assert len(validated) == 1
    retained = problem.configuration(y)
    count = driver.flow.calls
    torch.testing.assert_close(problem.residual(y), first)
    assert driver.flow.calls == count
    y.add_(1e-7)
    problem.residual(y)
    assert driver.flow.calls > count
    assert problem.configuration(y) is not retained
    # Mutation of a returned configuration invalidates the geometry cache.
    configuration = problem.configuration(y)
    configuration.add_(1.)
    regenerated = problem.configuration(y)
    torch.testing.assert_close(regenerated, state.x+y, rtol=0, atol=0)
    monkeypatch.setattr(torch, 'equal', equality)
    problem.residual(y)
    count = driver.flow.calls
    result, accepted_x = problem.accepted(y.clone())
    assert driver.flow.calls == count
    torch.testing.assert_close(result, problem.last_residual)
    torch.testing.assert_close(accepted_x, state.x+y, rtol=0, atol=0)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('backend', ['workspace', 'graph'])
@pytest.mark.parametrize('check_every', [1, 3, 4])
def test_be_workspace_matches_reference_and_owns_public_results(device, backend, check_every):
    grid = MACGrid((8,)*3, (2., 3., 4.))
    options = BEFlowOptions(convection=False, check_every=check_every)
    execution = 'fused' if device == 'cuda' else 'torch'
    reference = BackwardEulerFlow(grid, dt=.01, device=device, options=options, backend=execution)
    flow = BackwardEulerFlow(grid, dt=.01, device=device,
        options=replace(options, helmholtz_backend=backend), backend=execution)
    rhs = tuple(torch.randn_like(u) for u in grid.zeros(device=device))
    saved = tuple(u.clone() for u in rhs)
    actual, p, info = flow.solve_rhs(rhs)
    expected, q, baseline = reference.solve_rhs(rhs)
    for u, v in zip(actual, expected):
        torch.testing.assert_close(u, v, rtol=3e-9, atol=3e-10)
    torch.testing.assert_close(p, q, rtol=3e-9, atol=3e-9)
    assert info['helmholtz_sweeps'] == baseline['helmholtz_sweeps']
    assert all(r <= t for r, t in zip(info['helmholtz_residuals'], info['helmholtz_tolerances']))
    retained = tuple(u.clone() for u in (*actual, p))
    zero, _, _ = flow.solve_rhs(tuple(torch.zeros_like(v) for v in rhs))
    for u in zero:
        assert not u.any()
    for a, b in zip((*actual, p), retained):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(rhs, saved):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    if backend == 'graph' and device == 'cuda':
        assert flow.workspace.graph is not None


def test_solver_comparison_and_separate_profile_preserve_input_checkpoint(real_case, tmp_path):
    from validation.benchmark_paper_lv import benchmark
    cfg = config_for(real_case, 'jfnk')
    model = imported_model(cfg)
    driver = BEIBStepper(model, cfg, 'cpu')
    path = tmp_path/'checkpoint.npz'
    save(path, model, driver.initialize(model.mesh.X), cfg, dict(elapsed_seconds=0.))
    original = path.read_bytes()
    report = benchmark(path, device='cpu', warmup=1, steps=1, intervals=(5,),
        solvers=('jfnk', 'anderson-newton'), helmholtz_backends=('workspace',),
        profile=True, profile_steps=1)
    assert path.read_bytes() == original
    assert len(report['variants']) == 2
    for variant in report['variants']:
        assert variant['max_accepted_residual_to_tolerance'] <= 1.
        assert variant['profile']['phases']['fluid_total']['calls'] > 0
        assert variant['profile']['phases']['mass_solves']['calls'] == 2*variant['profile']['phases']['fluid_total']['calls']
    assert report['variants'][1]['end_state_max_abs_vs_first']['x_cm'] < 1e-8


def test_benchmark_keeps_successful_variant_when_later_configuration_fails(real_case, tmp_path):
    from validation.benchmark_paper_lv import benchmark
    cfg = config_for(real_case, 'jfnk')  # Fixed quadrature cannot use vector shared kernels.
    model = imported_model(cfg)
    driver = BEIBStepper(model, cfg, 'cpu')
    path = tmp_path/'checkpoint.npz'
    save(path, model, driver.initialize(model.mesh.X), cfg, dict(elapsed_seconds=0.))
    report = benchmark(path, device='cpu', warmup=0, steps=1, intervals=(5,),
        shared_executions=('reference', 'vector'))
    assert [v['status'] for v in report['variants']] == ['completed', 'failed']
    assert report['variants'][1]['failure_type'] == 'ValueError'
    assert 'measured_speedup_first_over_second' not in report
