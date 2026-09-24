"""Boundary lifting, convergence failures, gauges and full Chorin equations."""
import pytest
import torch
from afsi_torch.fluid import create_box, prepare_operators, ChorinSolver, SolverOptions, pcg
from afsi_torch.fluid.solvers import operator_diagonals


@pytest.fixture(params=['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA device unavailable'))])
def device(request):
    return request.param


def setup(device, n=2, **kwargs):
    mesh = create_box((n,)*3, (1.4, 1.1, .9), (-.3, .2, -.5), device=device)
    op = prepare_operators(mesh)
    return mesh, op, ChorinSolver(op, dt=.025, mu=.15, **kwargs)


def test_pcg_nonzero_component_boundary_and_true_residual(device):
    t = torch.arange(17, device=device, dtype=torch.float64)
    A = torch.diag(2+t/17)+.02*torch.cos(t[:, None]-t[None, :])
    b = torch.sin(t[:, None]+t.new_tensor([.2, .7, .9]))
    fixed = torch.zeros_like(b, dtype=torch.bool)
    fixed[0, :] = True
    fixed[-1, 1] = True
    values = torch.cos(b)
    result, info = pcg(lambda x: A@x, b, A.diag()[:, None].expand_as(b), fixed=fixed, values=values,
                       options=SolverOptions(rtol=1e-12, atol=1e-14, recompute_every=3))
    expected = torch.where(fixed, values, 0)
    for j in range(3):
        free = ~fixed[:, j]
        expected[free, j] = torch.linalg.solve(A[free][:, free], (b-A@expected)[free, j])
    torch.testing.assert_close(result, expected, atol=3e-12, rtol=2e-12)
    actual = torch.linalg.vector_norm((b-A@result).masked_fill(fixed, 0)).item()
    assert actual == pytest.approx(info.residual_norm, abs=1e-15)
    assert actual <= info.tolerance
    warm, warm_info = pcg(lambda x: A@x, b, A.diag()[:, None].expand_as(b), fixed=fixed,
                          values=values, initial=result)
    assert warm_info.iterations == 0
    torch.testing.assert_close(warm, result)


def test_fe_solve_manufactured_and_diagonals(device):
    mesh, op, _ = setup(device)
    X = mesh.velocity_coordinates
    exact = torch.sin(2*X)+X
    action = lambda u: 3*op.velocity_mass(u)+.2*op.velocity_stiffness(u)
    diagonals = operator_diagonals(op)
    diagonal = 3*diagonals['velocity_mass']+.2*diagonals['velocity_stiffness']
    result, info = pcg(action, action(exact), diagonal,
        fixed=mesh.velocity_boundary[:, None].expand_as(X), values=exact)
    torch.testing.assert_close(result, exact, atol=1e-9, rtol=1e-9)
    assert info.residual_norm <= info.tolerance
    # Selected assembled diagonal entries checked by independent unit actions.
    for i in (0, 31, 62):
        e = torch.zeros_like(X)
        e[i, 1] = 1
        torch.testing.assert_close(action(e)[i, 1], diagonal[i, 1], atol=1e-14, rtol=1e-13)


def test_rest_and_steady_shear_nonzero_boundary(device):
    mesh, op, solver = setup(device)
    X = mesh.velocity_coordinates
    rest = solver.step(torch.zeros_like(X))
    assert all(info['iterations'] == 0 for info in rest.diagnostics['solves'].values())
    shear = torch.zeros_like(X)
    shear[:, 0] = .2*X[:, 1]
    result = solver.step(shear, boundary_values=shear)
    torch.testing.assert_close(result.velocity, shear, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(result.pressure, torch.zeros_like(result.pressure), atol=1e-9, rtol=0)
    # Exact unsteady shear: u=(a(t)y,0,0), a'=constant, f=rho*a'*y.
    # Checks the new time level's nonzero boundary lift without nonlinear error.
    u = shear
    forcing = torch.zeros_like(X)
    forcing[:, 0] = solver.rho*.3*X[:, 1]
    for step in range(1, 4):
        target = torch.zeros_like(X)
        target[:, 0] = (.2+.3*step*solver.dt)*X[:, 1]
        u = solver.step(u, density=forcing, boundary_values=target).velocity
        torch.testing.assert_close(u, target, atol=2e-10, rtol=1e-9)


def test_chorin_equations_gauge_invariance_and_load_paths(device):
    mesh, op, solver = setup(device)
    X = mesh.velocity_coordinates
    s = (X-X.new_tensor(mesh.origin))/X.new_tensor(mesh.lengths)
    f = (s*(1-s)).prod(-1)[:, None]*X.new_tensor([20., -3., 5.])
    zero = torch.zeros_like(X)
    result = solver.step(zero, density=f)
    for info in result.diagnostics['solves'].values():
        assert info['residual_norm'] <= info['tolerance']
    free = ~mesh.velocity_boundary
    torch.testing.assert_close(solver.tentative_action(result.tentative_velocity)[free],
                               op.density_load(f)[free], atol=2e-12, rtol=1e-9)
    torch.testing.assert_close(op.pressure_stiffness(result.pressure),
        -solver.rho/solver.dt*op.divergence(result.tentative_velocity), atol=2e-11, rtol=1e-8)
    torch.testing.assert_close(op.velocity_mass(result.velocity)[free],
        (op.velocity_mass(result.tentative_velocity)-solver.dt/solver.rho*op.gradient(result.pressure))[free],
        atol=2e-12, rtol=1e-8)
    assert result.diagnostics['corrected_divergence_dual_norm'] < result.diagnostics['tentative_divergence_dual_norm']
    assert result.pressure[0] == 0
    direct = solver.step(zero, nodal_load=op.density_load(f))
    torch.testing.assert_close(result.velocity, direct.velocity, atol=1e-12, rtol=1e-10)
    shifted = ChorinSolver(op, dt=solver.dt, mu=solver.mu, pressure_dof=len(result.pressure)-1, pressure_value=2.)
    other = shifted.step(zero, density=f)
    torch.testing.assert_close(result.velocity, other.velocity, atol=2e-10, rtol=1e-7)
    offset = other.pressure-result.pressure
    torch.testing.assert_close(offset, offset[-1].expand_as(offset), atol=3e-9, rtol=1e-9)


def test_short_flow_and_pressure_iteration_refinement(device):
    for n in (2, 4):
        mesh, op, solver = setup(device, n=n)
        X = mesh.velocity_coordinates
        s = (X-X.new_tensor(mesh.origin))/X.new_tensor(mesh.lengths)
        f = (4*s*(1-s)).prod(-1)[:, None]*X.new_tensor([1., .3, -.2])
        u, p = torch.zeros_like(X), None
        for _ in range(3):
            result = solver.step(u, density=f, pressure_initial=p)
            u, p = result.velocity, result.pressure
            assert torch.isfinite(u).all() and torch.isfinite(p).all()
            assert result.diagnostics['solves']['pressure']['iterations'] < 150
        assert result.diagnostics['kinetic_energy'] > 0


def test_incompatible_boundary_flux_and_ambiguous_force_rejected(device):
    mesh, _, solver = setup(device)
    X = mesh.velocity_coordinates
    with pytest.raises(ValueError, match='compatibility'):
        solver.step(torch.zeros_like(X), boundary_values=X)
    with pytest.raises(ValueError, match='never both'):
        solver.step(torch.zeros_like(X), density=X, nodal_load=X)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device unavailable')
def test_full_step_cpu_cuda():
    results = []
    for device in ('cpu', 'cuda'):
        mesh, _, solver = setup(device)
        X = mesh.velocity_coordinates
        results.append(solver.step(torch.zeros_like(X), density=torch.sin(X)))
    for name in ('velocity', 'pressure', 'tentative_velocity'):
        torch.testing.assert_close(getattr(results[0], name), getattr(results[1], name).cpu(), atol=2e-9, rtol=1e-8)


def test_solver_failure_is_not_reported_as_convergence():
    b = torch.ones(3, dtype=torch.float64)
    A = torch.diag(b.new_tensor([1., 2., 4.]))
    with pytest.raises(RuntimeError, match='did not converge'):
        pcg(lambda x: A@x, b, b, options=SolverOptions(max_iterations=1))
    with pytest.raises(RuntimeError, match='positive definite'):
        pcg(lambda x: -x, b, b)
    with pytest.raises(ValueError, match='finite'):
        pcg(lambda x: x, b*float('nan'), b)
    x, info = pcg(lambda x: x, b, b, fixed=torch.ones_like(b, dtype=torch.bool), values=b*2)
    torch.testing.assert_close(x, b*2)
    assert info.iterations == 0


@pytest.mark.parametrize('kwargs', [dict(dt=0), dict(dt=-1), dict(dt=.1, rho=0),
    dict(dt=.1, mu=-1), dict(dt=.1, pressure_dof=-1)])
def test_invalid_physical_parameters(kwargs):
    with pytest.raises(ValueError):
        ChorinSolver(prepare_operators(create_box()), **kwargs)
