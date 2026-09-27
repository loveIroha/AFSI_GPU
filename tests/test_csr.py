"""CSR equivalence, blocked PCG stopping and diagnostic-free stepping."""
import pytest
import torch
from afsi_torch.fluid import (create_box, prepare_operators, CSRFluidOperators,
                              ChorinSolver, SolverOptions, pcg)


@pytest.fixture(params=['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA device unavailable'))])
def device(request):
    return request.param


def test_csr_all_fixed_actions_and_combined_matrix(device):
    mesh = create_box((2, 3, 2), (1.3, 2.1, .7), (-.2, .1, -.5), device=device)
    op = prepare_operators(mesh)
    csr = CSRFluidOperators(op)
    u, p = torch.sin(2.3*mesh.velocity_coordinates), torch.cos(mesh.pressure_coordinates).sum(-1)
    for name, value in [('velocity_mass', u), ('velocity_stiffness', u),
                        ('pressure_mass', p), ('pressure_stiffness', p),
                        ('gradient', p), ('divergence', u), ('density_load', u)]:
        torch.testing.assert_close(getattr(csr, name)(value), getattr(op, name)(value), atol=3e-14, rtol=2e-12)
    A = csr.tentative_matrix(230., .7)
    torch.testing.assert_close(torch.sparse.mm(A, u), 230*op.velocity_mass(u)+.7*op.velocity_stiffness(u),
                               atol=1e-12, rtol=2e-12)
    assert all(m.layout == torch.sparse_csr and m.device == u.device for m in csr.matrices.values())


def test_csr_chorin_three_steps_and_nonzero_boundary(device):
    mesh = create_box((3, 2, 2), (1.2, 1.1, .9), device=device)
    op = prepare_operators(mesh)
    reference = ChorinSolver(op, dt=.002, mu=.2)
    optimized = ChorinSolver(CSRFluidOperators(op), dt=.002, mu=.2,
        options=SolverOptions(check_every=8))
    X = mesh.velocity_coordinates
    boundary = torch.zeros_like(X)
    boundary[:, 0] = .1*X[:, 1]  # compatible shear, including a nonzero lift
    states = [(boundary.clone(), None), (boundary.clone(), None)]
    for _ in range(3):
        outputs = []
        for i, solver in enumerate((reference, optimized)):
            u, p = states[i]
            result = solver.step(u, density=torch.sin(X), boundary_values=boundary,
                                 pressure_initial=p, diagnostics=i == 0)
            states[i] = result.velocity, result.pressure
            outputs.append(result)
            assert all(s['residual_norm'] <= s['tolerance'] for s in result.diagnostics['solves'].values())
        for name in ('velocity', 'tentative_velocity', 'pressure'):
            torch.testing.assert_close(getattr(outputs[0], name), getattr(outputs[1], name), atol=2e-9, rtol=2e-8)
        assert set(outputs[1].diagnostics) == {'solves', 'net_flux'}


def test_blocked_pcg_exact_convergence_and_failures(device):
    b = torch.arange(1., 8., dtype=torch.float64, device=device)
    options = SolverOptions(check_every=8, recompute_every=200)
    # Identity converges at iteration 1; remaining device iterations must be harmless.
    x, info = pcg(lambda x: x, b, torch.ones_like(b), options=options)
    torch.testing.assert_close(x, b, atol=0, rtol=0)
    assert info.iterations == 8 and info.residual_norm == 0
    _, warm = pcg(lambda x: x, b, torch.ones_like(b), initial=b, options=options)
    assert warm.iterations == 0
    with pytest.raises(RuntimeError, match='positive definite'):
        pcg(lambda x: -x, b, torch.ones_like(b), options=options)
    with pytest.raises(RuntimeError, match='did not converge'):
        pcg(lambda x: b*x, b, torch.ones_like(b),
            options=SolverOptions(check_every=8, max_iterations=2))
    # A check interval larger than the recomputation interval still tests true residuals.
    x, info = pcg(lambda x: b*x, b, torch.ones_like(b),
                  options=SolverOptions(check_every=8, recompute_every=3))
    torch.testing.assert_close(x, torch.ones_like(b), atol=1e-9, rtol=1e-9)
    assert torch.linalg.vector_norm(b-b*x).item() <= info.tolerance


def test_sampled_cycle_failure_records_latest_state(tmp_path, monkeypatch):
    pytest.importorskip('gmsh')
    import json
    from examples.lv_cycle import run
    from afsi_torch.coupling import ExplicitIBStepper
    original = ExplicitIBStepper.step
    def fail(self, state, **kwargs):
        if state.step == 3:
            raise RuntimeError('sampling failure check')
        result = original(self, state, **kwargs)
        assert 'fe_power_erg_per_s' not in result.diagnostics
        assert 'kinetic_energy' not in result.diagnostics['fluid']
        return result
    monkeypatch.setattr(ExplicitIBStepper, 'step', fail)
    with pytest.raises(RuntimeError, match='sampling failure'):
        run(device='cpu', output=tmp_path, mesh_size=2.5, fluid_cells=3, end_time=.0003,
            write_vtk=False, history_every=20, log_every=100, checkpoint_every=200)
    report = json.loads((tmp_path/'report.json').read_text())
    assert report['last']['step'] == report['accepted_steps'] == 3
    assert report['last']['force_norm_dyn'] > 0
    assert report['settings']['backend'] == 'csr'
    assert report['settings']['solver']['check_every'] == 8


def test_nonlinear_ib_trajectory_without_diagnostics(device):
    from afsi_torch import solid
    from afsi_torch.tetrahedron import reference_nodes
    from afsi_torch.coupling import ExplicitIBStepper
    X = reference_nodes(device=device)*.4+.1
    geometry = solid.prepare_p2(X, torch.arange(10, device=device).reshape(1, 10))
    mesh = create_box((3,)*3, (3.,)*3, (-1.,)*3, device=device)
    op = prepare_operators(mesh)
    force = lambda x, t: solid.stress_force(x, geometry, 3., 5.)
    validate = lambda x: solid.validate_deformation(x, geometry)
    states = []
    for i, operators in enumerate((op, CSRFluidOperators(op))):
        solver = ChorinSolver(operators, dt=.002, mu=.1,
            options=SolverOptions(check_every=1 if i == 0 else 8))
        driver = ExplicitIBStepper(solver, force, validate)
        state = driver.initialize(X+.03*X.square())
        for _ in range(5):
            state = driver.step(state, diagnostics=i == 0).state
        states.append(state)
    for name in ('x', 'velocity', 'pressure', 'force'):
        torch.testing.assert_close(getattr(states[0], name), getattr(states[1], name), atol=2e-10, rtol=2e-8)

