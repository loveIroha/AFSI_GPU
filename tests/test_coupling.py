"""Causal update ordering, moving supports, density scaling and solid validity."""
from dataclasses import replace
import pytest
import torch
import numpy as np
from afsi_torch import ib, solid
from afsi_torch.tetrahedron import reference_nodes
from afsi_torch.fluid import create_box, prepare_operators, ChorinSolver
from afsi_torch.coupling import ExplicitIBStepper


@pytest.fixture(params=['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA device unavailable'))])
def device(request):
    return request.param


def setup(device, dt=.002):
    X = reference_nodes(device=device)*.4+.1
    geo = solid.prepare_p2(X, torch.arange(10, device=device).reshape(1, 10))
    mesh = create_box((3,)*3, (3.,)*3, (-1.,)*3, device=device)
    fluid = ChorinSolver(prepare_operators(mesh), dt=dt, mu=.1)
    force = lambda x, t: solid.stress_force(x, geo, 3., 5.)
    validate = lambda x: solid.validate_deformation(x, geo)
    return X, geo, fluid, force, validate


def test_zero_force_bootstrap_lag_and_manual_composition(device):
    X, _, fluid, force, validate = setup(device)
    times = []
    def callback(x, t):
        times.append(t)
        return force(x, t)+x.new_tensor([.01+t, 0., 0.]).expand_as(x)
    driver = ExplicitIBStepper(fluid, callback, validate)
    state = driver.initialize(X)
    before = state.x.clone()
    assert times == [] and state.force.count_nonzero() == 0
    first = driver.step(state)
    torch.testing.assert_close(first.state.x, X, atol=0, rtol=0)
    assert times == [0.] and first.state.force_time == 0.
    old = first.state
    stencil = ib.prepare_stencil(old.x, driver.grid)
    density = ib.spread_density(old.force, stencil)
    flow = fluid.step(old.velocity, density=density, pressure_initial=old.pressure)
    expected_x = old.x+fluid.dt*ib.interpolate(flow.velocity, stencil)
    result = driver.step(old)
    torch.testing.assert_close(result.state.x, expected_x, atol=1e-14, rtol=1e-12)
    torch.testing.assert_close(result.applied_density, density)
    torch.testing.assert_close(result.state.force, force(expected_x, fluid.dt)+
        expected_x.new_tensor([.01+fluid.dt, 0., 0.]).expand_as(expected_x))
    assert times == [0., fluid.dt]
    assert result.state.time == 2*fluid.dt
    assert (result.state.x-X).abs().max() > 1e-9
    torch.testing.assert_close(state.x, before, atol=0, rtol=0)
    assert result.diagnostics['lattice_power_error'] < 1e-13
    assert abs(result.diagnostics['fe_minus_solid_power']) > 1e-10


def test_diagnostic_dual_load_uses_weak_rhs_and_preserves_power(device):
    X, _, fluid, _, validate = setup(device)
    force = lambda x, t: x.new_tensor([.01, -.02, .03]).expand_as(x)
    driver = ExplicitIBStepper(fluid, force, validate, load_path='dual')
    first = driver.step(driver.initialize(X))
    state = first.state
    stencil = ib.prepare_stencil(state.x, driver.grid)
    dual = ib.spread_load(state.force, stencil)
    reference = fluid.step(state.velocity, nodal_load=dual, pressure_initial=state.pressure)
    result = driver.step(state)
    torch.testing.assert_close(result.state.velocity, reference.velocity, atol=1e-13, rtol=1e-11)
    torch.testing.assert_close(result.state.x, state.x + fluid.dt * ib.interpolate(reference.velocity, stencil),
                               atol=1e-13, rtol=1e-11)
    assert result.diagnostics['load_path'] == 'dual'
    assert abs(result.diagnostics['fe_minus_solid_power']) < 1e-13


def test_diagnostic_scaled_stencil_runs_moving_coupling(device):
    from validation.diagnose_ib import scaled_stencil
    mesh = create_box((6,)*3, (6.,)*3, (-3.,)*3, device=device)
    fluid = ChorinSolver(prepare_operators(mesh), dt=1e-4)
    X = torch.tensor([[-.2, .1, .2], [.2, -.1, -.2]], device=device, dtype=torch.float64)
    force = lambda x, t: x.new_tensor([1., 0., 0.]).expand_as(x)
    driver = ExplicitIBStepper(fluid, force, lambda x: None,
        stencil_factory=lambda x, grid: scaled_stencil(x, grid, 2), load_path='dual')
    state = driver.initialize(X)
    state = driver.step(state).state
    result = driver.step(state)
    assert result.state.step == 2
    assert torch.linalg.vector_norm(result.state.x - X).item() > 0
    assert result.diagnostics['lattice_power_error'] < 1e-13
    assert abs(result.diagnostics['fe_minus_solid_power']) < 1e-13


def test_translation_rebuilds_support_and_uses_old_positions(device):
    X, _, fluid, _, validate = setup(device, dt=.1)
    X = X+.37
    zero = lambda x, t: torch.zeros_like(x)
    driver = ExplicitIBStepper(fluid, zero, validate)
    bc = torch.full_like(fluid.op.mesh.velocity_coordinates, .5)
    state = driver.initialize(X, velocity=bc)
    original = ib.prepare_stencil(X, driver.grid)
    for _ in range(3):
        result = driver.step(state, boundary_values=bc)
        torch.testing.assert_close(result.solid_velocity, torch.full_like(X, .5), atol=1e-11, rtol=1e-11)
        state = result.state
    torch.testing.assert_close(state.x, X+.15, atol=1e-12, rtol=1e-12)
    new = ib.prepare_stencil(state.x, driver.grid)
    assert not torch.equal(original.indices, new.indices)


def test_deformed_elastic_body_short_run(device):
    X, geo, fluid, force, validate = setup(device)
    driver = ExplicitIBStepper(fluid, force, validate)
    state = driver.initialize(X+.03*X.square())
    initial_x = state.x.clone()
    for _ in range(4):
        result = driver.step(state)
        state = result.state
        assert torch.isfinite(state.force).all()
        assert result.diagnostics['spread_force_balance_max_abs'] < 1e-12
        for info in result.diagnostics['fluid']['solves'].values():
            assert info['residual_norm'] <= info['tolerance']
    assert (state.x-initial_x).abs().max() > 1e-10
    assert solid.energy(state.x, geo, 3., 5.).isfinite()


def test_rejected_step_does_not_mutate_state(device):
    X, _, fluid, force, validate = setup(device, dt=1.)
    driver = ExplicitIBStepper(fluid, force, validate)
    u = torch.ones_like(fluid.op.mesh.velocity_coordinates)
    state = driver.initialize(X, velocity=u)
    snapshot = state.x.clone()
    with pytest.raises(ValueError, match='displacement'):
        driver.step(state, boundary_values=u)
    torch.testing.assert_close(state.x, snapshot, atol=0, rtol=0)
    with pytest.raises(ValueError, match='time/step'):
        driver.step(replace(state, time=.1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device unavailable')
def test_coupled_cpu_gpu_parity():
    states = []
    for device in ('cpu', 'cuda'):
        X, _, fluid, force, validate = setup(device)
        driver = ExplicitIBStepper(fluid, force, validate)
        state = driver.initialize(X+.03*X.square())
        for _ in range(3):
            state = driver.step(state).state
        states.append(state)
    for name in ('x', 'velocity', 'pressure', 'force'):
        torch.testing.assert_close(getattr(states[0], name), getattr(states[1], name).cpu(), atol=2e-9, rtol=1e-7)


def test_generated_lv_coupled_and_output(device, tmp_path):
    pytest.importorskip('gmsh')
    meshio = pytest.importorskip('meshio')
    from examples.coupled_lv import run
    r = run(device=device, steps=4, mesh_size=1.4, output=str(tmp_path) if device == 'cpu' else None,
            output_every=2)
    assert r['coupled_time_stepping'] and not r['physiological_cycle']
    assert r['history'][-1]['minimum_detF'] > .99
    assert r['history'][-1]['max_total_displacement_cm'] > 1e-10
    assert r['history'][-1]['cavity_volume_ml'] > 0
    assert r['history'][0]['applied_force_norm_dyn'] == 0
    if device == 'cpu':
        import xml.etree.ElementTree as ET
        files = ET.parse(tmp_path/'solid.pvd').findall('.//DataSet')
        assert len(files) == 3
        assert float(files[-1].attrib['timestep']) == 4e-4
        volume = meshio.read(tmp_path/'solid_000004.vtu')
        fluid = meshio.read(tmp_path/'fluid_000004.vtu')
        with np.load(tmp_path/'final_state.npz', allow_pickle=False) as state:
            np.testing.assert_allclose(volume.points, state['x'])
            np.testing.assert_allclose(volume.point_data['next_force_dyn'], state['force'])
            np.testing.assert_allclose(fluid.point_data['velocity_cm_per_s'], state['velocity'])
            # Q1 pressure interpolated onto Q2 nodes must reproduce every corner.
            np.testing.assert_allclose(fluid.point_data['pressure_dyn_per_cm2'].reshape(13, 13, 13)[::2, ::2, ::2].reshape(-1),
                                       state['pressure'], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('kwargs', [dict(pressure_mmhg=-1), dict(tension=-1), dict(ramp_time=0)])
def test_invalid_loads(kwargs):
    from afsi_torch.lv_model import RampLoads
    with pytest.raises(ValueError):
        RampLoads(**kwargs)
