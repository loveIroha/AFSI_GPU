"""User UFL H-O derivatives, P1 weak form, basal constraints and real-LV demo."""
from dataclasses import asdict, replace
from math import factorial
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from afsi_torch.holzapfel_ogden import HOParameters, RealLVLoads, ho_energy, ho_pk1, active_energy
from afsi_torch.real_lv import RealLVConfig, imported_model
from afsi_torch import p1, boundary as bd
from afsi_torch.config import TimeConfig, FluidConfig, OutputConfig, LVExecutionConfig, load_config, save_config
from afsi_torch.mac.execution import build_driver
from afsi_torch.real_lv_checkpoint import save_real_lv, load_real_lv

DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))]


def ufl_energy(F, f, s, p):
    # Independent transcription: C, det(C), matrix invariants, no cofactor.
    C = F.T @ F
    I3 = torch.linalg.det(C)
    I1 = torch.trace(C)*I3.pow(-1/3)
    ef = torch.maximum(f @ C @ f, F.new_tensor(1.))-1
    es = torch.maximum(s @ C @ s, F.new_tensor(1.))-1
    I8 = f @ C @ s
    return (p.a/(2*p.b)*torch.exp(p.b*(I1-3))
            + p.a_f/(2*p.b_f)*(torch.exp(p.b_f*ef**2)-1)
            + p.a_s/(2*p.b_s)*(torch.exp(p.b_s*es**2)-1)
            + p.a_fs/(2*p.b_fs)*(torch.exp(p.b_fs*I8**2)-1)
            + p.kappa/4*torch.log(I3)**2)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('stretch', [.9, 1., 1.08])
def test_pk1_matches_supplied_ufl_and_fixed_active_potential(device, stretch):
    F = torch.tensor([[stretch, .035, .01], [.01, 1.02, .02], [0., .01, .99]],
                     dtype=torch.float64, device=device, requires_grad=True)
    f = F.new_tensor([1., .001, 0.]); s = F.new_tensor([.002, 1., 0.])
    p = HOParameters()
    T = 83000.
    expected = torch.autograd.grad(ufl_energy(F, f, s, p), F, retain_graph=True)[0]
    Ff = F @ f
    expected = expected+T*(1+4.9*(torch.linalg.vector_norm(Ff)-1))*torch.outer(Ff, f)
    actual = ho_pk1(F, f, s, p, T)
    torch.testing.assert_close(actual, expected, rtol=3e-12, atol=1e-8)
    energy = ho_energy(F, f, s, p)+active_energy(F, f, T, p)
    torch.testing.assert_close(actual, torch.autograd.grad(energy, F)[0], rtol=3e-12, atol=1e-8)


def test_ho_objectivity_and_compression_cutoff():
    p = HOParameters(kappa=17.)
    f, s = torch.eye(3, dtype=torch.float64)[:2]
    F = torch.diag(f.new_tensor([.8, .8, .8]))
    # Isochoric matrix term and anisotropic compressive terms give no stress.
    expected = 2*p.kappa*torch.log(torch.linalg.det(F))*torch.linalg.inv(F).T
    torch.testing.assert_close(ho_pk1(F, f, s, p), expected, atol=1e-10, rtol=1e-11)
    F = f.new_tensor([[1.02, .03, 0.], [.01, .99, .02], [0., 0., 1.01]])
    Q = f.new_tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    torch.testing.assert_close(ho_energy(Q @ F, f, s, p), ho_energy(F, f, s, p))
    torch.testing.assert_close(ho_pk1(Q @ F, f, s, p, 12.), Q @ ho_pk1(F, f, s, p, 12.), rtol=1e-11, atol=1e-9)


def test_three_cycle_waveform_matches_cpp_branches():
    loads = RealLVLoads()
    for cycle in range(3):
        for phase in (0., .1, .2, .499, .5, .6, .65, .7, .7999):
            p, t = loads.at(cycle*.8+phase)
            if phase < .2:
                expected_p, expected_t = 1.067*phase/.2, 0.
            elif phase < .5:
                expected_p, expected_t = 1.067, 0.
            else:
                d = phase-.5 if phase < .65 else .8-phase
                expected_p = 1.067+13.46*(1-np.exp(-d*d/.004))
                expected_t = 84.26*(1-np.exp(-d*d/.005))
            assert p == pytest.approx(expected_p*10000., rel=1e-11, abs=1e-8)
            assert t == pytest.approx(expected_t*10000., rel=1e-11, abs=1e-8)
    assert loads.at(2.4) == (0., 0.)
    assert loads.at(.8-1e-6)[0] > 10669.
    assert loads.at(.8)[0] == 0.  # Preserve the supplied pressure reset.


@pytest.mark.parametrize('degree,count', [(2, 4), (5, 15)])
def test_positive_simplex_rules_integrate_claimed_degree(degree, count):
    q, w = p1.tetra_rule(degree, torch.zeros((), dtype=torch.float64))
    assert len(q) == count and (w > 0).all()
    for a in range(degree+1):
        for b in range(degree+1-a):
            for c in range(degree+1-a-b):
                actual = (w*q[:, 0]**a*q[:, 1]**b*q[:, 2]**c).sum().item()
                exact = factorial(a)*factorial(b)*factorial(c)/factorial(a+b+c+3)
                assert actual == pytest.approx(exact, rel=3e-14, abs=1e-16)


@pytest.fixture
def real_case(tmp_path):
    pytest.importorskip('gmsh')
    from afsi_torch.geometry.ellipsoid import LVConfig, generate_lv
    from test_mesh_io import xdmf, collection
    generated = generate_lv(LVConfig(inner_axes=(.5, .5, .8), outer_axes=(.7, .7, 1.1),
                            base_height=.2, mesh_size=.45, center=(7.5, 7.5, 7.5)))
    X = generated.X[:generated.vertex_count].numpy()
    cells = generated.cells[:, :4].numpy()
    xdmf(tmp_path, coords=X, cells=cells, hdf=True).rename(tmp_path/'mesh_scale.xdmf')
    owners = {}
    facets = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
    for ci, row in enumerate(cells):
        for local, face in enumerate(facets):
            owners[tuple(sorted(row[face]))] = (ci, local)
    records = []
    for face, tag in zip(generated.faces[:, :3].tolist(), generated.facet_tags.tolist()):
        ci, local = owners[tuple(sorted(face))]
        records.append((ci, local, {1: 2, 2: 1, 3: 3}[tag]))
    collection(tmp_path/'boundaries.xml', 2, records, True)
    for i in range(3):
        collection(tmp_path/f'fibers_{i}.xml', 3, [(ci, 0, 1. if i == 0 else 0.) for ci in range(len(cells))])
        collection(tmp_path/f'sheets_{i}.xml', 3, [(ci, 0, 1. if i == 1 else 0.) for ci in range(len(cells))])
    return RealLVConfig(source_dir=str(tmp_path), time=TimeConfig(1e-4, 2e-4),
                        fluid=FluidConfig((8,)*3, (15.,)*3), execution=LVExecutionConfig(warm_start=True),
                        output=OutputConfig(1, 2, 1, True))


@pytest.mark.parametrize('device', DEVICES)
def test_p1_force_radial_base_and_follower_pressure(real_case, device):
    model = imported_model(real_case, device)
    X = model.mesh.X
    assert model.mesh.cells.shape[1] == 4
    center = X.new_tensor([7.5, 7.5, 7.5])
    A = X.new_tensor([[1.01, .001, 0.], [.002, .995, 0.], [0., 0., 1.005]])
    x = ((X-center) @ A.T+center).requires_grad_()
    F = model.element_gradient(x)
    energy = ((ho_energy(F, model.mesh.fiber, model.mesh.sheet, model.parameters)
               + active_energy(F, model.mesh.fiber, 123., model.parameters))*model.volumes).sum()+model.basal_energy(x)
    force = model.force_from_geometry(x, F, bd.area_vectors(x, model.endo), X.new_tensor([0., 123.]))
    torch.testing.assert_close(force, -torch.autograd.grad(energy, x)[0], rtol=2e-10, atol=1e-7)
    radial = X-center; radial[:, 2] = 0.
    assert model.basal_constraint(X+.01*radial).abs().max() < 2e-15
    z_motion = X+X.new_tensor([0., 0., .01])
    assert (model.basal_constraint(z_motion)[..., 2]+.01).abs().max() < 3e-15
    # Independent affine cofactor-normal reference pressure integration.
    x = (X-center) @ A.T+center
    cof = torch.linalg.det(A)*torch.linalg.inv(A).T
    expected = -321.*(model.endo.reference_area_vectors @ cof.T).sum(0)/2
    pressure = bd.pressure_force(x, model.endo, 321.)
    torch.testing.assert_close(pressure.sum(0), expected, rtol=1e-11, atol=1e-10)
    assert model.diagnostics(X)['cavity_volume_ml'] > 0
    torch.testing.assert_close(p1.cavity_volume(X+X.new_tensor([3., -2., 1.]), model.endo_faces, model.rim),
                               p1.cavity_volume(X, model.endo_faces, model.rim), rtol=1e-12, atol=1e-12)


def test_reference_optimized_steps_mass_and_checkpoint(real_case, tmp_path):
    from afsi_torch.simulation.real_lv_mac import run
    from afsi_torch.mac.output import MACWriter
    import meshio
    model = imported_model(real_case)
    settings = dict(dt=real_case.time.dt, fluid_shape=real_case.fluid.shape,
                    fluid_lengths=real_case.fluid.lengths, fluid_origin=real_case.fluid.origin,
                    rho=1., mu=1., interaction_degree=2, mass_solver=asdict(real_case.mass_solver),
                    **asdict(real_case.execution))
    reference = build_driver(model, settings, 'cpu')
    optimized_config = replace(real_case, execution=replace(RealLVConfig().execution, pressure_backend='workspace'))
    optimized = build_driver(model, {**settings, **asdict(optimized_config.execution)}, 'cpu')
    a, b = reference.initialize(model.mesh.X), optimized.initialize(model.mesh.X)
    M = reference.transfer.mass.to_dense()
    local = model.volumes[:, None, None]*(torch.ones((4, 4), dtype=torch.float64)+torch.eye(4, dtype=torch.float64))/20
    expected = torch.zeros_like(M)
    for ids, block in zip(model.mesh.cells, local):
        expected[ids[:, None], ids[None, :]] += block
    torch.testing.assert_close(M, expected, rtol=1e-13, atol=1e-15)
    for _ in range(3):
        a, ia = reference.step(a)
        b, ib = optimized.step(b)
        torch.testing.assert_close(a.x, b.x, rtol=1e-12, atol=1e-13)
        torch.testing.assert_close(a.force, b.force, rtol=1e-8, atol=1e-7)
        assert ia['power_error'] < 1e-9
    path = tmp_path/'checkpoint.npz'
    save_real_lv(path, model, a, settings, dict(elapsed_seconds=0., segments=[], summary={}), real_case)
    restored, state, _, _, cfg = load_real_lv(path)
    torch.testing.assert_close(restored.mesh.fiber, model.mesh.fiber, atol=0, rtol=0)
    torch.testing.assert_close(restored.mesh.sheet, model.mesh.sheet, atol=0, rtol=0)
    torch.testing.assert_close(state.force, a.force)
    assert cfg.material == real_case.material
    writer = MACWriter(tmp_path/'vtk', model, reference.flow.grid)
    writer.write(a)
    vtk = meshio.read(tmp_path/'vtk'/f'solid_{a.step:06d}.vtu')
    assert vtk.cells[0].type == 'tetra'
    np.testing.assert_array_equal(vtk.cell_data['fiber_reference'][0], model.mesh.fiber)
    np.testing.assert_array_equal(vtk.cell_data['sheet_reference'][0], model.mesh.sheet)
    # Fresh demo -> checkpoint -> restart; source files are not needed on restart.
    report = run(case_config=real_case, device='cpu', output=tmp_path/'demo')
    assert report['completed'] and report['accepted_steps'] == 2
    for p in Path(real_case.source_dir).glob('*.xml'):
        p.unlink()
    report = run(device='cpu', resume=tmp_path/'demo/checkpoint.npz', end_time=3e-4)
    assert report['completed'] and report['accepted_steps'] == 3


def test_public_config_and_default_three_cycles(tmp_path):
    cfg = RealLVConfig()
    assert cfg.time == TimeConfig(1e-4, 2.4) and cfg.fluid.shape == (128,)*3
    assert cfg.material.kappa == cfg.beta == 5e6
    assert cfg.basal_center_cm == (7.5, 7.5)
    path = tmp_path/'real.json'
    save_config(path, cfg)
    assert load_config(path, RealLVConfig) == cfg
    assert cfg.time.dt*cfg.fluid.mu/cfg.fluid.rho*sum(1/h**2 for h in cfg.fluid.spacing) < .25
    with pytest.raises(ValueError, match='consistent P1'):
        replace(cfg, interaction_degree=1)
    with pytest.raises(ValueError, match='Guccione'):
        replace(cfg, execution=replace(cfg.execution, solid_backend='pointwise'))


def test_transport_failure_saves_last_accepted_state_and_metrics(real_case, tmp_path, monkeypatch):
    from afsi_torch.simulation.real_lv_mac import run
    from afsi_torch.mac.flow import MACFlow, MACTransportGuardError
    from validation.diagnose_real_lv_guard import diagnose
    metrics = dict(courant=.01, cell_reynolds=1.2, triggered=['cell_reynolds'])
    def reject(*args, **kwargs):
        raise MACTransportGuardError(metrics)
    monkeypatch.setattr(MACFlow, 'step', reject)
    folder = tmp_path/'rejected'
    with pytest.raises(MACTransportGuardError):
        run(case_config=real_case, device='cpu', output=folder)
    report = json.loads((folder/'report.json').read_text())
    assert report['status'] == 'failed' and report['accepted_steps'] == 0
    assert report['failure']['transport_guard'] == metrics
    _, state, _, progress, _ = load_real_lv(folder/'checkpoint.npz')
    assert state.step == 0 and state.time == 0.
    assert progress['failure']['transport_guard'] == metrics
    assert diagnose(folder/'checkpoint.npz')['saved_failure']['transport_guard'] == metrics


@pytest.mark.parametrize('device', DEVICES)
def test_compiled_driver_keeps_ho_and_active_load(real_case, device):
    model = imported_model(real_case, device)
    settings = dict(dt=real_case.time.dt, fluid_shape=real_case.fluid.shape,
                    fluid_lengths=real_case.fluid.lengths, fluid_origin=real_case.fluid.origin,
                    rho=1., mu=1., interaction_degree=2, mass_solver=asdict(real_case.mass_solver),
                    **asdict(real_case.execution))
    reference = build_driver(model, settings, device)
    execution = RealLVConfig().execution
    if device == 'cpu':
        execution = replace(execution, pressure_backend='workspace')
    fast = build_driver(model, {**settings, **asdict(execution)}, device)
    a, b = reference.initialize(model.mesh.X), fast.initialize(model.mesh.X)
    force = model.force(model.mesh.X, .6-real_case.time.dt)
    a = replace(a, step=6000, time=.6, force=force, force_time=.6-real_case.time.dt)
    b = replace(b, step=6000, time=.6, force=force.clone(), force_time=.6-real_case.time.dt)
    for _ in range(3):
        a, ia = reference.step(a)
        b, ib = fast.step(b)
        torch.testing.assert_close(b.x, a.x, rtol=1e-10, atol=1e-11)
        torch.testing.assert_close(b.force, a.force, rtol=1e-8, atol=2e-6)
        torch.testing.assert_close(b.pressure, a.pressure, rtol=1e-8, atol=1e-6)
        for left, right in zip(a.velocity, b.velocity):
            torch.testing.assert_close(left, right, rtol=1e-8, atol=1e-10)
        scale = max(abs(ib['solid_power']), abs(ib['fluid_power']), 1.)
        assert ib['power_error']/scale < 1e-10
    if device == 'cpu':
        # Dynamo tracing catches graph breaks even on the local CPU-only host.
        kernel = torch.compile(model.geometry_state, backend='eager', fullgraph=True)
        geometry = kernel(a.x)
        force_kernel = torch.compile(model.force_from_geometry, backend='eager', fullgraph=True)
        value = force_kernel(a.x, geometry[0], geometry[1], a.x.new_tensor(model.loads.at(a.time)))
        torch.testing.assert_close(value, model.force(a.x, a.time))
