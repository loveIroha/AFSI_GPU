"""Generated LV geometry, CGS scaling, fibers and virtual-cap measurement."""
import numpy as np
import pytest
import torch
from afsi_torch.geometry import (LVConfig, generate_lv, ENDO, EPI, BASE,
    signed_cell_volumes, prepare_cavity, cavity_volume, rule_based_fibers)
from afsi_torch import solid, boundary as bd
from afsi_torch.fields import prepare_reference_fields
from afsi_torch.materials import GuccioneParameters


@pytest.fixture(scope='module')
def lv():
    pytest.importorskip('gmsh')
    return generate_lv(LVConfig(mesh_size=1.4))


@pytest.fixture(params=['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA device unavailable'))])
def device(request):
    return request.param


def test_tags_normals_and_closed_wall(lv):
    X, faces = lv.X, lv.faces
    assert set(lv.facet_tags.tolist()) == {ENDO, EPI, BASE}
    vols = signed_cell_volumes(X, lv.cells)
    assert (vols > 0).all()
    assert abs(vols.sum().item()/lv.config.wall_volume-1) < .06
    v = X[faces[:, :3]]
    n = torch.linalg.cross(v[:, 1]-v[:, 0], v[:, 2]-v[:, 0])
    centers = v.mean(1)-X.new_tensor(lv.config.center)
    for tag, axes, sign in ((ENDO, lv.config.inner_axes, -1), (EPI, lv.config.outer_axes, 1)):
        selected = lv.facet_tags == tag
        assert (sign*(n[selected]*centers[selected]/X.new_tensor(axes).square()).sum(-1) > 0).all()
    assert (n[lv.facet_tags == BASE, 2] > 0).all()
    edges = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2).sort(1).values
    _, counts = torch.unique(edges, dim=0, return_counts=True)
    assert (counts == 2).all()


def test_geometry_refinement_and_cm_volume(lv):
    fine = generate_lv(LVConfig(mesh_size=.9))
    exact = lv.config.cavity_volume
    assert 80 < exact < 100  # cm^3 == mL, not cubic metres.
    coarse_v = cavity_volume(lv.X, prepare_cavity(lv.X, lv.surface(ENDO))).item()
    fine_v = cavity_volume(fine.X, prepare_cavity(fine.X, fine.surface(ENDO))).item()
    assert abs(fine_v-exact) < abs(coarse_v-exact)
    assert abs(fine_v/exact-1) < .05
    assert len(fine.cells) > len(lv.cells)


def test_non_axisymmetric_shifted_geometry():
    pytest.importorskip('gmsh')
    config = LVConfig(inner_axes=(2.4, 2., 4.2), outer_axes=(3.3, 3., 5.3),
                      center=(4., -3., 2.), base_height=.8, mesh_size=1.5)
    m = generate_lv(config)
    c = prepare_cavity(m.X, m.surface(ENDO))
    assert abs(cavity_volume(m.X, c).item()/config.cavity_volume-1) < .15
    torch.testing.assert_close(m.X[m.surface(BASE)][:, :, 2],
        m.X.new_full(m.surface(BASE).shape, config.center[2]+config.base_height), atol=1e-10, rtol=0)


def test_affine_volume_and_fixed_rim_pressure_work(lv, device):
    m = lv.to(device)
    c = prepare_cavity(m.X, m.surface(ENDO))
    V = cavity_volume(m.X, c)
    A = m.X.new_tensor([[1.03, .02, 0], [0, .98, .01], [0, 0, 1.01]])
    x = m.X@A.T+m.X.new_tensor([2., -1., 3.])
    torch.testing.assert_close(cavity_volume(x, c), torch.linalg.det(A)*V, atol=1e-10, rtol=1e-11)
    curved = x+.001*m.X.square()
    torch.testing.assert_close(cavity_volume(curved+curved.new_tensor([3., 1., -2.]), c),
                               cavity_volume(curved, c), atol=1e-10, rtol=1e-11)
    d = torch.sin(torch.arange(x.numel(), device=device, dtype=x.dtype)).reshape_as(x)*.01
    d[torch.unique(c.rim)] = 0  # Hold the entire P2 base rim fixed.
    _, dV = torch.func.jvp(lambda y: cavity_volume(y, c), (x,), (d,))
    p = 10000.
    force = bd.pressure_force(x, c.endo, p)
    torch.testing.assert_close((force*d).sum(), p*dV, atol=1e-8, rtol=1e-10)
    eps = 1e-5
    fd = (cavity_volume(x+eps*d, c)-cavity_volume(x-eps*d, c))/(2*eps)
    torch.testing.assert_close(fd, dV, atol=2e-8, rtol=1e-6)


def test_fiber_frame_and_generated_solid(lv, device):
    m = lv.to(device)
    nodal = rule_based_fibers(m.X, m.config)
    for vector in (nodal.fiber, nodal.sheet):
        torch.testing.assert_close(vector.square().sum(-1), torch.ones_like(nodal.transmural), atol=1e-12, rtol=0)
    torch.testing.assert_close((nodal.fiber*nodal.sheet).sum(-1), torch.zeros_like(nodal.transmural), atol=1e-12, rtol=0)
    assert ((nodal.transmural >= 0) & (nodal.transmural <= 1)).all()
    geo = solid.prepare_p2(m.X, m.cells)
    fields = prepare_reference_fields(geo, nodal.fiber, nodal.sheet, 1000.)
    x = m.X*1.01
    solid.validate_deformation(x, geo)
    force = solid.guccione_force(x, geo, fields, GuccioneParameters())
    assert torch.isfinite(force).all()
    torch.testing.assert_close(force.sum(0), force.new_zeros(3), atol=2e-7, rtol=0)


def test_apex_has_finite_smooth_escape(device):
    cfg = LVConfig()
    X = torch.tensor([[0., 0., -5.], [1e-7, 0., -5.], [-1e-7, 0., -5.],
                      [0., 1e-7, -5.], [0., -1e-7, -5.]], dtype=torch.float64, device=device)
    f = rule_based_fibers(X, cfg)
    assert torch.isfinite(f.fiber).all() and torch.isfinite(f.sheet).all()
    torch.testing.assert_close(f.fiber, f.fiber[:1].expand_as(X), atol=2e-6, rtol=0)
    torch.testing.assert_close(f.sheet, f.sheet[:1].expand_as(X), atol=2e-6, rtol=0)
    assert (f.apical_weight > .99).all()
    with pytest.raises(ValueError):
        rule_based_fibers(torch.zeros((1, 3), device=device), cfg)


def test_cgs_to_si_force_and_energy_scaling(device):
    from afsi_torch.tetrahedron import reference_nodes
    X = reference_nodes(device=device)
    cells = torch.arange(10, device=device).reshape(1, 10)
    gc, gs = solid.prepare_p2(X, cells), solid.prepare_p2(X*.01, cells)
    fc = prepare_reference_fields(gc, [1., 0., 0.], [0., 1., 0.], 1000.)
    fs = prepare_reference_fields(gs, [1., 0., 0.], [0., 1., 0.], 100.)
    pc, ps = GuccioneParameters(), GuccioneParameters(C=2000., kappa=50000.)
    x = X*1.02
    torch.testing.assert_close(solid.guccione_force(x*.01, gs, fs, ps),
                               solid.guccione_force(x, gc, fc, pc)*1e-5, atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(solid.guccione_energy(x*.01, gs, fs, ps),
                               solid.guccione_energy(x, gc, fc, pc)*1e-7, atol=1e-13, rtol=1e-10)
    faces = bd.extract_boundary(X, cells)
    bc, bs = bd.prepare_surface(X, faces), bd.prepare_surface(.01*X, faces)
    torch.testing.assert_close(bd.spring_force(.01*x, bs, 5e6),
                               bd.spring_force(x, bc, 5e5)*1e-5, atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(bd.pressure_force(.01*x, bs, 1000.),
                               bd.pressure_force(x, bc, 10000.)*1e-5, atol=1e-11, rtol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device unavailable')
def test_generated_lv_cpu_gpu_parity(lv):
    results = []
    for device in ('cpu', 'cuda'):
        m = lv.to(device)
        nodal = rule_based_fibers(m.X, m.config)
        geo = solid.prepare_p2(m.X, m.cells)
        fields = prepare_reference_fields(geo, nodal.fiber, nodal.sheet, 1000.)
        f = solid.guccione_force(1.01*m.X, geo, fields, GuccioneParameters())
        c = prepare_cavity(m.X, m.surface(ENDO))
        results.append((nodal.fiber.cpu(), nodal.sheet.cpu(), f.cpu(), cavity_volume(m.X, c).cpu()))
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, atol=2e-7, rtol=1e-9)


def test_vtk_export_has_correct_p2_edges(lv, tmp_path):
    meshio = pytest.importorskip('meshio')
    from afsi_torch.geometry.output import write_lv
    fields = rule_based_fibers(lv.X, lv.config)
    write_lv(tmp_path, lv, fields)
    volume = meshio.read(tmp_path/'reference.vtu')
    ids = volume.cells_dict['tetra10']
    pairs = np.array([[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]])
    np.testing.assert_allclose(volume.points[ids[:, 4:]], volume.points[ids[:, pairs]].mean(2), atol=1e-12)
    surface = meshio.read(tmp_path/'surfaces.vtu')
    ids = surface.cells_dict['triangle6']
    pairs = np.array([[0, 1], [1, 2], [2, 0]])
    np.testing.assert_allclose(surface.points[ids[:, 3:]], surface.points[ids[:, pairs]].mean(2), atol=1e-12)
    assert set(surface.cell_data['surface_tag'][0]) == {ENDO, EPI, BASE}
    with np.load(tmp_path/'generated.npz', allow_pickle=False) as data:
        assert '"length": "cm"' in str(data['metadata'])


@pytest.mark.parametrize('kwargs', [dict(inner_axes=(4., 2., 4.)), dict(base_height=4.5),
    dict(mesh_size=0.), dict(center=(0., float('nan'), 0.))])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        LVConfig(**kwargs)


def test_does_not_clear_existing_gmsh_session():
    gmsh = pytest.importorskip('gmsh')
    gmsh.initialize([], readConfigFiles=False)
    try:
        gmsh.model.add('caller_owned')
        with pytest.raises(RuntimeError, match='own Gmsh session'):
            generate_lv()
        assert gmsh.model.getCurrent() == 'caller_owned'
    finally:
        gmsh.finalize()
