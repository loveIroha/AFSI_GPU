"""Pressure/cofactor, reference-area springs and full solid weak-force tests."""
from math import factorial
import pytest
import torch
from afsi_torch import boundary as bd, triangle, solid
from afsi_torch.fields import prepare_reference_fields
from afsi_torch.materials import GuccioneParameters
from afsi_torch.tetrahedron import promote_p1, reference_nodes


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device unavailable"))])
def device(request):
    return request.param


def mesh(device, two=False):
    if two:
        vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                                 [0., 0., 1.], [1., 1., 1.]], dtype=torch.float64, device=device)
        X, cells = promote_p1(vertices, torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], device=device))
    else:
        X = reference_nodes(device=device)
        cells = torch.arange(10, device=device).reshape(1, 10)
    faces = bd.extract_boundary(X, cells)
    return X, cells, faces


def bottom_surface(X, faces, degree=4):
    selected = (X[faces[:, :3], 2].abs() < 1e-12).all(-1)
    return bd.prepare_surface(X, faces[selected], degree)


def test_triangle_basis_and_polynomials(device):
    nodes = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [.5, 0.], [0., .5], [.5, .5]], dtype=torch.float64, device=device)
    N, _ = triangle.tabulate(nodes)
    torch.testing.assert_close(N, torch.eye(6, device=device, dtype=nodes.dtype), atol=1e-14, rtol=0)
    points, _ = triangle.quadrature(4, device=device)
    N, dN = triangle.tabulate(points)
    for i in range(3):
        for j in range(3-i):
            data = nodes[:, 0]**i*nodes[:, 1]**j
            torch.testing.assert_close(N@data, points[:, 0]**i*points[:, 1]**j, atol=1e-14, rtol=1e-13)
            grad = torch.zeros_like(points)
            if i:
                grad[:, 0] = i*points[:, 0]**(i-1)*points[:, 1]**j
            if j:
                grad[:, 1] = j*points[:, 0]**i*points[:, 1]**(j-1)
            torch.testing.assert_close(torch.einsum("qaj,a->qj", dN, data), grad, atol=1e-14, rtol=1e-13)


@pytest.mark.parametrize("degree", [0, 4, 6])
def test_triangle_quadrature_moments(device, degree):
    points, weights = triangle.quadrature(degree, device=device)
    assert (weights > 0).all()
    for i in range(degree+1):
        for j in range(degree+1-i):
            actual = (weights*points[:, 0]**i*points[:, 1]**j).sum()
            torch.testing.assert_close(actual, actual.new_tensor(factorial(i)*factorial(j)/factorial(i+j+2)), atol=1e-15, rtol=1e-12)


def test_exterior_faces_and_outward_orientation(device):
    X, cells, faces = mesh(device, two=True)
    assert faces.shape == (6, 6)  # 8 cell faces minus the two shared-face copies.
    surface = bd.prepare_surface(X, faces)
    for face, area in zip(faces, surface.reference_area_vectors):
        matches = torch.isin(cells[:, :4], face[:3]).sum(-1) == 3
        assert matches.sum() == 1
        owner = cells[matches, :4][0]
        opposite = owner[~torch.isin(owner, face[:3])][0]
        assert (area*(X[opposite]-X[face[0]])).sum() < 0
    # Reordering tetra vertices must preserve the physical outward surface.
    reordered_X, reordered_cells = promote_p1(X[:5], cells[:, [0, 2, 1, 3]])
    reordered_surface = bd.prepare_surface(reordered_X, bd.extract_boundary(reordered_X, reordered_cells))
    torch.testing.assert_close(surface.reference_weights.sum(), reordered_surface.reference_weights.sum())
    torch.testing.assert_close(bd.pressure_force(X, surface, 2.), bd.pressure_force(reordered_X, reordered_surface, 2.), atol=1e-13, rtol=1e-12)
    torch.testing.assert_close(bd.pressure_force(X, surface, 2.).sum(0), X.new_zeros(3), atol=1e-13, rtol=0)


def test_flat_pressure_nodal_distribution_and_sign(device):
    X, _, faces = mesh(device)
    surface = bottom_surface(X, faces)
    force = bd.pressure_force(X, surface, 12.)
    expected = torch.zeros_like(X)
    expected[surface.faces[0, 3:], 2] = 2.  # +z: solid normal is -z, area=1/2.
    torch.testing.assert_close(force, expected, atol=1e-13, rtol=0)


def test_affine_cofactor_area_mapping(device):
    X, _, faces = mesh(device)
    surface = bd.prepare_surface(X, faces)
    A = X.new_tensor([[1.2, .2, .1], [.1, .9, 0.], [0., .1, 1.1]])
    x = X@A.T+X.new_tensor([.3, -.2, .7])
    expected = surface.reference_area_vectors @ (torch.linalg.det(A)*torch.linalg.inv(A).T).T
    torch.testing.assert_close(bd.area_vectors(x, surface), expected[:, None, :].expand(-1, surface.values.shape[0], -1), atol=1e-13, rtol=1e-12)
    R = X.new_tensor([[.8, -.6, 0.], [.6, .8, 0.], [0., 0., 1.]])
    force = bd.pressure_force(x, surface, 3.)
    torch.testing.assert_close(bd.pressure_force(x@R.T+1., surface, 3.), force@R.T, atol=1e-12, rtol=1e-11)


def test_curved_surface_analytic_resultant(device):
    X, _, faces = mesh(device)
    surface = bottom_surface(X, faces)
    x = X.clone()
    x[:, 2] += .2*X[:, 0].square()+.1*X[:, 1].square()
    bd.validate_surface(x, surface)
    force = bd.pressure_force(x, surface, 6.)
    torch.testing.assert_close(force.sum(0), X.new_tensor([-.4, -.2, 3.]), atol=1e-13, rtol=1e-12)
    # Pressure is follower: holding the reference normal would miss x/y force.
    assert (force-bd.pressure_force(X, surface, 6.)).abs().max() > .01


def test_closed_pressure_volume_gradient_and_balance(device):
    X, cells, faces = mesh(device, two=True)
    surface = bd.prepare_surface(X, faces)
    x = X+.04*X.square()
    force = bd.pressure_force(x, surface, 2.)
    # Only a CLOSED, consistently oriented surface permits this volume test.
    volume = lambda y: (surface.quadrature_weights[None, :]*(bd.interpolate(y, surface)*bd.area_vectors(y, surface)).sum(-1)).sum()/3
    torch.testing.assert_close(volume(X), X.new_tensor(.5), atol=1e-13, rtol=0)
    torch.testing.assert_close(force, -2*torch.func.grad(volume)(x), atol=1e-12, rtol=1e-11)
    torch.testing.assert_close(force.sum(0), X.new_zeros(3), atol=1e-12, rtol=0)
    torch.testing.assert_close(torch.linalg.cross(x, force).sum(0), X.new_zeros(3), atol=1e-12, rtol=0)


def test_open_pressure_jvp_and_nonsymmetric_tangent(device):
    X, _, faces = mesh(device)
    surface = bottom_surface(X, faces)
    fn = lambda y: bd.pressure_force(y, surface, 3.)
    x = (X+.03*X.square()).requires_grad_()
    assert torch.autograd.gradcheck(fn, (x,), atol=1e-8, rtol=1e-5)
    direction = torch.sin(torch.arange(x.numel(), device=device, dtype=x.dtype)).reshape_as(x)
    _, Kv = torch.func.jvp(fn, (x,), (direction,))
    h = 1e-6
    torch.testing.assert_close(Kv, (fn(x+h*direction)-fn(x-h*direction))/(2*h), atol=1e-8, rtol=1e-6)
    K = torch.func.jacrev(fn)(x).reshape(x.numel(), x.numel())
    assert (K-K.T).abs().max() > .1  # Cannot replace this with an energy Hessian.


def test_spring_reference_measure_and_energy(device):
    X, _, faces = mesh(device)
    surface = bottom_surface(X, faces)
    beta = bd.prepare_coefficient(surface, 12., nonnegative=True)
    torch.testing.assert_close(bd.spring_force(X, surface, beta), torch.zeros_like(X), atol=1e-14, rtol=0)
    shift = X.new_tensor([.1, -.2, .3])
    torch.testing.assert_close(bd.spring_force(X+shift, surface, beta).sum(0), -6*shift, atol=1e-13, rtol=1e-12)
    torch.testing.assert_close(bd.spring_energy(X+shift, surface, beta), 3*shift.square().sum(), atol=1e-13, rtol=1e-12)
    x = X+.1*X.square()+shift
    fn = lambda y: bd.spring_energy(y, surface, beta)
    torch.testing.assert_close(bd.spring_force(x, surface, beta), -torch.func.grad(fn)(x), atol=1e-12, rtol=1e-11)
    # Integral X0^2 dA0=1/12; stretching x0 by 1.1 on this reference face.
    stretched = X.clone()
    stretched[:, 0] *= 1.1
    torch.testing.assert_close(bd.spring_energy(stretched, surface, beta), X.new_tensor(.005), atol=1e-13, rtol=1e-12)


def test_spatial_coefficients_and_validation(device):
    X, _, faces = mesh(device)
    surface = bottom_surface(X, faces, degree=6)
    p = bd.prepare_coefficient(surface, 1.+X[:, 0].square())
    torch.testing.assert_close(bd.pressure_force(X, surface, p).sum(0), X.new_tensor([0., 0., 7/12]), atol=1e-13, rtol=1e-12)
    for value in (-1., float("nan")):
        with pytest.raises(ValueError, match="finite"):
            bd.prepare_coefficient(surface, value, nonnegative=True)
    with pytest.raises(ValueError, match="coefficient"):
        bd.prepare_coefficient(surface, X)
    with pytest.raises(ValueError, match="degenerate"):
        bd.validate_surface(torch.zeros_like(X), surface)
    curved_ref = X.clone()
    curved_ref[surface.faces[0, 3], 2] += .01
    with pytest.raises(ValueError, match="midpoints"):
        bd.prepare_surface(curved_ref, surface.faces)
    with pytest.raises(ValueError, match="duplicate"):
        bd.prepare_surface(X, torch.cat((surface.faces, surface.faces)))


def test_reject_nonconforming_topology(device):
    X, cells, _ = mesh(device, two=True)
    shared_mid = cells[1, 4]
    bad_X = torch.cat((X, X[shared_mid:shared_mid+1]))
    bad_cells = cells.clone()
    bad_cells[1, 4] = X.shape[0]
    with pytest.raises(ValueError, match="nonconforming"):
        bd.extract_boundary(bad_X, bad_cells)
    with pytest.raises(ValueError, match="duplicate"):
        bd.extract_boundary(X, torch.cat((cells, cells[:1])))
    vertices = X.new_tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                              [0., 0., 1.], [0., 0., -1.], [0., 0., 2.]])
    three_X, three_cells = promote_p1(vertices, torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5]], device=device))
    with pytest.raises(ValueError, match="nonmanifold"):
        bd.extract_boundary(three_X, three_cells)


def test_combined_solid_force_tangent(device):
    X, cells, faces = mesh(device)
    geo = solid.prepare_p2(X, cells)
    fields = prepare_reference_fields(geo, [1., 0., 0.], [0., 1., 0.], .3)
    parameters = GuccioneParameters(C=2., kappa=30.)
    base = bottom_surface(X, faces)
    loaded = bd.prepare_surface(X, faces[(X[faces[:, :3]].sum(-1) > .9).all(-1)])
    pressure = bd.prepare_coefficient(loaded, 2.)
    beta = bd.prepare_coefficient(base, 5., nonnegative=True)
    force = lambda y: solid.guccione_force(y, geo, fields, parameters)+bd.pressure_force(y, loaded, pressure)+bd.spring_force(y, base, beta)
    x = X+.02*X.square()
    direction = torch.cos(torch.arange(x.numel(), device=device, dtype=x.dtype)).reshape_as(x)
    _, Kv = torch.func.jvp(lambda y: -force(y), (x,), (direction,))
    h = 1e-6
    torch.testing.assert_close(Kv, -(force(x+h*direction)-force(x-h*direction))/(2*h), atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(force(x).sum(0), (bd.pressure_force(x, loaded, pressure)+bd.spring_force(x, base, beta)).sum(0), atol=1e-12, rtol=1e-11)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_boundary_cpu_cuda_agreement():
    results = []
    for device in ("cpu", "cuda"):
        X, _, faces = mesh(device, two=True)
        surface = bd.prepare_surface(X, faces)
        x = X+.03*X.square()
        fn = lambda y: bd.pressure_force(y, surface, 1200.)+bd.spring_force(y, surface, 5000.)
        _, Kv = torch.func.jvp(fn, (x,), (torch.sin(x),))
        results.append((fn(x).cpu(), Kv.cpu()))
    for cpu, gpu in zip(*results):
        torch.testing.assert_close(cpu, gpu, atol=1e-8, rtol=1e-10)
