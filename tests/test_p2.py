"""Independent polynomial/continuum oracles in addition to AD consistency."""
from math import factorial
import pytest
import torch

from afsi_torch import solid
from afsi_torch.quadrature import tetrahedron_rule
from afsi_torch.tetrahedron import promote_p1, reference_nodes, tabulate


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device unavailable"))])
def device(request):
    return request.param


def two_cells(device="cpu"):
    X = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                      [0., 0., 1.], [1., 1., 1.]], dtype=torch.float64, device=device)
    cells = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], device=device)
    return promote_p1(X, cells)


def one_cell(device="cpu"):
    X = reference_nodes(device=device)
    return X, torch.arange(10, device=device).reshape(1, 10)


def test_p2_nodal_basis_and_polynomial_reproduction(device):
    nodes = reference_nodes(device=device)
    N, _ = tabulate(nodes)
    torch.testing.assert_close(N, torch.eye(10, dtype=nodes.dtype, device=device), atol=1e-14, rtol=0)
    points, _ = tetrahedron_rule(4, device=device)
    N, dN = tabulate(points)
    torch.testing.assert_close(N.sum(-1), torch.ones_like(N[:, 0]), atol=1e-14, rtol=0)
    torch.testing.assert_close(dN.sum(1), torch.zeros_like(points), atol=1e-14, rtol=0)
    # Complete P2 polynomial space, including each cross term and its derivative.
    for i in range(3):
        for j in range(3-i):
            for k in range(3-i-j):
                powers = (i, j, k)
                nodal = torch.prod(nodes ** nodes.new_tensor(powers), dim=1)
                exact = torch.prod(points ** points.new_tensor(powers), dim=1)
                torch.testing.assert_close(N @ nodal, exact, atol=1e-14, rtol=1e-13)
                expected_grad = torch.zeros_like(points)
                for d, exponent in enumerate(powers):
                    if exponent:
                        reduced = list(powers)
                        reduced[d] -= 1
                        expected_grad[:, d] = exponent * torch.prod(points ** points.new_tensor(reduced), dim=1)
                torch.testing.assert_close(torch.einsum("qaj,a->qj", dN, nodal), expected_grad, atol=1e-14, rtol=1e-13)


@pytest.mark.parametrize("degree", [0, 2, 4, 6])
def test_quadrature_exact_moments(device, degree):
    points, weights = tetrahedron_rule(degree, device=device)
    assert (weights > 0).all() and (points >= 0).all() and (points.sum(1) < 1).all()
    for i in range(degree+1):
        for j in range(degree+1-i):
            for k in range(degree+1-i-j):
                value = (weights * points[:, 0]**i * points[:, 1]**j * points[:, 2]**k).sum()
                exact = factorial(i)*factorial(j)*factorial(k)/factorial(i+j+k+3)
                torch.testing.assert_close(value, value.new_tensor(exact), atol=2e-15, rtol=1e-12)


def test_shared_dofs_and_affine_patch(device):
    X, cells = two_cells(device)
    assert X.shape == (14, 3)  # 5 vertices + 9 unique edges, not 12 edges.
    assert torch.isin(cells[0], cells[1]).sum() == 6  # Shared quadratic face.
    geo = solid.prepare_p2(X, cells)
    torch.testing.assert_close(geo.weights.sum(), X.new_tensor(0.5), atol=1e-14, rtol=0)
    A = X.new_tensor([[1.2, 0.1, 0.], [0., .9, .05], [0., 0., 1.1]])
    x = X @ A.T + X.new_tensor([.3, -.1, .2])
    F = solid.deformation_gradient(x, geo)
    torch.testing.assert_close(F, A.expand_as(F), atol=1e-13, rtol=1e-13)
    logJ = X.new_tensor(1.188).log()
    exact = .5*((1.2**2+.1**2+.9**2+.05**2+1.1**2-3)-2*logJ+2.5*logJ**2)
    torch.testing.assert_close(solid.energy(x, geo, 2., 5.), exact, atol=1e-13, rtol=1e-12)
    # Constant P has zero integrated vertex force in a P2 tetrahedron.
    force = solid.stress_force(x, geo, 2., 5.)
    torch.testing.assert_close(force[:5], torch.zeros_like(force[:5]), atol=1e-13, rtol=0)
    torch.testing.assert_close(force, -torch.func.grad(lambda y: solid.energy(y, geo, 2., 5.))(x), atol=1e-12, rtol=1e-11)


def test_rigid_motion_and_objectivity(device):
    X, cells = two_cells(device)
    geo = solid.prepare_p2(X, cells)
    R = X.new_tensor([[.8, -.6, 0.], [.6, .8, 0.], [0., 0., 1.]])
    rigid = X @ R.T + X.new_tensor([.2, -.3, .4])
    solid.validate_deformation(rigid, geo)
    torch.testing.assert_close(solid.energy(rigid, geo, 2., 5.), X.new_tensor(0.), atol=1e-13, rtol=0)
    torch.testing.assert_close(solid.stress_force(rigid, geo, 2., 5.), torch.zeros_like(X), atol=1e-12, rtol=0)
    x = X + .07*X.square()
    rotated = x @ R.T + X.new_tensor([.2, -.3, .4])
    torch.testing.assert_close(solid.energy(rotated, geo, 2., 5.), solid.energy(x, geo, 2., 5.), atol=1e-12, rtol=1e-11)
    force = solid.stress_force(x, geo, 2., 5.)
    torch.testing.assert_close(solid.stress_force(rotated, geo, 2., 5.), force @ R.T, atol=1e-12, rtol=1e-11)


def test_quadratic_shear_exact_continuum_energy(device):
    X, cells = one_cell(device)
    geo = solid.prepare_p2(X, cells)
    alpha, mu = .2, 2.
    x = X.clone()
    x[:, 0] += alpha*X[:, 1]**2
    solid.validate_deformation(x, geo)
    # x=(X0+alpha*X1^2,X1,X2): J=1, W=2*mu*alpha^2*X1^2.
    # Integral_unit_tet X1^2 dX = 1/60; independent of lambda.
    exact = X.new_tensor(mu*alpha**2/30)
    torch.testing.assert_close(solid.energy(x, geo, mu, 5.), exact, atol=1e-13, rtol=1e-12)
    force = solid.stress_force(x, geo, mu, 5.)
    torch.testing.assert_close(force.sum(0), X.new_zeros(3), atol=1e-12, rtol=0)
    torch.testing.assert_close(torch.linalg.cross(x, force).sum(0), X.new_zeros(3), atol=1e-12, rtol=0)


def test_force_tangent_and_shared_node_assembly(device):
    X, cells = two_cells(device)
    geo = solid.prepare_p2(X, cells)
    x = (X + .08*X.square()).requires_grad_()
    fn = lambda y: solid.energy(y, geo, 2., 5.)
    assert torch.autograd.gradcheck(fn, (x,), atol=1e-7, rtol=1e-5)
    residual = torch.func.grad(fn)
    force = solid.stress_force(x, geo, 2., 5.)
    torch.testing.assert_close(force, -residual(x), atol=1e-12, rtol=1e-11)
    # Independent per-cell accumulation verifies shared-node index_add.
    expected = torch.zeros_like(x)
    for cell in cells:
        local_geo = solid.prepare_p2(X[cell], torch.arange(10, device=device).reshape(1, 10))
        expected[cell] += solid.stress_force(x[cell], local_geo, 2., 5.)
    torch.testing.assert_close(force, expected, atol=1e-12, rtol=1e-11)
    direction = torch.sin(torch.arange(x.numel(), device=device, dtype=x.dtype)).reshape_as(x)
    _, Kv = torch.func.jvp(residual, (x,), (direction,))
    h = 1e-6
    fd = (residual(x+h*direction)-residual(x-h*direction))/(2*h)
    torch.testing.assert_close(Kv, fd, atol=2e-7, rtol=1e-5)
    # Stress-free reference state must also have a finite, correct tangent.
    _, K0v = torch.func.jvp(residual, (X,), (direction,))
    fd0 = (residual(X+h*direction)-residual(X-h*direction))/(2*h)
    torch.testing.assert_close(K0v, fd0, atol=2e-7, rtol=1e-5)


def test_quadrature_convergence_for_nonpolynomial_energy(device):
    X, cells = one_cell(device)
    x = X.clone()
    x[:, 0] += .3*X[:, 0]**2  # J=1+0.6*X0, so log(J) is nonpolynomial.
    values = [solid.energy(x, solid.prepare_p2(X, cells, degree=d), 2., 5.) for d in (2, 4, 6, 16)]
    errors = [(v-values[-1]).abs() for v in values[:-1]]
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 1e-7


def test_invalid_reference_and_deformation(device):
    X, cells = one_cell(device)
    with pytest.raises(ValueError, match="nondegenerate"):
        solid.prepare_p2(torch.zeros_like(X), cells)
    curved = X.clone()
    curved[4, 2] += .01
    with pytest.raises(ValueError, match="midpoints"):
        solid.prepare_p2(curved, cells)
    reordered = cells.clone()
    reordered[:, [4, 5]] = reordered[:, [5, 4]]
    with pytest.raises(ValueError, match="midpoints"):
        solid.prepare_p2(X, reordered)
    geo = solid.prepare_p2(X, cells)
    x = X.clone()
    x[:, 0] *= -1
    with pytest.raises(ValueError, match="det"):
        solid.validate_deformation(x, geo)


def test_reversed_cell_order_preserves_polynomial_energy(device):
    vertices = reference_nodes(device=device)[:4]
    results = []
    for ordering in ([0, 1, 2, 3], [0, 2, 1, 3]):
        X, cells = promote_p1(vertices, torch.tensor([ordering], device=device))
        geo = solid.prepare_p2(X, cells)
        x = X.clone()
        x[:, 0] += .2*X[:, 1]**2
        solid.validate_deformation(x, geo)
        results.append(solid.energy(x, geo, 2., 5.))
    torch.testing.assert_close(*results, atol=1e-13, rtol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_p2_cpu_cuda_agreement():
    results = []
    for device in ("cpu", "cuda"):
        X, cells = two_cells(device)
        geo = solid.prepare_p2(X, cells)
        x = X + .08*X.square()
        fn = lambda y: solid.energy(y, geo, 2., 5.)
        _, tangent = torch.func.jvp(torch.func.grad(fn), (x,), (torch.sin(x),))
        results.append((fn(x).cpu(), solid.stress_force(x, geo, 2., 5.).cpu(), tangent.cpu()))
    for cpu, gpu in zip(*results):
        torch.testing.assert_close(cpu, gpu, atol=1e-10, rtol=1e-9)
