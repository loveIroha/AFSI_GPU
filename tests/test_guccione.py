"""Pointwise constitutive oracles, reference fields, FEM forces and tangents."""
import pytest
import torch

from afsi_torch import materials as mat, solid
from afsi_torch.fields import interpolate_p2, prepare_reference_fields
from afsi_torch.tetrahedron import promote_p1


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device unavailable"))])
def device(request):
    return request.param


def patch(device):
    vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                             [0., 0., 1.], [1., 1., 1.]], dtype=torch.float64, device=device)
    X, cells = promote_p1(vertices, torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], device=device))
    return X, solid.prepare_p2(X, cells)


def varying_fields(X, geo, active=True):
    angle = .6*X[:, 0]
    zero = torch.zeros_like(angle)
    fiber = torch.stack((angle.cos(), angle.sin(), zero), -1)
    sheet = torch.stack((-angle.sin(), angle.cos(), zero), -1)
    tension = .4+.2*X[:, 1].square() if active else 0.
    return prepare_reference_fields(geo, fiber, sheet, tension)


def axes(device):
    eye = torch.eye(3, dtype=torch.float64, device=device)
    return eye[0], eye[1], -eye[2]


def test_reference_rigid_and_active_stress(device):
    f, s, n = axes(device)
    p = mat.GuccioneParameters()
    I = torch.eye(3, dtype=f.dtype, device=device)
    R = I.new_tensor([[.8, -.6, 0.], [.6, .8, 0.], [0., 0., 1.]])
    for F in (I, R):
        torch.testing.assert_close(mat.guccione_energy(F, f, s, n, p), I.new_tensor(0.), atol=1e-20, rtol=0)
        torch.testing.assert_close(mat.guccione_pk1(F, f, s, n, p), torch.zeros_like(I), atol=1e-9, rtol=0)
    # Active stress is NONZERO at reference, despite zero reference potential.
    torch.testing.assert_close(mat.active_pk1(I, f, 1200.), I.new_tensor([[1200., 0., 0.], [0., 0., 0.], [0., 0., 0.]]))
    torch.testing.assert_close(mat.active_potential(I, f, 1200.), I.new_tensor(0.))


def test_uniaxial_analytic_and_anisotropy(device):
    f, s, n = axes(device)
    p = mat.GuccioneParameters()
    stretch = 1.08
    E = (stretch**2-1)/2
    energies = []
    for axis, b in ((0, p.bf), (1, p.bt)):
        F = torch.eye(3, dtype=f.dtype, device=device)
        F[axis, axis] = stretch
        Q = F.new_tensor(b*E**2)
        expected_W = .5*p.C*Q.expm1()+p.kappa*(stretch-1)**2
        expected_P = p.C*b*E*stretch*Q.exp()+2*p.kappa*(stretch-1)
        W = mat.guccione_energy(F, f, s, n, p)
        torch.testing.assert_close(W, expected_W, atol=1e-10, rtol=1e-12)
        torch.testing.assert_close(mat.guccione_pk1(F, f, s, n, p)[axis, axis], expected_P, atol=1e-8, rtol=1e-12)
        energies.append(W)
    assert energies[0] > energies[1]


def test_constitutive_gradient_nonorthonormal_frame(device):
    f, s, _ = axes(device)
    f = 1.1*f + f.new_tensor([0., .12, .03])
    s = .9*s + s.new_tensor([.07, 0., .04])
    n = torch.linalg.cross(s, f)
    F = f.new_tensor([[1.07, .04, -.02], [.01, .96, .03], [.02, -.01, 1.03]])
    p = mat.GuccioneParameters(C=2., kappa=30.)
    fn = lambda A: mat.guccione_energy(A, f, s, n, p)+mat.active_potential(A, f, .8)
    assert torch.autograd.gradcheck(fn, (F.requires_grad_(),), atol=1e-7, rtol=1e-5)
    expected = mat.guccione_pk1(F, f, s, n, p)+mat.active_pk1(F, f, .8)
    torch.testing.assert_close(torch.func.grad(fn)(F), expected, atol=1e-12, rtol=1e-11)


def test_frame_objectivity(device):
    f, s, n = axes(device)
    p = mat.GuccioneParameters()
    F = f.new_tensor([[1.03, .05, .01], [0., .97, .02], [0., 0., 1.02]])
    R = f.new_tensor([[.8, -.6, 0.], [.6, .8, 0.], [0., 0., 1.]])
    fn = lambda A: mat.guccione_energy(A, f, s, n, p)+mat.active_potential(A, f, 500.)
    stress = lambda A: mat.guccione_pk1(A, f, s, n, p)+mat.active_pk1(A, f, 500.)
    torch.testing.assert_close(fn(R@F), fn(F), atol=1e-9, rtol=1e-11)
    torch.testing.assert_close(stress(R@F), R@stress(F), atol=1e-8, rtol=1e-10)


def test_interpolate_before_cross_and_no_normalization(device):
    X, geo = patch(device)
    fields = varying_fields(X, geo)
    torch.testing.assert_close(fields.normal, torch.linalg.cross(fields.sheet, fields.fiber))
    # Nodal cross products are all -e_z, but the interpolated vectors need not
    # have unit norm. Interpolating the nodal normal would change the model.
    nodal_normal = X.new_tensor([0., 0., -1.]).expand_as(X)
    wrong_normal = interpolate_p2(nodal_normal, geo)
    assert (fields.normal-wrong_normal).abs().max() > 1e-5
    assert (torch.linalg.vector_norm(fields.fiber, dim=-1)-1).abs().max() > 1e-5
    qX = interpolate_p2(X, geo)
    torch.testing.assert_close(fields.tension, .4+.2*qX[..., 1].square(), atol=1e-14, rtol=1e-13)
    constants = prepare_reference_fields(geo, [2., 0., 0.], [0., 3., 0.], 7.)
    torch.testing.assert_close(constants.normal, X.new_tensor([0., 0., -6.]).expand_as(constants.normal))
    torch.testing.assert_close(constants.tension, torch.full_like(constants.tension, 7.))


def test_active_force_sign_and_linear_tension(device):
    X, geo = patch(device)
    p = mat.GuccioneParameters()
    make = lambda t: prepare_reference_fields(geo, [1., 0., 0.], [0., 1., 0.], t)
    passive = solid.guccione_force(X, geo, make(0.), p)
    force = solid.guccione_force(X, geo, make(1000.), p)
    torch.testing.assert_close(passive, torch.zeros_like(X), atol=1e-8, rtol=0)
    assert force.abs().max() > 1
    torch.testing.assert_close(solid.guccione_force(X, geo, make(2000.), p)-passive,
                               2*(force-passive), atol=1e-9, rtol=1e-11)
    # Positive tension must oppose an infinitesimal extension along the fiber.
    extension = torch.zeros_like(X)
    extension[:, 0] = X[:, 0]
    torch.testing.assert_close((force*extension).sum(), X.new_tensor(-500.), atol=1e-8, rtol=1e-11)


def test_nodal_force_and_tangent(device):
    X, geo = patch(device)
    fields = varying_fields(X, geo)
    p = mat.GuccioneParameters(C=2., kappa=30.)
    x = (X+.04*X.square()).requires_grad_()
    fn = lambda y: solid.guccione_energy(y, geo, fields, p)
    residual = torch.func.grad(fn)
    assert torch.autograd.gradcheck(fn, (x,), atol=1e-7, rtol=1e-5)
    force = solid.guccione_force(x, geo, fields, p)
    torch.testing.assert_close(force, -residual(x), atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(force.sum(0), X.new_zeros(3), atol=1e-11, rtol=0)
    torch.testing.assert_close(torch.linalg.cross(x, force).sum(0), X.new_zeros(3), atol=1e-11, rtol=0)
    direction = torch.sin(torch.arange(x.numel(), dtype=x.dtype, device=device)).reshape_as(x)
    for state in (X, x):
        _, Kv = torch.func.jvp(residual, (state,), (direction,))
        h = 1e-6
        fd = (residual(state+h*direction)-residual(state-h*direction))/(2*h)
        torch.testing.assert_close(Kv, fd, atol=3e-7, rtol=1e-5)


def test_invalid_reference_fields(device):
    X, geo = patch(device)
    for fiber, sheet in (([0., 0., 0.], [0., 1., 0.]), ([1., 0., 0.], [2., 0., 0.])):
        with pytest.raises(ValueError, match="nonzero and nonparallel"):
            prepare_reference_fields(geo, fiber, sheet)
    with pytest.raises(ValueError, match="finite"):
        prepare_reference_fields(geo, [1., 0., 0.], [0., 1., 0.], float("nan"))
    with pytest.raises(ValueError, match="direction"):
        prepare_reference_fields(geo, torch.zeros((2, 3), device=device), [0., 1., 0.])
    with pytest.raises(ValueError, match="tension"):
        prepare_reference_fields(geo, [1., 0., 0.], [0., 1., 0.], X)


def test_parameters():
    for args in ({"C": 0.}, {"bf": -1.}, {"kappa": float("nan")}, {"bfs": float("inf")}):
        with pytest.raises(ValueError):
            mat.GuccioneParameters(**args)
    # No silently selected implementation of afsi's optional deviatoric branch.
    with pytest.raises(TypeError):
        mat.GuccioneParameters(deviatoric=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_guccione_cpu_cuda_agreement():
    outputs = []
    for device in ("cpu", "cuda"):
        X, geo = patch(device)
        fields = varying_fields(X, geo)
        p = mat.GuccioneParameters()
        x = X+.04*X.square()
        fn = lambda y: solid.guccione_energy(y, geo, fields, p)
        _, Kv = torch.func.jvp(torch.func.grad(fn), (x,), (torch.sin(x),))
        outputs.append((fn(x).cpu(), solid.guccione_force(x, geo, fields, p).cpu(), Kv.cpu()))
    for cpu, gpu in zip(*outputs):
        torch.testing.assert_close(cpu, gpu, atol=1e-7, rtol=1e-9)
