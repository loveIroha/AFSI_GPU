import pytest
import torch

from afsi_torch.mechanics import energy, reference_geometry, stress_force, validate_deformation


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device unavailable"))])
def mesh(request):
    X = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                      [0., 0., 1.], [1., 1., 1.]], dtype=torch.float64, device=request.param)
    cells = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], device=request.param)
    gradients, volumes = reference_geometry(X, cells)
    return X, cells, gradients, volumes


def test_reference_and_rigid_motion(mesh):
    X, cells, gradients, volumes = mesh
    rotation = X.new_tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    for x in (X, X @ rotation.T + X.new_tensor([0.2, -0.4, 0.7])):
        validate_deformation(x, cells, gradients)
        torch.testing.assert_close(energy(x, cells, gradients, volumes, 2., 5.), X.new_tensor(0.), atol=1e-12, rtol=0)
        torch.testing.assert_close(stress_force(x, cells, gradients, volumes, 2., 5.), torch.zeros_like(X), atol=1e-12, rtol=0)


def test_affine_energy_and_assembled_force(mesh):
    X, cells, gradients, volumes = mesh
    F = X.new_tensor([[1.2, 0.1, 0.], [0., 0.9, 0.05], [0., 0., 1.1]])
    x = (X @ F.T).requires_grad_()
    E = energy(x, cells, gradients, volumes, 2., 5.)
    # The two tetrahedra have volumes 1/6 and 1/3; J=1.2*0.9*1.1.
    logJ = X.new_tensor(1.188).log()
    expected = 0.5 * ((1.2**2 + 0.1**2 + 0.9**2 + 0.05**2 + 1.1**2 - 3) - 2*logJ + 2.5*logJ**2)
    torch.testing.assert_close(E, expected, atol=1e-12, rtol=1e-12)
    force = stress_force(x, cells, gradients, volumes, 2., 5.)
    torch.testing.assert_close(force, -torch.autograd.grad(E, x)[0], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(force.sum(0), X.new_zeros(3), atol=1e-12, rtol=0)
    torque = torch.linalg.cross(x, force).sum(0)
    torch.testing.assert_close(torque, X.new_zeros(3), atol=1e-12, rtol=0)


def test_finite_difference_and_tangent_action(mesh):
    X, cells, gradients, volumes = mesh
    x = X.clone()
    x[4] += X.new_tensor([0.12, -0.04, 0.07])  # Nonuniform deformation.
    fn = lambda y: energy(y, cells, gradients, volumes, 2., 5.)
    assert torch.autograd.gradcheck(fn, (x.requires_grad_(),), eps=1e-6, atol=1e-7, rtol=1e-5)
    residual = torch.func.grad(fn)
    direction = torch.arange(x.numel(), dtype=x.dtype, device=x.device).reshape_as(x) / 20
    _, tangent_action = torch.func.jvp(residual, (x,), (direction,))
    eps = 1e-6
    fd = (residual(x + eps * direction) - residual(x - eps * direction)) / (2 * eps)
    torch.testing.assert_close(tangent_action, fd, atol=1e-7, rtol=1e-5)


def test_invalid_geometry_and_inversion(mesh):
    X, cells, gradients, _ = mesh
    with pytest.raises(ValueError, match="nondegenerate"):
        reference_geometry(torch.zeros_like(X), cells)
    x = X.clone()
    x[:, 0] *= -1
    with pytest.raises(ValueError, match="det"):
        validate_deformation(x, cells, gradients)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cpu_cuda_agreement():
    X = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=torch.float64)
    cells = torch.tensor([[0, 1, 2, 3]])
    results = []
    for device in ("cpu", "cuda"):
        ref, conn = X.to(device), cells.to(device)
        gradients, volumes = reference_geometry(ref, conn)
        x = ref * ref.new_tensor([1.2, 0.9, 1.1])
        results.append(stress_force(x, conn, gradients, volumes, 2., 5.).cpu())
    torch.testing.assert_close(*results, atol=1e-11, rtol=1e-10)
