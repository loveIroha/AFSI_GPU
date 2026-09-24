"""Q2/Q1 polynomial, analytic matrix, boundary, nonlinear and device checks."""
import numpy as np
import pytest
import torch
from afsi_torch.fluid import create_box, prepare_operators
from afsi_torch.fluid.elements import basis, reference_nodes
from afsi_torch import ib


@pytest.fixture(params=['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA device unavailable'))])
def device(request):
    return request.param


def setup(device, counts=(2, 2, 2)):
    mesh = create_box(counts, (1.4, 1.1, .9), (-.3, .2, -.5), device=device)
    return mesh, prepare_operators(mesh)


def field(X):
    return torch.sin(torch.arange(X.numel(), device=X.device, dtype=X.dtype)*.7).reshape_as(X)


@pytest.mark.parametrize('degree', [1, 2])
def test_basis_kronecker_partition_and_polynomial(device, degree):
    nodes = reference_nodes(degree, device=device)
    values, _ = basis(nodes, degree)
    torch.testing.assert_close(values, torch.eye(len(nodes), device=device, dtype=nodes.dtype))
    q = nodes.new_tensor([[.13, .27, .72], [.8, .3, .1]])
    N, dN = basis(q, degree)
    f = nodes.prod(-1)**degree
    torch.testing.assert_close(N@f, q.prod(-1)**degree, atol=2e-15, rtol=1e-12)
    expected = degree*(q.prod(-1)**degree)[:, None]/q
    torch.testing.assert_close(torch.einsum('qaj,a->qj', dN, f), expected, atol=2e-15, rtol=1e-12)
    torch.testing.assert_close(N.sum(-1), torch.ones_like(q[:, 0]))
    torch.testing.assert_close(dN.sum(1), torch.zeros_like(q), atol=3e-15, rtol=0)


def test_lattices_shared_dofs_and_ib_order(device):
    mesh, _ = setup(device)
    torch.testing.assert_close(mesh.velocity_coordinates, mesh.velocity_grid.coordinates(device=device))
    assert len(torch.unique(mesh.velocity_cells)) == len(mesh.velocity_coordinates)
    assert len(torch.unique(mesh.pressure_cells)) == len(mesh.pressure_coordinates)
    assert int(mesh.velocity_boundary.sum()) == 125-27
    assert int(mesh.pressure_boundary.sum()) == 27-1
    torch.testing.assert_close(mesh.velocity_coordinates[mesh.velocity_cells[0]],
        reference_nodes(2, device=device)*mesh.velocity_coordinates.new_tensor(mesh.cell_sizes)+
        mesh.velocity_coordinates.new_tensor(mesh.origin), atol=1e-15, rtol=1e-14)


@pytest.mark.parametrize('degree', [1, 2])
def test_exact_tensor_mass_stiffness_assembly(device, degree):
    mesh, op = setup(device, (2, 1, 2))
    if degree == 1:
        M = np.array([[2., 1], [1, 2]])/6
        K = np.array([[1., -1], [-1, 1]])
        conn, X = mesh.pressure_cells, mesh.pressure_coordinates
        u = field(X)[:, 0]
        actual_M, actual_K = op.pressure_mass(u), op.pressure_stiffness(u)
    else:
        M = np.array([[4., 2, -1], [2, 16, 2], [-1, 2, 4]])/30
        K = np.array([[7., -8, 1], [-8, 16, -8], [1, -8, 7]])/3
        conn, X = mesh.velocity_cells, mesh.velocity_coordinates
        u = field(X)
        actual_M, actual_K = op.velocity_mass(u), op.velocity_stiffness(u)
    hx, hy, hz = mesh.cell_sizes
    local_M = np.kron(np.kron(M, M), M)*hx*hy*hz
    local_K = (np.kron(np.kron(M, M), K)*hy*hz/hx +
               np.kron(np.kron(M, K), M)*hx*hz/hy +
               np.kron(np.kron(K, M), M)*hx*hy/hz)
    matrices = [np.zeros((len(X), len(X))) for _ in range(2)]
    for cell in conn.cpu().numpy():
        for global_, local in zip(matrices, (local_M, local_K)):
            global_[np.ix_(cell, cell)] += local
    for actual, matrix in zip((actual_M, actual_K), matrices):
        expected = torch.as_tensor(matrix@u.cpu().numpy(), device=device)
        torch.testing.assert_close(actual, expected, atol=3e-14, rtol=1e-12)
        assert (u*actual).sum() > 0


def test_affine_fields_nullspaces_and_manufactured_convection(device):
    mesh, op = setup(device)
    X, P = mesh.velocity_coordinates, mesh.pressure_coordinates
    A = X.new_tensor([[.4, .7, -.2], [.3, -.6, .1], [0., .5, .9]])
    u = X@A.T+X.new_tensor([.3, .1, -.2])
    p = 1+P@P.new_tensor([.2, -.4, .6])
    torch.testing.assert_close(op.divergence(u), op.pressure_mass(torch.ones_like(p))*A.trace(), atol=2e-15, rtol=1e-12)
    torch.testing.assert_close(op.gradient(p), op.velocity_mass(X.new_tensor([.2, -.4, .6]).expand_as(X)), atol=3e-15, rtol=1e-12)
    torch.testing.assert_close(op.convection(u), op.velocity_mass(u@A.T), atol=3e-15, rtol=1e-12)
    for result in (op.velocity_stiffness(torch.ones_like(X)), op.gradient(torch.ones_like(p)),
                   op.pressure_stiffness(torch.ones_like(p)), op.convection(torch.ones_like(X))):
        torch.testing.assert_close(result, torch.zeros_like(result), atol=1e-14, rtol=0)
    volume = np.prod(mesh.lengths)
    torch.testing.assert_close(op.velocity_mass(torch.ones_like(X)).sum(0), X.new_full((3,), volume))
    torch.testing.assert_close(op.pressure_mass(torch.ones_like(p)).sum(), P.new_tensor(volume))


def test_mixed_transpose_and_boundary_integration_identity(device):
    mesh, op = setup(device)
    u, p = field(mesh.velocity_coordinates), field(mesh.pressure_coordinates)[:, 0]
    torch.testing.assert_close((p*op.divergence(u)).sum(), (u*op.divergence_transpose(p)).sum(), atol=2e-15, rtol=1e-12)
    boundary_load = op.gradient(p)+op.divergence_transpose(p)
    torch.testing.assert_close(boundary_load[~mesh.velocity_boundary],
        torch.zeros_like(boundary_load[~mesh.velocity_boundary]), atol=2e-15, rtol=0)
    assert boundary_load[mesh.velocity_boundary].abs().max() > 1e-3
    # Independent exact divergence theorem: p=1, u=x, net flux=3*volume.
    one = torch.ones(len(p), device=device, dtype=p.dtype)
    flux = (mesh.velocity_coordinates*(op.gradient(one)+op.divergence_transpose(one))).sum()
    torch.testing.assert_close(flux, p.new_tensor(3*np.prod(mesh.lengths)), atol=1e-14, rtol=1e-12)


def test_nonlinear_quadrature_and_jvp(device):
    mesh, op = setup(device)
    u = field(mesh.velocity_coordinates)
    direction = torch.cos(u*2)
    over = prepare_operators(mesh, quadrature_order=5)
    torch.testing.assert_close(op.convection(u), over.convection(u), atol=2e-14, rtol=1e-12)
    _, tangent = torch.func.jvp(op.convection, (u,), (direction,))
    expected = op.convection(u, direction)+op.convection(direction, u)
    torch.testing.assert_close(tangent, expected, atol=2e-14, rtol=1e-12)
    fd = (op.convection(u+1e-5*direction)-op.convection(u-1e-5*direction))/2e-5
    torch.testing.assert_close(tangent, fd, atol=2e-10, rtol=2e-8)
    grad = torch.func.grad(lambda v: .5*(v*op.velocity_mass(v)).sum())(u)
    torch.testing.assert_close(grad, op.velocity_mass(u), atol=2e-14, rtol=1e-12)


def test_ib_density_load_is_not_dual_spread(device):
    mesh, op = setup(device, (3, 3, 3))
    grid = mesh.velocity_grid
    X = mesh.velocity_coordinates
    x = X.new_tensor([[2.2, 2.4, 2.7]])*X.new_tensor(grid.spacing)+X.new_tensor(grid.origin)
    stencil = ib.prepare_stencil(x, grid)
    g = X.new_tensor([[2., -1., .3]])
    dual = ib.spread_load(g, stencil)
    density = ib.spread_density(g, stencil)
    weak = op.density_load(density)
    u = field(X)
    torch.testing.assert_close((u*dual).sum(), (ib.interpolate(u, stencil)*g).sum())
    torch.testing.assert_close(dual, density*grid.cell_volume)
    assert not torch.allclose(weak, dual, atol=1e-7, rtol=1e-7)
    # Consistent FE work is an integral, not a uniform-grid dot product.
    uq = torch.einsum('qa,eac->eqc', op.N, u[mesh.velocity_cells])
    fq = torch.einsum('qa,eac->eqc', op.N, density[mesh.velocity_cells])
    torch.testing.assert_close((u*weak).sum(), torch.einsum('eqc,eqc,q->', uq, fq, op.weights))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device unavailable')
def test_all_operator_cpu_cuda_parity():
    outputs = []
    for device in ('cpu', 'cuda'):
        mesh, op = setup(device)
        u, p = field(mesh.velocity_coordinates), field(mesh.pressure_coordinates)[:, 0]
        outputs.append([f(a).cpu() for f, a in [(op.velocity_mass, u), (op.velocity_stiffness, u),
            (op.pressure_mass, p), (op.pressure_stiffness, p), (op.gradient, p),
            (op.divergence, u), (op.divergence_transpose, p), (op.convection, u)]])
    for cpu, gpu in zip(*outputs):
        torch.testing.assert_close(cpu, gpu, atol=3e-14, rtol=1e-11)


@pytest.mark.parametrize('kwargs', [dict(counts=(0, 1, 1)), dict(counts=(1.5, 2, 2)),
    dict(lengths=(1., -1., 1.)), dict(origin=(0., float('nan'), 0.))])
def test_invalid_box(kwargs):
    with pytest.raises(ValueError):
        create_box(**kwargs)


def test_operator_shape_and_underintegration_rejected():
    mesh, op = setup('cpu')
    with pytest.raises(ValueError):
        op.divergence(torch.zeros(4, 3))
    with pytest.raises(ValueError):
        prepare_operators(mesh, quadrature_order=3)
