"""Check diagnostic alternatives independently of the production trajectory."""
import pytest
import torch
from afsi_torch import ib
from afsi_torch.fluid import create_box, prepare_operators, ChorinSolver
from validation.diagnose_ib import scaled_stencil, BoxMassInverse, discrete_projection


@pytest.fixture(params=['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA device unavailable'))])
def device(request):
    return request.param


@pytest.mark.parametrize('m', [1,2,3])
def test_scaled_kernel_moments_and_full_grid_sum(device, m):
    grid = ib.UniformGrid((19,20,21), (.2,.3,.4), (-1.,-2.,-3.))
    x = torch.tensor([[1.12,.77,.54],[.91,1.03,1.39]], device=device, dtype=torch.float64)
    stencil = scaled_stencil(x, grid, m)
    X = grid.coordinates(device=device)
    # Independent all-node sum, no sparse neighbor indexing.
    weights = (ib.peskin4((x[:,None]-X[None])/(x.new_tensor(grid.spacing)*m))/m).prod(-1)
    torch.testing.assert_close(stencil.weights.sum(-1), torch.ones_like(x[:,0]), atol=2e-14, rtol=0)
    torch.testing.assert_close(ib.interpolate(X,stencil), x, atol=2e-14, rtol=0)
    velocity = torch.sin(X)*torch.cos(2*X)
    torch.testing.assert_close(ib.interpolate(velocity,stencil), weights@velocity, atol=2e-14, rtol=1e-12)
    g = x.new_tensor([[1.,-2.,3.],[-4.,5.,-6.]])
    dual = ib.spread_load(g,stencil)
    torch.testing.assert_close(dual,weights.T@g,atol=2e-14,rtol=1e-12)
    torch.testing.assert_close((dual*velocity).sum(),(g*ib.interpolate(velocity,stencil)).sum())


def test_scaled_kernel_rejects_noninteger_and_truncated_support():
    grid = ib.UniformGrid((11,11,11), (1.,)*3)
    x = torch.tensor([[1.1,4.,4.]], dtype=torch.float64)
    with pytest.raises(ValueError,match='integer'):
        scaled_stencil(x,grid,1.5)
    with pytest.raises(ValueError,match='support'):
        scaled_stencil(x,grid,2)


def test_tensor_mass_inverse_anisotropic(device):
    mesh = create_box((2,3,2),(1.3,2.1,.8),device=device)
    op = prepare_operators(mesh)
    inverse = BoxMassInverse(mesh)
    u = torch.sin(mesh.velocity_coordinates*3.7).masked_fill(mesh.velocity_boundary[:,None],0)
    torch.testing.assert_close(inverse(op.velocity_mass(u)),u,atol=2e-13,rtol=1e-12)


def test_schur_projection_against_dense_constraint_solve(device):
    mesh = create_box((2,2,2),(1.3,1.7,2.1),device=device)
    op = prepare_operators(mesh)
    flow = ChorinSolver(op,dt=1e-4)
    X = mesh.velocity_coordinates
    star = (torch.sin(X*3.1)+torch.cos(X*1.7)).masked_fill(flow.velocity_fixed,0)
    projected, info = discrete_projection(flow,star,BoxMassInverse(mesh))
    # Independently assemble free M and D from quadrature actions, then dense
    # saddle constraint elimination on this tiny patch only.
    free = (~flow.velocity_fixed).reshape(-1).nonzero().flatten()
    basis = torch.eye(X.numel(),device=device,dtype=X.dtype)[:,free]
    M = torch.stack([op.velocity_mass(basis[:,i].reshape_as(X)).reshape(-1)[free]
                     for i in range(len(free))],1)
    D = torch.stack([op.divergence(basis[:,i].reshape_as(X)) for i in range(len(free))],1)[1:]
    MinvDT = torch.linalg.solve(M,D.T)
    target = star.reshape(-1)[free]
    reference = target-MinvDT@torch.linalg.solve(D@MinvDT,D@target)
    torch.testing.assert_close(projected.reshape(-1)[free],reference,atol=2e-10,rtol=1e-9)
    assert info['full_divergence_dual_norm'] < 1e-10
    assert (projected[flow.velocity_fixed]==0).all()
    again,_ = discrete_projection(flow,projected,BoxMassInverse(mesh))
    torch.testing.assert_close(again,projected,atol=2e-10,rtol=1e-9)
    correction = star-projected
    assert abs((projected*op.velocity_mass(correction)).sum().item()) < 1e-10


def test_projection_removes_discrete_gradient(device):
    mesh = create_box((2,3,2),(1.1,1.7,2.3),device=device)
    op = prepare_operators(mesh)
    flow = ChorinSolver(op,dt=1e-4)
    inverse = BoxMassInverse(mesh)
    p = torch.sin(mesh.pressure_coordinates[:,0]*2)+mesh.pressure_coordinates[:,2]**2
    gradient = inverse(op.divergence_transpose(p))
    projected,_ = discrete_projection(flow,gradient,inverse)
    assert torch.linalg.vector_norm(projected)/torch.linalg.vector_norm(gradient) < 1e-10
