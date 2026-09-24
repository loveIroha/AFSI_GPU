"""Kernel moments, conservation, duality and CPU/CUDA IB transfer tests."""
import pytest
import torch
from afsi_torch import ib, solid
from afsi_torch.tetrahedron import reference_nodes


@pytest.fixture(params=['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA device unavailable'))])
def device(request):
    return request.param


def setup(device):
    grid = ib.UniformGrid((11, 10, 9), (.2, .3, .4), (-.7, 1.1, -.3))
    indices = torch.tensor([[2.17, 3.41, 2.53], [5.32, 4.29, 3.78],
                            [6., 2., 5.], [2.17, 3.41, 2.53]], dtype=torch.float64, device=device)
    x = indices*indices.new_tensor(grid.spacing)+indices.new_tensor(grid.origin)
    return grid, x, ib.prepare_stencil(x, grid)


def test_peskin_moments_and_support(device):
    s = torch.linspace(0, 1, 51, dtype=torch.float64, device=device)[:, None]
    nodes = torch.arange(-2, 4, dtype=s.dtype, device=device)[None, :]
    weights = ib.peskin4(s-nodes)
    torch.testing.assert_close(weights.sum(1), torch.ones(51, device=device, dtype=s.dtype), atol=2e-15, rtol=0)
    torch.testing.assert_close((weights*(nodes-s)).sum(1), torch.zeros(51, device=device, dtype=s.dtype), atol=2e-15, rtol=0)
    torch.testing.assert_close(weights.square().sum(1), s.new_full((51,), 3/8), atol=2e-15, rtol=0)
    r = s.new_tensor([-3., -2., -1., 0., 1., 2., 3.])
    torch.testing.assert_close(ib.peskin4(r), s.new_tensor([0., 0., .25, .5, .25, 0., 0.]), atol=1e-15, rtol=0)
    assert (weights >= 0).all()


def test_constant_affine_velocity_and_grid_order(device):
    grid, x, stencil = setup(device)
    X = grid.coordinates(device=device)
    torch.testing.assert_close(X[1]-X[0], X.new_tensor([.2, 0., 0.]))
    torch.testing.assert_close(X[grid.shape[0]]-X[0], X.new_tensor([0., .3, 0.]))
    A = X.new_tensor([[1., .3, -.2], [.7, -.8, .2], [0., .1, .5]])
    b = X.new_tensor([.2, -.4, .7])
    torch.testing.assert_close(ib.interpolate(b.expand_as(X), stencil), b.expand_as(x), atol=1e-14, rtol=1e-13)
    torch.testing.assert_close(ib.interpolate(X@A.T+b, stencil), x@A.T+b, atol=1e-14, rtol=1e-13)


def test_force_torque_and_power_conservation(device):
    grid, x, stencil = setup(device)
    X = grid.coordinates(device=device)
    u = torch.sin(torch.arange(X.numel(), dtype=X.dtype, device=device)).reshape_as(X)
    g = torch.cos(torch.arange(x.numel(), dtype=x.dtype, device=device)).reshape_as(x)
    f = ib.spread_density(g, stencil)
    torch.testing.assert_close(f.sum(0)*grid.cell_volume, g.sum(0), atol=1e-13, rtol=1e-12)
    torch.testing.assert_close(torch.linalg.cross(X, f).sum(0)*grid.cell_volume,
                               torch.linalg.cross(x, g).sum(0), atol=1e-13, rtol=1e-12)
    torch.testing.assert_close((u*f).sum()*grid.cell_volume, (ib.interpolate(u, stencil)*g).sum(), atol=1e-13, rtol=1e-12)
    load = ib.spread_load(g, stencil)
    torch.testing.assert_close(load, f*grid.cell_volume, atol=1e-14, rtol=1e-13)


def test_transpose_is_velocity_autograd(device):
    grid, x, stencil = setup(device)
    u = grid.coordinates(device=device).sin()
    g = x.cos()
    adjoint = torch.func.grad(lambda v: (ib.interpolate(v, stencil)*g).sum())(u)
    torch.testing.assert_close(adjoint, ib.spread_load(g, stencil), atol=1e-14, rtol=1e-12)
    reverse = torch.func.grad(lambda f: (ib.spread_density(f, stencil)*u).sum()*grid.cell_volume)(g)
    torch.testing.assert_close(reverse, ib.interpolate(u, stencil), atol=1e-14, rtol=1e-12)


def test_origin_translation_and_rebuild(device):
    grid, x, stencil = setup(device)
    shift = x.new_tensor([10., -2., 4.])
    moved = ib.UniformGrid(grid.shape, grid.spacing, tuple(a+b for a, b in zip(grid.origin, [10., -2., 4.])))
    translated = ib.prepare_stencil(x+shift, moved)
    # Exact lattice nodes can select either zero-weight support endpoint after
    # rounding; compare the resulting operator, not the stored integer indices.
    u = grid.coordinates(device=device).cos()
    torch.testing.assert_close(ib.interpolate(u, stencil), ib.interpolate(u, translated), atol=2e-14, rtol=1e-12)
    moved_x = x+x.new_tensor([.23, .02, -.03])
    rebuilt = ib.prepare_stencil(moved_x, grid)
    torch.testing.assert_close(ib.interpolate(grid.coordinates(device=device), rebuilt), moved_x, atol=1e-14, rtol=1e-12)


def test_position_derivatives_inside_stencil(device):
    grid, x, _ = setup(device)
    x = x[:2].clone().requires_grad_()
    u = grid.coordinates(device=device).sin()
    assert torch.autograd.gradcheck(lambda y: ib.interpolate(u, ib.prepare_stencil(y, grid)),
                                    (x,), eps=1e-6, atol=1e-7, rtol=1e-5)
    # All inactive sqrt branches must also have finite backward derivatives.
    r = x.new_tensor([-3., -2., -1., 0., 1., 2., 3.], requires_grad=True)
    derivative = torch.autograd.grad(ib.peskin4(r).sum(), r)[0]
    assert torch.isfinite(derivative).all()


def test_solid_energy_rate_through_ib(device):
    X = reference_nodes(device=device)
    cells = torch.arange(10, device=device).reshape(1, 10)
    geometry = solid.prepare_p2(X, cells)
    x = X+.04*X.square()
    grid = ib.UniformGrid((13, 13, 13), (.2, .2, .2), (-.6, -.6, -.6))
    stencil = ib.prepare_stencil(x, grid)
    u = .1*grid.coordinates(device=device).sin()
    velocity = ib.interpolate(u, stencil)
    force = solid.stress_force(x, geometry, 2., 10.)
    _, energy_rate = torch.func.jvp(lambda y: solid.energy(y, geometry, 2., 10.), (x,), (velocity,))
    fluid_power = (ib.spread_density(force, stencil)*u).sum()*grid.cell_volume
    torch.testing.assert_close(fluid_power, -energy_rate, atol=1e-12, rtol=1e-11)


@pytest.mark.parametrize('mode', ['near_wall', 'outside', 'nan', 'shape', 'dtype'])
def test_invalid_positions_rejected(mode):
    grid, x, _ = setup('cpu')
    if mode == 'near_wall':
        x[0] = x.new_tensor(grid.origin)+.1*x.new_tensor(grid.spacing)
    elif mode == 'outside':
        x[0, 0] = 1e100
    elif mode == 'nan':
        x[0, 0] = float('nan')
    elif mode == 'shape':
        x = x[:, :2]
    else:
        x = x.to(torch.int64)
    with pytest.raises(ValueError):
        ib.prepare_stencil(x, grid)


def test_grid_and_field_validation():
    for kwargs in [dict(shape=(3, 5, 5), spacing=(1., 1., 1.)),
                   dict(shape=(5, 5, 5), spacing=(0., 1., 1.)),
                   dict(shape=(5, 5, 5), spacing=(1., 1., 1.), origin=(float('nan'), 0., 0.))]:
        with pytest.raises(ValueError):
            ib.UniformGrid(**kwargs)
    grid, x, stencil = setup('cpu')
    with pytest.raises(ValueError):
        ib.interpolate(torch.zeros(grid.node_count, 3), stencil)
    with pytest.raises(ValueError):
        ib.spread_load(x[:, :2], stencil)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device unavailable')
def test_cpu_gpu_transfer_parity():
    grid, x, cpu = setup('cpu')
    gpu = ib.prepare_stencil(x.cuda(), grid)
    u, g = grid.coordinates().sin(), x.cos()
    torch.testing.assert_close(ib.interpolate(u, cpu), ib.interpolate(u.cuda(), gpu).cpu(), atol=1e-13, rtol=1e-12)
    torch.testing.assert_close(ib.spread_density(g, cpu), ib.spread_density(g.cuda(), gpu).cpu(), atol=1e-12, rtol=1e-11)
