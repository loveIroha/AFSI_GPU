"""Compare buffered/Triton multigrid against the existing torch discretization."""
from math import pi
import pytest
import torch
import torch.nn.functional as F
from afsi_torch.mac import MACGrid, MACFlow, GeometricMultigrid, MGOptions, divergence
from afsi_torch.mac.grid import negative_laplacian, zero_normal

DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))]


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('shape', [(6,10,14), (8,12,16)])
@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
def test_kernels_boundary_rows_restriction_and_prolongation(device, shape, dtype):
    grid = MACGrid(shape, (2.,3.,5.))
    mg = GeometricMultigrid(grid, device=device, dtype=dtype, backend='fused')
    kernels = mg.workspace.kernels
    coordinates = grid.coordinates(device=device,dtype=dtype)
    p = torch.sin(coordinates.sum(-1))
    rhs = torch.cos(1.3*coordinates[...,0]-.4*coordinates[...,2])
    original = p.clone()
    out = torch.full_like(p, float('nan'))
    tol = dict(rtol=2e-5, atol=2e-5) if dtype == torch.float32 else dict(rtol=2e-12, atol=2e-12)
    kernels.residual(p, rhs, out, grid.spacing)
    torch.testing.assert_close(out, rhs-negative_laplacian(p,grid.spacing), **tol)
    kernels.smooth(p, rhs, mg.diagonals[0], out, grid.spacing)
    torch.testing.assert_close(out, p+(2/3)*(rhs-negative_laplacian(p,grid.spacing))/mg.diagonals[0], **tol)
    torch.testing.assert_close(p, original, rtol=0, atol=0)
    coarse = p.new_empty(tuple(n//2 for n in shape))
    kernels.restrict(p, coarse)
    torch.testing.assert_close(coarse, F.avg_pool3d(p[None,None],2,2)[0,0], **tol)
    # Compare all corners, edges, faces and interior cells to PyTorch's exact
    # align_corners=False convention; the 105-cell coarse case tests tail masks.
    expected = original+F.interpolate(coarse[None,None],size=shape,
                                      mode='trilinear',align_corners=False)[0,0]
    kernels.prolong_add(coarse, p)
    torch.testing.assert_close(p, expected, **tol)
    constant = torch.full_like(p, 7.)
    kernels.residual(constant, torch.zeros_like(p), out, grid.spacing)
    torch.testing.assert_close(out, torch.zeros_like(p), atol=0, rtol=0)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('shape', [(4,4,4), (8,12,16), (16,16,16)])
def test_v_cycle_solve_and_reused_storage_preserve_reference(device, shape):
    grid = MACGrid(shape, (2.,3.,4.))
    options = MGOptions(smooth=3)  # odd count exercises ping-pong buffer swaps
    reference = GeometricMultigrid(grid, device=device, options=options)
    fused = GeometricMultigrid(grid, device=device, options=options, backend='fused')
    xyz = grid.coordinates(device=device)
    exact = torch.cos(pi*xyz[...,0]/2)*torch.cos(2*pi*xyz[...,1]/3)*torch.cos(pi*xyz[...,2]/4)
    exact -= exact.mean()
    rhs = negative_laplacian(exact,grid.spacing)
    start = .1*exact
    rhs_before, start_before = rhs.clone(), start.clone()
    with torch.no_grad():
        expected_cycle = reference._cycle(0,start,rhs)
        fused.workspace.initialize(start,rhs)
        actual_cycle = fused.workspace.cycle().clone()
    torch.testing.assert_close(actual_cycle,expected_cycle,rtol=2e-11,atol=2e-12)
    p_ref, _ = reference.solve(rhs,start)
    p, info = fused.solve(rhs,start)
    torch.testing.assert_close(p,p_ref,rtol=2e-8,atol=2e-9)
    torch.testing.assert_close(p,exact,rtol=2e-8,atol=2e-9)
    true_residual = torch.linalg.vector_norm(rhs-rhs.mean()-negative_laplacian(p,grid.spacing)).item()
    assert true_residual <= info['tolerance']
    assert info['backend'] == ('triton' if device == 'cuda' else 'buffered-cpu')
    buffers_before = sorted(t.data_ptr() for group in
        (fused.workspace.p, fused.workspace.ping, fused.workspace.rhs, fused.workspace.residuals) for t in group)
    p_before = p.clone()
    p2, _ = fused.solve(-rhs,p)
    torch.testing.assert_close(p2,-exact,rtol=2e-8,atol=2e-9)
    torch.testing.assert_close(p,p_before,rtol=0,atol=0)
    torch.testing.assert_close(start,start_before,rtol=0,atol=0)
    torch.testing.assert_close(rhs,rhs_before,rtol=0,atol=0)
    buffers_after = sorted(t.data_ptr() for group in
        (fused.workspace.p, fused.workspace.ping, fused.workspace.rhs, fused.workspace.residuals) for t in group)
    assert buffers_before == buffers_after
    zero, zero_info = fused.solve(torch.zeros_like(rhs))
    assert zero_info['cycles'] == 0 and zero.count_nonzero().item() == 0
    with pytest.raises(ValueError,match='incompatible'):
        fused.solve(torch.ones_like(rhs))
    with pytest.raises(ValueError,match='invalid pressure RHS'):
        fused.solve(torch.full_like(rhs,float('nan')))


@pytest.mark.parametrize('device', DEVICES)
def test_fused_projection_has_small_divergence_and_matches_reference(device):
    grid = MACGrid((16,16,16),(2.,3.,4.))
    reference = MACFlow(grid,dt=1e-4,device=device)
    fused = MACFlow(grid,dt=1e-4,device=device,pressure_backend='fused')
    velocity = zero_normal(tuple(torch.sin((c+1)*grid.coordinates(c,device=device).sum(-1)) for c in range(3)))
    a, b = reference.project(velocity), fused.project(velocity)
    torch.testing.assert_close(a.pressure,b.pressure,rtol=2e-8,atol=2e-8)
    for u,v in zip(a.velocity,b.velocity):
        torch.testing.assert_close(u,v,rtol=2e-8,atol=2e-9)
    assert divergence(b.velocity,grid.spacing).abs().max().item() < 1e-8


@pytest.mark.parametrize('device', DEVICES)
def test_fused_lv_checkpoint_resumes_with_backend_and_warm_start(tmp_path,device):
    pytest.importorskip('gmsh')
    from examples.lv_mac import run
    from afsi_torch.mac.checkpoint import load_mac
    common = dict(device=device,mesh_size=.4,fluid_cells=16,log_every=2,
                  checkpoint_every=2,warm_start=True)
    run(**common,output=tmp_path/'ref',end_time=.0003)
    run(**common,pressure_backend='fused',output=tmp_path/'fused',end_time=.00015)
    resumed = run(device=device,resume=tmp_path/'fused'/'checkpoint.npz',end_time=.0003,log_every=2)
    assert resumed['settings']['pressure_backend'] == 'fused'
    assert resumed['settings']['warm_start'] is True
    _, a, _, _ = load_mac(tmp_path/'ref'/'checkpoint.npz',device)
    _, b, _, _ = load_mac(tmp_path/'fused'/'checkpoint.npz',device)
    for name in ('x','force','pressure'):
        torch.testing.assert_close(getattr(a,name),getattr(b,name),rtol=1e-7,atol=1e-8)
    for u,v in zip(a.velocity,b.velocity):
        torch.testing.assert_close(u,v,rtol=1e-7,atol=1e-9)
