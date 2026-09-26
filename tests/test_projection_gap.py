"""Independent small-box checks for the projection attribution."""
import pytest
import torch

from afsi_torch.fluid import ChorinSolver, create_box, prepare_operators
from validation.diagnose_ib import BoxMassInverse, discrete_projection
from validation.diagnose_projection_gap import reconstruct_laplacian_projection, run


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA device unavailable'))])
def test_laplacian_reconstruction_matches_chorin_correction(device):
    mesh = create_box((2, 2, 2), (2., 2., 2.), (-1., -1., -1.), device=device)
    flow = ChorinSolver(prepare_operators(mesh), dt=5e-5)
    x = mesh.velocity_coordinates
    density = torch.sin(1.7*x) + .4*torch.cos(2.3*x)
    result = flow.step(torch.zeros_like(x), density=density)
    inverse = BoxMassInverse(mesh)
    reconstructed, info = reconstruct_laplacian_projection(flow, result, inverse)
    torch.testing.assert_close(reconstructed, result.velocity, atol=5e-12, rtol=2e-8)
    assert info['free_gradient_plus_divergence_transpose_relative'] < 1e-12
    assert info['laplacian_multiplier_residual_relative'] < 1e-8
    schur, _ = discrete_projection(flow, result.tentative_velocity, inverse)
    assert torch.linalg.vector_norm(flow.op.divergence(schur)) < 1e-10


def test_projection_study_rejects_incompatible_kernel_before_loading_preload(tmp_path):
    with pytest.raises(ValueError, match='integer dilation'):
        run(preload=tmp_path / 'missing', output=tmp_path / 'report.json', epsilon=.75)
    assert not (tmp_path / 'report.json').exists()
