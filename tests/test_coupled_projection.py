"""The experimental Schur branch changes only the corrected fluid velocity."""
import numpy as np
import pytest
import torch

from afsi_torch.fluid import ChorinSolver, create_box, prepare_operators
from validation.compare_coupled_projection import ReferenceSchurFlow, _difference, run
from validation.diagnose_ib import BoxMassInverse, discrete_projection


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA device unavailable'))])
def test_schur_wrapper_uses_same_tentative_step_and_reference_projection(device):
    mesh = create_box((2, 2, 2), (2., 2., 2.), (-1., -1., -1.), device=device)
    chorin = ChorinSolver(prepare_operators(mesh), dt=5e-5)
    schur = ReferenceSchurFlow(chorin)
    velocity = torch.zeros_like(mesh.velocity_coordinates)
    density = torch.sin(1.7*mesh.velocity_coordinates) + .4*torch.cos(2.3*mesh.velocity_coordinates)
    baseline = chorin.step(velocity, density=density)
    wrapped = schur.step(velocity, density=density)
    expected, info = discrete_projection(chorin, baseline.tentative_velocity,
                                         BoxMassInverse(mesh), options=schur.options)
    # GPU index_add/einsum accumulation can vary by a few float64 ulps across
    # repeated solves; the two tentative fields must agree numerically.
    torch.testing.assert_close(wrapped.tentative_velocity, baseline.tentative_velocity,
                               atol=1e-18, rtol=1e-12)
    torch.testing.assert_close(wrapped.velocity, expected, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(wrapped.pressure, baseline.pressure, atol=1e-12, rtol=1e-10)
    assert wrapped.diagnostics['corrected_divergence_dual_norm'] == pytest.approx(
        info['full_divergence_dual_norm'], abs=1e-12)
    assert wrapped.diagnostics['corrected_divergence_dual_norm'] < 1e-10


def test_projection_comparison_uses_displacements_from_same_preload(tmp_path):
    reference = dict(X=np.zeros((2, 3)), x_preload=np.zeros((2, 3)),
                     cells=np.zeros((1, 4), dtype=np.int64))
    paths = [tmp_path / f'{i}.npz' for i in range(2)]
    np.savez(paths[0], **reference, x=np.array([[1., 0., 0.], [0., 0., 0.]]))
    np.savez(paths[1], **reference, x=np.array([[2., 0., 0.], [0., 0., 0.]]))
    def case(path):
        return dict(time_s=.001, reference_sha256='same', completed=True,
                    snapshot=str(path), summary=dict(delta_cavity_ml=1. if path == paths[0] else 2.))
    result = _difference(case(paths[0]), case(paths[1]))
    assert result['displacement_relative_to_first'] == pytest.approx(1.)
    assert result['displacement_relative_to_second'] == pytest.approx(.5)
    assert result['delta_cavity_relative_to_first'] == pytest.approx(1.)


def test_coupled_projection_rejects_noninteger_kernel_before_preload(tmp_path):
    with pytest.raises(ValueError, match='integer dilation'):
        run(preload=tmp_path / 'missing', output=tmp_path / 'unused', epsilon=.75)
    assert not (tmp_path / 'unused').exists()
