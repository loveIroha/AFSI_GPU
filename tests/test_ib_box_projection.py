"""Checks for the controlled fluid-box diagnostic setup and comparison."""
import numpy as np
import pytest
import torch

from afsi_torch.fluid import create_box
from validation.compare_ib_box_projection import box_spec, response_comparisons, run
from validation.diagnose_ib import scaled_stencil


def test_box_expansion_preserves_physical_ib_support():
    point = torch.tensor([[.37, -.41, -1.23]], dtype=torch.float64)
    support = None
    for n in (12, 18, 24):
        mesh = create_box(*box_spec(n))
        assert mesh.velocity_grid.spacing == (.5, .5, .5)
        stencil = scaled_stencil(point, mesh.velocity_grid, 2)
        current = (mesh.velocity_coordinates[stencil.indices].numpy(), stencil.weights.numpy())
        if support is not None:
            np.testing.assert_array_equal(current[0], support[0])
            np.testing.assert_allclose(current[1], support[1], rtol=0, atol=1e-15)
        support = current


def test_pairwise_box_difference_keeps_reference_denominator_explicit():
    responses = {12: np.array([[1., 2., 0.]]), 18: np.array([[2., 2., 0.]]),
                 24: np.array([[2., 3., 0.]])}
    pairs = response_comparisons(responses)
    assert set(pairs) == {'12_to_18', '12_to_24', '18_to_24'}
    assert pairs['12_to_18']['absolute_nodal_l2_cm_per_s'] == pytest.approx(1.)
    assert pairs['12_to_18']['relative_to_small'] == pytest.approx(1 / np.sqrt(5))
    assert pairs['12_to_18']['relative_to_large'] == pytest.approx(1 / np.sqrt(8))


def test_box_study_rejects_changed_kernel_width_before_loading_preload(tmp_path):
    with pytest.raises(ValueError, match='integer dilation'):
        run(preload=tmp_path / 'missing', output=tmp_path / 'report.json', epsilon=.75)
    assert not (tmp_path / 'report.json').exists()
