"""Independent regularized IB force integral against Q2 basis functions."""
import numpy as np
import pytest
import torch

from afsi_torch import ib
from afsi_torch.fluid import create_box, prepare_operators
from validation.diagnose_ib import scaled_stencil
from validation.verify_ib_weak import axis_weak_load, reference_weak_load, run


@pytest.mark.parametrize('n', [6, 12, 18])
def test_quadrature_reproduces_resultant_and_first_moment(n):
    origin, length, point = -6., 12., .37
    b = axis_weak_load(n, length, origin, point, 1.)
    nodes = origin + np.arange(2*n+1)*length/(2*n)
    assert b.sum() == pytest.approx(1., abs=2e-13)
    assert b @ nodes == pytest.approx(point, abs=2e-13)
    with pytest.raises(ValueError, match='support'):
        axis_weak_load(n, length, origin, origin+1., 1.)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA device unavailable'))])
def test_q2_density_load_approaches_independent_regularized_weak_rhs(device):
    point = (.37, -.41, -1.23)
    force = (1., -.7, .4)
    x = torch.tensor([point], device=device, dtype=torch.float64)
    g = torch.tensor([force], device=device, dtype=torch.float64)
    errors = []
    for n in (12, 18):
        mesh = create_box((n,)*3, (12.,)*3, (-6., -6., -8.), device=device)
        stencil = scaled_stencil(x, mesh.velocity_grid, n//6)
        dual = ib.spread_load(g, stencil)
        density = prepare_operators(mesh).density_load(dual/mesh.velocity_grid.cell_volume)
        reference = reference_weak_load(mesh.counts, mesh.lengths, mesh.origin, point, force)
        direct_error = np.linalg.norm(dual.cpu().numpy()-reference)/np.linalg.norm(reference)
        density_error = np.linalg.norm(density.cpu().numpy()-reference)/np.linalg.norm(reference)
        assert direct_error > .4
        errors.append(density_error)
    assert errors[0] < .03
    assert errors[1] < .01
    assert errors[1] < errors[0]/2


def test_box_control_rejects_incompatible_physical_kernel_width(tmp_path):
    with pytest.raises(ValueError, match='integer dilation'):
        run(output=tmp_path/'unused.json', levels=(18,), epsilon=2/3)
    assert not (tmp_path/'unused.json').exists()
