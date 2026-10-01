"""Distinguish conservative cell-Re stops from CFL failures and inspect saved states."""
import json
import numpy as np
import pytest
import torch
from afsi_torch.mac import MACGrid, MACFlow
from afsi_torch.mac.flow import MACTransportGuardError
from validation.diagnose_real_lv_guard import diagnose

DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))]


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('backend', ['torch', 'fused'])
@pytest.mark.parametrize('dt', [1e-4, 5e-5])
def test_cell_re_guard_is_independent_of_dt_and_precedes_predictor(device, backend, dt, monkeypatch):
    grid = MACGrid((4,)*3, (15/32,)*3)  # Same h as the real 128^3 demo.
    flow = MACFlow(grid, dt=dt, device=device, execution_backend=backend)
    velocity = tuple(torch.full_like(v, 8.6) for v in grid.zeros(device=device))
    before = tuple(v.clone() for v in velocity)
    def forbid_prediction(*args):
        pytest.fail('a rejected step must not execute its predictor')
    monkeypatch.setattr(flow, '_predict', forbid_prediction)
    monkeypatch.setattr(flow, 'project', forbid_prediction)
    with pytest.raises(MACTransportGuardError, match='Reducing dt affects CFL, not cell_Re') as error:
        flow.step(velocity, grid.zeros(device=device))
    info = error.value.diagnostics
    assert info['triggered'] == ['cell_reynolds']
    assert info['cell_reynolds'] == pytest.approx(8.6*15/128)
    assert info['courant'] == pytest.approx(.022016*dt/1e-4)
    assert info['courant'] < .25
    assert info['component_velocity_at_cell_re_limit'] == pytest.approx([128/15]*3)
    for a, b in zip(velocity, before):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_checkpoint_inspection_finds_staggered_peak_and_preserves_file(tmp_path):
    grid = MACGrid((4, 8, 4), (.5, 1., .5), (1., 2., 3.))
    velocity = [np.zeros(grid.face_shape(c)) for c in range(3)]
    velocity[1][1, 2, 3] = -9.
    meta = dict(producer='afsi-torch-real-lv-mac', schema=1, units='cm-g-s', step=234, time=.0234,
                settings=dict(dt=1e-4, rho=1., mu=1., fluid_shape=grid.shape,
                              fluid_lengths=grid.lengths, fluid_origin=grid.origin),
                progress=dict(failure={'message': 'old generic guard error'}))
    path = tmp_path/'checkpoint.npz'
    np.savez_compressed(path, metadata=json.dumps(meta),
                        **{f'velocity_{c}': u for c, u in enumerate(velocity)})
    saved = path.read_bytes()
    result = diagnose(path)
    assert path.read_bytes() == saved
    assert result['step'] == 234 and result['attempted_next_step'] == 235
    assert result['triggered'] == ['cell_reynolds']
    assert result['courant'] == pytest.approx(.0072)
    assert result['cell_reynolds'] == pytest.approx(1.125)
    assert result['component_peaks'][1]['location_cm'] == pytest.approx([1.1875, 2.25, 3.4375])
    assert result['saved_failure']['message'] == 'old generic guard error'
    json.dumps(result, allow_nan=False)


def test_both_transport_limits_are_reported():
    grid = MACGrid((4,)*3, (.5,)*3)
    flow = MACFlow(grid, dt=1e-4)
    velocity = tuple(torch.full_like(v, 1000.) for v in grid.zeros())
    with pytest.raises(MACTransportGuardError) as error:
        flow.step(velocity, grid.zeros())
    assert error.value.diagnostics['triggered'] == ['courant', 'cell_reynolds']
