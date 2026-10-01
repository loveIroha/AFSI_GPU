"""Centered advection/diffusion screening, old checkpoints and unchanged updates."""
import json
import numpy as np
import pytest
import torch
from afsi_torch.mac import MACGrid, MACFlow
from afsi_torch.mac.flow import MACTransportGuardError
from afsi_torch.mac.grid import zero_normal, convection, velocity_laplacian
from afsi_torch.transport import transport_numbers, transport_violations
from validation.diagnose_real_lv_guard import diagnose

DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))]


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('backend', ['torch', 'fused'])
def test_cell_re_above_one_keeps_original_centered_mac_update(device, backend):
    grid = MACGrid((4,)*3, (15/32,)*3)
    flow = MACFlow(grid, dt=1e-4, device=device, execution_backend=backend)
    velocity = zero_normal(tuple(torch.full_like(v, 8.6) for v in grid.zeros(device=device)))
    before = tuple(v.clone() for v in velocity)
    adv = convection(velocity, grid.spacing)
    star = zero_normal(tuple(u+flow.dt*(-a+velocity_laplacian(u,c,grid.spacing))
                             for c,(u,a) in enumerate(zip(velocity,adv))))
    expected = flow.project(star)
    result = flow.step(velocity, grid.zeros(device=device))
    assert result.diagnostics['cell_reynolds'] > 1.
    assert result.diagnostics['advection_diffusion_number'] < .25
    torch.testing.assert_close(result.pressure, expected.pressure, atol=1e-8, rtol=1e-8)
    for actual, reference, old, saved in zip(result.velocity, expected.velocity, velocity, before):
        torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-8)
        torch.testing.assert_close(old, saved, atol=0, rtol=0)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('backend', ['torch', 'fused'])
def test_small_cfl_still_rejects_insufficient_viscosity_and_halving_dt_helps(device, backend, monkeypatch):
    grid = MACGrid((4,)*3, (1.,)*3)
    flow = MACFlow(grid, dt=1e-4, mu=.001, device=device, execution_backend=backend)
    velocity = zero_normal(tuple(torch.full_like(v, 1.5) for v in grid.zeros(device=device)))
    def forbid_prediction(*args):
        pytest.fail('a rejected step must not execute its predictor')
    monkeypatch.setattr(flow, '_predict', forbid_prediction)
    monkeypatch.setattr(flow, 'project', forbid_prediction)
    with pytest.raises(MACTransportGuardError, match='Reduce dt') as error:
        flow.step(velocity, grid.zeros(device=device))
    info = error.value.diagnostics
    assert info['triggered'] == ['advection_diffusion']
    assert info['courant'] < .01
    assert info['advection_diffusion_number'] == pytest.approx(.3375)
    half = MACFlow(grid, dt=5e-5, mu=.001, device=device, execution_backend=backend)
    result = half.step(velocity, grid.zeros(device=device))
    assert result.diagnostics['advection_diffusion_number'] == pytest.approx(.16875)
    assert result.diagnostics['cell_reynolds'] == pytest.approx(info['cell_reynolds'])


def test_checkpoint_inspection_reassesses_legacy_guard_without_mutating_file(tmp_path):
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
    assert result['triggered'] == [] and result['legacy_cell_re_gt_one']
    assert result['courant'] == pytest.approx(.0072)
    assert result['cell_reynolds'] == pytest.approx(1.125)
    assert result['advection_diffusion_number'] == pytest.approx(.00405)
    assert result['component_peaks'][1]['location_cm'] == pytest.approx([1.1875, 2.25, 3.4375])
    assert result['saved_failure']['message'] == 'old generic guard error'
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('speed,mu,triggered', [
    (22., 5., ['courant']), (1000., 1., ['courant', 'advection_diffusion'])])
def test_time_step_violations_are_distinguished(speed, mu, triggered):
    grid = MACGrid((4,)*3, (1.,)*3)
    flow = MACFlow(grid, dt=.001, mu=mu)
    velocity = tuple(torch.full_like(v, speed) for v in grid.zeros())
    with pytest.raises(MACTransportGuardError) as error:
        flow.step(velocity, grid.zeros())
    assert error.value.diagnostics['triggered'] == triggered


@pytest.mark.parametrize('spacing,dt,nu,speeds', [
    ((.1,.2,.3), 5e-5, .5, (10.,15.,20.)),
    ((.25,.4,.5), .001, 3., (5.,4.,3.)),
    ((.25,.25,.25), 1e-4, .001, (.25,.5,.75))])
def test_accepted_screen_is_inside_independent_frozen_fourier_stability_region(spacing, dt, nu, speeds):
    numbers = transport_numbers(speeds, spacing, dt, nu)
    assert not transport_violations(numbers['courant'], numbers['advection_diffusion_number'],
                                    numbers['viscous_number'])
    theta = np.meshgrid(*(np.linspace(-np.pi,np.pi,33),)*3, indexing='ij')
    # Independent forward-Euler/centered symbol; frozen periodic scalar model only.
    symbol = 1-sum(4*nu*dt/h**2*np.sin(t/2)**2 for h,t in zip(spacing,theta))
    symbol = symbol-1j*sum(dt*u/h*np.sin(t) for u,h,t in zip(speeds,spacing,theta))
    assert np.max(np.abs(symbol)) <= 1+1e-12


def test_unsafe_frozen_mode_rejected_even_with_small_cfl():
    dt, nu, h, speed = 1e-4, .001, .25, 30.
    numbers = transport_numbers((speed,0.,0.), (h,)*3, dt, nu)
    assert numbers['courant'] < .25
    theta = .1
    symbol = 1-4*nu*dt/h**2*np.sin(theta/2)**2-1j*dt*speed/h*np.sin(theta)
    assert abs(symbol) > 1.
    assert transport_violations(numbers['courant'], numbers['advection_diffusion_number'],
                                numbers['viscous_number']) == ['advection_diffusion']
