"""Time refinement must compare the same reference, load and physical times."""
import copy

import numpy as np
import pytest

from validation.study_coupled_timestep import compare_reports


def _report(tmp_path, *, steps, dt, displacement):
    snapshot = tmp_path / f'{steps}.npz'
    np.savez(snapshot, X=np.zeros((2, 3)), x_preload=np.zeros((2, 3)),
             cells=np.zeros((1, 4), dtype=np.int64),
             x=np.array([[displacement, 0., 0.], [0., 0., 0.]]))
    cases = {}
    for projection in ('chorin', 'schur_reference'):
        history = [dict(step=i, time_s=i * dt,
                        applied_pressure_mmhg=.2 + (i - 1) * dt,
                        cavity_volume_ml=10. + displacement * i / steps,
                        max_incremental_displacement_cm=displacement * i / steps,
                        minimum_detF=1., corrected_divergence_dual_l2=0.)
                   for i in range(1, steps + 1)]
        cases[f'{projection}_24'] = dict(completed=True, accepted_steps=steps,
            time_s=steps * dt, reference_sha256='same', snapshot=str(snapshot),
            history=history, summary=dict(delta_cavity_ml=displacement))
    return dict(completed=True, device='cuda', steps=steps, time_step_s=dt,
        final_time_s=steps * dt, levels=[24], load_path='density',
        force_order='lagged AFSI', velocity_spacing_cm=.5, epsilon_cm=1.,
        pressure_increment_mmhg=.02, pressure_base_mmhg=.2,
        loads=dict(hold_time=.0002, ramp_time=.0001),
        reference_sha256='same', initial={'cavity_volume_ml': 10.},
        preload=dict(checkpoint_sha256='checkpoint', report_sha256='preload'),
        cases=cases, projection_comparisons={'24': {'available': True}})


def test_time_study_compares_shared_times_and_endpoint(tmp_path):
    coarse = _report(tmp_path, steps=8, dt=5e-5, displacement=1.)
    fine = _report(tmp_path, steps=16, dt=2.5e-5, displacement=1.1)
    result = compare_reports(coarse, fine)
    assert result['final_time_s'] == pytest.approx(.0004)
    assert len(result['comparisons']['chorin']['common_times']) == 8
    endpoint = result['comparisons']['chorin']['endpoint']
    assert endpoint['absolute_displacement_nodal_l2_cm'] == pytest.approx(.1)
    row = result['comparisons']['chorin']['common_times'][0]
    assert row['time_s'] == pytest.approx(5e-5)
    assert row['fine_applied_pressure_mmhg'] != row['coarse_applied_pressure_mmhg']


def test_time_study_rejects_changed_load_and_missing_snapshot(tmp_path):
    coarse = _report(tmp_path, steps=8, dt=5e-5, displacement=1.)
    fine = _report(tmp_path, steps=16, dt=2.5e-5, displacement=1.1)
    altered = copy.deepcopy(fine)
    altered['loads']['ramp_time'] = .0002
    with pytest.raises(ValueError, match='pressure schedule'):
        compare_reports(coarse, altered)
    altered = copy.deepcopy(fine)
    altered['cases']['chorin_24']['snapshot'] = str(tmp_path / 'missing.npz')
    with pytest.raises(ValueError, match='missing chorin endpoint'):
        compare_reports(coarse, altered)
