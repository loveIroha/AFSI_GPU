"""Study comparability and honest failure/response reporting."""
from dataclasses import asdict
import json
import numpy as np
import pytest
import torch
from validation.study_lv import StudyCase, StudyRunner, study_cases, compare_pair, assess, relative_difference


@pytest.mark.parametrize('kwargs', [dict(dt=0), dict(final_time=.00015), dict(fluid_cells=2.5),
                                   dict(name='../outside'), dict(mesh_size=float('nan'))])
def test_invalid_study_case(kwargs):
    with pytest.raises(ValueError):
        StudyCase(**(dict(name='test') | kwargs))


def record(case, delta=1., volume=80., completed=True):
    return dict(case=asdict(case), completed=completed, reached_time_s=case.final_time,
                initial=dict(cavity_volume_ml=volume), final=dict(cavity_volume_ml=volume+delta),
                summary=dict(delta_cavity_ml=delta, max_displacement_cm=.01))


def test_response_comparison_removes_initial_geometry_offset(tmp_path):
    coarse = record(StudyCase('a', mesh_size=1.4), delta=.001, volume=81.)
    fine = record(StudyCase('b', mesh_size=.9), delta=.002, volume=84.)
    result = compare_pair(coarse, fine, tmp_path, 'solid')
    assert result['delta_cavity_difference_ml'] == pytest.approx(.001)
    assert result['delta_cavity_relative_difference'] == pytest.approx(.5)
    assert relative_difference(0., 0., 1e-10) is None


def test_comparison_rejects_changed_time_or_multiple_parameters(tmp_path):
    baseline = record(StudyCase('a'))
    with pytest.raises(ValueError, match='physical time'):
        compare_pair(baseline, record(StudyCase('b', final_time=.002)), tmp_path, 'time')
    with pytest.raises(ValueError, match='only one'):
        compare_pair(baseline, record(StudyCase('b', dt=5e-5, fluid_cells=8)), tmp_path, 'time')


def test_nodal_comparison_rejects_unmatched_mesh(tmp_path):
    for name, X in [('a', np.zeros((2, 3))), ('b', np.ones((2, 3)))]:
        (tmp_path/name).mkdir()
        np.savez(tmp_path/name/'last_accepted.npz', X=X, x=X+.1, cells=np.array([[0, 1]]))
    with pytest.raises(ValueError, match='identical reference mesh'):
        compare_pair(record(StudyCase('a')), record(StudyCase('b', dt=5e-5)), tmp_path, 'time')


def test_failed_runs_cannot_pass_screen(tmp_path):
    reports = {c.name: record(c, completed=False) for c in study_cases('smoke')}
    result = assess(reports, tmp_path, 'smoke')
    assert result['axes']['time']['screen'] == 'inconclusive'
    assert not result['all_requested_runs_completed']
    assert not result['all_refinement_screens_met']
    assert not result['full_cycle_ready']


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA device unavailable'))])
def test_study_case_records_real_state(device, tmp_path):
    pytest.importorskip('gmsh')
    r = StudyRunner(device, tmp_path).run_case(StudyCase('small', final_time=.0002))
    assert r['completed'] and r['accepted_steps'] == 2
    assert r['summary']['max_solver_residual_ratio'] <= 1
    assert r['reached_time_s'] == .0002
    with np.load(tmp_path/'small'/'last_accepted.npz', allow_pickle=False) as state:
        assert state['step'] == 2
        assert np.isfinite(state['x']).all()


def test_failed_step_preserves_last_accepted_state(tmp_path, monkeypatch):
    pytest.importorskip('gmsh')
    from afsi_torch.coupling import ExplicitIBStepper
    step = ExplicitIBStepper.step
    def fail(self, state):
        if state.step == 1:
            raise ValueError('injected rejected step')
        return step(self, state)
    monkeypatch.setattr(ExplicitIBStepper, 'step', fail)
    r = StudyRunner('cpu', tmp_path).run_case(StudyCase('reject', final_time=.0004))
    assert not r['completed'] and r['accepted_steps'] == 1
    assert r['failure']['attempted_step'] == 2
    saved = json.loads((tmp_path/'reject'/'report.json').read_text())
    assert len(saved['history']) == 1
    with np.load(tmp_path/'reject'/'last_accepted.npz', allow_pickle=False) as state:
        assert state['step'] == 1
        assert state['time'] == 1e-4


def test_setup_failure_is_recorded_without_fake_state(tmp_path, monkeypatch):
    runner = StudyRunner('cpu', tmp_path)
    (tmp_path/'setup').mkdir()
    (tmp_path/'setup'/'last_accepted.npz').write_bytes(b'stale previous result')
    (tmp_path/'setup'/'history.csv').write_text('stale previous result')
    def fail(_):
        raise RuntimeError('injected setup failure')
    monkeypatch.setattr(runner, 'model', fail)
    r = runner.run_case(StudyCase('setup'))
    assert not r['completed'] and r['accepted_steps'] == 0
    assert r['failure']['phase'] == 'setup'
    assert not (tmp_path/'setup'/'last_accepted.npz').exists()
    assert not (tmp_path/'setup'/'history.csv').exists()
