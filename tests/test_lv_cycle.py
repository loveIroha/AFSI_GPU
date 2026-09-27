"""Cycle phase boundaries and persistence of the nonlinear coupled trajectory."""
import csv
from dataclasses import replace
import json
import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

from afsi_torch.cycle_checkpoint import load_cycle
from afsi_torch.cycle_loads import AFSICycleLoads
from afsi_torch.units import MMHG_TO_DYN_PER_CM2
from examples.lv_cycle import run


def test_afsi_cycle_phase_values_and_continuity():
    load = AFSICycleLoads()
    assert load.at(0.) == (0., 0.)
    assert load.at(.1) == pytest.approx((4*MMHG_TO_DYN_PER_CM2, 0.))
    for t in (.2, .4, .5, .8, 1., 1.3):
        assert load.at(t) == pytest.approx((8*MMHG_TO_DYN_PER_CM2, 0.), abs=1e-8)
    peak_p, peak_t = load.at(.65)
    assert peak_p == pytest.approx(load.diastole_pressure +
        (150000.-load.diastole_pressure)*(1-math.exp(-.15**2/.004)))
    assert peak_t == pytest.approx(600000.*(1-math.exp(-.15**2/.005)))
    assert load.at(.6) == pytest.approx(load.at(.7))
    # No return to zero pressure at the first-cycle boundary.
    for t in (.2, .5, .65, .8, 1.6):
        assert load.at(t-1e-10) == pytest.approx(load.at(t+1e-10), rel=1e-7, abs=1e-5)
    for t in (.3, .6, .75):
        assert load.at(t) == pytest.approx(load.at(t+.8))
    preloaded = replace(load, initial_pressure=.2*MMHG_TO_DYN_PER_CM2)
    assert preloaded.at(0.)[0] == pytest.approx(.2*MMHG_TO_DYN_PER_CM2)
    for t in (-1., float('nan'), float('inf')):
        with pytest.raises(ValueError):
            load.at(t)


@pytest.mark.parametrize('kwargs', [dict(period=.4), dict(filling_end=0.),
    dict(initial_pressure=-1.), dict(diastole_pressure=2e5), dict(max_tension=-1.),
    dict(pressure_width=0.), dict(tension_width=float('nan'))])
def test_cycle_rejects_invalid_loads(kwargs):
    with pytest.raises(ValueError):
        AFSICycleLoads(**kwargs)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA device unavailable'))])
def test_cycle_resume_matches_uninterrupted_run(tmp_path, device, monkeypatch):
    pytest.importorskip('gmsh')
    pytest.importorskip('meshio')
    dt = 5e-5
    direct, split = tmp_path/'direct', tmp_path/'split'
    config = dict(device=device, dt=dt, mesh_size=2.5, fluid_cells=3,
                  output_every=2, checkpoint_every=2, history_every=1, log_every=20)
    whole = run(**config, output=direct, end_time=6*dt)
    partial = run(**config, output=split, end_time=3*dt)
    assert partial['completed'] and not partial['full_cycle_completed']
    # Simulate an output row and frame newer than the durable checkpoint.
    with (split/'history.csv').open('a', newline='', encoding='utf-8') as stream:
        with (split/'history.csv').open(newline='', encoding='utf-8') as source:
            rows = list(csv.DictReader(source))
        extra = dict(rows[-1], step=100, time_s=100*dt)
        csv.DictWriter(stream, fieldnames=extra.keys()).writerow(extra)
    def no_remesh(*args, **kwargs):
        raise AssertionError('resume must not regenerate geometry')
    monkeypatch.setattr('examples.lv_cycle.generate_lv', no_remesh)
    resumed = run(device=device, resume=split/'checkpoint.npz', end_time=6*dt,
                  output_every=2, checkpoint_every=2, history_every=1, log_every=20)
    assert resumed['accepted_steps'] == whole['accepted_steps'] == 6
    assert resumed['last']['next_force_time_s'] == 5*dt
    assert resumed['last']['applied_force_time_s'] == 4*dt
    assert resumed['last']['force_norm_dyn'] > 0
    assert resumed['resumptions'] == 1
    with np.load(direct/'checkpoint.npz', allow_pickle=False) as a, np.load(
            split/'checkpoint.npz', allow_pickle=False) as b:
        for name in ('X', 'cells', 'x_start'):
            np.testing.assert_array_equal(a[name], b[name])
        for name in ('x', 'velocity', 'pressure', 'force'):
            np.testing.assert_allclose(a[name], b[name], rtol=1e-9, atol=1e-8)
    with (split/'history.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row['step']) for row in rows] == list(range(7))
    frames = list(ET.parse(split/'solid.pvd').getroot().iter('DataSet'))
    times = [float(frame.attrib['timestep']) for frame in frames]
    assert times == sorted(set(times)) and times[-1] == 6*dt
    with pytest.raises(ValueError, match='resume restores'):
        run(device=device, resume=split/'checkpoint.npz', dt=dt, end_time=8*dt)
    # Numeric checkpoint tampering must be detected before reusing any force.
    with np.load(split/'checkpoint.npz', allow_pickle=False) as archive:
        fields = {name: archive[name] for name in archive.files}
    fields['force'][0, 0] += 1.
    bad = tmp_path/'bad.npz'
    np.savez_compressed(bad, **fields)
    with pytest.raises(ValueError, match='checksum'):
        load_cycle(bad, device)


def test_cycle_failure_saves_last_accepted_state(tmp_path, monkeypatch):
    pytest.importorskip('gmsh')
    from afsi_torch.coupling import ExplicitIBStepper
    original = ExplicitIBStepper.step
    def fail_at_third(self, state, **kwargs):
        if state.step == 2:
            raise RuntimeError('injected step failure')
        return original(self, state, **kwargs)
    monkeypatch.setattr(ExplicitIBStepper, 'step', fail_at_third)
    with pytest.raises(RuntimeError, match='injected'):
        run(device='cpu', output=tmp_path, end_time=.0003, mesh_size=2.5,
            fluid_cells=3, write_vtk=False, checkpoint_every=10, history_every=1)
    report = json.loads((tmp_path/'report.json').read_text(encoding='utf-8'))
    assert report['status'] == 'failed' and not report['completed']
    assert report['accepted_steps'] == 2
    _, state, _, _, progress = load_cycle(tmp_path/'checkpoint.npz')
    assert state.step == 2 and progress['failure']['message'] == 'injected step failure'

