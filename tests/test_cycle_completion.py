"""Completion checker must reject partial or rejected trajectories."""
import csv
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('cycle_check',Path(__file__).resolve().parents[1]/'validation/check_real_lv_cycles.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def output(path):
    report=dict(completed=True,status='completed',accepted_steps=24000,reached_time_s=2.4,requested_end_time_s=2.4,
                configuration=dict(time=dict(dt=1e-4),loads=dict(period=.8)))
    (path/'report.json').write_text(json.dumps(report))
    rows=[dict(step=step,time_s=step*1e-4,minimum_detF=.9,maximum_detF=1.1,cavity_volume_ml=100.,
               wall_volume_cm3=77.,divergence_l2=1e-9,courant=.1,
               nonlinear_residual=1e-10 if step else '',nonlinear_tolerance=1e-9 if step else '')
          for step in (0,8000,16000,24000)]
    def write():
        with (path/'history.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]); writer.writeheader();writer.writerows(rows)
    write()
    (path/'runtime.txt').write_text('elapsed_seconds=11000.123 exit_code=0\n')
    return report,rows,write


def test_three_cycles_accepted(tmp_path):
    output(tmp_path)
    r=module.check(tmp_path)
    assert r['passed'] and len(r['cycle_end_samples'])==3
    assert r['elapsed_seconds']==11000.123


@pytest.mark.parametrize('problem',['partial','negative-J','nan-volume','residual','missing-residual','clock','exit','resumed'])
def test_reject_incomplete_or_invalid_samples(tmp_path,problem):
    report,rows,write=output(tmp_path)
    if problem=='partial':
        report.update(completed=False,status='failed',accepted_steps=16000,reached_time_s=1.6)
    elif problem=='negative-J': rows[2]['minimum_detF']=-.1
    elif problem=='nan-volume': rows[2]['cavity_volume_ml']=float('nan')
    elif problem=='residual': rows[2]['nonlinear_residual']=1e-5
    elif problem=='missing-residual': rows[2]['nonlinear_residual']=''
    elif problem=='clock': rows[2]['time_s']=1.59
    elif problem=='resumed': rows.pop(0)
    elif problem=='exit': (tmp_path/'runtime.txt').write_text('elapsed_seconds=11000.123 exit_code=130')
    (tmp_path/'report.json').write_text(json.dumps(report));write()
    assert not module.check(tmp_path)['passed']
