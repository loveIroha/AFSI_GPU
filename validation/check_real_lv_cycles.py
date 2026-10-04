"""Check completion and sampled acceptance of a fresh real-LV cycle run.

This checks existing output only; it never advances a simulation or claims
mesh/time convergence, periodic steady state, or physical validation.
"""
import argparse
import csv
import json
from math import isclose, isfinite
from pathlib import Path
import re


def check(folder, *, cycles=3):
    if type(cycles) is not int or cycles<1:
        raise ValueError('positive integer cycles required')
    folder=Path(folder)
    report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
    config=report['configuration']
    dt,period=config['time']['dt'],config['loads']['period']
    target=cycles*period
    expected=round(target/dt)
    errors=[]
    if not isclose(expected*dt,target,abs_tol=1e-12,rel_tol=0):
        errors.append('cycle horizon is not an integer number of steps')
    if not report.get('completed') or report.get('status')!='completed' or report.get('failure'):
        errors.append('simulation did not complete successfully')
    if report.get('accepted_steps')!=expected or not isclose(report.get('reached_time_s',-1),target,abs_tol=1e-12,rel_tol=0):
        errors.append('accepted state does not reach the requested cycle horizon')
    if not isclose(report.get('requested_end_time_s',-1),target,abs_tol=1e-12,rel_tol=0):
        errors.append('report requested a different horizon')
    with (folder/'history.csv').open(newline='',encoding='utf-8') as stream:
        history=list(csv.DictReader(stream))
    if not history or int(history[0]['step'])!=0:
        errors.append('history does not start from the initial state; check a resumed/branched run separately')
    if not history or int(history[-1]['step'])!=expected:
        errors.append('history does not include the final accepted step')
    previous=-1
    for row in history:
        step=int(row['step']); t=float(row['time_s'])
        if step<=previous or not isclose(t,step*dt,abs_tol=1e-12,rel_tol=0):
            errors.append(f'inconsistent history clock at step {step}')
        previous=step
        for key in ('minimum_detF','maximum_detF','cavity_volume_ml','wall_volume_cm3','divergence_l2','courant'):
            value=float(row[key])
            if not isfinite(value) or value<0 or (key in ('minimum_detF','maximum_detF','cavity_volume_ml','wall_volume_cm3') and value==0):
                errors.append(f'invalid {key} at sampled step {step}')
        if step>0:
            if not row.get('nonlinear_residual') or not row.get('nonlinear_tolerance'):
                errors.append(f'missing nonlinear acceptance at sampled step {step}')
            else:
                residual,tolerance=float(row['nonlinear_residual']),float(row['nonlinear_tolerance'])
                if not isfinite(residual) or not isfinite(tolerance) or residual<0 or tolerance<=0 or residual>tolerance:
                    errors.append(f'nonlinear tolerance failed at sampled step {step}')
    runtime=(folder/'runtime.txt').read_text(encoding='utf-8')
    match=re.search(r'elapsed_seconds=([0-9.]+) exit_code=(\d+)',runtime)
    if not match or int(match[2])!=0:
        errors.append('runtime.txt does not confirm exit_code=0')
    cycle_samples=[]
    for c in range(1,cycles+1):
        rows=[r for r in history if isclose(float(r['time_s']),c*period,abs_tol=1e-12,rel_tol=0)]
        if rows:
            r=rows[-1]
            cycle_samples.append(dict(cycle=c,time_s=float(r['time_s']),volume_ml=float(r['cavity_volume_ml']),minimum_detF=float(r['minimum_detF'])))
    return dict(passed=not errors,errors=errors,accepted_steps=report.get('accepted_steps'),
                target_time_s=target,elapsed_seconds=float(match[1]) if match else None,
                cycle_end_samples=cycle_samples,
                scope='completion and sampled acceptance only; not convergence or periodic steady state')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder')
    p.add_argument('--cycles',type=int,default=3)
    args=p.parse_args()
    try:
        result=check(args.folder,cycles=args.cycles)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        result=dict(passed=False,errors=[str(exc)])
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':
    main()
