"""Compare coupled IB/FEM time steps at fixed geometry and fluid spacing.

The coarse trajectory may be reused from compare_coupled_projection.py. Both
trajectories must have the same physical duration and prescribed load schedule.
"""
import argparse
import hashlib
import json
from math import isclose, isfinite
from pathlib import Path

if __package__:
    from .compare_coupled_projection import _difference, run as run_projection
else:
    from compare_coupled_projection import _difference, run as run_projection


PROJECTIONS = ('chorin', 'schur_reference')


def _write(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _same_time(a, b):
    return isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-14)


def _check_report(report, *, box, steps, dt, pressure_increment, epsilon):
    if not report.get('completed') or report.get('device') != 'cuda':
        raise ValueError('a completed CUDA trajectory report is required')
    if (report.get('steps') != steps or not _same_time(report.get('time_step_s', -1), dt)
            or not _same_time(report.get('final_time_s', -1), steps * dt)):
        raise ValueError('trajectory time step or duration differs from study')
    if (box not in report.get('levels', []) or
            report.get('load_path') != 'density' or
            report.get('force_order') != 'lagged AFSI' or
            report.get('velocity_spacing_cm') != .5 or
            report.get('epsilon_cm') != epsilon or
            report.get('pressure_increment_mmhg') != pressure_increment):
        raise ValueError('trajectory spatial or coupling configuration differs from study')
    loads = report.get('loads', {})
    if (not _same_time(loads.get('hold_time', -1), steps * dt / 2) or
            not _same_time(loads.get('ramp_time', -1), steps * dt / 4)):
        raise ValueError('trajectory pressure schedule differs from study')
    for projection in PROJECTIONS:
        case = report.get('cases', {}).get(f'{projection}_{box}', {})
        if (not case.get('completed') or case.get('accepted_steps') != steps or
                len(case.get('history', [])) != steps or
                not _same_time(case.get('time_s', -1), steps * dt) or
                case.get('reference_sha256') != report.get('reference_sha256')):
            raise ValueError(f'incomplete or mismatched {projection} trajectory')
        if not Path(case['snapshot']).is_file():
            raise ValueError(f'missing {projection} endpoint snapshot: {case["snapshot"]}')
        for number, row in enumerate(case['history'], 1):
            if row.get('step') != number or not _same_time(row.get('time_s', -1), number * dt):
                raise ValueError(f'{projection} history has a missing or mis-timed step')


def compare_reports(coarse, fine, *, box=24, refinement=2):
    """Require matching provenance and compare endpoint fields and common times."""
    if type(refinement) is not int or refinement < 2:
        raise ValueError('integer refinement >=2 required')
    coarse_steps, coarse_dt = coarse['steps'], coarse['time_step_s']
    _check_report(coarse, box=box, steps=coarse_steps, dt=coarse_dt,
                  pressure_increment=coarse['pressure_increment_mmhg'],
                  epsilon=coarse['epsilon_cm'])
    _check_report(fine, box=box, steps=coarse_steps * refinement,
                  dt=coarse_dt / refinement,
                  pressure_increment=coarse['pressure_increment_mmhg'],
                  epsilon=coarse['epsilon_cm'])
    for field in ('reference_sha256', 'pressure_base_mmhg', 'loads'):
        if coarse[field] != fine[field]:
            raise ValueError(f'coarse and fine {field} differ')
    if coarse['initial'].keys() != fine['initial'].keys() or any(
            not isclose(coarse['initial'][key], fine['initial'][key], rel_tol=1e-12, abs_tol=1e-12)
            for key in coarse['initial']):
        raise ValueError('coarse and fine initial diagnostics differ')
    for field in ('checkpoint_sha256', 'report_sha256'):
        if coarse['preload'][field] != fine['preload'][field]:
            raise ValueError(f'coarse and fine preload {field} differ')
    comparisons = {}
    for projection in PROJECTIONS:
        first = coarse['cases'][f'{projection}_{box}']
        second = fine['cases'][f'{projection}_{box}']
        common = []
        for index, a in enumerate(first['history']):
            b = second['history'][(index + 1) * refinement - 1]
            if not _same_time(a['time_s'], b['time_s']):
                raise ValueError('common physical times do not match')
            common.append(dict(time_s=a['time_s'],
                coarse_applied_pressure_mmhg=a['applied_pressure_mmhg'],
                fine_applied_pressure_mmhg=b['applied_pressure_mmhg'],
                cavity_volume_difference_ml=b['cavity_volume_ml'] - a['cavity_volume_ml'],
                max_displacement_difference_cm=(b['max_incremental_displacement_cm'] -
                                                a['max_incremental_displacement_cm']),
                coarse_minimum_detF=a['minimum_detF'],
                fine_minimum_detF=b['minimum_detF'],
                coarse_divergence_dual_l2=a['corrected_divergence_dual_l2'],
                fine_divergence_dual_l2=b['corrected_divergence_dual_l2']))
        comparisons[projection] = dict(endpoint=_difference(first, second), common_times=common,
            coarse_summary=first['summary'], fine_summary=second['summary'])
    return dict(completed=True, full_cycle_ready=False, box_length_cm=box,
        velocity_spacing_cm=.5, coarse_dt_s=coarse_dt, fine_dt_s=coarse_dt / refinement,
        final_time_s=coarse_steps * coarse_dt, reference_sha256=coarse['reference_sha256'],
        preload_checkpoint_sha256=coarse['preload']['checkpoint_sha256'],
        comparisons=comparisons,
        coarse_projection_gap=coarse['projection_comparisons'][str(box)],
        fine_projection_gap=fine['projection_comparisons'][str(box)])


def run(*, preload='results/lv_equilibrium', device='cuda', output='results/coupled_timestep',
        reuse_coarse=None, box=24, steps=20, dt=5e-5, refinement=2,
        epsilon=1., pressure_increment=.02):
    if (device != 'cuda' or type(box) is not int or box < 12 or type(steps) is not int or
            steps < 8 or type(refinement) is not int or refinement < 2 or
            not all(isfinite(v) and v > 0 for v in (dt, epsilon, pressure_increment))):
        raise ValueError('CUDA, box>=12, steps>=8, refinement>=2 and positive finite inputs required')
    folder = Path(output)
    if reuse_coarse is None:
        coarse = run_projection(preload, device, folder / 'coarse', (box,), steps, dt,
                                epsilon, pressure_increment)
    else:
        coarse = json.loads(Path(reuse_coarse).read_text(encoding='utf-8'))
        _check_report(coarse, box=box, steps=steps, dt=dt,
                      pressure_increment=pressure_increment, epsilon=epsilon)
        for name, field in (('last_converged.npz', 'checkpoint_sha256'),
                            ('report.json', 'report_sha256')):
            actual = hashlib.sha256((Path(preload) / name).read_bytes()).hexdigest()
            if actual != coarse['preload'][field]:
                raise ValueError(f'reused coarse trajectory has a different preload {name}')
    fine = run_projection(preload, device, folder / 'fine', (box,), steps * refinement,
                          dt / refinement, epsilon, pressure_increment)
    result = compare_reports(coarse, fine, box=box, refinement=refinement)
    result['coarse_report'] = str(reuse_coarse or folder / 'coarse' / 'report.json')
    result['fine_report'] = str(folder / 'fine' / 'report.json')
    _write(folder / 'report.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preload', default='results/lv_equilibrium')
    parser.add_argument('--device', default='cuda', choices=['cuda'])
    parser.add_argument('--output', default='results/coupled_timestep')
    parser.add_argument('--reuse-coarse', help='Existing completed compare_coupled_projection.py report')
    parser.add_argument('--box', type=int, default=24)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--dt', type=float, default=5e-5)
    parser.add_argument('--refinement', type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(preload=args.preload, device=args.device, output=args.output,
                         reuse_coarse=args.reuse_coarse, box=args.box, steps=args.steps,
                         dt=args.dt, refinement=args.refinement), indent=2, allow_nan=False))
