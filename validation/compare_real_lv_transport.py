"""Continue two time steps from a common real-LV checkpoint without overwriting it."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from afsi_torch.config import TimeConfig
from afsi_torch.real_lv_checkpoint import load_real_lv, save_real_lv
from afsi_torch.simulation.real_lv_mac import run
from afsi_torch.cycle_checkpoint import atomic_json
from afsi_torch.transport import transport_policy


def fork_case(path, model, state, settings, config, dt, end_time, write_vtk=True):
    """Retain common x/u/p; reset the clock and lagged load for the new dt."""
    original_dt = settings['dt']
    step = round(state.time/dt)
    if abs(step*dt-state.time) > 1e-12:
        raise ValueError('checkpoint time must be an integer multiple of both time steps')
    config = replace(config, time=TimeConfig(dt, end_time), output=replace(config.output,
        log_every=max(1, round(.001/dt)), output_every=max(1, round(.01/dt)),
        checkpoint_every=max(1, round(.01/dt)), write_vtk=write_vtk))
    force_time = None if step == 0 else (step-1)*dt
    force = state.force if dt == original_dt or step == 0 else model.force(state.x, force_time)
    branch = replace(state, step=step, time=step*dt, force_time=force_time, force=force)
    settings = dict(settings, dt=dt)
    progress = dict(elapsed_seconds=0., segments=[], summary={})
    save_real_lv(path, model, branch, settings, progress, config)
    return dict(dt=dt, start_step=step, start_time_s=branch.time,
                force_time_s=force_time, lagged_force_resampled=dt != original_dt and step > 0)


@torch.no_grad()
def compare(checkpoint, *, output, device='cuda', end_time=.05, write_vtk=True):
    folder = Path(output)
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('comparison output must be a new or empty directory')
    model, initial, settings, source_progress, config = load_real_lv(checkpoint, device)
    if end_time <= initial.time:
        raise ValueError('comparison end time must follow the common checkpoint time')
    original_dt = settings['dt']
    folder.mkdir(parents=True, exist_ok=True)
    cases = {}
    for name, dt in (('original_dt', original_dt), ('half_dt', original_dt/2)):
        case_dir = folder/name
        case_dir.mkdir()
        seed = fork_case(case_dir/'checkpoint.npz', model, initial, settings, config,
                         dt, end_time, write_vtk)
        print(f'{name}: resume t={initial.time:.8g} s with dt={dt:g}', flush=True)
        try:
            report = run(device=device, resume=case_dir/'checkpoint.npz', end_time=end_time)
        except Exception as exc:
            report_path = case_dir/'report.json'
            report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else dict(
                status='failed', completed=False, failure=dict(type=type(exc).__name__, message=str(exc)))
        cases[name] = dict(initial=seed, status=report['status'],
                           completed=report['completed'] and report['status'] == 'completed',
                           report=str(case_dir/'report.json'), last=report.get('last'),
                           failure=report.get('failure'), elapsed_seconds=report.get('elapsed_seconds'))
        atomic_json(folder/'report.json', dict(status='running', cases=cases))
    both_completed = all(case['completed'] for case in cases.values())
    differences = None
    if both_completed:
        _, a, _, _, _ = load_real_lv(folder/'original_dt/checkpoint.npz', device)
        _, b, _, _, _ = load_real_lv(folder/'half_dt/checkpoint.npz', device)
        increment_norm = torch.linalg.vector_norm(b.x-initial.x).item()
        delta_norm = torch.linalg.vector_norm(a.x-b.x).item()
        differences = dict(x_max_abs_cm=(a.x-b.x).abs().max().item(),
            increment_relative_l2=delta_norm/max(increment_norm, 1e-30),
            cavity_volume_abs_ml=abs(cases['original_dt']['last']['cavity_volume_ml']-
                                    cases['half_dt']['last']['cavity_volume_ml']),
            velocity_component_max_abs_cm_per_s=[(u-v).abs().max().item() for u,v in zip(a.velocity,b.velocity)])
    report = dict(schema=1, experiment='real-LV transport recovery from a common saved state',
        status='completed' if both_completed else 'failed', both_completed=both_completed,
        source_checkpoint=str(Path(checkpoint)), common_start_time_s=initial.time,
        source_failure=source_progress.get('failure'),
        requested_end_time_s=end_time, transport_policy=transport_policy(), cases=cases,
        final_state_differences=differences,
        interpretation='common-state continuation, not whole-history temporal convergence; '
                       'half-dt branch resamples its lagged force at start_time-half_dt; '
                       'two successful short trajectories do not validate a full cardiac cycle')
    atomic_json(folder/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--end-time', type=float, default=.05)
    parser.add_argument('--output', default='results/real_lv_transport_recovery')
    parser.add_argument('--no-vtk', action='store_true')
    args = parser.parse_args()
    report = compare(args.checkpoint, device=args.device, end_time=args.end_time,
                     output=args.output, write_vtk=not args.no_vtk)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report['both_completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
