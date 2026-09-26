"""Controlled short FSI trajectories for IB kernel and FE load-path factors.

This is an experimental comparison, not a replacement for production coupling.
Every case starts from the same validated preloaded solid and zero fluid state.
The prescribed pressure schedule, Chorin solver, force lag, box, dt and solid
mesh are unchanged. Only kernel width and/or FE load mapping vary.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from math import isfinite
from pathlib import Path

import numpy as np
import torch

from afsi_torch.coupling import ExplicitIBStepper
from afsi_torch.fluid import ChorinSolver, create_box, prepare_operators
from afsi_torch.preload import load_preload
from afsi_torch.units import CGS_UNITS, MMHG_TO_DYN_PER_CM2
if __package__:
    from .diagnose_ib import scaled_stencil
    from .study_preloaded_ib import compare
else:
    from diagnose_ib import scaled_stencil
    from study_preloaded_ib import compare


METHODS = ('native_density', 'fixed_density', 'native_dual', 'fixed_dual')


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _method_difference(baseline, candidate):
    """Compare two methods at the same fluid resolution and physical time."""
    if baseline['time_s'] != candidate['time_s'] or baseline['fluid_cells'] != candidate['fluid_cells']:
        raise ValueError('method comparison requires identical resolution and time')
    if baseline['reference_sha256'] != candidate['reference_sha256']:
        raise ValueError('method comparison changed the preloaded solid')
    if not baseline['completed'] or not candidate['completed']:
        return dict(available=False, reason='incomplete case')
    with np.load(baseline['snapshot'], allow_pickle=False) as a, np.load(candidate['snapshot'], allow_pickle=False) as b:
        for name in ('X', 'x_preload', 'cells'):
            if not np.array_equal(a[name], b[name]):
                raise ValueError(f'method comparison changed solid {name}')
        ua = a['x'] - a['x_preload']
        ub = b['x'] - b['x_preload']
        denominator = float(np.linalg.norm(ua))
        displacement_difference = float(np.linalg.norm(ub - ua))
    volume_a = baseline['summary']['delta_cavity_ml']
    volume_b = candidate['summary']['delta_cavity_ml']
    return dict(available=True, fluid_cells=baseline['fluid_cells'],
                displacement_nodal_l2_difference_cm=displacement_difference,
                displacement_relative_to_baseline=None if denominator <= 1e-12 else displacement_difference / denominator,
                delta_cavity_difference_ml=abs(volume_b - volume_a),
                delta_cavity_relative_to_baseline=None if abs(volume_a) <= 1e-10 else abs(volume_b - volume_a) / abs(volume_a))


@torch.no_grad()
def run(preload='results/lv_equilibrium', device='cpu', output='results/coupled_ib_factors',
        fluid_levels=(6, 12, 18), methods=METHODS, steps=20, dt=5e-5,
        pressure_increment=.02, fixed_epsilon_cm=1.):
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if (not fluid_levels or any(type(n) is not int or n < 6 for n in fluid_levels) or
        list(fluid_levels) != sorted(set(fluid_levels)) or
        not methods or len(methods) != len(set(methods)) or any(m not in METHODS for m in methods) or
        type(steps) is not int or steps < 8 or not isfinite(dt) or dt <= 0 or
        not isfinite(pressure_increment) or pressure_increment <= 0 or
        not isfinite(fixed_epsilon_cm) or fixed_epsilon_cm <= 0):
        raise ValueError('ordered fluid levels >=6, valid unique methods, steps>=8 and positive parameters required')
    # Exact Peskin zero/first moments for this diagnostic width require integer dilation.
    dilations = {}
    for n in fluid_levels:
        ratio = fixed_epsilon_cm / (6. / n)
        if abs(ratio - round(ratio)) > 1e-12 or round(ratio) < 1:
            raise ValueError(f'fixed width at {n}^3 requires integer dilation of Q2 velocity spacing')
        dilations[n] = round(ratio)

    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    model, x0, tolerance, provenance = load_preload(preload, device=device)
    initial = model.diagnostics(x0)
    baseline = model.loads
    duration = steps * dt
    model.loads = replace(baseline, pressure_increment_mmhg=pressure_increment,
                          hold_time=duration / 2, ramp_time=duration / 4)
    reference = hashlib.sha256(model.mesh.X.cpu().numpy().tobytes() +
                               model.mesh.cells.cpu().numpy().tobytes() + x0.cpu().numpy().tobytes()).hexdigest()
    report = dict(device=str(device), torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
                  units=CGS_UNITS, preload=provenance, reference_sha256=reference,
                  initial=initial, pressure_base_mmhg=baseline.pressure_mmhg,
                  pressure_increment_mmhg=pressure_increment, time_step_s=dt,
                  steps=steps, final_time_s=duration, fluid_levels=list(fluid_levels),
                  methods=list(methods), fixed_kernel_epsilon_cm=fixed_epsilon_cm,
                  fixed_kernel_dilation={str(n): dilations[n] for n in fluid_levels},
                  loads=asdict(model.loads), cases={}, grid_comparisons={},
                  method_comparisons={}, completed=False, full_cycle_ready=False,
                  scope='short prescribed-traction perturbation; experimental factor isolation, not mesh convergence')
    _write(folder / 'report.json', report)

    for n in fluid_levels:
        mesh = create_box((n,) * 3, (12.,) * 3, (-6., -6., -8.), device=device)
        flow = ChorinSolver(prepare_operators(mesh), dt=dt)
        for method in methods:
            kernel, load_path = method.split('_')
            case_folder = folder / method / f'fluid_{n}'
            case_folder.mkdir(parents=True, exist_ok=True)
            snapshot = case_folder / 'last_accepted.npz'
            case = dict(method=method, kernel=kernel, load_path=load_path,
                        fluid_cells=n, time_s=duration, reference_sha256=reference,
                        snapshot=str(snapshot), completed=False, accepted_steps=0, history=[])
            report['cases'][f'{method}_{n}'] = case
            stencil_factory = None if kernel == 'native' else (
                lambda x, grid, dilation=dilations[n]: scaled_stencil(x, grid, dilation))
            driver = ExplicitIBStepper(flow, model.force, model.validate,
                                       stencil_factory=stencil_factory, load_path=load_path)
            state = None
            try:
                state = driver.initialize_equilibrium(x0, force_tolerance=tolerance)
                for _ in range(steps):
                    result = driver.step(state)
                    state = result.state
                    diagnostics = model.diagnostics(state.x)
                    solves = result.diagnostics['fluid']['solves']
                    case['history'].append(dict(step=state.step, time_s=state.time,
                        applied_pressure_mmhg=model.loads.at(result.diagnostics['used_force_time_s'])[0] / MMHG_TO_DYN_PER_CM2,
                        cavity_volume_ml=diagnostics['cavity_volume_ml'],
                        wall_volume_cm3=diagnostics['wall_volume_cm3'],
                        minimum_detF=diagnostics['minimum_detF'],
                        max_incremental_displacement_cm=torch.linalg.vector_norm(state.x - x0, dim=-1).max().item(),
                        max_fluid_speed_cm_per_s=torch.linalg.vector_norm(state.velocity, dim=-1).max().item(),
                        corrected_divergence_l2=result.diagnostics['fluid']['corrected_divergence_l2'],
                        fe_minus_solid_power_erg_per_s=result.diagnostics['fe_minus_solid_power'],
                        lattice_power_error_erg_per_s=result.diagnostics['lattice_power_error'],
                        solver_residual_ratio=max(s['residual_norm'] / s['tolerance'] for s in solves.values())))
                rows = case['history']
                case['summary'] = dict(delta_cavity_ml=rows[-1]['cavity_volume_ml'] - initial['cavity_volume_ml'],
                    max_incremental_displacement_cm=max(r['max_incremental_displacement_cm'] for r in rows),
                    max_fluid_speed_cm_per_s=max(r['max_fluid_speed_cm_per_s'] for r in rows),
                    minimum_detF=min(r['minimum_detF'] for r in rows),
                    max_absolute_wall_volume_change_cm3=max(abs(r['wall_volume_cm3'] - initial['wall_volume_cm3']) for r in rows),
                    max_corrected_divergence_l2=max(r['corrected_divergence_l2'] for r in rows),
                    max_solver_residual_ratio=max(r['solver_residual_ratio'] for r in rows),
                    accumulated_fe_minus_solid_work_erg=dt * sum(r['fe_minus_solid_power_erg_per_s'] for r in rows),
                    max_lattice_power_error_erg_per_s=max(r['lattice_power_error_erg_per_s'] for r in rows),
                    final_applied_pressure_mmhg=rows[-1]['applied_pressure_mmhg'])
                case['completed'] = True
            except (ValueError, RuntimeError) as exc:
                case['failure'] = dict(type=type(exc).__name__, message=str(exc),
                                       attempted_step=0 if state is None else state.step + 1)
                raise
            finally:
                case['accepted_steps'] = 0 if state is None else state.step
                if state is not None:
                    np.savez_compressed(snapshot, X=model.mesh.X.cpu().numpy(),
                        x_preload=x0.cpu().numpy(), x=state.x.cpu().numpy(),
                        cells=model.mesh.cells.cpu().numpy(), velocity=state.velocity.cpu().numpy(),
                        pressure=state.pressure.cpu().numpy(), force=state.force.cpu().numpy(),
                        time=state.time, step=state.step)
                _write(folder / 'report.json', report)
            print(f'{method} {n}^3: ' + json.dumps(case['summary']), flush=True)

    for method in methods:
        cases = [report['cases'][f'{method}_{n}'] for n in fluid_levels]
        report['grid_comparisons'][method] = [compare(a, b) for a, b in zip(cases, cases[1:])]
    if 'native_density' in methods:
        for n in fluid_levels:
            baseline_case = report['cases'][f'native_density_{n}']
            report['method_comparisons'][str(n)] = {method: _method_difference(
                baseline_case, report['cases'][f'{method}_{n}']) for method in methods if method != 'native_density'}
    report['completed'] = True
    _write(folder / 'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preload', default='results/lv_equilibrium')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='results/coupled_ib_factors')
    parser.add_argument('--fluid-levels', type=int, nargs='+', default=[6, 12, 18])
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--dt', type=float, default=5e-5)
    parser.add_argument('--pressure-increment-mmhg', type=float, default=.02)
    parser.add_argument('--fixed-epsilon-cm', type=float, default=1.)
    args = parser.parse_args()
    result = run(args.preload, args.device, args.output, tuple(args.fluid_levels), tuple(args.methods),
                 args.steps, args.dt, args.pressure_increment_mmhg, args.fixed_epsilon_cm)
    print(json.dumps({**result, 'cases': {k: {key: value for key, value in case.items() if key != 'history'}
                                        for k, case in result['cases'].items()}}, indent=2, allow_nan=False))
