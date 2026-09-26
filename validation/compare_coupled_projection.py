"""Short preloaded-LV IB/FEM trajectories with Chorin or reference Schur projection.

Every trajectory uses the same Q2 tentative solve, fixed 1 cm IB kernel,
density load, pressure schedule, solid model, time step and force lag. The
Schur branch replaces only the corrected fluid velocity. It is diagnostic,
not a new production fluid solver or a full cardiac cycle.
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
from afsi_torch.fluid.chorin import StepResult
from afsi_torch.fluid.solvers import SolverOptions
from afsi_torch.preload import load_preload
from afsi_torch.units import CGS_UNITS, MMHG_TO_DYN_PER_CM2

if __package__:
    from .compare_ib_box_projection import box_spec
    from .diagnose_ib import BoxMassInverse, discrete_projection, scaled_stencil
    from .verify_ib_weak import integer_dilation
else:
    from compare_ib_box_projection import box_spec
    from diagnose_ib import BoxMassInverse, discrete_projection, scaled_stencil
    from verify_ib_weak import integer_dilation


class ReferenceSchurFlow:
    """Wrap the unchanged Chorin tentative step and replace its projection.

    The stored pressure is Chorin's auxiliary pressure, used only as the next
    Poisson initial guess. It is not the Schur multiplier or a cavity pressure.
    The wrapper computes an otherwise unused Chorin correction for a controlled
    comparison; no fluid pressure feeds into the solid force callback.
    """
    def __init__(self, chorin, *, options=None):
        self.chorin = chorin
        self.op, self.dt = chorin.op, chorin.dt
        self.pressure_values = chorin.pressure_values
        self.inverse = BoxMassInverse(chorin.op.mesh)
        self.options = (SolverOptions(rtol=1e-12, atol=1e-14,
                                     max_iterations=4000, recompute_every=200)
                        if options is None else options)

    @torch.no_grad()
    def step(self, velocity, *, density=None, nodal_load=None, boundary_values=None,
             pressure_initial=None):
        baseline = self.chorin.step(velocity, density=density, nodal_load=nodal_load,
                                    boundary_values=boundary_values,
                                    pressure_initial=pressure_initial)
        projected, info = discrete_projection(self.chorin, baseline.tentative_velocity,
                                              self.inverse, options=self.options)
        diagnostics = dict(baseline.diagnostics)
        diagnostics['projection'] = 'schur_reference'
        diagnostics['solves'] = {**diagnostics['solves'], 'schur_projection': info['solve']}
        diagnostics['corrected_divergence_dual_norm'] = info['full_divergence_dual_norm']
        diagnostics['corrected_divergence_l2'] = self.chorin.divergence_l2(projected)
        diagnostics['kinetic_energy'] = (.5 * self.chorin.rho *
                                          (projected * self.op.velocity_mass(projected)).sum().item())
        diagnostics['chorin_auxiliary_velocity_difference_l2_cm_per_s'] = (
            torch.linalg.vector_norm(projected - baseline.velocity).item())
        return StepResult(projected, baseline.pressure, baseline.tentative_velocity, diagnostics)


def _write(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _difference(a, b):
    """Compare final solid responses from matched times and references."""
    if a['time_s'] != b['time_s'] or a['reference_sha256'] != b['reference_sha256']:
        raise ValueError('comparison requires identical physical time and solid reference')
    if not a['completed'] or not b['completed']:
        return dict(available=False, reason='incomplete case')
    with np.load(a['snapshot'], allow_pickle=False) as aa, np.load(b['snapshot'], allow_pickle=False) as bb:
        for name in ('X', 'x_preload', 'cells'):
            if not np.array_equal(aa[name], bb[name]):
                raise ValueError(f'comparison changed solid {name}')
        ua, ub = aa['x'] - aa['x_preload'], bb['x'] - bb['x_preload']
        absolute = float(np.linalg.norm(ub - ua))
        norm_a, norm_b = float(np.linalg.norm(ua)), float(np.linalg.norm(ub))
    va, vb = a['summary']['delta_cavity_ml'], b['summary']['delta_cavity_ml']
    return dict(available=True, absolute_displacement_nodal_l2_cm=absolute,
        displacement_relative_to_first=None if norm_a <= 1e-12 else absolute / norm_a,
        displacement_relative_to_second=None if norm_b <= 1e-12 else absolute / norm_b,
        absolute_delta_cavity_difference_ml=abs(vb - va),
        delta_cavity_relative_to_first=None if abs(va) <= 1e-10 else abs(vb - va) / abs(va),
        delta_cavity_relative_to_second=None if abs(vb) <= 1e-10 else abs(vb - va) / abs(vb))


@torch.no_grad()
def run(preload='results/lv_equilibrium', device='cpu', output='results/coupled_projection',
        levels=(18, 24), steps=20, dt=5e-5, epsilon=1., pressure_increment=.02):
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if (not levels or list(levels) != sorted(set(levels)) or
            any(type(n) is not int or n < 12 for n in levels) or
            type(steps) is not int or steps < 8 or not isfinite(dt) or dt <= 0 or
            not isfinite(epsilon) or epsilon <= 0 or
            not isfinite(pressure_increment) or pressure_increment <= 0):
        raise ValueError('increasing unique box levels >=12, steps>=8 and positive finite parameters required')
    dilation = integer_dilation(epsilon, .5)
    folder = Path(output)
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
        initial=initial, levels=list(levels), projections=['chorin', 'schur_reference'],
        velocity_spacing_cm=.5, epsilon_cm=epsilon, kernel_dilation=dilation,
        time_step_s=dt, steps=steps, final_time_s=duration,
        pressure_base_mmhg=baseline.pressure_mmhg,
        pressure_increment_mmhg=pressure_increment, loads=asdict(model.loads),
        load_path='density', force_order='lagged AFSI',
        schur_pressure_field='auxiliary Chorin pressure used only as Poisson initial guess',
        cases={}, projection_comparisons={}, box_comparisons={},
        completed=False, full_cycle_ready=False,
        scope='short prescribed-traction coupled trajectories; boundary and projection sensitivity')
    _write(folder / 'report.json', report)
    support = None
    options = SolverOptions(max_iterations=4000, recompute_every=200)
    for n in levels:
        counts, lengths, origin = box_spec(n)
        mesh = create_box(counts, lengths, origin, device=device)
        current_stencil = scaled_stencil(x0, mesh.velocity_grid, dilation)
        current_support = (mesh.velocity_coordinates[current_stencil.indices].cpu().numpy(),
                           current_stencil.weights.cpu().numpy())
        if support is None:
            support = current_support
        elif (not np.allclose(current_support[0], support[0], rtol=0, atol=1e-12) or
              not np.allclose(current_support[1], support[1], rtol=0, atol=1e-12)):
            raise RuntimeError('box expansion changed initial IB support')
        for projection in ('chorin', 'schur_reference'):
            chorin = ChorinSolver(prepare_operators(mesh), dt=dt, options=options)
            flow = chorin if projection == 'chorin' else ReferenceSchurFlow(chorin)
            case_folder = folder / projection / f'box_{n}'
            case_folder.mkdir(parents=True, exist_ok=True)
            snapshot = case_folder / 'last_accepted.npz'
            case = dict(projection=projection, box_length_cm=lengths[0], fluid_cells=n,
                        velocity_spacing_cm=.5, kernel_epsilon_cm=epsilon,
                        time_s=duration, reference_sha256=reference,
                        snapshot=str(snapshot), completed=False, accepted_steps=0, history=[])
            report['cases'][f'{projection}_{n}'] = case
            driver = ExplicitIBStepper(flow, model.force, model.validate,
                stencil_factory=lambda x, grid: scaled_stencil(x, grid, dilation),
                load_path='density')
            state = None
            try:
                state = driver.initialize_equilibrium(x0, force_tolerance=tolerance)
                for _ in range(steps):
                    result = driver.step(state)
                    state = result.state
                    solid = model.diagnostics(state.x)
                    fluid = result.diagnostics['fluid']
                    case['history'].append(dict(step=state.step, time_s=state.time,
                        applied_pressure_mmhg=(model.loads.at(result.diagnostics['used_force_time_s'])[0] /
                                               MMHG_TO_DYN_PER_CM2),
                        cavity_volume_ml=solid['cavity_volume_ml'],
                        wall_volume_cm3=solid['wall_volume_cm3'],
                        minimum_detF=solid['minimum_detF'],
                        max_incremental_displacement_cm=torch.linalg.vector_norm(state.x - x0, dim=-1).max().item(),
                        max_fluid_speed_cm_per_s=torch.linalg.vector_norm(state.velocity, dim=-1).max().item(),
                        corrected_divergence_dual_l2=fluid['corrected_divergence_dual_norm'],
                        corrected_divergence_l2=fluid['corrected_divergence_l2'],
                        fe_minus_solid_power_erg_per_s=result.diagnostics['fe_minus_solid_power'],
                        max_solver_residual_ratio=max(s['residual_norm'] / s['tolerance']
                            for s in fluid['solves'].values())))
                    _write(folder / 'report.json', report)
                rows = case['history']
                case['summary'] = dict(delta_cavity_ml=rows[-1]['cavity_volume_ml'] - initial['cavity_volume_ml'],
                    max_incremental_displacement_cm=max(v['max_incremental_displacement_cm'] for v in rows),
                    max_fluid_speed_cm_per_s=max(v['max_fluid_speed_cm_per_s'] for v in rows),
                    minimum_detF=min(v['minimum_detF'] for v in rows),
                    max_corrected_divergence_dual_l2=max(v['corrected_divergence_dual_l2'] for v in rows),
                    max_corrected_divergence_l2=max(v['corrected_divergence_l2'] for v in rows),
                    max_solver_residual_ratio=max(v['max_solver_residual_ratio'] for v in rows),
                    accumulated_fe_minus_solid_work_erg=dt * sum(
                        v['fe_minus_solid_power_erg_per_s'] for v in rows))
                case['completed'] = True
            except (RuntimeError, ValueError) as exc:
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
    for n in levels:
        report['projection_comparisons'][str(n)] = _difference(
            report['cases'][f'chorin_{n}'], report['cases'][f'schur_reference_{n}'])
    for projection in ('chorin', 'schur_reference'):
        report['box_comparisons'][projection] = {}
        for small, large in zip(levels, levels[1:]):
            report['box_comparisons'][projection][f'{small}_to_{large}'] = _difference(
                report['cases'][f'{projection}_{small}'],
                report['cases'][f'{projection}_{large}'])
    report['completed'] = True
    _write(folder / 'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preload', default='results/lv_equilibrium')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='results/coupled_projection')
    parser.add_argument('--levels', type=int, nargs='+', default=[18, 24])
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args()
    result = run(args.preload, args.device, args.output, tuple(args.levels), args.steps)
    print(json.dumps({**result, 'cases': {k: {key: value for key, value in v.items() if key != 'history'}
                                         for k, v in result['cases'].items()}}, indent=2, allow_nan=False))
