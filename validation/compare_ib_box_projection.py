"""Fixed-spacing fluid-box and projection study for a frozen preloaded LV.

This is one forced fluid step, not a coupled trajectory. The preloaded solid,
incremental nodal force, IB lattice phase, kernel width and fluid spacing are
identical in all cases. Only outer wall location and projection are compared.
"""
import argparse
from dataclasses import replace
import json
from itertools import combinations
from math import isfinite
from pathlib import Path

import numpy as np
import torch

from afsi_torch import ib
from afsi_torch.fluid import ChorinSolver, create_box, prepare_operators
from afsi_torch.fluid.solvers import SolverOptions
from afsi_torch.preload import load_preload
from afsi_torch.units import CGS_UNITS

if __package__:
    from .diagnose_ib import BoxMassInverse, discrete_projection, scaled_stencil
    from .verify_ib_weak import integer_dilation
else:
    from diagnose_ib import BoxMassInverse, discrete_projection, scaled_stencil
    from verify_ib_weak import integer_dilation


def box_spec(n):
    """Center each n-cm box on the same LV and retain the 0.5-cm Q2 lattice."""
    length = float(n)
    return (n,) * 3, (length,) * 3, (-length / 2, -length / 2, -length / 2 - 2.)


def response_comparisons(responses):
    """Pairwise nodal-vector differences with both denominators made explicit."""
    comparisons = {}
    for small, large in combinations(sorted(responses), 2):
        a, b = responses[small], responses[large]
        difference = float(np.linalg.norm(b - a))
        small_norm, large_norm = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        comparisons[f'{small}_to_{large}'] = dict(
            absolute_nodal_l2_cm_per_s=difference,
            relative_to_small=None if small_norm <= 1e-14 else difference / small_norm,
            relative_to_large=None if large_norm <= 1e-14 else difference / large_norm)
    return comparisons


def _write(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')


@torch.no_grad()
def run(preload='results/lv_equilibrium', device='cpu', output='results/ib_box_projection/report.json',
        levels=(12, 18, 24), dt=5e-5, epsilon=1., pressure_increment=.02):
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if (not levels or list(levels) != sorted(set(levels)) or
            any(type(n) is not int or n < 12 for n in levels) or
            not isfinite(dt) or dt <= 0 or not isfinite(epsilon) or epsilon <= 0 or
            not isfinite(pressure_increment) or pressure_increment <= 0):
        raise ValueError('increasing unique box levels >=12 and positive finite parameters required')
    dilation = integer_dilation(epsilon, .5)
    output = Path(output)
    model, x0, tolerance, provenance = load_preload(preload, device=device)
    original = model.loads
    initial_force = model.force(x0, 0.)
    if torch.linalg.vector_norm(initial_force).item() > tolerance:
        raise ValueError('preload force balance failed')
    try:
        model.loads = replace(original, pressure_mmhg=original.pressure_mmhg + pressure_increment)
        incremental_force = (model.force(x0, 0.) - initial_force).detach()
    finally:
        model.loads = original
    chorin_options = SolverOptions(max_iterations=4000, recompute_every=200)
    schur_options = SolverOptions(rtol=1e-12, atol=1e-14,
                                  max_iterations=4000, recompute_every=200)
    report = dict(device=str(device), torch=torch.__version__,
        gpu=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
        units=CGS_UNITS, preload=provenance, levels=list(levels),
        velocity_spacing_cm=.5, epsilon_cm=epsilon, kernel_dilation=dilation,
        dt_s=dt, pressure_increment_mmhg=pressure_increment,
        incremental_force_norm_dyn=torch.linalg.vector_norm(incremental_force).item(),
        load_path='Q2 density: M_f H^T g / delta_V',
        chorin_solver_options=dict(rtol=chorin_options.rtol, atol=chorin_options.atol,
                                   max_iterations=chorin_options.max_iterations,
                                   recompute_every=chorin_options.recompute_every),
        schur_solver_options=dict(rtol=schur_options.rtol, atol=schur_options.atol,
                                  max_iterations=schur_options.max_iterations,
                                  recompute_every=schur_options.recompute_every),
        cases={}, box_comparisons={}, completed=False, full_cycle_ready=False,
        scope='frozen preloaded LV, zero initial fluid velocity, one incremental-force step')
    _write(output, report)
    responses = {'chorin': {}, 'schur': {}}
    support = None
    raw = dict(x_preload_cm=x0.cpu().numpy(), incremental_force_dyn=incremental_force.cpu().numpy())
    for n in levels:
        counts, lengths, origin = box_spec(n)
        mesh = create_box(counts, lengths, origin, device=device)
        flow = ChorinSolver(prepare_operators(mesh), dt=dt, options=chorin_options)
        grid = mesh.velocity_grid
        if any(abs(h - .5) > 1e-12 for h in grid.spacing):
            raise RuntimeError('fluid boxes changed velocity-node spacing')
        stencil = scaled_stencil(x0, grid, dilation)
        current_support = (mesh.velocity_coordinates[stencil.indices].cpu().numpy(),
                           stencil.weights.cpu().numpy())
        if support is None:
            support = current_support
        elif (not np.allclose(current_support[0], support[0], rtol=0, atol=1e-12) or
              not np.allclose(current_support[1], support[1], rtol=0, atol=1e-12)):
            raise RuntimeError('box expansion changed IB support coordinates or weights')
        spread = ib.spread_load(incremental_force, stencil)
        density = spread / grid.cell_volume
        result = flow.step(torch.zeros_like(mesh.velocity_coordinates), density=density)
        projected, projection_info = discrete_projection(
            flow, result.tentative_velocity, BoxMassInverse(mesh), options=schur_options)
        for name, velocity in (('chorin', result.velocity), ('schur', projected)):
            response = ib.interpolate(velocity, stencil).cpu().numpy()
            if not np.isfinite(response).all():
                raise RuntimeError(f'nonfinite {name} LV velocity for {n} cm box')
            responses[name][n] = response
            raw[f'{name}_{n}_cm_per_s'] = response
        gap = float(np.linalg.norm(responses['chorin'][n] - responses['schur'][n]))
        schur_norm = float(np.linalg.norm(responses['schur'][n]))
        report['cases'][str(n)] = dict(box_length_cm=lengths[0], origin_cm=list(origin),
            velocity_spacing_cm=grid.spacing[0], ib_support_identical=True,
            ib_support_min_wall_distance_cm=float(min(
                *(x0[:, axis].min().item() - origin[axis] - 2 * epsilon for axis in range(3)),
                *(origin[axis] + lengths[axis] - x0[:, axis].max().item() - 2 * epsilon for axis in range(3)))),
            density_resultant_dyn=flow.op.density_load(density).sum(dim=0).cpu().tolist(),
            chorin_solid_velocity_nodal_norm_cm_per_s=float(np.linalg.norm(responses['chorin'][n])),
            schur_solid_velocity_nodal_norm_cm_per_s=schur_norm,
            chorin_vs_schur_absolute_nodal_l2_cm_per_s=gap,
            chorin_vs_schur_relative_to_schur=None if schur_norm <= 1e-14 else gap / schur_norm,
            chorin_divergence_dual_l2=result.diagnostics['corrected_divergence_dual_norm'],
            schur_divergence_dual_l2=projection_info['full_divergence_dual_norm'],
            chorin_solver_max_residual_ratio=max(
                solve['residual_norm'] / solve['tolerance'] for solve in result.diagnostics['solves'].values()),
            schur_solver_residual_ratio=(projection_info['solve']['residual_norm'] /
                                         projection_info['solve']['tolerance']))
        _write(output, report)
    report['box_comparisons'] = {name: response_comparisons(values) for name, values in responses.items()}
    if len(levels) == 3:
        small, middle, large = levels
        report['adjacent_change_ratio'] = {}
        for name, comparisons in report['box_comparisons'].items():
            first = comparisons[f'{small}_to_{middle}']['absolute_nodal_l2_cm_per_s']
            second = comparisons[f'{middle}_to_{large}']['absolute_nodal_l2_cm_per_s']
            report['adjacent_change_ratio'][name] = None if first <= 1e-14 else second / first
    response_path = output.with_name('responses.npz')
    np.savez_compressed(response_path, **raw)
    report['response_file'] = str(response_path)
    report['completed'] = True
    _write(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preload', default='results/lv_equilibrium')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='results/ib_box_projection/report.json')
    parser.add_argument('--levels', type=int, nargs='+', default=[12, 18, 24])
    args = parser.parse_args()
    print(json.dumps(run(args.preload, args.device, args.output, tuple(args.levels)), indent=2, allow_nan=False))
