"""Attribute the frozen-LV Chorin/Schur gap to projection components.

Both corrected velocities start from one identical tentative fluid field.
The intermediate Laplacian reconstruction changes only the algebraic form of
the Chorin correction; the Schur reference changes the pressure operator.
"""
import argparse
from dataclasses import replace
import json
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
    from .compare_ib_box_projection import box_spec
    from .diagnose_ib import BoxMassInverse, discrete_projection, scaled_stencil
    from .verify_ib_weak import integer_dilation
else:
    from compare_ib_box_projection import box_spec
    from diagnose_ib import BoxMassInverse, discrete_projection, scaled_stencil
    from verify_ib_weak import integer_dilation


def relative_norm(actual, reference):
    denominator = torch.linalg.vector_norm(reference).item()
    return None if denominator <= 1e-14 else (
        torch.linalg.vector_norm(actual - reference).item() / denominator)


@torch.no_grad()
def reconstruct_laplacian_projection(flow, result, mass_inverse):
    """Use Chorin's pressure multiplier with the exact free Q2 mass inverse.

    The Chorin pressure equation is K p = -(rho/dt) D u*. Thus
    lambda_K = -(dt/rho) p solves K lambda_K = D u*. For zero outer velocity,
    the free components of G p and -D.T p coincide by integration by parts.
    """
    op = flow.op
    star, pressure = result.tentative_velocity, result.pressure
    multiplier = -(flow.dt / flow.rho) * pressure
    reconstructed = star - mass_inverse(op.divergence_transpose(multiplier))
    free = ~flow.velocity_fixed
    gradient_identity = (op.gradient(pressure) + op.divergence_transpose(pressure))[free]
    transpose_norm = torch.linalg.vector_norm(op.divergence_transpose(pressure)[free]).item()
    gradient_relative = (torch.linalg.vector_norm(gradient_identity).item() /
                         max(transpose_norm, 1e-30))
    divergence_star = op.divergence(star)
    k_residual = (op.pressure_stiffness(multiplier) - divergence_star).masked_fill(
        flow.pressure_fixed, 0)
    k_rhs_norm = torch.linalg.vector_norm(divergence_star.masked_fill(flow.pressure_fixed, 0)).item()
    schur_rhs_error = op.divergence(mass_inverse(op.divergence_transpose(multiplier))) - divergence_star
    diagnostics = dict(
        free_gradient_plus_divergence_transpose_relative=gradient_relative,
        laplacian_multiplier_residual_relative=(torch.linalg.vector_norm(k_residual).item() /
                                                max(k_rhs_norm, 1e-30)),
        laplacian_multiplier_schur_residual_relative=(
            torch.linalg.vector_norm(schur_rhs_error).item() /
            max(torch.linalg.vector_norm(divergence_star).item(), 1e-30)),
        chorin_vs_reconstructed_full_velocity_relative=relative_norm(
            result.velocity, reconstructed),
        reconstructed_divergence_dual_l2=torch.linalg.vector_norm(op.divergence(reconstructed)).item())
    return reconstructed, diagnostics


def _write(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')


@torch.no_grad()
def run(preload='results/lv_equilibrium', device='cpu',
        output='results/projection_gap/report.json', levels=(18, 24),
        dt=5e-5, epsilon=1., pressure_increment=.02):
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if (not levels or list(levels) != sorted(set(levels)) or
            any(type(n) is not int or n < 12 for n in levels) or
            not isfinite(dt) or dt <= 0 or not isfinite(epsilon) or epsilon <= 0 or
            not isfinite(pressure_increment) or pressure_increment <= 0):
        raise ValueError('increasing unique box levels >=12 and positive finite parameters required')
    dilation = integer_dilation(epsilon, .5)
    model, x0, tolerance, provenance = load_preload(preload, device=device)
    original = model.loads
    baseline_force = model.force(x0, 0.)
    if torch.linalg.vector_norm(baseline_force).item() > tolerance:
        raise ValueError('preload force balance failed')
    try:
        model.loads = replace(original, pressure_mmhg=original.pressure_mmhg + pressure_increment)
        force = (model.force(x0, 0.) - baseline_force).detach()
    finally:
        model.loads = original
    chorin_options = SolverOptions(max_iterations=4000, recompute_every=200)
    schur_options = SolverOptions(rtol=1e-12, atol=1e-14,
                                  max_iterations=4000, recompute_every=200)
    output = Path(output)
    report = dict(device=str(device), torch=torch.__version__,
        gpu=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
        units=CGS_UNITS, preload=provenance, levels=list(levels),
        velocity_spacing_cm=.5, epsilon_cm=epsilon, dt_s=dt,
        pressure_increment_mmhg=pressure_increment,
        incremental_force_norm_dyn=torch.linalg.vector_norm(force).item(),
        kernel_dilation=dilation, load_path='Q2 density: M_f H^T g / delta_V',
        chorin_solver_options=vars(chorin_options), schur_solver_options=vars(schur_options),
        cases={}, completed=False, full_cycle_ready=False,
        scope='frozen preloaded LV, zero initial fluid velocity, one pressure-projection attribution')
    _write(output, report)
    support = None
    raw = dict(x_preload_cm=x0.cpu().numpy(), incremental_force_dyn=force.cpu().numpy())
    for n in levels:
        counts, lengths, origin = box_spec(n)
        mesh = create_box(counts, lengths, origin, device=device)
        flow = ChorinSolver(prepare_operators(mesh), dt=dt, options=chorin_options)
        grid = mesh.velocity_grid
        stencil = scaled_stencil(x0, grid, dilation)
        current_support = (mesh.velocity_coordinates[stencil.indices].cpu().numpy(),
                           stencil.weights.cpu().numpy())
        if support is None:
            support = current_support
        elif (not np.allclose(current_support[0], support[0], atol=1e-12, rtol=0) or
              not np.allclose(current_support[1], support[1], atol=1e-12, rtol=0)):
            raise RuntimeError('box expansion changed IB support coordinates or weights')
        density = ib.spread_load(force, stencil) / grid.cell_volume
        result = flow.step(torch.zeros_like(mesh.velocity_coordinates), density=density)
        inverse = BoxMassInverse(mesh)
        reconstructed, attribution = reconstruct_laplacian_projection(flow, result, inverse)
        schur, schur_info = discrete_projection(
            flow, result.tentative_velocity, inverse, options=schur_options)
        if (attribution['free_gradient_plus_divergence_transpose_relative'] > 1e-9 or
                attribution['chorin_vs_reconstructed_full_velocity_relative'] > 1e-6):
            raise RuntimeError('Laplacian reconstruction does not reproduce Chorin; inspect BCs and mass solve')
        responses = {name: ib.interpolate(velocity, stencil)
                     for name, velocity in [('chorin', result.velocity),
                                            ('laplacian_reconstructed', reconstructed),
                                            ('schur', schur)]}
        for name, response in responses.items():
            raw[f'{name}_{n}_cm_per_s'] = response.cpu().numpy()
        schur_norm = torch.linalg.vector_norm(responses['schur']).item()
        gap = lambda name: torch.linalg.vector_norm(
            responses[name] - responses['schur']).item() / max(schur_norm, 1e-30)
        report['cases'][str(n)] = dict(box_length_cm=lengths[0],
            velocity_spacing_cm=grid.spacing[0], ib_support_identical=True,
            force_resultant_dyn=flow.op.density_load(density).sum(dim=0).cpu().tolist(),
            tentative_divergence_dual_l2=result.diagnostics['tentative_divergence_dual_norm'],
            chorin_divergence_dual_l2=result.diagnostics['corrected_divergence_dual_norm'],
            schur_divergence_dual_l2=schur_info['full_divergence_dual_norm'],
            chorin_solid_velocity_nodal_norm_cm_per_s=torch.linalg.vector_norm(responses['chorin']).item(),
            schur_solid_velocity_nodal_norm_cm_per_s=schur_norm,
            chorin_vs_schur_solid_velocity_relative=gap('chorin'),
            reconstructed_vs_schur_solid_velocity_relative=gap('laplacian_reconstructed'),
            chorin_vs_reconstructed_solid_velocity_relative=relative_norm(
                responses['chorin'], responses['laplacian_reconstructed']),
            chorin_solver_max_residual_ratio=max(
                info['residual_norm'] / info['tolerance'] for info in result.diagnostics['solves'].values()),
            schur_solver_residual_ratio=(schur_info['solve']['residual_norm'] /
                                         schur_info['solve']['tolerance']),
            **attribution)
        _write(output, report)
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
    parser.add_argument('--output', default='results/projection_gap/report.json')
    parser.add_argument('--levels', type=int, nargs='+', default=[18, 24])
    args = parser.parse_args()
    print(json.dumps(run(args.preload, args.device, args.output, tuple(args.levels)), indent=2, allow_nan=False))
