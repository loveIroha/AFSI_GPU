"""Independent weak-load quadrature and fixed-spacing fluid-box controls.

The reference integrates a fixed 1 cm tensor-product Peskin force density
against Q2 Lagrange basis functions using NumPy only. It does not reuse the
torch IB stencil or fluid mass assembly. This checks a *regularized* weak load,
not convergence to an unregularized point force or the original AFSI model.
"""
import argparse
from dataclasses import replace
import json
from math import isfinite, sqrt
from pathlib import Path

import numpy as np
import torch

from afsi_torch import ib
from afsi_torch.fluid import ChorinSolver, create_box, prepare_operators
from afsi_torch.preload import load_preload
if __package__:
    from .diagnose_ib import BoxMassInverse, discrete_projection, scaled_stencil
else:
    from diagnose_ib import BoxMassInverse, discrete_projection, scaled_stencil


def scalar_peskin4(distance):
    """Scalar reference implementation, separate from the torch IB kernel."""
    r = abs(float(distance))
    if r < 1.:
        return (3. - 2.*r + sqrt(1. + 4.*r - 4.*r*r)) / 8.
    if r < 2.:
        return (5. - 2.*r - sqrt(-7. + 12.*r - 4.*r*r)) / 8.
    return 0.


def axis_weak_load(n, length, origin, point, epsilon, order=32):
    """Integrate phi((s-point)/epsilon)/epsilon times each 1D Q2 basis."""
    if (type(n) is not int or n < 1 or not all(isfinite(x) for x in
            (length, origin, point, epsilon)) or length <= 0 or epsilon <= 0 or
            point-2*epsilon <= origin or point+2*epsilon >= origin+length or
            type(order) is not int or order < 4):
        raise ValueError('positive lengths, interior full support and quadrature order>=4 required')
    nodes, weights = np.polynomial.legendre.leggauss(order)
    result = np.zeros(2*n+1, dtype=np.float64)
    h = length/n
    breaks = [point-2*epsilon, point-epsilon, point, point+epsilon, point+2*epsilon]
    for element in range(n):
        left = origin+element*h
        right = left+h
        if right <= breaks[0] or left >= breaks[-1]:
            continue
        cuts = [left, *[b for b in breaks if left < b < right], right]
        for lo, hi in zip(cuts, cuts[1:]):
            x = (lo+hi)/2 + (hi-lo)*nodes/2
            s = (x-left)/h
            basis = np.stack((2*(s-.5)*(s-1), 4*s*(1-s), 2*s*(s-.5)), axis=1)
            delta = np.array([scalar_peskin4((v-point)/epsilon)/epsilon for v in x])
            result[2*element:2*element+3] += (hi-lo)/2 * (weights*delta) @ basis
    return result


def reference_weak_load(counts, lengths, origin, point, force, epsilon=1.):
    """Independent tensor-product Q2 weak RHS in x-fast DOF ordering."""
    axes = [axis_weak_load(n, length, start, position, epsilon)
            for n, length, start, position in zip(counts, lengths, origin, point)]
    return np.einsum('k,j,i,c->kjic', axes[2], axes[1], axes[0], np.asarray(force, dtype=np.float64)).reshape(-1, 3)


def relative_norm(actual, reference):
    denominator = np.linalg.norm(reference)
    return None if denominator < 1e-14 else float(np.linalg.norm(actual-reference)/denominator)


def integer_dilation(epsilon, spacing):
    ratio = epsilon/spacing
    if abs(ratio-round(ratio)) > 1e-12 or round(ratio) < 1:
        raise ValueError('physical kernel width requires integer dilation on every probed grid')
    return round(ratio)


def _write(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')


@torch.no_grad()
def run(device='cpu', output='results/ib_weak_reference/report.json', levels=(6, 12, 18, 24),
        point=(.37, -.41, -1.23), force=(1., -.7, .4), epsilon=1., dt=5e-5,
        preload=None, pressure_increment=.02):
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if (not levels or list(levels) != sorted(set(levels)) or
        any(type(n) is not int or n < 6 or n % 6 for n in levels) or
        len(point) != 3 or len(force) != 3 or not all(isfinite(v) for v in (*point, *force)) or
        not isfinite(epsilon) or epsilon <= 0 or not isfinite(dt) or dt <= 0 or
        not isfinite(pressure_increment) or pressure_increment <= 0):
        raise ValueError('ordered fluid levels divisible by 6, finite point/force and positive epsilon/dt required')
    for n in levels:
        integer_dilation(epsilon, 6./n)
    integer_dilation(epsilon, .5)
    report = dict(device=str(device), torch=torch.__version__,
        gpu=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
        point_cm=list(point), force_dyn=list(force), epsilon_cm=epsilon, dt_s=dt,
        levels=list(levels), weak_loads={}, box_control={}, lv_box_control={}, completed=False,
        full_cycle_ready=False,
        scope='independent Q2 weak-load reference and fixed-spacing box/projection controls; no coupled trajectory')
    _write(output, report)
    x = torch.tensor([point], dtype=torch.float64, device=device)
    g = torch.tensor([force], dtype=torch.float64, device=device)
    lengths = (12., 12., 12.)
    origin = (-6., -6., -8.)
    for n in levels:
        mesh = create_box((n,)*3, lengths, origin, device=device)
        op = prepare_operators(mesh)
        h = mesh.velocity_grid.spacing[0]
        stencil = scaled_stencil(x, mesh.velocity_grid, integer_dilation(epsilon, h))
        direct = ib.spread_load(g, stencil)
        density = op.density_load(direct/mesh.velocity_grid.cell_volume)
        weak = reference_weak_load(mesh.counts, mesh.lengths, mesh.origin, point, force, epsilon)
        direct_np = direct.cpu().numpy()
        density_np = density.cpu().numpy()
        coordinates = mesh.velocity_coordinates.cpu().numpy()
        report['weak_loads'][str(n)] = dict(fluid_cells=n, velocity_spacing_cm=h,
            reference_resultant_dyn=weak.sum(axis=0).tolist(),
            reference_force_moment_dyn_cm=(weak*coordinates).sum(axis=0).tolist(),
            direct_resultant_dyn=direct_np.sum(axis=0).tolist(),
            density_resultant_dyn=density_np.sum(axis=0).tolist(),
            direct_relative_to_quadrature=relative_norm(direct_np, weak),
            density_relative_to_quadrature=relative_norm(density_np, weak),
            direct_vs_density_relative_to_quadrature=relative_norm(direct_np, density_np))
        _write(output, report)

    # Same h=0.5 cm and same lattice phase. Only outer walls move by 3 cm.
    controls = ((12, (12.,)*3, (-6., -6., -8.)),
                (18, (18.,)*3, (-9., -9., -11.)))
    responses = {}
    support = None
    for n, lengths, origin in controls:
        mesh = create_box((n,)*3, lengths, origin, device=device)
        flow = ChorinSolver(prepare_operators(mesh), dt=dt)
        stencil = scaled_stencil(x, mesh.velocity_grid, integer_dilation(epsilon, mesh.velocity_grid.spacing[0]))
        current_support = (mesh.velocity_coordinates[stencil.indices].cpu().numpy(),
                           stencil.weights.cpu().numpy())
        if support is None:
            support = current_support
        elif (not np.allclose(current_support[0], support[0], rtol=0, atol=1e-12) or
              not np.allclose(current_support[1], support[1], rtol=0, atol=1e-12)):
            raise RuntimeError('box control changed lattice phase or IB support')
        dual = ib.spread_load(g, stencil)
        result = flow.step(torch.zeros_like(mesh.velocity_coordinates), nodal_load=dual)
        chorin = ib.interpolate(result.velocity, stencil)
        schur, schur_info = discrete_projection(flow, result.tentative_velocity, BoxMassInverse(mesh))
        projected = ib.interpolate(schur, stencil)
        responses[n] = chorin.cpu().numpy().reshape(3)
        report['box_control'][str(n)] = dict(fluid_cells=n, box_length_cm=lengths[0],
            velocity_spacing_cm=mesh.velocity_grid.spacing[0],
            chorin_solid_velocity_cm_per_s=responses[n].tolist(),
            schur_solid_velocity_cm_per_s=projected.cpu().numpy().reshape(3).tolist(),
            chorin_vs_schur_solid_velocity_relative=relative_norm(
                chorin.cpu().numpy(), projected.cpu().numpy()),
            chorin_divergence_dual_l2=result.diagnostics['corrected_divergence_dual_norm'],
            schur_divergence_dual_l2=schur_info['full_divergence_dual_norm'],
            chorin_solver_max_residual_ratio=max(v['residual_norm']/v['tolerance']
                for v in result.diagnostics['solves'].values()),
            schur_solver_residual_ratio=schur_info['solve']['residual_norm']/schur_info['solve']['tolerance'])
        _write(output, report)
    report['box_control']['large_vs_small_chorin_relative'] = relative_norm(responses[18], responses[12])
    if preload is not None:
        model, x0, tolerance, provenance = load_preload(preload, device=device)
        initial_force = model.force(x0, 0.)
        if torch.linalg.vector_norm(initial_force).item() > tolerance:
            raise ValueError('preload force balance failed')
        baseline = model.loads
        try:
            model.loads = replace(baseline, pressure_mmhg=baseline.pressure_mmhg+pressure_increment)
            incremental_force = (model.force(x0, 0.)-initial_force).detach()
        finally:
            model.loads = baseline
        report['lv_box_control']['preload'] = provenance
        report['lv_box_control']['incremental_force_norm_dyn'] = torch.linalg.vector_norm(incremental_force).item()
        boxes = {}
        support = None
        for n, lengths, origin in controls:
            mesh = create_box((n,)*3, lengths, origin, device=device)
            flow = ChorinSolver(prepare_operators(mesh), dt=dt)
            stencil = scaled_stencil(x0, mesh.velocity_grid, integer_dilation(epsilon, mesh.velocity_grid.spacing[0]))
            current_support = (mesh.velocity_coordinates[stencil.indices].cpu().numpy(),
                               stencil.weights.cpu().numpy())
            if support is None:
                support = current_support
            elif (not np.allclose(current_support[0], support[0], rtol=0, atol=1e-12) or
                  not np.allclose(current_support[1], support[1], rtol=0, atol=1e-12)):
                raise RuntimeError('LV box control changed lattice phase or IB support')
            boxes[n] = mesh, flow, stencil
        for path in ('density', 'dual'):
            velocities = {}
            report['lv_box_control'][path] = {}
            for n, lengths, origin in controls:
                mesh, flow, stencil = boxes[n]
                dual = ib.spread_load(incremental_force, stencil)
                load = ({'density': dual/mesh.velocity_grid.cell_volume} if path == 'density'
                        else {'nodal_load': dual})
                flow_result = flow.step(torch.zeros_like(mesh.velocity_coordinates), **load)
                chorin = ib.interpolate(flow_result.velocity, stencil).cpu().numpy()
                schur, schur_info = discrete_projection(flow, flow_result.tentative_velocity, BoxMassInverse(mesh))
                schur_velocity = ib.interpolate(schur, stencil).cpu().numpy()
                velocities[n] = chorin
                report['lv_box_control'][path][str(n)] = dict(fluid_cells=n, box_length_cm=lengths[0],
                    velocity_spacing_cm=mesh.velocity_grid.spacing[0],
                    chorin_solid_velocity_nodal_norm_cm_per_s=float(np.linalg.norm(chorin)),
                    schur_solid_velocity_nodal_norm_cm_per_s=float(np.linalg.norm(schur_velocity)),
                    chorin_vs_schur_relative=relative_norm(chorin, schur_velocity),
                    chorin_divergence_dual_l2=flow_result.diagnostics['corrected_divergence_dual_norm'],
                    schur_divergence_dual_l2=schur_info['full_divergence_dual_norm'],
                    chorin_solver_max_residual_ratio=max(v['residual_norm']/v['tolerance']
                        for v in flow_result.diagnostics['solves'].values()),
                    schur_solver_residual_ratio=schur_info['solve']['residual_norm']/schur_info['solve']['tolerance'])
                _write(output, report)
            report['lv_box_control'][path]['large_vs_small_chorin_relative'] = relative_norm(
                velocities[18], velocities[12])
    report['completed'] = True
    _write(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='results/ib_weak_reference/report.json')
    parser.add_argument('--levels', type=int, nargs='+', default=[6,12,18,24])
    parser.add_argument('--preload', default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.device, args.output, tuple(args.levels), preload=args.preload), indent=2, allow_nan=False))
