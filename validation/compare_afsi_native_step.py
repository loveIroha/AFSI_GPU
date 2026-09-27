"""Locate the first mismatch in one AFSI-native versus PyTorch IB/Chorin step."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from afsi_torch import ib
from afsi_torch.fluid import ChorinSolver, create_box, prepare_operators
from afsi_torch.fluid.solvers import SolverOptions

if __package__:
    from .reference_io import match_points
else:
    from reference_io import match_points


STAGES = ('fluid_density', 'fluid_weak_load', 'tentative_velocity', 'pressure', 'velocity',
          'solid_velocity', 'solid_next')


def _metrics(actual, expected, *, rtol, atol):
    difference = actual - expected
    absolute = torch.linalg.vector_norm(difference).item()
    reference = torch.linalg.vector_norm(expected).item()
    maximum = difference.abs().max().item()
    return dict(max_abs=maximum, l2=absolute,
                relative_l2=None if reference < 1e-14 else absolute / reference,
                tolerance_l2=atol * expected.numel()**.5 + rtol * reference,
                passed=absolute <= atol * expected.numel()**.5 + rtol * reference)


def compare(path, device='cpu', *, rtol=2e-7, atol=2e-9):
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
    if rtol <= 0 or atol <= 0:
        raise ValueError('positive comparison tolerances required')
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    meta = json.loads(str(data.pop('metadata')))
    if meta.get('schema') != 1 or meta.get('producer') != 'afsi-native-ib-chorin':
        raise ValueError('expected AFSI-native IB/Chorin reference')
    if (meta.get('rho') != 1. or meta.get('velocity_boundary') !=
            'zero Dirichlet on all outer faces' or meta.get('ib_kernel') !=
            'Peskin four point, epsilon=velocity lattice spacing'):
        raise ValueError('unsupported AFSI-native numerical setup')
    mesh = create_box(meta['counts'], meta['lengths'], meta['origin'], device=device)
    index_v = match_points(mesh.velocity_coordinates.cpu().numpy(), data['velocity_coordinates'])
    index_p = match_points(mesh.pressure_coordinates.cpu().numpy(), data['pressure_coordinates'])
    gauge = int(match_points(np.array([meta['pressure_gauge_coordinate']]),
                             mesh.pressure_coordinates.cpu().numpy())[0])
    n_velocity = len(mesh.velocity_coordinates)
    n_pressure = len(mesh.pressure_coordinates)
    n_solid = len(data['solid_coordinates'])
    shapes = dict(velocity_coordinates=(n_velocity, 3),
        pressure_coordinates=(n_pressure, 3), solid_coordinates=(n_solid, 3),
        solid_force=(n_solid, 3), fluid_density=(n_velocity, 3),
        fluid_weak_load=(n_velocity, 3),
        tentative_velocity=(n_velocity, 3), pressure=(n_pressure,),
        velocity=(n_velocity, 3), solid_velocity=(n_solid, 3),
        solid_next=(n_solid, 3))
    for name, shape in shapes.items():
        if (name not in data or data[name].shape != shape or
                data[name].dtype != np.float64 or not np.isfinite(data[name]).all()):
            raise ValueError(f'invalid native reference array {name}')
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
    x = tensor(data['solid_coordinates'])
    force = tensor(data['solid_force'])
    stencil = ib.prepare_stencil(x, mesh.velocity_grid)
    density = ib.spread_density(force, stencil)
    flow = ChorinSolver(prepare_operators(mesh), dt=meta['dt'], rho=meta['rho'],
        mu=meta['mu'], pressure_dof=gauge,
        options=SolverOptions(rtol=1e-12, atol=1e-13,
                              max_iterations=4000, recompute_every=200))
    result = flow.step(torch.zeros_like(mesh.velocity_coordinates), density=density)
    vs = ib.interpolate(result.velocity, stencil)
    actual = dict(fluid_density=density,
                  fluid_weak_load=flow.op.density_load(density),
                  tentative_velocity=result.tentative_velocity,
                  pressure=result.pressure, velocity=result.velocity,
                  solid_velocity=vs, solid_next=x+meta['dt']*vs)
    metrics = {}
    for name in STAGES:
        expected = tensor(data[name][index_p if name == 'pressure' else
                                      index_v if name in STAGES[:5] else slice(None)])
        metrics[name] = _metrics(actual[name], expected, rtol=rtol, atol=atol)
    first = next((name for name in STAGES if not metrics[name]['passed']), None)
    return dict(completed=True, passed=first is None, first_mismatch=first,
        reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        device=str(device), torch=torch.__version__, native=meta,
        comparison_rtol=rtol, comparison_atol=atol, stages=metrics,
        flow_diagnostics=result.diagnostics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', default='results/afsi_native_step/reference.npz')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='results/afsi_native_step/report.json')
    args = parser.parse_args()
    report = compare(args.reference, args.device)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report['passed']:
        raise SystemExit(f"first mismatched stage: {report['first_mismatch']}")
