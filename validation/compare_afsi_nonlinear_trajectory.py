"""Find the first discrepant step in a moving nonlinear AFSI/PyTorch trajectory."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from afsi_torch import boundary as bd, solid
from afsi_torch.coupling import ExplicitIBStepper
from afsi_torch.fields import prepare_reference_fields
from afsi_torch.fluid import ChorinSolver, create_box, prepare_operators
from afsi_torch.fluid.solvers import SolverOptions
from afsi_torch.materials import GuccioneParameters

if __package__:
    from .reference_io import match_points
else:
    from reference_io import match_points


STAGES = ('fluid_density', 'velocity', 'pressure', 'solid_velocity',
          'displacement', 'solid_force')


def _metric(actual, expected, *, rtol, atol):
    difference = actual-expected
    l2 = torch.linalg.vector_norm(difference).item()
    reference = torch.linalg.vector_norm(expected).item()
    tolerance = atol*expected.numel()**.5 + rtol*reference
    return dict(max_abs=difference.abs().max().item(), l2=l2,
                relative_l2=None if reference < 1e-14 else l2/reference,
                tolerance_l2=tolerance, passed=l2 <= tolerance)


def _load(time, config):
    fraction = min(time/config['ramp_time'], 1.)
    return (config['pressure_initial']+config['pressure_rise']*fraction,
            config['tension_initial']+config['tension_rise']*fraction)


def compare(path, device='cpu', *, rtol=2e-5, atol=2e-9):
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
    if rtol <= 0 or atol <= 0:
        raise ValueError('positive comparison tolerances required')
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    meta = json.loads(str(data.pop('metadata')))
    if meta.get('schema') != 1 or meta.get('producer') != 'afsi-native-nonlinear-trajectory':
        raise ValueError('expected AFSI-native nonlinear trajectory reference')
    if (meta.get('rho') != 1. or meta.get('velocity_boundary') !=
            'zero Dirichlet on all outer faces' or meta.get('ib_kernel') !=
            'Peskin four point, epsilon=velocity lattice spacing'):
        raise ValueError('unsupported AFSI-native numerical setup')
    steps = meta['steps']
    if not isinstance(steps, int) or steps < 2:
        raise ValueError('invalid trajectory step count')
    mesh = create_box(meta['counts'], meta['lengths'], meta['origin'], device=device)
    index_v = match_points(mesh.velocity_coordinates.cpu().numpy(), data['velocity_coordinates'])
    index_p = match_points(mesh.pressure_coordinates.cpu().numpy(), data['pressure_coordinates'])
    gauge = int(match_points(np.array([meta['pressure_gauge_coordinate']]),
                             mesh.pressure_coordinates.cpu().numpy())[0])
    Xn = data['X']
    nsolid, nvelocity, npressure = len(Xn), len(index_v), len(index_p)
    shapes = dict(X=(nsolid, 3), cells=(len(data['cells']), 10),
        velocity_coordinates=(nvelocity, 3), pressure_coordinates=(npressure, 3),
        positions=(steps+1, nsolid, 3), forces=(steps+1, nsolid, 3),
        fluid_densities=(steps, nvelocity, 3), velocities=(steps, nvelocity, 3),
        pressures=(steps, npressure), solid_velocities=(steps, nsolid, 3))
    for name, shape in shapes.items():
        value = data.get(name)
        dtype = np.int64 if name == 'cells' else np.float64
        if value is None or value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f'invalid native reference array {name}')
    for name, dimension in (('volume_points', 3), ('surface_points', 2)):
        points = data.get(name)
        weights = data.get(name.replace('points', 'weights'))
        if (points is None or weights is None or points.ndim != 2 or
                points.shape[1] != dimension or weights.shape != (len(points),) or
                points.dtype != np.float64 or weights.dtype != np.float64 or
                not np.isfinite(points).all() or not np.isfinite(weights).all()):
            raise ValueError(f'invalid quadrature {name}')
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
    X = tensor(Xn)
    geometry = solid.prepare_p2(X, torch.as_tensor(data['cells'], dtype=torch.int64,
        device=device), quadrature=(data['volume_points'], data['volume_weights']))
    faces = bd.extract_boundary(X, geometry.cells)
    xface = X[faces[:, :3], 0]
    high = X[:, 0].max()
    low = X[:, 0].min()
    endo = bd.prepare_surface(X, faces[torch.isclose(xface, high).all(1)],
        quadrature=(data['surface_points'], data['surface_weights']))
    base = bd.prepare_surface(X, faces[torch.isclose(xface, low).all(1)],
        quadrature=(data['surface_points'], data['surface_weights']))
    fields = prepare_reference_fields(geometry, [1., 0., 0.], [0., 1., 0.])
    parameters = GuccioneParameters()

    def force(x, time):
        pressure, tension = _load(time, meta['load'])
        current = replace(fields, tension=torch.full_like(fields.tension, tension))
        return (solid.guccione_force(x, geometry, current, parameters) +
                bd.pressure_force(x, endo, pressure) +
                bd.spring_force(x, base, meta['beta']))

    def validate(x):
        solid.validate_deformation(x, geometry)
        bd.validate_surface(x, endo)
        bd.validate_surface(x, base)

    flow = ChorinSolver(prepare_operators(mesh), dt=meta['dt'], rho=meta['rho'],
        mu=meta['mu'], pressure_dof=gauge,
        options=SolverOptions(rtol=1e-12, atol=1e-13,
                              max_iterations=4000, recompute_every=200))
    stepper = ExplicitIBStepper(flow, force, validate)
    state = stepper.initialize(X)
    first_mismatch = None
    history = []
    bootstrap = _metric(state.force, tensor(data['forces'][0]), rtol=rtol, atol=atol)
    if not bootstrap['passed']:
        first_mismatch = dict(step=0, stage='bootstrap_force')
    for index in range(steps):
        result = stepper.step(state)
        state = result.state
        actual = dict(fluid_density=result.applied_density,
            velocity=state.velocity, pressure=state.pressure,
            solid_velocity=result.solid_velocity, displacement=state.x-X,
            solid_force=state.force)
        expected = dict(fluid_density=tensor(data['fluid_densities'][index, index_v]),
            velocity=tensor(data['velocities'][index, index_v]),
            pressure=tensor(data['pressures'][index, index_p]),
            solid_velocity=tensor(data['solid_velocities'][index]),
            displacement=tensor(data['positions'][index+1]-Xn),
            solid_force=tensor(data['forces'][index+1]))
        metrics = {}
        for name in STAGES:
            # Compare displacement, not absolute coordinates: cm-scale X would
            # mask a much smaller but dynamically important motion error.
            local_atol = 2e-11 if name == 'displacement' else atol
            metrics[name] = _metric(actual[name], expected[name], rtol=rtol, atol=local_atol)
            if first_mismatch is None and not metrics[name]['passed']:
                first_mismatch = dict(step=index+1, stage=name)
        history.append(dict(step=index+1, time_s=state.time, stages=metrics,
            max_displacement_cm=torch.linalg.vector_norm(state.x-X, dim=-1).max().item(),
            force_norm_dyn=torch.linalg.vector_norm(state.force).item(),
            minimum_detF=torch.det(solid.deformation_gradient(state.x, geometry)).min().item()))
    native_motion = float(np.linalg.norm(data['positions'][-1]-Xn, axis=-1).max())
    if first_mismatch is None and native_motion <= 1e-9:
        first_mismatch = dict(step=steps, stage='insufficient_native_motion')
    return dict(completed=True, passed=first_mismatch is None,
        first_mismatch=first_mismatch, reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        device=str(device), torch=torch.__version__, native=meta,
        comparison_rtol=rtol, comparison_atol=atol,
        native_max_displacement_cm=native_motion, bootstrap_force=bootstrap, history=history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', default='results/afsi_nonlinear_trajectory/reference.npz')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='results/afsi_nonlinear_trajectory/report.json')
    args = parser.parse_args()
    report = compare(args.reference, args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report['passed']:
        raise SystemExit(f"first mismatched step/stage: {report['first_mismatch']}")

