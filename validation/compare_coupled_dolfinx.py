"""Compare the entire evolving coupled state with the independent CPU oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from afsi_torch import solid, boundary as bd
from afsi_torch.materials import active_pk1
from afsi_torch.fluid import create_box, prepare_operators, ChorinSolver
from afsi_torch.coupling import ExplicitIBStepper
try:
    from .reference_io import match_points
    from .compare_dolfinx import select_tagged_faces
except ImportError:
    from reference_io import match_points
    from compare_dolfinx import select_tagged_faces


def compare(path, device='cpu'):
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    meta = json.loads(str(data.pop('metadata')))
    if meta.get('schema') != 1 or meta.get('producer') != 'dolfinx-numpy-ib-coupled':
        raise ValueError('unsupported coupled reference')
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
    X = tensor(data['X'])
    cells = torch.as_tensor(data['cells'], dtype=torch.int64, device=device)
    geometry = solid.prepare_p2(X, cells, quadrature=(data['volume_points'], data['volume_weights']))
    regions = select_tagged_faces(bd.extract_boundary(X, cells).cpu().numpy(), data['tagged_vertices'], data['facet_tags'])
    surfaces = {tag: bd.prepare_surface(X, torch.as_tensor(faces, device=device),
        quadrature=(data['surface_points'], data['surface_weights'])) for tag, faces in regions.items()}
    def validate(x):
        solid.validate_deformation(x, geometry)
        for surface in surfaces.values():
            bd.validate_surface(x, surface)
    def force(x, time):
        F = solid.deformation_gradient(x, geometry)
        return (solid.stress_force(x, geometry, meta['solid_mu'], meta['solid_lam'])+
            solid.assemble_pk1(active_pk1(F, X.new_tensor([1., 0., 0.]), .1+time), geometry)+
            bd.pressure_force(x, surfaces[1], .2+2*time)+bd.spring_force(x, surfaces[2], meta['beta']))
    mesh = create_box(meta['counts'], meta['lengths'], meta['origin'], device=device)
    iv = match_points(mesh.velocity_coordinates.cpu().numpy(), data['velocity_coordinates'])
    ip = match_points(mesh.pressure_coordinates.cpu().numpy(), data['pressure_coordinates'])
    gauge = int(match_points(np.array([meta['pressure_gauge']]), mesh.pressure_coordinates.cpu().numpy())[0])
    T, N = meta['steps'], len(X)
    shapes = dict(initial_x=(N, 3), x=(T, N, 3), force=(T, N, 3), velocity=(T, len(iv), 3),
                  pressure=(T, len(ip)), applied_density=(T, len(iv), 3))
    if T < 1 or len(iv) != len(data['velocity_coordinates']) or len(ip) != len(data['pressure_coordinates']):
        raise ValueError('invalid reference counts')
    for name, shape in shapes.items():
        if data[name].shape != shape or data[name].dtype != np.float64 or not np.isfinite(data[name]).all():
            raise ValueError(f'invalid reference array {name}')
    flow = ChorinSolver(prepare_operators(mesh), dt=meta['dt'], rho=meta['rho'], mu=meta['mu'], pressure_dof=gauge)
    driver = ExplicitIBStepper(flow, force, validate)
    state = driver.initialize(tensor(data['initial_x']))
    reports = []
    for step in range(T):
        result = driver.step(state)
        state = result.state
        errors = {}
        for name in ('x', 'force', 'velocity', 'pressure', 'applied_density'):
            actual = result.applied_density if name == 'applied_density' else getattr(state, name)
            expected = data[name][step]
            if name in ('velocity', 'applied_density'):
                expected = expected[iv]
            elif name == 'pressure':
                expected = expected[ip]
            expected = tensor(expected)
            torch.testing.assert_close(actual, expected, atol=3e-9, rtol=3e-8,
                msg=lambda msg: f'step {step}/{name}: {msg}')
            errors[name] = (actual-expected).abs().max().item()
        reports.append(dict(max_abs=errors, **result.diagnostics))
    return dict(status='passed', device=str(device), torch=torch.__version__, reference=meta,
        reference_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), comparisons=reports)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', default='validation/results/coupled_patch.npz')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output')
    args = parser.parse_args()
    report = compare(args.reference, args.device)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
