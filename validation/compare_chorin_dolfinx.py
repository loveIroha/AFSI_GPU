"""Compare every Chorin substep to independently solved DOLFINx fields."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from afsi_torch.fluid import create_box, prepare_operators, ChorinSolver
try:
    from .reference_io import match_points
except ImportError:
    from reference_io import match_points


def compare(path, device='cpu'):
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    meta = json.loads(str(data.pop('metadata')))
    if meta.get('schema') != 1 or meta.get('producer') != 'dolfinx-ufl-chorin':
        raise ValueError('unsupported Chorin reference')
    mesh = create_box(meta['counts'], meta['lengths'], meta['origin'], device=device)
    iv = match_points(mesh.velocity_coordinates.cpu().numpy(), data['velocity_coordinates'])
    ip = match_points(mesh.pressure_coordinates.cpu().numpy(), data['pressure_coordinates'])
    if len(iv) != len(data['velocity_coordinates']) or len(ip) != len(data['pressure_coordinates']):
        raise ValueError('unexpected reference DOFs')
    gauge = int(match_points(np.array([meta['gauge_coordinate']]), mesh.pressure_coordinates.cpu().numpy())[0])
    solver = ChorinSolver(prepare_operators(mesh), dt=meta['dt'], rho=meta['rho'], mu=meta['mu'],
                          pressure_dof=gauge, pressure_value=meta['pressure_value'])
    K, T, N, P = len(meta['cases']), meta['steps'], len(iv), len(ip)
    shapes = dict(initial_velocity=(K, N, 3), density=(K, N, 3), boundary_values=(K, N, 3),
                  tentative_velocity=(K, T, N, 3), velocity=(K, T, N, 3), pressure=(K, T, P),
                  divergence_l2=(K, T, 2))
    if not K or T < 1 or len(set(meta['cases'])) != K:
        raise ValueError('invalid reference cases/steps')
    for name, shape in shapes.items():
        if data[name].shape != shape or data[name].dtype != np.float64 or not np.isfinite(data[name]).all():
            raise ValueError(f'invalid reference array {name}')
    reports = {}
    for case_index, case in enumerate(meta['cases']):
        tensor = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
        u, f, bc = (tensor(data[name][case_index][iv]) for name in ('initial_velocity', 'density', 'boundary_values'))
        reports[case] = []
        p = None
        for step in range(T):
            result = solver.step(u, density=f, boundary_values=bc, pressure_initial=p)
            errors = {}
            for name in ('tentative_velocity', 'pressure', 'velocity'):
                actual = getattr(result, name)
                expected = tensor(data[name][case_index, step][ip if name == 'pressure' else iv])
                torch.testing.assert_close(actual, expected, atol=3e-9, rtol=3e-8,
                    msg=lambda msg: f'{case}/{step}/{name}: {msg}')
                errors[name] = (actual-expected).abs().max().item()
            divs = tensor([result.diagnostics['tentative_divergence_l2'], result.diagnostics['corrected_divergence_l2']])
            torch.testing.assert_close(divs, tensor(data['divergence_l2'][case_index, step]), atol=3e-9, rtol=3e-8)
            reports[case].append(dict(max_abs=errors, **result.diagnostics))
            u, p = result.velocity, result.pressure
    return dict(status='passed', device=str(device), torch=torch.__version__, reference=meta,
        reference_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), comparisons=reports)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', default='validation/results/chorin_coarse.npz')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output')
    args = parser.parse_args()
    report = compare(args.reference, args.device)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
