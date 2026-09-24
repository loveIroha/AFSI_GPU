"""Compare independent DOLFINx fluid vectors with torch CPU/CUDA actions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from afsi_torch.fluid import create_box, prepare_operators
try:
    from .reference_io import match_points
except ImportError:
    from reference_io import match_points


def compare(path, device='cpu'):
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    meta = json.loads(str(data.pop('metadata')))
    if meta.get('schema') != 1 or meta.get('producer') != 'dolfinx-ufl-fluid':
        raise ValueError('unsupported reference schema or producer')
    mesh = create_box(meta['counts'], meta['lengths'], meta['origin'], device=device)
    op = prepare_operators(mesh)
    iv = match_points(mesh.velocity_coordinates.cpu().numpy(), data['velocity_coordinates'])
    ip = match_points(mesh.pressure_coordinates.cpu().numpy(), data['pressure_coordinates'])
    if len(iv) != len(data['velocity_coordinates']) or len(ip) != len(data['pressure_coordinates']):
        raise ValueError('reference must contain exactly the expected DOFs')
    scalar = {'p', 'pressure_mass', 'pressure_stiffness', 'divergence'}
    terms = ('velocity_mass', 'velocity_stiffness', 'pressure_mass', 'pressure_stiffness',
             'divergence', 'gradient', 'divergence_transpose', 'convection', 'density_load', 'convection_tangent')
    if set(meta['terms']) != set(terms) or not meta['cases'] or len(set(meta['cases'])) != len(meta['cases']):
        raise ValueError('invalid reference terms or cases')
    for name in ('u', 'p', 'density', 'direction', *terms):
        shape = (len(meta['cases']), len(ip)) if name in scalar else (len(meta['cases']), len(iv), 3)
        if data[name].shape != shape or data[name].dtype != np.float64 or not np.isfinite(data[name]).all():
            raise ValueError(f'invalid reference array: {name}')
    reports = {}
    for i, case in enumerate(meta['cases']):
        tensor = lambda name: torch.as_tensor(data[name][i][ip if name in scalar else iv], device=device)
        u, p, density, direction = (tensor(name) for name in ('u', 'p', 'density', 'direction'))
        actions = {name: getattr(op, name)(p if name in ('pressure_mass', 'pressure_stiffness',
            'gradient', 'divergence_transpose') else density if name == 'density_load' else u)
            for name in terms if name != 'convection_tangent'}
        actions['convection_tangent'] = torch.func.jvp(op.convection, (u,), (direction,))[1]
        reports[case] = {}
        for name, actual in actions.items():
            expected = tensor(name)
            torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-10,
                                       msg=lambda msg: f'{case}/{name}: {msg}')
            reports[case][name] = (actual-expected).abs().max().item()
    return dict(status='passed', device=str(device), torch=torch.__version__,
        reference_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), reference=meta,
        cells=len(mesh.velocity_cells), comparisons=reports)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', default='validation/results/fluid_coarse.npz')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output')
    args = parser.parse_args()
    result = compare(args.reference, args.device)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
