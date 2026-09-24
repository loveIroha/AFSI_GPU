"""Independent scalar NumPy full-grid reference for four-point IB transfer.

The oracle checks every grid node: it does not reuse Torch's stencil indices,
kernel implementation, gather/scatter or derivatives. No original afsi C++
binary is executed; this validates the same mathematical kernel and scaling.
"""
import argparse
import json
from math import sqrt
from pathlib import Path
import numpy as np
import torch
from afsi_torch import ib


def scalar_phi(r):
    r = abs(float(r))
    if r >= 2:
        return 0.
    if r >= 1:
        return (5-2*r-sqrt(-7+12*r-4*r*r))/8
    return (3-2*r+sqrt(1+4*r-4*r*r))/8


def compare(device='cpu'):
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    shape, spacing, origin = (10, 9, 8), (.2, .3, .4), (-.7, 1.1, -.3)
    X = np.array([[origin[0]+i*spacing[0], origin[1]+j*spacing[1], origin[2]+k*spacing[2]]
                  for k in range(shape[2]) for j in range(shape[1]) for i in range(shape[0])])
    s = np.array([[2.17, 3.41, 2.53], [5.32, 4.29, 3.78], [6., 2., 5.], [2.17, 3.41, 2.53]])
    x = np.array(origin)+s*np.array(spacing)
    H = np.empty((len(x), len(X)))
    for a, position in enumerate(x):
        for i, node in enumerate(X):
            d = (position-node)/np.array(spacing)
            H[a, i] = scalar_phi(d[0])*scalar_phi(d[1])*scalar_phi(d[2])
    rng = np.random.default_rng(260924)
    u, g = rng.normal(size=X.shape), rng.normal(size=x.shape)
    volume = float(np.prod(spacing))
    expected_U, expected_f = H@u, H.T@g/volume
    grid = ib.UniformGrid(shape, spacing, origin)
    tensor = lambda a: torch.tensor(a, dtype=torch.float64, device=device)
    stencil = ib.prepare_stencil(tensor(x), grid)
    actual_U = ib.interpolate(tensor(u), stencil).cpu().numpy()
    actual_f = ib.spread_density(tensor(g), stencil).cpu().numpy()
    np.testing.assert_allclose(actual_U, expected_U, atol=1e-13, rtol=1e-12)
    np.testing.assert_allclose(actual_f, expected_f, atol=1e-12, rtol=1e-11)
    np.testing.assert_allclose(actual_f.sum(0)*volume, g.sum(0), atol=1e-12, rtol=1e-12)
    power_error = abs(np.sum(u*actual_f)*volume-np.sum(actual_U*g))
    np.testing.assert_allclose(power_error, 0, atol=1e-12)
    return dict(status='passed', device=str(device), torch=torch.__version__,
                reference='scalar NumPy full-grid kernel evaluation',
                velocity_max_abs=float(np.max(np.abs(actual_U-expected_U))),
                force_density_max_abs=float(np.max(np.abs(actual_f-expected_f))),
                power_error=float(power_error))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output')
    args = parser.parse_args()
    report = json.dumps(compare(args.device), indent=2)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report+'\n', encoding='utf-8')
    print(report)
