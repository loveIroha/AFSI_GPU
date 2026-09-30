"""Run the generated AFSI demo_337 left ventricle with MAC fluid for 2 s.

The fluid is PyTorch MAC finite differences with geometric multigrid;
the solid is the existing nonlinear finite-element model. Run from the
repository root after installing the package in the active environment.
"""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples.lv_mac import run  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', help='fresh-run output directory')
    parser.add_argument('--resume', help='MAC checkpoint.npz to resume')
    parser.add_argument('--end-time', type=float, default=2.,
                        help='final simulation time in seconds; default: 2')
    parser.add_argument('--dt', type=float, help='fresh run only; default: 5e-5 s')
    parser.add_argument('--mesh-size', type=float,
                        help='fresh solid mesh only; default: 0.1 cm')
    parser.add_argument('--fluid-cells', type=int,
                        help='fresh MAC grid cells per axis; default: 64')
    parser.add_argument('--interaction-degree', type=int,
                        help='fresh quadrature only; default: 14-point rule')
    parser.add_argument('--log-every', type=int, default=200)
    parser.add_argument('--checkpoint-every', type=int, default=1000)
    parser.add_argument('--vtk', dest='write_vtk', action=argparse.BooleanOptionalAction, default=None,
                        help='write ParaView time series; enabled for a fresh demo, inherited on resume')
    parser.add_argument('--output-every', type=int, default=None,
                        help='VTK frame interval in steps; default: 400 (0.02 s)')
    parser.add_argument('--warm-start', action=argparse.BooleanOptionalAction, default=None,
                        help='reuse previous IB mass-solve coefficients (experimental)')
    parser.add_argument('--pressure-backend', choices=('torch','fused','workspace','graph'), default=None,
                        help='graph: fused workspace and CUDA Graph V-cycle blocks')
    parser.add_argument('--coupling-backend', choices=('reference','optimized'), default=None,
                        help='optimized: IB stencil cache and combined checks; requires fused execution')
    parser.add_argument('--execution-backend', choices=('torch','fused'), default=None,
                        help='fused: compact IB, buffered CSR PCG and compiled fluid/solid kernels')
    parser.add_argument('--solid-backend', choices=('reference','pointwise'), default=None,
                        help='pointwise: fused scalar Guccione stress; requires fused execution')
    parser.add_argument('--mass-backend', choices=('pcg','graph'), default=None,
                        help='graph: captured CSR-PCG blocks; requires fused execution')
    options = vars(parser.parse_args(argv))
    if options['write_vtk'] is None and options['resume'] is None:
        options['write_vtk'] = True
    if options['output'] is None and options['resume'] is None:
        options['output'] = 'results/demo_ideal_lv/mac'
    report = run(**options)
    print(f"{report['status']}: step={report['accepted_steps']}, "
          f"t={report['reached_time_s']:.6f} s, "
          f"elapsed_seconds={report['elapsed_seconds']:.3f}", flush=True)
    return report


if __name__ == '__main__':
    main()
