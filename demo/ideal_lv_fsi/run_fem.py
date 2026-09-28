"""Run the generated AFSI demo_337 left ventricle with FEM fluid for 2 s.

The fluid is the existing PyTorch Q2/Q1 Chorin finite-element backend.
The nonlinear finite-element solid and AFSI demo_337 loads are shared
with the MAC example.
"""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples.lv_cycle import run  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', help='fresh-run output directory')
    parser.add_argument('--resume', help='FEM checkpoint.npz to resume')
    parser.add_argument('--end-time', type=float, default=2.,
                        help='final simulation time in seconds; default: 2')
    parser.add_argument('--dt', type=float, help='fresh run only; default: 5e-5 s')
    parser.add_argument('--mesh-size', type=float,
                        help='fresh solid mesh only; default: 0.1 cm')
    parser.add_argument('--fluid-cells', type=int,
                        help='fresh Q2/Q1 grid cells per axis; default: 32')
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--history-every', type=int, default=20)
    parser.add_argument('--output-every', type=int, default=200)
    parser.add_argument('--checkpoint-every', type=int, default=200)
    parser.add_argument('--no-vtk', action='store_true')
    options = vars(parser.parse_args(argv))
    write_vtk = not options.pop('no_vtk')
    if options['output'] is None and options['resume'] is None:
        options['output'] = 'results/demo_ideal_lv/fem'
    report = run(**options, profile='afsi337', backend='csr', write_vtk=write_vtk)
    print(f"{report['status']}: step={report['accepted_steps']}, "
          f"t={report['reached_time_s']:.6f} s, "
          f"elapsed_seconds={report['elapsed_seconds']:.3f}", flush=True)
    return report


if __name__ == '__main__':
    main()
