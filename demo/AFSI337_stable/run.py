"""AFSI337_stable: P2 solid / Q2-Q1 fluid FEM with implicit grad-div."""
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from demo.ideal_lv_fsi.run_fem import CONFIG as BASE_CONFIG
from afsi_torch.simulation.cli import fem_main

CONFIG = replace(BASE_CONFIG, fluid=replace(BASE_CONFIG.fluid, grad_div_alpha=1.))


def main(argv=None):
    return fem_main(argv, defaults=CONFIG, default_output='results/AFSI337_stable')


if __name__ == '__main__':
    main()
