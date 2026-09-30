"""Compatibility entry: short generated LV MAC validation segment."""
from dataclasses import replace
from afsi_torch.config import LVSimulationConfig, TimeConfig, OutputConfig
from afsi_torch.simulation.lv_mac import run
from afsi_torch.simulation.cli import lv_main


def main(argv=None):
    config = replace(LVSimulationConfig(), time=TimeConfig(5e-5,.005),
                     output=OutputConfig(20,200,400,False))
    return lv_main(argv, defaults=config, default_output='results/lv_mac')


if __name__ == '__main__':
    main()
