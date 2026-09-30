"""Compatibility entry: short generated valve MAC validation segment."""
from dataclasses import replace
from afsi_torch.config import ValveSimulationConfig, TimeConfig
from afsi_torch.simulation.valve_mac import run
from afsi_torch.simulation.cli import valve_main


def main(argv=None, *, default_end_time=.005, default_output='results/valve_mac'):
    config = replace(ValveSimulationConfig(), time=TimeConfig(1/16000,default_end_time))
    return valve_main(argv, defaults=config, default_output=default_output)


if __name__ == '__main__':
    main()
