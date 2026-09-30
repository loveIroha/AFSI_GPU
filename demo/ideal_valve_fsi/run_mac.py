"""Generated AFSI demo_340 valve. Edit CONFIG or pass --config JSON."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from afsi_torch.config import (ValveSimulationConfig, TimeConfig, FluidConfig, OutputConfig,
                               ValveExecutionConfig, ValveConfig, InletConfig, MGOptions, mass_options)
from afsi_torch.simulation.cli import valve_main

# Unit out-of-plane thickness, cm-g-s. Fluid height and solid.height must agree.
CONFIG = ValveSimulationConfig(
    time=TimeConfig(dt=1/16000, end_time=3.),
    fluid=FluidConfig(shape=(256,64), lengths=(8.,1.61), origin=(0.,0.), rho=1., mu=.1),
    solid=ValveConfig(width=.0212, length=.7, x_right=2., height=1.61, mesh_size=.01,
                      C0=200000., C1=1000000., kappa=400000., beta=100000000.),
    inlet=InletConfig(amplitude=5., period=1., offset=1.1),
    pressure_solver=MGOptions(rtol=1e-10, atol=1e-12, max_cycles=100, smooth=4, check_every=2),
    mass_solver=mass_options(),
    execution=ValveExecutionConfig(),
    output=OutputConfig(log_every=160, checkpoint_every=1600, output_every=160,
                        write_vtk=True, fluid_fields=False),
)


def main(argv=None):
    return valve_main(argv, defaults=CONFIG)


if __name__ == '__main__':
    main()
