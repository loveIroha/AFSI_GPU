"""Generated AFSI demo_337 left ventricle. Edit CONFIG or pass --config JSON."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from afsi_torch.config import (LVSimulationConfig, TimeConfig, FluidConfig, OutputConfig,
                               LVExecutionConfig, LVConfig, GuccioneParameters, AFSI337Loads, MGOptions, mass_options)
from afsi_torch.simulation.cli import lv_main

# Units: cm, g, s. Keep the solid inside the full IB support region.
CONFIG = LVSimulationConfig(
    time=TimeConfig(dt=5e-5, end_time=2.),
    fluid=FluidConfig(shape=(64,64,64), lengths=(5.,5.,5.), origin=(0.,0.,0.), rho=1., mu=1.),
    geometry=LVConfig(inner_axes=(.7,.7,1.7), outer_axes=(1.,1.,2.), base_height=.5,
                      center=(3.5,2.5,2.5), long_axis='x', mesh_size=.1),
    material=GuccioneParameters(C=20000., bf=8., bt=2., bfs=4., kappa=500000.),
    loads=AFSI337Loads(pressure=150000., tension=600000., ramp_time=1.5),
    beta=500000.,basal_constraint='spring',
    pressure_solver=MGOptions(rtol=1e-10, atol=1e-12, max_cycles=100, smooth=4, check_every=2),
    mass_solver=mass_options(),  # rtol=1e-12, atol=1e-13, max_iterations=500
    execution=LVExecutionConfig(execution_backend='fused',pressure_backend='graph',
        solid_backend='pointwise',mass_backend='graph',coupling_backend='optimized',warm_start=True,ib_backend='cuda'),
    output=OutputConfig(log_every=200, checkpoint_every=1000, output_every=400, write_vtk=True),
)


def main(argv=None):
    return lv_main(argv, defaults=CONFIG)


if __name__ == '__main__':
    main()
