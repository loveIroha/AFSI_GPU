"""Generated AFSI demo_337 with Q2/Q1 FEM fluid. Edit CONFIG or use --config."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from afsi_torch.config import (LVFEMSimulationConfig,TimeConfig,FEMFluidConfig,OutputConfig,
                               LVConfig,GuccioneParameters,AFSI337Loads,SolverOptions)
from afsi_torch.simulation.cli import fem_main

CONFIG=LVFEMSimulationConfig(
    time=TimeConfig(dt=5e-5,end_time=2.),
    fluid=FEMFluidConfig(shape=(32,32,32),lengths=(5.,5.,5.),origin=(0.,0.,0.),rho=1.,mu=1.),
    geometry=LVConfig(inner_axes=(.7,.7,1.7),outer_axes=(1.,1.,2.),base_height=.5,
                      center=(3.5,2.5,2.5),long_axis='x',mesh_size=.1),
    material=GuccioneParameters(C=20000.,bf=8.,bt=2.,bfs=4.,kappa=500000.),
    loads=AFSI337Loads(pressure=150000.,tension=600000.,ramp_time=1.5),
    beta=500000.,backend='csr',ib_backend='cuda',
    solver=SolverOptions(rtol=1e-10,atol=1e-12,max_iterations=4000,recompute_every=200,check_every=8),
    history_every=20,
    output=OutputConfig(log_every=100,checkpoint_every=200,output_every=200,write_vtk=True),
)


def main(argv=None):
    return fem_main(argv,defaults=CONFIG)


if __name__ == '__main__':
    main()
