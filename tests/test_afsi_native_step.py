"""The staged AFSI comparison must expose the first altered field."""
import json

import numpy as np
import torch

from afsi_torch import ib
from afsi_torch.fluid import ChorinSolver, create_box, prepare_operators
from afsi_torch.fluid.solvers import SolverOptions
from validation.compare_afsi_native_step import compare


def test_native_stage_report_identifies_first_mismatch(tmp_path):
    mesh = create_box((2, 2, 2), (4., 4., 4.), (0., 0., 0.))
    x = torch.tensor([[1.6, 1.7, 1.8]], dtype=torch.float64)
    force = torch.tensor([[70., -20., 16.]], dtype=torch.float64)
    stencil = ib.prepare_stencil(x, mesh.velocity_grid)
    density = ib.spread_density(force, stencil)
    flow = ChorinSolver(prepare_operators(mesh), dt=5e-5,
        options=SolverOptions(rtol=1e-12, atol=1e-13,
                              max_iterations=4000, recompute_every=200))
    result = flow.step(torch.zeros_like(mesh.velocity_coordinates), density=density)
    vs = ib.interpolate(result.velocity, stencil)
    meta = dict(schema=1, producer='afsi-native-ib-chorin', counts=[2, 2, 2],
        lengths=[4., 4., 4.], origin=[0., 0., 0.], dt=5e-5, rho=1., mu=1.,
        pressure_gauge_coordinate=[0., 0., 0.],
        velocity_boundary='zero Dirichlet on all outer faces',
        ib_kernel='Peskin four point, epsilon=velocity lattice spacing')
    fields = dict(metadata=json.dumps(meta),
        velocity_coordinates=mesh.velocity_coordinates.numpy(),
        pressure_coordinates=mesh.pressure_coordinates.numpy(),
        solid_coordinates=x.numpy(), solid_force=force.numpy(),
        fluid_density=density.numpy(),
        fluid_weak_load=flow.op.density_load(density).numpy(),
        tentative_velocity=result.tentative_velocity.numpy(),
        pressure=result.pressure.numpy(), velocity=result.velocity.numpy(),
        solid_velocity=vs.numpy(), solid_next=(x+flow.dt*vs).numpy())
    # DOLFINx and PyTorch do not generally number their DOFs in the same order.
    velocity_order = np.arange(len(fields['velocity_coordinates']))[::-1]
    pressure_order = np.arange(len(fields['pressure_coordinates']))[::-1]
    for name in ('velocity_coordinates', 'fluid_density', 'fluid_weak_load',
                 'tentative_velocity', 'velocity'):
        fields[name] = fields[name][velocity_order]
    for name in ('pressure_coordinates', 'pressure'):
        fields[name] = fields[name][pressure_order]
    path = tmp_path / 'reference.npz'
    np.savez_compressed(path, **fields)
    matched = compare(path)
    assert matched['passed'] and matched['first_mismatch'] is None
    changed = dict(fields)
    changed['fluid_density'] = fields['fluid_density'].copy()
    changed['fluid_density'][0, 0] += 1.
    np.savez_compressed(path, **changed)
    failed = compare(path)
    assert not failed['passed'] and failed['first_mismatch'] == 'fluid_density'
