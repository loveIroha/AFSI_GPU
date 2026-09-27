"""The trajectory comparator must track lagged nonlinear force across steps."""
from dataclasses import replace
import json

import numpy as np
import torch

from afsi_torch import boundary as bd, solid
from afsi_torch.coupling import ExplicitIBStepper
from afsi_torch.fields import prepare_reference_fields
from afsi_torch.fluid import ChorinSolver, create_box, prepare_operators
from afsi_torch.fluid.solvers import SolverOptions
from afsi_torch.materials import GuccioneParameters
from afsi_torch.tetrahedron import promote_p1
from validation.compare_afsi_nonlinear_trajectory import compare, _load


def test_nonlinear_trajectory_locates_first_feedback_error(tmp_path):
    low = torch.tensor([1.82, 1.86, 1.9], dtype=torch.float64)
    high = torch.tensor([2.18, 2.14, 2.1], dtype=torch.float64)
    vertices = torch.stack([torch.where(torch.tensor([i & 1, i & 2, i & 4],
        dtype=torch.bool), high, low) for i in range(8)])
    tetrahedra = torch.tensor([[0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7],
                               [0, 6, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7]])
    X, cells = promote_p1(vertices, tetrahedra)
    geometry = solid.prepare_p2(X, cells)
    faces = bd.extract_boundary(X, cells)
    xface = X[faces[:, :3], 0]
    endo = bd.prepare_surface(X, faces[torch.isclose(xface, high[0]).all(1)])
    base = bd.prepare_surface(X, faces[torch.isclose(xface, low[0]).all(1)])
    fields = prepare_reference_fields(geometry, [1., 0., 0.], [0., 1., 0.])
    load = dict(pressure_initial=1000., pressure_rise=200.,
                tension_initial=5000., tension_rise=1000., ramp_time=.0003)

    def force(x, time):
        pressure, tension = _load(time, load)
        current = replace(fields, tension=torch.full_like(fields.tension, tension))
        return (solid.guccione_force(x, geometry, current, GuccioneParameters())+
                bd.pressure_force(x, endo, pressure)+bd.spring_force(x, base, 1000.))

    mesh = create_box((2, 2, 2), (4., 4., 4.), (0., 0., 0.))
    flow = ChorinSolver(prepare_operators(mesh), dt=5e-5,
        options=SolverOptions(rtol=1e-12, atol=1e-13,
                              max_iterations=4000, recompute_every=200))
    driver = ExplicitIBStepper(flow, force,
        lambda x: solid.validate_deformation(x, geometry))
    state = driver.initialize(X)
    positions, forces = [X.numpy().copy()], [state.force.numpy().copy()]
    densities, velocities, pressures, solid_velocities = [], [], [], []
    for _ in range(2):
        result = driver.step(state)
        state = result.state
        positions.append(state.x.numpy().copy())
        forces.append(state.force.numpy().copy())
        densities.append(result.applied_density.numpy().copy())
        velocities.append(state.velocity.numpy().copy())
        pressures.append(state.pressure.numpy().copy())
        solid_velocities.append(result.solid_velocity.numpy().copy())
    metadata = dict(schema=1, producer='afsi-native-nonlinear-trajectory',
        counts=[2, 2, 2], origin=[0., 0., 0.], lengths=[4., 4., 4.],
        dt=5e-5, rho=1., mu=1., beta=1000., steps=2, load=load,
        pressure_gauge_coordinate=[0., 0., 0.],
        ib_kernel='Peskin four point, epsilon=velocity lattice spacing',
        velocity_boundary='zero Dirichlet on all outer faces')
    # Reverse fluid DOFs to exercise coordinate mapping, as with DOLFINx.
    vorder = np.arange(len(mesh.velocity_coordinates))[::-1]
    porder = np.arange(len(mesh.pressure_coordinates))[::-1]
    fields_np = dict(metadata=json.dumps(metadata), X=X.numpy(), cells=cells.numpy(),
        volume_points=solid.tetrahedron_rule(4)[0].numpy(),
        volume_weights=solid.tetrahedron_rule(4)[1].numpy(),
        surface_points=bd.triangle.quadrature(4)[0].numpy(),
        surface_weights=bd.triangle.quadrature(4)[1].numpy(),
        velocity_coordinates=mesh.velocity_coordinates.numpy()[vorder],
        pressure_coordinates=mesh.pressure_coordinates.numpy()[porder],
        positions=np.asarray(positions), forces=np.asarray(forces),
        fluid_densities=np.asarray(densities)[:, vorder],
        velocities=np.asarray(velocities)[:, vorder],
        pressures=np.asarray(pressures)[:, porder],
        solid_velocities=np.asarray(solid_velocities))
    path = tmp_path/'trajectory.npz'
    np.savez_compressed(path, **fields_np)
    matched = compare(path)
    assert matched['passed'] and matched['native_max_displacement_cm'] > 1e-9
    fields_np['forces'] = fields_np['forces'].copy()
    fields_np['forces'][1, 0, 0] += 1.
    np.savez_compressed(path, **fields_np)
    failed = compare(path)
    assert not failed['passed']
    assert failed['first_mismatch'] == dict(step=1, stage='solid_force')

