"""Export one AFSI-native IB/Chorin step from a serial DOLFINx environment.

Run this with the installed afsic C++ extension, not with afsi-torch. The
solid nodes carry prescribed integrated forces so every transfer and fluid
stage can be compared without importing an unrelated LV geometry.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import dolfinx
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_vector, apply_lifting, set_bc
from mpi4py import MPI
from petsc4py import PETSc
from basix.ufl import element

from afsic import ChorinSolver, IBMesh3D, IBInterpolation3D


def _staged_native_step(flow):
    """Use the live AFSI forms/solvers, retaining fields before correction."""
    solves = {}
    for name, vector, form, bilinear, bcs, solver, target in (
            ('tentative', flow.b1, flow.L1, flow.a1, flow.bcu, flow.solver1, flow.u_),
            ('pressure', flow.b2, flow.L2, flow.a2, flow.bcp, flow.solver2, flow.p_),
            ('correction', flow.b3, flow.L3, flow.a3, flow.bcu, flow.solver3, flow.u_)):
        with vector.localForm() as local:
            local.set(0)
        assemble_vector(vector, form)
        apply_lifting(vector, [bilinear], [bcs])
        vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                           mode=PETSc.ScatterMode.REVERSE)
        set_bc(vector, bcs)
        solver.solve(vector, target.x.petsc_vec)
        if solver.getConvergedReason() <= 0:
            raise RuntimeError('AFSI native Chorin substep did not converge')
        solves[name] = dict(iterations=solver.getIterationNumber(),
                            residual_norm=solver.getResidualNorm())
        target.x.scatter_forward()
        if vector is flow.b1:
            tentative = target.x.array.copy().reshape(-1, 3)
        elif vector is flow.b2:
            pressure = target.x.array.copy()
    return tentative, pressure, flow.u_.x.array.copy().reshape(-1, 3), solves


def export(output):
    if MPI.COMM_WORLD.size != 1 or np.dtype(dolfinx.default_scalar_type) != np.dtype(np.float64):
        raise RuntimeError('serial real-float64 DOLFINx is required')
    counts = (4, 4, 4)
    origin = (0., 0., 0.)
    lengths = (4., 4., 4.)
    dt, rho, mu = 5e-5, 1., 1.
    domain = mesh.create_box(MPI.COMM_WORLD, [np.array(origin), np.array(lengths)],
                             counts, cell_type=mesh.CellType.hexahedron)
    V = fem.functionspace(domain, element('Lagrange', 'hexahedron', 2, shape=(3,)))
    Q = fem.functionspace(domain, element('Lagrange', 'hexahedron', 1))
    zero = fem.Function(V)
    outer = fem.locate_dofs_geometrical(V, lambda x: np.any(
        np.isclose(x, np.array(origin)[:, None]) |
        np.isclose(x, np.array(lengths)[:, None]), axis=0))
    gauge = fem.locate_dofs_geometrical(Q, lambda x: np.all(
        np.isclose(x, np.array(origin)[:, None]), axis=0))
    if len(gauge) != 1:
        raise RuntimeError('exactly one pressure gauge expected')
    bcu = [fem.dirichletbc(zero, outer)]
    bcp = [fem.dirichletbc(dolfinx.default_scalar_type(0.), gauge, Q)]

    solid_domain = mesh.create_box(MPI.COMM_WORLD,
        [np.array((1.82, 1.86, 1.9)), np.array((2.18, 2.14, 2.1))],
        (1, 1, 1), cell_type=mesh.CellType.tetrahedron)
    W = fem.functionspace(solid_domain, element('Lagrange', 'tetrahedron', 1, shape=(3,)))
    solid_x, solid_force, solid_velocity = (fem.Function(W) for _ in range(3))
    solid_x.interpolate(lambda x: x)
    x = solid_x.x.array.reshape(-1, 3)
    g = solid_force.x.array.reshape(-1, 3)
    g[:, 0] = 50. + 10.*x[:, 0]
    g[:, 1] = -30. + 5.*x[:, 1]
    g[:, 2] = 20. - 2.*x[:, 2]

    flow = ChorinSolver(V, Q, bcu, bcp, dt, rho, mu)
    for solver in (flow.solver1, flow.solver2, flow.solver3):
        solver.setTolerances(rtol=1e-12, atol=1e-13, max_it=10000)
    ibmesh = IBMesh3D(*(origin[0], lengths[0], origin[1], lengths[1],
                        origin[2], lengths[2]), *counts, 2)
    transfer = IBInterpolation3D(ibmesh)
    fluid_coordinates = fem.Function(V)
    fluid_coordinates.interpolate(lambda x: x)
    ibmesh.build_map(fluid_coordinates._cpp_object)
    transfer.evaluate_current_points(solid_x._cpp_object)
    transfer.solid_to_fluid(flow.f._cpp_object, solid_force._cpp_object)
    flow.f.x.scatter_forward()
    density = flow.f.x.array.copy().reshape(-1, 3)
    load_form = fem.form(ufl.inner(flow.f, ufl.TestFunction(V))*ufl.dx)
    load_vector = assemble_vector(load_form)
    with load_vector.localForm() as local:
        weak_load = local.array.copy().reshape(-1, 3)

    tentative, pressure, corrected, native_solves = _staged_native_step(flow)
    # Confirm instrumentation reproduces the unmodified AFSI public method.
    flow.u_.x.array[:] = 0.
    flow.p_.x.array[:] = 0.
    flow.solve_one_step()
    np.testing.assert_allclose(flow.u_.x.array.reshape(-1, 3), corrected,
                               rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(flow.p_.x.array, pressure, rtol=2e-10, atol=2e-11)
    transfer.fluid_to_solid(flow.u_._cpp_object, solid_velocity._cpp_object)
    vs = solid_velocity.x.array.copy().reshape(-1, 3)
    next_x = x.copy() + dt*vs

    metadata = dict(schema=1, producer='afsi-native-ib-chorin',
        afsi_chorin_source=inspect.getsourcefile(ChorinSolver),
        afsi_chorin_sha256=hashlib.sha256(inspect.getsource(ChorinSolver).encode()).hexdigest(),
        dolfinx=dolfinx.__version__, counts=counts, origin=origin, lengths=lengths,
        dt=dt, rho=rho, mu=mu, pressure_gauge_coordinate=origin,
        ib_kernel='Peskin four point, epsilon=velocity lattice spacing',
        solid_force='prescribed integrated nodal force',
        velocity_boundary='zero Dirichlet on all outer faces', native_solves=native_solves)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, metadata=json.dumps(metadata),
        velocity_coordinates=V.tabulate_dof_coordinates().copy(),
        pressure_coordinates=Q.tabulate_dof_coordinates().copy(),
        solid_coordinates=x.copy(), solid_force=g.copy(), fluid_density=density,
        fluid_weak_load=weak_load,
        tentative_velocity=tentative, pressure=pressure, velocity=corrected,
        solid_velocity=vs, solid_next=next_x)
    return dict(status='exported', file=str(output), fluid_cells=int(np.prod(counts)),
                solid_nodes=len(x), dolfinx=dolfinx.__version__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/afsi_native_step/reference.npz')
    args = parser.parse_args()
    print(json.dumps(export(args.output), indent=2))
