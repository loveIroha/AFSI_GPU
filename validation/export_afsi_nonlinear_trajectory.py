"""Export a small moving P2 Guccione solid coupled to native AFSI IB/Chorin.

Run only in a serial DOLFINx/afsic environment. This is a controlled numerical
alignment case, not the original externally meshed LV or a cardiac cycle.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import basix
import dolfinx
import numpy as np
import ufl
from basix.ufl import element
from dolfinx import fem, mesh
from mpi4py import MPI

from afsic import ChorinSolver, IBInterpolation3D, IBMesh3D
from reference_io import EDGES, match_points


def _p2_cells(domain, space, X):
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    nodes = np.concatenate((vertices, vertices[np.asarray(EDGES)].mean(1)))
    result = []
    for cell in range(domain.topology.index_map(3).size_local):
        physical = domain.geometry.cmap.push_forward(
            nodes, domain.geometry.x[domain.geometry.dofmap[cell]])
        dofs = space.dofmap.cell_dofs(cell)
        result.append(dofs[match_points(physical, X[dofs])])
    return np.asarray(result, dtype=np.int64)


def _load(time):
    # AFSI's step-n load is assembled after moving the solid and used in step n+1.
    return 1000. + 200.*min(time/.0003, 1.), 5000. + 1000.*min(time/.0003, 1.)


def export(output, *, steps=6):
    if MPI.COMM_WORLD.size != 1 or np.dtype(dolfinx.default_scalar_type) != np.dtype(np.float64):
        raise RuntimeError('serial real-float64 DOLFINx is required')
    if steps < 2:
        raise ValueError('at least two steps needed to test delayed nonlinear force')
    counts, origin, lengths = (4, 4, 4), (0., 0., 0.), (4., 4., 4.)
    dt, rho, mu, beta = 5e-5, 1., 1., 1000.
    domain = mesh.create_box(MPI.COMM_WORLD, [np.array(origin), np.array(lengths)],
                             counts, cell_type=mesh.CellType.hexahedron)
    V = fem.functionspace(domain, element('Lagrange', 'hexahedron', 2, shape=(3,)))
    Q = fem.functionspace(domain, element('Lagrange', 'hexahedron', 1))
    zero = fem.Function(V)
    outer = fem.locate_dofs_geometrical(V, lambda y: np.any(
        np.isclose(y, np.array(origin)[:, None]) |
        np.isclose(y, np.array(lengths)[:, None]), axis=0))
    gauge = fem.locate_dofs_geometrical(Q, lambda y: np.all(
        np.isclose(y, np.array(origin)[:, None]), axis=0))
    if len(gauge) != 1:
        raise RuntimeError('exactly one pressure gauge expected')
    flow = ChorinSolver(V, Q, [fem.dirichletbc(zero, outer)],
                        [fem.dirichletbc(np.float64(0.), gauge, Q)], dt, rho, mu)
    for solver in (flow.solver1, flow.solver2, flow.solver3):
        solver.setTolerances(rtol=1e-12, atol=1e-13, max_it=10000)

    low, high = (1.82, 1.86, 1.9), (2.18, 2.14, 2.1)
    body = mesh.create_box(MPI.COMM_WORLD, [np.array(low), np.array(high)],
                           (1, 1, 1), cell_type=mesh.CellType.tetrahedron)
    W = fem.functionspace(body, element('Lagrange', 'tetrahedron', 2, shape=(3,),
                                        lagrange_variant=basix.LagrangeVariant.equispaced))
    X = W.tabulate_dof_coordinates().copy()
    cells = _p2_cells(body, W, X)
    solid_x, solid_force, solid_velocity = (fem.Function(W) for _ in range(3))
    solid_x.x.array[:] = X.ravel()
    solid_x.x.scatter_forward()

    endo = mesh.locate_entities_boundary(body, 2, lambda y: np.isclose(y[0], high[0]))
    base = mesh.locate_entities_boundary(body, 2, lambda y: np.isclose(y[0], low[0]))
    if not len(endo) or not len(base):
        raise RuntimeError('missing loaded or spring boundary')
    facets = np.concatenate((endo, base))
    tags = mesh.meshtags(body, 2, facets[np.argsort(facets)],
        np.concatenate((np.ones(len(endo), dtype=np.int32),
                        np.full(len(base), 2, dtype=np.int32)))[np.argsort(facets)])
    q, w = basix.make_quadrature(basix.CellType.tetrahedron, 6)
    sq, sw = basix.make_quadrature(basix.CellType.triangle, 6)
    dx = ufl.Measure('dx', domain=body, metadata={
        'quadrature_rule': 'custom', 'quadrature_points': q, 'quadrature_weights': w})
    ds = ufl.Measure('ds', domain=body, subdomain_data=tags,
                     metadata={'quadrature_degree': 6, 'quadrature_rule': 'default'})
    F = ufl.variable(ufl.grad(solid_x))
    E = .5*(F.T*F - ufl.Identity(3))
    f = ufl.as_vector((1., 0., 0.))
    s = ufl.as_vector((0., 1., 0.))
    n = ufl.cross(s, f)
    strain = lambda a, b: ufl.dot(a, E*b)
    Qexp = (8*strain(f, f)**2 + 2*(strain(s, s)**2+strain(n, n)**2+
            2*strain(s, n)**2) + 8*(strain(f, s)**2+strain(f, n)**2))
    energy = 10000*(ufl.exp(Qexp)-1) + 500000*(ufl.det(F)-1)**2
    p = fem.Constant(body, np.float64(1000.))
    tension = fem.Constant(body, np.float64(5000.))
    test = ufl.TestFunction(W)
    P = ufl.diff(energy, F) + tension*ufl.outer(F*f, f)
    force_form = fem.form(-ufl.inner(P, ufl.grad(test))*dx
        - p*ufl.inner(test, ufl.cofac(F)*ufl.FacetNormal(body))*ds(1)
        - beta*ufl.inner(solid_x-ufl.SpatialCoordinate(body), test)*ds(2))

    ibmesh = IBMesh3D(*(origin[0], lengths[0], origin[1], lengths[1],
                        origin[2], lengths[2]), *counts, 2)
    transfer = IBInterpolation3D(ibmesh)
    fluid_coordinates = fem.Function(V)
    fluid_coordinates.interpolate(lambda y: y)
    ibmesh.build_map(fluid_coordinates._cpp_object)
    transfer.evaluate_current_points(solid_x._cpp_object)

    positions = [X.copy()]
    forces = [np.zeros_like(X)]  # AFSI's source-compatible g=0 bootstrap.
    densities, velocities, pressures, solid_velocities = [], [], [], []
    for step in range(steps):
        solid_force.x.array[:] = forces[-1].ravel()
        solid_force.x.scatter_forward()
        flow.f.x.array[:] = 0.
        transfer.solid_to_fluid(flow.f._cpp_object, solid_force._cpp_object)
        flow.f.x.scatter_forward()
        densities.append(flow.f.x.array.copy().reshape(-1, 3))
        flow.solve_one_step()
        velocities.append(flow.u_.x.array.copy().reshape(-1, 3))
        pressures.append(flow.p_.x.array.copy())
        transfer.fluid_to_solid(flow.u_._cpp_object, solid_velocity._cpp_object)
        solid_velocities.append(solid_velocity.x.array.copy().reshape(-1, 3))
        solid_x.x.array[:] += dt*solid_velocity.x.array
        solid_x.x.scatter_forward()
        positions.append(solid_x.x.array.copy().reshape(-1, 3))
        p.value, tension.value = map(np.float64, _load(step*dt))
        forces.append(fem.assemble_vector(force_form).array.copy().reshape(-1, 3))
        transfer.evaluate_current_points(solid_x._cpp_object)
    for name, array in [('positions', positions), ('forces', forces),
                        ('velocities', velocities), ('pressures', pressures)]:
        if not np.isfinite(array).all():
            raise RuntimeError(f'nonfinite native {name}')
    metadata = dict(schema=1, producer='afsi-native-nonlinear-trajectory',
        afsi_chorin_source=inspect.getsourcefile(ChorinSolver),
        afsi_chorin_sha256=hashlib.sha256(inspect.getsource(ChorinSolver).encode()).hexdigest(),
        dolfinx=dolfinx.__version__, counts=counts, origin=origin, lengths=lengths,
        dt=dt, rho=rho, mu=mu, beta=beta, steps=steps,
        pressure_gauge_coordinate=origin,
        ib_kernel='Peskin four point, epsilon=velocity lattice spacing',
        solid='P2 tetrahedra, Guccione plus active fiber stress, follower pressure and base spring',
        velocity_boundary='zero Dirichlet on all outer faces')
    metadata['load'] = dict(pressure_initial=1000., pressure_rise=200.,
                            tension_initial=5000., tension_rise=1000., ramp_time=.0003)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, metadata=json.dumps(metadata), X=X, cells=cells,
        volume_points=q, volume_weights=w, surface_points=sq, surface_weights=sw,
        velocity_coordinates=V.tabulate_dof_coordinates().copy(),
        pressure_coordinates=Q.tabulate_dof_coordinates().copy(),
        positions=np.asarray(positions), forces=np.asarray(forces),
        fluid_densities=np.asarray(densities), velocities=np.asarray(velocities),
        pressures=np.asarray(pressures), solid_velocities=np.asarray(solid_velocities))
    return dict(status='exported', file=str(output), steps=steps,
                solid_nodes=len(X), dolfinx=dolfinx.__version__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/afsi_nonlinear_trajectory/reference.npz')
    parser.add_argument('--steps', type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(export(args.output, steps=args.steps), indent=2))

