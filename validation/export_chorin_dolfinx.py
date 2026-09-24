"""Actual DOLFINx/PETSc LU reference for three full Chorin time steps.

Independent UFL forms, full boundary lifting and a single pressure gauge.
No project fluid code or torch is imported.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import basix
import basix.ufl
import dolfinx
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem


def export(output, subdivisions=1):
    if MPI.COMM_WORLD.size != 1 or np.dtype(dolfinx.default_scalar_type) != np.dtype(np.float64):
        raise RuntimeError('serial real-float64 DOLFINx required')
    if not dolfinx.__version__.startswith('0.10.') or subdivisions < 1:
        raise ValueError('DOLFINx 0.10.x and positive subdivisions required')
    counts = (2*subdivisions,)*3
    origin, lengths = np.array([-.3, .2, -.5]), np.array([1.4, 1.1, .9])
    domain = mesh.create_box(MPI.COMM_WORLD, [origin, origin+lengths], counts, cell_type=mesh.CellType.hexahedron)
    V = fem.functionspace(domain, basix.ufl.element('Lagrange', 'hexahedron', 2,
        shape=(3,), lagrange_variant=basix.LagrangeVariant.equispaced))
    Q = fem.functionspace(domain, basix.ufl.element('Lagrange', 'hexahedron', 1,
        lagrange_variant=basix.LagrangeVariant.equispaced))
    old, star, current, density, boundary = [fem.Function(V) for _ in range(5)]
    pressure = fem.Function(Q)
    outer = fem.locate_dofs_geometrical(V, lambda x: np.any(
        np.isclose(x, origin[:, None]) | np.isclose(x, (origin+lengths)[:, None]), axis=0))
    gauge = fem.locate_dofs_geometrical(Q, lambda x: np.all(np.isclose(x, origin[:, None]), axis=0))
    if len(gauge) != 1:
        raise RuntimeError('expected one pressure gauge')
    bcu = [fem.dirichletbc(boundary, outer)]
    pressure_value = .7
    bcp = [fem.dirichletbc(dolfinx.default_scalar_type(pressure_value), gauge, Q)]
    dt, rho, mu = .025, 1.1, .15
    u, v, p, q = ufl.TrialFunction(V), ufl.TestFunction(V), ufl.TrialFunction(Q), ufl.TestFunction(Q)
    dx = ufl.Measure('dx', domain=domain, metadata={'quadrature_degree': 8})
    a1 = rho/dt*ufl.inner(u, v)*dx+mu*ufl.inner(ufl.grad(u), ufl.grad(v))*dx
    L1 = (rho/dt*ufl.inner(old, v)-rho*ufl.inner(ufl.dot(ufl.grad(old), old), v)+ufl.inner(density, v))*dx
    a2, L2 = ufl.inner(ufl.grad(p), ufl.grad(q))*dx, -rho/dt*ufl.div(star)*q*dx
    a3, L3 = ufl.inner(u, v)*dx, (ufl.inner(star, v)-dt/rho*ufl.inner(ufl.grad(pressure), v))*dx
    options = {'ksp_type': 'preonly', 'pc_type': 'lu', 'ksp_error_if_not_converged': True}
    problems = [LinearProblem(a, L, u=result, bcs=bc, petsc_options=options,
        petsc_options_prefix=f'chorin_reference_{i}_') for i, (a, L, result, bc) in enumerate(
        [(a1, L1, star, bcu), (a2, L2, pressure, bcp), (a3, L3, current, bcu)])]
    data = dict(velocity_coordinates=V.tabulate_dof_coordinates().copy(),
                pressure_coordinates=Q.tabulate_dof_coordinates().copy())
    for name in ('initial_velocity', 'density', 'boundary_values', 'tentative_velocity', 'pressure', 'velocity', 'divergence_l2'):
        data[name] = []
    for drift in (np.zeros(3), np.array([.05, -.02, .03])):
        def initial(x):
            s = (x-origin[:, None])/lengths[:, None]
            b = np.prod(4*s*(1-s), axis=0)
            return drift[:, None]+np.array([.03, -.01, .02])[:, None]*b
        old.interpolate(initial)
        boundary.interpolate(lambda x: np.broadcast_to(drift[:, None], (3, x.shape[1])))
        density.interpolate(lambda x: np.array([np.sin(x[0]+x[1]), np.cos(x[2])*.3, .2*x[0]*x[2]]))
        for f in (old, boundary, density):
            f.x.scatter_forward()
        for name, f in [('initial_velocity', old), ('density', density), ('boundary_values', boundary)]:
            data[name].append(f.x.array.copy().reshape(-1, 3))
        snapshots = {name: [] for name in ('tentative_velocity', 'pressure', 'velocity', 'divergence_l2')}
        for _ in range(3):
            for problem in problems:
                solution = problem.solve()
                solution.x.scatter_forward()
                if problem.solver.getConvergedReason() <= 0:
                    raise RuntimeError('reference LU solve failed')
            for name, f in [('tentative_velocity', star), ('pressure', pressure), ('velocity', current)]:
                snapshots[name].append(f.x.array.copy().reshape((-1,) if name == 'pressure' else (-1, 3)))
            snapshots['divergence_l2'].append([np.sqrt(fem.assemble_scalar(fem.form(ufl.div(f)**2*dx)))
                                               for f in (star, current)])
            old.x.array[:] = current.x.array
            old.x.scatter_forward()
        for name, values in snapshots.items():
            data[name].append(values)
    for name in list(data):
        data[name] = np.asarray(data[name])
    metadata = dict(schema=1, producer='dolfinx-ufl-chorin', dolfinx=dolfinx.__version__, basix=basix.__version__,
        ufl=ufl.__version__, counts=counts, origin=origin.tolist(), lengths=lengths.tolist(), dt=dt, rho=rho, mu=mu,
        pressure_value=pressure_value, gauge_coordinate=origin.tolist(), steps=3,
        cases=['stationary_boundary', 'translating_boundary'], quadrature_degree=8, linear_solver='PETSc LU')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, metadata=json.dumps(metadata), **data)
    return dict(status='exported', file=str(output), cells=int(np.prod(counts)), cases=metadata['cases'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='validation/results/chorin_coarse.npz')
    parser.add_argument('--subdivisions', type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(export(args.output, args.subdivisions), indent=2))
