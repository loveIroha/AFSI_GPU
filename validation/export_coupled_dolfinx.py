"""Independent small coupled body: UFL solid + NumPy IB + PETSc fluid LU.

Zero bootstrap and lagged force update mirror the source demo. This executes
neither the original afsi C++ binary nor a full LV/cardiac cycle.
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
from dolfinx import fem, mesh, la
from dolfinx.fem.petsc import LinearProblem
from reference_io import EDGES, match_points
from ib_reference import weights


def export(output):
    if MPI.COMM_WORLD.size != 1 or np.dtype(dolfinx.default_scalar_type) != np.dtype(np.float64):
        raise RuntimeError('serial real-float64 DOLFINx required')
    if not dolfinx.__version__.startswith('0.10.'):
        raise RuntimeError('DOLFINx 0.10.x required')
    origin, lengths, counts = np.array([-1.5]*3), np.array([4.]*3), (3,)*3
    spacing = lengths/(2*np.array(counts))
    fluid = mesh.create_box(MPI.COMM_WORLD, [origin, origin+lengths], counts, cell_type=mesh.CellType.hexahedron)
    V = fem.functionspace(fluid, basix.ufl.element('Lagrange', 'hexahedron', 2,
        shape=(3,), lagrange_variant=basix.LagrangeVariant.equispaced))
    Q = fem.functionspace(fluid, basix.ufl.element('Lagrange', 'hexahedron', 1,
        lagrange_variant=basix.LagrangeVariant.equispaced))
    old, star, current, density = [fem.Function(V) for _ in range(4)]
    pressure = fem.Function(Q)
    outer = fem.locate_dofs_geometrical(V, lambda y: np.any(
        np.isclose(y, origin[:, None]) | np.isclose(y, (origin+lengths)[:, None]), axis=0))
    gauge = fem.locate_dofs_geometrical(Q, lambda y: np.all(np.isclose(y, origin[:, None]), axis=0))
    bcu = [fem.dirichletbc(np.zeros(3, dtype=np.float64), outer, V)]
    bcp = [fem.dirichletbc(dolfinx.default_scalar_type(0.), gauge, Q)]
    dt, rho, mu = .002, 1., .1
    u, v, p, q = ufl.TrialFunction(V), ufl.TestFunction(V), ufl.TrialFunction(Q), ufl.TestFunction(Q)
    dx = ufl.Measure('dx', domain=fluid, metadata={'quadrature_degree': 8})
    a1 = rho/dt*ufl.inner(u, v)*dx+mu*ufl.inner(ufl.grad(u), ufl.grad(v))*dx
    L1 = (rho/dt*ufl.inner(old, v)-rho*ufl.inner(ufl.dot(ufl.grad(old), old), v)+ufl.inner(density, v))*dx
    forms = [(a1, L1, star, bcu), (ufl.inner(ufl.grad(p), ufl.grad(q))*dx, -rho/dt*ufl.div(star)*q*dx, pressure, bcp),
        (ufl.inner(u, v)*dx, (ufl.inner(star, v)-dt/rho*ufl.inner(ufl.grad(pressure), v))*dx, current, bcu)]
    problems = [LinearProblem(a, L, u=f, bcs=bc, petsc_options_prefix=f'coupled_ref_{i}_',
        petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu', 'ksp_error_if_not_converged': True})
        for i, (a, L, f, bc) in enumerate(forms)]

    body = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1, cell_type=mesh.CellType.tetrahedron)
    S = fem.functionspace(body, basix.ufl.element('Lagrange', 'tetrahedron', 2,
        shape=(3,), lagrange_variant=basix.LagrangeVariant.equispaced))
    X = S.tabulate_dof_coordinates().copy()
    position = fem.Function(S)
    position.x.array[:] = (X+.01*X**2).reshape(-1)
    position.x.scatter_forward()
    rv = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    rn = np.concatenate((rv, rv[np.array(EDGES)].mean(1)))
    cells = []
    for c in range(body.topology.index_map(3).size_local):
        xyz = body.geometry.cmap.push_forward(rn, body.geometry.x[body.geometry.dofmap[c]])
        dofs = S.dofmap.cell_dofs(c)
        cells.append(dofs[match_points(xyz, X[dofs])])
    body.topology.create_connectivity(2, 3)
    endo = mesh.locate_entities_boundary(body, 2, lambda y: np.isclose(y[0], 1.))
    base = mesh.locate_entities_boundary(body, 2, lambda y: np.isclose(y[2], 0.))
    facets = np.concatenate((endo, base)).astype(np.int32)
    markers = np.concatenate((np.ones(len(endo)), 2*np.ones(len(base)))).astype(np.int32)
    order = np.argsort(facets)
    facets, markers = facets[order], markers[order]
    tags = mesh.meshtags(body, 2, facets, markers)
    facet_geometry = mesh.entities_to_geometry(body, 2, facets, permute=False)
    tagged_vertices = np.array([match_points(body.geometry.x[g], X) for g in facet_geometry])
    vp, vw = basix.make_quadrature(basix.CellType.tetrahedron, 6)
    sp, sw = basix.make_quadrature(basix.CellType.triangle, 6)
    dX = ufl.Measure('dx', domain=body, metadata={'quadrature_rule': 'custom', 'quadrature_points': vp, 'quadrature_weights': vw})
    ds = ufl.Measure('ds', domain=body, subdomain_data=tags, metadata={'quadrature_degree': 6})
    F = ufl.variable(ufl.grad(position))
    J = ufl.det(F)
    solid_mu, solid_lam, beta = 3., 5., 2.
    W = .5*solid_mu*(ufl.inner(F, F)-3)-solid_mu*ufl.ln(J)+.5*solid_lam*ufl.ln(J)**2
    T = fem.Constant(body, dolfinx.default_scalar_type(.1))
    Pendo = fem.Constant(body, dolfinx.default_scalar_type(.2))
    fiber = ufl.as_vector((1., 0., 0.))
    PK1 = ufl.diff(W, F)+T*ufl.outer(F*fiber, fiber)
    test = ufl.TestFunction(S)
    force_form = fem.form(-ufl.inner(PK1, ufl.grad(test))*dX-
        beta*ufl.inner(position-ufl.SpatialCoordinate(body), test)*ds(2)-
        Pendo*ufl.inner(test, ufl.cofac(F)*ufl.FacetNormal(body))*ds(1))
    fluid_X = V.tabulate_dof_coordinates().copy()
    data = dict(X=X, cells=np.asarray(cells, dtype=np.int64), tagged_vertices=tagged_vertices, facet_tags=markers,
        volume_points=vp, volume_weights=vw, surface_points=sp, surface_weights=sw,
        velocity_coordinates=fluid_X, pressure_coordinates=Q.tabulate_dof_coordinates().copy(),
        initial_x=position.x.array.copy().reshape(-1, 3))
    for name in ('x', 'velocity', 'pressure', 'force', 'applied_density'):
        data[name] = []
    for step in range(3):
        old_x = position.x.array.copy().reshape(-1, 3)
        H = weights(old_x, fluid_X, spacing)
        data['applied_density'].append(density.x.array.copy().reshape(-1, 3))
        for problem in problems:
            result = problem.solve()
            result.x.scatter_forward()
            if problem.solver.getConvergedReason() <= 0:
                raise RuntimeError('reference fluid solve failed')
        current_u = current.x.array.copy().reshape(-1, 3)
        new_x = old_x+dt*(H@current_u)
        position.x.array[:] = new_x.reshape(-1)
        position.x.scatter_forward()
        T.value, Pendo.value = .1+step*dt, .2+2*step*dt
        assembled = fem.assemble_vector(force_form)
        assembled.scatter_reverse(la.InsertMode.add)
        force = assembled.array.copy().reshape(-1, 3)
        density.x.array[:] = (weights(new_x, fluid_X, spacing).T@force/np.prod(spacing)).reshape(-1)
        density.x.scatter_forward()
        for name, value in [('x', new_x), ('velocity', current_u), ('pressure', pressure.x.array.copy()), ('force', force)]:
            data[name].append(value)
        old.x.array[:] = current.x.array
        old.x.scatter_forward()
    data = {key: np.asarray(value) for key, value in data.items()}
    metadata = dict(schema=1, producer='dolfinx-numpy-ib-coupled', dolfinx=dolfinx.__version__, basix=basix.__version__,
        counts=counts, origin=origin.tolist(), lengths=lengths.tolist(), dt=dt, rho=rho, mu=mu,
        solid_mu=solid_mu, solid_lam=solid_lam, beta=beta, steps=3, pressure_gauge=origin.tolist(),
        pressure_formula='.2+2*t', tension_formula='.1+t', force_bootstrap='zero',
        update='flow, interpolate at old x, update x, assemble force at new x and old t, spread at new x',
        reference_solver='DOLFINx/PETSc LU + scalar NumPy full-grid Peskin kernel')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, metadata=json.dumps(metadata), **data)
    return dict(status='exported', file=str(output), steps=3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='validation/results/coupled_patch.npz')
    print(json.dumps(export(parser.parse_args().output), indent=2))
