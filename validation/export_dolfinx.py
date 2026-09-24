"""Assemble actual UFL/DOLFINx forces and directional derivatives, without Torch.

Run with real-scalar DOLFINx 0.10.x, one MPI rank. The generated fixture uses
P1 geometry and continuous blocked P2 displacement/material Functions.
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
from reference_io import EDGES, TERMS, match_points


def export(output, subdivisions=1):
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("Reference export currently requires one MPI rank")
    if np.dtype(dolfinx.default_scalar_type) != np.dtype(np.float64):
        raise RuntimeError("Reference export requires real float64 DOLFINx")
    if not dolfinx.__version__.startswith("0.10."):
        raise RuntimeError("Use the pinned DOLFINx 0.10.x reference environment")
    if subdivisions < 1:
        raise ValueError("subdivisions must be positive")
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, subdivisions, subdivisions,
                                  subdivisions, cell_type=mesh.CellType.tetrahedron)
    V = fem.functionspace(domain, basix.ufl.element(
        "Lagrange", "tetrahedron", 2, shape=(3,), lagrange_variant=basix.LagrangeVariant.equispaced))
    S = fem.functionspace(domain, basix.ufl.element(
        "Lagrange", "tetrahedron", 2, lagrange_variant=basix.LagrangeVariant.equispaced))
    if V.dofmap.index_map_bs != 3 or V.dofmap.bs != 3:
        raise RuntimeError("Expected blocked vector P2 space")
    X = V.tabulate_dof_coordinates().copy()
    scalar_order = match_points(X, S.tabulate_dof_coordinates())

    def function(space, expression):
        result = fem.Function(space)
        result.interpolate(expression)
        result.x.scatter_forward()
        return result

    fiber = function(V, lambda y: np.array([np.cos(.6*y[0]), np.sin(.6*y[0]), 0*y[0]]))
    sheet = function(V, lambda y: np.array([-np.sin(.6*y[0]), np.cos(.6*y[0]), 0*y[0]]))
    tension = function(S, lambda y: 1000+500*y[1]**2)
    pressure = function(S, lambda y: 1200+200*y[0]**2)
    beta = function(S, lambda y: 5000+1000*y[1]**2)
    x = fem.Function(V)
    direction = function(V, lambda y: np.array([
        .1+.2*y[0]*y[1], -.2+.1*y[2]**2, .15-.1*y[0]*y[2]]))

    # Recover our local P2 ordering by physical coordinates in EACH cell.
    # Geometry push_forward preserves DOLFINx's reference vertex orientation.
    rv = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    rn = np.concatenate((rv, rv[np.array(EDGES)].mean(1)))
    cells = []
    for c in range(domain.topology.index_map(3).size_local):
        geom = domain.geometry.x[domain.geometry.dofmap[c]]
        physical = domain.geometry.cmap.push_forward(rn, geom)
        local = V.dofmap.cell_dofs(c)
        cells.append(local[match_points(physical, X[local])])
    cells = np.asarray(cells, dtype=np.int64)

    domain.topology.create_connectivity(2, 3)
    loaded = mesh.locate_entities_boundary(domain, 2, lambda y: np.isclose(y[0], 1.))
    base = mesh.locate_entities_boundary(domain, 2, lambda y: np.isclose(y[2], 0.))
    facet_ids = np.concatenate((loaded, base)).astype(np.int32)
    tag_values = np.concatenate((np.ones(len(loaded)), 2*np.ones(len(base)))).astype(np.int32)
    order = np.argsort(facet_ids)
    facet_ids, tag_values = facet_ids[order], tag_values[order]
    tags = mesh.meshtags(domain, 2, facet_ids, tag_values)
    facet_geometry = mesh.entities_to_geometry(domain, 2, facet_ids, permute=False)
    tagged_vertices = np.array([match_points(domain.geometry.x[g], X)
                                for g in facet_geometry], dtype=np.int64)

    # Explicit volume rule avoids automatic nonlinear-integrand degree guesses.
    q, w = basix.make_quadrature(basix.CellType.tetrahedron, 6)
    sq, sw = basix.make_quadrature(basix.CellType.triangle, 6)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_rule": "custom",
        "quadrature_points": q, "quadrature_weights": w})
    ds = ufl.Measure("ds", domain=domain, subdomain_data=tags,
                     metadata={"quadrature_degree": 6, "quadrature_rule": "default"})
    v = ufl.TestFunction(V)
    F = ufl.variable(ufl.grad(x))
    E = .5*(F.T*F-ufl.Identity(3))
    n = ufl.cross(sheet, fiber)  # Cross AFTER interpolation, no normalization.
    eff = ufl.dot(fiber, E*fiber)
    ess, enn = ufl.dot(sheet, E*sheet), ufl.dot(n, E*n)
    efs, efn, esn = ufl.dot(fiber, E*sheet), ufl.dot(fiber, E*n), ufl.dot(sheet, E*n)
    parameters = dict(C=20000., bf=8., bt=2., bfs=4., kappa=500000.)
    p = parameters
    Q = p['bf']*eff**2+p['bt']*(ess**2+enn**2+2*esn**2)+2*p['bfs']*(efs**2+efn**2)
    W = .5*p['C']*(ufl.exp(Q)-1)+p['kappa']*(ufl.det(F)-1)**2
    P = ufl.diff(W, F)
    active = tension*ufl.outer(F*fiber, fiber)
    normal = ufl.FacetNormal(domain)
    forms = {
        "passive": -ufl.inner(P, ufl.grad(v))*dx,
        "active": -ufl.inner(active, ufl.grad(v))*dx,
        "pressure": -ufl.inner(v, pressure*ufl.cofac(F)*normal)*ds(1),
        "spring": -beta*ufl.inner(x-ufl.SpatialCoordinate(domain), v)*ds(2),
    }
    forms["total"] = sum(forms.values())
    force_forms = {key: fem.form(value) for key, value in forms.items()}
    tangent_forms = {key: fem.form(ufl.derivative(value, x, direction))
                     for key, value in forms.items()}

    def assemble(form):
        result = fem.assemble_vector(form)
        result.scatter_reverse(la.InsertMode.add)
        return result.array.copy().reshape(-1, 3)

    deformed = X+.04*X**2
    deformed[:, 2] += .03*X[:, 0]**2
    configurations = np.stack((X, deformed))
    data = dict(X=X, cells=cells, x=configurations,
                direction=direction.x.array.reshape(-1, 3).copy(),
                fiber=fiber.x.array.reshape(-1, 3).copy(),
                sheet=sheet.x.array.reshape(-1, 3).copy(),
                tension=tension.x.array[scalar_order].copy(),
                pressure=pressure.x.array[scalar_order].copy(),
                beta=beta.x.array[scalar_order].copy(),
                tagged_vertices=tagged_vertices, facet_tags=tag_values,
                volume_points=q, volume_weights=w, surface_points=sq, surface_weights=sw)
    for key in TERMS:
        data['force_'+key], data['tangent_'+key] = [], []
    for current in configurations:
        x.x.array[:] = current.ravel()
        x.x.scatter_forward()
        for key in TERMS:
            data['force_'+key].append(assemble(force_forms[key]))
            data['tangent_'+key].append(assemble(tangent_forms[key]))
    for key in TERMS:
        data['force_'+key] = np.stack(data['force_'+key])
        data['tangent_'+key] = np.stack(data['tangent_'+key])
    metadata = dict(schema=1, producer="dolfinx-ufl", dolfinx=dolfinx.__version__,
                    basix=basix.__version__, ufl=ufl.__version__, numpy=np.__version__,
                    cases=["reference", "quadratic_deformation"], parameters=parameters,
                    mpi_size=1, subdivisions=subdivisions, volume_quadrature="explicit Basix degree 6",
                    surface_quadrature="Basix default degree 6", tangent="derivative of nodal force")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, metadata=json.dumps(metadata), **data)
    return dict(status="exported", file=str(output), nodes=len(X), cells=len(cells),
                cases=metadata['cases'], dolfinx=dolfinx.__version__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="validation/results/dolfinx_reference.npz")
    parser.add_argument("--subdivisions", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(export(args.output, args.subdivisions), indent=2))
