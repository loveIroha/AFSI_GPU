"""Independent UFL/DOLFINx Q2/Q1 actions on a physical hexahedral box.

No torch or project fluid modules are imported. No essential BCs are imposed:
the full boundary loads must be preserved in the gradient/transpose comparison.
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


def export(output, subdivisions=1):
    if MPI.COMM_WORLD.size != 1 or np.dtype(dolfinx.default_scalar_type) != np.dtype(np.float64):
        raise RuntimeError('reference requires one MPI rank and real float64')
    if not dolfinx.__version__.startswith('0.10.'):
        raise RuntimeError('use the pinned DOLFINx 0.10.x environment')
    if subdivisions < 1:
        raise ValueError('subdivisions must be positive')
    counts = (2*subdivisions, subdivisions, subdivisions)
    origin, lengths = np.array([-.3, .2, -.5]), np.array([1.4, 1.1, .9])
    domain = mesh.create_box(MPI.COMM_WORLD, [origin, origin+lengths], counts,
                             cell_type=mesh.CellType.hexahedron)
    V = fem.functionspace(domain, basix.ufl.element('Lagrange', 'hexahedron', 2,
        shape=(3,), lagrange_variant=basix.LagrangeVariant.equispaced))
    Q = fem.functionspace(domain, basix.ufl.element('Lagrange', 'hexahedron', 1,
        lagrange_variant=basix.LagrangeVariant.equispaced))
    if V.dofmap.index_map_bs != 3 or Q.dofmap.index_map_bs != 1:
        raise RuntimeError('unexpected velocity/pressure blocking')
    u, p, f, direction = fem.Function(V), fem.Function(Q), fem.Function(V), fem.Function(V)
    v, q = ufl.TestFunction(V), ufl.TestFunction(Q)
    # Overintegrate relative to the independent torch 4x4x4 Gauss rule.
    dx = ufl.Measure('dx', domain=domain, metadata={'quadrature_degree': 8})
    forms = dict(velocity_mass=ufl.inner(u, v)*dx,
        velocity_stiffness=ufl.inner(ufl.grad(u), ufl.grad(v))*dx,
        pressure_mass=p*q*dx, pressure_stiffness=ufl.inner(ufl.grad(p), ufl.grad(q))*dx,
        divergence=q*ufl.div(u)*dx, gradient=ufl.inner(v, ufl.grad(p))*dx,
        divergence_transpose=p*ufl.div(v)*dx,
        convection=ufl.inner(ufl.dot(ufl.grad(u), u), v)*dx,
        density_load=ufl.inner(f, v)*dx)
    forms['convection_tangent'] = ufl.derivative(forms['convection'], u, direction)
    compiled = {name: fem.form(form) for name, form in forms.items()}
    scalar = {'pressure_mass', 'pressure_stiffness', 'divergence'}
    data = dict(velocity_coordinates=V.tabulate_dof_coordinates().copy(),
                pressure_coordinates=Q.tabulate_dof_coordinates().copy())
    for name in ('u', 'p', 'density', 'direction', *forms):
        data[name] = []
    expressions = [
        (lambda x: np.array([.3+.4*x[0]+.7*x[1]-.2*x[2],
                            .1+.3*x[0]-.6*x[1]+.1*x[2], -.2+.5*x[1]+.9*x[2]]),
         lambda x: 1+.2*x[0]-.4*x[1]+.6*x[2]),
        (lambda x: np.array([np.sin(1.3*x[0]+.7*x[1])+.2*x[2]**2,
                            np.cos(.6*x[1]-1.2*x[2])+.1*x[0]*x[1],
                            np.sin(x[0]+x[1]+x[2])+.3*x[0]**2*x[1]**2*x[2]**2]),
         lambda x: np.cos(.8*x[0]+.9*x[1]-1.1*x[2]))]
    for velocity, pressure in expressions:
        u.interpolate(velocity)
        p.interpolate(pressure)
        f.interpolate(lambda x: np.array([np.cos(x[0]+x[2]), np.sin(x[1]), .2+x[0]*x[1]*x[2]]))
        direction.interpolate(lambda x: np.array([.3+x[1]**2, -.2+x[0]*x[2], .1+x[0]**2*x[2]**2]))
        for function in (u, p, f, direction):
            function.x.scatter_forward()
        for name, function in [('u', u), ('p', p), ('density', f), ('direction', direction)]:
            data[name].append(function.x.array.copy().reshape((-1,) if name == 'p' else (-1, 3)))
        for name, form in compiled.items():
            result = fem.assemble_vector(form)
            result.scatter_reverse(la.InsertMode.add)
            data[name].append(result.array.copy().reshape((-1,) if name in scalar else (-1, 3)))
    for name in ('u', 'p', 'density', 'direction', *forms):
        data[name] = np.stack(data[name])
    metadata = dict(schema=1, producer='dolfinx-ufl-fluid', dolfinx=dolfinx.__version__,
        basix=basix.__version__, ufl=ufl.__version__, counts=counts, lengths=lengths.tolist(),
        origin=origin.tolist(), cases=['affine', 'nonlinear_nodal'], terms=list(forms),
        quadrature_degree=8, boundary_conditions='none', length_unit='cm')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, metadata=json.dumps(metadata), **data)
    return dict(status='exported', file=str(output), cells=int(np.prod(counts)),
                velocity_nodes=len(data['velocity_coordinates']), pressure_nodes=len(data['pressure_coordinates']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='validation/results/fluid_coarse.npz')
    parser.add_argument('--subdivisions', type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(export(args.output, args.subdivisions), indent=2))
