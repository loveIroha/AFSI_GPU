"""Independent small P2 Guccione/follower equilibrium using UFL Jacobians.

Dense NumPy solves are ONLY an independent tiny CPU reference, not production.
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
from dolfinx import fem,mesh,la
from dolfinx.fem import petsc
from reference_io import EDGES,match_points


def export(output):
    if MPI.COMM_WORLD.size != 1 or np.dtype(dolfinx.default_scalar_type)!=np.dtype(np.float64):
        raise RuntimeError('single-rank real float64 reference required')
    domain = mesh.create_unit_cube(MPI.COMM_WORLD,1,1,1,cell_type=mesh.CellType.tetrahedron)
    V = fem.functionspace(domain,basix.ufl.element('Lagrange','tetrahedron',2,shape=(3,),
        lagrange_variant=basix.LagrangeVariant.equispaced))
    X = V.tabulate_dof_coordinates().copy()
    x = fem.Function(V)
    x.x.array[:] = X.ravel()
    x.x.scatter_forward()
    free = np.repeat(~np.isclose(X[:,0],0.),3)
    rv = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    rn = np.concatenate((rv,rv[np.array(EDGES)].mean(1)))
    cells=[]
    for c in range(domain.topology.index_map(3).size_local):
        physical = domain.geometry.cmap.push_forward(rn,domain.geometry.x[domain.geometry.dofmap[c]])
        local = V.dofmap.cell_dofs(c)
        cells.append(local[match_points(physical,X[local])])
    cells=np.asarray(cells)
    facets=mesh.locate_entities_boundary(domain,2,lambda y:np.isclose(y[0],1.))
    facets=np.sort(facets)
    tags=mesh.meshtags(domain,2,facets,np.ones(len(facets),dtype=np.int32))
    q,w=basix.make_quadrature(basix.CellType.tetrahedron,6)
    sq,sw=basix.make_quadrature(basix.CellType.triangle,6)
    dx=ufl.Measure('dx',domain=domain,metadata={'quadrature_rule':'custom','quadrature_points':q,'quadrature_weights':w})
    ds=ufl.Measure('ds',domain=domain,subdomain_data=tags,metadata={'quadrature_degree':6,'quadrature_rule':'default'})
    F=ufl.variable(ufl.grad(x))
    E=.5*(F.T*F-ufl.Identity(3))
    Q=8*E[0,0]**2+2*(E[1,1]**2+E[2,2]**2+2*E[1,2]**2)+8*(E[0,1]**2+E[0,2]**2)
    W=10000*(ufl.exp(Q)-1)+500000*(ufl.det(F)-1)**2
    test=ufl.TestFunction(V)
    R=ufl.inner(ufl.diff(W,F),ufl.grad(test))*dx+1000*ufl.inner(test,ufl.cofac(F)*ufl.FacetNormal(domain))*ds(1)
    residual_form=fem.form(R)
    jacobian_form=fem.form(ufl.derivative(R,x,ufl.TrialFunction(V)))
    history=[]
    for iteration in range(15):
        vector=fem.assemble_vector(residual_form)
        vector.scatter_reverse(la.InsertMode.add)
        r=vector.array.copy()
        norm=float(np.linalg.norm(r[free]))
        history.append(norm)
        if norm<1e-7:
            break
        matrix=petsc.assemble_matrix(jacobian_form)
        matrix.assemble()
        indptr,indices,values=matrix.getValuesCSR()
        J=np.zeros(matrix.getSize(),dtype=values.dtype)
        rows=np.repeat(np.arange(len(indptr)-1),np.diff(indptr))
        J[rows,indices]=values
        increment=np.linalg.solve(J[np.ix_(free,free)],-r[free])
        x.x.array[free]+=increment
        x.x.scatter_forward()
        matrix.destroy()
    else:
        raise RuntimeError('independent nonlinear reference did not converge')
    if not np.isfinite(x.x.array).all():
        raise RuntimeError('nonfinite reference')
    metadata=dict(producer='UFL/DOLFINx residual and full Jacobian, NumPy direct Newton',
        dolfinx=dolfinx.__version__,basix=basix.__version__,ufl=ufl.__version__,history=history,
        pressure_dyn_per_cm2=1000.,parameters=dict(C=20000.,bf=8.,bt=2.,bfs=4.,kappa=500000.))
    path=Path(output)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,X=X,cells=cells,x=x.x.array.copy().reshape(-1,3),
        volume_points=q,volume_weights=w,surface_points=sq,surface_weights=sw,metadata=json.dumps(metadata))
    return dict(status='exported',file=str(path),nodes=len(X),history=history)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='validation/results/nonlinear.npz')
    print(json.dumps(export(parser.parse_args().output),indent=2))
