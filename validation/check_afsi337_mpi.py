"""Check native 3D MPI IB against an independent global Peskin stencil."""
import argparse
import json
from pathlib import Path
import numpy as np


def stencil(points, shape=(65,65,65), lengths=(5.,5.,5.)):
    shape = np.array(shape, dtype=np.int64)
    h = np.array(lengths)/(shape-1)
    scaled = np.asarray(points)/h
    offsets = np.stack(np.meshgrid(*([np.arange(4)]*3), indexing='ij'), -1).reshape(-1,3)
    nodes = np.floor(scaled-1).astype(np.int64)[:,None]+offsets
    r = np.abs(scaled[:,None]-nodes)
    phi = np.where(r < 1, (3-2*r+np.sqrt(np.maximum(1+4*r-4*r*r,0)))/8,
                   np.where(r < 2, (5-2*r-np.sqrt(np.maximum(-7+12*r-4*r*r,0)))/8,0))
    valid = ((nodes >= 0)&(nodes < shape)).all(-1)
    weights = phi.prod(-1)*valid
    nodes = np.clip(nodes,0,shape-1)
    indices = (nodes[...,0]*shape[1]+nodes[...,1])*shape[2]+nodes[...,2]
    return indices, weights, float(h.prod())


def velocity_field(x):
    return np.array([np.sin(1.7*x[0])+.4*np.cos(2.3*x[1]),
                     np.cos(.9*x[1])-.3*np.sin(1.1*x[2]),
                     np.sin(.7*x[2])+.2*np.cos(1.3*x[0])])


def check(input_path, output):
    import dolfinx
    from dolfinx import fem, mesh, io
    from mpi4py import MPI
    from basix.ufl import element
    from afsic import IBMesh3D, IBInterpolation3D
    comm = MPI.COMM_WORLD
    fluid = mesh.create_box(comm, ((0.,0.,0.),(5.,5.,5.)), (32,32,32),
                            cell_type=mesh.CellType.hexahedron)
    with io.XDMFFile(comm, str(input_path), 'r') as file:
        solid = file.read_mesh(name='Mesh')
    # Native inputs are mm before the demo's scale and translation.
    solid.geometry.x[:] = solid.geometry.x/10 + np.array([3.5,2.5,2.5])
    V = fem.functionspace(fluid, element('Lagrange','hexahedron',2,shape=(3,)))
    W = fem.functionspace(solid, element('Lagrange','tetrahedron',2,shape=(3,)))
    xyz,u,density = (fem.Function(V) for _ in range(3))
    X,force,velocity = (fem.Function(W) for _ in range(3))
    xyz.interpolate(lambda x:x[:3]); X.interpolate(lambda x:x[:3])
    u.interpolate(velocity_field)
    force.interpolate(lambda x:np.array([10*np.sin(1.3*x[0]+x[1]),
                                         7*np.cos(x[1]-1.1*x[2]),
                                         5*np.sin(.9*x[2]+x[0])]))
    ib = IBMesh3D(0.,5.,0.,5.,0.,5.,32,32,32,2)
    transfer = IBInterpolation3D(ib)
    ib.build_map(xyz._cpp_object)
    transfer.evaluate_current_points(X._cpp_object)
    transfer.fluid_to_solid(u._cpp_object,velocity._cpp_object)
    velocity.x.scatter_forward()
    transfer.solid_to_fluid(density._cpp_object,force._cpp_object)
    density.x.scatter_forward()
    ns, nf = W.dofmap.index_map.size_local, V.dofmap.index_map.size_local
    points = X.x.array.reshape(-1,3)[:ns]
    ids,w,volume = stencil(points)
    gx,gy,gz = np.meshgrid(*([np.linspace(0,5,65)]*3),indexing='ij')
    values = velocity_field(np.array([gx,gy,gz])).reshape(3,-1).T
    expected_velocity = (values[ids]*w[...,None]).sum(1)
    error_u = np.max(np.abs(expected_velocity-velocity.x.array.reshape(-1,3)[:ns]),initial=0)
    local = np.zeros_like(values)
    np.add.at(local,ids.reshape(-1),(w[...,None]*force.x.array.reshape(-1,3)[:ns,None]/volume).reshape(-1,3))
    expected = np.empty_like(local)
    comm.Allreduce(local,expected,op=MPI.SUM)
    coords = xyz.x.array.reshape(-1,3)[:nf]
    nodes = np.rint(coords/(5/64)).astype(np.int64)
    global_ids = (nodes[:,0]*65+nodes[:,1])*65+nodes[:,2]
    owners_local = np.bincount(global_ids,minlength=len(values)).astype(np.int64)
    owners = np.empty_like(owners_local)
    comm.Allreduce(owners_local,owners,op=MPI.SUM)
    error_f = np.max(np.abs(expected[global_ids]-density.x.array.reshape(-1,3)[:nf]),initial=0)
    error_u = comm.allreduce(float(error_u),op=MPI.MAX)
    error_f = comm.allreduce(float(error_f),op=MPI.MAX)
    scale = float(np.abs(expected).max(initial=0))
    passed = bool(np.all(owners==1) and error_u<1e-10 and error_f<=1e-9+1e-11*scale)
    report = dict(passed=passed,mpi_ranks=comm.size,dolfinx=dolfinx.__version__,
                  interpolation_max_abs=error_u,spread_max_abs=error_f,
                  spread_tolerance=1e-9+1e-11*scale,unique_fluid_owners=bool(np.all(owners==1)),
                  global_solid_nodes=comm.allreduce(ns,op=MPI.SUM),fluid_nodes=len(values),
                  native_ib='owned gather -> rank 0 C++ kernel -> owned scatter; ghost synchronization')
    if comm.rank==0:
        Path(output).write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2),flush=True)
    if not passed:
        raise RuntimeError('3D MPI IB preflight failed; full run has not started')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    check(args.input,args.output)
