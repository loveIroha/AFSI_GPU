"""Verify live native 2D IB gather/scatter against an independent global stencil.

Run under the same mpirun command used for the native demo, inside its container.
Owned nodes contribute once; expected stencils omit out-of-domain links exactly
as native AFSI does. This is not the GPU demo's reflected quadrature transfer.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def stencil(points,shape=(257,65),lengths=(8.,1.61)):
    h=np.array(lengths)/(np.array(shape)-1)
    scaled=np.asarray(points)/h
    offsets=np.array([(i,j) for i in range(4) for j in range(4)])
    nodes=np.floor(scaled-1).astype(np.int64)[:,None]+offsets
    r=np.abs(scaled[:,None]-nodes)
    root=np.sqrt(np.maximum(1+4*np.minimum(r,1)-4*np.minimum(r,1)**2,0))
    # Evaluate each branch with its own argument to avoid invalid square roots.
    phi=np.where(r<1,(3-2*r+root)/8,
        np.where(r<2,(5-2*r-np.sqrt(np.maximum(-7+12*r-4*r*r,0)))/8,0))
    valid=((nodes>=0)&(nodes<np.array(shape))).all(-1)
    weights=phi.prod(-1)*valid
    indices=np.clip(nodes[...,0],0,shape[0]-1)*shape[1]+np.clip(nodes[...,1],0,shape[1]-1)
    return indices,weights,float(np.prod(h))


def check(input_path,output):
    import dolfinx
    from dolfinx import fem,mesh,io
    from mpi4py import MPI
    from basix.ufl import element
    from afsic import IBMesh,IBInterpolation
    comm=MPI.COMM_WORLD
    fluid=mesh.create_rectangle(comm,((0.,0.),(8.,1.61)),(128,32),cell_type=mesh.CellType.quadrilateral)
    with io.XDMFFile(comm,str(input_path),'r') as file:
        solid=file.read_mesh(name='mesh')
    V=fem.functionspace(fluid,element('Lagrange','quadrilateral',2,shape=(2,)))
    W=fem.functionspace(solid,element('Lagrange','triangle',2,shape=(2,)))
    xyz,u,density=(fem.Function(V) for _ in range(3))
    X,force,velocity=(fem.Function(W) for _ in range(3))
    xyz.interpolate(lambda x:x[:2]); X.interpolate(lambda x:x[:2])
    u.interpolate(lambda x:np.array([np.sin(1.7*x[0])+.4*np.cos(2.3*x[1]),np.cos(.9*x[0])-.3*np.sin(1.1*x[1])]))
    force.interpolate(lambda x:np.array([10*np.sin(1.3*x[0]+x[1]),7*np.cos(x[0]-1.1*x[1])]))
    ib=IBMesh(0.,8.,0.,1.61,128,32,2)
    transfer=IBInterpolation(ib)
    ib.build_map(xyz._cpp_object)
    transfer.evaluate_current_points(X._cpp_object)
    transfer.fluid_to_solid(u._cpp_object,velocity._cpp_object)
    velocity.x.scatter_forward()
    transfer.solid_to_fluid(density._cpp_object,force._cpp_object)
    density.x.scatter_forward()
    ns=W.dofmap.index_map.size_local
    nf=V.dofmap.index_map.size_local
    points=X.x.array.reshape(-1,2)[:ns]
    ids,w,area=stencil(points)
    gx,gy=np.meshgrid(np.linspace(0,8,257),np.linspace(0,1.61,65),indexing='ij')
    values=np.stack([np.sin(1.7*gx)+.4*np.cos(2.3*gy),np.cos(.9*gx)-.3*np.sin(1.1*gy)],-1).reshape(-1,2)
    expected_velocity=(values[ids]*w[...,None]).sum(1)
    error_u=np.max(np.abs(expected_velocity-velocity.x.array.reshape(-1,2)[:ns]),initial=0)
    local=np.zeros_like(values)
    np.add.at(local,ids.reshape(-1),(w[...,None]*force.x.array.reshape(-1,2)[:ns,None]/area).reshape(-1,2))
    expected=np.empty_like(local)
    comm.Allreduce(local,expected,op=MPI.SUM)
    coords=xyz.x.array.reshape(-1,2)[:nf]
    nodes=np.rint(coords/np.array([8/256,1.61/64])).astype(np.int64)
    global_ids=nodes[:,0]*65+nodes[:,1]
    local_owners=np.bincount(global_ids,minlength=len(values)).astype(np.int64)
    owners=np.empty_like(local_owners); comm.Allreduce(local_owners,owners,op=MPI.SUM)
    error_f=np.max(np.abs(expected[global_ids]-density.x.array.reshape(-1,2)[:nf]),initial=0)
    error_u=comm.allreduce(float(error_u),op=MPI.MAX)
    error_f=comm.allreduce(float(error_f),op=MPI.MAX)
    scale=comm.allreduce(float(np.abs(expected).max(initial=0)),op=MPI.MAX)
    passed=bool(np.all(owners==1) and error_u<1e-10 and error_f<=1e-9+1e-11*scale)
    report=dict(passed=passed,mpi_ranks=comm.size,dolfinx=dolfinx.__version__,
                interpolation_max_abs=error_u,spread_max_abs=error_f,
                spread_tolerance=1e-9+1e-11*scale,unique_fluid_owners=bool(np.all(owners==1)),
                global_solid_nodes=comm.allreduce(ns,op=MPI.SUM),fluid_nodes=len(values),
                native_ib='owned gather -> rank 0 kernel -> owned scatter; ghost synchronization')
    if comm.rank==0:
        Path(output).write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2),flush=True)
    if not passed:
        raise RuntimeError('native MPI IB validation failed; do not run full comparison')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True); p.add_argument('--output',required=True)
    args=p.parse_args(); check(args.input,args.output)
