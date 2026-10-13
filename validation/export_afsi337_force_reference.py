"""Use the installed demo_337 material and native inputs to export loaded P2 forces.

Run serially inside the AFSI container. No simulation, uploads or mesh regeneration.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import ufl
from basix.ufl import element
from dolfinx import fem,io
from dolfinx.fem.petsc import assemble_vector
from mpi4py import MPI
from petsc4py import PETSc
from export_afsi337_solid import load_fields


def export(root,material_path,output):
    if MPI.COMM_WORLD.size!=1:
        raise RuntimeError('serial export required')
    root=Path(root).expanduser();material_path=Path(material_path)
    geometry=root/'lv_ellipsoid/geometry'
    markers=json.loads((geometry/'markers.json').read_text())
    with io.XDMFFile(MPI.COMM_WORLD,str(geometry/'mesh.xdmf'),'r') as f:
        domain=f.read_mesh(name='Mesh')
        domain.topology.create_connectivity(2,3)
        tags=f.read_meshtags(domain,name='Facet tags')
    V=fem.functionspace(domain,element('Lagrange','tetrahedron',2,shape=(3,)))
    raw=V.tabulate_dof_coordinates().copy()
    fiber,sheet=load_fields(root,raw)
    f0,s0,x=(fem.Function(V) for _ in range(3))
    f0.x.array[:]=fiber.ravel();s0.x.array[:]=sheet.ravel()
    domain.geometry.x[:]=domain.geometry.x/10+np.array([3.5,2.5,2.5])
    X=V.tabulate_dof_coordinates().copy()
    spec=importlib.util.spec_from_file_location('native_guccione',material_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    material=module.GuccioneMaterial(kappa=5e5,deviatoric=False,contraction=True,
        C=20000.,bf=8.,bt=2.,bfs=4.,f0=f0,s0=s0,n0=ufl.cross(s0,f0))
    v=ufl.TestFunction(V);F=ufl.grad(x)
    pressure=fem.Constant(domain,PETSc.ScalarType(0.))
    tension=fem.Constant(domain,PETSc.ScalarType(0.))
    dx=ufl.Measure('dx',domain=domain,metadata={'quadrature_degree':4})
    ds=ufl.Measure('ds',domain=domain,subdomain_data=tags,metadata={'quadrature_degree':4})
    P=material.first_piola_kirchhoff_stress_v1(domain,x)+tension*F*ufl.outer(f0,f0)
    form=fem.form(-ufl.inner(P,ufl.grad(v))*dx
        -5e5*ufl.inner(x-ufl.SpatialCoordinate(domain),v)*ds(markers['BASE'][0])
        -pressure*ufl.inner(ufl.cofac(F)*ufl.FacetNormal(domain),v)*ds(markers['ENDO'][0]))
    states=[];forces=[]
    times=[0.,.5,1.5]
    for t,amplitude in zip(times,[0.,.025,.06]):
        current=X+amplitude*np.column_stack((np.sin(X[:,0])*np.cos(X[:,1]),
            .6*np.sin(1.3*X[:,1])*.5*X[:,2],-.4*np.cos(X[:,0])*np.sin(X[:,2])))
        x.x.array[:]=current.ravel();x.x.scatter_forward()
        pressure.value=150000.*min(t/1.5,1.);tension.value=600000.*min(t/1.5,1.)
        b=assemble_vector(form)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,mode=PETSc.ScatterMode.REVERSE)
        forces.append(b.array.copy().reshape(-1,3));states.append(current);b.destroy()
    metadata=dict(producer='afsi337-native-force',schema=1,
        material_source=str(material_path),material_sha256=hashlib.sha256(material_path.read_bytes()).hexdigest(),
        kappa=5e5,beta=5e5,quadrature_degree=4,units='cm-g-s')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,metadata=json.dumps(metadata),reference_coordinates=X,
        coordinates=np.array(states),force=np.array(forces),time=np.array(times),fiber=fiber,sheet=sheet)
    return dict(output=str(output),states=len(states),nodes=len(X),metadata=metadata)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',default='/root/afsi-data/337_ideal_left_ventricle')
    p.add_argument('--material',default='/root/afsi/afsic/demo/demo_337/Guccione.py')
    p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(export(a.data_root,a.material,a.output),indent=2))
