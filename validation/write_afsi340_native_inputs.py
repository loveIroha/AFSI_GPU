"""Write the GPU demo's exact reference triangles for native AFSI, inside Docker."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def prepare(source):
    with np.load(source,allow_pickle=False) as archive:
        data={k:archive[k] for k in archive.files}
    meta=json.loads(str(data.pop('metadata')))
    expected=meta.pop('sha256')
    checksum=hashlib.sha256(json.dumps(meta,sort_keys=True,allow_nan=False).encode())
    for k in sorted(data):
        checksum.update(k.encode()); checksum.update(data[k].tobytes())
    if (checksum.hexdigest()!=expected or meta.get('schema')!=1 or
        meta.get('producer')!='afsi-torch-valve-mac' or meta.get('units')!='cm-g-s-per-unit-thickness'):
        raise ValueError('expected an intact valve MAC checkpoint')
    defaults=dict(width=.0212,length=.7,x_right=2.,height=1.61,C0=2e5,C1=1e6,kappa=4e5,beta=1e8)
    if any(meta['config'][k]!=v for k,v in defaults.items()):
        raise ValueError('source geometry/material differs from native demo_340 defaults')
    settings=dict(dt=1/16000,nx=256,ny=64,rho=1.,mu=.1)
    if any(meta['settings'][k]!=v for k,v in settings.items()):
        raise ValueError('GPU time step/fluid settings differ from the default comparison')
    X,cells,tags=data['X'],data['cells'],data['cell_tags']
    n=meta['vertex_count']
    if (X.dtype!=np.float64 or cells.dtype!=np.int64 or tags.dtype!=np.int64 or
        X.shape!=(len(X),2) or cells.shape!=(len(cells),6) or tags.shape!=(len(cells),) or
        not np.isfinite(X).all() or not 3<=n<len(X) or np.any(cells<0) or np.any(cells>=len(X)) or
        set(np.unique(cells[:,:3]))!=set(range(n)) or set(tags)!={1,11}):
        raise ValueError('invalid valve mesh')
    return X[:n],cells[:,:3],tags,meta,expected


def write(source,output):
    import dolfinx
    import ufl
    from basix.ufl import element
    from dolfinx import mesh,io,fem
    from mpi4py import MPI
    if MPI.COMM_WORLD.size!=1 or np.dtype(dolfinx.default_scalar_type)!=np.dtype(np.float64):
        raise RuntimeError('serial real FP64 DOLFINx required')
    output=Path(output)
    if output.exists():
        raise FileExistsError(f'input already exists; select a new directory: {output}')
    X,cells,tags,meta,sha=prepare(source)
    domain=mesh.create_mesh(MPI.COMM_WORLD,cells=cells.copy(),x=X.copy(),
                           e=ufl.Mesh(element('Lagrange','triangle',1,shape=(2,))))
    domain.name='mesh'
    domain.topology.create_connectivity(1,2)
    domain.topology.create_connectivity(1,0)
    domain.topology.create_connectivity(0,2)
    V=fem.functionspace(domain,element('Lagrange','triangle',2,shape=(2,)))
    p2_nodes=V.dofmap.index_map.size_global
    count=domain.topology.index_map(2).size_local
    original=domain.topology.original_cell_index[:count]
    ct=mesh.meshtags(domain,2,np.arange(count,dtype=np.int32),tags[original].astype(np.int32))
    ct.name='Cell markers'
    facets=[]; values=[]
    for marker,y in ((4,0.),(15,1.61)):
        ids=mesh.locate_entities_boundary(domain,1,lambda x,y=y:np.isclose(x[1],y,rtol=0,atol=1e-12))
        if not len(ids):
            raise ValueError(f'missing valve root {marker}')
        facets.extend(ids); values.extend([marker]*len(ids))
    order=np.argsort(facets)
    ft=mesh.meshtags(domain,1,np.asarray(facets,dtype=np.int32)[order],np.asarray(values,dtype=np.int32)[order])
    ft.name='Facet markers'
    output.mkdir(parents=True)
    with io.XDMFFile(MPI.COMM_WORLD,str(output/'mesh-340.xdmf'),'w') as file:
        file.write_mesh(domain)
        file.write_meshtags(ft,domain.geometry)
        file.write_meshtags(ct,domain.geometry)
    # Read the exact objects used by the native demo before starting a long run.
    with io.XDMFFile(MPI.COMM_WORLD,str(output/'mesh-340.xdmf'),'r') as file:
        check=file.read_mesh(name='mesh')
        check.topology.create_connectivity(1,2)
        c=file.read_meshtags(check,'Cell markers')
        f=file.read_meshtags(check,'Facet markers')
        if set(c.values)!={1,11} or set(f.values)!={4,15}:
            raise ValueError('XDMF marker round trip failed')
    report=dict(source_sha256=sha,source=str(source),config=meta['config'],
                reference_vertices=len(X),triangles=len(cells),solid_p2_nodes=p2_nodes,dolfinx=dolfinx.__version__,
                same_gpu_reference_mesh=True,initial_state='undeformed; zero fluid velocity and force')
    (output/'input_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(json.dumps(write(args.source,args.output),indent=2))
