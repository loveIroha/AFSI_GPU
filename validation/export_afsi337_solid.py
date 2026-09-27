"""Convert demo_337 external solid/fiber data in serial DOLFINx; no FSI run.

Standalone: only numpy, DOLFINx, Basix and MPI from the AFSI container required.
Coordinate lookup reproduces ReadFibers.py's hash; ambiguous keys are rejected.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

EDGES = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])


def coordinate_key(point):
    value = round(100*point[0]+10*point[1]+point[2], 6)
    return int(hashlib.sha256(f'{value:.6f}'.encode()).hexdigest()[:8], 16)


def load_fields(root, coordinates):
    fiber = np.loadtxt(root/'f0.txt').reshape(-1,3)
    sheet = np.loadtxt(root/'s0.txt').reshape(-1,3)
    keys = np.loadtxt(root/'cdm.txt', dtype=np.int64).reshape(-1)
    if len(fiber) != len(keys) or sheet.shape != fiber.shape or len(np.unique(keys)) != len(keys):
        raise ValueError('inconsistent fiber data or ambiguous source coordinate hashes')
    query = np.array([coordinate_key(x) for x in coordinates], dtype=np.int64)
    if len(np.unique(query)) != len(query):
        raise ValueError('target coordinates have ambiguous hashes; cannot safely reproduce mapping')
    order = np.argsort(keys)
    index = np.searchsorted(keys[order], query)
    if np.any(index >= len(keys)) or not np.array_equal(keys[order[index]], query):
        raise ValueError('P2 coordinate missing from cdm.txt; use original unscaled millimetre mesh')
    return fiber[order[index]], sheet[order[index]]


def match(query, candidates):
    distance = np.linalg.norm(query[:,None,:]-candidates[None,:,:],axis=-1)
    near = distance < 1e-9  # raw mm coordinates
    if not np.all(near.sum(1) == 1):
        raise ValueError('cannot uniquely match local P2 coordinates')
    result = np.argmax(near, axis=1)
    if len(np.unique(result)) != len(result):
        raise ValueError('repeated local coordinate match')
    return result


def export(root, output):
    import basix
    import dolfinx
    from basix.ufl import element
    from dolfinx import fem, mesh, io
    from mpi4py import MPI
    if MPI.COMM_WORLD.size != 1 or np.dtype(dolfinx.default_scalar_type) != np.dtype('float64'):
        raise RuntimeError('run using serial real-float64 DOLFINx')
    root = Path(root).expanduser()
    geometry = root/'lv_ellipsoid'/'geometry'
    markers = json.loads((geometry/'markers.json').read_text())
    with io.XDMFFile(MPI.COMM_WORLD, str(geometry/'mesh.xdmf'), 'r') as stream:
        domain = stream.read_mesh(name='Mesh')
        if domain.topology.cell_name() != 'tetrahedron':
            raise ValueError('tetrahedral solid required')
        domain.topology.create_connectivity(2,3)
        tags = stream.read_meshtags(domain, name='Facet tags')
    space = fem.functionspace(domain, element('Lagrange', 'tetrahedron', 2, shape=(3,)))
    Xraw = space.tabulate_dof_coordinates().copy()
    fiber, sheet = load_fields(root, Xraw)
    vertices = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    nodes = np.concatenate((vertices, vertices[EDGES].mean(1)))
    cells = []
    for cell in range(domain.topology.index_map(3).size_local):
        physical = domain.geometry.cmap.push_forward(nodes,domain.geometry.x[domain.geometry.dofmap[cell]])
        dofs = space.dofmap.cell_dofs(cell)
        cells.append(dofs[match(physical,Xraw[dofs])])
    cells = np.asarray(cells,dtype=np.int64)
    domain.topology.create_connectivity(2,0)
    connectivity = domain.topology.connectivity(2,0)
    tagged_vertices, facet_tags = [], []
    for name, label in (('ENDO',1),('EPI',2),('BASE',3)):
        original = markers[name][0]
        for facet in tags.find(original):
            dofs = fem.locate_dofs_topological(space,2,np.array([facet],dtype=np.int32))
            vertex_ids = connectivity.links(facet)
            geometry_ids = mesh.entities_to_geometry(domain,0,vertex_ids,False).reshape(-1)
            corners = domain.geometry.x[geometry_ids]
            tagged_vertices.append(dofs[match(corners,Xraw[dofs])])
            facet_tags.append(label)
    q,w = basix.make_quadrature(basix.CellType.tetrahedron,4)
    sq,sw = basix.make_quadrature(basix.CellType.triangle,4)
    files = [root/name for name in ('f0.txt','s0.txt','cdm.txt')]
    files += [geometry/'mesh.xdmf',geometry/'markers.json']
    files += sorted(geometry.glob('*.h5'))
    metadata = dict(schema=1,producer='afsi337-solid-export',units='cm',
        dolfinx=dolfinx.__version__,basix=basix.__version__,
        coordinate_transform='Xcm=Xmm/10+(3.5,2.5,2.5); vector components unchanged',
        files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        source_root=str(root),quadrature='installed Basix default degree 4')
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    data = dict(X=Xraw/10+np.array([3.5,2.5,2.5]),cells=cells,
        tagged_vertices=np.asarray(tagged_vertices,dtype=np.int64),
        facet_tags=np.asarray(facet_tags,dtype=np.int64),fiber=fiber,sheet=sheet,
        volume_points=q,volume_weights=w,surface_points=sq,surface_weights=sw)
    if not all(np.isfinite(a).all() for a in data.values()):
        raise ValueError('nonfinite exported data')
    with output.open('wb') as stream:
        np.savez_compressed(stream,metadata=json.dumps(metadata),**data)
    return dict(file=str(output),solid_nodes=len(Xraw),solid_cells=len(cells),metadata=metadata)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',default='~/afsi-data/337_ideal_left_ventricle')
    parser.add_argument('--output',default='/tmp/afsi337_solid.npz')
    args = parser.parse_args()
    print(json.dumps(export(args.data_root,args.output),indent=2))

