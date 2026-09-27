"""Read the optional serial DOLFINx numeric export, preserving P2 coefficients."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .afsi337 import AFSI337Loads, geometry_config
from .boundary import extract_boundary
from .geometry.ellipsoid import LVMesh
from .geometry.fibers import FiberField
from .lv_model import LVSolid


def load_native_solid(path, *, device='cpu'):
    path = Path(path)
    with np.load(path,allow_pickle=False) as archive:
        data = {name:archive[name] for name in archive.files}
    metadata = json.loads(str(data.pop('metadata')))
    if metadata.get('schema') != 1 or metadata.get('producer') != 'afsi337-solid-export' or metadata.get('units') != 'cm':
        raise ValueError('unsupported AFSI337 solid export')
    required = {'X','cells','tagged_vertices','facet_tags','fiber','sheet',
                'volume_points','volume_weights','surface_points','surface_weights'}
    if set(data) != required:
        raise ValueError('unexpected AFSI337 export fields')
    for name,value in data.items():
        dtype = np.int64 if name in ('cells','tagged_vertices','facet_tags') else np.float64
        if value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f'invalid native field {name}')
    cast = lambda name: torch.as_tensor(data[name],device=device)
    X,cells = cast('X'),cast('cells')
    faces = extract_boundary(X,cells)
    tagged,tags = data['tagged_vertices'],data['facet_tags']
    if tagged.shape != (len(faces),3) or tags.shape != (len(faces),) or set(tags.tolist()) != {1,2,3}:
        raise ValueError('invalid native boundary tags')
    mapping = {tuple(sorted(f)):int(t) for f,t in zip(tagged.tolist(),tags.tolist())}
    keys = [tuple(sorted(f)) for f in faces[:,:3].cpu().tolist()]
    if len(mapping) != len(faces) or set(mapping) != set(keys):
        raise ValueError('native tags must cover exactly the solid exterior')
    facet_tags = torch.tensor([mapping[k] for k in keys],dtype=torch.int64,device=device)
    for name in ('fiber','sheet'):
        if data[name].shape != tuple(X.shape):
            raise ValueError('native fiber/sheet shape differs from P2 nodes')
    zeros = X.new_zeros(len(X))
    fields = FiberField(cast('fiber'),cast('sheet'),zeros,zeros.clone(),zeros.clone())
    # Config records the expected companion CAD dimensions, not a remeshing
    # instruction. Actual reference coordinates and connectivity are preserved.
    mesh = LVMesh(geometry_config(),X,cells,faces,facet_tags,
                  len(torch.unique(cells[:,:4])),'native-DOLFINx')
    return LVSolid(mesh,loads=AFSI337Loads(),fibers=fields,
        volume_quadrature=(cast('volume_points'),cast('volume_weights')),
        surface_quadrature=(cast('surface_points'),cast('surface_weights')),
        fiber_metadata=dict(mode='native',external_fields_matched=True,
            export_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source=metadata,
            auxiliary_fiber_arrays='not available; visualization-only arrays set to zero',
            solid_mesh_size='external mesh preserved; config.mesh_size is not applied'))

