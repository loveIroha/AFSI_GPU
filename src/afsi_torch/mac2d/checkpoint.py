"""Numeric, checksummed valve checkpoints, independent of external mesh files."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from ..afsi340 import ValveConfig,ValveMesh,ValveSolid
from ..mac.checkpoint import digest
from .coupling import ValveState


def save(path,solid,state,settings,progress):
    array=lambda t:t.detach().cpu().numpy()
    m=solid.mesh
    data={name:array(getattr(m,name)) for name in ('X','cells','cell_tags','roots','root_tags')}
    data.update({name:array(getattr(state,name)) for name in ('x','pressure','force')})
    data.update({f'velocity_{c}':array(v) for c,v in enumerate(state.velocity)})
    metadata=dict(schema=1,producer='afsi-torch-valve-mac',units='cm-g-s-per-unit-thickness',
        config=asdict(solid.config),vertex_count=m.vertex_count,gmsh=m.gmsh_version,
        step=state.step,time=state.time,settings=settings,progress=progress)
    metadata['sha256']=digest(metadata,data)
    path=Path(path)
    temporary=path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,metadata=json.dumps(metadata,allow_nan=False),**data)
    temporary.replace(path)


def load(path,device='cpu'):
    with np.load(path,allow_pickle=False) as archive:
        data={k:archive[k] for k in archive.files}
    metadata=json.loads(str(data.pop('metadata')))
    sha=metadata.pop('sha256')
    if metadata['schema']!=1 or metadata['producer']!='afsi-torch-valve-mac' or metadata['units']!='cm-g-s-per-unit-thickness' or digest(metadata,data)!=sha:
        raise ValueError('invalid valve checkpoint/checksum')
    integer={'cells','cell_tags','roots','root_tags'}
    for name,v in data.items():
        if v.dtype!=(np.int64 if name in integer else np.float64) or not np.isfinite(v).all():
            raise ValueError('invalid valve checkpoint array')
    tensor=lambda name:torch.as_tensor(data[name],device=device)
    mesh=ValveMesh(*(tensor(k) for k in ('X','cells','cell_tags','roots','root_tags')),metadata['vertex_count'],metadata['gmsh'])
    solid=ValveSolid(mesh,ValveConfig(**metadata['config']),fused=metadata['settings']['fused'])
    state=ValveState(metadata['step'],metadata['time'],tensor('x'),
        tuple(tensor(f'velocity_{c}') for c in range(2)),tensor('pressure'),tensor('force'))
    if type(state.step) is not int or state.step<0 or abs(state.time-state.step*metadata['settings']['dt'])>1e-12:
        raise ValueError('invalid valve checkpoint clock')
    solid.validate(state.x)
    expected=torch.zeros_like(state.x) if state.step==0 else solid.force(state.x)
    if not torch.allclose(expected,state.force,rtol=1e-9,atol=1e-7):
        raise ValueError('valve checkpoint force is inconsistent')
    return solid,state,metadata['settings'],metadata['progress']
