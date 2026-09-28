"""Self-contained numeric checkpoints for the MAC experiment (not FEM flow)."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from ..afsi337 import AFSI337Loads
from ..geometry import LVConfig
from ..geometry.ellipsoid import LVMesh
from ..geometry.fibers import FiberField
from ..lv_model import LVSolid
from ..materials import GuccioneParameters
from .coupling import MACState


def digest(metadata, data):
    result=hashlib.sha256(json.dumps(metadata,sort_keys=True,allow_nan=False).encode())
    for name in sorted(data):
        result.update(name.encode())
        result.update(data[name].tobytes())
    return result.hexdigest()


def save_mac(path, model, state, settings, progress):
    array=lambda t:t.detach().cpu().numpy()
    mesh=model.mesh
    data={name:array(getattr(mesh,name)) for name in ('X','cells','faces','facet_tags')}
    data.update({name:array(getattr(model.fibers,name)) for name in
                 ('fiber','sheet','transmural','helix_angle','apical_weight')})
    data.update({name:array(getattr(state,name)) for name in ('x','pressure','force')})
    data.update({f'velocity_{c}':array(u) for c,u in enumerate(state.velocity)})
    for name in ('volume_quadrature','surface_quadrature'):
        rule=getattr(model,name)
        if rule is not None:
            data[name+'_points'],data[name+'_weights']=map(array,rule)
    metadata=dict(producer='afsi-torch-mac',schema=1,units='cm-g-s',
        geometry=asdict(mesh.config),vertex_count=mesh.vertex_count,gmsh=mesh.gmsh_version,
        material=asdict(model.parameters),beta=model.beta,loads=asdict(model.loads),
        fiber_metadata=model.fiber_metadata,step=state.step,time=state.time,force_time=state.force_time,
        settings=settings,progress=progress)
    metadata['sha256']=digest(metadata,data)
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,metadata=json.dumps(metadata,allow_nan=False),**data)
    temporary.replace(path)


def load_mac(path, device='cpu'):
    with np.load(path,allow_pickle=False) as archive:
        data={name:archive[name] for name in archive.files}
    metadata=json.loads(str(data.pop('metadata')))
    checksum=metadata.pop('sha256')
    if (metadata['producer']!='afsi-torch-mac' or metadata['schema']!=1 or
            metadata['units']!='cm-g-s' or digest(metadata,data)!=checksum):
        raise ValueError('invalid MAC checkpoint or checksum')
    for name,value in data.items():
        dtype=np.int64 if name in ('cells','faces','facet_tags') else np.float64
        if value.dtype!=dtype or not np.isfinite(value).all():
            raise ValueError('invalid MAC checkpoint array')
    tensor=lambda name:torch.as_tensor(data[name],device=device)
    mesh=LVMesh(LVConfig(**metadata['geometry']),*(tensor(name) for name in
        ('X','cells','faces','facet_tags')),metadata['vertex_count'],metadata['gmsh'])
    fields=FiberField(*(tensor(name) for name in ('fiber','sheet','transmural','helix_angle','apical_weight')))
    rules={}
    for name in ('volume_quadrature','surface_quadrature'):
        if name+'_points' in data:
            rules[name]=(tensor(name+'_points'),tensor(name+'_weights'))
    model=LVSolid(mesh,loads=AFSI337Loads(**metadata['loads']),beta=metadata['beta'],
        parameters=GuccioneParameters(**metadata['material']),fibers=fields,
        fiber_metadata=metadata['fiber_metadata'],**rules)
    state=MACState(metadata['step'],metadata['time'],tensor('x'),
        tuple(tensor(f'velocity_{c}') for c in range(3)),tensor('pressure'),tensor('force'),metadata['force_time'])
    dt=metadata['settings']['dt']
    if (type(state.step) is not int or state.step<0 or abs(state.time-state.step*dt)>1e-12 or
            (state.step==0 and state.force_time is not None) or
            (state.step>0 and (state.force_time is None or abs(state.force_time-(state.step-1)*dt)>1e-12))):
        raise ValueError('invalid MAC checkpoint clock')
    model.validate(state.x)
    expected=torch.zeros_like(state.x) if state.force_time is None else model.force(state.x,state.force_time)
    if not torch.allclose(expected,state.force,rtol=1e-10,atol=1e-7):
        raise ValueError('MAC checkpoint force is inconsistent')
    return model,state,metadata['settings'],metadata['progress']
