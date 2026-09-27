"""Atomic, numeric-only checkpoints for a prescribed-load LV trajectory."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import boundary as bd
from .coupling import CoupledState
from .cycle_loads import AFSICycleLoads
from .geometry import LVConfig
from .geometry.ellipsoid import LVMesh
from .lv_model import LVSolid
from .materials import GuccioneParameters
from .units import CGS_UNITS


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def save_cycle(path, model, state, x_start, settings, progress):
    """Keep the reference X and exact lagged force/time, including bootstrap."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = lambda x: x.detach().cpu().numpy()
    mesh = model.mesh
    data = dict(X=array(mesh.X), cells=array(mesh.cells), faces=array(mesh.faces),
        facet_tags=array(mesh.facet_tags), fiber=array(model.fibers.fiber),
        sheet=array(model.fibers.sheet), x_start=array(x_start))
    data.update({name: array(getattr(state, name)) for name in ('x', 'velocity', 'pressure', 'force')})
    metadata = dict(schema=1, producer='afsi-torch-lv-cycle', units=CGS_UNITS,
        geometry=asdict(mesh.config), vertex_count=mesh.vertex_count, gmsh=mesh.gmsh_version,
        material=asdict(model.parameters), beta=model.beta, loads=asdict(model.loads),
        step=state.step, time=state.time, force_time=state.force_time,
        settings=settings, progress=progress)
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
    for name in sorted(data):
        digest.update(name.encode())
        digest.update(data[name].tobytes())
    metadata['sha256'] = digest.hexdigest()
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, metadata=json.dumps(metadata, allow_nan=False), **data)
    temporary.replace(path)


def load_cycle(path, device='cpu'):
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(data.pop('metadata')))
    saved_digest = metadata.pop('sha256')
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
    for name in sorted(data):
        digest.update(name.encode())
        digest.update(data[name].tobytes())
    if digest.hexdigest() != saved_digest:
        raise ValueError('cycle checkpoint checksum mismatch')
    if (metadata['schema'] != 1 or metadata['producer'] != 'afsi-torch-lv-cycle' or
            metadata['units'] != CGS_UNITS):
        raise ValueError('unsupported cycle checkpoint')
    for name, value in data.items():
        integer = name in ('cells', 'faces', 'facet_tags')
        if value.dtype != (np.int64 if integer else np.float64) or not np.isfinite(value).all():
            raise ValueError(f'invalid checkpoint field {name}')
    tensor = lambda name: torch.as_tensor(data[name], device=device)
    config = LVConfig(**metadata['geometry'])
    X, cells, faces, tags = (tensor(name) for name in ('X', 'cells', 'faces', 'facet_tags'))
    if (not torch.equal(bd.extract_boundary(X, cells), faces) or
            tags.shape != (len(faces),) or set(tags.cpu().tolist()) != {1, 2, 3}):
        raise ValueError('invalid cycle checkpoint boundary')
    mesh = LVMesh(config, X, cells, faces, tags, metadata['vertex_count'], metadata['gmsh'])
    model = LVSolid(mesh, loads=AFSICycleLoads(**metadata['loads']), beta=metadata['beta'],
                    parameters=GuccioneParameters(**metadata['material']))
    for name in ('fiber', 'sheet'):
        expected = tensor(name)
        actual = getattr(model.fibers, name)
        if expected.shape != actual.shape or not torch.allclose(expected, actual, rtol=0, atol=1e-12):
            raise ValueError('cycle checkpoint material fields changed')
    dt = metadata['settings']['dt']
    step, time, force_time = (metadata[name] for name in ('step', 'time', 'force_time'))
    if (type(step) is not int or step < 0 or abs(time-step*dt) > 1e-12 or
            (step > 0 and (force_time is None or abs(force_time-(step-1)*dt) > 1e-12)) or
            (step == 0 and force_time not in (None, 0.))):
        raise ValueError('cycle checkpoint has inconsistent load/step clocks')
    state = CoupledState(step, time, tensor('x'), tensor('velocity'), tensor('pressure'),
                         tensor('force'), force_time)
    model.validate(state.x)
    model.validate(tensor('x_start'))
    if force_time is None:
        expected_force = torch.zeros_like(X)
    else:
        expected_force = model.force(state.x, force_time)
    if state.force.shape != X.shape or not torch.allclose(state.force, expected_force, rtol=1e-10, atol=1e-7):
        raise ValueError('stored force disagrees with its recorded geometry and load time')
    return model, state, tensor('x_start'), metadata['settings'], metadata['progress']

