"""Numeric-only real-LV checkpoints: P1 mesh, DG0 fields and H-O model ID."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from .mesh_io import ImportedSolidMesh
from .real_lv import RealLVConfig, RealLVSolid
from .config import _decode
from .mac.checkpoint import digest
from .mac.coupling import MACState


def save_real_lv(path, model, state, settings, progress, config):
    array = lambda t: t.detach().cpu().numpy()
    mesh = model.mesh
    data = {name: array(getattr(mesh, name)) for name in
            ('X', 'cells', 'faces', 'facet_tags', 'boundary_cells', 'boundary_local_facets', 'fiber', 'sheet')}
    data.update({name: array(getattr(state, name)) for name in ('x', 'pressure', 'force')})
    data.update({f'velocity_{c}': array(u) for c, u in enumerate(state.velocity)})
    metadata = dict(producer='afsi-torch-real-lv-mac', schema=1, units='cm-g-s',
                    material_model='user-HO-iso-I1-DG0', element='P1', direction_location='cell-DG0',
                    config=asdict(config), mesh_metadata=mesh.metadata, vertex_count=mesh.vertex_count,
                    step=state.step, time=state.time, force_time=state.force_time,
                    settings=settings, progress=progress)
    metadata['sha256'] = digest(metadata, data)
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, metadata=json.dumps(metadata, allow_nan=False), **data)
    temporary.replace(path)


def load_real_lv(path, device='cpu'):
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(data.pop('metadata')))
    checksum = metadata.pop('sha256')
    if (metadata.get('producer') != 'afsi-torch-real-lv-mac' or metadata.get('schema') != 1
            or metadata.get('units') != 'cm-g-s' or metadata.get('element') != 'P1'
            or metadata.get('material_model') != 'user-HO-iso-I1-DG0'
            or metadata.get('direction_location') != 'cell-DG0' or digest(metadata, data) != checksum):
        raise ValueError('invalid real LV checkpoint or checksum')
    integer = {'cells', 'faces', 'facet_tags', 'boundary_cells', 'boundary_local_facets'}
    for name, value in data.items():
        if value.dtype != (np.int64 if name in integer else np.float64) or not np.isfinite(value).all():
            raise ValueError('invalid checkpoint array dtype or value')
    tensor = lambda name: torch.as_tensor(data[name], device=device)
    mesh = ImportedSolidMesh(*(tensor(name) for name in
        ('X', 'cells', 'faces', 'facet_tags', 'boundary_cells', 'boundary_local_facets', 'fiber', 'sheet')),
        metadata['vertex_count'], metadata['mesh_metadata'])
    config = _decode(RealLVConfig, metadata['config'])
    model = RealLVSolid(mesh, config)
    state = MACState(metadata['step'], metadata['time'], tensor('x'),
                     tuple(tensor(f'velocity_{c}') for c in range(3)), tensor('pressure'),
                     tensor('force'), metadata['force_time'])
    dt = config.time.dt
    if (type(state.step) is not int or state.step < 0 or abs(state.time-state.step*dt) > 1e-12
            or (state.step == 0 and state.force_time is not None)
            or (state.step > 0 and (state.force_time is None or abs(state.force_time-(state.step-1)*dt) > 1e-12))):
        raise ValueError('inconsistent real LV checkpoint clock')
    model.validate(state.x)
    expected = torch.zeros_like(state.x) if state.force_time is None else model.force(state.x, state.force_time)
    if not torch.allclose(state.force, expected, rtol=1e-10, atol=1e-7):
        raise ValueError('checkpoint force is inconsistent with the H-O model')
    return model, state, metadata['settings'], metadata['progress'], config
