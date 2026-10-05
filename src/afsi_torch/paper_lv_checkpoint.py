"""Checksummed paper BE-BE checkpoints; incompatible with old CNAB runs."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from .mac.checkpoint import digest
from .config import _decode
from .mesh_io import ImportedSolidMesh
from .paper_lv import PaperLVConfig, PaperLVSolid
from .mac.paper_coupling import PaperState


def save(path, model, state, config, progress):
    array = lambda t: t.detach().cpu().numpy()
    names = ('X', 'cells', 'faces', 'facet_tags', 'boundary_cells', 'boundary_local_facets', 'fiber', 'sheet')
    data = {n: array(getattr(model.mesh, n)) for n in names}
    data.update({n: array(getattr(state, n)) for n in ('x', 'pressure', 'force')})
    data.update({f'velocity_{c}': array(v) for c, v in enumerate(state.velocity)})
    if state.previous_x is not None:
        data['previous_x'] = array(state.previous_x)
    metadata = dict(producer='afsi-torch-ma2024-real-lv', schema=1, units='cm-g-s',
        material_model='paper-eq81-eq82-raw-I1', time_scheme='BE-BE-semilagrangian',
        config=asdict(config), progress=progress, mesh_metadata=model.mesh.metadata,
        vertex_count=model.mesh.vertex_count, step=state.step, time=state.time,
        force_time=state.force_time, pressure_time=state.pressure_time)
    metadata['sha256'] = digest(metadata, data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, metadata=json.dumps(metadata, allow_nan=False), **data)
    temporary.replace(path)


def load(path, device='cpu'):
    with np.load(path, allow_pickle=False) as archive:
        data = {n: archive[n] for n in archive.files}
    metadata = json.loads(str(data.pop('metadata')))
    checksum = metadata.pop('sha256')
    if (metadata.get('producer') != 'afsi-torch-ma2024-real-lv' or metadata.get('schema') != 1
            or metadata.get('material_model') != 'paper-eq81-eq82-raw-I1'
            or metadata.get('time_scheme') != 'BE-BE-semilagrangian'
            or metadata.get('units') != 'cm-g-s' or digest(metadata, data) != checksum):
        raise ValueError('paper BE-BE checkpoint required; old CNAB/active-cycle checkpoints cannot be resumed')
    integers = {'cells', 'faces', 'facet_tags', 'boundary_cells', 'boundary_local_facets'}
    for n, a in data.items():
        if a.dtype != (np.int64 if n in integers else np.float64) or not np.isfinite(a).all():
            raise ValueError('invalid paper checkpoint array')
    tensor = lambda n: torch.as_tensor(data[n], device=device)
    mesh = ImportedSolidMesh(*(tensor(n) for n in ('X', 'cells', 'faces', 'facet_tags',
        'boundary_cells', 'boundary_local_facets', 'fiber', 'sheet')), metadata['vertex_count'], metadata['mesh_metadata'])
    config = _decode(PaperLVConfig, metadata['config'])
    model = PaperLVSolid(mesh, config)
    state = PaperState(metadata['step'], metadata['time'], tensor('x'),
        tuple(tensor(f'velocity_{c}') for c in range(3)), tensor('pressure'), tensor('force'),
        metadata['force_time'], pressure_time=metadata['pressure_time'],
        previous_x=tensor('previous_x') if 'previous_x' in data else None)
    if (type(state.step) is not int or state.step < 0 or abs(state.time-state.step*config.time.dt) > 1e-12
            or state.force_time != state.time or (state.step > 0 and (state.pressure_time != state.time or state.previous_x is None))):
        raise ValueError('invalid paper checkpoint clock or predictor history')
    model.validate(state.x)
    from .mac.grid import MACGrid
    grid = MACGrid(config.fluid.shape, config.fluid.lengths, config.fluid.origin)
    grid.check_velocity(state.velocity)
    if state.pressure.shape != grid.shape or state.force.shape != state.x.shape:
        raise ValueError('paper checkpoint field shape mismatch')
    if not torch.allclose(state.force, model.force(state.x, state.time), rtol=1e-10, atol=1e-7):
        raise ValueError('paper checkpoint force disagrees with material/load time')
    return model, state, config, metadata['progress']
