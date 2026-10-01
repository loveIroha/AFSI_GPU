"""Numeric-only real-LV checkpoints: P1 mesh, DG0 fields and H-O model ID."""
from dataclasses import asdict, replace
from math import isfinite
import json
from pathlib import Path
import numpy as np
import torch
from .mesh_io import ImportedSolidMesh
from .real_lv import RealLVConfig, RealLVSolid
from .config import _decode, TimeConfig
from .mac.checkpoint import digest
from .mac.coupling import MACState


@torch.no_grad()
def refine_checkpoint_dt(model, state, settings, config, dt, end_time=None):
    """Retain accepted x/u/p and retime the existing lagged-force scheme.

    Only integer subdivisions of the saved dt are supported. Output intervals
    grow by that factor so their spacing in physical time does not change.
    The caller must save the continuation in a separate directory.
    """
    old_dt = config.time.dt
    if isinstance(dt, bool) or not isinstance(dt, (float, int)) or not isfinite(dt) or dt <= 0 or dt >= old_dt:
        raise ValueError('resume dt must be positive, finite and smaller than the saved dt')
    ratio = old_dt/dt
    if not isfinite(ratio):
        raise ValueError('resume dt refinement factor must be finite')
    factor = round(ratio)
    if factor < 2 or abs(factor*dt-old_dt) > 1e-12*old_dt:
        raise ValueError('saved dt must be an integer multiple of resume dt')
    if settings['dt'] != old_dt:
        raise ValueError('saved solver dt differs from the configuration')
    target = config.time.end_time if end_time is None else end_time
    time = TimeConfig(dt, target)
    if target < state.time-1e-12:
        raise ValueError('end time precedes checkpoint')
    step = state.step*factor
    if abs(step*dt-state.time) > 1e-12:
        raise ValueError('checkpoint time must be an integer multiple of resume dt')
    output = replace(config.output, **{name: getattr(config.output, name)*factor for name in
                     ('log_every', 'output_every', 'checkpoint_every')})
    config = replace(config, time=time, output=output)
    force_time = None if step == 0 else state.time if config.coupling.scheme == 'implicit-newton' else (step-1)*dt
    force = state.force if step == 0 else model.force(state.x, force_time)
    branch = replace(state, step=step, force_time=force_time, force=force)
    details = dict(old_dt_s=old_dt, new_dt_s=dt, refinement_factor=factor,
                   source_step=state.step, start_step=step, start_time_s=state.time,
                   force_time_s=force_time,force_resampled=step > 0,
                   lagged_force_resampled=step > 0 and config.coupling.scheme == 'explicit-lagged',
                   output_intervals_preserved_in_seconds=True)
    return branch, dict(settings, dt=dt), config, details


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
    expected_time = state.time if config.coupling.scheme == 'implicit-newton' else (state.step-1)*dt
    if (type(state.step) is not int or state.step < 0 or abs(state.time-state.step*dt) > 1e-12
            or (state.step == 0 and state.force_time is not None)
            or (state.step > 0 and (state.force_time is None or abs(state.force_time-expected_time) > 1e-12))):
        raise ValueError('inconsistent real LV checkpoint clock')
    if metadata['settings'].get('coupling', {}).get('scheme', 'explicit-lagged') != config.coupling.scheme:
        raise ValueError('checkpoint solver and configuration coupling schemes differ')
    model.validate(state.x)
    expected = torch.zeros_like(state.x) if state.force_time is None else model.force(state.x, state.force_time)
    if not torch.allclose(state.force, expected, rtol=1e-10, atol=1e-7):
        raise ValueError('checkpoint force is inconsistent with the H-O model')
    return model, state, metadata['settings'], metadata['progress'], config
