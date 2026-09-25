"""Validated transfer of an LV solid equilibrium to a prescribed-load startup.

Checkpoint X is the ORIGINAL reference; x is the preloaded current position.
New snapshots carry facets and material fields. Legacy v0.13 snapshots require
Gmsh only to recover tags, and must match every reference coordinate/DOF before
acceptance. Loading never re-solves the solid equilibrium.
"""
from dataclasses import asdict,replace
import hashlib
import json
from math import isfinite,isclose
from pathlib import Path
import numpy as np
import torch
from .geometry import LVConfig,generate_lv
from .geometry.ellipsoid import LVMesh
from . import boundary as bd
from .lv_model import LVSolid,PreloadedLoads
from .materials import GuccioneParameters
from .units import CGS_UNITS


def save_checkpoint(path,model,x,fraction):
    pressure,tension=model.loads.at(fraction)
    from .units import MMHG_TO_DYN_PER_CM2
    metadata=dict(schema=1,config=asdict(model.mesh.config),material=asdict(model.parameters),
        beta=model.beta,pressure_mmhg=pressure/MMHG_TO_DYN_PER_CM2,tension=tension,
        gmsh=model.mesh.gmsh_version,vertex_count=model.mesh.vertex_count,units=CGS_UNITS)
    array=lambda v:v.detach().cpu().numpy()
    np.savez_compressed(path,X=array(model.mesh.X),cells=array(model.mesh.cells),x=array(x),
        load_fraction=fraction,faces=array(model.mesh.faces),facet_tags=array(model.mesh.facet_tags),
        fiber=array(model.fibers.fiber),sheet=array(model.fibers.sheet),metadata=json.dumps(metadata))


def load_preload(directory,device='cpu'):
    """Return model, preloaded coordinates, tolerance and provenance.

    Only a fully converged final load with preserved model parameters is
    accepted. Fresh residual evaluation, not the JSON flag alone, is decisive.
    """
    folder=Path(directory)
    report_path=folder/'report.json'
    checkpoint=folder/'last_converged.npz'
    report=json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('converged') is not True or not report.get('history'):
        raise ValueError('a fully converged preload report is required')
    last=report['history'][-1]
    tol=float(last['tolerance'])
    if (not isfinite(tol) or tol<=0 or last['load_fraction']!=1. or
            not isfinite(last['residual_norm']) or last['residual_norm']>tol):
        raise ValueError('preload final load/residual did not converge')
    if report['units']!=CGS_UNITS:
        raise ValueError('checkpoint must use project CGS units')
    config=LVConfig(**report['solid_config'])
    parameters=GuccioneParameters(**report['material'])
    loads=PreloadedLoads(report['target_pressure_mmhg'],report['target_tension_dyn_per_cm2'])
    with np.load(checkpoint,allow_pickle=False) as saved:
        if saved['load_fraction'].shape!=() or float(saved['load_fraction'])!=1.:
            raise ValueError('checkpoint is not at the final load')
        if saved['X'].dtype!=np.float64 or saved['x'].dtype!=np.float64 or saved['cells'].dtype!=np.int64:
            raise ValueError('checkpoint requires float64 coordinates and int64 connectivity')
        X=torch.tensor(saved['X'],device=device,dtype=torch.float64)
        x=torch.tensor(saved['x'],device=device,dtype=torch.float64)
        cells=torch.tensor(saved['cells'],device=device,dtype=torch.int64)
        if x.shape!=X.shape or not torch.isfinite(x).all():
            raise ValueError('invalid preloaded coordinates')
        legacy='metadata' not in saved.files
        if legacy:
            recovered=generate_lv(config,device=device)
            if (X.shape!=recovered.X.shape or cells.shape!=recovered.cells.shape or
                    not torch.equal(cells,recovered.cells) or
                    not torch.allclose(X,recovered.X,atol=1e-12,rtol=0)):
                raise ValueError('legacy checkpoint mesh/numbering differs from tag reconstruction; no remapping performed')
            mesh=replace(recovered,X=X,cells=cells)
        else:
            metadata=json.loads(str(saved['metadata']))
            if (metadata['schema']!=1 or LVConfig(**metadata['config'])!=config or
                metadata['material']!=report['material'] or metadata['beta']!=report['beta'] or
                not isclose(metadata['pressure_mmhg'],report['target_pressure_mmhg'],rel_tol=1e-14,abs_tol=0) or
                metadata['tension']!=report['target_tension_dyn_per_cm2'] or metadata['units']!=CGS_UNITS):
                raise ValueError('checkpoint and report describe different preload models')
            if saved['faces'].dtype!=np.int64 or saved['facet_tags'].dtype!=np.int64:
                raise ValueError('facet arrays must have int64 dtype')
            faces=torch.tensor(saved['faces'],device=device)
            tags=torch.tensor(saved['facet_tags'],device=device)
            exterior=bd.extract_boundary(X,cells)
            if not torch.equal(exterior,faces) or tags.shape!=(len(faces),) or set(tags.cpu().tolist())!={1,2,3}:
                raise ValueError('invalid checkpoint exterior facets or tags')
            mesh=LVMesh(config,X,cells,faces,tags,int(metadata['vertex_count']),metadata['gmsh'])
        model=LVSolid(mesh,loads=loads,beta=report['beta'],parameters=parameters)
        if not legacy:
            for name in ('fiber','sheet'):
                expected=torch.as_tensor(saved[name],device=device,dtype=X.dtype)
                actual=getattr(model.fibers,name)
                if expected.shape!=actual.shape or not torch.allclose(expected,actual,atol=1e-12,rtol=0):
                    raise ValueError('reference material fields changed; checkpoint cannot be silently reinterpreted')
    if len(X)!=report['nodes'] or len(cells)!=report['cells']:
        raise ValueError('report mesh dimensions disagree with checkpoint')
    model.validate(x)
    force=model.force(x,0.)
    norm=torch.linalg.vector_norm(force).item()
    if not isfinite(norm) or norm>tol:
        raise ValueError(f'loaded preload fails fresh force balance: {norm:.6g} > {tol:.6g}')
    actual=model.diagnostics(x)
    for key,value in actual.items():
        expected=report['final'][key]
        if not isfinite(expected) or abs(value-expected)>1e-9*max(1.,abs(expected)):
            raise ValueError(f'checkpoint and report final state differ: {key}')
    return model,x,tol,dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),legacy_tag_reconstruction=legacy,
        recomputed_force_norm_dyn=norm,force_tolerance_dyn=tol,reference_rebased=False,
        equilibrium_resolved=False,reference_nodes=len(X),reference_cells=len(cells))
