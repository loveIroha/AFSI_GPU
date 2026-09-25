"""Fixed-preload IB phase and FE power probe; production solver is unchanged.

Moving only interaction points relative to the same fluid box is a diagnostic:
it also changes their distance to outer boundaries. It is not a translated FSI
trajectory or a mathematical proof of periodic-grid phase error.
"""
import argparse
import json
from math import isfinite
from pathlib import Path
import numpy as np
import torch
from afsi_torch import ib
from afsi_torch.fluid import create_box,prepare_operators,ChorinSolver
if __package__:
    from .diagnose_ib import scaled_stencil
else:
    from diagnose_ib import scaled_stencil


PHASES={'base':(0.,0.,0.),'x_quarter':(.25,0.,0.),
        'x_half':(.5,0.,0.),'diagonal_half':(.5,.5,.5)}


def ratio(a,b,floor=1e-14):
    size=torch.linalg.vector_norm(b).item()
    return None if size<=floor else torch.linalg.vector_norm(a-b).item()/size


def _save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


@torch.no_grad()
def run(study='results/preloaded_ib_study',device='cpu',output='results/preloaded_ib_phase',levels=(12,18),dt=2.5e-5):
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if type(dt) not in (int,float) or not isfinite(dt) or dt<=0:
        raise ValueError('positive finite dt required')
    root=Path(study)
    coupled=json.loads((root/'report.json').read_text(encoding='utf-8'))
    frozen=json.loads((root/'frozen'/'diagnosis.json').read_text(encoding='utf-8'))
    if not coupled['completed'] or not frozen['completed'] or not coupled['frozen_completed']:
        raise ValueError('completed preloaded frozen/coupled study required')
    source=frozen['metadata']['preload']['provenance']
    for key in ('checkpoint_sha256','report_sha256','reference_nodes','reference_cells'):
        if source[key]!=coupled['preload'][key]:
            raise ValueError('frozen and coupled studies use different preloads')
    if (tuple(levels)!=(12,18) or list(levels)!=sorted(set(levels)) or
        any(type(n) is not int or n not in frozen['metadata']['levels'] for n in levels)):
        raise ValueError('phase study requires the recorded 12 and 18 cell levels')
    if abs(dt-frozen['metadata']['dt_s'])>1e-15:
        raise ValueError('phase step must match frozen baseline dt')
    with np.load(root/'frozen'/'solid_probe.npz',allow_pickle=False) as probe:
        X=torch.as_tensor(probe['X'],device=device,dtype=torch.float64)
        x=torch.as_tensor(probe['x_preload'],device=device,dtype=torch.float64)
        force=torch.as_tensor(probe['force'],device=device,dtype=torch.float64)
        grad=torch.as_tensor(probe['volume_gradient'],device=device,dtype=torch.float64)
    if (len(x)!=source['reference_nodes'] or X.shape!=x.shape or grad.shape!=x.shape or
        force.shape!=x.shape or not all(torch.isfinite(v).all() for v in (X,x,force,grad))):
        raise ValueError('invalid frozen solid probe arrays')
    folder=Path(output)
    folder.mkdir(parents=True,exist_ok=True)
    report=dict(device=str(device),torch=torch.__version__,
        gpu=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
        source=dict(preload=source,source_sha256=frozen['metadata']['source_sha256'],
                    frozen_levels=frozen['metadata']['levels'],frozen_dt_s=frozen['metadata']['dt_s']),
        levels=list(levels),phases=PHASES,dt_s=dt,case_count_expected=len(levels)*len(PHASES)*4,
        source_production_chorin_vs_schur={str(n):frozen['cases'][f'native_density_{n}']
            ['chorin_vs_schur_solid_velocity_relative_norm'] for n in levels},
        cases={},phase_comparisons={},completed=False,full_cycle_ready=False,
        limitations=['frozen interaction-point shifts are not coupled trajectories',
                     'solid/fluid relative position also changes distance to outer box walls',
                     'density and dual FE loads are distinct discretizations; no production replacement',
                     'algebraic lattice adjointness does not imply weak FE power consistency'])
    _save(folder/'report.json',report)
    arrays={}
    for n in levels:
        mesh=create_box((n,)*3,(12.,)*3,(-6.,-6.,-8.),device=device)
        op=prepare_operators(mesh)
        flow=ChorinSolver(op,dt=dt)
        velocity0=torch.zeros_like(mesh.velocity_coordinates)
        h=mesh.velocity_grid.spacing[0]
        for phase,offset in PHASES.items():
            # Forces and volume gradient stay fixed; only IB interaction points
            # shift within the unchanged Eulerian FE problem.
            xp=x+x.new_tensor(offset)*h
            for mode in ('native','fixed'):
                stencil=scaled_stencil(xp,mesh.velocity_grid,1 if mode=='native' else n//6)
                moment0=(stencil.weights.sum(-1)-1).abs().max().item()
                moment1=(ib.interpolate(mesh.velocity_coordinates,stencil)-xp).abs().max().item()
                if moment0>1e-12 or moment1>1e-11:
                    raise RuntimeError('phase-shifted kernel moment test failed')
                dual=ib.spread_load(force,stencil)
                density=dual/mesh.velocity_grid.cell_volume
                weak_density=op.density_load(density)
                mapping_gap=ratio(weak_density,dual)
                for path in ('density','dual'):
                    name=f'{n}_{phase}_{mode}_{path}'
                    try:
                        flow_result=flow.step(velocity0,**({'density':density} if path=='density'
                                                             else {'nodal_load':dual}))
                        u=flow_result.velocity
                        U=ib.interpolate(u,stencil)
                        arrays[(n,phase,mode,path)]=U.cpu()
                        solid_power=(U*force).sum().item()
                        lattice_power=(u*dual).sum().item()
                        fe_power=(u*(weak_density if path=='density' else dual)).sum().item()
                        case=dict(completed=True,fluid_cells=n,phase=phase,phase_grid_steps=list(offset),
                            kernel=mode,load_path=path,moment_zero_error=moment0,
                            moment_one_error_cm=moment1,
                            fe_density_vs_dual_rhs_relative_norm=mapping_gap,
                            solid_velocity_nodal_norm_cm_per_s=torch.linalg.vector_norm(U).item(),
                            cavity_rate_ml_per_s=(U*grad).sum().item(),
                            solid_power_erg_per_s=solid_power,
                            lattice_power_error_erg_per_s=abs(lattice_power-solid_power),
                            fe_minus_solid_power_erg_per_s=fe_power-solid_power,
                            max_solver_residual_ratio=max(s['residual_norm']/s['tolerance']
                                                          for s in flow_result.diagnostics['solves'].values()))
                        if case['max_solver_residual_ratio']>1:
                            raise RuntimeError('fluid true residual failed')
                        report['cases'][name]=case
                    except (ValueError,RuntimeError) as exc:
                        report['cases'][name]=dict(completed=False,error_type=type(exc).__name__,message=str(exc))
                        _save(folder/'report.json',report)
                        raise
                    _save(folder/'report.json',report)
            print(f'phase {n}^3 {phase}: 4 load/kernel cases',flush=True)
    for n in levels:
        for mode in ('native','fixed'):
            for path in ('density','dual'):
                base=arrays[(n,'base',mode,path)]
                key=f'{n}_{mode}_{path}'
                report['phase_comparisons'][key]=[dict(phase=phase,
                    solid_velocity_relative_to_base=ratio(arrays[(n,phase,mode,path)],base),
                    cavity_rate_relative_to_base=(None if abs(report['cases'][f'{n}_base_{mode}_{path}']
                        ['cavity_rate_ml_per_s'])<=1e-10 else abs(report['cases'][f'{n}_{phase}_{mode}_{path}']
                        ['cavity_rate_ml_per_s']-report['cases'][f'{n}_base_{mode}_{path}']
                        ['cavity_rate_ml_per_s'])/abs(report['cases'][f'{n}_base_{mode}_{path}']
                        ['cavity_rate_ml_per_s']))) for phase in PHASES if phase!='base']
    report['completed']=True
    _save(folder/'report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',default='results/preloaded_ib_study')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',default='results/preloaded_ib_phase')
    args=parser.parse_args()
    result=run(args.study,args.device,args.output)
    print(json.dumps(dict(result,cases={k:{kk:vv for kk,vv in c.items() if kk not in
        ('solid_velocity_nodal_norm_cm_per_s','solid_power_erg_per_s')} for k,c in result['cases'].items()}),indent=2))
