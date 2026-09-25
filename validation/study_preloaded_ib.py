"""Preloaded-LV frozen IB factors followed by short, source-compatible FSI grid study.

Uses an existing converged checkpoint; never resolves the solid equilibrium.
The frozen alternatives remain diagnostic. Coupled trajectories retain the
production Peskin kernel, density load path, Chorin solver and force lag.
"""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from math import isfinite
from pathlib import Path
import numpy as np
import torch
from afsi_torch.fluid import create_box,prepare_operators,ChorinSolver
from afsi_torch.coupling import ExplicitIBStepper
from afsi_torch.preload import load_preload
from afsi_torch.units import CGS_UNITS,MMHG_TO_DYN_PER_CM2
if __package__:
    from .diagnose_ib import run as frozen_run
else:
    from diagnose_ib import run as frozen_run


def relative_change(coarse,fine,floor):
    """Response-relative change; near-zero signals cannot establish convergence."""
    magnitude=abs(fine)
    return None if magnitude<=floor else abs(coarse-fine)/magnitude


def compare(coarse,fine):
    if coarse['time_s']!=fine['time_s']:
        raise ValueError('compare responses at identical physical time')
    if coarse['reference_sha256']!=fine['reference_sha256']:
        raise ValueError('fluid comparison needs identical solid reference and numbering')
    if not coarse['completed'] or not fine['completed']:
        return dict(available=False,reason='incomplete case')
    a,b=coarse['summary'],fine['summary']
    # x_preload is identical across levels; compare the induced response only.
    with np.load(coarse['snapshot'],allow_pickle=False) as ca, np.load(fine['snapshot'],allow_pickle=False) as fb:
        for name in ('X','x_preload','cells'):
            if not np.array_equal(ca[name],fb[name]):
                raise ValueError(f'fluid comparison changed solid {name}')
        ua=ca['x']-ca['x_preload']
        ub=fb['x']-fb['x_preload']
        nodal_diff=float(np.linalg.norm(ua-ub))
        nodal_response=float(np.linalg.norm(ub))
    metrics=dict(delta_cavity_response=relative_change(a['delta_cavity_ml'],b['delta_cavity_ml'],1e-10),
        max_incremental_displacement=relative_change(a['max_incremental_displacement_cm'],
                                                     b['max_incremental_displacement_cm'],1e-12),
        displacement_nodal_l2=None if nodal_response<=1e-12 else nodal_diff/nodal_response)
    return dict(available=True,coarse_cells=coarse['fluid_cells'],fine_cells=fine['fluid_cells'],
        absolute_delta_cavity_difference_ml=abs(a['delta_cavity_ml']-b['delta_cavity_ml']),
        displacement_nodal_l2_difference_cm=nodal_diff,
        fine_displacement_nodal_l2_cm=nodal_response,relative_metrics=metrics,
        screen='inconclusive' if any(v is None for v in metrics.values()) else
               'within_5pct' if max(metrics.values())<=.05 else 'needs_refinement')


def _write(path,report):
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')


@torch.no_grad()
def run(preload='results/lv_equilibrium',device='cpu',output='results/preloaded_ib_study',
        frozen_levels=(6,12,18),fluid_levels=(6,8,10),steps=20,dt=5e-5,
        pressure_increment=.02):
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if (len(fluid_levels)<2 or any(type(n) is not int or n<6 for n in fluid_levels) or
        list(fluid_levels)!=sorted(set(fluid_levels)) or type(steps) is not int or steps<8 or
        not isfinite(dt) or dt<=0 or not isfinite(pressure_increment) or pressure_increment<=0):
        raise ValueError('ordered fluid levels >=6, steps>=8 and positive dt/increment required')
    folder=Path(output)
    folder.mkdir(parents=True,exist_ok=True)
    model,x0,tolerance,provenance=load_preload(preload,device=device)
    initial=model.diagnostics(x0)
    baseline=model.loads
    report=dict(device=str(device),torch=torch.__version__,
        gpu=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
        units=CGS_UNITS,preload=provenance,initial=initial,
        pressure_base_mmhg=baseline.pressure_mmhg,pressure_increment_mmhg=pressure_increment,
        time_step_s=dt,steps=steps,final_time_s=steps*dt,
        frozen_levels=list(frozen_levels),fluid_levels=list(fluid_levels),
        frozen_completed=False,cases={},comparisons=[],completed=False,
        all_refinement_screens_met=False,full_cycle_ready=False,
        scope='short prescribed-traction perturbation; no full-cycle or mesh-convergence claim')
    _write(folder/'report.json',report)
    # Same prestressed shape and incremental pressure force at every frozen level.
    frozen=frozen_run(device=device,output=str(folder/'frozen'),levels=frozen_levels,
                      preload=preload,pressure_increment=pressure_increment)
    report['frozen_completed']=frozen['completed']
    report['frozen_production_chorin_comparisons']={k:v for k,v in frozen['comparisons'].items()
                                                 if k=='native_density_chorin'}
    _write(folder/'report.json',report)
    reference=hashlib.sha256(model.mesh.X.cpu().numpy().tobytes()+
                             model.mesh.cells.cpu().numpy().tobytes()+x0.cpu().numpy().tobytes()).hexdigest()
    duration=steps*dt
    model.loads=replace(baseline,pressure_increment_mmhg=pressure_increment,
                        hold_time=duration/2,ramp_time=duration/4)
    for n in fluid_levels:
        case_folder=folder/f'fluid_{n}'
        case_folder.mkdir(parents=True,exist_ok=True)
        snapshot=case_folder/'last_accepted.npz'
        case=dict(fluid_cells=n,time_s=duration,reference_sha256=reference,
                  snapshot=str(snapshot),completed=False,accepted_steps=0,
                  loads=asdict(model.loads),history=[])
        report['cases'][str(n)]=case
        state=None
        try:
            mesh=create_box((n,)*3,(12.,)*3,(-6.,-6.,-8.),device=device)
            flow=ChorinSolver(prepare_operators(mesh),dt=dt)
            driver=ExplicitIBStepper(flow,model.force,model.validate)
            state=driver.initialize_equilibrium(x0,force_tolerance=tolerance)
            for _ in range(steps):
                result=driver.step(state)
                state=result.state
                diag=model.diagnostics(state.x)
                solves=result.diagnostics['fluid']['solves']
                case['history'].append(dict(step=state.step,time_s=state.time,
                    applied_pressure_mmhg=model.loads.at(result.diagnostics['used_force_time_s'])[0]/MMHG_TO_DYN_PER_CM2,
                    cavity_volume_ml=diag['cavity_volume_ml'],wall_volume_cm3=diag['wall_volume_cm3'],
                    minimum_detF=diag['minimum_detF'],
                    max_incremental_displacement_cm=torch.linalg.vector_norm(state.x-x0,dim=-1).max().item(),
                    max_fluid_speed_cm_per_s=torch.linalg.vector_norm(state.velocity,dim=-1).max().item(),
                    corrected_divergence_l2=result.diagnostics['fluid']['corrected_divergence_l2'],
                    fe_minus_solid_power_erg_per_s=result.diagnostics['fe_minus_solid_power'],
                    solver_residual_ratio=max(s['residual_norm']/s['tolerance'] for s in solves.values())))
            rows=case['history']
            case['summary']=dict(delta_cavity_ml=rows[-1]['cavity_volume_ml']-initial['cavity_volume_ml'],
                max_incremental_displacement_cm=max(r['max_incremental_displacement_cm'] for r in rows),
                max_fluid_speed_cm_per_s=max(r['max_fluid_speed_cm_per_s'] for r in rows),
                minimum_detF=min(r['minimum_detF'] for r in rows),
                max_absolute_wall_volume_change_cm3=max(abs(r['wall_volume_cm3']-initial['wall_volume_cm3']) for r in rows),
                max_corrected_divergence_l2=max(r['corrected_divergence_l2'] for r in rows),
                max_solver_residual_ratio=max(r['solver_residual_ratio'] for r in rows),
                accumulated_fe_minus_solid_work_erg=dt*sum(r['fe_minus_solid_power_erg_per_s'] for r in rows),
                final_applied_pressure_mmhg=rows[-1]['applied_pressure_mmhg'])
            case['completed']=True
        except (ValueError,RuntimeError) as exc:
            case['failure']=dict(type=type(exc).__name__,message=str(exc),attempted_step=0 if state is None else state.step+1)
            raise
        finally:
            case['accepted_steps']=0 if state is None else state.step
            if state is not None:
                np.savez_compressed(snapshot,X=model.mesh.X.cpu().numpy(),x_preload=x0.cpu().numpy(),
                    x=state.x.cpu().numpy(),cells=model.mesh.cells.cpu().numpy(),
                    velocity=state.velocity.cpu().numpy(),pressure=state.pressure.cpu().numpy(),
                    force=state.force.cpu().numpy(),time=state.time,step=state.step)
            _write(folder/'report.json',report)
        print(f'fluid {n}^3: '+json.dumps(case['summary']),flush=True)
    cases=[report['cases'][str(n)] for n in fluid_levels]
    report['comparisons']=[compare(a,b) for a,b in zip(cases,cases[1:])]
    report['all_refinement_screens_met']=all(c['screen']=='within_5pct' for c in report['comparisons'])
    report['completed']=True
    _write(folder/'report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preload',default='results/lv_equilibrium')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',default='results/preloaded_ib_study')
    parser.add_argument('--frozen-levels',type=int,nargs='+',default=[6,12,18])
    parser.add_argument('--fluid-levels',type=int,nargs='+',default=[6,8,10])
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--dt',type=float,default=5e-5)
    parser.add_argument('--pressure-increment-mmhg',type=float,default=.02)
    args=parser.parse_args()
    result=run(args.preload,args.device,args.output,tuple(args.frozen_levels),tuple(args.fluid_levels),
               args.steps,args.dt,args.pressure_increment_mmhg)
    compact=dict(result,cases={k:{kk:vv for kk,vv in c.items() if kk!='history'} for k,c in result['cases'].items()})
    print(json.dumps(compact,indent=2,allow_nan=False))
