"""Reuse an existing solid preload for hold and pressure-increment IB tests.

Prescribed cavity traction is already inside the solid force. The background
pressure starts at zero and is NOT a reconstructed diastolic cavity pressure.
"""
import argparse
from dataclasses import asdict,replace
import json
from math import isfinite
from pathlib import Path
import numpy as np
import torch
from afsi_torch.preload import load_preload
from afsi_torch.lv_model import RampLoads
from afsi_torch.fluid import create_box,prepare_operators,ChorinSolver
from afsi_torch.coupling import ExplicitIBStepper
from afsi_torch.coupling_output import CoupledWriter
from afsi_torch import ib
from afsi_torch.units import CGS_UNITS,MMHG_TO_DYN_PER_CM2


def run(preload='results/lv_equilibrium',device='cpu',output='results/preloaded_lv',
        steps=20,dt=5e-5,fluid_cells=6,pressure_increment=.02,write_output=True):
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if (not isinstance(steps,int) or isinstance(steps,bool) or steps<4 or
        not isinstance(fluid_cells,int) or isinstance(fluid_cells,bool) or fluid_cells<2 or
        not isfinite(dt) or dt<=0 or not isfinite(pressure_increment) or pressure_increment<=0):
        raise ValueError('steps>=4, positive dt/increment and integer fluid_cells>=2 required')
    folder=Path(output)
    folder.mkdir(parents=True,exist_ok=True)
    model,x0,tolerance,provenance=load_preload(preload,device)
    initial=model.diagnostics(x0)
    fluid_mesh=create_box((fluid_cells,)*3,(12.,)*3,(-6.,-6.,-8.),device=device)
    flow=ChorinSolver(prepare_operators(fluid_mesh),dt=dt)
    base_loads=model.loads
    report=dict(device=str(device),torch=torch.__version__,
        gpu=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
        units=CGS_UNITS,provenance=provenance,initial=initial,dt_s=dt,steps=steps,fluid_cells=fluid_cells,
        source_preload_pressure_mmhg=base_loads.pressure_mmhg,
        source_preload_tension_dyn_per_cm2=base_loads.tension,
        initial_pressure_field='zero background Chorin pressure; prescribed cavity traction remains in solid force',
        fluid_pressure_represents_prescribed_lv_pressure=False,
        initialization='actual balanced nodal force at physical time zero',
        reference_geometry='original unloaded X retained',
        hold_screen=dict(max_displacement_cm=1e-8,max_cavity_change_ml=1e-7,max_fluid_speed_cm_per_s=1e-6),
        cases={},completed=False,hold_screen_passed=False,perturbation_detected=False,full_cycle_ready=False)
    def save():
        (folder/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    save()
    # Negative control: reverting to the old zero-based load ramp should fail
    # the same force-balance gate on this preloaded geometry.
    model.loads=RampLoads(pressure_mmhg=base_loads.pressure_mmhg,tension=base_loads.tension)
    bad_force=model.force(x0,0.)
    report['zero_ramp_control']=dict(force_norm_dyn=torch.linalg.vector_norm(bad_force).item(),
        rejected_by_balance_gate=False)
    try:
        ExplicitIBStepper(flow,model.force,model.validate).initialize_equilibrium(x0,force_tolerance=tolerance)
    except ValueError as exc:
        if 'preload is not balanced' not in str(exc):
            raise
        report['zero_ramp_control']['rejected_by_balance_gate']=True
    model.loads=base_loads
    for name,increment in [('hold',0.),('pressure_increment',pressure_increment)]:
        duration=steps*dt
        model.loads=replace(base_loads,pressure_increment_mmhg=increment,hold_time=duration/2,ramp_time=duration/4)
        driver=ExplicitIBStepper(flow,model.force,model.validate)
        state=driver.initialize_equilibrium(x0,force_tolerance=tolerance)
        case_folder=folder/name
        case_folder.mkdir(parents=True,exist_ok=True)
        writer=CoupledWriter(case_folder,model.mesh,fluid_mesh) if write_output else None
        stencil=ib.prepare_stencil(x0,driver.grid)
        weak_load=flow.op.density_load(ib.spread_density(state.force,stencil))
        data=dict(loads=asdict(model.loads),completed=False,history=[],
            initial_force_norm_dyn=torch.linalg.vector_norm(state.force).item(),
            initial_fluid_free_rhs_norm_dyn=torch.linalg.vector_norm(weak_load.masked_fill(flow.velocity_fixed,0)).item(),
            initial_solid_force_time_s=state.force_time)
        report['cases'][name]=data
        if writer:
            writer.write(state)
        try:
            for k in range(steps):
                result=driver.step(state)
                state=result.state
                row=dict(step=state.step,**result.diagnostics,**model.diagnostics(state.x),
                    applied_prescribed_pressure_mmhg=model.loads.at(result.diagnostics['used_force_time_s'])[0]/MMHG_TO_DYN_PER_CM2,
                    next_prescribed_pressure_mmhg=model.loads.at(state.force_time)[0]/MMHG_TO_DYN_PER_CM2,
                    max_incremental_displacement_cm=torch.linalg.vector_norm(state.x-x0,dim=-1).max().item(),
                    max_fluid_speed_cm_per_s=torch.linalg.vector_norm(state.velocity,dim=-1).max().item())
                data['history'].append(row)
                if writer and (state.step%5==0 or state.step==steps):
                    writer.write(state)
            data['completed']=True
        except (ValueError,RuntimeError) as exc:
            data['failure']=dict(type=type(exc).__name__,message=str(exc),attempted_step=state.step+1)
            raise
        finally:
            data['accepted_steps']=state.step
            data['reached_time_s']=state.time
            np.savez_compressed(case_folder/'last_accepted.npz',X=model.mesh.X.cpu().numpy(),
                x_preload=x0.cpu().numpy(),x=state.x.cpu().numpy(),cells=model.mesh.cells.cpu().numpy(),
                velocity=state.velocity.cpu().numpy(),pressure=state.pressure.cpu().numpy(),
                force=state.force.cpu().numpy(),time=state.time,step=state.step)
            save()
        rows=data['history']
        data['summary']=dict(max_incremental_displacement_cm=max(r['max_incremental_displacement_cm'] for r in rows),
            max_abs_cavity_change_ml=max(abs(r['cavity_volume_ml']-initial['cavity_volume_ml']) for r in rows),
            final_cavity_change_ml=rows[-1]['cavity_volume_ml']-initial['cavity_volume_ml'],
            max_fluid_speed_cm_per_s=max(r['max_fluid_speed_cm_per_s'] for r in rows),
            min_detF=min(r['minimum_detF'] for r in rows),
            max_solver_residual_ratio=max(s['residual_norm']/s['tolerance'] for r in rows for s in r['fluid']['solves'].values()))
        print(name,json.dumps(data['summary']),flush=True)
        save()
    hold=report['cases']['hold']['summary']
    perturb=report['cases']['pressure_increment']['summary']
    screen=report['hold_screen']
    report['hold_screen_passed']=(hold['max_incremental_displacement_cm']<=screen['max_displacement_cm'] and
        hold['max_abs_cavity_change_ml']<=screen['max_cavity_change_ml'] and
        hold['max_fluid_speed_cm_per_s']<=screen['max_fluid_speed_cm_per_s'])
    report['perturbation_detected']=perturb['max_incremental_displacement_cm']>max(1e-10,100*hold['max_incremental_displacement_cm'])
    report['completed']=True
    save()
    if not report['hold_screen_passed'] or not report['perturbation_detected']:
        raise RuntimeError('startup diagnostic failed; inspect saved report, no tolerances changed')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preload',default='results/lv_equilibrium')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',default='results/preloaded_lv')
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--dt',type=float,default=5e-5)
    parser.add_argument('--fluid-cells',type=int,default=6)
    parser.add_argument('--pressure-increment-mmhg',type=float,default=.02)
    parser.add_argument('--no-vtk',action='store_true')
    args=parser.parse_args()
    r=run(args.preload,args.device,args.output,args.steps,args.dt,args.fluid_cells,args.pressure_increment_mmhg,not args.no_vtk)
    compact=dict(r,cases={name:{k:v for k,v in c.items() if k!='history'} for name,c in r['cases'].items()})
    print(json.dumps(compact,indent=2))
