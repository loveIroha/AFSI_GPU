"""Experimental generated LV: PyTorch MAC + geometric MG + quadrature FE/IB.

Default is a 0.005 s validation segment, not a claim of a validated full run.
Uses AFSI337 geometry/material/loads in cm-g-s. No Taichi dependency.
"""
import argparse
import csv
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from time import perf_counter
import torch
from afsi_torch.afsi337 import generated_model
from afsi_torch.cycle_checkpoint import atomic_json
from afsi_torch.mac import MACGrid,divergence
from afsi_torch.mac.checkpoint import save_mac,load_mac


@torch.no_grad()
def run(*,device='cuda',output=None,end_time=.005,dt=None,fluid_cells=None,mesh_size=None,
        interaction_degree=None,log_every=20,checkpoint_every=200,resume=None,warm_start=None,
        pressure_backend=None,execution_backend=None,solid_backend=None,mass_backend=None,coupling_backend=None,
        write_vtk=None,output_every=None):
    if execution_backend is not None and execution_backend not in ('torch','fused'):
        raise ValueError('execution_backend must be torch or fused')
    if solid_backend is not None and solid_backend not in ('reference','pointwise'):
        raise ValueError('solid_backend must be reference or pointwise')
    if mass_backend is not None and mass_backend not in ('pcg','graph'):
        raise ValueError('mass_backend must be pcg or graph')
    if coupling_backend is not None and coupling_backend not in ('reference','optimized'):
        raise ValueError('coupling_backend must be reference or optimized')
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if any(type(n) is not int or n<1 for n in (log_every,checkpoint_every)):
        raise ValueError('positive integer log/checkpoint intervals required')
    if warm_start is not None and type(warm_start) is not bool:
        raise ValueError('warm_start must be a bool or None')
    if write_vtk is not None and type(write_vtk) is not bool:
        raise ValueError('write_vtk must be a bool or None')
    if output_every is not None and (type(output_every) is not int or output_every<1):
        raise ValueError('positive integer output_every required')
    if pressure_backend is not None and pressure_backend not in ('torch','fused','workspace','graph'):
        raise ValueError('pressure_backend must be torch, fused, workspace or graph')
    started=perf_counter()
    folder=Path(output) if output else Path(resume).parent if resume else Path('results/lv_mac')
    if resume:
        if any(v is not None for v in (dt,fluid_cells,mesh_size,interaction_degree)):
            raise ValueError('resume restores mesh, dt and interaction quadrature')
        if folder.resolve()!=Path(resume).resolve().parent:
            raise ValueError('resume in the checkpoint directory')
        model,state,settings,progress=load_mac(resume,device)
        settings['warm_start']=(settings.get('warm_start',False) if warm_start is None else warm_start)
        settings['pressure_backend']=(settings.get('pressure_backend','torch')
                                      if pressure_backend is None else pressure_backend)
        settings['execution_backend']=(settings.get('execution_backend','torch')
                                       if execution_backend is None else execution_backend)
        settings['solid_backend']=(settings.get('solid_backend','reference')
                                   if solid_backend is None else solid_backend)
        settings['mass_backend']=(settings.get('mass_backend','pcg')
                                  if mass_backend is None else mass_backend)
        settings['coupling_backend']=(settings.get('coupling_backend','reference')
                                      if coupling_backend is None else coupling_backend)
    else:
        if (folder/'report.json').exists() or (folder/'checkpoint.npz').exists():
            raise ValueError('output contains a run; choose a new folder or resume')
        dt=5e-5 if dt is None else dt
        fluid_cells=64 if fluid_cells is None else fluid_cells
        mesh_size=.1 if mesh_size is None else mesh_size
        model=generated_model(mesh_size=mesh_size,device=device)
        settings=dict(dt=dt,fluid_cells=fluid_cells,box_length=5.,rho=1.,mu=1.,
                      interaction_degree=interaction_degree,
                      warm_start=False if warm_start is None else warm_start,
                      pressure_backend='torch' if pressure_backend is None else pressure_backend,
                      execution_backend='torch' if execution_backend is None else execution_backend,
                      solid_backend='reference' if solid_backend is None else solid_backend,
                      mass_backend='pcg' if mass_backend is None else mass_backend,
                      coupling_backend='reference' if coupling_backend is None else coupling_backend)
        progress=dict(elapsed_seconds=0.,segments=[],summary={})
    settings['write_vtk']=settings.get('write_vtk',False) if write_vtk is None else write_vtk
    settings['output_every']=settings.get('output_every',400) if output_every is None else output_every
    if type(settings['output_every']) is not int or settings['output_every']<1:
        raise ValueError('positive integer output_every required')
    dt=settings['dt']
    if not isfinite(end_time) or end_time<=0 or not isfinite(dt) or dt<=0:
        raise ValueError('positive finite dt/end time required')
    steps=round(end_time/dt)
    if steps<1 or abs(steps*dt-end_time)>1e-12:
        raise ValueError('end time must be an integer multiple of dt')
    grid=MACGrid((settings['fluid_cells'],)*3,(settings['box_length'],)*3)
    degree=settings['interaction_degree']
    if degree is not None and (type(degree) is not int or degree<4):
        raise ValueError('interaction degree must be >=4 for the P2 consistent mass')
    from afsi_torch.mac.execution import build_driver
    driver=build_driver(model,settings,device)
    flow,transfer=driver.flow,driver.transfer
    geometry=transfer.geometry
    if resume:
        if state.step>steps:
            raise ValueError('end time precedes checkpoint')
        grid.check_velocity(state.velocity)
        if state.pressure.shape!=grid.shape:
            raise ValueError('checkpoint pressure shape mismatch')
        transfer.check_support(transfer.interaction_points(state.x))
    else:
        state=driver.initialize(model.mesh.X)
    folder.mkdir(parents=True,exist_ok=True)
    writer=None
    if settings['write_vtk']:
        from afsi_torch.mac.output import MACWriter
        writer=MACWriter(folder/'vtk',model,grid,resume_time=state.time if resume else None)
    history=[]
    if (folder/'history.csv').exists():
        with (folder/'history.csv').open(newline='',encoding='utf-8') as stream:
            history=[r for r in csv.DictReader(stream) if int(r['step'])<=state.step]
    previous_elapsed=progress['elapsed_seconds']
    progress.pop('failure',None)
    progress['segments'].append(dict(start_step=state.step,device=str(device),log_every=log_every,
                                      warm_start=settings['warm_start'],
                                      execution_backend=settings['execution_backend'],
                                      solid_backend=settings['solid_backend'],mass_backend=settings['mass_backend'],
                                      coupling_backend=settings['coupling_backend'],
                                      pressure_backend=flow.pressure_solver.backend,
                                      write_vtk=settings['write_vtk'],output_every=settings['output_every']))
    info={}

    def row():
        return dict(step=state.step,time_s=state.time,**model.diagnostics(state.x),
            max_fluid_component_cm_per_s=max(u.abs().max().item() for u in state.velocity),
            divergence_l2=(grid.volume*divergence(state.velocity,grid.spacing).square().sum()).sqrt().item(),
            pressure_cycles=info.get('flow',{}).get('pressure',{}).get('cycles',0),
            power_error=info.get('power_error',0.))

    if not history or int(history[-1]['step'])!=state.step:
        history.append(row())

    def save(status):
        if int(history[-1]['step'])!=state.step:
            history.append(row())
        with (folder/'history.csv').open('w',newline='',encoding='utf-8') as stream:
            csv_writer=csv.DictWriter(stream,fieldnames=history[0].keys())
            csv_writer.writeheader()
            csv_writer.writerows(history)
        progress['elapsed_seconds']=previous_elapsed+perf_counter()-started
        report=dict(schema=1,demo='experimental-afsi337-mac',status=status,
            completed=state.step==steps,accepted_steps=state.step,reached_time_s=state.time,
            requested_end_time_s=end_time,device=str(device),torch=torch.__version__,
            gpu=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
            units='cm-g-s',settings=settings,solid_nodes=len(state.x),solid_cells=len(model.mesh.cells),
            fluid_velocity_dofs=sum(u.numel() for u in state.velocity),fluid_pressure_cells=state.pressure.numel(),
            fluid_spacing_cm=grid.spacing,mg_levels=flow.pressure_solver.shapes,
            pressure_backend=flow.pressure_solver.backend,
            pressure_cuda_graphs=len(getattr(flow.pressure_solver.workspace,'graphs',{})),
            pressure_workspace_bytes=getattr(flow.pressure_solver.workspace,'allocated_bytes',0),
            coupling_backend=settings['coupling_backend'],ib_stencil_builds=driver.stencil_builds,
            ib_point_evaluations=driver.point_evaluations,
            execution_backend=settings['execution_backend'],
            solid_backend=settings['solid_backend'],mass_backend=settings['mass_backend'],
            mass_cuda_graphs=len(getattr(getattr(transfer,'mass_solver',None),'graphs',{})),
            interaction_points=geometry.weights.numel(),interaction_rule='fixed; increase --interaction-degree for refinement',
            coupling='Griffith-Luo unified quadrature transfer with consistent FE mass solves',
            pressure_gauge='zero mean, homogeneous Neumann, A=-D G',
            time_scheme='first-order, centered conservative convection, explicit viscosity; AFSI lagged force order',
            loads=asdict(model.loads),material=asdict(model.parameters),last=history[-1],last_solver_info=info,
            visualization=writer.summary() if writer else dict(enabled=False),
            full_horizon_validated=False,mesh_convergence_established=False,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None,
            **progress)
        save_mac(folder/'checkpoint.npz',model,state,settings,progress)
        atomic_json(folder/'report.json',report)
        return report

    try:
        if writer:
            writer.write(state)
        save('running')
        print(f'MAC LV: {device}, grid={grid.shape}, solid nodes={len(state.x)}, '
              f'interaction points={geometry.weights.numel()}, dt={dt:g}; steps {state.step}->{steps}',flush=True)
        print(f'Execution: {settings["execution_backend"]}; solid={settings["solid_backend"]}, '
              f'mass={settings["mass_backend"]}, pressure={flow.pressure_solver.backend}, '
              f'coupling={settings["coupling_backend"]}',flush=True)
        for _ in range(state.step,steps):
            sample=(state.step+1)%log_every==0 or state.step+1==steps
            state,info=driver.step(state,diagnostics=sample)
            if writer and (state.step%settings['output_every']==0 or state.step==steps):
                writer.write(state)
            for key,value in (('max_pressure_cycles',info['flow']['pressure']['cycles']),
                              ('max_grid_displacement',info['max_grid_displacement'])):
                progress['summary'][key]=max(progress['summary'].get(key,0),value)
            if sample:
                history.append(row())
                progress['summary']['max_sampled_power_error']=max(progress['summary'].get('max_sampled_power_error',0),info['power_error'])
                last=history[-1]
                print(f'step {state.step}/{steps}, t={state.time:.6f}, V={last["cavity_volume_ml"]:.8g} mL, '
                      f'minJ={last["minimum_detF"]:.6g}, div={last["divergence_l2"]:.3g}, '
                      f'MG cycles={last["pressure_cycles"]}, power error={last["power_error"]:.3g}',flush=True)
            if state.step%checkpoint_every==0:
                save('running')
        return save('completed')
    except (Exception,KeyboardInterrupt) as exc:
        progress['failure']=dict(type=type(exc).__name__,message=str(exc),last_accepted_step=state.step)
        save('failed')
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--output')
    parser.add_argument('--end-time',type=float,default=.005)
    parser.add_argument('--dt',type=float)
    parser.add_argument('--fluid-cells',type=int)
    parser.add_argument('--mesh-size',type=float)
    parser.add_argument('--interaction-degree',type=int)
    parser.add_argument('--log-every',type=int,default=20)
    parser.add_argument('--checkpoint-every',type=int,default=200)
    parser.add_argument('--resume')
    parser.add_argument('--vtk',dest='write_vtk',action=argparse.BooleanOptionalAction,default=None,
                        help='write ParaView solid/fluid time series at output steps')
    parser.add_argument('--output-every',type=int,default=None,
                        help='VTK frame interval in steps; default: 400')
    parser.add_argument('--warm-start',action=argparse.BooleanOptionalAction,default=None,
                        help='reuse previous IB mass-solve coefficients as PCG initial guesses')
    parser.add_argument('--pressure-backend',choices=('torch','fused','workspace','graph'),default=None,
                        help='graph: fused workspace and CUDA Graph V-cycle blocks')
    parser.add_argument('--coupling-backend',choices=('reference','optimized'),default=None,
                        help='optimized: cached accepted IB stencils and combined checks; requires fused execution')
    parser.add_argument('--execution-backend',choices=('torch','fused'),default=None,
                        help='fused: compact IB, buffered CSR PCG and compiled fluid/solid kernels')
    parser.add_argument('--solid-backend',choices=('reference','pointwise'),default=None,
                        help='pointwise: fused scalar 3x3 Guccione algebra; requires fused execution')
    parser.add_argument('--mass-backend',choices=('pcg','graph'),default=None,
                        help='graph: CUDA Graph PCG blocks; requires fused execution')
    args=parser.parse_args()
    report=run(**vars(args))
    print(f'{report["status"]}: elapsed_seconds={report["elapsed_seconds"]:.3f}',flush=True)


if __name__=='__main__':
    main()
