"""Generated AFSI demo_340 valve: P2/FRH solid, 2D MAC/MG fluid, quadrature IB."""
import argparse
import csv
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from time import perf_counter
import torch
from afsi_torch.afsi340 import ValveConfig,generate_valve,ValveSolid
from afsi_torch.mac2d import ChannelGrid,ChannelFlow,divergence
from afsi_torch.mac2d.transfer import TriangleTransfer
from afsi_torch.mac2d.coupling import ValveStepper
from afsi_torch.mac2d import checkpoint
from afsi_torch.cycle_checkpoint import atomic_json


@torch.no_grad()
def run(*,device='cuda',output=None,resume=None,end_time=.005,dt=None,nx=None,ny=None,
        mesh_size=None,mass_backend=None,fused=None,warm_start=None,
        log_every=160,checkpoint_every=1600,field_every=160,fluid_fields=False):
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if dt is not None and (not isfinite(dt) or dt<=0):
        raise ValueError('positive finite dt required')
    if any(type(n) is not int or n<1 for n in (log_every,checkpoint_every)) or type(field_every) is not int or field_every<0:
        raise ValueError('positive log/checkpoint intervals and nonnegative field interval required')
    if mass_backend is not None and mass_backend not in ('pcg','graph'):
        raise ValueError('mass backend must be pcg or graph')
    if any(v is not None and type(v) is not bool for v in (fused,warm_start)):
        raise ValueError('fused/warm_start must be bool or None')
    started=perf_counter()
    folder=Path(output) if output else Path(resume).parent if resume else Path('results/lv_valve_mac')
    if resume:
        if any(v is not None for v in (dt,nx,ny,mesh_size)) or folder.resolve()!=Path(resume).resolve().parent:
            raise ValueError('resume restores geometry/grid/dt in its checkpoint directory')
        solid,state,settings,progress=checkpoint.load(resume,device)
        if fused is not None and fused!=settings['fused']:
            solid=ValveSolid(solid.mesh,solid.config,fused=fused)
        for key,value in (('mass_backend',mass_backend),('fused',fused),('warm_start',warm_start)):
            if value is not None:
                settings[key]=value
    else:
        if (folder/'report.json').exists() or (folder/'checkpoint.npz').exists():
            raise ValueError('choose a new output directory or resume')
        config=ValveConfig(mesh_size=.01 if mesh_size is None else mesh_size)
        settings=dict(dt=1/16000 if dt is None else dt,nx=256 if nx is None else nx,ny=64 if ny is None else ny,
                      rho=1.,mu=.1,mass_backend='graph' if mass_backend is None else mass_backend,
                      fused=True if fused is None else fused,warm_start=True if warm_start is None else warm_start)
        solid=ValveSolid(generate_valve(config,device=device),config,fused=settings['fused'])
        progress=dict(elapsed_seconds=0.,segments=[],frames=[],summary={})
    if not isfinite(end_time) or end_time<=0:
        raise ValueError('positive finite end time required')
    steps=round(end_time/settings['dt'])
    if steps<1 or abs(steps*settings['dt']-end_time)>1e-12:
        raise ValueError('end time must be an integer multiple of dt')
    grid=ChannelGrid((settings['nx'],settings['ny']),(8.,solid.config.height))
    flow=ChannelFlow(grid,dt=settings['dt'],rho=settings['rho'],mu=settings['mu'],device=device,fused=settings['fused'])
    transfer=TriangleTransfer(grid,solid.geometry,mass_backend=settings['mass_backend'],
                              warm_start=settings['warm_start'],fused=settings['fused'])
    driver=ValveStepper(flow,transfer,solid)
    if not resume:
        state=driver.initialize()
    if state.step>steps:
        raise ValueError('end time precedes checkpoint')
    grid.check_velocity(state.velocity)
    if state.pressure.shape!=grid.shape:
        raise ValueError('checkpoint pressure shape mismatch')
    folder.mkdir(parents=True,exist_ok=True)
    previous_elapsed=progress['elapsed_seconds']
    progress.pop('failure',None)
    progress['segments'].append(dict(start_step=state.step,device=str(device),mass_backend=settings['mass_backend'],
                                     fused=settings['fused'],field_every=field_every,fluid_fields=fluid_fields))
    history=[]
    if (folder/'history.csv').exists():
        with (folder/'history.csv').open(newline='',encoding='utf-8') as stream:
            history=[r for r in csv.DictReader(stream) if int(r['step'])<=state.step]
    info={}

    def row():
        u,v=state.velocity
        qin=(grid.spacing[1]*u[0].sum()).item()
        qout=(grid.spacing[1]*u[-1].sum()).item()
        return dict(step=state.step,time_s=state.time,**solid.diagnostics(state.x),
            inlet_flow_cm2_per_s=qin,outlet_flow_cm2_per_s=qout,flux_error_cm2_per_s=abs(qout-qin),
            max_velocity_component_cm_per_s=max(a.abs().max().item() for a in (u,v)),
            divergence_l2=(grid.volume*divergence(state.velocity,grid.spacing).square().sum()).sqrt().item(),
            pressure_cycles=info.get('flow',{}).get('pressure',{}).get('cycles',0),
            power_relative_error=info.get('power_relative_error',0.))

    if not history or int(history[-1]['step'])!=state.step:
        history.append(row())

    def write_fields():
        from afsi_torch.mac2d.output import fields,collection
        frame=fields(folder/'fields',solid,state,grid,fluid=fluid_fields)
        progress['frames']=[r for r in progress['frames'] if r['step']<state.step]+[frame]
        collection(folder/'fields',progress['frames'])

    def save(status):
        if int(history[-1]['step'])!=state.step:
            history.append(row())
        with (folder/'history.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=history[0].keys())
            writer.writeheader(); writer.writerows(history)
        progress['elapsed_seconds']=previous_elapsed+perf_counter()-started
        report=dict(schema=1,demo='afsi340-generated-valve-mac',status=status,completed=state.step==steps,
            accepted_steps=state.step,reached_time_s=state.time,requested_end_time_s=end_time,
            device=str(device),torch=torch.__version__,gpu=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
            units='cm-g-s-per-unit-out-of-plane-thickness',settings=settings,config=asdict(solid.config),
            solid_nodes=len(state.x),solid_cells=len(solid.mesh.cells),interaction_points=solid.geometry.weights.numel(),
            fluid_cells=grid.shape,fluid_spacing_cm=grid.spacing,mg_levels=flow.pressure_solver.shapes,
            pressure_backend=flow.pressure_solver.backend,mass_cuda_graphs=len(getattr(transfer.solver,'graphs',{})),
            quadrature='positive six-point triangle degree 4; three-point root Gauss',
            coupling='quadrature IB with consistent CSR P2 mass; adjoint odd-wall extension',
            inlet='u_x=5*(sin(2*pi*t)+1.1)*y*(1.61-y), u_y=0',outlet='p=0; tentative velocity zero normal derivative',
            source='https://github.com/loveIroha/afsi/tree/main/afsic/demo/demo_340',
            differences=['MAC finite differences replace source Q2/Q1 FEM fluid.',
                         'MAC viscosity is explicit; source Chorin tentative velocity uses implicit viscosity.',
                         'Centered conservative MAC transport replaces source explicit advective FEM transport.',
                         'Quadrature IB replaces source nodal IB implementation.',
                         'Odd wall ghost extension replaces source out-of-grid IB stencil omission.',
                         'GPU clock labels accepted state at (step+1)*dt; inlet is sampled at step*dt.',
                         'Energy output subtracts the constant undeformed FRH density.',
                         'No contact model; source demo has no explicit contact model.'],
            last=history[-1],last_solver_info=info,cpu_afsi_agreement_established=False,mesh_convergence_established=False,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None,**progress)
        checkpoint.save(folder/'checkpoint.npz',solid,state,settings,progress)
        atomic_json(folder/'report.json',report)
        return report

    try:
        if field_every and not progress['frames']:
            write_fields()
        save('running')
        print(f'Valve MAC: {device}, grid={grid.shape}, solid nodes={len(state.x)}, cells={len(solid.mesh.cells)}, '
              f'dt={flow.dt:g}; steps {state.step}->{steps}; mass={settings["mass_backend"]}',flush=True)
        for _ in range(state.step,steps):
            sample=(state.step+1)%log_every==0 or state.step+1==steps
            state,info=driver.step(state,diagnostics=sample)
            progress['summary']['max_pressure_cycles']=max(progress['summary'].get('max_pressure_cycles',0),info['flow']['pressure']['cycles'])
            if sample:
                history.append(row())
                last=history[-1]
                print(f'step {state.step}/{steps}, t={state.time:.6f}, upper tip=({last["upper_tip_dx_cm"]:.5g}, '
                      f'{last["upper_tip_dy_cm"]:.5g}) cm, gap={last["probe_gap_cm"]:.5g}, '
                      f'minJ={last["minimum_detF"]:.6g}, div={last["divergence_l2"]:.3g}',flush=True)
            if field_every and (state.step%field_every==0 or state.step==steps):
                write_fields()
            if state.step%checkpoint_every==0:
                save('running')
        return save('completed')
    except (Exception,KeyboardInterrupt) as exc:
        progress['failure']=dict(type=type(exc).__name__,message=str(exc),last_accepted_step=state.step)
        save('failed')
        raise


def main(argv=None,*,default_end_time=.005,default_output='results/valve_mac'):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda')
    p.add_argument('--output')
    p.add_argument('--resume')
    p.add_argument('--end-time',type=float,default=default_end_time)
    p.add_argument('--dt',type=float)
    p.add_argument('--nx',type=int)
    p.add_argument('--ny',type=int)
    p.add_argument('--mesh-size',type=float)
    p.add_argument('--mass-backend',choices=('pcg','graph'))
    p.add_argument('--fused',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--warm-start',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--log-every',type=int,default=160)
    p.add_argument('--checkpoint-every',type=int,default=1600)
    p.add_argument('--field-every',type=int,default=160,help='0 disables VTU/PVD output')
    p.add_argument('--fluid-fields',action='store_true',help='also export cell pressure/velocity')
    options=vars(p.parse_args(argv))
    if not options['output'] and not options['resume']:
        options['output']=default_output
    report=run(**options)
    print(f'{report["status"]}: step={report["accepted_steps"]}, t={report["reached_time_s"]:.6f} s, '
          f'elapsed_seconds={report["elapsed_seconds"]:.3f}',flush=True)
    return report


if __name__=='__main__':
    main()
