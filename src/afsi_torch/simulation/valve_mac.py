"""Generated AFSI demo_340 valve: P2/FRH solid, 2D MAC/MG fluid, quadrature IB."""
import csv
from dataclasses import asdict,replace
from math import isfinite
from pathlib import Path
from time import perf_counter
import torch
from afsi_torch.afsi340 import generate_valve,ValveSolid
from afsi_torch.mac2d import divergence
from afsi_torch.mac2d import checkpoint
from afsi_torch.mac2d.execution import build_driver
from afsi_torch.cycle_checkpoint import atomic_json
from afsi_torch.config import (ValveSimulationConfig, TimeConfig, FluidConfig, OutputConfig,
    ValveExecutionConfig, InletConfig, MGOptions, SolverOptions, mass_options)


@torch.no_grad()
def run(*,device='cuda',output=None,resume=None,end_time=None,dt=None,nx=None,ny=None,
        mesh_size=None,mass_backend=None,fused=None,warm_start=None,execution_backend=None,pressure_backend=None,ib_backend=None,
        log_every=None,checkpoint_every=None,field_every=None,fluid_fields=None,
        case_config=None,fluid_lengths=None,rho=None,mu=None):
    if case_config is not None and not isinstance(case_config,ValveSimulationConfig):
        raise TypeError('case_config must be ValveSimulationConfig')
    if resume and case_config is not None:
        raise ValueError('resume restores physical configuration; omit case_config/--config')
    inherit_field_every=bool(resume) and field_every is None
    inherit_fluid_fields=bool(resume) and fluid_fields is None
    end_time=(case_config.time.end_time if case_config else .005) if end_time is None else end_time
    log_every=(case_config.output.log_every if case_config else 160) if log_every is None else log_every
    checkpoint_every=(case_config.output.checkpoint_every if case_config else 1600) if checkpoint_every is None else checkpoint_every
    if field_every is None:
        field_every=(case_config.output.output_every if case_config.output.write_vtk else 0) if case_config else 160
    fluid_fields=(case_config.output.fluid_fields if case_config else False) if fluid_fields is None else fluid_fields
    if type(fluid_fields) is not bool:
        raise ValueError('fluid_fields must be bool')
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if dt is not None and (not isfinite(dt) or dt<=0):
        raise ValueError('positive finite dt required')
    if any(type(n) is not int or n<1 for n in (log_every,checkpoint_every)) or type(field_every) is not int or field_every<0:
        raise ValueError('positive log/checkpoint intervals and nonnegative field interval required')
    if mass_backend is not None and mass_backend not in ('pcg','graph'):
        raise ValueError('mass backend must be pcg or graph')
    if execution_backend is not None and execution_backend not in ('reference','optimized'):
        raise ValueError('execution backend must be reference or optimized')
    if pressure_backend is not None and pressure_backend not in ('auto','reference','workspace','graph'):
        raise ValueError('invalid pressure backend')
    if any(v is not None and type(v) is not bool for v in (fused,warm_start)):
        raise ValueError('fused/warm_start must be bool or None')
    started=perf_counter()
    folder=Path(output) if output else Path(resume).parent if resume else Path('results/lv_valve_mac')
    if resume:
        if any(v is not None for v in (dt,nx,ny,mesh_size,fluid_lengths,rho,mu)) or folder.resolve()!=Path(resume).resolve().parent:
            raise ValueError('resume restores geometry/grid/dt in its checkpoint directory')
        solid,state,settings,progress=checkpoint.load(resume,device)
        settings.setdefault('ib_backend','reference')
        if inherit_field_every:
            field_every=settings.get('field_every',160)
        if inherit_fluid_fields:
            fluid_fields=settings.get('fluid_fields',False)
        if fused is not None and fused!=settings['fused']:
            solid=ValveSolid(solid.mesh,solid.config,fused=fused)
        for key,value in (('mass_backend',mass_backend),('fused',fused),('warm_start',warm_start),
                          ('execution_backend',execution_backend),('pressure_backend',pressure_backend),('ib_backend',ib_backend)):
            if value is not None:
                settings[key]=value
    else:
        if (folder/'report.json').exists() or (folder/'checkpoint.npz').exists():
            raise ValueError('choose a new output directory or resume')
        config=ValveSimulationConfig() if case_config is None else case_config
        fluid=replace(config.fluid,shape=(config.fluid.shape[0] if nx is None else nx,config.fluid.shape[1] if ny is None else ny),
            lengths=config.fluid.lengths if fluid_lengths is None else tuple(fluid_lengths),
            rho=config.fluid.rho if rho is None else rho,mu=config.fluid.mu if mu is None else mu)
        execution=replace(config.execution,**{k:v for k,v in dict(mass_backend=mass_backend,fused=fused,ib_backend=ib_backend,
            warm_start=warm_start,execution_backend=execution_backend,pressure_backend=pressure_backend).items() if v is not None})
        config=replace(config,time=TimeConfig(config.time.dt if dt is None else dt,end_time),fluid=fluid,execution=execution,
            solid=replace(config.solid,mesh_size=config.solid.mesh_size if mesh_size is None else mesh_size))
        settings=dict(dt=config.time.dt,nx=fluid.shape[0],ny=fluid.shape[1],fluid_lengths=fluid.lengths,
                      rho=fluid.rho,mu=fluid.mu,inlet=asdict(config.inlet),
                      pressure_solver=asdict(config.pressure_solver),mass_solver=asdict(config.mass_solver),**asdict(execution))
        solid=ValveSolid(generate_valve(config.solid,device=device),config.solid,fused=settings['fused'])
        progress=dict(elapsed_seconds=0.,segments=[],frames=[],summary={})
    if not isfinite(end_time) or end_time<=0:
        raise ValueError('positive finite end time required')
    steps=round(end_time/settings['dt'])
    if steps<1 or abs(steps*settings['dt']-end_time)>1e-12:
        raise ValueError('end time must be an integer multiple of dt')
    settings.setdefault('execution_backend','optimized')
    settings.setdefault('pressure_backend','auto')
    settings['output_every']=field_every if field_every else (
        case_config.output.output_every if case_config else settings.get('output_every',160))
    settings['field_every']=field_every
    settings['fluid_fields']=fluid_fields
    driver=build_driver(solid,settings,device)
    flow,transfer=driver.flow,driver.transfer
    grid=flow.grid
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
                                     fused=settings['fused'],execution_backend=settings['execution_backend'],
                                     pressure_backend=flow.pressure_solver.backend,field_every=field_every,fluid_fields=fluid_fields))
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
        configuration=asdict(ValveSimulationConfig(time=TimeConfig(settings['dt'],end_time),
            fluid=FluidConfig(grid.shape,grid.lengths,(0.,0.),settings['rho'],settings['mu']),solid=solid.config,
            inlet=InletConfig(**settings.get('inlet',{})),
            pressure_solver=MGOptions(**settings.get('pressure_solver',{})),
            mass_solver=SolverOptions(**settings['mass_solver']) if 'mass_solver' in settings else mass_options(),
            execution=ValveExecutionConfig(**{k:settings[k] for k in asdict(ValveExecutionConfig())}),
            output=OutputConfig(log_every,checkpoint_every,settings['output_every'],field_every>0,fluid_fields)))
        report=dict(schema=1,demo='afsi340-generated-valve-mac',status=status,configuration=configuration,completed=state.step==steps,
            accepted_steps=state.step,reached_time_s=state.time,requested_end_time_s=end_time,
            device=str(device),torch=torch.__version__,gpu=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
            units='cm-g-s-per-unit-out-of-plane-thickness',settings=settings,config=asdict(solid.config),
            solid_nodes=len(state.x),solid_cells=len(solid.mesh.cells),interaction_points=solid.geometry.weights.numel(),
            fluid_cells=grid.shape,fluid_spacing_cm=grid.spacing,mg_levels=flow.pressure_solver.shapes,
            pressure_backend=flow.pressure_solver.backend,mass_cuda_graphs=len(getattr(transfer.solver,'graphs',{})),
            pressure_cuda_graphs=len(getattr(flow.pressure_solver.workspace,'graphs',{})),
            pressure_workspace_bytes=getattr(flow.pressure_solver.workspace,'allocated_bytes',0),
            ib_stencil_builds=driver.stencil_builds,
            quadrature='positive six-point triangle degree 4; three-point root Gauss',
            coupling='quadrature IB with consistent CSR P2 mass; adjoint odd-wall extension',
            inlet=dict(formula='amplitude*(sin(2*pi*t/period)+offset)*y*(height-y), u_y=0',
                       **asdict(InletConfig(**settings.get('inlet',{})))),outlet='p=0; tentative velocity zero normal derivative',
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
        atomic_json(folder/'configuration.json',configuration)
        atomic_json(folder/'report.json',report)
        return report

    try:
        if field_every and not progress['frames']:
            write_fields()
        save('running')
        print(f'Valve MAC: {device}, grid={grid.shape}, solid nodes={len(state.x)}, cells={len(solid.mesh.cells)}, '
              f'dt={flow.dt:g}; steps {state.step}->{steps}; mass={settings["mass_backend"]}; '
              f'execution={settings["execution_backend"]}, pressure={flow.pressure_solver.backend}; IB={settings["ib_backend"]}',flush=True)
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

