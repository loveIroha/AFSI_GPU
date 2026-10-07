"""Generated ideal-LV prescribed-load cycle with PyTorch IB/FEM time stepping.

Default: 0.8 s, dt=5e-5 s, fresh generated geometry, no input mesh/fiber files.
The first cycle includes filling from zero pressure. No valve/circulation
model or periodic steady-state solution is implied by completing one cycle.
"""
import argparse
import csv
from dataclasses import asdict,replace
import json
from math import isfinite
from pathlib import Path
from time import perf_counter

import torch

from afsi_torch import ib
from afsi_torch.coupling import ExplicitIBStepper
from afsi_torch.coupling_output import CoupledWriter
from afsi_torch.cycle_checkpoint import atomic_json, load_cycle, save_cycle
from afsi_torch.cycle_loads import AFSICycleLoads
from afsi_torch.fluid import ChorinSolver, create_box, prepare_operators, CSRFluidOperators
from afsi_torch.fluid.solvers import SolverOptions
from afsi_torch.geometry import LVConfig, generate_lv
from afsi_torch.geometry.output import write_lv
from afsi_torch.lv_model import LVSolid
from afsi_torch.preload import load_preload
from afsi_torch.step_timing import StepTimingRecorder
from afsi_torch.units import CGS_UNITS, MMHG_TO_DYN_PER_CM2
from afsi_torch.config import LVFEMSimulationConfig,TimeConfig,FEMFluidConfig,OutputConfig


COLUMNS = ('step', 'time_s', 'prescribed_pressure_mmhg', 'prescribed_tension_dyn_per_cm2',
    'applied_force_time_s', 'applied_pressure_mmhg', 'next_force_time_s',
    'cavity_volume_ml', 'wall_volume_cm3', 'minimum_detF', 'maximum_detF',
    'max_incremental_displacement_cm', 'max_fluid_speed_cm_per_s', 'fluid_courant',
    'max_grid_displacement', 'force_norm_dyn', 'kinetic_energy_erg',
    'corrected_divergence_l2', 'max_solver_residual_ratio', 'max_solver_iterations')


def _history(path, resume_step):
    """Discard only journal rows newer than the accepted checkpoint."""
    rows = []
    if path.exists():
        with path.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != COLUMNS:
                raise ValueError('existing history.csv has an incompatible schema')
            rows = [row for row in reader if int(row['step']) <= resume_step]
        if any(int(a['step']) >= int(b['step']) for a, b in zip(rows, rows[1:])):
            raise ValueError('history steps must increase strictly')
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return None if not rows else int(rows[-1]['step'])


def _row(model, state, x_start, flow, result=None):
    solid = model.diagnostics(state.x)
    pressure, tension = model.loads.at(state.time)
    used_time = None if result is None else result.diagnostics['used_force_time_s']
    diagnostics = {} if result is None else result.diagnostics['fluid']
    solves = diagnostics.get('solves', {}).values()
    kinetic = diagnostics.get('kinetic_energy')
    divergence = diagnostics.get('corrected_divergence_l2')
    if kinetic is None:
        kinetic = .5*flow.rho*(state.velocity*flow.op.velocity_mass(state.velocity)).sum().item()
    if divergence is None:
        divergence = flow.divergence_l2(state.velocity)
    return dict(step=state.step, time_s=state.time,
        prescribed_pressure_mmhg=pressure/MMHG_TO_DYN_PER_CM2,
        prescribed_tension_dyn_per_cm2=tension,
        applied_force_time_s=used_time,
        applied_pressure_mmhg=0. if used_time is None else model.loads.at(used_time)[0]/MMHG_TO_DYN_PER_CM2,
        next_force_time_s=state.force_time,
        **{key: solid[key] for key in ('cavity_volume_ml', 'wall_volume_cm3', 'minimum_detF', 'maximum_detF')},
        max_incremental_displacement_cm=torch.linalg.vector_norm(state.x-x_start, dim=-1).max().item(),
        max_fluid_speed_cm_per_s=torch.linalg.vector_norm(state.velocity, dim=-1).max().item(),
        fluid_courant=(flow.dt*(state.velocity.abs()/state.velocity.new_tensor(
            flow.op.mesh.velocity_grid.spacing)).sum(-1).max()).item(),
        max_grid_displacement=0. if result is None else result.diagnostics['max_grid_displacement'],
        force_norm_dyn=torch.linalg.vector_norm(state.force).item(),
        kinetic_energy_erg=kinetic,
        corrected_divergence_l2=divergence,
        max_solver_residual_ratio=max((s['residual_norm']/s['tolerance'] for s in solves), default=0.),
        max_solver_iterations=max((s['iterations'] for s in diagnostics.get('solves', {}).values()), default=0))


def _accumulate(summary, row):
    for name, field, reduction in (
            ('minimum_detF', 'minimum_detF', min), ('maximum_detF', 'maximum_detF', max),
            ('minimum_cavity_volume_ml', 'cavity_volume_ml', min),
            ('maximum_cavity_volume_ml', 'cavity_volume_ml', max),
            ('maximum_fluid_speed_cm_per_s', 'max_fluid_speed_cm_per_s', max),
            ('maximum_fluid_courant', 'fluid_courant', max),
            ('maximum_grid_displacement', 'max_grid_displacement', max),
            ('maximum_solver_residual_ratio', 'max_solver_residual_ratio', max),
            ('maximum_solver_iterations', 'max_solver_iterations', max)):
        summary[name] = reduction(summary.get(name, row[field]), row[field])


@torch.no_grad()
def run(*, device='cuda', output=None, end_time=None, dt=None, mesh_size=None,
        fluid_cells=None, box_length=None, loads=None, preload=None, resume=None,
        output_every=None, checkpoint_every=None, log_every=None, history_every=None,
        write_vtk=None, backend=None, check_every=None, profile=None, solid_input=None,
        timing=False,case_config=None,ib_backend=None):
    if case_config is not None:
        if not isinstance(case_config,LVFEMSimulationConfig):
            raise TypeError('case_config must be LVFEMSimulationConfig')
        if resume or solid_input or preload or loads is not None or profile not in (None,'afsi337'):
            raise ValueError('case_config requires a fresh generated afsi337 run')
        profile='afsi337'
        case_config=replace(case_config,
            time=TimeConfig(case_config.time.dt if dt is None else dt,
                            case_config.time.end_time if end_time is None else end_time),
            fluid=replace(case_config.fluid,
                shape=case_config.fluid.shape if fluid_cells is None else (fluid_cells,)*3,
                lengths=case_config.fluid.lengths if box_length is None else (box_length,)*3),
            geometry=replace(case_config.geometry,mesh_size=case_config.geometry.mesh_size if mesh_size is None else mesh_size))
        dt,end_time=case_config.time.dt,case_config.time.end_time
        fluid_cells,box_length=case_config.fluid.shape[0],case_config.fluid.lengths[0]
    default_output=case_config.output if case_config else OutputConfig(100,200,200,True)
    output_every=default_output.output_every if output_every is None else output_every
    checkpoint_every=default_output.checkpoint_every if checkpoint_every is None else checkpoint_every
    log_every=default_output.log_every if log_every is None else log_every
    history_every=(case_config.history_every if case_config else 20) if history_every is None else history_every
    write_vtk=default_output.write_vtk if write_vtk is None else write_vtk
    backend=(case_config.backend if case_config else 'csr') if backend is None else backend
    check_every=(case_config.solver.check_every if case_config else 8) if check_every is None else check_every
    if profile not in (None, 'cycle', 'afsi337'):
        raise ValueError('unknown LV profile')
    if not resume:
        profile = 'cycle' if profile is None else profile
    if solid_input and (profile != 'afsi337' or mesh_size is not None):
        raise ValueError('solid-input requires afsi337 and preserves its mesh size')
    if profile == 'afsi337' and (loads is not None or preload is not None):
        raise ValueError('afsi337 starts unloaded with its prescribed ramp; no cycle loads/preload')
    if backend not in ('csr', 'quadrature'):
        raise ValueError('backend must be csr or quadrature')
    if type(check_every) is not int or check_every < 1:
        raise ValueError('check_every must be a positive integer')
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
    for value in (output_every, checkpoint_every, log_every, history_every):
        if type(value) is not int or value < 1:
            raise ValueError('output/checkpoint/log/history intervals must be positive integers')
    started = perf_counter()
    folder = Path(output) if output is not None else Path(resume).parent if resume else Path(
        'results/lv_afsi337' if profile == 'afsi337' else 'results/lv_cycle')
    if resume:
        if any(v is not None for v in (dt, mesh_size, fluid_cells, box_length, loads, preload, solid_input)):
            raise ValueError('resume restores physical settings; do not also specify mesh, dt, loads or preload')
        if folder.resolve() != Path(resume).resolve().parent:
            raise ValueError('resume in the checkpoint directory to preserve the output history')
        model, state, x_start, settings, progress = load_cycle(resume, device)
        stored_profile = settings.get('profile', 'cycle')
        if profile is not None and profile != stored_profile:
            raise ValueError('resume profile disagrees with checkpoint')
        profile = stored_profile
        progress['resumptions'] += 1
        progress.pop('failure', None)
    else:
        if (folder/'checkpoint.npz').exists() or (folder/'report.json').exists():
            raise ValueError('output contains a run; use --resume or choose a new output directory')
        dt = 5e-5 if dt is None else dt
        fluid_cells = (32 if profile == 'afsi337' else 24) if fluid_cells is None else fluid_cells
        box_length = (5. if profile == 'afsi337' else 12.) if box_length is None else box_length
        if (not isfinite(dt) or dt <= 0 or type(fluid_cells) is not int or fluid_cells < 2 or
                not isfinite(box_length) or box_length <= 0):
            raise ValueError('positive dt/box length and integer fluid_cells >= 2 required')
        if profile == 'afsi337':
            from afsi_torch.afsi337 import generated_model
            if solid_input:
                from afsi_torch.afsi337_io import load_native_solid
                model = load_native_solid(solid_input, device=device)
            else:
                if case_config:
                    model=generated_model(geometry=case_config.geometry,parameters=case_config.material,
                        loads=case_config.loads,beta=case_config.beta,basal_constraint=case_config.basal_constraint,device=device)
                else:
                    model = generated_model(mesh_size=.1 if mesh_size is None else mesh_size, device=device)
            x_start = model.mesh.X.clone()
            provenance = dict(initialization='AFSI337 unloaded reference and zero-force bootstrap',
                              solid_input=None if solid_input is None else str(solid_input))
        elif preload:
            if mesh_size is not None:
                raise ValueError('preload preserves its mesh; do not specify mesh-size')
            model, x_start, tolerance, provenance = load_preload(preload, device)
            initial_pressure, initial_tension = model.loads.at(0.)
            loads = AFSICycleLoads(initial_pressure=initial_pressure) if loads is None else loads
            if abs(loads.at(0.)[0]-initial_pressure) > 1e-9 or initial_tension != 0.:
                raise ValueError('cycle initial pressure/tension must match the saved preload')
            model.loads = loads
        else:
            loads = AFSICycleLoads() if loads is None else loads
            if loads.initial_pressure != 0.:
                raise ValueError('nonzero initial pressure requires a matching preload')
            config = LVConfig(mesh_size=1.2 if mesh_size is None else mesh_size)
            model = LVSolid(generate_lv(config, device=device), loads=loads)
            x_start = model.mesh.X.clone()
            provenance = dict(initialization='generated unloaded reference, AFSI zero-force bootstrap')
        config = model.mesh.config
        center = list(config.center)
        center[2] += .5*(config.base_height-config.outer_axes[2])
        settings = dict(profile=profile, dt=dt, fluid_cells=fluid_cells, box_length=box_length,
            origin=[0.,0.,0.] if profile == 'afsi337' else [value-box_length/2 for value in center], rho=1., mu=1.,
            solver=asdict(SolverOptions(max_iterations=4000, recompute_every=200)))
        if case_config:
            settings.update(fluid_shape=case_config.fluid.shape,fluid_lengths=case_config.fluid.lengths,
                origin=case_config.fluid.origin,rho=case_config.fluid.rho,mu=case_config.fluid.mu,
                solver=asdict(case_config.solver))
        progress = dict(initial=model.diagnostics(x_start), summary={}, provenance=provenance,
                        elapsed_seconds=0., resumptions=0)
    end_time = (2. if profile == 'afsi337' else .8) if end_time is None else end_time
    if not isfinite(end_time) or end_time <= 0:
        raise ValueError('end_time must be positive and finite')
    dt = settings['dt']
    steps = round(end_time/dt)
    if steps < 1 or abs(steps*dt-end_time) > 1e-10*max(1., end_time):
        raise ValueError('end_time must be an integer multiple of dt')
    if resume and state.step > steps:
        raise ValueError('requested end time precedes the checkpoint')
    fluid_mesh = create_box(tuple(settings.get('fluid_shape',(settings['fluid_cells'],)*3)),
                            tuple(settings.get('fluid_lengths',(settings['box_length'],)*3)),
                            settings['origin'], device=device)
    settings['ib_backend'] = (case_config.ib_backend if case_config else settings.get('ib_backend','reference')) if ib_backend is None else ib_backend
    if settings['ib_backend'] not in ('reference','cuda'):
        raise ValueError('invalid IB backend')
    settings['backend'] = backend
    settings['solver']['check_every'] = check_every
    progress.setdefault('execution_segments', []).append(dict(
        start_step=state.step if resume else 0, backend=backend, check_every=check_every,
        history_every=history_every, timing=bool(timing),
        diagnostic_sampling='history/log/checkpoint/output/end'))
    operators = prepare_operators(fluid_mesh)
    if backend == 'csr':
        operators = CSRFluidOperators(operators)
    flow = ChorinSolver(operators, dt=dt, rho=settings['rho'], mu=settings['mu'],
                        options=SolverOptions(**settings['solver']))
    from functools import partial
    print(f'BASE: {model.basal_constraint_mode}, beta={model.beta:g}; '
          f'long axis={model.mesh.config.long_axis}, center={model.mesh.config.center}',flush=True)
    driver = ExplicitIBStepper(flow, model.force, model.validate,
        stencil_factory=partial(ib.prepare_stencil,backend=settings['ib_backend']))
    timer = StepTimingRecorder(device, progress.get('timing')) if timing else None
    if not resume:
        state = (driver.initialize_equilibrium(x_start, force_tolerance=tolerance)
                 if preload else driver.initialize(x_start))
    else:
        flow.op._field(state.velocity, vector=True)
        flow.op._field(state.pressure, pressure=True)
        ib.prepare_stencil(state.x, fluid_mesh.velocity_grid)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'loads.csv').open('w', newline='', encoding='utf-8') as load_stream:
        load_writer = csv.writer(load_stream)
        load_writer.writerow(['time_s', 'prescribed_pressure_mmhg', 'prescribed_tension_dyn_per_cm2'])
        for sample in range(1001):
            time = end_time*sample/1000
            pressure, tension = model.loads.at(time)
            load_writer.writerow([time, pressure/MMHG_TO_DYN_PER_CM2, tension])
    previous_step = _history(folder/'history.csv', state.step)
    stream = (folder/'history.csv').open('a', newline='', encoding='utf-8')
    journal = csv.DictWriter(stream, fieldnames=COLUMNS)
    writer = None
    if write_vtk:
        if not resume:
            write_lv(folder/'geometry', model.mesh, model.fibers)
        writer = CoupledWriter(folder, model.mesh, fluid_mesh,
                               resume_time=state.time if resume else None)
        if not writer.frames or writer.frames[-1][0] != state.time:
            writer.write(state)
    row = progress['last'] if 'last' in progress else _row(model, state, x_start, flow)
    if previous_step != state.step:
        journal.writerow(row)
    _accumulate(progress['summary'], row)
    progress['last'] = row
    segment_step = state.step
    previous_elapsed = progress['elapsed_seconds']
    status = 'running'
    result = None

    def save():
        if progress['last']['step'] != state.step:
            progress['last'] = _row(model, state, x_start, flow, result)
            _accumulate(progress['summary'], progress['last'])
        stream.flush()
        if timer is not None:
            progress['timing'] = timer.snapshot()
        progress['elapsed_seconds'] = previous_elapsed + perf_counter()-started
        save_cycle(folder/'checkpoint.npz', model, state, x_start, settings, progress)
        period = getattr(model.loads, 'period', None)
        report = dict(schema=1, demo='afsi337-ramp-hold' if profile == 'afsi337' else 'generated-ideal-lv-prescribed-cycle', status=status,
            completed=state.step == steps, full_cycle_completed=period is not None and state.time+1e-12 >= period,
            requested_end_time_s=end_time, reached_time_s=state.time, accepted_steps=state.step,
            requested_steps=steps, period_s=period, device=str(device), torch=torch.__version__,
            gpu=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None,
            units=CGS_UNITS, settings=settings, solid_config=asdict(model.mesh.config),
            loads=asdict(model.loads), material=asdict(model.parameters), beta=model.beta,
            basal_constraint=model.basal_constraint_mode,
            solid_nodes=len(state.x), solid_cells=len(model.mesh.cells),
            fluid_velocity_nodes=len(state.velocity), fluid_pressure_nodes=len(state.pressure),
            fluid_element_size_cm=settings['box_length']/settings['fluid_cells'],
            fluid_element_sizes_cm=tuple(2*h for h in fluid_mesh.velocity_grid.spacing),
            velocity_lattice_spacing_cm=fluid_mesh.velocity_grid.spacing,
            ib_kernel='AFSI four-point Peskin, epsilon equals velocity lattice spacing',
            force_path='integrated solid force -> IB density -> Q2 consistent mass',
            force_order='flow(g_n), interpolate at x_n, update x, force(x_new,t_n)',
            pressure_output='background Chorin pressure is distinct from prescribed endocardial traction',
            pressure_volume_curve='prescribed pressure at state time versus cavity volume',
            periodic_steady_state_established=False, circulation_model=False,
            grid_convergence_established=False,
            diagnostic_scope=dict(expensive_extrema='sampled at history/log/checkpoint/output/end',
                every_step='true solver residual, finite fields, deformation and IB support/displacement guards',
                ib_power_diagnostics=False,
                timing='CUDA events on GPU or perf_counter on CPU, read at report saves'
                if timing else 'disabled'),
            source_alignment=dict(demo='afsic/demo/demo_337/fsi_paralell_fibers_contraction.py',
                demo_blob='283b23f5155dbc57043edd2aa7280d61c3c8e985',
                load_shapes='afsic/demo/demo_337/PressureEndo.py',
                load_shapes_blob='8bf2cce74346522f2d10139765ecec0781f1d2a5',
                original_dt=5e-5, original_fluid_element_size_cm=5./32,
                differences=['generated geometry and rule-based fibers replace external files',
                    '0.8 s helper waveform replaces active demo 1.5 s linear ramp',
                    'initial filling ramp occurs only once to avoid pressure jump on later cycles',
                    '8 mmHg diastolic level follows helper; pressure/tension amplitudes follow demo',
                    'larger fluid box contains this generated LV; report records actual mesh spacing']),
            **progress)
        if profile == 'afsi337':
            configuration=asdict(LVFEMSimulationConfig(time=TimeConfig(dt,end_time),
                fluid=FEMFluidConfig(tuple(settings.get('fluid_shape',(settings['fluid_cells'],)*3)),
                    tuple(settings.get('fluid_lengths',(settings['box_length'],)*3)),
                    tuple(settings['origin']),settings['rho'],settings['mu']),
                geometry=model.mesh.config,material=model.parameters,loads=model.loads,beta=model.beta,
                basal_constraint=model.basal_constraint_mode,
                solver=SolverOptions(**settings['solver']),backend=backend,history_every=history_every,ib_backend=settings['ib_backend'],
                output=OutputConfig(log_every,checkpoint_every,output_every,write_vtk)))
            report['configuration']=configuration
            atomic_json(folder/'configuration.json',configuration)
            from afsi_torch.afsi337 import alignment
            report.update(source_alignment=alignment(model, settings),
                load_protocol='ramp-and-hold', reference_load_completed=state.time+1e-12 >= 1.5,
                reference_horizon_completed=state.time+1e-12 >= 2.,
                static_equilibrium_established=False)
        atomic_json(folder/'report.json', report)
        return report

    try:
        save()
        print(f'ideal LV: {device}, {len(state.x)} solid nodes, '
              f'{tuple(settings.get("fluid_shape",(settings["fluid_cells"],)*3))} Q2 fluid cells, dt={dt:g} s; '
              f'continuing at step {state.step} toward {steps} ({end_time:g} s)', flush=True)
        for _ in range(state.step, steps):
            try:
                result = driver.step(state, diagnostics=False, timing=timer)
            except BaseException:
                if timer is not None:
                    timer.abort()
                raise
            state = result.state
            solves = result.diagnostics['fluid']['solves'].values()
            for key, value in (
                    ('maximum_grid_displacement', result.diagnostics['max_grid_displacement']),
                    ('maximum_solver_residual_ratio', max(s['residual_norm']/s['tolerance'] for s in solves)),
                    ('maximum_solver_iterations', max(s['iterations'] for s in solves))):
                progress['summary'][key] = max(progress['summary'].get(key, value), value)
            sample = state.step == steps or any(state.step % n == 0 for n in
                (history_every, log_every, checkpoint_every)) or (writer and state.step % output_every == 0)
            if sample:
                row = _row(model, state, x_start, flow, result)
                progress['last'] = row
                _accumulate(progress['summary'], row)
            if state.step % history_every == 0 or state.step == steps:
                journal.writerow(row)
            if writer and (state.step % output_every == 0 or state.step == steps):
                writer.write(state)
            if state.step % checkpoint_every == 0:
                save()
            if state.step % log_every == 0 or state.step == steps:
                elapsed = perf_counter()-started
                rate = (state.step-segment_step)/max(elapsed, 1e-12)
                remaining = (steps-state.step)/max(rate, 1e-12)
                print(f'step {state.step}/{steps}, t={state.time:.6f} s, '
                      f'V={row["cavity_volume_ml"]:.6g} mL, minJ={row["minimum_detF"]:.6g}, '
                      f'p={row["prescribed_pressure_mmhg"]:.4g} mmHg, '
                      f'Ta={row["prescribed_tension_dyn_per_cm2"]:.4g} dyn/cm2, '
                      f'estimated remaining={remaining/60:.1f} min', flush=True)
        status = 'completed'
        return save()
    except (Exception, KeyboardInterrupt) as exc:
        status = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        progress['failure'] = dict(type=type(exc).__name__, message=str(exc),
                                   last_accepted_step=state.step)
        save()
        raise
    finally:
        stream.close()


def main(default_profile=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--end-time', type=float, help='default: 2 s for afsi337, 0.8 s for cycle')
    parser.add_argument('--profile', choices=('cycle', 'afsi337'), default=default_profile)
    parser.add_argument('--solid-input', help='optional native AFSI337 numeric export; otherwise generate geometry')
    parser.add_argument('--dt', type=float)
    parser.add_argument('--mesh-size', type=float)
    parser.add_argument('--fluid-cells', type=int)
    parser.add_argument('--box-length', type=float)
    parser.add_argument('--preload')
    parser.add_argument('--resume')
    parser.add_argument('--output')
    parser.add_argument('--output-every', type=int, default=200)
    parser.add_argument('--checkpoint-every', type=int, default=200)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--history-every', type=int, default=20)
    parser.add_argument('--no-vtk', action='store_true')
    parser.add_argument('--backend', choices=('csr', 'quadrature'), default='csr')
    parser.add_argument('--check-every', type=int, default=8, help='PCG host check interval; true residual verified on return')
    parser.add_argument('--timing', action='store_true', help='record cumulative IB, solid and fluid stage timings')
    parser.add_argument('--diastole-pressure-mmhg', type=float)
    parser.add_argument('--systole-pressure-mmhg', type=float)
    parser.add_argument('--max-tension', type=float, help='active stress amplitude in dyn/cm^2')
    args = parser.parse_args()
    options = vars(args).copy()
    write_vtk = not options.pop('no_vtk')
    overrides = {}
    for arg, key, factor in (('diastole_pressure_mmhg', 'diastole_pressure', MMHG_TO_DYN_PER_CM2),
                            ('systole_pressure_mmhg', 'systole_pressure', MMHG_TO_DYN_PER_CM2),
                            ('max_tension', 'max_tension', 1.)):
        value = options.pop(arg)
        if value is not None:
            overrides[key] = value*factor
    if overrides:
        if args.preload:
            # Preload is validated again by load_preload inside run; only its
            # prescribed initial pressure is needed to construct the schedule.
            preload_report = json.loads((Path(args.preload)/'report.json').read_text(encoding='utf-8'))
            overrides['initial_pressure'] = preload_report['target_pressure_mmhg']*MMHG_TO_DYN_PER_CM2
        options['loads'] = AFSICycleLoads(**overrides)
    report = run(**options, write_vtk=write_vtk)
    print(json.dumps(dict(status=report['status'], reached_time_s=report['reached_time_s'],
                         full_cycle_completed=report['full_cycle_completed'],
                         summary=report['summary']), indent=2))


if __name__ == '__main__':
    main()
