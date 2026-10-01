"""Three-cycle imported P1 H--O LV demo using the shared GPU MAC/IB solver."""
import csv
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
import torch
from ..real_lv import RealLVConfig, imported_model
from ..real_lv_checkpoint import save_real_lv, load_real_lv, refine_checkpoint_dt
from ..config import TimeConfig, lv_grid
from ..cycle_checkpoint import atomic_json
from ..mac.execution import build_driver
from ..mac.grid import divergence
from ..transport import transport_numbers, transport_policy


@torch.no_grad()
def run(*, case_config=None, device='cuda', output=None, resume=None, end_time=None, resume_dt=None):
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if resume and case_config is not None:
        raise ValueError('resume restores physical configuration; omit --config and physical overrides')
    if resume_dt is not None and not resume:
        raise ValueError('resume_dt requires a checkpoint')
    started = perf_counter()
    folder = Path(output) if output else Path(resume).parent if resume else Path('results/demo_real_lv/mac')
    refinement = resume_dt is not None
    if refinement:
        if output is None or folder.resolve() == Path(resume).resolve().parent:
            raise ValueError('smaller-dt resume requires a new output directory; source is preserved')
    if not resume or refinement:
        if any((folder/name).exists() for name in ('report.json', 'checkpoint.npz', 'history.csv', 'vtk')):
            raise ValueError('output contains a run; use a new directory or resume')
    if resume:
        if not refinement and folder.resolve() != Path(resume).resolve().parent:
            raise ValueError('resume in the checkpoint directory')
        model, state, settings, progress, config = load_real_lv(resume, device)
        if refinement:
            state, settings, config, details = refine_checkpoint_dt(
                model, state, settings, config, resume_dt, end_time)
            progress = dict(elapsed_seconds=0., segments=[], summary={}, restart_from=dict(
                checkpoint=str(Path(resume).resolve()), source_elapsed_seconds=progress['elapsed_seconds'],
                source_failure=progress.get('failure'), **details))
        elif end_time is not None:
            config = replace(config, time=TimeConfig(config.time.dt, end_time))
    else:
        config = RealLVConfig() if case_config is None else case_config
        if not isinstance(config, RealLVConfig):
            raise TypeError('case_config must be RealLVConfig')
        if end_time is not None:
            config = replace(config, time=TimeConfig(config.time.dt, end_time))
        model = imported_model(config, device)
        settings = dict(dt=config.time.dt, fluid_shape=config.fluid.shape,
                        fluid_lengths=config.fluid.lengths, fluid_origin=config.fluid.origin,
                        rho=config.fluid.rho, mu=config.fluid.mu,
                        interaction_degree=config.interaction_degree,
                        pressure_solver=asdict(config.pressure_solver), mass_solver=asdict(config.mass_solver),
                        **asdict(config.execution))
        progress = dict(elapsed_seconds=0., segments=[], summary={})
    steps = round(config.time.end_time/config.time.dt)
    driver = build_driver(model, settings, device)
    grid, flow, transfer = driver.flow.grid, driver.flow, driver.transfer
    if resume:
        if state.step > steps:
            raise ValueError('end time precedes checkpoint')
        grid.check_velocity(state.velocity)
        if state.pressure.shape != grid.shape:
            raise ValueError('checkpoint pressure shape differs from saved grid')
        transfer.check_support(transfer.interaction_points(state.x))
    else:
        state = driver.initialize(model.mesh.X)
    folder.mkdir(parents=True, exist_ok=True)
    writer = None
    if config.output.write_vtk:
        from ..mac.output import MACWriter
        writer = MACWriter(folder/'vtk', model, grid, resume_time=state.time if resume else None)
    history = []
    if resume and (folder/'history.csv').exists():
        with (folder/'history.csv').open(newline='', encoding='utf-8') as stream:
            history = [r for r in csv.DictReader(stream) if int(r['step']) <= state.step]
    previous_elapsed = progress['elapsed_seconds']
    progress.pop('failure', None)
    progress['segments'].append(dict(start_step=state.step, start_time_s=state.time,
                                    dt_s=config.time.dt, device=str(device),
                                    execution=asdict(config.execution), output=asdict(config.output),
                                    transport_policy=transport_policy()))
    info = {}

    def row():
        pressure, tension = model.loads.at(state.time)
        speeds = torch.stack([u.abs().max() for u in state.velocity]).tolist()
        numbers = transport_numbers(speeds, grid.spacing, flow.dt, flow.mu/flow.rho)
        return dict(step=state.step, time_s=state.time,
                    force_time_s=-1. if state.force_time is None else state.force_time,
                    pressure_load_dyn_per_cm2=pressure, active_tension_dyn_per_cm2=tension,
                    **model.diagnostics(state.x),
                    max_fluid_component_cm_per_s=max(speeds),
                    courant=numbers['courant'], cell_reynolds=numbers['cell_reynolds'],
                    advection_diffusion_number=numbers['advection_diffusion_number'],
                    divergence_l2=(grid.volume*divergence(state.velocity, grid.spacing).square().sum()).sqrt().item(),
                    pressure_cycles=info.get('flow', {}).get('pressure', {}).get('cycles', 0),
                    power_error=info.get('power_error'), power_error_sampled='power_error' in info)

    if not history or int(history[-1]['step']) != state.step:
        history.append(row())

    def save(status):
        if int(history[-1]['step']) != state.step:
            history.append(row())
        temporary = folder/'history.csv.tmp'
        with temporary.open('w', newline='', encoding='utf-8') as stream:
            fields = list(dict.fromkeys(key for record in history for key in record))
            csv_writer = csv.DictWriter(stream, fieldnames=fields)
            csv_writer.writeheader(); csv_writer.writerows(history)
        temporary.replace(folder/'history.csv')
        progress['elapsed_seconds'] = previous_elapsed+perf_counter()-started
        report = dict(schema=1, demo='real-left-ventricle-HO-P1-MAC', status=status,
                      completed=state.step == steps, accepted_steps=state.step,
                      reached_time_s=state.time, requested_end_time_s=config.time.end_time,
                      requested_cycles=config.time.end_time/config.loads.period,
                      device=str(device), torch=torch.__version__, units='cm-g-s',
                      gpu=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
                      configuration=asdict(config), settings=settings,
                      source_mesh=model.mesh.metadata, solid_nodes=len(state.x), solid_cells=len(model.mesh.cells),
                      solid_element='P1', material_model='user-HO-iso-I1-DG0',
                      direction_location='cell-DG0; unmodified fiber/sheet',
                      solid_quadrature_degree=config.solid_degree,
                      solid_quadrature_points_per_cell=model.geometry.values.shape[0],
                      interaction_quadrature_degree=config.interaction_degree,
                      interaction_points=transfer.geometry.weights.numel(),
                      fluid_pressure_cells=state.pressure.numel(),
                      fluid_velocity_dofs=sum(v.numel() for v in state.velocity),
                      fluid_spacing_cm=grid.spacing, viscous_number=flow.viscous_number,
                      transport_policy=transport_policy(),
                      pressure_backend=flow.pressure_solver.backend, mg_levels=flow.pressure_solver.shapes,
                      coupling='quadrature FE/IB with consistent CSR mass and shared MAC solver',
                      boundary='endo follower pressure; basal radial projection in xy plus fixed z; free epi',
                      time_scheme='explicit partitioned MAC; updated solid force sampled at preceding state time',
                      pressure_gauge='closed box, homogeneous Neumann, zero mean',
                      cavity_measurement='endocardium plus virtual mean-rim triangle fan; no cap traction',
                      volume_penalty='kappa*(ln J)^2; finite penalty, not a mixed incompressible constraint',
                      last=history[-1], last_solver_info=info,
                      visualization=writer.summary() if writer else dict(enabled=False),
                      full_horizon_validated=False, mesh_convergence_established=False, **progress)
        save_real_lv(folder/'checkpoint.npz', model, state, settings, progress, config)
        atomic_json(folder/'configuration.json', asdict(config))
        atomic_json(folder/'report.json', report)
        return report

    try:
        if writer:
            writer.write(state)
        save('running')
        print(f'Real LV H-O/P1: {device}, grid={grid.shape}, nodes={len(state.x)}, '
              f'cells={len(model.mesh.cells)}, IB points={transfer.geometry.weights.numel()}, '
              f'dt={config.time.dt:g}; steps {state.step}->{steps}', flush=True)
        print(f'kappa={config.material.kappa:g}, beta={config.beta:g}; '
              f'execution={config.execution.execution_backend}, pressure={flow.pressure_solver.backend}, '
              f'mass={config.execution.mass_backend}, coupling={config.execution.coupling_backend}', flush=True)
        for _ in range(state.step, steps):
            sample = (state.step+1) % config.output.log_every == 0 or state.step+1 == steps
            state, info = driver.step(state, diagnostics=sample)
            if writer and (state.step % config.output.output_every == 0 or state.step == steps):
                writer.write(state)
            if sample:
                history.append(row())
                last = history[-1]
                print(f'step {state.step}/{steps}, t={state.time:.6f}, V={last["cavity_volume_ml"]:.8g} mL, '
                      f'minJ={last["minimum_detF"]:.6g}, div={last["divergence_l2"]:.3g}, '
                      f'MG cycles={last["pressure_cycles"]}, CFL={last["courant"]:.3g}, '
                      f'A={last["advection_diffusion_number"]:.3g}, Re_h={last["cell_reynolds"]:.3g}', flush=True)
            progress['summary']['max_grid_displacement'] = max(
                progress['summary'].get('max_grid_displacement', 0.), info['max_grid_displacement'])
            if state.step % config.output.checkpoint_every == 0:
                save('running')
        return save('completed')
    except (Exception, KeyboardInterrupt) as exc:
        progress['failure'] = dict(type=type(exc).__name__, message=str(exc), last_accepted_step=state.step)
        if hasattr(exc, 'diagnostics'):
            progress['failure']['transport_guard'] = exc.diagnostics
        save('failed')
        raise
