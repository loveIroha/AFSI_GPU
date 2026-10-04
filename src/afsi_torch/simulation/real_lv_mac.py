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
from ..transport import transport_numbers, coupling_policy


@torch.no_grad()
def run(*, case_config=None, device='cuda', output=None, resume=None, end_time=None, resume_dt=None,
        coupling_scheme=None, nonlinear_solver=None):
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if resume and case_config is not None:
        raise ValueError('resume restores physical configuration; omit --config and physical overrides')
    if resume_dt is not None and not resume:
        raise ValueError('resume_dt requires a checkpoint')
    started = perf_counter()
    folder = Path(output) if output else Path(resume).parent if resume else Path('results/demo_real_lv/mac')
    refinement = resume_dt is not None
    branch = refinement or resume and (coupling_scheme is not None or nonlinear_solver is not None)
    if branch:
        if output is None or folder.resolve() == Path(resume).resolve().parent:
            raise ValueError('dt/scheme change requires a new output directory; source is preserved')
    if not resume or branch:
        if any((folder/name).exists() for name in ('report.json', 'checkpoint.npz', 'history.csv', 'vtk')):
            raise ValueError('output contains a run; use a new directory or resume')
    if resume:
        if not branch and folder.resolve() != Path(resume).resolve().parent:
            raise ValueError('resume in the checkpoint directory')
        model, state, settings, progress, config = load_real_lv(resume, device)
        old_scheme = config.coupling.scheme
        details = dict(old_dt_s=config.time.dt,new_dt_s=config.time.dt,source_step=state.step,
                       start_step=state.step,start_time_s=state.time,refinement_factor=1)
        if refinement:
            state, settings, config, details = refine_checkpoint_dt(
                model, state, settings, config, resume_dt, end_time)
        elif end_time is not None:
            config = replace(config, time=TimeConfig(config.time.dt, end_time))
        if coupling_scheme is not None:
            config = replace(config,coupling=replace(config.coupling,scheme=coupling_scheme))
            force_time = None if state.step == 0 else state.time if coupling_scheme in ('implicit-newton','cnab-midpoint','cnab-semiimplicit') else (state.step-1)*config.time.dt
            state = replace(state, force_time=force_time, force=state.force if force_time is None else model.force(state.x,force_time),
                            previous_advection=state.previous_advection if old_scheme==coupling_scheme and not refinement else None)
            settings = dict(settings,coupling=asdict(config.coupling))
            details.update(old_scheme=old_scheme,new_scheme=config.coupling.scheme,force_time_s=force_time,
                           force_resampled=state.step > 0)
        if nonlinear_solver is not None:
            details.update(old_nonlinear_solver=config.coupling.semiimplicit_solver,new_nonlinear_solver=nonlinear_solver)
            config = replace(config,coupling=replace(config.coupling,semiimplicit_solver=nonlinear_solver))
            settings = dict(settings,coupling=asdict(config.coupling))
        if branch:
            progress = dict(elapsed_seconds=0., segments=[], summary={}, restart_from=dict(
                checkpoint=str(Path(resume).resolve()), source_elapsed_seconds=progress['elapsed_seconds'],
                source_failure=progress.get('failure'), **details))
    else:
        config = replace(RealLVConfig(),coupling=replace(RealLVConfig().coupling,
            scheme='cnab-semiimplicit',semiimplicit_solver='anderson-newton')) if case_config is None else case_config
        if not isinstance(config, RealLVConfig):
            raise TypeError('case_config must be RealLVConfig')
        if end_time is not None:
            config = replace(config, time=TimeConfig(config.time.dt, end_time))
        if coupling_scheme is not None:
            config = replace(config,coupling=replace(config.coupling,scheme=coupling_scheme))
        if nonlinear_solver is not None:
            config = replace(config,coupling=replace(config.coupling,semiimplicit_solver=nonlinear_solver))
        model = imported_model(config, device)
        settings = dict(dt=config.time.dt, fluid_shape=config.fluid.shape,
                        fluid_lengths=config.fluid.lengths, fluid_origin=config.fluid.origin,
                        rho=config.fluid.rho, mu=config.fluid.mu,
                        interaction_degree=config.interaction_degree,interaction_quadrature=asdict(config.interaction_quadrature),
                        coupling=asdict(config.coupling),
                        pressure_solver=asdict(config.pressure_solver), mass_solver=asdict(config.mass_solver),
                        **asdict(config.execution))
        progress = dict(elapsed_seconds=0., segments=[], summary={})
    implicit = config.coupling.scheme == 'implicit-newton'
    cnab = config.coupling.scheme in ('cnab-midpoint','cnab-semiimplicit')
    semiimplicit = config.coupling.scheme == 'cnab-semiimplicit'
    if nonlinear_solver is not None and not semiimplicit:
        raise ValueError('nonlinear_solver override requires cnab-semiimplicit')
    policy = coupling_policy(config.coupling.scheme)
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
                                    coupling=asdict(config.coupling),transport_policy=policy))
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
                    stokes_iterations=info.get('flow',{}).get('pressure',{}).get('schur_iterations'),
                    momentum_residual=info.get('flow',{}).get('stokes',{}).get('momentum_residual'),
                    power_error=info.get('power_error'), power_error_sampled='power_error' in info,
                    newton_iterations=info.get('nonlinear',{}).get('newton_iterations',info.get('nonlinear',{}).get('iterations')),
                    anderson_iterations=info.get('nonlinear',{}).get('anderson_iterations'),
                    gmres_iterations=sum(h.get('linear',{}).get('iterations',0) for h in info.get('nonlinear',{}).get('history',[])),
                    jacobian_actions=info.get('nonlinear',{}).get('jacobian_actions'),
                    tangent_assemblies=info.get('nonlinear',{}).get('tangent_assemblies'),
                    residual_evaluations=info.get('nonlinear',{}).get('residual_evaluations'),
                    stokes_solves=info.get('nonlinear',{}).get('stokes_solves'),
                    nonlinear_residual=info.get('nonlinear',{}).get('residual_norm'),
                    nonlinear_tolerance=info.get('nonlinear',{}).get('tolerance'),
                    final_evaluation_reused=info.get('nonlinear',{}).get('final_evaluation_reused'),
                    interaction_points=transfer.quadrature_summary()['point_count'] if hasattr(transfer,'quadrature_summary') else transfer.geometry.weights.numel())

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
                      interaction_quadrature_degree=None if config.interaction_quadrature.mode=='adaptive' else config.interaction_degree,
                      reference_mass_quadrature_degree=config.interaction_degree,
                      interaction_points=transfer.quadrature_summary()['point_count'] if hasattr(transfer,'quadrature_summary') else transfer.geometry.weights.numel(),
                      interaction_quadrature=transfer.quadrature_summary() if hasattr(transfer,'quadrature_summary') else dict(mode='fixed'),
                      fluid_pressure_cells=state.pressure.numel(),
                      fluid_velocity_dofs=sum(v.numel() for v in state.velocity),
                      fluid_spacing_cm=grid.spacing, viscous_number=flow.viscous_number,
                      transport_policy=policy,
                      coupled_unknown_dofs=state.x.numel() if semiimplicit else sum(v.numel() for v in state.velocity) if implicit else 0,
                      pressure_backend=flow.pressure_solver.backend, mg_levels=flow.pressure_solver.shapes,
                      coupling='quadrature FE/IB with consistent CSR mass and shared MAC solver',
                      boundary='endo follower pressure; basal radial projection in xy plus fixed z; free epi',
                      time_scheme=('CN viscosity/AB2 convection; solved nonlinear midpoint FE force; IB geometry frozen at predicted midpoint; solid-node CSR tangent correction' if semiimplicit else 'CN viscosity/AB2 convection; predicted-midpoint FE force/IB geometry; average-velocity structure update; predictor-corrector startup'
                                   if cnab else 'backward Euler transport and new-time FE force; reduced coupled Newton; IB geometry frozen at old position'
                                   if implicit else 'SSPRK3 fluid with step-frozen force; first-order explicit partitioned FE/IB; preceding-time force sampling'
                                   if config.coupling.scheme == 'explicit-rk3' else 'explicit partitioned MAC; updated solid force sampled at preceding state time'),
                      pressure_time_meaning='half-time Stokes multiplier' if cnab else 'RK-weighted step average' if config.coupling.scheme == 'explicit-rk3' else 'projection multiplier',
                      advection=config.coupling.cnab.advection if cnab else 'centered',
                      pressure_gauge='closed box, homogeneous Neumann, zero mean',
                      cavity_measurement='endocardium plus virtual mean-rim triangle fan; no cap traction',
                      volume_penalty='kappa*(ln J)^2; finite penalty, not a mixed incompressible constraint',
                      last=history[-1], last_solver_info=info,
                      visualization=writer.summary() if writer else dict(enabled=False),
                      full_horizon_validated=False, mesh_convergence_established=False, **progress)
        save_real_lv(folder/'checkpoint.npz', model, state, settings, progress, config)
        if semiimplicit and status=='running' and state.step % config.output.checkpoint_every == 0:
            # Scheduled snapshots survive the failure-save of checkpoint.npz.
            # A restart candidate is not a claim of numerical convergence.
            save_real_lv(folder/'recovery_checkpoint.npz', model, state, settings, progress, config)
        atomic_json(folder/'configuration.json', asdict(config))
        atomic_json(folder/'report.json', report)
        return report

    try:
        if writer:
            writer.write(state)
        save('running')
        print(f'Real LV H-O/P1: {device}, grid={grid.shape}, nodes={len(state.x)}, '
              f'cells={len(model.mesh.cells)}, IB points={history[-1]["interaction_points"]}, '
              f'dt={config.time.dt:g}; steps {state.step}->{steps}', flush=True)
        print(f'kappa={config.material.kappa:g}, beta={config.beta:g}; '
              f'execution={config.execution.execution_backend}, pressure={flow.pressure_solver.backend}, '
              f'mass={config.execution.mass_backend}, coupling={config.execution.coupling_backend}', flush=True)
        print(f'Time scheme: {config.coupling.scheme}', flush=True)
        print(f'Constitutive law: supplied H-O UFL, active stretch factor={config.material.active_stretch_slope:g}',flush=True)
        print(f'Interaction quadrature: {config.interaction_quadrature.mode}; '
              f'point density={config.interaction_quadrature.point_density:g}; '
              f'rule family={config.interaction_quadrature.rule_family}',flush=True)
        if semiimplicit:
            print(f'Nonlinear solver: {config.coupling.semiimplicit_solver}',flush=True)
        for _ in range(state.step, steps):
            sample = (state.step+1) % config.output.log_every == 0 or state.step+1 == steps
            state, info = driver.step(state, diagnostics=sample)
            if writer and (state.step % config.output.output_every == 0 or state.step == steps):
                writer.write(state)
            if sample:
                history.append(row())
                last = history[-1]
                nonlinear = info.get('nonlinear',{})
                suffix = (f', Newton={nonlinear["iterations"]}, residual={nonlinear["residual_norm"]:.3g}' if nonlinear else '')
                if semiimplicit and nonlinear:
                    suffix = (f', AA={nonlinear["anderson_iterations"]}, Newton={nonlinear["newton_iterations"]}, '
                        f'GMRES={last["gmres_iterations"]}, Stokes calls={nonlinear["stokes_solves"]}, '
                        f'residual={nonlinear["residual_norm"]:.3g}')
                if cnab and not semiimplicit:
                    suffix = f', Stokes={last["stokes_iterations"]}, momentum={last["momentum_residual"]:.3g}'
                print(f'step {state.step}/{steps}, t={state.time:.6f}, V={last["cavity_volume_ml"]:.8g} mL, '
                      f'minJ={last["minimum_detF"]:.6g}, div={last["divergence_l2"]:.3g}, '
                      f'MG cycles={last["pressure_cycles"]}, CFL={last["courant"]:.3g}, '
                      f'A={last["advection_diffusion_number"]:.3g}, Re_h={last["cell_reynolds"]:.3g}{suffix}', flush=True)
            progress['summary']['max_grid_displacement'] = max(
                progress['summary'].get('max_grid_displacement', 0.), info['max_grid_displacement'])
            if state.step % config.output.checkpoint_every == 0:
                save('running')
        return save('completed')
    except (Exception, KeyboardInterrupt) as exc:
        progress['failure'] = dict(type=type(exc).__name__, message=str(exc), last_accepted_step=state.step)
        if hasattr(exc, 'diagnostics'):
            key = 'stokes' if exc.diagnostics.get('stage')=='CN Stokes' else 'transport_guard'
            progress['failure'][key] = exc.diagnostics
        if hasattr(exc, 'result'):
            result = exc.result
            progress['failure']['nonlinear'] = dict(iterations=result.iterations,
                residual_norm=result.residual_norm,tolerance=result.tolerance,history=result.history)
        if hasattr(exc, 'coupled_diagnostics'):
            progress['failure']['coupled_acceptance'] = exc.coupled_diagnostics
        save('failed')
        raise
