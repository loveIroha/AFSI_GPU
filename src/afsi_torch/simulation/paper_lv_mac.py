"""Real-LV BE-BE driver for paper inflation and prescribed active cycles."""
import csv
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
import torch
from ..paper_lv import PaperHOParameters, PaperLVConfig, imported_model
from ..paper_lv_checkpoint import save as save_checkpoint, load as load_checkpoint
from ..config import TimeConfig
from ..cycle_checkpoint import atomic_json
from ..mac.paper_coupling import BEIBStepper
from ..mac.memory import allocator_sample


@torch.no_grad()
def run(*, case_config=None, device='cuda', output=None, resume=None, end_time=None,
        anderson_policy=None, newton_preconditioner=None,linear_policy=None,ib_response_backend=None,
        ib_csr_assembly_backend=None,ib_csr_contraction_backend=None,
        anderson_max_iterations=None,nonlinear_solver=None,ib_max_order=None,ib_max_points=None):
    if anderson_max_iterations is not None and (type(anderson_max_iterations) is not int or anderson_max_iterations<1):
        raise ValueError('positive integer anderson_max_iterations required')
    if nonlinear_solver is not None and nonlinear_solver not in ('jfnk','newton','anderson-newton'):
        raise ValueError('invalid paper nonlinear_solver')
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if resume and case_config is not None:
        raise ValueError('resume restores the saved paper configuration')
    started = perf_counter()
    folder = Path(output) if output else Path(resume).parent if resume else Path('results/real_lv_ma2024')
    if resume:
        if folder.resolve() != Path(resume).resolve().parent:
            raise ValueError('resume in the checkpoint directory')
        model, state, config, progress = load_checkpoint(resume, device)
        if end_time is not None:
            config = replace(config, time=TimeConfig(config.time.dt, end_time))
        if config.time.end_time < state.time:
            raise ValueError('end time precedes checkpoint')
    else:
        if any((folder/n).exists() for n in ('report.json', 'checkpoint.npz', 'history.csv', 'vtk')):
            raise ValueError('output contains a run; choose a new directory or resume')
        config = case_config or PaperLVConfig()
        if not isinstance(config, PaperLVConfig):
            raise TypeError('paper real-LV demo requires PaperLVConfig')
        if end_time is not None:
            config = replace(config, time=TimeConfig(config.time.dt, end_time))
        model = imported_model(config, device)
        progress = dict(elapsed_seconds=0.)
    if anderson_policy is not None:
        from ..mac.midpoint_solver import anderson_policy as select_policy
        config = replace(config,anderson=select_policy(config.anderson,anderson_policy))
    if anderson_max_iterations is not None:
        config = replace(config,anderson=replace(config.anderson,max_iterations=anderson_max_iterations))
    if nonlinear_solver is not None:
        config = replace(config,nonlinear_solver=nonlinear_solver)
    if newton_preconditioner is not None:
        config = replace(config,newton_preconditioner=newton_preconditioner)
    if linear_policy is not None:
        from ..nonlinear import coupled_linear_policy
        config = replace(config,nonlinear=coupled_linear_policy(config.nonlinear,linear_policy))
    config = replace(config,**{k:v for k,v in dict(ib_response_backend=ib_response_backend,
        ib_csr_assembly_backend=ib_csr_assembly_backend,
        ib_csr_contraction_backend=ib_csr_contraction_backend).items() if v is not None})
    from ..mac.adaptive_transfer import increase_quadrature_budget
    quadrature = increase_quadrature_budget(config.interaction_quadrature,
        max_order=ib_max_order,max_points=ib_max_points)
    if quadrature!=config.interaction_quadrature:
        progress.setdefault('quadrature_budget_changes',[]).append(dict(
            accepted_step=state.step if resume else 0,
            previous=asdict(config.interaction_quadrature),updated=asdict(quadrature)))
        config = replace(config,interaction_quadrature=quadrature)
    driver = BEIBStepper(model, config, device)
    if not resume:
        state = driver.initialize(model.mesh.X)
    if resume and 'failure' in progress:
        progress.setdefault('previous_failures',[]).append(progress.pop('failure'))
    folder.mkdir(parents=True, exist_ok=True)
    previous_elapsed = progress.get('elapsed_seconds', 0.)
    history = []
    if resume and (folder/'history.csv').exists():
        with (folder/'history.csv').open(newline='') as stream:
            history = [dict(r) for r in csv.DictReader(stream) if float(r['time_s']) <= state.time]
    info = {}
    from ..mac.output import MACWriter
    writer = MACWriter(folder/'vtk', model, driver.grid, resume_time=state.time if resume else None,
        pressure_description='physical MAC projection pressure; p=0 at outer box faces; endpoint time') if config.output.write_vtk else None
    steps = round(config.time.end_time/config.time.dt)
    active_cycle = config.load_protocol=='active-cycle'

    def row():
        nonlinear = info.get('nonlinear', {})
        pressure, tension = model.loads.at(state.time)
        return dict(step=state.step, time_s=state.time,
            endocardial_pressure_mmhg=pressure/1333.22387415,
            endocardial_pressure_dyn_per_cm2=pressure,
            active_tension_dyn_per_cm2=tension, **model.diagnostics(state.x),
            divergence_l2=info.get('divergence_l2', 0.), power_error=info.get('power_error', 0.),
            nonlinear_iterations=nonlinear.get('iterations', 0), nonlinear_residual=nonlinear.get('residual_norm', 0.),
            fluid_solves=nonlinear.get('fluid_solves', 0), courant=info.get('courant', 0.),
            anderson_iterations=nonlinear.get('anderson_iterations', 0),
            newton_iterations=nonlinear.get('newton_iterations', nonlinear.get('iterations',0) if config.nonlinear_solver=='newton' else 0),
            gmres_iterations=nonlinear.get('gmres_iterations', 0),
            linear_true_checks=nonlinear.get('true_residual_checks',0),
            tangent_assemblies=nonlinear.get('tangent_assemblies', 0),
            mass_solves=nonlinear.get('mass_solves', 0),
            total_pressure_cycles=nonlinear.get('pressure_cycles', 0),
            pressure_cycles=info.get('flow', {}).get('pressure', {}).get('cycles', 0))

    def save(status):
        progress['elapsed_seconds'] = previous_elapsed+perf_counter()-started
        if not history or float(history[-1]['time_s']) != state.time:
            history.append(row())
        with (folder/'history.csv').open('w', newline='') as stream:
            # Resume older histories that did not yet contain solver counters.
            fields = list(dict.fromkeys(k for record in history for k in record))
            w = csv.DictWriter(stream, fieldnames=fields); w.writeheader(); w.writerows(history)
        report = dict(status=status, accepted_steps=state.step, reached_time_s=state.time,
            requested_end_time_s=config.time.end_time, dt_s=config.time.dt, elapsed_seconds=progress['elapsed_seconds'],
            reference='Ma et al., Physics of Fluids 36, 081914 (2024), section V.F', doi='10.1063/5.0225605',
            units='cm-g-s', solid_nodes=len(state.x), solid_cells=len(model.mesh.cells), fluid_shape=driver.grid.shape,
            config=asdict(config), material='eq81/82 RAW I1 H-O with normal-stress removal and log(I3) penalty',
            load_protocol=config.load_protocol,
            case_purpose='user active-cycle extension' if active_cycle else 'paper passive-inflation reproduction',
            active_stress=dict(enabled=active_cycle, form='T(t)*(1+slope*(lambda_f-1))*(F*f0) outer f0',
                stretch_slope=config.material.active_stretch_slope, coefficient_units='dyn/cm2',
                stretch_multiplier_clipped=False),
            time_scheme='BE-BE; endpoint FE force; old-geometry dual IB; BE diffusion/Chorin projection',
            convection='first-order characteristic tracing and trilinear semi-Lagrangian MAC sampling' if config.flow.convection else 'disabled',
            fluid_boundaries='homogeneous Neumann velocity diffusion; homogeneous Dirichlet pressure on physical box faces',
            reproduction_differences=['user radial basal spring retained; paper V.F does not specify the basal constraint',
                'fixed dt rather than the adaptive time sequence shown in Fig26',
                'characteristic tracing/interpolation reconstructed; authors implementation unavailable',
                'interaction quadrature density/rule and solver tolerances are explicit implementation choices']
                + ([] if config.nonlinear_solver == 'jfnk' else ['Anderson/assembled-CSR Newton instead of paper JFNK/BiCGSTAB; same BE-BE residual'])
                + (['user material coefficients differ from paper V.F; see config.material'] if config.material != PaperHOParameters() else [])
                + (['user 0.8 s periodic pressure and active PK1 added; not the paper V.F passive benchmark'] if active_cycle else []),
            energy_claim='no unconditional energy theorem asserted for this loaded open-boundary corrected H-O experiment',
            interaction_quadrature=driver.transfer.quadrature_summary() if hasattr(driver.transfer, 'quadrature_summary') else dict(mode='fixed'),
            nonlinear_solver=config.nonlinear_solver, newton_preconditioner=config.newton_preconditioner,
            pressure_backend=driver.flow.pressure_solver.backend,
            gpu_memory=allocator_sample(state.x.device), last=history[-1], last_solver_info=info,
            visualization=writer.summary() if writer else dict(enabled=False),
            full_horizon_validated=status == 'completed' and state.time >= max(config.time.end_time,
                config.cyclic_loads.period if active_cycle else 1.5)-1e-12,
            completed_active_cycles=int((state.time+1e-12)/config.cyclic_loads.period) if active_cycle else 0,
            published_results_reproduced=False, **{k:v for k,v in progress.items() if k != 'elapsed_seconds'})
        save_checkpoint(folder/'checkpoint.npz', model, state, config, progress)
        atomic_json(folder/'configuration.json', asdict(config)); atomic_json(folder/'report.json', report)
        return report

    try:
        if writer:
            writer.write(state)
        if not history:
            history.append(row())
        save('running')
        print(f'Real LV Ma2024 BE-BE/P1: {device}, grid={driver.grid.shape}, nodes={len(state.x)}, '
              f'cells={len(model.mesh.cells)}, dt={config.time.dt:g}; steps {state.step}->{steps}', flush=True)
        load_description = (f'Active cycle: period={config.cyclic_loads.period:g} s, user pressure/tension waveform, '
                            f'active stretch slope={config.material.active_stretch_slope:g}' if active_cycle else
                            f'Passive inflation: 0->{config.loads.target_mmhg:g} mmHg over {config.loads.ramp_seconds:g} s, then held')
        print(f'{load_description}; mu={config.fluid.mu:g}; '
              f'solver={config.nonlinear_solver}; open pressure boundary; radial base retained', flush=True)
        print(f'Execution: support={config.support_backend}, Helmholtz={config.flow.helmholtz_backend}, '
              f'IB shared={config.interaction_quadrature.shared_execution}, response={config.ib_response_backend}, '
              f'CSR assembly={config.ib_csr_assembly_backend}, contraction={config.ib_csr_contraction_backend}; '
              f'GMRES checks={config.nonlinear.linear.check_policy}, forcing={config.nonlinear.linear_forcing}', flush=True)
        if config.interaction_quadrature.mode=='adaptive':
            q = config.interaction_quadrature
            print(f'IB quadrature: density={q.point_density:g}, max order={q.max_order}, '
                  f'max points={q.max_points}; orders above 8 use positive conical rules '
                  'when compact tables are unavailable',flush=True)
        if config.nonlinear_solver != 'jfnk':
            print(f'Anderson budget={config.anderson.max_iterations}+{config.anderson.extra_iterations}, '
                  f'stagnation window={config.anderson.stall_iterations}; '
                  f'Newton preconditioner={config.newton_preconditioner}',flush=True)
        for _ in range(state.step, steps):
            sample = (state.step+1) % config.output.log_every == 0 or state.step+1 == steps
            state, info = driver.step(state, diagnostics=sample)
            if writer and (state.step % config.output.output_every == 0 or state.step == steps):
                writer.write(state)
            if sample:
                history.append(row()); last = history[-1]
                active_text = f'T={last["active_tension_dyn_per_cm2"]/10000:.4g} kPa, ' if active_cycle else ''
                solver_text = (f'AA={last["anderson_iterations"]}, Newton={last["newton_iterations"]}, '
                               f'GMRES={last["gmres_iterations"]}, ' if config.nonlinear_solver!='jfnk' else '')
                print(f'step {state.step}/{steps}, t={state.time:.6f}, V={last["cavity_volume_ml"]:.8g} mL, '
                    f'p_endo={last["endocardial_pressure_mmhg"]:.4g} mmHg, {active_text}minJ={last["minimum_detF"]:.6g}, '
                    f'div={last["divergence_l2"]:.3g}, iterations={last["nonlinear_iterations"]}, '
                    f'{solver_text}fluid solves={last["fluid_solves"]}, '
                    f'mass solves={last["mass_solves"]}, residual={last["nonlinear_residual"]:.3g}', flush=True)
            if state.step % config.output.checkpoint_every == 0:
                save('running')
        return save('completed')
    except (Exception, KeyboardInterrupt) as exc:
        progress['failure'] = dict(type=type(exc).__name__, message=str(exc), last_accepted_step=state.step)
        if hasattr(exc, 'diagnosis'):
            progress['failure']['diagnosis'] = exc.diagnosis
        if hasattr(exc, 'result'):
            progress['failure']['nonlinear'] = dict(iterations=exc.result.iterations,
                residual_norm=exc.result.residual_norm, tolerance=exc.result.tolerance, history=exc.result.history)
        save('failed')
        raise
