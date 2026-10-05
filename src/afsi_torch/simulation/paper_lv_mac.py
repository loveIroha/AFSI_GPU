"""Real LV passive-inflation reproduction, Ma et al. 2024 section V.F."""
import csv
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
import torch
from ..paper_lv import PaperLVConfig, imported_model
from ..paper_lv_checkpoint import save as save_checkpoint, load as load_checkpoint
from ..config import TimeConfig
from ..cycle_checkpoint import atomic_json
from ..mac.paper_coupling import BEIBStepper
from ..mac.memory import allocator_sample


@torch.no_grad()
def run(*, case_config=None, device='cuda', output=None, resume=None, end_time=None):
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
    driver = BEIBStepper(model, config, device)
    if not resume:
        state = driver.initialize(model.mesh.X)
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

    def row():
        nonlinear = info.get('nonlinear', {})
        return dict(step=state.step, time_s=state.time,
            endocardial_pressure_mmhg=model.loads.at(state.time)[0]/1333.22387415,
            endocardial_pressure_dyn_per_cm2=model.loads.at(state.time)[0],
            active_tension_dyn_per_cm2=0., **model.diagnostics(state.x),
            divergence_l2=info.get('divergence_l2', 0.), power_error=info.get('power_error', 0.),
            nonlinear_iterations=nonlinear.get('iterations', 0), nonlinear_residual=nonlinear.get('residual_norm', 0.),
            fluid_solves=nonlinear.get('fluid_solves', 0), courant=info.get('courant', 0.),
            pressure_cycles=info.get('flow', {}).get('pressure', {}).get('cycles', 0))

    def save(status):
        progress['elapsed_seconds'] = previous_elapsed+perf_counter()-started
        if not history or float(history[-1]['time_s']) != state.time:
            history.append(row())
        with (folder/'history.csv').open('w', newline='') as stream:
            w = csv.DictWriter(stream, fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
        report = dict(status=status, accepted_steps=state.step, reached_time_s=state.time,
            requested_end_time_s=config.time.end_time, dt_s=config.time.dt, elapsed_seconds=progress['elapsed_seconds'],
            reference='Ma et al., Physics of Fluids 36, 081914 (2024), section V.F', doi='10.1063/5.0225605',
            units='cm-g-s', solid_nodes=len(state.x), solid_cells=len(model.mesh.cells), fluid_shape=driver.grid.shape,
            config=asdict(config), material='eq81/82 RAW I1 H-O with normal-stress removal and log(I3) penalty',
            time_scheme='BE-BE; endpoint FE force; old-geometry dual IB; BE diffusion/Chorin projection',
            convection='first-order characteristic tracing and trilinear semi-Lagrangian MAC sampling' if config.flow.convection else 'disabled',
            fluid_boundaries='homogeneous Neumann velocity diffusion; homogeneous Dirichlet pressure on physical box faces',
            reproduction_differences=['user radial basal spring retained; paper V.F does not specify the basal constraint',
                'fixed dt rather than the adaptive time sequence shown in Fig26',
                'characteristic tracing/interpolation reconstructed; authors implementation unavailable',
                'interaction quadrature density/rule and solver tolerances are explicit implementation choices']
                + ([] if config.nonlinear_solver == 'jfnk' else ['Anderson/assembled-CSR Newton instead of paper JFNK/BiCGSTAB; same BE-BE residual']),
            energy_claim='no unconditional energy theorem asserted for this loaded open-boundary corrected H-O experiment',
            interaction_quadrature=driver.transfer.quadrature_summary() if hasattr(driver.transfer, 'quadrature_summary') else dict(mode='fixed'),
            nonlinear_solver=config.nonlinear_solver, pressure_backend=driver.flow.pressure_solver.backend,
            gpu_memory=allocator_sample(state.x.device), last=history[-1], last_solver_info=info,
            visualization=writer.summary() if writer else dict(enabled=False),
            full_horizon_validated=status == 'completed' and state.time >= 1.5,
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
        print(f'Passive inflation: 0->{config.loads.target_mmhg:g} mmHg over {config.loads.ramp_seconds:g} s, then held; mu={config.fluid.mu:g}; '
              f'solver={config.nonlinear_solver}; open pressure boundary; radial base retained', flush=True)
        for _ in range(state.step, steps):
            sample = (state.step+1) % config.output.log_every == 0 or state.step+1 == steps
            state, info = driver.step(state, diagnostics=sample)
            if writer and (state.step % config.output.output_every == 0 or state.step == steps):
                writer.write(state)
            if sample:
                history.append(row()); last = history[-1]
                print(f'step {state.step}/{steps}, t={state.time:.6f}, V={last["cavity_volume_ml"]:.8g} mL, '
                    f'p_endo={last["endocardial_pressure_mmhg"]:.4g} mmHg, minJ={last["minimum_detF"]:.6g}, '
                    f'div={last["divergence_l2"]:.3g}, iterations={last["nonlinear_iterations"]}, '
                    f'fluid solves={last["fluid_solves"]}, residual={last["nonlinear_residual"]:.3g}', flush=True)
            if state.step % config.output.checkpoint_every == 0:
                save('running')
        return save('completed')
    except (Exception, KeyboardInterrupt) as exc:
        progress['failure'] = dict(type=type(exc).__name__, message=str(exc), last_accepted_step=state.step)
        if hasattr(exc, 'result'):
            progress['failure']['nonlinear'] = dict(iterations=exc.result.iterations,
                residual_norm=exc.result.residual_norm, tolerance=exc.result.tolerance, history=exc.result.history)
        save('failed')
        raise
