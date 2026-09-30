"""Compare reference and fused pressure MG with identical warm-started IB replays."""
import argparse
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
from time import perf_counter
import torch
from benchmark_mac import PhaseRecorder, record_phases, replay, compare_states, synchronize
from afsi_torch.cycle_checkpoint import atomic_json
from afsi_torch.mac import MACGrid, MACFlow, divergence
from afsi_torch.mac.checkpoint import load_mac
from afsi_torch.mac.coupling import MACIBStepper
from afsi_torch.mac.transfer import FETransfer
from afsi_torch.solid import prepare_p2


def comparison(a, b):
    differences = {name+'_max_abs': (getattr(a,name)-getattr(b,name)).abs().max().item()
                   for name in ('x', 'force', 'pressure')}
    differences.update({f'velocity_{c}_max_abs': (u-v).abs().max().item()
                        for c,(u,v) in enumerate(zip(a.velocity,b.velocity))})
    try:
        compare_states(a, b)
        return dict(passed=True, max_abs=differences)
    except AssertionError as exc:
        return dict(passed=False, max_abs=differences, error=str(exc))


@torch.no_grad()
def benchmark(checkpoint, *, device='cuda', steps=20, warmup=3):
    if steps < 1 or warmup < 1:
        raise ValueError('steps and warmup must be positive')
    device = torch.device(device)
    model, initial, settings, _ = load_mac(checkpoint, device)
    from afsi_torch.config import lv_grid
    grid = lv_grid(settings)
    degree = settings['interaction_degree']
    geometry = (model.geometry if degree is None else
                prepare_p2(model.mesh.X, model.mesh.cells, degree=degree))
    variants, states = {}, {}
    for backend in ('torch', 'fused'):
        print(f'{backend}: setup and warmup (first CUDA use compiles kernels)...', flush=True)
        synchronize(device)
        started = perf_counter()
        from afsi_torch.mac.multigrid import MGOptions
        from afsi_torch.fluid.solvers import SolverOptions
        flow = MACFlow(grid, dt=settings['dt'], rho=settings['rho'], mu=settings['mu'],
                       device=device, pressure_backend=backend,options=MGOptions(**settings.get('pressure_solver',{})))
        transfer = FETransfer(grid, geometry, warm_start=True,
                              options=SolverOptions(**settings['mass_solver']) if 'mass_solver' in settings else None)
        driver = MACIBStepper(flow, transfer, model.force, model.validate)
        replay(driver, initial, warmup, device)
        setup_warmup = perf_counter()-started
        transfer.reset_warm_start()
        end, plain = replay(driver, initial, steps, device)
        states[backend] = end
        transfer.reset_warm_start()
        recorder = PhaseRecorder(device)
        with record_phases(driver, recorder):
            profiled, measured = replay(driver, initial, steps, device)
        phases = recorder.summary(steps)
        workspace = flow.pressure_solver.workspace
        variants[backend] = dict(**plain, actual_backend=flow.pressure_solver.backend,
            setup_and_warmup_seconds=setup_warmup,
            ms_per_step=1000*plain['wall_seconds']/steps,
            instrumented_wall_seconds=measured['wall_seconds'], phase_timings=phases,
            workspace_bytes=workspace.allocated_bytes if workspace is not None else 0,
            repeat_equivalence=comparison(end, profiled),
            final_solid=model.diagnostics(end.x),
            final_divergence_l2=(grid.volume*divergence(end.velocity,grid.spacing).square().sum()).sqrt().item())
        print(f"{backend}: {variants[backend]['ms_per_step']:.3f} ms/step; "
              f"pressure={phases['pressure_solve']['ms_per_step']:.3f} ms/step; "
              f"cycles={plain['pressure_cycles_mean']:.1f}", flush=True)
        del driver, transfer, flow, workspace, profiled
    equivalent = comparison(states['torch'], states['fused'])
    try:
        triton_version = version('triton')
    except PackageNotFoundError:
        triton_version = None
    return dict(schema=1, benchmark='mac-pressure-backend-replay',
        checkpoint=str(Path(checkpoint).resolve()), device=str(device),
        torch=torch.__version__, triton=triton_version,
        gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        dtype=str(initial.x.dtype), cpu_threads=torch.get_num_threads(),
        steps=steps, warmup_steps=warmup, start_step=initial.step,
        start_time_s=initial.time, end_time_s=states['fused'].time,
        settings=settings, ib_warm_start=True, variants=variants, equivalence=equivalent,
        passed=equivalent['passed'] and all(v['repeat_equivalence']['passed'] for v in variants.values()),
        whole_step_speedup=variants['torch']['wall_seconds']/variants['fused']['wall_seconds'],
        pressure_speedup=variants['torch']['phase_timings']['pressure_solve']['ms_per_step']/
                         variants['fused']['phase_timings']['pressure_solve']['ms_per_step'],
        notes=['Identical checkpoints, separable IB and warm-started mass solves for both backends.',
               'Setup/compilation, final diagnostics and file I/O excluded from throughput.',
               'Phase events use a separate replay, without added per-phase synchronization.',
               'pressure_solve is included in fluid_total; do not double count.',
               'Original PyTorch true-residual checks and tolerances are retained.',
               'CPU fused is buffered reference emulation, not a GPU performance result.',
               'A 2 s checkpoint measures held-load tail behavior, not the loading trajectory.',
               'Input checkpoint is never modified. Short replay speedup is not full-run speedup.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--output', default='results/mac_pressure_fused/report.json')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'choose a new benchmark output: {output}')
    report = benchmark(args.checkpoint, device=args.device, steps=args.steps, warmup=args.warmup)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    print(json.dumps({key: report[key] for key in
                     ('passed', 'whole_step_speedup', 'pressure_speedup', 'equivalence')}, indent=2))
    print(f'Report: {output}', flush=True)
    if not report['passed']:
        raise SystemExit('Pressure backend comparison failed; inspect the saved report.')


if __name__ == '__main__':
    main()
