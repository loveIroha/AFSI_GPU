"""Warmed BE-BE JFNK timings without visualization or checkpoint writes."""
import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))
from afsi_torch.paper_lv_checkpoint import load
from afsi_torch.mac.paper_coupling import BEIBStepper
from afsi_torch.cycle_checkpoint import atomic_json


@torch.no_grad()
def benchmark(checkpoint, *, device='cuda', warmup=5, steps=20, intervals=(1, 5)):
    if warmup < 0 or steps < 1 or not intervals or any(n < 1 for n in intervals):
        raise ValueError('nonnegative warmup, positive steps and check intervals required')
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    synchronize = lambda: torch.cuda.synchronize(device) if str(device).startswith('cuda') else None
    started = perf_counter()
    model, initial, config, _ = load(checkpoint, device)
    synchronize()
    load_seconds = perf_counter()-started
    variants, reference = [], None
    for interval in intervals:
        setup = perf_counter()
        cfg = replace(config, nonlinear_solver='jfnk', nonlinear=replace(config.nonlinear,
            linear=replace(config.nonlinear.linear, check_every=interval)))
        driver = BEIBStepper(model, cfg, device)
        state = initial
        for _ in range(warmup):
            state, _ = driver.step(state, diagnostics=False)
        synchronize()
        setup_seconds = perf_counter()-setup
        print(f'check interval={interval}: warmup completed; measuring {steps} steps', flush=True)
        totals = {key: 0 for key in ('iterations', 'fluid_solves', 'jacobian_actions',
            'true_residual_checks', 'residual_restarts', 'scalar_reads')}
        worst_ratio = 0.
        synchronize()
        start = perf_counter()
        for _ in range(steps):
            state, info = driver.step(state, diagnostics=False)
            nonlinear = info['nonlinear']
            for key in totals:
                totals[key] += nonlinear.get(key, 0)
            worst_ratio = max(worst_ratio, nonlinear['residual_norm']/nonlinear['tolerance'])
        synchronize()
        elapsed = perf_counter()-start
        result = dict(check_every=interval, measured_steps=steps,
            setup_and_warmup_seconds=setup_seconds, elapsed_seconds=elapsed,
            milliseconds_per_step=1000*elapsed/steps,
            per_step={key: value/steps for key, value in totals.items()},
            max_accepted_residual_to_tolerance=worst_ratio,
            start_time_s=initial.time+warmup*cfg.time.dt, end_time_s=state.time)
        if reference is None:
            reference = (state.x.clone(), state.pressure.clone(), tuple(v.clone() for v in state.velocity))
        else:
            result['end_state_max_abs_vs_first'] = dict(
                x_cm=(state.x-reference[0]).abs().max().item(),
                pressure_dyn_per_cm2=(state.pressure-reference[1]).abs().max().item(),
                velocity_cm_per_s=max((a-b).abs().max().item() for a, b in zip(state.velocity, reference[2])))
        result['interaction_quadrature'] = (driver.transfer.quadrature_summary()
            if hasattr(driver.transfer, 'quadrature_summary') else dict(mode='fixed'))
        variants.append(result)
        print(f'check interval={interval}: {result["milliseconds_per_step"]:.3f} ms/step; '
              f'fluid solves={result["per_step"]["fluid_solves"]:.2f}, '
              f'true checks={result["per_step"]["true_residual_checks"]:.2f}', flush=True)
        # Release this driver's workspaces before constructing the next one.
        del driver
    report = dict(checkpoint=str(checkpoint), device=str(device), dtype=str(initial.x.dtype),
        scheme='BE-BE', nonlinear_solver='jfnk', warmup_steps=warmup,
        checkpoint_load_seconds=load_seconds, variants=variants,
        timing='wall time with CUDA synchronization only at measurement boundaries; '
               'no VTK, checkpoint, history or per-step log writes during measurement',
        scalar_reads_scope='BiCGSTAB scalar transfers only; excludes nested fluid/mass/validation checks')
    if len(variants) == 2:
        report['measured_speedup_first_over_second'] = variants[0]['milliseconds_per_step']/variants[1]['milliseconds_per_step']
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--linear-check-intervals', type=int, nargs='+', default=[1, 5])
    p.add_argument('--output', default='results/paper_lv_performance/report.json')
    args = p.parse_args()
    report = benchmark(args.checkpoint, device=args.device, warmup=args.warmup,
        steps=args.steps, intervals=tuple(args.linear_check_intervals))
    atomic_json(args.output, report)
    print(f'report={args.output}', flush=True)


if __name__ == '__main__':
    main()
