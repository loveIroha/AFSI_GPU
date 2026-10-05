"""Warmed BE-BE solver/execution comparisons without per-step file writes."""
import argparse
from contextlib import contextmanager
from dataclasses import replace
from itertools import product
from pathlib import Path
from time import perf_counter
import sys
import gc
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))
from afsi_torch.paper_lv_checkpoint import load
from afsi_torch.mac.paper_coupling import BEIBStepper
from afsi_torch.cycle_checkpoint import atomic_json


@contextmanager
def profile_phases(driver, recorder):
    targets = [(driver.solid, 'validate', 'solid_validation'),
        (driver.solid, 'force', 'solid_force'),
        (driver, 'check_support', 'ib_support'),
        (driver.transfer, 'prepare', 'ib_prepare'),
        (driver.transfer, 'spread', 'ib_spread_with_mass'),
        (driver.transfer, 'interpolate', 'ib_interpolate_with_mass'),
        (driver.transfer, 'solve_mass', 'mass_solves'),
        (driver.flow, 'advect', 'advection'),
        (driver.flow, 'solve_rhs', 'fluid_total'),
        (driver.flow.pressure_solver, 'solve', 'pressure_solve'),
        (driver.flow, '_metrics', 'helmholtz_initial_metrics'),
        (driver.flow, '_residual_norms', 'helmholtz_residual_metrics')]
    targets.append((driver.flow, '_smooth', 'helmholtz_smoothing') if driver.flow.workspace is None
        else (driver.flow.workspace, 'advance', 'helmholtz_smoothing'))
    originals = [(obj, name, getattr(obj, name)) for obj, name, _ in targets]
    try:
        for (obj, name, label), (_, _, function) in zip(targets, originals):
            setattr(obj, name, recorder.wrap(function, label))
        yield
    finally:
        for obj, name, function in originals:
            setattr(obj, name, function)


@torch.no_grad()
def benchmark(checkpoint, *, device='cuda', warmup=5, steps=20, intervals=(1, 5),
              solvers=('jfnk',), shared_executions=(None,), support_backends=(None,),
              helmholtz_backends=(None,), profile=False, profile_steps=3):
    if warmup < 0 or steps < 1 or not intervals or any(n < 1 for n in intervals):
        raise ValueError('nonnegative warmup, positive steps and check intervals required')
    if (not solvers or any(s not in ('jfnk', 'newton', 'anderson-newton') for s in solvers)
            or not shared_executions or any(s not in (None, 'reference', 'vector', 'reduced') for s in shared_executions)
            or not support_backends or any(s not in (None, 'points', 'vertices') for s in support_backends)
            or not helmholtz_backends or any(s not in (None, 'reference', 'workspace', 'graph') for s in helmholtz_backends)
            or profile_steps < 1):
        raise ValueError('invalid solver/execution selections or profile steps')
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    synchronize = lambda: torch.cuda.synchronize(device) if str(device).startswith('cuda') else None
    started = perf_counter()
    model, initial, config, _ = load(checkpoint, device)
    synchronize()
    load_seconds = perf_counter()-started
    variants, reference = [], None
    for solver, interval, shared, support, helmholtz in product(solvers, intervals,
            shared_executions, support_backends, helmholtz_backends):
        # Vary check intervals only for JFNK in this comparison; the other
        # solvers use the first interval, including their GMRES fallback.
        if solver != 'jfnk' and interval != intervals[0]:
            continue
        driver = None
        state = initial
        try:
            setup = perf_counter()
            cfg = replace(config, nonlinear_solver=solver, nonlinear=replace(config.nonlinear,
                linear=replace(config.nonlinear.linear, check_every=interval)))
            if shared is not None:
                cfg = replace(cfg, interaction_quadrature=replace(cfg.interaction_quadrature, shared_execution=shared))
            if support is not None:
                cfg = replace(cfg, support_backend=support)
            if helmholtz is not None:
                cfg = replace(cfg, flow=replace(cfg.flow, helmholtz_backend=helmholtz))
            label = f'{solver}/check={interval}/ib={cfg.interaction_quadrature.shared_execution}/support={cfg.support_backend}/helmholtz={cfg.flow.helmholtz_backend}'
            driver = BEIBStepper(model, cfg, device)
            state = initial
            for _ in range(warmup):
                state, _ = driver.step(state, diagnostics=False)
            synchronize()
            setup_seconds = perf_counter()-setup
            warm_state = state
            print(f'{label}: warmup completed; measuring {steps} steps', flush=True)
            totals = {key: 0 for key in ('iterations', 'fluid_solves', 'jacobian_actions',
                'true_residual_checks', 'residual_restarts', 'scalar_reads',
                'anderson_iterations', 'newton_iterations')}
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
            result = dict(status='completed', solver=solver, check_every=interval,
                shared_execution=cfg.interaction_quadrature.shared_execution,
                support_backend=cfg.support_backend, helmholtz_backend=cfg.flow.helmholtz_backend,
                measured_steps=steps,
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
            print(f'{label}: {result["milliseconds_per_step"]:.3f} ms/step; '
                  f'fluid solves={result["per_step"]["fluid_solves"]:.2f}, '
                  f'true checks={result["per_step"]["true_residual_checks"]:.2f}', flush=True)
            if profile:
                from validation.benchmark_mac import PhaseRecorder
                recorder = PhaseRecorder(initial.x.device)
                probe = warm_state
                synchronize()
                profile_start = perf_counter()
                with profile_phases(driver, recorder):
                    for _ in range(profile_steps):
                        probe, _ = driver.step(probe, diagnostics=False)
                synchronize()
                result['profile'] = dict(steps=profile_steps,
                    milliseconds_per_step=1000*(perf_counter()-profile_start)/profile_steps,
                    phases=recorder.summary(profile_steps),
                    interpretation='nested inclusive event intervals; do not sum parent and child phases; '
                        'CUDA intervals can include CPU launch gaps, not pure kernel execution time')
                del recorder, probe
            variants.append(result)
            # Release this driver's workspaces before constructing the next one.
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            variants.append(dict(status='failed', solver=solver, check_every=interval,
                shared_execution=shared, support_backend=support, helmholtz_backend=helmholtz,
                last_accepted_time_s=state.time, failure_type=type(exc).__name__, failure=str(exc)))
            print(f'{solver}/check={interval}: failed: {exc}', flush=True)
        finally:
            del driver
            gc.collect()
    report = dict(checkpoint=str(checkpoint), device=str(device), dtype=str(initial.x.dtype),
        scheme='BE-BE', nonlinear_solver=solvers[0] if len(solvers)==1 else 'comparison',
        nonlinear_solvers=list(solvers), warmup_steps=warmup,
        checkpoint_load_seconds=load_seconds, variants=variants,
        timing='wall time with CUDA synchronization only at measurement boundaries; '
               'no VTK, checkpoint, history or per-step log writes during measurement',
        scalar_reads_scope='BiCGSTAB scalar transfers only; excludes nested fluid/mass/validation checks')
    if len(variants) == 2 and all(v['status']=='completed' for v in variants):
        report['measured_speedup_first_over_second'] = variants[0]['milliseconds_per_step']/variants[1]['milliseconds_per_step']
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--linear-check-intervals', type=int, nargs='+', default=[1, 5])
    p.add_argument('--solvers', nargs='+', choices=('jfnk', 'newton', 'anderson-newton'), default=['jfnk'])
    p.add_argument('--ib-shared-executions', nargs='+', choices=('reference', 'vector', 'reduced'))
    p.add_argument('--support-backends', nargs='+', choices=('points', 'vertices'))
    p.add_argument('--helmholtz-backends', nargs='+', choices=('reference', 'workspace', 'graph'))
    p.add_argument('--profile', action='store_true')
    p.add_argument('--profile-steps', type=int, default=3)
    p.add_argument('--output', default='results/paper_lv_performance/report.json')
    args = p.parse_args()
    report = benchmark(args.checkpoint, device=args.device, warmup=args.warmup,
        steps=args.steps, intervals=tuple(args.linear_check_intervals), solvers=tuple(args.solvers),
        shared_executions=tuple(args.ib_shared_executions or [None]),
        support_backends=tuple(args.support_backends or [None]),
        helmholtz_backends=tuple(args.helmholtz_backends or [None]),
        profile=args.profile, profile_steps=args.profile_steps)
    atomic_json(args.output, report)
    print(f'report={args.output}', flush=True)
    if any(v['status']=='failed' for v in report['variants']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
