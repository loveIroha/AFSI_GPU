"""Warmed BE-BE solver/execution comparisons without per-step file writes."""
import argparse
from contextlib import contextmanager
from dataclasses import replace
from itertools import product
from pathlib import Path
from time import perf_counter
import sys
import gc
import traceback
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))
from afsi_torch.paper_lv_checkpoint import load
from afsi_torch.mac.paper_coupling import BEIBStepper, BEProblem
from afsi_torch.mac.midpoint_solver import anderson_policy
from afsi_torch.nonlinear import coupled_linear_policy
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
    targets += [(BEProblem,'linearization','tangent_assembly'),
                (BEProblem,'preconditioner','preconditioner_build')]
    if hasattr(driver.transfer,'assemble_stencil'):
        targets.append((driver.transfer,'assemble_stencil','ib_csr_assembly'))
        if driver.transfer.assembly_backend=='hash':
            from afsi_torch.mac import hash_transfer_assembly as builder
            targets += [(builder,'accumulate_plans','ib_hash_accumulate'),
                        (builder,'finish_component','ib_hash_finalize')]
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
              helmholtz_backends=(None,), anderson_policies=(None,),
              newton_preconditioners=(None,), linear_policies=(None,),
              ib_response_backends=(None,),csr_assembly_backends=(None,),
              anderson_budgets=(None,),profile=False, profile_steps=3):
    if warmup < 0 or steps < 1 or not intervals or any(n < 1 for n in intervals):
        raise ValueError('nonnegative warmup, positive steps and check intervals required')
    if (not solvers or any(s not in ('jfnk', 'newton', 'anderson-newton') for s in solvers)
            or not shared_executions or any(s not in (None, 'reference', 'vector', 'reduced') for s in shared_executions)
            or not support_backends or any(s not in (None, 'points', 'vertices') for s in support_backends)
            or not helmholtz_backends or any(s not in (None, 'reference', 'workspace', 'graph') for s in helmholtz_backends)
            or not anderson_policies or any(s not in (None,'legacy','adaptive') for s in anderson_policies)
            or not newton_preconditioners or any(s not in (None,'none','solid-block') for s in newton_preconditioners)
            or not linear_policies or any(s not in (None,'reference','estimated','inexact') for s in linear_policies)
            or not ib_response_backends or any(s not in (None,'quadrature','csr') for s in ib_response_backends)
            or not csr_assembly_backends or any(s not in (None,'coalesce','hash') for s in csr_assembly_backends)
            or not anderson_budgets or any(n is not None and (type(n) is not int or n<1) for n in anderson_budgets)
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
    for solver, interval, shared, support, helmholtz, policy, preconditioner, linear_policy, ib_backend, csr_builder, budget in product(solvers, intervals,
            shared_executions, support_backends, helmholtz_backends,anderson_policies,newton_preconditioners,
            linear_policies,ib_response_backends,csr_assembly_backends,anderson_budgets):
        # Vary check intervals only for JFNK in this comparison; the other
        # solvers use the first interval, including their GMRES fallback.
        if solver != 'jfnk' and interval != intervals[0]:
            continue
        if solver != 'anderson-newton' and policy != anderson_policies[0]:
            continue
        if solver != 'anderson-newton' and budget != anderson_budgets[0]:
            continue
        if solver == 'jfnk' and preconditioner != newton_preconditioners[0]:
            continue
        if solver == 'jfnk' and linear_policy != linear_policies[0]:
            continue
        if (ib_backend or config.ib_response_backend)!='csr' and csr_builder!=csr_assembly_backends[0]:
            continue
        driver = None
        cfg = None
        state = initial
        phase = 'setup'
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
            if policy is not None:
                cfg = replace(cfg,anderson=anderson_policy(cfg.anderson,policy))
            if solver=='anderson-newton' and budget is not None:
                cfg = replace(cfg,anderson=replace(cfg.anderson,max_iterations=budget))
            if preconditioner is not None:
                cfg = replace(cfg,newton_preconditioner=preconditioner)
            if linear_policy is not None:
                cfg = replace(cfg,nonlinear=coupled_linear_policy(cfg.nonlinear,linear_policy))
            if ib_backend is not None:
                cfg = replace(cfg,ib_response_backend=ib_backend)
            if csr_builder is not None:
                cfg = replace(cfg,ib_csr_assembly_backend=csr_builder)
            label = (f'{solver}/aa={policy or "saved"}/pc={cfg.newton_preconditioner}/check={interval}'
                     f'/aa-budget={cfg.anderson.max_iterations}+{cfg.anderson.extra_iterations}'
                     f'/ib={cfg.interaction_quadrature.shared_execution}/support={cfg.support_backend}'
                     f'/helmholtz={cfg.flow.helmholtz_backend}/linear={linear_policy or "saved"}'
                     f'/response={cfg.ib_response_backend}/csr={cfg.ib_csr_assembly_backend}')
            driver = BEIBStepper(model, cfg, device)
            state = initial
            phase = 'warmup'
            for _ in range(warmup):
                state, _ = driver.step(state, diagnostics=False)
            synchronize()
            setup_seconds = perf_counter()-setup
            warm_state = state
            print(f'{label}: warmup completed; measuring {steps} steps', flush=True)
            totals = {key: 0 for key in ('iterations', 'fluid_solves', 'jacobian_actions',
                'true_residual_checks', 'estimated_residual_checks','residual_restarts', 'scalar_reads',
                'anderson_iterations', 'newton_iterations','anderson_extra_iterations',
                'seeded_residual_reuses','residual_evaluations','gmres_iterations',
                'tangent_assemblies','preconditioner_applications','mass_solves','mass_iterations',
                'pressure_solves','pressure_cycles','helmholtz_sweeps')}
            fallback_count = 0
            sampled_histories = []
            worst_ratio = 0.
            synchronize()
            if str(device).startswith('cuda'):
                torch.cuda.reset_peak_memory_stats(device)
            start = perf_counter()
            phase = 'measurement'
            for _ in range(steps):
                state, info = driver.step(state, diagnostics=False)
                nonlinear = info['nonlinear']
                fallback_count += int(nonlinear.get('newton_fallback',False))
                if len(sampled_histories)<3:
                    sampled_histories.append(dict(step=state.step,time_s=state.time,
                        nonlinear=nonlinear))
                for key in totals:
                    totals[key] += nonlinear.get(key, 0)
                worst_ratio = max(worst_ratio, nonlinear['residual_norm']/nonlinear['tolerance'])
            synchronize()
            elapsed = perf_counter()-start
            result = dict(status='completed', solver=solver, check_every=interval,
                anderson_policy=policy or 'saved',newton_preconditioner=cfg.newton_preconditioner,
                linear_policy=linear_policy or 'saved',ib_response_backend=cfg.ib_response_backend,
                ib_csr_assembly_backend=cfg.ib_csr_assembly_backend,
                linear_options=dict(check_policy=cfg.nonlinear.linear.check_policy,
                    true_check_interval=cfg.nonlinear.linear.true_check_interval,
                    forcing=cfg.nonlinear.linear_forcing,
                    tolerance_fraction=cfg.nonlinear.linear_tolerance_fraction),
                anderson_budget=cfg.anderson.max_iterations,anderson_extra_budget=cfg.anderson.extra_iterations,
                anderson_stall_iterations=cfg.anderson.stall_iterations,
                shared_execution=cfg.interaction_quadrature.shared_execution,
                support_backend=cfg.support_backend, helmholtz_backend=cfg.flow.helmholtz_backend,
                measured_steps=steps,
                setup_and_warmup_seconds=setup_seconds, elapsed_seconds=elapsed,
                milliseconds_per_step=1000*elapsed/steps,
                per_step={key: value/steps for key, value in totals.items()},
                newton_fallback_fraction=fallback_count/steps,sampled_histories=sampled_histories,
                max_accepted_residual_to_tolerance=worst_ratio,
                start_time_s=initial.time+warmup*cfg.time.dt, end_time_s=state.time)
            if str(device).startswith('cuda'):
                result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
                result['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
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
                  f'AA={result["per_step"]["anderson_iterations"]:.2f}, '
                  f'Newton={result["per_step"]["newton_iterations"]:.2f}, '
                  f'GMRES={result["per_step"]["gmres_iterations"]:.2f}, '
                  f'mass solves={result["per_step"]["mass_solves"]:.2f}', flush=True)
            if profile:
                phase = 'profile'
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
                anderson_policy=policy,newton_preconditioner=preconditioner,
                anderson_budget=cfg.anderson.max_iterations if cfg is not None else budget,
                linear_policy=linear_policy,ib_response_backend=ib_backend,
                ib_csr_assembly_backend=csr_builder,
                shared_execution=shared, support_backend=support, helmholtz_backend=helmholtz,
                last_accepted_time_s=state.time,failure_phase=phase,
                failure_type=type(exc).__name__,failure=str(exc),traceback=traceback.format_exc()))
            print(f'{solver}/check={interval}/aa-budget={budget or config.anderson.max_iterations}'
                  f'/csr={csr_builder or config.ib_csr_assembly_backend}: '
                  f'failed during {phase}: {exc}',flush=True)
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
    successful = [v for v in variants if v['status']=='completed']
    if successful:
        baseline = successful[0]
        report['comparison_baseline'] = dict(solver=baseline['solver'],
            anderson_budget=baseline['anderson_budget'],
            milliseconds_per_step=baseline['milliseconds_per_step'])
        for variant in successful:
            variant['speedup_vs_first_completed'] = baseline['milliseconds_per_step']/variant['milliseconds_per_step']
        fastest = min(successful,key=lambda v:v['milliseconds_per_step'])
        report['fastest_variant'] = {k:fastest[k] for k in ('solver','anderson_policy',
            'anderson_budget',
            'newton_preconditioner','milliseconds_per_step','shared_execution','support_backend','helmholtz_backend',
            'linear_policy','ib_response_backend','ib_csr_assembly_backend')}
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
    p.add_argument('--anderson-policies', nargs='+', choices=('legacy','adaptive'))
    p.add_argument('--anderson-budgets', nargs='+',type=int,
                   help='trial budgets before Newton; compared only for anderson-newton')
    p.add_argument('--newton-preconditioners', nargs='+', choices=('none','solid-block'))
    p.add_argument('--linear-policies', nargs='+', choices=('reference','estimated','inexact'))
    p.add_argument('--ib-response-backends', nargs='+', choices=('quadrature','csr'))
    p.add_argument('--csr-assembly-backends', nargs='+', choices=('coalesce','hash'))
    p.add_argument('--profile', action='store_true')
    p.add_argument('--profile-steps', type=int, default=3)
    p.add_argument('--output', default='results/paper_lv_performance/report.json')
    args = p.parse_args()
    report = benchmark(args.checkpoint, device=args.device, warmup=args.warmup,
        steps=args.steps, intervals=tuple(args.linear_check_intervals), solvers=tuple(args.solvers),
        shared_executions=tuple(args.ib_shared_executions or [None]),
        support_backends=tuple(args.support_backends or [None]),
        helmholtz_backends=tuple(args.helmholtz_backends or [None]),
        anderson_policies=tuple(args.anderson_policies or [None]),
        newton_preconditioners=tuple(args.newton_preconditioners or [None]),
        linear_policies=tuple(args.linear_policies or [None]),
        ib_response_backends=tuple(args.ib_response_backends or [None]),
        csr_assembly_backends=tuple(args.csr_assembly_backends or [None]),
        anderson_budgets=tuple(args.anderson_budgets) if args.anderson_budgets is not None else (None,),
        profile=args.profile, profile_steps=args.profile_steps)
    atomic_json(args.output, report)
    print(f'report={args.output}', flush=True)
    if any(v['status']=='failed' for v in report['variants']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
