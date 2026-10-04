"""Compare warmed coupled step costs from one real-LV checkpoint, without I/O per step."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
import sys
import gc
import torch
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Direct execution adds validation/, not the repository root, to sys.path.
# Make both local package code and the optional profiling helper importable.
sys.path.insert(0,str(PROJECT_ROOT))
sys.path.insert(0,str(PROJECT_ROOT/'src'))
from afsi_torch.real_lv_checkpoint import load_real_lv
from afsi_torch.mac.execution import build_driver
from afsi_torch.cycle_checkpoint import atomic_json


@contextmanager
def record_phases(driver,recorder):
    targets = [(driver.transfer,'prepare','ib_prepare'),
        (driver.transfer,'spread','ib_spread_with_mass'),
        (driver.transfer,'interpolate','ib_interpolate_with_mass'),
        (driver.transfer,'solve_mass','mass_solves'),
        (driver.flow,'stokes','stokes'),(driver.flow.pressure_solver,'solve','pressure_solves'),
        (driver,'_force_geometry','solid_force'),(driver,'validate','solid_validation'),
        (driver.flow,'check_transport','transport_checks')]
    if hasattr(driver.transfer,'_direct_prepare'):
        targets += [(driver.transfer,'_rule','ib_rule_selection'),
            (driver.transfer,'_points','ib_point_coordinates'),
            (driver.transfer,'from_points','ib_table_generation'),
            (driver.transfer,'_direct_prepare','ib_direct_prepare')]
    originals = [(obj,name,getattr(obj,name)) for obj,name,_ in targets]
    try:
        for (obj,name,label),(_,_,fn) in zip(targets,originals):
            setattr(obj,name,recorder.wrap(fn,label))
        yield
    finally:
        for obj,name,fn in originals:
            setattr(obj,name,fn)


@torch.no_grad()
def benchmark(checkpoint,*,device='cuda',schemes=('explicit-rk3','implicit-newton'),warmup=2,steps=10,
              nonlinear_solvers=None,initial_state='checkpoint',execution_variants=None,profile=False):
    if type(warmup) is not int or warmup<0 or type(steps) is not int or steps<1:
        raise ValueError('warmup must be nonnegative and steps positive')
    if initial_state not in ('checkpoint','reference'):
        raise ValueError('initial_state must be checkpoint or reference')
    if not schemes or len(set(schemes))!=len(schemes) or any(s not in
            ('explicit-lagged','explicit-rk3','implicit-newton','cnab-midpoint','cnab-semiimplicit') for s in schemes):
        raise ValueError('choose distinct supported schemes')
    if nonlinear_solvers is not None and (not nonlinear_solvers or len(set(nonlinear_solvers))!=len(nonlinear_solvers)
            or any(s not in ('newton','anderson-newton') for s in nonlinear_solvers) or tuple(schemes)!=('cnab-semiimplicit',)):
        raise ValueError('nonlinear_solvers requires only cnab-semiimplicit and distinct supported solvers')
    model,initial,settings,_,config = load_real_lv(checkpoint,device)
    if execution_variants is not None:
        if (not execution_variants or len(set(execution_variants))!=len(execution_variants)
                or any(v not in ('baseline','reuse','compact','fused','cell','shared','pressure-warm','shared-warm','prepare-warm','shared-fused-warm','shared-fused-checked') for v in execution_variants)
                or tuple(schemes)!=('cnab-semiimplicit',) or config.interaction_quadrature.mode!='adaptive'):
            raise ValueError('execution_variants requires adaptive cnab-semiimplicit and distinct supported variants')
    if profile and tuple(schemes)!=('cnab-semiimplicit',):
        raise ValueError('phase profiling currently requires only cnab-semiimplicit')
    synchronize = lambda:torch.cuda.synchronize(device) if torch.device(device).type=='cuda' else None
    report = dict(checkpoint=str(checkpoint),device=str(device),torch=torch.__version__,
        gpu=torch.cuda.get_device_name(device) if torch.device(device).type=='cuda' else None,dt_s=config.time.dt,
        initial_state=initial_state,source_checkpoint_step=initial.step,source_checkpoint_time_s=initial.time,
        start_time_s=0. if initial_state=='reference' else initial.time,warmup_steps=warmup,measured_steps=steps,
        timing_scope='warmed wall time; excludes compilation, output and optional subsequent profile replay',
        note='Performance comparison, not full-cycle stability validation. Reference mode starts at t=0 with undeformed solid and zero flow; checkpoint mode retains the saved physical state. Different schemes need not be numerically equivalent.',cases={})
    cases = [(s,s,None) for s in schemes] if nonlinear_solvers is None else [
        ('cnab-semiimplicit',f'cnab-semiimplicit/{s}',s) for s in nonlinear_solvers]
    cases = [(scheme,label,solver,None) for scheme,label,solver in cases] if execution_variants is None else [
        (scheme,f'{label}/{variant}',solver,variant) for scheme,label,solver in cases for variant in execution_variants]
    final_states = {}
    for scheme,label,solver,variant in cases:
        print(f'{label}: preparing and warming up...',flush=True)
        started = perf_counter()
        coupling = replace(config.coupling,scheme=scheme)
        if solver is not None:
            coupling = replace(coupling,semiimplicit_solver=solver)
        quadrature = config.interaction_quadrature
        if variant is not None:
            coupling = replace(coupling,reuse_final_evaluation=variant!='baseline',
                stokes_warm_start=variant in ('pressure-warm','shared-warm','prepare-warm','shared-fused-warm','shared-fused-checked'),
                reuse_validation=variant=='shared-fused-checked')
            quadrature = replace(quadrature,
                rule_family='conical' if variant in ('baseline','reuse') else 'xiao-gimbutas',
                transfer_backend='fused' if variant in ('shared-fused-warm','shared-fused-checked') else variant if variant in ('fused','cell') else 'reference',
                stencil_backend='shared' if variant in ('shared','shared-warm','prepare-warm','shared-fused-warm','shared-fused-checked') else 'component',
                prepare_backend='triton' if variant in ('prepare-warm','shared-fused-warm','shared-fused-checked') else 'torch')
        driver = build_driver(model,dict(settings,coupling=asdict(coupling),interaction_quadrature=asdict(quadrature)),device)
        if initial_state=='reference':
            state = driver.initialize(model.mesh.X)
        else:
            force_time = None if initial.step==0 else initial.time if scheme in ('implicit-newton','cnab-midpoint','cnab-semiimplicit') else (initial.step-1)*config.time.dt
            state = replace(initial,force_time=force_time,
                            previous_advection=initial.previous_advection if scheme==config.coupling.scheme else None,
                            force=torch.zeros_like(initial.x) if force_time is None else model.force(initial.x,force_time))
        counters = dict(pressure_solves=0,pressure_cycles=0,mass_solves=0,mass_iterations=0,stokes_solves=0)
        pressure_solve,mass_solve = driver.flow.pressure_solver.solve,driver.transfer.solve_mass
        stokes_solve = getattr(driver.flow,'stokes',None)
        def stokes(*args,**kwargs):
            counters['stokes_solves']+=1
            return stokes_solve(*args,**kwargs)
        def pressure(*args,**kwargs):
            value,info = pressure_solve(*args,**kwargs)
            counters['pressure_solves']+=1; counters['pressure_cycles']+=info['cycles']
            return value,info
        def mass(*args,**kwargs):
            value,info = mass_solve(*args,**kwargs)
            counters['mass_solves']+=1; counters['mass_iterations']+=info.iterations
            return value,info
        driver.flow.pressure_solver.solve,driver.transfer.solve_mass = pressure,mass
        if stokes_solve is not None:
            driver.flow.stokes = stokes
        count,newton_iterations,gmres_iterations,stokes_iterations,tangent_assemblies,jacobian_actions,anderson_iterations,fallback_steps = 0,0,0,0,0,0,0,0
        measured_start = None
        final_reuses = 0
        pressure_warm_starts = pressure_warm_fallbacks = 0
        validation_evaluations = validation_reuses = 0
        try:
            for _ in range(warmup):
                state,_ = driver.step(state,diagnostics=False)
            synchronize()
            setup_seconds = perf_counter()-started
            for key in counters:
                counters[key]=0
            measured_start = perf_counter()
            for _ in range(steps):
                state,info = driver.step(state,diagnostics=False)
                count+=1
                nonlinear = info.get('nonlinear',{})
                final_reuses += int(nonlinear.get('final_evaluation_reused',False))
                pressure_warm_starts += nonlinear.get('stokes_pressure_warm_starts',0)
                pressure_warm_fallbacks += nonlinear.get('stokes_pressure_warm_fallbacks',0)
                validation_evaluations += nonlinear.get('validation_evaluations',0)
                validation_reuses += nonlinear.get('validation_reuses',0)
                newton_iterations+=nonlinear.get('newton_iterations',nonlinear.get('iterations',0))
                anderson_iterations+=nonlinear.get('anderson_iterations',0)
                fallback_steps+=int(nonlinear.get('newton_fallback',False))
                gmres_iterations+=sum(h.get('linear',{}).get('iterations',0) for h in nonlinear.get('history',[]))
                tangent_assemblies+=nonlinear.get('tangent_assemblies',0)
                jacobian_actions+=nonlinear.get('jacobian_actions',0)
                stokes_iterations+=info.get('flow',{}).get('pressure',{}).get('schur_iterations',0)
            synchronize()
            elapsed = perf_counter()-measured_start
            case = dict(completed=True,setup_warmup_seconds=setup_seconds,elapsed_seconds=elapsed,
                measured_steps=count,ms_per_step=elapsed/count*1000,end_time_s=state.time,
                counts=dict(counters),newton_iterations=newton_iterations,gmres_iterations=gmres_iterations,
                stokes_iterations=stokes_iterations,
                tangent_assemblies=tangent_assemblies,jacobian_actions=jacobian_actions,
                nonlinear_solver=coupling.semiimplicit_solver if scheme=='cnab-semiimplicit' else None,
                execution_variant=variant,final_evaluation_reuses=final_reuses,
                reuse_final_evaluation=coupling.reuse_final_evaluation,
                stokes_warm_start=coupling.stokes_warm_start,
                stokes_pressure_warm_starts=pressure_warm_starts,stokes_pressure_warm_fallbacks=pressure_warm_fallbacks,
                reuse_validation=coupling.reuse_validation,validation_evaluations=validation_evaluations,validation_reuses=validation_reuses,
                interaction_quadrature=driver.transfer.quadrature_summary() if hasattr(driver.transfer,'quadrature_summary') else dict(mode='fixed'),
                anderson_iterations=anderson_iterations,newton_fallback_steps=fallback_steps,
                outer_unknown_dofs=initial.x.numel() if scheme=='cnab-semiimplicit' else sum(v.numel() for v in initial.velocity) if scheme=='implicit-newton' else 0,
                final_solid=model.diagnostics(state.x))
            if execution_variants is not None:
                final_states[label] = dict(x=state.x.clone(),pressure=state.pressure.clone(),
                    force=state.force.clone(),velocity=tuple(u.clone() for u in state.velocity))
            print(f'{label}: {case["ms_per_step"]:.3f} ms/step; '
                  f'Stokes={counters["stokes_solves"]/count:.1f}/step, '
                  f'pressure solves={counters["pressure_solves"]/count:.1f}/step, '
                  f'mass solves={counters["mass_solves"]/count:.1f}/step',flush=True)
            if profile:
                from validation.benchmark_mac import PhaseRecorder
                recorder = PhaseRecorder(torch.device(device))
                profile_before = dict(counters)
                profile_start_time = state.time
                try:
                    with record_phases(driver,recorder):
                        for _ in range(steps):
                            state,_ = driver.step(state,diagnostics=False)
                    synchronize()
                    case['phases'] = recorder.summary(steps)
                    case['profile_counts'] = {k:counters[k]-profile_before[k] for k in counters}
                    case['profile_start_time_s'],case['profile_end_time_s'] = profile_start_time,state.time
                    case['phase_scope'] = 'Separate subsequent replay; mass is nested inside IB and pressure inside Stokes. Do not sum nested phases.'
                    for name,values in case['phases'].items():
                        print(f'  {name}: {values["ms_per_step"]:.3f} ms/step',flush=True)
                except (ValueError,RuntimeError,FloatingPointError) as exc:
                    case['profile_failure'] = dict(type=type(exc).__name__,message=str(exc))
        except (ValueError,RuntimeError,FloatingPointError) as exc:
            case = dict(completed=False,measured_steps=count,counts=dict(counters),
                        failure=dict(type=type(exc).__name__,message=str(exc)))
            print(f'{label}: failed: {exc}',flush=True)
        report['cases'][label] = case
        # Release bound-method closures/graph owners before the next scheme.
        driver.flow.pressure_solver.solve,driver.transfer.solve_mass = pressure_solve,mass_solve
        if stokes_solve is not None:
            driver.flow.stokes = stokes_solve
        del driver,pressure_solve,mass_solve,pressure,mass,stokes_solve,stokes
        gc.collect()
    if all(report['cases'].get(s,{}).get('completed') for s in ('explicit-rk3','implicit-newton')):
        report['implicit_over_rk3_speedup'] = report['cases']['implicit-newton']['ms_per_step']/report['cases']['explicit-rk3']['ms_per_step']
    old,new = (report['cases'].get(f'cnab-semiimplicit/{s}',{}) for s in ('newton','anderson-newton'))
    if old.get('completed') and new.get('completed'):
        report['nonlinear_solver_speedup'] = old['ms_per_step']/new['ms_per_step']
    if execution_variants is not None:
        report['execution_comparisons'] = {}
        for label in final_states:
            prefix = label.rsplit('/',1)[0]
            base = prefix+('/prepare-warm' if label.endswith('/shared-fused-warm') else
                '/shared-fused-warm' if label.endswith('/shared-fused-checked') else
                '/shared-warm' if label.endswith('/prepare-warm') else
                '/compact' if label.endswith(('/fused','/cell','/shared','/pressure-warm','/shared-warm')) else '/baseline')
            if label==base or base not in final_states:
                continue
            reference,candidate = final_states[base],final_states[label]
            differences = {f'{name}_max_abs':(candidate[name]-reference[name]).abs().max().item()
                           for name in ('x','pressure','force')}
            differences['velocity_max_abs'] = max((a-b).abs().max().item() for a,b in zip(candidate['velocity'],reference['velocity']))
            scales = {f'{name}_max_abs':reference[name].abs().max().item() for name in ('pressure','force')}
            scales['velocity_max_abs'] = max(u.abs().max().item() for u in reference['velocity'])
            relative = {name:(torch.linalg.vector_norm(candidate[name]-reference[name])/
                torch.linalg.vector_norm(reference[name]).clamp_min(1e-30)).item() for name in ('pressure','force')}
            velocity_error = sum((a-b).square().sum() for a,b in zip(candidate['velocity'],reference['velocity']))
            velocity_scale = sum(u.square().sum() for u in reference['velocity'])
            relative['velocity'] = (velocity_error/velocity_scale.clamp_min(1e-60)).sqrt().item()
            relative['displacement'] = (torch.linalg.vector_norm(candidate['x']-reference['x'])/
                torch.linalg.vector_norm(reference['x']-model.mesh.X).clamp_min(1e-30)).item()
            report['execution_comparisons'][label] = dict(reference=base,
                measured_speedup=report['cases'][base]['ms_per_step']/report['cases'][label]['ms_per_step'],
                final_state_differences=differences,
                reference_field_max_abs=scales,relative_l2=relative,
                interpretation='Execution/warm-start variants retain reference quadrature and acceptance; compact vs baseline changes IB sampling')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--initial-state',choices=('checkpoint','reference'),default='checkpoint',
                        help='checkpoint: saved physical state; reference: embedded reference mesh, zero flow and t=0 (startup cost only)')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--schemes',nargs='+',default=['explicit-rk3','implicit-newton'])
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--steps',type=int,default=10)
    parser.add_argument('--profile',action='store_true',help='separate subsequent replay with phase timings; excluded from speedup')
    parser.add_argument('--nonlinear-solvers',nargs='+',choices=('newton','anderson-newton'))
    parser.add_argument('--execution-variants',nargs='+',choices=('baseline','reuse','compact','fused','cell','shared','pressure-warm','shared-warm','prepare-warm','shared-fused-warm','shared-fused-checked'),
                        help='adaptive CNAB: compact controls shared stencils and same-step pressure warm starts; input checkpoint is read only')
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    report = benchmark(args.checkpoint,device=args.device,schemes=tuple(args.schemes),warmup=args.warmup,steps=args.steps,
                       nonlinear_solvers=args.nonlinear_solvers,initial_state=args.initial_state,
                       execution_variants=args.execution_variants,profile=args.profile)
    path = Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(path,report)
    print(f'report={path}',flush=True)
    if not all(c['completed'] for c in report['cases'].values()):
        raise SystemExit(1)


if __name__=='__main__':
    main()
