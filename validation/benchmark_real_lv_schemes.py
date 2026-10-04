"""Compare warmed coupled step costs from one real-LV checkpoint, without I/O per step."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
import sys
import gc
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from afsi_torch.real_lv_checkpoint import load_real_lv
from afsi_torch.mac.execution import build_driver
from afsi_torch.cycle_checkpoint import atomic_json


@torch.no_grad()
def benchmark(checkpoint,*,device='cuda',schemes=('explicit-rk3','implicit-newton'),warmup=2,steps=10,
              nonlinear_solvers=None):
    if type(warmup) is not int or warmup<0 or type(steps) is not int or steps<1:
        raise ValueError('warmup must be nonnegative and steps positive')
    if not schemes or len(set(schemes))!=len(schemes) or any(s not in
            ('explicit-lagged','explicit-rk3','implicit-newton','cnab-midpoint','cnab-semiimplicit') for s in schemes):
        raise ValueError('choose distinct supported schemes')
    if nonlinear_solvers is not None and (not nonlinear_solvers or len(set(nonlinear_solvers))!=len(nonlinear_solvers)
            or any(s not in ('newton','anderson-newton') for s in nonlinear_solvers) or tuple(schemes)!=('cnab-semiimplicit',)):
        raise ValueError('nonlinear_solvers requires only cnab-semiimplicit and distinct supported solvers')
    model,initial,settings,_,config = load_real_lv(checkpoint,device)
    synchronize = lambda:torch.cuda.synchronize(device) if torch.device(device).type=='cuda' else None
    report = dict(checkpoint=str(checkpoint),device=str(device),torch=torch.__version__,
        gpu=torch.cuda.get_device_name(device) if torch.device(device).type=='cuda' else None,dt_s=config.time.dt,
        start_time_s=initial.time,warmup_steps=warmup,measured_steps=steps,
        note='Different time schemes: performance comparison, not numerical equivalence or full-cycle stability validation.',cases={})
    cases = [(s,s,None) for s in schemes] if nonlinear_solvers is None else [
        ('cnab-semiimplicit',f'cnab-semiimplicit/{s}',s) for s in nonlinear_solvers]
    for scheme,label,solver in cases:
        print(f'{label}: preparing and warming up...',flush=True)
        started = perf_counter()
        coupling = replace(config.coupling,scheme=scheme)
        if solver is not None:
            coupling = replace(coupling,semiimplicit_solver=solver)
        driver = build_driver(model,dict(settings,coupling=asdict(coupling)),device)
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
                anderson_iterations=anderson_iterations,newton_fallback_steps=fallback_steps,
                outer_unknown_dofs=initial.x.numel() if scheme=='cnab-semiimplicit' else sum(v.numel() for v in initial.velocity) if scheme=='implicit-newton' else 0,
                final_solid=model.diagnostics(state.x))
            print(f'{label}: {case["ms_per_step"]:.3f} ms/step; '
                  f'Stokes={counters["stokes_solves"]/count:.1f}/step, '
                  f'pressure solves={counters["pressure_solves"]/count:.1f}/step, '
                  f'mass solves={counters["mass_solves"]/count:.1f}/step',flush=True)
        except (ValueError,RuntimeError,FloatingPointError) as exc:
            case = dict(completed=False,measured_steps=count,failure=dict(type=type(exc).__name__,message=str(exc)))
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
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--schemes',nargs='+',default=['explicit-rk3','implicit-newton'])
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--steps',type=int,default=10)
    parser.add_argument('--nonlinear-solvers',nargs='+',choices=('newton','anderson-newton'))
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    report = benchmark(args.checkpoint,device=args.device,schemes=tuple(args.schemes),warmup=args.warmup,steps=args.steps,
                       nonlinear_solvers=args.nonlinear_solvers)
    path = Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(path,report)
    print(f'report={path}',flush=True)
    if not all(c['completed'] for c in report['cases'].values()):
        raise SystemExit(1)


if __name__=='__main__':
    main()
