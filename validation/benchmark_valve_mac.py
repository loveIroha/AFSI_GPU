"""Read-only A/B replay: original compiled 2D execution vs workspace/graph.

Wall timings exclude setup, warmup, output and optional phase instrumentation.
CUDA events, when requested, are read only after a separate short replay.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
from statistics import mean
from time import perf_counter
import torch
if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from afsi_torch.mac2d import checkpoint
from afsi_torch.mac2d.execution import build_driver
from afsi_torch.cycle_checkpoint import atomic_json
from validation.benchmark_mac import PhaseRecorder,synchronize


@contextmanager
def phases(driver,recorder):
    force='force_and_det' if driver.optimized else 'force'
    targets=[(driver.flow,'step','fluid_total'),(driver.flow.pressure_solver,'solve','pressure_solve'),
        (driver.transfer,'spread','ib_spread_with_mass'),(driver.transfer,'interpolate','ib_interpolate_with_mass'),
        (driver.transfer,'_prepare','ib_stencil'),(driver.solid,force,'solid_force'),
        (driver,'_accept_metrics','acceptance_metrics')] if driver.optimized else [
        (driver.flow,'step','fluid_total'),(driver.flow.pressure_solver,'solve','pressure_solve'),
        (driver.transfer,'spread','ib_spread_with_mass'),(driver.transfer,'interpolate','ib_interpolate_with_mass'),
        (driver.transfer,'_prepare','ib_stencil'),(driver.solid,force,'solid_force'),
        (driver.solid,'validate','solid_validation')]
    originals=[(obj,name,getattr(obj,name)) for obj,name,_ in targets]
    try:
        for (obj,name,label),(_,_,fn) in zip(targets,originals):
            setattr(obj,name,recorder.wrap(fn,label))
        yield
    finally:
        for obj,name,fn in originals:
            setattr(obj,name,fn)


def replay(driver,state,steps,device):
    cycles=[]; force_iterations=[]; velocity_iterations=[]
    synchronize(device)
    start=perf_counter()
    for _ in range(steps):
        state,info=driver.step(state)
        cycles.append(info['flow']['pressure']['cycles'])
        force_iterations.append(info['force_mass']['iterations'])
        velocity_iterations.append(info['velocity_mass']['iterations'])
    synchronize(device)
    elapsed=perf_counter()-start
    return state,dict(elapsed_seconds=elapsed,ms_per_step=1000*elapsed/steps,
        pressure_cycles_mean=mean(cycles),pressure_cycles_max=max(cycles),
        force_mass_iterations_mean=mean(force_iterations),velocity_mass_iterations_mean=mean(velocity_iterations))


def equivalence(a,b):
    results={}
    for name,x,y in [(n,getattr(a,n),getattr(b,n)) for n in ('x','pressure','force')]+[
        (f'velocity_{c}',u,v) for c,(u,v) in enumerate(zip(a.velocity,b.velocity))]:
        rtol,atol=(1e-9,1e-9) if name=='x' else (1e-7,1e-7)
        torch.testing.assert_close(y,x,rtol=rtol,atol=atol)
        results[name]=dict(max_abs=(y-x).abs().max().item(),
            relative_l2=(torch.linalg.vector_norm(y-x)/torch.linalg.vector_norm(x).clamp_min(1e-30)).item(),
            comparison_rtol=rtol,comparison_atol=atol)
    return results


@torch.no_grad()
def benchmark(path,*,device='cuda',steps=100,warmup=10,profile=False):
    device=torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if steps<1 or warmup<1:
        raise ValueError('positive steps/warmup required')
    variants=[('reference','reference','reference'),('workspace','optimized','workspace')]
    if device.type=='cuda':
        variants.append(('graph','optimized','graph'))
    records={}; reference=None
    for name,execution,pressure in variants:
        solid,state,settings,_=checkpoint.load(path,device)
        settings=dict(settings,execution_backend=execution,pressure_backend=pressure)
        driver=build_driver(solid,settings,device)
        # Warm all kernels/graphs and keep the same physical warmup for each variant.
        state,_=replay(driver,state,warmup,device)
        builds=driver.stencil_builds
        final,timing=replay(driver,state,steps,device)
        entry=dict(settings=settings,pressure_backend=driver.flow.pressure_solver.backend,
            pressure_graphs=len(getattr(driver.flow.pressure_solver.workspace,'graphs',{})),
            mass_graphs=len(getattr(driver.transfer.solver,'graphs',{})),
            stencil_builds=driver.stencil_builds-builds,**timing)
        if reference is None:
            reference=final
        else:
            entry['equivalence']=equivalence(reference,final)
        if profile:
            recorder=PhaseRecorder(device)
            with phases(driver,recorder):
                replay(driver,final,steps,device)
            entry['phases']=recorder.summary(steps)
        records[name]=entry
        print(f'{name}: {timing["ms_per_step"]:.3f} ms/step, '
              f'MG cycles={timing["pressure_cycles_mean"]:.2f}, stencil builds={entry["stencil_builds"]}',flush=True)
        if profile:
            for label,value in entry['phases'].items():
                print(f'  {label}: {value["ms_per_step"]:.3f} ms/step',flush=True)
        del driver,solid,state,final
    fastest='graph' if device.type=='cuda' else 'workspace'
    speedup=records['reference']['elapsed_seconds']/records[fastest]['elapsed_seconds']
    return dict(schema=1,checkpoint=str(Path(path).resolve()),device=str(device),torch=torch.__version__,
        gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        steps=steps,warmup=warmup,variants=records,measured_speedup=speedup,
        equivalence_passed=True,physical_settings_changed=False,
        timing_scope='warm wall time; excludes initialization, output and optional phase pass',
        phase_scope='separate subsequent replay; pressure is included in fluid_total; do not sum overlapping phases')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True); p.add_argument('--device',default='cuda')
    p.add_argument('--steps',type=int,default=100); p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--output',default='results/valve_mac_performance/report.json')
    args=p.parse_args()
    report=benchmark(args.checkpoint,device=args.device,steps=args.steps,warmup=args.warmup,profile=args.profile)
    target=Path(args.output)
    if target.resolve()==Path(args.checkpoint).resolve():
        raise ValueError('benchmark output must not overwrite checkpoint')
    target.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(target,report)
    print(f'report={target}, measured_speedup={report["measured_speedup"]:.3f}',flush=True)


if __name__=='__main__':
    main()
