"""Read-only same-state LV replay: prior fused execution vs workspace/graphs."""
import argparse
from contextlib import contextmanager
from pathlib import Path
from statistics import mean
from time import perf_counter
import sys
import torch
if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from afsi_torch.mac.checkpoint import load_mac
from afsi_torch.mac.execution import build_driver
from afsi_torch.cycle_checkpoint import atomic_json
from validation.benchmark_mac import PhaseRecorder,synchronize,compare_states


@contextmanager
def phases(driver,recorder):
    solid=driver.solid_execution
    targets=[(driver.flow,'step','fluid_total'),(driver.flow.pressure_solver,'solve','pressure_solve'),
        (driver.transfer,'spread','ib_spread_with_mass'),(driver.transfer,'interpolate','ib_interpolate_with_mass'),
        (driver.transfer,'_prepare_kernel','ib_stencil'),(driver.transfer,'_evaluate_kernel','fe_point_evaluation'),
        (driver.transfer,'solve_mass','ib_mass_solves'),(solid,'_geometry_kernel','solid_geometry_checks'),
        (solid,'_force_kernel','solid_force')]
    if driver.optimized:
        targets.append((driver,'_accept_metrics','coupled_acceptance'))
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
        state,info=driver.step(state,diagnostics=False)
        cycles.append(info['flow']['pressure']['cycles'])
        force_iterations.append(info['force_mass']['iterations'])
        velocity_iterations.append(info['velocity_mass']['iterations'])
    synchronize(device)
    elapsed=perf_counter()-start
    return state,dict(elapsed_seconds=elapsed,ms_per_step=1000*elapsed/steps,
        pressure_cycles_mean=mean(cycles),pressure_cycles_max=max(cycles),
        force_mass_iterations_mean=mean(force_iterations),velocity_mass_iterations_mean=mean(velocity_iterations))


def equivalence(reference,candidate):
    differences={name+'_max_abs':(getattr(candidate,name)-getattr(reference,name)).abs().max().item()
                 for name in ('x','force','pressure')}
    differences.update({f'velocity_{c}_max_abs':(v-u).abs().max().item()
                        for c,(u,v) in enumerate(zip(reference.velocity,candidate.velocity))})
    relative={name:(torch.linalg.vector_norm(getattr(candidate,name)-getattr(reference,name))/
        torch.linalg.vector_norm(getattr(reference,name)).clamp_min(1e-30)).item() for name in ('x','force','pressure')}
    relative.update({f'velocity_{c}':(torch.linalg.vector_norm(v-u)/torch.linalg.vector_norm(u).clamp_min(1e-30)).item()
                     for c,(u,v) in enumerate(zip(reference.velocity,candidate.velocity))})
    result=dict(passed=True,max_abs=differences,relative_l2=relative,
        state_rtol=1e-7,state_atol=1e-8,velocity_rtol=1e-7,velocity_atol=1e-9)
    try:
        compare_states(reference,candidate)
        assert (reference.step,reference.time,reference.force_time)==(candidate.step,candidate.time,candidate.force_time)
    except AssertionError as exc:
        result.update(passed=False,error=str(exc))
    return result


@torch.no_grad()
def benchmark(path,*,device='cuda',steps=100,warmup=10,profile=False):
    if type(steps) is not int or steps<1 or type(warmup) is not int or warmup<1:
        raise ValueError('positive integer steps/warmup required')
    device=torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    variants=[('reference','fused','reference'),('workspace','workspace','optimized')]
    if device.type=='cuda':
        variants.append(('graph','graph','optimized'))
    records={}; reference=None
    for name,pressure,coupling in variants:
        print(f'{name}: setup and warmup; first CUDA use compiles kernels...',flush=True)
        model,state,settings,_=load_mac(path,device)
        source_settings=dict(settings)
        # Compare against the previous fused baseline, retaining the checkpoint's
        # solid/mass backends and warm-start choice; no spatial/physical edits.
        settings=dict(settings,execution_backend='fused',pressure_backend=pressure,coupling_backend=coupling)
        driver=build_driver(model,settings,device)
        start_step,start_time=state.step,state.time
        state,_=replay(driver,state,warmup,device)
        builds,evaluations=driver.stencil_builds,driver.point_evaluations
        final,timing=replay(driver,state,steps,device)
        entry=dict(settings=settings,pressure_backend=driver.flow.pressure_solver.backend,
            pressure_cuda_graphs=len(driver.flow.pressure_solver.workspace.graphs),
            pressure_workspace_bytes=driver.flow.pressure_solver.workspace.allocated_bytes,
            mass_cuda_graphs=len(getattr(driver.transfer.mass_solver,'graphs',{})),
            stencil_builds=driver.stencil_builds-builds,point_evaluations=driver.point_evaluations-evaluations,**timing)
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
        print(f'{name}: {timing["ms_per_step"]:.3f} ms/step, MG cycles={timing["pressure_cycles_mean"]:.2f}, '
              f'IB point evaluations={entry["point_evaluations"]}',flush=True)
        if profile:
            for label,values in entry['phases'].items():
                print(f'  {label}: {values["ms_per_step"]:.3f} ms/step',flush=True)
        end_step,end_time=final.step,final.time
        del driver,model,state,final
    candidate='graph' if device.type=='cuda' else 'workspace'
    return dict(schema=1,benchmark='lv-mac-graph-replay',checkpoint=str(Path(path).resolve()),
        device=str(device),torch=torch.__version__,gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        steps=steps,warmup=warmup,start_step=start_step,start_time_s=start_time,end_step=end_step,end_time_s=end_time,
        source_settings=source_settings,variants=records,
        equivalence_passed=all(r.get('equivalence',{}).get('passed',True) for r in records.values()),physical_settings_changed=False,
        measured_speedup=records['reference']['elapsed_seconds']/records[candidate]['elapsed_seconds'],
        timing_scope='warm wall time; excludes setup, compilation, output and optional profile pass',
        phase_scope='separate subsequent replay; pressure is within fluid_total, mass within IB phases; do not sum nested phases',
        pressure_gauge='closed-box homogeneous Neumann with original zero-mean constraints at every level',
        notes=['All variants use fused execution, the same solid/mass backends, and the same warm-start setting.',
               'A completed 2 s checkpoint samples held loads; early loading must be checked separately.',
               'No input checkpoint or full-demo files are modified.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True); p.add_argument('--device',default='cuda')
    p.add_argument('--steps',type=int,default=100); p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--output',default='results/lv_mac_graph_performance/report.json')
    args=p.parse_args()
    target=Path(args.output)
    if target.resolve()==Path(args.checkpoint).resolve():
        raise ValueError('benchmark output must not overwrite checkpoint')
    report=benchmark(args.checkpoint,device=args.device,steps=args.steps,warmup=args.warmup,profile=args.profile)
    target.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(target,report)
    print(f'report={target}, measured_speedup={report["measured_speedup"]:.3f}',flush=True)
    if not report['equivalence_passed']:
        raise SystemExit('LV execution comparison failed; inspect the saved report.')


if __name__=='__main__':
    main()
