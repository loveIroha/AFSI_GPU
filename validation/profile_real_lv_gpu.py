"""Warmed GPU timeline or isolated frozen-stencil IB capture from a checkpoint."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from statistics import median
from time import perf_counter
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
from afsi_torch.cycle_checkpoint import atomic_json
from afsi_torch.mac.execution import build_driver
from afsi_torch.real_lv_checkpoint import load_real_lv
from validation.benchmark_real_lv_schemes import record_phases
from validation.gpu_trace import RangeRecorder, summarize_trace


def prepare(checkpoint, device, shared_execution):
    """Restore all physical/history fields; change only two execution choices."""
    model,state,settings,_,config = load_real_lv(checkpoint,device)
    q = config.interaction_quadrature
    if (config.coupling.scheme!='cnab-semiimplicit' or q.mode!='adaptive'
            or q.stencil_backend!='shared' or q.transfer_backend!='fused'):
        raise ValueError('requires a cnab-semiimplicit adaptive/shared/fused checkpoint')
    coupling = replace(config.coupling,cnab=replace(config.coupling.cnab,helmholtz_backend='torch'))
    q = replace(q,shared_execution=shared_execution)
    driver = build_driver(model,dict(settings,coupling=asdict(coupling),interaction_quadrature=asdict(q)),device)
    return model,state,driver


def ib_operations(transfer, velocity, coefficient, stencil):
    """Pure FE/IB kernels; neither operation includes a mass solve or prepare."""
    from afsi_torch.mac import adaptive_cell
    return dict(gather=lambda:adaptive_cell.assemble_velocity(transfer,velocity,stencil),
                spread=lambda:adaptive_cell.spread(transfer,coefficient,stencil,reduced=False))


def differences(candidate, reference):
    a = candidate if isinstance(candidate,tuple) else (candidate,)
    b = reference if isinstance(reference,tuple) else (reference,)
    error = sum((x-y).square().sum() for x,y in zip(a,b))
    scale = sum(y.square().sum() for y in b)
    return dict(max_abs=max((x-y).abs().max().item() for x,y in zip(a,b)),
                relative_l2=(error/scale.clamp_min(1e-60)).sqrt().item())


def time_batches(fn, *, device, repeats, batches):
    """Uninstrumented batches; one event pair per batch, no per-call sync."""
    wall, gpu = [], []
    cuda = torch.device(device).type=='cuda'
    for _ in range(batches):
        if cuda:
            torch.cuda.synchronize(device)
            begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record()
        start = perf_counter()
        for _ in range(repeats):
            fn()
        if cuda:
            end.record()
            end.synchronize()
            gpu.append(begin.elapsed_time(end)/repeats)
        wall.append((perf_counter()-start)*1000/repeats)
    return dict(repeats_per_batch=repeats,batches=batches,wall_ms_samples=wall,
                wall_ms_median=median(wall),gpu_stream_ms_samples=gpu,
                gpu_stream_ms_median=median(gpu) if gpu else None,
                scope='Uninstrumented warmed batches, including output allocation/zeroing. CUDA stream span can include host enqueue gaps; it is not the sum of active kernel time.')


@torch.no_grad()
def profile_checkpoint(checkpoint, *, device='cuda', scope='coupled', operation='both',
                       shared_execution='reference', warmup=5, steps=2, repeats=10,
                       batches=3, capture='torch', output='results/real_lv_gpu_profile'):
    if any(type(v) is not int or v<1 for v in (steps,repeats,batches)) or type(warmup) is not int or warmup<1:
        raise ValueError('warmup, steps, repeats and batches must be positive integers')
    if scope not in ('coupled','ib') or operation not in ('gather','spread','both') or capture not in ('torch','external'):
        raise ValueError('invalid profiling scope, operation or capture mode')
    target = Path(output)
    if Path(checkpoint).resolve().parent==target.resolve():
        raise ValueError('use a separate profiling output directory; source checkpoint is read only')
    if (target/'report.json').exists() or (target/'trace.json').exists():
        raise FileExistsError('profiling output already exists; choose a new directory')
    dev = torch.device(device)
    cuda = dev.type=='cuda'
    if capture=='external' and not cuda:
        raise ValueError('external capture requires CUDA')
    # A selected CUDA device also selects the event/profiler/NVTX context.
    context = torch.cuda.device(dev) if cuda else nullcontext()
    with context:
        return _profile(checkpoint,dev,scope,operation,shared_execution,warmup,steps,repeats,batches,capture,target)


def _profile(checkpoint,device,scope,operation,shared_execution,warmup,steps,repeats,batches,capture,target):
    cuda = device.type=='cuda'
    synchronize = lambda:torch.cuda.synchronize(device) if cuda else None
    model,state,driver = prepare(checkpoint,device,shared_execution)
    report = dict(checkpoint=str(checkpoint),source_time_s=state.time,source_step=state.step,
        device=str(device),torch=torch.__version__,gpu=torch.cuda.get_device_name(device) if cuda else None,
        scope=scope,shared_execution=shared_execution,helmholtz_backend=driver.flow.helmholtz_backend,
        capture=capture,warmup=warmup,dt_s=driver.flow.dt,
        numerical_changes=False,source_checkpoint_written=False,
        note='Diagnostic replay, not a full-cycle simulation. Profiling overhead is excluded from uninstrumented timings.')
    if cuda:
        props = torch.cuda.get_device_properties(device)
        report['hardware'] = dict(sm_count=props.multi_processor_count,total_memory_bytes=props.total_memory,
            compute_capability=list(torch.cuda.get_device_capability(device)),cuda_runtime=torch.version.cuda)
    if scope=='coupled':
        def work():
            nonlocal state
            state,info = driver.step(state,diagnostics=False)
            return info
        for _ in range(warmup):work()
        report['uninstrumented_start_time_s'] = state.time
        report['timing'] = time_batches(work,device=device,repeats=steps,batches=batches)
        report['uninstrumented_end_time_s'] = state.time
        report['timing']['comparison_note'] = 'Sequential coupled batches advance physical time; sample variation also reflects changing load/geometry.'
        captured_steps = []
        def capture_work(recorder):
            with record_phases(driver,recorder):
                for _ in range(steps):
                    with recorder.range('step'):
                        info = work()
                    nonlinear = info.get('nonlinear',{})
                    captured_steps.append(dict(step=state.step,time_s=state.time,
                        anderson_iterations=nonlinear.get('anderson_iterations',0),
                        newton_iterations=nonlinear.get('newton_iterations',0),
                        residual_norm=nonlinear.get('residual_norm'),tolerance=nonlinear.get('tolerance'),
                        final_evaluation_reused=nonlinear.get('final_evaluation_reused',False)))
        report['captured_steps'] = captured_steps
        report['capture_start_time_s'] = state.time
    else:
        stencil = driver.transfer.prepare(state.x)
        # The coefficient solve happens once, before timing/capture. Use the
        # current checkpoint load, not a potentially lagged cached force.
        coefficient,_ = driver.transfer.solve_mass(model.force(state.x,state.time),None)
        operations = ib_operations(driver.transfer,state.velocity,coefficient,stencil)
        names = ('gather','spread') if operation=='both' else (operation,)
        original = driver.transfer.quadrature_options
        driver.transfer.quadrature_options = replace(original,shared_execution='reference')
        try:
            baseline = {name:operations[name]() for name in names}
        finally:
            driver.transfer.quadrature_options = original
        report['equivalence'] = {name:differences(operations[name](),baseline[name]) for name in names}
        del baseline
        for name in names:
            for _ in range(warmup):operations[name]()
        report['timing'] = {name:time_batches(operations[name],device=device,repeats=repeats,batches=batches)
                            for name in names}
        report['frozen_stencil'] = dict(configuration='checkpoint x (not a nonlinear midpoint predictor)',
            point_count=stencil.rule.point_count,mass_solve_in_timing=False,prepare_in_timing=False,
            physical_time_advanced=False,coefficient='consistent mass solve of force(x, checkpoint time)')
        def capture_work(recorder):
            for _ in range(steps):
                for name in names:
                    with recorder.range('ib_'+name+'_kernel'):
                        operations[name]()
    synchronize()
    target.mkdir(parents=True,exist_ok=True)
    recorder = RangeRecorder(nvtx=capture=='external')
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    if capture=='torch':
        activities = [torch.profiler.ProfilerActivity.CPU]
        if cuda:activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities,record_shapes=False,
                                    profile_memory=False,with_stack=False) as prof:
            with recorder.range('capture'):
                capture_work(recorder)
                synchronize()
        trace = target/'trace.json'
        prof.export_chrome_trace(str(trace))
        report['trace'] = summarize_trace(trace)
        averages = prof.key_averages()
        (target/'operators.txt').write_text(averages.table(sort_by='self_device_time_total' if cuda else
                                                         'self_cpu_time_total',row_limit=40),encoding='utf-8')
    else:
        # Do not run Kineto and Nsight/CUPTI collection together. Compilation,
        # coefficient solve and uninstrumented batches are outside this range.
        torch.cuda.profiler.start()
        try:
            with recorder.range('capture'):
                capture_work(recorder)
                synchronize()
        finally:
            torch.cuda.profiler.stop()
        report['trace'] = dict(external=True,note='Collect using nsys --capture-range=cudaProfilerApi or ncu --profile-from-start off. External instrumentation distorts timings; do not compare throughput from this run.')
    report['ranges'] = dict(recorder.calls)
    report['interaction_quadrature'] = driver.transfer.quadrature_summary()
    report['capture_end_time_s'] = state.time
    if cuda:
        report['memory'] = dict(allocated_bytes=torch.cuda.memory_allocated(device),
            reserved_bytes=torch.cuda.memory_reserved(device),peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
    atomic_json(target/'report.json',report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--scope',choices=('coupled','ib'),default='coupled')
    p.add_argument('--operation',choices=('gather','spread','both'),default='both')
    p.add_argument('--ib-shared-execution',choices=('reference','vector','reduced'),default='reference')
    p.add_argument('--capture',choices=('torch','external'),default='torch')
    p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--steps',type=int,default=2,help='short capture steps; compilation excluded')
    p.add_argument('--repeats',type=int,default=10,help='IB calls per uninstrumented batch')
    p.add_argument('--batches',type=int,default=3)
    p.add_argument('--output',required=True)
    a = p.parse_args()
    r = profile_checkpoint(a.checkpoint,device=a.device,scope=a.scope,operation=a.operation,
        shared_execution=a.ib_shared_execution,warmup=a.warmup,steps=a.steps,repeats=a.repeats,
        batches=a.batches,capture=a.capture,output=a.output)
    print(f'report={Path(a.output)/"report.json"}',flush=True)
    print(f'timing={r["timing"]}',flush=True)
    if not r['trace'].get('gpu_events_available',True):
        print('No GPU events recorded: trace cannot diagnose GPU occupancy/idle time. Use external Nsight capture.',flush=True)


if __name__=='__main__':main()
