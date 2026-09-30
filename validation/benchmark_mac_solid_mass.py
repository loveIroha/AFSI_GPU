"""Read-only four-way experiment: pointwise solid algebra and graph CSR-PCG."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import torch
from benchmark_mac import PhaseRecorder,record_phases,replay,synchronize
from benchmark_mac_execution import detail_phases
from benchmark_mac_pressure import comparison
from afsi_torch.cycle_checkpoint import atomic_json
from afsi_torch.mac.checkpoint import load_mac
from afsi_torch.mac.execution import build_driver,tensor_kernel
from afsi_torch.mac.mass_graph import GraphMassSolver
from afsi_torch.mac.solid_pointwise import PointwiseSolidExecution
from afsi_torch.mac.grid import divergence


def experiment_driver(model,settings,device,variant):
    driver=build_driver(model,settings,device)
    if variant in ('pointwise','combined'):
        solid=PointwiseSolidExecution(model)
        driver.force,driver.validate=solid.force,solid.validate
    if variant in ('pcg_graph','combined'):
        t=driver.transfer
        t.mass_solver=GraphMassSolver(t.mass,t.diagonal,t.options)
    return driver


@contextmanager
def mass_details(solver,recorder):
    names=('action','advance','direction','restart')
    original={n:getattr(solver,n) for n in names}
    try:
        for n in names:
            setattr(solver,n,recorder.wrap(original[n],'mass_'+n))
        yield
    finally:
        for n,f in original.items():
            setattr(solver,n,f)


def measure(function,args,device,steps):
    for _ in range(3):
        function(*args)
    synchronize(device)
    started=perf_counter()
    for _ in range(steps):
        output=function(*args)
    synchronize(device)
    return output,1000*(perf_counter()-started)/steps


def solid_details(solid,state,device,steps):
    # Isolated kernels materialize P at a boundary. Their timings must not be
    # added or subtracted from the fully fused force timing.
    solid.validate(state.x)
    F,area,_=solid._cached_geometry
    p,t=solid.model.loads.at(state.time)
    solid.loads[0].fill_(p)
    solid.loads[1].fill_(t)
    stress=tensor_kernel(solid._stress,device)
    assemble=tensor_kernel(solid._assemble,device)
    P,stress_ms=measure(stress,(F,solid.loads),device,steps)
    result,assembly_ms=measure(assemble,(state.x,P,area,solid.loads),device,steps)
    reference=solid.force(state.x,state.time)
    torch.testing.assert_close(result,reference,rtol=1e-7,atol=1e-8)
    return dict(constitutive_ms_per_call=stress_ms,assembly_ms_per_call=assembly_ms,
                split_force_max_abs=(result-reference).abs().max().item())


@torch.no_grad()
def benchmark(checkpoint,device='cuda',steps=20,warmup=3):
    if steps<1 or warmup<3:
        raise ValueError('positive steps and at least three warmup steps required')
    device=torch.device(device)
    model,initial,settings,_=load_mac(checkpoint,device)
    settings=dict(settings,warm_start=True,pressure_backend='fused',execution_backend='fused')
    variants,states={},{}
    for variant in ('reference','pointwise','pcg_graph','combined'):
        print(f'{variant}: setup, compilation and warmup...',flush=True)
        synchronize(device)
        started=perf_counter()
        driver=experiment_driver(model,settings,device,variant)
        replay(driver,initial,warmup,device)
        setup_seconds=perf_counter()-started
        driver.transfer.reset_warm_start()
        end,plain=replay(driver,initial,steps,device)
        states[variant]=end
        driver.transfer.reset_warm_start()
        recorder=PhaseRecorder(device)
        with record_phases(driver,recorder),detail_phases(driver.transfer,recorder):
            repeated,profiled=replay(driver,initial,steps,device)
        phases=recorder.summary(steps)
        # Detailed per-iteration event recording perturbs submission overhead.
        # Keep this in a separate replay and never use it as the speedup result.
        mass_profile=None
        if variant in ('reference','pointwise'):
            driver.transfer.reset_warm_start()
            detail=PhaseRecorder(device)
            with mass_details(driver.transfer.mass_solver,detail):
                replay(driver,initial,steps,device)
            mass_profile=detail.summary(steps)
        stencil=driver.transfer.prepare(end.x)
        density,force_info=driver.transfer.spread(end.force,stencil)
        velocity,velocity_info=driver.transfer.interpolate(end.velocity,stencil)
        solid_power=(velocity*end.force).sum().item()
        fluid_power=(driver.flow.grid.volume*sum((u*f).sum() for u,f in zip(end.velocity,density))).item()
        variants[variant]=dict(**plain,ms_per_step=1000*plain['wall_seconds']/steps,
            setup_warmup_seconds=setup_seconds,phase_timings=phases,
            instrumented_wall_seconds=profiled['wall_seconds'],
            mass_instrumented_details=mass_profile,
            solid_isolated_details=solid_details(driver.force.__self__,end,device,steps),
            cuda_graphs=len(getattr(driver.transfer.mass_solver,'graphs',{})),
            repeat_equivalence=comparison(end,repeated),final_solid=model.diagnostics(end.x),
            final_force_mass=asdict(force_info),final_velocity_mass=asdict(velocity_info),
            final_divergence_l2=(driver.flow.grid.volume*divergence(end.velocity,driver.flow.grid.spacing).square().sum()).sqrt().item(),
            final_power_relative_error=abs(solid_power-fluid_power)/max(abs(solid_power),abs(fluid_power),1e-30))
        print(f"{variant}: {variants[variant]['ms_per_step']:.3f} ms/step",flush=True)
        for name in ('solid_force','ib_mass_solves','fluid_total'):
            print(f"  {name}: {phases[name]['ms_per_step']:.3f} ms/step",flush=True)
        del driver,repeated,stencil,density,velocity
    equivalence={v:comparison(states['reference'],states[v]) for v in variants if v!='reference'}
    speedups={v:variants['reference']['wall_seconds']/variants[v]['wall_seconds']
              for v in equivalence}
    return dict(schema=1,benchmark='mac-solid-mass-execution-experiment',
        checkpoint=str(Path(checkpoint).resolve()),device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        torch=torch.__version__,dtype=str(initial.x.dtype),settings=settings,
        steps=steps,warmup_steps=warmup,start_time_s=initial.time,
        end_time_s=states['reference'].time,variants=variants,equivalence=equivalence,
        speedups=speedups,passed=all(v['passed'] for v in equivalence.values()) and
        all(v['repeat_equivalence']['passed'] for v in variants.values()),
        notes=['Four variants share the existing optimized IB/fluid execution and warm starts.',
               'Pointwise changes only the execution of 3x3 Guccione stress algebra.',
               'Graph PCG retains CSR/Jacobi, check/recompute schedules, and true residual acceptance.',
               'CPU tests exercise blocked recurrence, not CUDA Graph capture or GPU performance.',
               'Use uninstrumented wall times for speedups. Nested phase rows cannot be summed.',
               'Per-iteration event profiling perturbs launch overhead and excludes host checks.',
               'Isolated stress/assembly timings introduce a materialized stress boundary; not additive with fused force.',
               'Setup/compilation/capture, final diagnostics and I/O are outside timed replay.',
               'Input checkpoint is read-only. A tail checkpoint does not test the load ramp.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--steps',type=int,default=20)
    p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--output',default='results/mac_solid_mass/report.json')
    args=p.parse_args()
    output=Path(args.output)
    if output.exists():
        raise FileExistsError(f'choose a new report path: {output}')
    report=benchmark(args.checkpoint,args.device,args.steps,args.warmup)
    output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(output,report)
    print(json.dumps(dict(passed=report['passed'],speedups=report['speedups']),indent=2))
    print(f'Report: {output}',flush=True)
    if not report['passed']:
        raise SystemExit('Execution comparison failed; inspect report.')


if __name__=='__main__':
    main()
