"""Same-equation MAC/IB/FEM execution comparison, without modifying checkpoints."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import torch
from benchmark_mac import PhaseRecorder,record_phases,replay,synchronize
from benchmark_mac_pressure import comparison
from afsi_torch.cycle_checkpoint import atomic_json
from afsi_torch.mac.checkpoint import load_mac
from afsi_torch.mac.execution import build_driver
from afsi_torch.mac.grid import divergence


@contextmanager
def detail_phases(transfer,recorder):
    names={'solve_mass':'ib_mass_solves','weighted_force':'fe_force_evaluation',
           'spread_grid':'ib_spread_grid','gather_grid':'ib_gather_grid',
           'assemble_velocity':'fe_velocity_assembly'}
    original={name:getattr(transfer,name) for name in names}
    try:
        for name,label in names.items():
            setattr(transfer,name,recorder.wrap(original[name],label))
        yield
    finally:
        for name,method in original.items():
            setattr(transfer,name,method)


@torch.no_grad()
def benchmark(checkpoint,device='cuda',steps=20,warmup=3):
    if steps<1 or warmup<3:
        raise ValueError('positive steps and at least three warmup steps required')
    device=torch.device(device)
    model,initial,settings,_=load_mac(checkpoint,device)
    settings=dict(settings,warm_start=True,pressure_backend='fused')
    variants,states={},{}
    for backend in ('torch','fused'):
        print(f'{backend}: setup and warmup; first use compiles GPU kernels...',flush=True)
        synchronize(device)
        started=perf_counter()
        driver=build_driver(model,dict(settings,execution_backend=backend),device)
        replay(driver,initial,warmup,device)
        compile_seconds=perf_counter()-started
        driver.transfer.reset_warm_start()
        if device.type=='cuda':
            torch.cuda.reset_peak_memory_stats(device)
        base_bytes=torch.cuda.memory_allocated(device) if device.type=='cuda' else None
        end,plain=replay(driver,initial,steps,device)
        peak=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None
        states[backend]=end
        driver.transfer.reset_warm_start()
        recorder=PhaseRecorder(device)
        with record_phases(driver,recorder),detail_phases(driver.transfer,recorder):
            repeated,profiled=replay(driver,initial,steps,device)
        phases=recorder.summary(steps)
        transfer=driver.transfer
        stencil=transfer.prepare(end.x)
        stencil_bytes=(stencil.storage_bytes if hasattr(stencil,'storage_bytes') else
                       sum(v.numel()*v.element_size() for v in (*stencil.indices,*stencil.weights)))
        density,force_info=transfer.spread(end.force,stencil)
        velocity,velocity_info=transfer.interpolate(end.velocity,stencil)
        solid_power=(velocity*end.force).sum().item()
        fluid_power=(driver.flow.grid.volume*sum((u*f).sum() for u,f in zip(end.velocity,density))).item()
        variants[backend]=dict(**plain,ms_per_step=1000*plain['wall_seconds']/steps,
            setup_warmup_seconds=compile_seconds,instrumented_wall_seconds=profiled['wall_seconds'],
            phase_timings=phases,stencil_bytes=stencil_bytes,
            actual_backend=getattr(transfer,'execution_backend','torch'),
            pressure_backend=driver.flow.pressure_solver.backend,
            peak_allocated_bytes=peak,peak_extra_bytes=peak-base_bytes if peak is not None else None,
            repeat_equivalence=comparison(end,repeated),final_solid=model.diagnostics(end.x),
            final_divergence_l2=(driver.flow.grid.volume*divergence(end.velocity,driver.flow.grid.spacing).square().sum()).sqrt().item(),
            final_power_abs_error=abs(solid_power-fluid_power),
            final_power_relative_error=abs(solid_power-fluid_power)/max(abs(solid_power),abs(fluid_power),1e-30),
            final_force_mass=asdict(force_info),final_velocity_mass=asdict(velocity_info))
        print(f"{backend}: {variants[backend]['ms_per_step']:.3f} ms/step; stencil={stencil_bytes/2**20:.2f} MiB",flush=True)
        for name,values in phases.items():
            print(f"  {name}: {values['ms_per_step']:.3f} ms/step",flush=True)
        del driver,transfer,stencil,density,velocity,repeated
    equivalent=comparison(states['torch'],states['fused'])
    return dict(schema=1,benchmark='mac-execution-replay',checkpoint=str(Path(checkpoint).resolve()),
        device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        torch=torch.__version__,dtype=str(initial.x.dtype),settings=settings,
        steps=steps,warmup_steps=warmup,start_time_s=initial.time,end_time_s=states['fused'].time,
        variants=variants,equivalence=equivalent,
        passed=equivalent['passed'] and all(v['repeat_equivalence']['passed'] for v in variants.values()),
        whole_step_speedup=variants['torch']['wall_seconds']/variants['fused']['wall_seconds'],
        notes=['Both variants use fused pressure MG and identical warm starts; only execution differs.',
               'Full phase timings include nested subphases: do not add all rows.',
               'ib_mass_solves is the sum of force and velocity consistent mass solves.',
               'Pressure true residual uses the same operator, compiled on the optimized CUDA path.',
               'CPU buffered execution does not validate CUDA compilation/performance.',
               'Setup/compilation, final diagnostics, I/O are excluded from timed replay.',
               'GPU atomics and fused arithmetic change rounding; compare with numerical tolerances.',
               'This benchmark never changes its input checkpoint. A 2 s checkpoint samples the held-load tail.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--output',default='results/mac_execution/report.json')
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():
        raise FileExistsError(f'choose a new report path: {output}')
    report=benchmark(args.checkpoint,args.device,args.steps,args.warmup)
    output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(output,report)
    print(json.dumps({key:report[key] for key in ('passed','whole_step_speedup','equivalence')},indent=2))
    print(f'Report: {output}',flush=True)
    if not report['passed']:
        raise SystemExit('Execution comparison failed; inspect report.')


if __name__=='__main__':
    main()
