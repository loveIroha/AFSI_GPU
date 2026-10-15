"""Same-checkpoint A/B replay: P2 CUDA fusion and FGMRES workspaces only."""
import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
from time import perf_counter
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from afsi_torch.mac.checkpoint import load_mac
from afsi_torch.mac.execution import build_driver
from validation.benchmark_mac import PhaseRecorder


def execution_settings(saved,variant):
    settings=deepcopy(saved)
    if settings.get('coupling',{}).get('scheme')!='cnab-midpoint':
        raise ValueError('benchmark requires the P2 CN-AB2 midpoint checkpoint')
    if settings.get('ib_backend')!='cuda':
        raise ValueError('benchmark requires the native CUDA IB checkpoint')
    quadrature=settings['interaction_quadrature']
    quadrature.update(transfer_backend='reference',stencil_backend='component',
        prepare_backend='torch',reuse_stencil_buffers=False,shared_execution='reference')
    if variant!='reference':
        quadrature.update(transfer_backend='fused',stencil_backend='shared',prepare_backend='cuda',reuse_stencil_buffers=True)
    settings['coupling']['cnab']['stokes_vector_backend']='workspace' if variant=='optimized' else 'reference'
    return settings


def arrays(state):
    return {**{name:getattr(state,name).detach().cpu().clone() for name in ('x','force','pressure')},
        **{f'velocity_{c}':u.detach().cpu().clone() for c,u in enumerate(state.velocity)}}


def profile(driver,state,steps,device):
    recorder=PhaseRecorder(torch.device(device))
    targets=[(driver.transfer,'prepare','ib_prepare'),(driver.transfer,'interpolate','ib_interpolate_with_mass'),
        (driver.transfer,'spread','ib_spread_with_mass'),(driver.transfer,'solve_mass','mass_solves'),
        (driver.flow,'advance','fluid_advance'),(driver.flow.pressure_solver,'solve','pressure_mg'),
        (driver,'_force_geometry','solid_force_checks'),(driver.flow,'advection','advection')]
    originals=[]
    try:
        for obj,name,label in targets:
            f=getattr(obj,name);originals.append((obj,name,f));setattr(obj,name,recorder.wrap(f,label))
        for _ in range(steps):state,_=driver.step(state,diagnostics=False)
        torch.cuda.synchronize()
        return state,recorder.summary(steps)
    finally:
        for obj,name,f in originals:setattr(obj,name,f)


@torch.no_grad()
def benchmark(args):
    results={};ends={}
    for variant in args.variants:
        print(variant+': setup and warmup',flush=True)
        model,state,saved,_=load_mac(args.checkpoint,args.device)
        settings=execution_settings(saved,variant)
        if model.mesh.cells.shape[1]!=10:raise ValueError('ten-node P2 tetrahedra required')
        driver=build_driver(model,settings,args.device)
        for _ in range(args.warmup):state,_=driver.step(state,diagnostics=False)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start=perf_counter();counters=[]
        for _ in range(args.steps):
            state,info=driver.step(state,diagnostics=False)
            counters.append((info['flow']['stokes']['iterations'],info['flow']['pressure']['cycles']))
        torch.cuda.synchronize()
        elapsed=perf_counter()-start
        ends[variant]=arrays(state)
        result=dict(ms_per_step=1000*elapsed/args.steps,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            settings=settings,quadrature=driver.transfer.quadrature_summary(),
            stokes_iterations_mean=sum(v[0] for v in counters)/len(counters),
            pressure_cycles_mean=sum(v[1] for v in counters)/len(counters),
            final_solid=model.diagnostics(state.x))
        if args.profile:
            state,result['phases']=profile(driver,state,min(10,args.steps),args.device)
        _,checked=driver.step(state,diagnostics=True)
        result['power_error']=checked['power_error']
        result['last_solver_info']=checked
        results[variant]=result
        print(f"{variant}: {result['ms_per_step']:.3f} ms/step; allocated={result['peak_allocated_bytes']/1024**3:.3f} GiB",flush=True)
        del driver,model,state
        gc.collect();torch.cuda.empty_cache()
    differences={};failures={}
    if 'reference' in ends:
        for variant,end in ends.items():
            if variant=='reference':continue
            differences[variant]={}
            for key,value in end.items():
                expected=ends['reference'][key]
                atol=2e-10 if key=='x' else 2e-4 if key=='pressure' else 2e-6 if key=='force' else 2e-8
                differences[variant][key+'_max_abs']=float((value-expected).abs().max())
                try:
                    torch.testing.assert_close(value,expected,rtol=2e-7,atol=atol)
                except AssertionError as exc:
                    failures[variant+'/'+key]=str(exc)
    report=dict(checkpoint=str(args.checkpoint),warmup=args.warmup,steps=args.steps,
        note='Identical quadrature selection, mass matrix, CN-AB2/xsPPM7, midpoint stages, tolerances and FP64. No simulation files are overwritten.',
        variants=results,equivalence=differences,passed=not failures,equivalence_failures=failures)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('report='+str(path),flush=True)
    if failures:
        raise RuntimeError('P2 execution equivalence failed: '+', '.join(failures))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--device',default='cuda')
    p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--steps',type=int,default=20)
    p.add_argument('--variants',nargs='+',choices=('reference','p2','optimized'),default=['reference','p2','optimized'])
    p.add_argument('--profile',action='store_true')
    p.add_argument('--output',default='results/p2_mac_execution/report.json')
    args=p.parse_args()
    if args.warmup<1 or args.steps<1 or not str(args.device).startswith('cuda'):p.error('positive CUDA warmup/steps required')
    benchmark(args)
