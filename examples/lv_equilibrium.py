"""Small-load generated LV solid equilibrium, separate from explicit IB/FEM.

Default 0.2 mmHg is a numerical preload smoke test, not a calibrated diastolic
state or an unloaded-reference reconstruction. Load fraction is not time.
"""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import torch
from afsi_torch.geometry import LVConfig,generate_lv
from afsi_torch.lv_model import LVSolid,RampLoads
from afsi_torch.nonlinear import NewtonOptions,GMRESOptions,newton,NonlinearFailure
from afsi_torch.solid_preconditioner import guccione_blocks,block_inverse
from afsi_torch.units import CGS_UNITS
from afsi_torch.preload import save_checkpoint


def run(device='cpu',mesh_size=1.8,pressure_mmhg=.2,tension=0.,load_steps=2,output='results/lv_equilibrium'):
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if not isinstance(load_steps,int) or isinstance(load_steps,bool) or load_steps<1:
        raise ValueError('positive integer load steps required')
    config = LVConfig(mesh_size=mesh_size)
    loads = RampLoads(pressure_mmhg=pressure_mmhg,tension=tension,ramp_time=1.)
    mesh = generate_lv(config,device=device)
    model = LVSolid(mesh,loads=loads)
    options = NewtonOptions(rtol=1e-8,atol=1e-5,linear=GMRESOptions(restart=80,max_iterations=2400,rtol=1e-3))
    x = mesh.X.clone()
    folder = Path(output)
    folder.mkdir(parents=True,exist_ok=True)
    report = dict(device=str(device),torch=torch.__version__,units=CGS_UNITS,
        gpu=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
        solid_config=asdict(config),nodes=len(x),cells=len(mesh.cells),gmsh=mesh.gmsh_version,
        target_pressure_mmhg=pressure_mmhg,target_tension_dyn_per_cm2=tension,
        material=asdict(model.parameters),beta=model.beta,options=asdict(options),
        load_steps=load_steps,history=[],initial=model.diagnostics(x),converged=False,
        production_ib_changed=False,physiological_preload_validated=False,full_cycle_ready=False)
    def save():
        (folder/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    def snapshot(path,y,fraction):
        save_checkpoint(path,model,y,fraction)
    snapshot(folder/'last_converged.npz',x,0.)
    (folder/'failed_load_last_iterate.npz').unlink(missing_ok=True)
    save()
    for step in range(1,load_steps+1):
        fraction = step/load_steps
        print(f'LV equilibrium load {step}/{load_steps}: {fraction*pressure_mmhg:g} mmHg',flush=True)
        fields = replace(model.fields,tension=torch.full_like(model.fields.tension,fraction*tension))
        precondition = lambda y: block_inverse(guccione_blocks(y,model.geometry,fields,model.parameters,
                                                               base=model.base,beta=model.beta))
        try:
            result = newton(lambda y:-model.force(y,fraction),x,validate=model.validate,
                            preconditioner_factory=precondition,options=options)
        except NonlinearFailure as exc:
            report['failure'] = dict(load_fraction=fraction,message=str(exc),history=exc.result.history)
            snapshot(folder/'failed_load_last_iterate.npz',exc.result.x,fraction)
            save()
            raise
        x = result.x
        report['history'].append(dict(load_fraction=fraction,pressure_mmhg=fraction*pressure_mmhg,
            tension_dyn_per_cm2=fraction*tension,iterations=result.iterations,
            residual_norm=result.residual_norm,tolerance=result.tolerance,
            newton_history=result.history,**model.diagnostics(x)))
        snapshot(folder/'last_converged.npz',x,fraction)
        save()
    report['converged'] = True
    report['final'] = model.diagnostics(x)
    save()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--mesh-size',type=float,default=1.8)
    parser.add_argument('--pressure-mmhg',type=float,default=.2)
    parser.add_argument('--tension',type=float,default=0.)
    parser.add_argument('--load-steps',type=int,default=2)
    parser.add_argument('--output',default='results/lv_equilibrium')
    args = parser.parse_args()
    print(json.dumps(run(args.device,args.mesh_size,args.pressure_mmhg,args.tension,args.load_steps,args.output),indent=2))
