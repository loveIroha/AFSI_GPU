"""Compare frozen IB matrices before running the nonlinear GPU benchmark.

Read-only checkpoint diagnostic: no time advance or checkpoint overwrite.
Checks the complete sparse matrices without allocating a dense FE/grid matrix.
"""
import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import sys
import traceback
from math import isfinite
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from afsi_torch.paper_lv_checkpoint import load
from afsi_torch.mac.paper_coupling import BEIBStepper
from afsi_torch.mac.adaptive_transfer import AdaptiveP1Transfer
from afsi_torch.cycle_checkpoint import atomic_json


def json_safe(value):
    # The most useful failed report may contain NaN/Inf diagnostics; strict
    # JSON output must still succeed so that the original failure is retained.
    if isinstance(value,float) and not isfinite(value):
        return str(value)
    if isinstance(value,dict):
        return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [json_safe(v) for v in value]
    return value


def field_summary(fields):
    return [dict(finite=bool(torch.isfinite(v).all().item()),
                 max_abs=float(v.abs().max().item()),
                 norm=float(torch.linalg.vector_norm(v).item())) for v in fields]


def compare_matrices(actual,expected):
    same = (actual.shape==expected.shape
            and torch.equal(actual.crow_indices(),expected.crow_indices())
            and torch.equal(actual.col_indices(),expected.col_indices()))
    finite = bool(torch.isfinite(actual.values()).all().item())
    result = dict(nnz=actual.values().numel(),reference_nnz=expected.values().numel(),
                  identical_structure=same,finite=finite)
    if same and finite:
        diff = actual.values()-expected.values()
        scale = torch.linalg.vector_norm(expected.values())
        result.update(max_abs=diff.abs().max().item() if len(diff) else 0.,
            relative_l2=(torch.linalg.vector_norm(diff)/scale.clamp_min(torch.finfo(diff.dtype).tiny)).item(),
            within_tolerance=bool(torch.allclose(actual.values(),expected.values(),
                rtol=2e-5 if diff.dtype==torch.float32 else 2e-11,
                atol=2e-6 if diff.dtype==torch.float32 else 2e-12)))
    else:
        result['within_tolerance'] = False
    return result


@torch.no_grad()
def diagnose(checkpoint,device):
    model,state,config,_ = load(checkpoint,device)
    config = replace(config,ib_response_backend='csr',ib_csr_assembly_backend='coalesce',
                     ib_csr_contraction_backend='sites')
    driver = BEIBStepper(model,config,device)
    sync = lambda: torch.cuda.synchronize(device) if state.x.is_cuda else None
    report = dict(checkpoint=str(checkpoint),time_s=state.time,device=str(device),
                  dtype=str(state.x.dtype),status='running',stage='quadrature')
    try:
        # Evaluate the same quadrature/stencil once; both builders consume it.
        raw = AdaptiveP1Transfer.prepare(driver.transfer,state.x)
        report['stage'] = 'coalesce_assembly'
        print('assembling coalesce reference...',flush=True)
        sync(); start = perf_counter()
        reference = driver.transfer.assemble_stencil(raw)
        sync(); report['coalesce_seconds'] = perf_counter()-start
        report['coalesce_assembly'] = reference.assembly
        driver.transfer.assembly_backend = 'hash'
        report['stage'] = 'hash_assembly'
        print('assembling hash with the same frozen quadrature...',flush=True)
        sync(); start = perf_counter()
        hashed = driver.transfer.assemble_stencil(raw)
        sync(); report['hash_seconds'] = perf_counter()-start
        report['hash_assembly'] = hashed.assembly
        report['stage'] = 'matrix_comparison'
        report['matrices'] = [compare_matrices(a,b) for a,b in
            zip(hashed.gather+hashed.spread,reference.gather+reference.spread)]
        if not all(v['within_tolerance'] for v in report['matrices']):
            raise RuntimeError('hash/coalesce matrix comparison failed before the fluid solve')
        print('all six CSR matrices agree; checking force propagation...',flush=True)
        force = driver.solid.force(state.x,(state.step+1)*config.time.dt)
        report['force'] = field_summary((force,))
        outputs = []
        for name,stencil in (('coalesce',reference),('hash',hashed)):
            report['stage'] = name+'_spread'
            driver.transfer.reset_warm_start()
            density,_ = driver.transfer.spread(force,stencil)
            summary = field_summary(density)
            report[name+'_density'] = summary
            if not all(v['finite'] for v in summary):
                raise FloatingPointError(name+' produced nonfinite spread density')
            outputs.append(density)
        report['density_max_abs_difference'] = max((a-b).abs().max().item()
            for a,b in zip(*outputs))
        report['stage'] = 'fluid_solves'
        departure = driver.flow.advect(state.velocity)
        fields = []
        for name,density in zip(('coalesce','hash'),outputs):
            report['stage'] = name+'_fluid_solve'
            rhs = tuple(a+driver.flow.dt/driver.flow.rho*f for a,f in zip(departure,density))
            report[name+'_rhs'] = field_summary(rhs)
            # Inspect raw data independently of the compiled residual kernel.
            report[name+'_initial_helmholtz_metrics'] = driver.flow._measure(rhs,rhs).tolist()
            velocity,pressure,info = driver.flow.solve_rhs(rhs,state.pressure)
            fields.append((tuple(v.clone() for v in velocity),pressure.clone()))
            report[name+'_flow'] = info
        report['velocity_max_abs_difference'] = max((a-b).abs().max().item()
            for a,b in zip(fields[0][0],fields[1][0]))
        report['pressure_max_abs_difference'] = (fields[0][1]-fields[1][1]).abs().max().item()
        report.update(status='passed',stage='complete')
    except (ValueError,RuntimeError,FloatingPointError) as exc:
        report.update(status='failed',failure_type=type(exc).__name__,failure=str(exc),
                      traceback=traceback.format_exc())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--output',default='results/paper_lv_hash_diagnosis/report.json')
    args = parser.parse_args()
    report = diagnose(args.checkpoint,args.device)
    atomic_json(args.output,json_safe(report))
    print(f"{report['status']}: stage={report['stage']}; report={args.output}",flush=True)
    if report['status']!='passed':
        print(report['failure'],flush=True)
        raise SystemExit(1)


if __name__=='__main__':
    main()
