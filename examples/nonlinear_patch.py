"""Actual P2 nonlinear equilibrium: analytic affine traction and follower load.

Centimetre/gram/second units. These are solid equilibrium tests, not IB steps.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from afsi_torch import solid, boundary as bd
from afsi_torch.tetrahedron import promote_p1
from afsi_torch.fields import prepare_reference_fields
from afsi_torch.materials import GuccioneParameters, guccione_pk1
from afsi_torch.nonlinear import newton, NewtonOptions, GMRESOptions, NonlinearFailure
from afsi_torch.solid_preconditioner import guccione_blocks, block_inverse
from afsi_torch.mechanics import determinant3
from afsi_torch.units import CGS_UNITS


def setup(device):
    vertices = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.],
                             [0.,0.,1.],[1.,0.,1.],[0.,1.,1.],[1.,1.,1.]],device=device,dtype=torch.float64)
    cells = torch.tensor([[0,1,3,7],[0,3,2,7],[0,2,6,7],[0,6,4,7],[0,4,5,7],[0,5,1,7]],device=device)
    X,cells = promote_p1(vertices,cells)
    geo = solid.prepare_p2(X,cells)
    faces = bd.extract_boundary(X,cells)
    surface = bd.prepare_surface(X,faces)
    loaded = bd.prepare_surface(X,faces[(X[faces[:,:3],0]>.999).all(-1)])
    fixed = (X[:,0]<1e-12)[:,None].expand_as(X)
    fiber = X.new_tensor([1.,0.,0.]).expand_as(X)
    sheet = X.new_tensor([0.,1.,0.]).expand_as(X)
    fields = prepare_reference_fields(geo,fiber,sheet,0.)
    return X,cells,geo,surface,loaded,fixed,fields


def run(device='cpu',output='results/nonlinear_patch'):
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    X,cells,geo,surface,loaded,fixed,fields = setup(device)
    parameters = GuccioneParameters()
    options = NewtonOptions(atol=1e-7,rtol=1e-9,linear=GMRESOptions(rtol=1e-5,restart=70,max_iterations=700))
    # Analytic constant PK1 traction integrated on reference triangles. This
    # external force is NOT manufactured by negating the assembled nodal force.
    F = X.new_tensor([[1.08,0.,0.],[.025,1.,0.],[0.,0.,1.]])
    target = X@F.T
    P = guccione_pk1(F,X.new_tensor([1.,0.,0.]),X.new_tensor([0.,1.,0.]),X.new_tensor([0.,0.,1.]),parameters)
    # P2 triangle vertex basis integrates to zero; edge basis to area/3.
    traction_area = .5*surface.reference_area_vectors@P.T
    local = X.new_zeros((len(surface.faces),6,3))
    local[:,3:] = traction_area[:,None,:]/3
    dead = torch.zeros_like(X).index_add(0,surface.faces.reshape(-1),local.reshape(-1,3))
    internal = lambda y: solid.guccione_force(y,geo,fields,parameters)
    validate = lambda y: solid.validate_deformation(y,geo)
    precondition = lambda y: block_inverse(guccione_blocks(y,geo,fields,parameters))
    report = dict(device=str(device),torch=torch.__version__,units=CGS_UNITS,options=asdict(options),
                  nodes=len(X),cells=len(cells),material=asdict(parameters),cases={},converged=False,
                  production_ib_changed=False,full_cycle_ready=False)
    folder = Path(output)
    folder.mkdir(parents=True,exist_ok=True)
    def save():
        (folder/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for name,force in [('affine',lambda y:internal(y)+dead),
                       ('follower',lambda y:internal(y)+bd.pressure_force(y,loaded,1000.))]:
        residual = lambda y: -force(y)
        try:
            solved = newton(residual,X,validate=validate,fixed=fixed,values=X,
                            preconditioner_factory=precondition,options=options)
        except NonlinearFailure as exc:
            solved = exc.result
            report['cases'][name] = dict(converged=False,error=str(exc),history=solved.history)
            np.savez_compressed(folder/(name+'_last_accepted.npz'),X=X.cpu().numpy(),x=solved.x.cpu().numpy(),cells=cells.cpu().numpy())
            save()
            raise
        entry = dict(converged=solved.converged,iterations=solved.iterations,residual_norm=solved.residual_norm,
            tolerance=solved.tolerance,history=solved.history,
            minimum_detF=determinant3(solid.deformation_gradient(solved.x,geo)).min().item(),
            max_displacement_cm=torch.linalg.vector_norm(solved.x-X,dim=-1).max().item())
        if name == 'affine':
            entry['analytic_position_max_error_cm'] = (solved.x-target).abs().max().item()
            if entry['analytic_position_max_error_cm'] > 1e-8:
                raise RuntimeError('analytic equilibrium mismatch')
        report['cases'][name] = entry
        np.savez_compressed(folder/(name+'.npz'),X=X.cpu().numpy(),x=solved.x.cpu().numpy(),cells=cells.cpu().numpy())
        save()
    report['converged'] = True
    save()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',default='results/nonlinear_patch')
    args = parser.parse_args()
    print(json.dumps(run(args.device,args.output),indent=2))
