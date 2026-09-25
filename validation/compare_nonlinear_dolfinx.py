"""Compare converged Newton--GMRES positions with independent UFL equilibrium."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from afsi_torch import solid,boundary as bd
from afsi_torch.fields import prepare_reference_fields
from afsi_torch.materials import GuccioneParameters
from afsi_torch.nonlinear import newton,NewtonOptions,GMRESOptions
from afsi_torch.solid_preconditioner import guccione_blocks,block_inverse


def compare(path,device='cpu'):
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    with np.load(path,allow_pickle=False) as data:
        metadata=json.loads(str(data['metadata']))
        X=torch.as_tensor(data['X'],device=device,dtype=torch.float64)
        cells=torch.as_tensor(data['cells'],device=device,dtype=torch.int64)
        geometry=solid.prepare_p2(X,cells,quadrature=(data['volume_points'],data['volume_weights']))
        faces=bd.extract_boundary(X,cells)
        surface=bd.prepare_surface(X,faces[(X[faces[:,:3],0]>.999).all(-1)],
                                  quadrature=(data['surface_points'],data['surface_weights']))
        fields=prepare_reference_fields(geometry,[1.,0.,0.],[0.,1.,0.],0.)
        parameters=GuccioneParameters(**metadata['parameters'])
        residual=lambda x:-solid.guccione_force(x,geometry,fields,parameters)-bd.pressure_force(x,surface,metadata['pressure_dyn_per_cm2'])
        fixed=(X[:,0]<1e-12)[:,None].expand_as(X)
        result=newton(residual,X,validate=lambda x:solid.validate_deformation(x,geometry),fixed=fixed,
            preconditioner_factory=lambda x:block_inverse(guccione_blocks(x,geometry,fields,parameters)),
            options=NewtonOptions(rtol=1e-10,atol=1e-8,linear=GMRESOptions(rtol=1e-5,restart=70)))
        actual=result.x.cpu().numpy()
        np.testing.assert_allclose(actual,data['x'],atol=2e-9,rtol=1e-9)
        return dict(status='passed',device=str(device),torch=torch.__version__,reference=metadata,
            reference_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            position_max_abs_cm=float(np.abs(actual-data['x']).max()),
            residual_norm=result.residual_norm,tolerance=result.tolerance,history=result.history)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',default='validation/results/nonlinear.npz')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',default='validation/results/nonlinear.json')
    args=parser.parse_args()
    report=compare(args.reference,args.device)
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
