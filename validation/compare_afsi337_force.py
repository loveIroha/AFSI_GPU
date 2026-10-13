"""Compare GPU assembled solid forces against the actual native demo material."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from afsi_torch.afsi337_io import load_native_solid
if __package__:
    from .compare_afsi_native_step import _metrics
else:
    from compare_afsi_native_step import _metrics


def match_coordinates(query,candidates):
    """Same native export uses identical DOFs; hash a permutation without N² memory."""
    if (query.ndim!=2 or candidates.ndim!=2 or query.shape[1:]!=(3,) or candidates.shape[1:]!=(3,)
            or not np.isfinite(query).all() or not np.isfinite(candidates).all()):
        raise ValueError('finite 3D coordinate arrays required')
    keys=[tuple(v) for v in np.round(candidates,10)]
    lookup={key:i for i,key in enumerate(keys)}
    if len(lookup)!=len(keys):raise ValueError('ambiguous native coordinate keys')
    try:index=np.array([lookup[tuple(v)] for v in np.round(query,10)],dtype=np.int64)
    except KeyError as exc:raise ValueError('native coordinate missing') from exc
    if len(np.unique(index))!=len(index) or not np.allclose(query,candidates[index],rtol=0,atol=1e-11):
        raise ValueError('native coordinate permutation failed verification')
    return index


@torch.no_grad()
def compare(solid_input,reference,device='cuda'):
    model=load_native_solid(solid_input,device=device)
    with np.load(reference,allow_pickle=False) as a:
        data={k:a[k] for k in a.files}
    meta=json.loads(str(data['metadata']))
    if meta.get('producer')!='afsi337-native-force' or meta.get('schema')!=1:
        raise ValueError('native force reference required')
    index=match_coordinates(model.mesh.X.cpu().numpy(),data['reference_coordinates'])
    cast=lambda a:torch.as_tensor(a,device=device,dtype=torch.float64)
    fibers={name:_metrics(getattr(model.fibers,name),cast(data[name][index]),rtol=1e-12,atol=1e-12)
            for name in ('fiber','sheet')}
    results=[]
    for t,x,force in zip(data['time'],data['coordinates'],data['force']):
        current=cast(x[index]);model.validate(current)
        results.append(dict(time_s=float(t),**_metrics(model.force(current,float(t)),cast(force[index]),rtol=2e-9,atol=2e-8)))
    return dict(passed=all(x['passed'] for x in [*fibers.values(),*results]),device=str(device),
        native=meta,solid_input_sha256=hashlib.sha256(Path(solid_input).read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256(Path(reference).read_bytes()).hexdigest(),fibers=fibers,states=results)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--solid-input',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--output',required=True)
    a=p.parse_args();r=compare(a.solid_input,a.reference,a.device)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
    if not r['passed']:raise SystemExit('native solid force comparison failed')
