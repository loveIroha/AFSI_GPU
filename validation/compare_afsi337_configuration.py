"""Compare accepted GPU FEM coordinates against native AFSI absolute coordinates.

No Warp, re-solving or interpolation between time frames. Compare common vertices.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import h5py
import numpy as np
from afsi_torch.cycle_checkpoint import load_cycle
from afsi_torch.geometry import ENDO,BASE
if __package__:
    from .compare_afsi337_force import match_coordinates
else:
    from compare_afsi337_force import match_coordinates


def cavity(x,faces,base):
    edges=np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1)
    unique,counts=np.unique(edges,axis=0,return_counts=True);rim=unique[counts==1]
    base_nodes=set(np.unique(base).tolist())
    if not len(rim) or any(int(i) not in base_nodes for i in rim.ravel()):
        raise ValueError('cavity has an unexpected open rim')
    center=x[np.unique(rim)].mean(0)
    total=0.
    # ENDO faces point out of the wall and into the cavity; reverse for cavity orientation.
    for face in faces:
        tri=x[face[[0,2,1]]]-center
        total+=np.dot(tri[0],np.cross(tri[1],tri[2]))/6
    return abs(float(total))  # cap triangles have zero signed volume about center


def compare(checkpoint,native_xdmf,native_dt=5e-5):
    model,state,_,settings,_=load_cycle(checkpoint,'cpu')
    if settings.get('profile')!='afsi337':raise ValueError('AFSI337 FEM checkpoint required')
    native_xdmf=Path(native_xdmf)
    tree=ET.parse(native_xdmf).getroot()
    temporal=tree.find('./Domain/Grid[@Name="solid_coords_io"]')
    if temporal is None:raise ValueError('native absolute coordinate field missing')
    sequence=[(float(g.find('Time').attrib['Value'])+native_dt,g.find('./Attribute/DataItem').text.strip())
              for g in temporal.findall('Grid')]
    time,location=min(sequence,key=lambda item:abs(item[0]-state.time))
    if abs(time-state.time)>1e-10:raise ValueError('no matching accepted native time')
    filename,dataset=location.split(':',1)
    with h5py.File(native_xdmf.parent/filename,'r') as h:
        X=h['/Mesh/Mesh/geometry'][:];cells=h['/Mesh/Mesh/topology'][:];cpu=h[dataset][:]
    reference=model.mesh.X.numpy();ids=match_coordinates(X,reference)
    native_topology=np.unique(np.sort(ids[cells],axis=1),axis=0)
    gpu_topology=np.unique(np.sort(model.mesh.cells[:,:4].numpy(),axis=1),axis=0)
    if not np.array_equal(native_topology,gpu_topology):raise ValueError('reference connectivity differs')
    gpu=state.x.numpy()[ids];difference=gpu-cpu
    volumes=abs(np.linalg.det(X[cells[:,1:]]-X[cells[:,:1]]))/6
    weights=np.zeros(len(X))
    for k in range(4):np.add.at(weights,cells[:,k],volumes/4)
    weights/=weights.sum()
    rms=np.sqrt(np.sum(weights*np.sum(difference**2,axis=1)))
    denom=np.sqrt(np.sum(weights*np.sum((cpu-X)**2,axis=1)))
    inverse=np.full(len(reference),-1,dtype=np.int64);inverse[ids]=np.arange(len(ids))
    endo=inverse[model.mesh.surface(ENDO)[:,:3].numpy()]
    base=inverse[model.mesh.surface(BASE)[:,:3].numpy()]
    if endo.min()<0 or base.min()<0:raise ValueError('native output omits boundary corners')
    cpu_volume=cavity(cpu,endo,base);gpu_volume=cavity(gpu,endo,base)
    return dict(schema=1,scope='Matched accepted time; common P1 vertex coordinates and cavity surfaces',
        time_s=state.time,checkpoint=str(checkpoint),native_xdmf=str(native_xdmf),
        same_reference_connectivity=True,reference_max_abs_cm=float(abs(X-reference[ids]).max()),
        coordinate_rms_mm=float(10*rms),coordinate_max_mm=float(10*np.linalg.norm(difference,axis=1).max()),
        displacement_relative_rms=None if denom<1e-14 else float(rms/denom),
        cpu_cavity_p1_ml=cpu_volume,gpu_cavity_p1_ml=gpu_volume,
        cavity_relative_difference=float((gpu_volume-cpu_volume)/cpu_volume),
        cpu_base_mean_displacement_mm=(10*(cpu-X)[np.unique(base)].mean(0)).tolist(),
        gpu_base_mean_displacement_mm=(10*(gpu-X)[np.unique(base)].mean(0)).tolist(),
        gpu_diagnostics=model.diagnostics(state.x),
        visualization_note='Native solid_coords_io is absolute x; GPU checkpoint stores absolute x; no extra warp.',
        limitation='Native saved solid field is P1 vertex sampling of P2. No convergence threshold or full P2 norm claimed.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--native-xdmf',required=True)
    p.add_argument('--native-dt',type=float,default=5e-5);p.add_argument('--output',required=True)
    a=p.parse_args();r=compare(a.checkpoint,a.native_xdmf,a.native_dt)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
