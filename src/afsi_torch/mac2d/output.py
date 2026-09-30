"""VTU/PVD output at sampled times; no force or velocity solve for output."""
from pathlib import Path
import numpy as np
import meshio


def fields(folder,solid,state,grid,*,fluid=False):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    array=lambda t:t.detach().cpu().numpy()
    pad=lambda v:np.column_stack((v,np.zeros(len(v))))
    name=f'solid_{state.step:06d}.vtu'
    meshio.write_points_cells(folder/name,pad(array(state.x)),
        [('triangle6',array(solid.mesh.cells)[:,[0,1,2,3,5,4]])],
        point_data={'displacement_cm':pad(array(state.x-solid.mesh.X)),
                    'integrated_force_per_thickness':pad(array(state.force))},
        cell_data={'leaflet_marker':[array(solid.mesh.cell_tags)]},binary=True)
    fluid_name=None
    if fluid:
        nx,ny=grid.shape
        hx,hy=grid.spacing
        x,y=np.meshgrid(hx*np.arange(nx+1),hy*np.arange(ny+1),indexing='ij')
        points=np.column_stack((x.reshape(-1),y.reshape(-1),np.zeros(x.size)))
        ids=np.arange((nx+1)*(ny+1)).reshape(nx+1,ny+1)
        cells=np.stack((ids[:-1,:-1],ids[1:,:-1],ids[1:,1:],ids[:-1,1:]),-1).reshape(-1,4)
        u,v=state.velocity
        centered=np.stack((array(.5*(u[1:]+u[:-1])).reshape(-1),array(.5*(v[:,1:]+v[:,:-1])).reshape(-1)),1)
        fluid_name=f'fluid_{state.step:06d}.vtu'
        meshio.write_points_cells(folder/fluid_name,points,[('quad',cells)],
            cell_data={'velocity_cm_per_s':[pad(centered)],'pressure_dyn_per_cm2':[array(state.pressure).reshape(-1)]},binary=True)
    return dict(step=state.step,time=state.time,solid=name,fluid=fluid_name)


def collection(folder,frames):
    for kind in ('solid','fluid'):
        entries=[f'<DataSet timestep="{r["time"]:.16g}" group="" part="0" file="{r[kind]}"/>' for r in frames if r.get(kind)]
        if entries:
            path=Path(folder)/(kind+'.pvd')
            path.write_text('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n<Collection>\n'+
                            '\n'.join(entries)+'\n</Collection>\n</VTKFile>\n',encoding='utf-8')
