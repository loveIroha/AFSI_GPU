"""Optional ParaView output; visualization transfers to CPU only when requested."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
from .fluid.elements import basis, reference_nodes


class CoupledWriter:
    def __init__(self, directory, solid_mesh, fluid_mesh):
        import meshio
        self.meshio = meshio
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.solid_mesh, self.fluid_mesh = solid_mesh, fluid_mesh
        self.frames = []

    @torch.no_grad()
    def write(self, state):
        array = lambda x: x.detach().cpu().numpy()
        sm, fm = self.solid_mesh, self.fluid_mesh
        solid_name, fluid_name = f'solid_{state.step:06d}.vtu', f'fluid_{state.step:06d}.vtu'
        tetra = array(sm.cells)[:, [0, 1, 2, 3, 4, 7, 5, 6, 8, 9]]
        self.meshio.write(self.directory/solid_name, self.meshio.Mesh(array(state.x), [('tetra10', tetra)],
            point_data=dict(displacement_cm=array(state.x-sm.X), next_force_dyn=array(state.force))))
        # Visualize on eight linear sub-hexahedra per Q2 cell; velocities are
        # exact nodal values, rendering between nodes is only trilinear.
        nx, ny, nz = (2*n+1 for n in fm.counts)
        ids = np.arange(nx*ny*nz).reshape(nz, ny, nx)
        hexa = np.stack([ids[:-1, :-1, :-1], ids[:-1, :-1, 1:], ids[:-1, 1:, 1:], ids[:-1, 1:, :-1],
                         ids[1:, :-1, :-1], ids[1:, :-1, 1:], ids[1:, 1:, 1:], ids[1:, 1:, :-1]], -1).reshape(-1, 8)
        q1, _ = basis(reference_nodes(2, device=state.x.device, dtype=state.x.dtype), 1)
        local_p = torch.einsum('qa,ea->eq', q1, state.pressure[fm.pressure_cells])
        indices = fm.velocity_cells.reshape(-1)
        count = state.x.new_zeros(len(fm.velocity_coordinates)).index_add(0, indices, torch.ones_like(local_p).reshape(-1))
        pressure = state.x.new_zeros(len(fm.velocity_coordinates)).index_add(0, indices, local_p.reshape(-1))/count
        self.meshio.write(self.directory/fluid_name, self.meshio.Mesh(array(fm.velocity_coordinates), [('hexahedron', hexa)],
            point_data=dict(velocity_cm_per_s=array(state.velocity), pressure_dyn_per_cm2=array(pressure))))
        self.frames.append((state.time, solid_name, fluid_name))
        for index, name in ((1, 'solid.pvd'), (2, 'fluid.pvd')):
            root = ET.Element('VTKFile', type='Collection', version='0.1', byte_order='LittleEndian')
            collection = ET.SubElement(root, 'Collection')
            for frame in self.frames:
                ET.SubElement(collection, 'DataSet', timestep=str(frame[0]), group='', part='0', file=frame[index])
            ET.ElementTree(root).write(self.directory/name, encoding='utf-8', xml_declaration=True)
