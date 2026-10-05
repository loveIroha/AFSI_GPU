"""Optional FP64 ParaView frames for the generated 3D FE/MAC demo."""
import base64
from pathlib import Path
import xml.etree.ElementTree as ET
import zlib
import numpy as np
import torch
from ..cycle_checkpoint import atomic_json
from ..solid import deformation_gradient
from ..mechanics import determinant3


def _xml(path, root):
    temporary = path.with_suffix(path.suffix+'.tmp')
    ET.ElementTree(root).write(temporary, encoding='utf-8', xml_declaration=True)
    temporary.replace(path)


def _binary_array(parent, name, values, components=1):
    """VTK zlib block header and payload, each separately base64 encoded."""
    raw = np.asarray(values, dtype='<f8').tobytes()
    block_size = 32768
    blocks = [zlib.compress(raw[i:i+block_size], level=1)
              for i in range(0, len(raw), block_size)]
    header = np.asarray([len(blocks), block_size,
                         len(raw)-(len(blocks)-1)*block_size,
                         *map(len, blocks)], dtype='<u8')
    node = ET.SubElement(parent, 'DataArray', type='Float64', Name=name,
                         NumberOfComponents=str(components),
                         NumberOfTuples=str(np.asarray(values).size//components), format='binary')
    node.text = (base64.b64encode(header.tobytes()).decode('ascii')
                 + base64.b64encode(b''.join(blocks)).decode('ascii'))


class MACWriter:
    def __init__(self, directory, model, grid, *, resume_time=None):
        try:
            import meshio
        except ImportError as exc:
            raise ImportError('VTK output requires: pip install -e ".[geometry]"') from exc
        self.meshio, self.model, self.grid = meshio, model, grid
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        array = lambda t: t.detach().cpu().numpy()
        self.reference = array(model.mesh.X).copy()
        self.p1 = model.mesh.cells.shape[1] == 4
        self.cells = (array(model.mesh.cells) if self.p1 else
                      array(model.mesh.cells)[:, [0,1,2,3,4,7,5,6,8,9]]).copy()
        self.fiber = array(model.mesh.fiber if self.p1 else model.fibers.fiber).copy()
        self.sheet = array(model.mesh.sheet).copy() if self.p1 else None
        self.frames = []
        if resume_time is not None:
            collections = []
            for kind, extension in (('solid', '.vtu'), ('fluid', '.vti')):
                path = self.directory/(kind+'.pvd')
                entries = {}
                if path.exists():
                    for entry in ET.parse(path).getroot().iter('DataSet'):
                        time, filename = float(entry.attrib['timestep']), entry.attrib['file']
                        if (time <= resume_time and Path(filename).name == filename
                                and filename.startswith(kind+'_') and filename.endswith(extension)
                                and (self.directory/filename).is_file()):
                            entries[time] = filename
                collections.append(entries)
            for time in sorted(collections[0].keys() & collections[1].keys()):
                self.frames.append((time, collections[0][time], collections[1][time]))
            self._collections()
        atomic_json(self.directory/'fields.json', dict(
            units='cm-g-s', solid_geometry=f'deformed {"P1" if self.p1 else "P2"} tetrahedra; no Warp By Vector needed',
            displacement_cm='current minus reference coordinates',
            nodal_force_dyn='integrated FE nodal force, sampled at force_time_s',
            fiber_reference='reference DG0 cell fiber' if self.p1 else 'reference nodal fiber, not a deformed direction',
            sheet_reference='reference DG0 cell sheet' if self.p1 else 'not exported',
            J_min='minimum det(F) over quadrature points within each cell',
            J_max='maximum det(F) over quadrature points within each cell',
            pressure_dyn_per_cm2='original MAC cell pressure; zero-mean gauge',
            velocity_cm_per_s='visualization only: arithmetic average of opposite MAC faces',
            divergence_per_s='divergence of original staggered velocities',
            fluid_storage='VTK ImageData cell fields; x varies fastest; FP64 zlib binary',
            mesh='fixed connectivity; solid points change with time',
        ))

    def _collections(self):
        for index, kind in ((1, 'solid'), (2, 'fluid')):
            root = ET.Element('VTKFile', type='Collection', version='0.1', byte_order='LittleEndian')
            collection = ET.SubElement(root, 'Collection')
            for frame in self.frames:
                ET.SubElement(collection, 'DataSet', timestep=repr(frame[0]),
                              group='', part='0', file=frame[index])
            _xml(self.directory/(kind+'.pvd'), root)

    def _fluid(self, path, state):
        velocity = tuple(u.detach().cpu().numpy() for u in state.velocity)
        centered = np.stack([.5*(np.take(u, range(u.shape[c]-1), axis=c)
                                 + np.take(u, range(1,u.shape[c]), axis=c))
                             for c,u in enumerate(velocity)], axis=-1)
        div = sum(np.diff(u, axis=c)/self.grid.spacing[c] for c,u in enumerate(velocity))
        extent = ' '.join(f'0 {n}' for n in self.grid.shape)
        root = ET.Element('VTKFile', type='ImageData', version='1.0',
                          byte_order='LittleEndian', header_type='UInt64',
                          compressor='vtkZLibDataCompressor')
        image = ET.SubElement(root, 'ImageData', WholeExtent=extent,
                             Origin=' '.join(map(str,self.grid.origin)),
                             Spacing=' '.join(map(str,self.grid.spacing)))
        fields = ET.SubElement(image, 'FieldData')
        _binary_array(fields, 'TimeValue', [state.time])
        # -1 denotes the initial zero force before its first load evaluation.
        _binary_array(fields, 'force_time_s', [-1. if state.force_time is None else state.force_time])
        if state.pressure_time is not None:
            _binary_array(fields, 'pressure_time_s', [state.pressure_time])
        piece = ET.SubElement(image, 'Piece', Extent=extent)
        ET.SubElement(piece, 'PointData')
        data = ET.SubElement(piece, 'CellData', Scalars='pressure_dyn_per_cm2', Vectors='velocity_cm_per_s')
        # Solver storage is x,y,z with z contiguous; VTK expects x contiguous.
        _binary_array(data, 'pressure_dyn_per_cm2', state.pressure.detach().cpu().numpy().transpose(2,1,0))
        _binary_array(data, 'velocity_cm_per_s', centered.transpose(2,1,0,3), 3)
        _binary_array(data, 'divergence_per_s', div.transpose(2,1,0))
        _xml(path, root)

    @torch.no_grad()
    def write(self, state):
        if any(time == state.time for time,_,_ in self.frames):
            return
        array = lambda t: t.detach().cpu().numpy()
        solid_name, fluid_name = f'solid_{state.step:06d}.vtu', f'fluid_{state.step:06d}.vti'
        solid_path = self.directory/solid_name
        temporary = solid_path.with_suffix('.vtu.tmp')
        J = (determinant3(self.model.element_gradient(state.x))[:, None] if self.p1 else
             determinant3(deformation_gradient(state.x,self.model.geometry)))
        x = array(state.x)
        point_data = dict(displacement_cm=x-self.reference, nodal_force_dyn=array(state.force))
        cell_data = dict(J_min=[array(J.amin(1))], J_max=[array(J.amax(1))])
        if self.p1:
            cell_data.update(fiber_reference=[self.fiber], sheet_reference=[self.sheet])
        else:
            point_data['fiber_reference'] = self.fiber
        self.meshio.write(temporary, self.meshio.Mesh(x, [('tetra' if self.p1 else 'tetra10',self.cells)],
            point_data=point_data, cell_data=cell_data),
            file_format='vtu', binary=True, compression='zlib')
        temporary.replace(solid_path)
        self._fluid(self.directory/fluid_name,state)
        # Publish the pair only after both files are complete.
        self.frames = [frame for frame in self.frames if frame[0] < state.time]
        self.frames.append((state.time, solid_name, fluid_name))
        self._collections()

    def summary(self):
        return dict(enabled=True,format='VTK', directory='vtk', frames=len(self.frames),
                    first_time_s=self.frames[0][0] if self.frames else None,
                    last_time_s=self.frames[-1][0] if self.frames else None,
                    solid_collection='vtk/solid.pvd', fluid_collection='vtk/fluid.pvd')
