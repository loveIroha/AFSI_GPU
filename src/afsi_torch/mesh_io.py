"""External tetrahedral meshes and DOLFIN cell/facet data, without FEniCS.

Reading and validation happen on CPU. Cell order is never changed: legacy
MeshValueCollection data refer to cell_index and local_entity, not DOF order.
Coordinates are converted explicitly to cm; DG0 directions are not smoothed,
normalized, or converted to nodal fields. No sheet direction is invented.
"""
from dataclasses import dataclass, replace
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
from .fields import ReferenceFields
from .tetrahedron import EDGES, promote_p1


# DOLFIN tetrahedral facet i is opposite local vertex i.
_FACETS = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
_SCALE_CM = {'cm': 1., 'mm': .1, 'm': 100.}


def _name(element):
    return element.tag.rsplit('}', 1)[-1]


def _child(element, name):
    matches = [e for e in element if _name(e) == name]
    if len(matches) != 1:
        raise ValueError(f'expected exactly one {name}')
    return matches[0]


def _data_item(element, directory):
    item = _child(element, 'DataItem')
    shape = tuple(int(n) for n in item.attrib['Dimensions'].split())
    if not shape or any(n <= 0 for n in shape):
        raise ValueError('DataItem dimensions must be positive')
    text = (item.text or '').strip()
    fmt = item.get('Format', 'XML').upper()
    if fmt == 'HDF':
        try:
            import h5py
        except ImportError as exc:
            raise ImportError('XDMF/HDF5 input requires pip install -e ".[mesh]"') from exc
        # Split at the dataset separator, retaining Windows drive letters.
        if ':/' not in text:
            raise ValueError('HDF DataItem must contain filename:/dataset')
        filename, dataset = text.rsplit(':/', 1)
        with h5py.File(directory / filename, 'r') as archive:
            data = np.asarray(archive['/' + dataset])
    elif fmt == 'XML':
        kind = item.get('NumberType', item.get('DataType', 'Float')).lower()
        dtype = np.int64 if kind in ('int', 'uint') else np.float64
        tokens = text.split()
        data = np.asarray(tokens, dtype=dtype)
        if data.size == int(np.prod(shape)):
            data = data.reshape(shape)
    else:
        raise ValueError(f'unsupported DataItem Format={fmt}; use XML or HDF')
    if data.shape != shape:
        raise ValueError('DataItem dimensions do not match stored data')
    return data


def read_xdmf(path, *, grid_name=None):
    """Read one static P1 tetrahedral grid; return raw CPU NumPy (X, cells).

    Supports inline XML and companion HDF5 arrays, including FEniCS XDMF.
    Multiple grids require grid_name; temporal collections and curved/P2
    topology are rejected instead of guessing their layout or node ordering.
    """
    path = Path(path)
    root = ET.parse(path).getroot()
    if _name(root) != 'Xdmf':
        raise ValueError('expected Xdmf document')
    grids = [g for g in root.iter() if _name(g) == 'Grid'
             and any(_name(e) == 'Topology' for e in g)]
    if any(g.get('CollectionType') == 'Temporal' for g in root.iter() if _name(g) == 'Grid'):
        raise ValueError('temporal XDMF is unsupported; export a static reference mesh')
    if grid_name is not None:
        grids = [g for g in grids if g.get('Name') == grid_name]
    if len(grids) != 1:
        raise ValueError('select exactly one static mesh with grid_name')
    topology, geometry = _child(grids[0], 'Topology'), _child(grids[0], 'Geometry')
    if topology.get('TopologyType', '').lower() != 'tetrahedron':
        raise ValueError('only P1 Tetrahedron topology is supported')
    if geometry.get('GeometryType', '').upper() != 'XYZ':
        raise ValueError('expected XYZ geometry')
    X, cells = _data_item(geometry, path.parent), _data_item(topology, path.parent)
    if 'NumberOfElements' in topology.attrib and int(topology.get('NumberOfElements')) != len(cells):
        raise ValueError('Topology NumberOfElements does not match connectivity')
    return X, cells


def read_dolfin_collection(path, *, dimension, cell_count, integer=False):
    """Read a legacy XML MeshValueCollection (also XML stored as .txt).

    dim=2: sparse or complete cell-local facet markers, zero if unspecified.
    dim=3: complete DG0 scalar field, one value per original tetrahedron.
    A global-index MeshFunction or Function vector is deliberately rejected:
    its indices cannot be mapped to this mesh without additional ordering data.
    """
    if dimension not in (2, 3):
        raise ValueError('only tetrahedral facets (dim=2) and cells (dim=3) are supported')
    width = 4 if dimension == 2 else 1
    result = np.zeros((cell_count, width), dtype=np.int64 if integer else np.float64)
    seen = np.zeros(result.shape, dtype=bool)
    collection, declared, count, collections = None, None, 0, 0
    for event, element in ET.iterparse(path, events=('start', 'end')):
        name = _name(element)
        if event == 'start' and name == 'mesh_value_collection':
            collections += 1
            collection = element
            if int(element.get('dim', '-1')) != dimension:
                raise ValueError(f'expected MeshValueCollection dim={dimension}')
            kind = element.get('type')
            if kind not in (('uint', 'int', 'size_t') if integer else ('double', 'float')):
                raise ValueError('MeshValueCollection has an incompatible value type')
            declared = int(element.attrib['size'])
        elif event == 'end' and name == 'value':
            if collection is None:
                raise ValueError('expected cell-local MeshValueCollection, not global entity indices')
            cell = int(element.attrib['cell_index'])
            local = int(element.attrib['local_entity'])
            if not 0 <= cell < cell_count or not 0 <= local < width:
                raise ValueError('cell_index or local_entity is out of range')
            if seen[cell, local]:
                raise ValueError('duplicate cell-local MeshValueCollection entry')
            value = int(element.attrib['value']) if integer else float(element.attrib['value'])
            if not np.isfinite(value) or (integer and value < 0):
                raise ValueError('nonfinite field value or negative boundary marker')
            result[cell, local], seen[cell, local] = value, True
            count += 1
            # The collection owns potentially hundreds of thousands of values.
            # Clear completed children to bound streaming parser memory.
            collection.clear()
    if collections != 1 or declared != count:
        raise ValueError('expected one MeshValueCollection with matching declared size')
    if dimension == 3 and not seen.all():
        raise ValueError('DG0 field must contain exactly one value per cell')
    return result if dimension == 2 else result[:, 0]


def _exterior(X, cells):
    raw = cells[:, _FACETS].reshape(-1, 3)
    _, first, inverse, counts = np.unique(np.sort(raw, axis=1), axis=0,
                                        return_index=True, return_inverse=True, return_counts=True)
    if counts.max() > 2:
        raise ValueError('nonmanifold mesh: more than two tetrahedra share a face')
    keep = counts[inverse] == 1
    indices = np.flatnonzero(keep)
    faces = raw[keep].copy()
    a, b, c = (X[faces[:, i]] for i in range(3))
    inside = X[cells.reshape(-1)[keep]] - a
    flip = (np.cross(b-a, c-a) * inside).sum(axis=1) > 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return faces, indices, keep, first, inverse


@dataclass(frozen=True)
class ImportedSolidMesh:
    """Reference mesh tensors; fiber/sheet are DG0 (E,3), not nodal arrays.

    faces are outward from the solid. boundary_cells/local_facets identify
    their original owner, independently of outward orientation. units are cm.
    to_p2() adds shared midpoint DOFs but preserves cells, tags and DG0 fields.
    """
    X: torch.Tensor
    cells: torch.Tensor
    faces: torch.Tensor
    facet_tags: torch.Tensor
    boundary_cells: torch.Tensor
    boundary_local_facets: torch.Tensor
    fiber: torch.Tensor | None
    sheet: torch.Tensor | None
    vertex_count: int
    metadata: dict

    def to(self, device):
        updates = {name: value.to(device) for name, value in vars(self).items()
                   if isinstance(value, torch.Tensor)}
        return replace(self, **updates)

    def surface(self, tag):
        return self.faces[self.facet_tags == tag]

    def to_p2(self):
        if self.cells.shape[1] == 10:
            return self
        X, cells = promote_p1(self.X, self.cells)
        # Use each original facet owner's new local edge DOFs. Keep boundary
        # row order and orientation, making its tag/owner correspondence exact.
        pairs = self.cells[self.boundary_cells][:, torch.tensor(EDGES, device=self.X.device)]
        vertices = self.faces[:, :3]
        local_edges = torch.tensor([[0, 1], [0, 2], [1, 2]], device=self.X.device)
        desired = vertices[:, local_edges].sort(-1).values
        match = (desired[:, :, None] == pairs.sort(-1).values[:, None]).all(-1)
        edge_local = match.to(torch.int64).argmax(-1)
        mids = cells[self.boundary_cells, 4:].gather(1, edge_local)
        return replace(self, X=X, cells=cells, faces=torch.cat((vertices, mids), dim=1),
                       metadata={**self.metadata, 'displacement_element': 'P2',
                                 'p2_order': 'v0,v1,v2,v3,e01,e02,e03,e12,e13,e23'})

    def reference_fields(self, geometry, *, sheet=None, tension=0.):
        """Evaluate unchanged DG0 directions at this mesh's FE quadrature.

        An explicit sheet (E,3) is required if no sheet files were imported.
        This is a mechanics assembly interface, not an automatic LVSolid/demo
        constructor or checkpoint registration for a patient-specific model.
        """
        if geometry.cells.shape[0] != len(self.cells) or not torch.equal(
                geometry.cells[:, :4], self.cells[:, :4]):
            raise ValueError('quadrature geometry must preserve original cell order')
        if self.cells.shape[1] == 10 and (geometry.node_count != len(self.X)
                or not torch.equal(geometry.cells, self.cells)):
            raise ValueError('quadrature geometry must use this imported P2 mesh')
        if geometry.values.device != self.X.device or geometry.values.dtype != self.X.dtype:
            raise ValueError('quadrature geometry must match mesh device and dtype')
        if self.fiber is None:
            raise ValueError('fiber components have not been imported')
        s = self.sheet if sheet is None else torch.as_tensor(sheet, device=self.X.device, dtype=self.X.dtype)
        if s is None:
            raise ValueError('sheet direction is missing; supply measured data or an explicit rule')
        if s.shape != self.fiber.shape or not torch.isfinite(s).all():
            raise ValueError('sheet must be finite DG0 data with shape (E,3)')
        n = torch.linalg.cross(s, self.fiber)
        scale = torch.linalg.vector_norm(s, dim=-1)*torch.linalg.vector_norm(self.fiber, dim=-1)
        if (torch.linalg.vector_norm(n, dim=-1) <= 100*torch.finfo(self.X.dtype).eps*scale).any():
            raise ValueError('fiber and sheet must be nonzero and nonparallel')
        E, Q = geometry.weights.shape
        t = torch.as_tensor(tension, device=self.X.device, dtype=self.X.dtype)
        if t.ndim == 0:
            t = t.expand(E, Q)
        elif t.shape == (E,):
            t = t[:, None].expand(E, Q)
        else:
            raise ValueError('tension must be scalar or DG0 (E,)')
        if not torch.isfinite(t).all():
            raise ValueError('tension must be finite')
        expand = lambda v: v[:, None, :].expand(E, Q, 3)
        return ReferenceFields(expand(self.fiber), expand(s), expand(n), t)


def read_solid_mesh(path, *, units, boundaries=None, fiber_components=None,
                    sheet_components=None, tag_map=None, translation_cm=(0., 0., 0.),
                    grid_name=None, device='cpu', require_full_boundary=True):
    """Load static P1 XDMF + optional DOLFIN facet and DG0 direction files.

    units must be explicit ('cm', 'mm', 'm'). tag_map is source->target, e.g.
    {1:2, 2:1, 3:3} ONLY after confirming 1=epi, 2=endo, 3=base in the source.
    Input files are read-only. Neither geometry nor directions are regenerated.
    """
    if units not in _SCALE_CM:
        raise ValueError('explicit source units must be cm, mm, or m')
    X, cells = read_xdmf(path, grid_name=grid_name)
    if X.ndim != 2 or X.shape[1] != 3 or not len(X) or not np.isfinite(X).all():
        raise ValueError('expected finite coordinates with shape (N,3)')
    if (cells.ndim != 2 or cells.shape[1] != 4 or not len(cells)
            or cells.dtype.kind not in 'iu' or cells.min() < 0 or cells.max() >= len(X)):
        raise ValueError('expected integer tetrahedra with valid shape (E,4) and vertex indices')
    cells = np.asarray(cells, dtype=np.int64)
    keys = np.sort(cells, axis=1)
    if np.any(np.diff(keys, axis=1) == 0) or len(np.unique(keys, axis=0)) != len(cells):
        raise ValueError('repeated vertices or duplicate tetrahedra')
    shift = np.asarray(translation_cm, dtype=np.float64)
    if shift.shape != (3,) or not np.isfinite(shift).all():
        raise ValueError('translation_cm must contain three finite coordinates')
    X = np.asarray(X, dtype=np.float64)*_SCALE_CM[units] + shift
    singular = np.linalg.svd(np.transpose(X[cells[:, 1:]]-X[cells[:, :1]], (0, 2, 1)), compute_uv=False)
    if (singular[:, -1] <= 100*np.finfo(np.float64).eps*singular[:, 0]).any():
        raise ValueError('degenerate or ill-conditioned reference tetrahedra')
    faces, indices, exterior, first, inverse = _exterior(X, cells)
    tags = np.zeros(len(faces), dtype=np.int64)
    if boundaries is not None:
        markers = read_dolfin_collection(boundaries, dimension=2, cell_count=len(cells), integer=True).ravel()
        if np.any(markers != markers[first[inverse]]):
            raise ValueError('conflicting tags on the same shared facet')
        if np.any(markers[~exterior] != 0):
            raise ValueError('nonzero boundary markers occur on interior facets')
        tags = markers[indices]
        if require_full_boundary and np.any(tags == 0):
            raise ValueError('boundary markers must cover the complete exterior')
    elif tag_map is not None:
        raise ValueError('tag_map requires a boundary file')
    source_tags = tags.copy()
    if tag_map is not None:
        if any(type(k) is not int or type(v) is not int or v <= 0 for k, v in tag_map.items()):
            raise ValueError('tag_map must map integer source tags to positive integer target tags')
        labels = set(tags.tolist()) - {0}
        if not labels <= set(tag_map):
            raise ValueError('tag_map must include every nonzero source boundary tag')
        tags = np.array([tag_map[int(t)] if t else 0 for t in tags], dtype=np.int64)

    def directions(paths):
        if paths is None:
            return None
        if isinstance(paths, (str, Path)) or len(paths) != 3:
            raise ValueError('directions require exactly three component files')
        data = np.stack([read_dolfin_collection(p, dimension=3, cell_count=len(cells)) for p in paths], axis=1)
        if (np.linalg.norm(data, axis=1) <= np.finfo(np.float64).tiny).any():
            raise ValueError('zero fiber/sheet direction in source data')
        return torch.from_numpy(data)

    f, s = directions(fiber_components), directions(sheet_components)
    metadata = dict(source=str(Path(path).resolve()), source_units=units, units='cm',
                    scale_to_cm=_SCALE_CM[units], translation_cm=shift.tolist(),
                    source_vertex_count=len(X), source_cell_count=len(cells),
                    displacement_element='P1', direction_location='cell-DG0',
                    directions_modified=False, cell_order_preserved=True,
                    source_boundary_counts={str(k): int((source_tags == k).sum()) for k in np.unique(source_tags)},
                    tag_map=None if tag_map is None else {str(k): v for k, v in tag_map.items()},
                    boundaries=None if boundaries is None else str(Path(boundaries).resolve()),
                    fiber_sources=None if fiber_components is None else [str(Path(p).resolve()) for p in fiber_components],
                    sheet_sources=None if sheet_components is None else [str(Path(p).resolve()) for p in sheet_components])
    mesh = ImportedSolidMesh(torch.from_numpy(X), torch.from_numpy(cells), torch.from_numpy(faces),
                             torch.from_numpy(tags), torch.from_numpy(indices//4),
                             torch.from_numpy(indices%4), f, s, len(X), metadata)
    return mesh.to(device)
