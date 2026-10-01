"""Cell-local data alignment, outward tags, DG0 persistence and FE assembly."""
from pathlib import Path
import numpy as np
import pytest
import torch
from afsi_torch.mesh_io import read_solid_mesh, read_dolfin_collection, read_xdmf
from afsi_torch import solid
from afsi_torch.materials import GuccioneParameters

X = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., -1.]])
# Opposite local orientations and deliberately non-sorted vertex order.
CELLS = np.array([[2, 0, 1, 3], [1, 2, 0, 4]])


def xdmf(tmp, coords=X, cells=CELLS, hdf=False):
    if hdf:
        h5py = pytest.importorskip('h5py')
        with h5py.File(tmp/'mesh.h5', 'w') as f:
            f['/Mesh/mesh/geometry'] = coords
            f['/Mesh/mesh/topology'] = cells
        data_X, data_C, fmt = 'mesh.h5:/Mesh/mesh/geometry', 'mesh.h5:/Mesh/mesh/topology', 'HDF'
    else:
        data_X = ' '.join(str(v) for v in coords.ravel())
        data_C = ' '.join(str(v) for v in cells.ravel())
        fmt = 'XML'
    p = tmp/'mesh.xdmf'
    p.write_text(f'<Xdmf Version="3.0"><Domain><Grid Name="mesh" GridType="Uniform">'
                 f'<Topology TopologyType="Tetrahedron" NumberOfElements="{len(cells)}">'
                 f'<DataItem Dimensions="{len(cells)} 4" NumberType="Int" Format="{fmt}">{data_C}</DataItem></Topology>'
                 f'<Geometry GeometryType="XYZ"><DataItem Dimensions="{len(coords)} 3" Format="{fmt}">{data_X}</DataItem>'
                 '</Geometry></Grid></Domain></Xdmf>')
    return p


def collection(path, dim, rows, integer=False, size=None):
    values = ''.join(f'<value cell_index="{c}" local_entity="{l}" value="{v}" />' for c, l, v in rows)
    path.write_text(f'<dolfin><mesh_function><mesh_value_collection name="f" type="{"uint" if integer else "double"}"'
                    f' dim="{dim}" size="{len(rows) if size is None else size}">{values}'
                    '</mesh_value_collection></mesh_function></dolfin>')
    return path


def inputs(tmp, hdf=False):
    mesh = xdmf(tmp, hdf=hdf)
    # Local facet 3 is shared and unmarked; exterior source tags are arbitrary.
    boundary = collection(tmp/'boundary.txt', 2,
                          [(c, l, 10*(l+1) if l != 3 else 0) for c in range(2) for l in range(4)], True)
    fiber = [collection(tmp/f'f{i}.xml', 3, [(1, 0, v[1]), (0, 0, v[0])])
             for i, v in enumerate([[1., 0.], [0., 1.], [0., 0.]])]
    return mesh, boundary, fiber


@pytest.mark.parametrize('hdf', [False, True])
def test_raw_order_boundary_mapping_and_units(tmp_path, hdf):
    path, boundary, fiber = inputs(tmp_path, hdf)
    mesh = read_solid_mesh(path, units='mm', boundaries=boundary, fiber_components=fiber,
                           tag_map={10: 2, 20: 1, 30: 3}, translation_cm=(2., 3., 4.))
    np.testing.assert_array_equal(mesh.cells, CELLS)
    np.testing.assert_allclose(mesh.X, X*.1+[2., 3., 4.])
    assert mesh.facet_tags.tolist() == [2, 1, 3, 2, 1, 3]
    np.testing.assert_array_equal(mesh.fiber, [[1., 0., 0.], [0., 1., 0.]])
    assert mesh.sheet is None and mesh.metadata['directions_modified'] is False
    for face, owner, local in zip(mesh.faces, mesh.boundary_cells, mesh.boundary_local_facets):
        a, b, c = mesh.X[face]
        assert torch.dot(torch.linalg.cross(b-a, c-a), mesh.X[mesh.cells[owner, local]]-a) < 0


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_p2_dg0_quadrature_and_energy_gradient(tmp_path, device):
    path, boundary, fiber = inputs(tmp_path)
    mesh = read_solid_mesh(path, units='cm', boundaries=boundary, fiber_components=fiber).to_p2().to(device)
    assert len(mesh.X) == 14 and mesh.to_p2() is mesh
    np.testing.assert_array_equal(mesh.cells[:, :4].cpu(), CELLS)
    geom = solid.prepare_p2(mesh.X, mesh.cells)
    with pytest.raises(ValueError, match='sheet direction is missing'):
        mesh.reference_fields(geom)
    fields = mesh.reference_fields(geom, sheet=[[0., 0., 1.], [0., 0., 1.]], tension=[2., 3.])
    torch.testing.assert_close(fields.fiber, mesh.fiber[:, None].expand_as(fields.fiber), rtol=0, atol=0)
    torch.testing.assert_close(fields.tension[:, 0], mesh.X.new_tensor([2., 3.]))
    # This imported DG0 material feeds the existing FE weak form unchanged.
    x = (mesh.X @ mesh.X.new_tensor([[1.02, .01, 0.], [0., .99, .01], [0., 0., 1.01]]).T).requires_grad_()
    p = GuccioneParameters(C=2., kappa=10.)
    energy = solid.guccione_energy(x, geom, fields, p)
    force = solid.guccione_force(x, geom, fields, p)
    torch.testing.assert_close(force, -torch.autograd.grad(energy, x)[0], rtol=1e-11, atol=1e-12)
    for face in mesh.faces:
        torch.testing.assert_close(mesh.X[face[3:]], mesh.X[face[:3][torch.tensor([[0, 1], [0, 2], [1, 2]], device=device)]].mean(1))


def test_sheet_import_is_preserved_and_wrong_order_rejected(tmp_path):
    path, boundary, fiber = inputs(tmp_path)
    sheets = [collection(tmp_path/f's{i}.xml', 3, [(c, 0, 2. if i == 2 else 0.) for c in range(2)]) for i in range(3)]
    mesh = read_solid_mesh(path, units='cm', fiber_components=fiber, sheet_components=sheets).to_p2()
    geom = solid.prepare_p2(mesh.X, mesh.cells)
    assert (mesh.reference_fields(geom).sheet[:, :, 2] == 2.).all()
    with pytest.raises(ValueError, match='cell order'):
        mesh.reference_fields(solid.prepare_p2(mesh.X, mesh.cells.flip(0)))
    with pytest.raises(ValueError, match='nonzero and nonparallel'):
        mesh.reference_fields(geom, sheet=mesh.fiber)


@pytest.mark.parametrize('rows,dim,match', [
    ([(0, 0, 1.), (0, 0, 2.)], 3, 'duplicate'),
    ([(2, 0, 1.)], 3, 'out of range'),
    ([(0, 1, 1.)], 3, 'out of range'),
    ([(0, 0, 1.)], 3, 'exactly one value'),
    ([(0, 0, float('nan')), (1, 0, 1.)], 3, 'nonfinite'),
])
def test_invalid_cell_fields_fail(tmp_path, rows, dim, match):
    p = collection(tmp_path/'bad.xml', dim, rows)
    with pytest.raises(ValueError, match=match):
        read_dolfin_collection(p, dimension=dim, cell_count=2)


def test_wrong_dimension_count_and_global_dof_input_fail(tmp_path):
    p = collection(tmp_path/'bad.xml', 2, [(0, 0, 1)], True)
    with pytest.raises(ValueError, match='dim=3'):
        read_dolfin_collection(p, dimension=3, cell_count=2)
    p = collection(p, 3, [(0, 0, 1.), (1, 0, 1.)], size=3)
    with pytest.raises(ValueError, match='declared size'):
        read_dolfin_collection(p, dimension=3, cell_count=2)
    p.write_text('<dolfin><mesh_function><entity index="0" value="1"/></mesh_function></dolfin>')
    with pytest.raises(ValueError, match='MeshValueCollection'):
        read_dolfin_collection(p, dimension=2, cell_count=2, integer=True)


def test_bad_markers_and_mapping_fail(tmp_path):
    path, boundary, fiber = inputs(tmp_path)
    with pytest.raises(ValueError, match='every nonzero'):
        read_solid_mesh(path, units='cm', boundaries=boundary, tag_map={10: 1})
    boundary = collection(boundary, 2, [(0, 0, 1)], True)
    with pytest.raises(ValueError, match='complete exterior'):
        read_solid_mesh(path, units='cm', boundaries=boundary)
    assert (read_solid_mesh(path, units='cm', boundaries=boundary, require_full_boundary=False).facet_tags == 0).sum() == 5
    boundary = collection(boundary, 2, [(0, 3, 1), (1, 3, 2)], True)
    with pytest.raises(ValueError, match='conflicting'):
        read_solid_mesh(path, units='cm', boundaries=boundary)
    boundary = collection(boundary, 2, [(0, 3, 1), (1, 3, 1)], True)
    with pytest.raises(ValueError, match='interior'):
        read_solid_mesh(path, units='cm', boundaries=boundary)


@pytest.mark.parametrize('cells,match', [
    (np.array([[0, 0, 1, 3]]), 'repeated'),
    (np.array([[0, 1, 2, 5]]), 'indices'),
    (np.array([[0, 1, 2, 3], [3, 2, 1, 0]]), 'duplicate'),
    (np.array([[0, 1, 3, 4]]), 'degenerate'),
])
def test_invalid_topology_fails(tmp_path, cells, match):
    with pytest.raises(ValueError, match=match):
        read_solid_mesh(xdmf(tmp_path, cells=cells), units='cm')


def test_nonmanifold_and_invalid_direction_files_fail(tmp_path):
    coords = np.vstack((X, [.2, .2, 2.]))
    cells = np.array([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5]])
    with pytest.raises(ValueError, match='nonmanifold'):
        read_solid_mesh(xdmf(tmp_path, coords=coords, cells=cells), units='cm')
    path, boundary, fiber = inputs(tmp_path)
    with pytest.raises(ValueError, match='three component'):
        read_solid_mesh(path, units='cm', fiber_components=fiber[:2])
    for p in fiber:
        collection(p, 3, [(0, 0, 0.), (1, 0, 0.)])
    with pytest.raises(ValueError, match='zero fiber'):
        read_solid_mesh(path, units='cm', fiber_components=fiber)


def test_grid_selection_and_units_are_explicit(tmp_path):
    p = xdmf(tmp_path)
    with pytest.raises(ValueError, match='source units'):
        read_solid_mesh(p, units='unknown')
    with pytest.raises(ValueError, match='grid_name'):
        read_xdmf(p, grid_name='not-present')
    p.write_text(p.read_text().replace('Tetrahedron', 'Tetrahedron_10'))
    with pytest.raises(ValueError, match='P1'):
        read_xdmf(p)


def test_inspection_writes_tagged_vtk_with_cell_fibers(tmp_path):
    pytest.importorskip('meshio')
    import meshio
    from argparse import Namespace
    from examples.read_lv_mesh import run
    path, boundary, fiber = inputs(tmp_path, hdf=True)
    args = Namespace(mesh=path, units='cm', boundaries=boundary, fibers=fiber, sheets=None,
                     endo_tag=20, epi_tag=10, base_tag=30, translation_cm=(0., 0., 0.),
                     grid_name=None, p2=True, output=tmp_path/'output')
    mesh, report = run(args)
    written = meshio.read(Path(args.output)/'solid.vtu')
    boundary_vtk = meshio.read(Path(args.output)/'boundary.vtu')
    np.testing.assert_array_equal(written.cell_data['fiber'][0], mesh.fiber)
    np.testing.assert_array_equal(written.cells[0].data[:, :4], CELLS)
    vtk_edges = np.array([[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]])
    vtk_nodes = written.points[written.cells[0].data]
    np.testing.assert_allclose(vtk_nodes[:, 4:], vtk_nodes[:, :4][:, vtk_edges].mean(2))
    vtk_faces = boundary_vtk.points[boundary_vtk.cells[0].data]
    np.testing.assert_allclose(vtk_faces[:, 3:], vtk_faces[:, :3][:, [[0, 1], [1, 2], [2, 0]]].mean(2))
    np.testing.assert_array_equal(boundary_vtk.cell_data['tag'][0], mesh.facet_tags)
    assert report['nodes'] == 14 and report['simulation_started'] is False
