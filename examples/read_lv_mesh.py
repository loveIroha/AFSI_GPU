"""Inspect external reference LV inputs; does not launch an FSI simulation."""
import argparse
import json
from pathlib import Path
import numpy as np
from afsi_torch.mesh_io import read_solid_mesh


def run(args):
    requested = (args.endo_tag, args.epi_tag, args.base_tag)
    if any(t is not None for t in requested) and (any(t is None for t in requested) or len(set(requested)) != 3):
        raise ValueError('provide three distinct source tags: --endo-tag, --epi-tag, --base-tag')
    mapping = None if requested == (None, None, None) else dict(zip(requested, (1, 2, 3)))
    mesh = read_solid_mesh(args.mesh, units=args.units, boundaries=args.boundaries,
                           fiber_components=args.fibers, sheet_components=args.sheets,
                           tag_map=mapping, translation_cm=args.translation_cm,
                           grid_name=args.grid_name)
    if args.p2:
        mesh = mesh.to_p2()
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    X, cells = mesh.X.numpy(), mesh.cells.numpy()
    corners = X[cells[:, :4]]
    D = np.transpose(corners[:, 1:]-corners[:, :1], (0, 2, 1))
    volume = np.abs(np.linalg.det(D))/6
    report = dict(metadata=mesh.metadata, nodes=len(X), vertices=mesh.vertex_count,
                  cells=len(cells), exterior_facets=len(mesh.faces),
                  bbox_min_cm=X.min(0).tolist(), bbox_max_cm=X.max(0).tolist(),
                  wall_volume_cm3=float(volume.sum()), minimum_cell_volume_cm3=float(volume.min()),
                  facet_counts={str(t): int((mesh.facet_tags == t).sum()) for t in mesh.facet_tags.unique().tolist()},
                  fiber_imported=mesh.fiber is not None, sheet_imported=mesh.sheet is not None,
                  simulation_started=False)
    if mesh.fiber is not None:
        norms = np.linalg.norm(mesh.fiber.numpy(), axis=1)
        report['fiber_norm_range'] = [float(norms.min()), float(norms.max())]
    try:
        import meshio
    except ImportError as exc:
        raise ImportError('VTK inspection output requires pip install -e ".[mesh]"') from exc
    cell_data = {'original_cell_index': [np.arange(len(cells), dtype=np.int64)],
                 'reference_cell_volume_cm3': [volume]}
    for name in ('fiber', 'sheet'):
        field = getattr(mesh, name)
        if field is not None:
            cell_data[name] = [field.numpy()]
    vtk_cells = cells if cells.shape[1] == 4 else cells[:, [0, 1, 2, 3, 4, 7, 5, 6, 8, 9]]
    meshio.write(folder/'solid.vtu', meshio.Mesh(X, [('tetra' if cells.shape[1] == 4 else 'tetra10', vtk_cells)], cell_data=cell_data))
    boundary_data = {'tag': [mesh.facet_tags.numpy()],
                     'owner_cell': [mesh.boundary_cells.numpy()],
                     'local_facet': [mesh.boundary_local_facets.numpy()]}
    # VTK triangle6 edges are 01,12,20; project edges are 01,02,12.
    vtk_faces = mesh.faces.numpy() if mesh.faces.shape[1] == 3 else mesh.faces[:, [0, 1, 2, 3, 5, 4]].numpy()
    meshio.write(folder/'boundary.vtu', meshio.Mesh(X, [('triangle' if vtk_faces.shape[1] == 3 else 'triangle6', vtk_faces)], cell_data=boundary_data))
    (folder/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))
    return mesh, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', required=True, help='static tetrahedral XDMF reference mesh')
    parser.add_argument('--units', required=True, choices=('cm', 'mm', 'm'), help='units in the original file')
    parser.add_argument('--boundaries', help='DOLFIN facet MeshValueCollection XML or XML .txt')
    parser.add_argument('--fibers', nargs=3, metavar=('FX', 'FY', 'FZ'), help='three DG0 component XML files')
    parser.add_argument('--sheets', nargs=3, metavar=('SX', 'SY', 'SZ'), help='optional three DG0 sheet component XML files')
    parser.add_argument('--endo-tag', type=int)
    parser.add_argument('--epi-tag', type=int)
    parser.add_argument('--base-tag', type=int)
    parser.add_argument('--translation-cm', nargs=3, type=float, default=(0., 0., 0.))
    parser.add_argument('--grid-name')
    parser.add_argument('--p2', action='store_true', help='append shared straight-edge midpoint DOFs')
    parser.add_argument('--output', default='results/imported_lv')
    run(parser.parse_args())
