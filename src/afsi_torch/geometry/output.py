"""Optional VTK output of generated geometry; no solver input-file dependency."""
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from ..units import CGS_UNITS


def write_lv(directory, mesh, fields, *, current=None, force=None, report=None):
    """Export P2 volume/surfaces and an in-memory-generated NPZ for inspection."""
    try:
        import meshio
    except ImportError as exc:
        raise ImportError('Install output dependencies: pip install -e ".[geometry]"') from exc
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    array = lambda x: x.detach().cpu().numpy()
    X, cells, faces, tags = map(array, (mesh.X, mesh.cells, mesh.faces, mesh.facet_tags))
    point_data = dict(fiber=array(fields.fiber), sheet=array(fields.sheet),
                      transmural=array(fields.transmural), apical_weight=array(fields.apical_weight))
    # VTK tetra10 edges: 01,12,20,03,13,23. VTK triangle6: 01,12,20.
    vtk_cells = cells[:, [0, 1, 2, 3, 4, 7, 5, 6, 8, 9]]
    vtk_faces = faces[:, [0, 1, 2, 3, 5, 4]]
    meshio.write(directory/'reference.vtu', meshio.Mesh(X, [('tetra10', vtk_cells)], point_data=point_data))
    meshio.write(directory/'surfaces.vtu', meshio.Mesh(X, [('triangle6', vtk_faces)],
                  cell_data={'surface_tag': [tags]}, point_data=point_data))
    if current is not None:
        extra = dict(point_data, displacement_cm=array(current)-X)
        if force is not None:
            extra['nodal_force_dyn'] = array(force)
        meshio.write(directory/'prescribed_deformation.vtu', meshio.Mesh(
            array(current), [('tetra10', vtk_cells)], point_data=extra))
    metadata = dict(config=asdict(mesh.config), units=CGS_UNITS, gmsh=mesh.gmsh_version,
                    facet_tags=dict(ENDO=1, EPI=2, BASE=3),
                    p2_order='v0,v1,v2,v3,e01,e02,e03,e12,e13,e23')
    np.savez_compressed(directory/'generated.npz', X=X, cells=cells, faces=faces,
                        facet_tags=tags, **point_data, metadata=json.dumps(metadata))
    (directory/'report.json').write_text(json.dumps(dict(metadata=metadata, results=report),
                                                    indent=2)+'\n', encoding='utf-8')
