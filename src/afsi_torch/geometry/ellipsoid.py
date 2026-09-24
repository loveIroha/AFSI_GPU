"""Generate a truncated, concentric ellipsoidal wall, without input mesh files."""
from dataclasses import dataclass, replace
from math import isfinite, pi
import numpy as np
import torch
from ..tetrahedron import promote_p1
from ..boundary import extract_boundary

ENDO, EPI, BASE = 1, 2, 3


@dataclass(frozen=True)
class LVConfig:
    """Lengths in centimetres; keep z <= center_z + base_height. Apex is -z.

    Defaults are numerical example parameters, not a calibrated patient model.
    The two concentric ellipsoids need not be homothetic.
    """
    inner_axes: tuple[float, float, float] = (2.5, 2.5, 4.5)
    outer_axes: tuple[float, float, float] = (3.5, 3.5, 5.5)
    base_height: float = 1.5
    mesh_size: float = 1.2
    center: tuple[float, float, float] = (0., 0., 0.)

    def __post_init__(self):
        for name in ('inner_axes', 'outer_axes', 'center'):
            data = tuple(float(v) for v in getattr(self, name))
            if len(data) != 3 or not all(isfinite(v) for v in data):
                raise ValueError(f'{name} must contain three finite coordinates')
            object.__setattr__(self, name, data)
        if any(not 0 < a < b for a, b in zip(self.inner_axes, self.outer_axes)):
            raise ValueError('all inner semi-axes must be positive and strictly inside outer axes')
        if not isfinite(self.base_height) or not -self.inner_axes[2] < self.base_height < self.inner_axes[2]:
            raise ValueError('base plane must cut through the inner ellipsoid')
        if not isfinite(self.mesh_size) or self.mesh_size <= 0:
            raise ValueError('mesh_size must be finite and positive')

    def enclosed_volume(self, axes):
        """Analytic ellipsoid volume below the base plane, closed by a disk."""
        a, b, c = axes
        z = self.base_height
        return pi*a*b*(z-z**3/(3*c*c)+2*c/3)

    @property
    def cavity_volume(self):
        return self.enclosed_volume(self.inner_axes)

    @property
    def wall_volume(self):
        return self.enclosed_volume(self.outer_axes)-self.cavity_volume


@dataclass(frozen=True)
class LVMesh:
    config: LVConfig
    X: torch.Tensor          # N,3 reference coordinates, straight P2 edges
    cells: torch.Tensor      # E,10 in project ordering
    faces: torch.Tensor      # B,6, outward from SOLID
    facet_tags: torch.Tensor # B, ENDO/EPI/BASE
    vertex_count: int
    gmsh_version: str

    def surface(self, tag):
        if tag not in (ENDO, EPI, BASE):
            raise ValueError('unknown LV surface tag')
        return self.faces[self.facet_tags == tag]

    def to(self, device):
        return replace(self, X=self.X.to(device), cells=self.cells.to(device),
                       faces=self.faces.to(device), facet_tags=self.facet_tags.to(device))


def generate_lv(config=None, *, device='cpu') -> LVMesh:
    """Gmsh CPU generation -> validated connectivity -> optional device transfer.

    Own the Gmsh session, refuse to clear an existing caller-owned model.
    CAD is built at unit scale to keep geometry tolerances scale-independent.
    No files or GUI are required. Only first-order tetrahedra are meshed; shared
    midpoint DOFs are appended afterwards, without projecting onto CAD curves.
    """
    config = LVConfig() if config is None else config
    try:
        import gmsh
    except ImportError as exc:
        raise ImportError('Install geometry dependencies: pip install -e ".[geometry]"') from exc
    if gmsh.isInitialized():
        raise RuntimeError('generate_lv requires its own Gmsh session; finalize the existing session first')
    scale = max(config.outer_axes)
    inner = np.array(config.inner_axes)/scale
    outer = np.array(config.outer_axes)/scale
    base = config.base_height/scale
    gmsh.initialize([], readConfigFiles=False)
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.option.setNumber('General.NumThreads', 1)
        gmsh.model.add('ideal_lv')
        occ = gmsh.model.occ
        volumes = []
        for axes in (outer, inner):
            v = occ.addSphere(0, 0, 0, 1)
            occ.dilate([(3, v)], 0, 0, 0, *axes)
            volumes.append(v)
        wall, _ = occ.cut([(3, volumes[0])], [(3, volumes[1])])
        upper = occ.addBox(-2, -2, base, 4, 4, 4)
        wall, _ = occ.cut(wall, [(3, upper)])
        occ.synchronize()
        if len(wall) != 1 or wall[0][0] != 3:
            raise RuntimeError('expected one connected LV wall volume')
        gmsh.option.setNumber('Mesh.ElementOrder', 1)
        gmsh.option.setNumber('Mesh.MeshSizeMin', config.mesh_size/scale)
        gmsh.option.setNumber('Mesh.MeshSizeMax', config.mesh_size/scale)
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
        gmsh.option.setNumber('Mesh.Algorithm3D', 1)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize('')
        types, _, connectivity = gmsh.model.mesh.getElements(3)
        if list(types) != [4]:
            raise RuntimeError('expected first-order tetrahedral volume mesh only')
        raw_cells = np.asarray(connectivity[0]).reshape(-1, 4)
        used = np.unique(raw_cells)
        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(tags)
        sorted_tags = np.asarray(tags)[order]
        node_indices = np.searchsorted(sorted_tags, used)
        if not np.array_equal(sorted_tags[node_indices], used):
            raise RuntimeError('Gmsh node tags could not be mapped')
        points = np.asarray(coordinates).reshape(-1, 3)[order[node_indices]]
        cells = np.searchsorted(used, raw_cells).astype(np.int64)
        D = points[cells[:, 1:]]-points[cells[:, :1]]
        determinant = np.linalg.det(D)
        if not np.isfinite(determinant).all() or (determinant == 0).any():
            raise RuntimeError('Gmsh produced a degenerate tetrahedron')
        flip = determinant < 0
        cells[flip] = cells[flip][:, [0, 2, 1, 3]]

        # Identify whole CAD surfaces using their on-CAD mesh nodes, including
        # base curves; never guess tags from the centroid of a flat triangle.
        labels = {}
        groups = {ENDO: [], EPI: [], BASE: []}
        for dim, tag in gmsh.model.getBoundary(wall, oriented=False):
            _, surface_xyz, _ = gmsh.model.mesh.getNodes(dim, tag, includeBoundary=True)
            samples = np.asarray(surface_xyz).reshape(-1, 3)
            if not len(samples):
                raise RuntimeError('empty CAD surface')
            if np.max(np.abs(samples[:, 2]-base)) < 1e-7:
                marker = BASE
            else:
                # OCC's affine conversion of a general ellipsoid to B-splines
                # has small CAD errors (~1e-6 implicit residual in test cases).
                # Require a unique match, never choose an ambiguous thin wall.
                errors = [np.max(np.abs(np.sum((samples/axes)**2, axis=1)-1))
                          for axes in (inner, outer)]
                matches = [label for label, error in zip((ENDO, EPI), errors) if error < 1e-5]
                if len(matches) != 1:
                    raise RuntimeError(f'ambiguous LV surface {tag}; inner/outer residuals={errors}')
                marker = matches[0]
            groups[marker].append(tag)
            stypes, _, sconn = gmsh.model.mesh.getElements(2, tag)
            if list(stypes) != [2]:
                raise RuntimeError('expected first-order surface triangles')
            tri = np.asarray(sconn[0]).reshape(-1, 3)
            mapped = np.searchsorted(used, tri)
            if (mapped >= len(used)).any() or not np.array_equal(used[mapped], tri):
                raise RuntimeError('surface node missing from wall volume')
            for face in mapped:
                key = tuple(sorted(face.tolist()))
                if key in labels:
                    raise RuntimeError('duplicate tagged surface triangle')
                labels[key] = marker
        for marker, entities in groups.items():
            if not entities:
                raise RuntimeError('missing ENDO/EPI/BASE surface')
            gmsh.model.addPhysicalGroup(2, entities, marker)
            gmsh.model.setPhysicalName(2, marker, {ENDO: 'ENDO', EPI: 'EPI', BASE: 'BASE'}[marker])
        version = gmsh.__version__
    finally:
        gmsh.finalize()
    vertices = torch.from_numpy(points*scale+np.array(config.center))
    X, p2_cells = promote_p1(vertices, torch.from_numpy(cells))
    faces = extract_boundary(X, p2_cells)
    keys = [tuple(sorted(face)) for face in faces[:, :3].tolist()]
    if len(keys) != len(labels) or set(keys) != set(labels):
        raise RuntimeError('CAD surface tags and tetrahedral exterior disagree')
    facet_tags = torch.tensor([labels[key] for key in keys], dtype=torch.int64)
    return LVMesh(config, X, p2_cells, faces, facet_tags, len(vertices), version).to(device)
