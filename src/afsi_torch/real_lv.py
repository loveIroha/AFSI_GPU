"""Imported P1 left ventricle: H--O UFL stress and radial-only basal motion."""
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
import torch
from . import boundary as bd, p1
from .mesh_io import read_solid_mesh
from .holzapfel_ogden import HOParameters, RealLVLoads, ho_energy, ho_pk1
from .mechanics import determinant3
from .config import (TimeConfig, FluidConfig, OutputConfig, LVExecutionConfig,
                     MGOptions, mass_options, validate_graph_options)
from .fluid.solvers import SolverOptions
from .mac.implicit import MACCouplingOptions
from .mac.adaptive_transfer import InteractionQuadratureOptions


@dataclass(frozen=True)
class RealLVConfig:
    source_dir: str = '/mnt/large2/gjh/realistic_left_ventricle'
    mesh_file: str = 'mesh_scale.xdmf'
    boundary_file: str = 'boundaries.xml'
    fiber_files: tuple = ('fibers_0.xml', 'fibers_1.xml', 'fibers_2.xml')
    sheet_files: tuple = ('sheets_0.xml', 'sheets_1.xml', 'sheets_2.xml')
    source_units: str = 'cm'
    endo_tag: int = 2
    epi_tag: int = 1
    base_tag: int = 3
    basal_center_cm: tuple = (7.5, 7.5)
    beta: float = 5e6
    time: TimeConfig = field(default_factory=lambda: TimeConfig(1e-4, 2.4))
    fluid: FluidConfig = field(default_factory=lambda: FluidConfig((128,)*3, (15.,)*3))
    material: HOParameters = field(default_factory=HOParameters)
    loads: RealLVLoads = field(default_factory=RealLVLoads)
    solid_degree: int = 5
    interaction_degree: int = 2
    interaction_quadrature: InteractionQuadratureOptions = field(default_factory=InteractionQuadratureOptions)
    pressure_solver: MGOptions = field(default_factory=MGOptions)
    mass_solver: SolverOptions = field(default_factory=mass_options)
    execution: LVExecutionConfig = field(default_factory=lambda: LVExecutionConfig(
        execution_backend='fused', pressure_backend='graph', solid_backend='reference',
        mass_backend='graph', coupling_backend='optimized', warm_start=True))
    output: OutputConfig = field(default_factory=lambda: OutputConfig(100, 1000, 200, True))
    coupling: MACCouplingOptions = field(default_factory=MACCouplingOptions)

    def __post_init__(self):
        for name in ('fiber_files', 'sheet_files', 'basal_center_cm'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if len(self.fiber_files) != 3 or len(self.sheet_files) != 3:
            raise ValueError('three fiber and three sheet files are required')
        if self.source_units not in ('cm', 'mm', 'm'):
            raise ValueError('source_units must be cm, mm, or m')
        tags = (self.endo_tag, self.epi_tag, self.base_tag)
        if len(set(tags)) != 3 or any(type(t) is not int or t <= 0 for t in tags):
            raise ValueError('three distinct positive source tags are required')
        if len(self.basal_center_cm) != 2 or not all(isfinite(c) for c in self.basal_center_cm):
            raise ValueError('basal_center_cm requires two finite coordinates')
        if not isfinite(self.beta) or self.beta < 0:
            raise ValueError('beta must be finite and nonnegative')
        if type(self.solid_degree) is not int or self.solid_degree < 5:
            raise ValueError('solid quadrature degree must be >=5 as in the supplied UFL')
        if type(self.interaction_degree) is not int or self.interaction_degree < 2:
            raise ValueError('interaction degree must be >=2 for consistent P1 mass')
        if len(self.fluid.shape) != 3 or not self.output.fluid_fields:
            raise ValueError('real LV requires 3D fluid and paired output')
        if self.execution.solid_backend != 'reference':
            raise ValueError('real LV uses the compiled H-O kernel, not Guccione pointwise execution')
        if not isinstance(self.coupling, MACCouplingOptions):
            raise ValueError('coupling must be MACCouplingOptions')
        if not isinstance(self.interaction_quadrature,InteractionQuadratureOptions):
            raise ValueError('interaction_quadrature must be InteractionQuadratureOptions')
        if (self.interaction_quadrature.mode=='adaptive' and self.coupling.scheme not in
                ('cnab-midpoint','cnab-semiimplicit')):
            raise ValueError('adaptive quadrature currently requires cnab-midpoint or cnab-semiimplicit')
        if self.coupling.scheme not in ('implicit-newton','cnab-midpoint','cnab-semiimplicit'):
            self.fluid.validate_time(self.time)
        validate_graph_options(self)


def imported_model(config, device='cpu'):
    folder = Path(config.source_dir)
    mesh = read_solid_mesh(folder/config.mesh_file, units=config.source_units,
        boundaries=folder/config.boundary_file,
        fiber_components=[folder/p for p in config.fiber_files],
        sheet_components=[folder/p for p in config.sheet_files],
        tag_map={config.endo_tag: 1, config.epi_tag: 2, config.base_tag: 3}, device=device)
    return RealLVSolid(mesh, config)


class RealLVSolid:
    def __init__(self, mesh, config):
        if mesh.cells.shape[1] != 4 or mesh.fiber is None or mesh.sheet is None:
            raise ValueError('real LV requires P1 tetrahedra and DG0 fiber/sheet fields')
        self.mesh, self.config = mesh, config
        self.parameters, self.loads, self.beta = config.material, config.loads, config.beta
        if set(mesh.facet_tags.cpu().tolist()) != {1, 2, 3}:
            raise ValueError('real LV requires mapped endo/epi/base tags')
        self.geometry = p1.prepare_p1(mesh.X, mesh.cells, config.solid_degree)
        self.gradients = self.geometry.gradients[:, 0]
        self.volumes = self.geometry.weights.sum(1)
        # Check directions, but do not interpolate/normalize these DG0 data.
        mesh.reference_fields(self.geometry)
        self.endo_faces = mesh.surface(1)
        self.endo = p1.prepare_surface(mesh.X, self.endo_faces, config.solid_degree)
        self.base = p1.prepare_surface(mesh.X, mesh.surface(3), config.solid_degree)
        self.rim = p1.cavity_rim(self.endo_faces)
        if not torch.isin(self.rim, mesh.surface(3)).all():
            raise ValueError('endocardial rim must lie on the tagged base')
        R = self.base.reference_positions[..., :2]-mesh.X.new_tensor(config.basal_center_cm)
        if (R.square().sum(-1) <= torch.finfo(mesh.X.dtype).eps).any():
            raise ValueError('radial basal constraint undefined at its center')
        self.radial = R/R.square().sum(-1, keepdim=True).sqrt()
        initial_F = self.element_gradient(mesh.X)
        self.reference_energy = ho_energy(initial_F, mesh.fiber, mesh.sheet, self.parameters)
        self.validate(mesh.X)

    def element_gradient(self, x):
        # grad(current position) == I+grad(displacement) in the supplied UFL.
        return torch.einsum('eai,eaJ->eiJ', x[self.mesh.cells], self.gradients)

    def geometry_state(self, x):
        F = self.element_gradient(x)
        J = determinant3(F)
        endo_area, base_area = bd.area_vectors(x, self.endo), bd.area_vectors(x, self.base)
        volume = p1.cavity_volume(x, self.endo_faces, self.rim)
        def surface_ok(area, surface):
            scale = torch.linalg.vector_norm(surface.reference_area_vectors, dim=-1)[:, None]
            return torch.isfinite(area).all() & (torch.linalg.vector_norm(area, dim=-1) > 100*torch.finfo(x.dtype).eps*scale).all()
        flags = torch.stack((torch.isfinite(J).all() & (J > 0).all(),
                             surface_ok(endo_area, self.endo), surface_ok(base_area, self.base),
                             torch.isfinite(volume) & (volume > 0)))
        return F, endo_area, flags

    def validate(self, x):
        if x.shape != self.mesh.X.shape or x.device != self.mesh.X.device or x.dtype != self.mesh.X.dtype:
            raise ValueError('coordinates must match the prepared P1 mesh')
        if not self.geometry_state(x)[-1].all():
            raise ValueError('invalid real LV deformation: det(F), boundary area or cavity volume')

    def basal_constraint(self, x):
        u = bd.interpolate(x, self.base)-self.base.reference_positions
        radial_u = (u[..., :2]*self.radial).sum(-1, keepdim=True)*self.radial
        return torch.cat((radial_u-u[..., :2], -u[..., 2:]), -1)

    def basal_force(self, x):
        traction = self.beta*self.basal_constraint(x)
        return bd._scatter(torch.einsum('bq,qa,bqi->bai', self.base.reference_weights,
                                       self.base.values, traction), self.base)

    def basal_energy(self, x):
        return .5*self.beta*(self.base.reference_weights*self.basal_constraint(x).square().sum(-1)).sum()

    def force_from_geometry(self, x, F, endo_area, loads):
        P = ho_pk1(F, self.mesh.fiber, self.mesh.sheet, self.parameters, loads[1])
        # P1 gradients and DG0 material make P constant within each cell.
        # Sum the degree-5 weights exactly; avoid repeating 3x3 stress work.
        local = -self.volumes[:, None, None]*torch.einsum('eiJ,eaJ->eai', P, self.gradients)
        force = torch.zeros_like(x).index_add(0, self.mesh.cells.reshape(-1), local.reshape(-1, 3))
        pressure = bd._scatter(torch.einsum('q,qa,bqi->bai', self.endo.quadrature_weights,
                                           self.endo.values, -loads[0]*endo_area), self.endo)
        return force+pressure+self.basal_force(x)

    def force(self, x, time):
        F, endo, _ = self.geometry_state(x)
        return self.force_from_geometry(x, F, endo, x.new_tensor(self.loads.at(time)))

    def diagnostics(self, x):
        F = self.element_gradient(x)
        J = determinant3(F)
        W = ho_energy(F, self.mesh.fiber, self.mesh.sheet, self.parameters)-self.reference_energy
        return dict(cavity_volume_ml=p1.cavity_volume(x, self.endo_faces, self.rim).item(),
                    wall_volume_cm3=(J*self.volumes).sum().item(),
                    minimum_detF=J.min().item(), maximum_detF=J.max().item(),
                    passive_energy_erg=(W*self.volumes).sum().item(),
                    basal_constraint_energy_erg=self.basal_energy(x).item(),
                    max_basal_constraint_cm=torch.linalg.vector_norm(self.basal_constraint(x), dim=-1).max().item(),
                    max_total_displacement_cm=torch.linalg.vector_norm(x-self.mesh.X, dim=-1).max().item())

    def execution_factory(self):
        return HOExecution(self)


from .mac.solid_execution import SolidExecution


class HOExecution(SolidExecution):
    """Reuse checked geometry/cache protocol; compile the actual H-O weak form."""
    def __init__(self, model):
        from .mac.execution import tensor_kernel
        self.model = model
        self.loads = model.mesh.X.new_empty(2)
        self._geometry_kernel = tensor_kernel(model.geometry_state, model.mesh.X.device)
        self._force_kernel = tensor_kernel(model.force_from_geometry, model.mesh.X.device)
        self._cached_x = self._cached_version = self._cached_geometry = None
