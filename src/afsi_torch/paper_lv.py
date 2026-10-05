"""Ma et al. (2024), section V.F, real-LV passive inflation configuration.

The mesh and DG0 directions remain user input. The previously requested
radial-only base spring is retained by agreement; the paper does not specify
this boundary condition for its real ventricle, so it is a declared difference.
"""
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
import torch
from .config import TimeConfig, FluidConfig, OutputConfig, LVExecutionConfig, mass_options
from .mac.multigrid import MGOptions
from .mac.adaptive_transfer import InteractionQuadratureOptions
from .mac.backward_euler import BEFlowOptions
from .nonlinear import NewtonOptions, GMRESOptions
from .mac.midpoint_solver import AndersonOptions
from .holzapfel_ogden import invariants, cofactor
from .real_lv import RealLVSolid
from .mesh_io import read_solid_mesh
from .ho_tangent import HOTangentAssembler
from .fluid.solvers import SolverOptions
from . import boundary as bd, p1


@dataclass(frozen=True)
class PaperHOParameters:
    a: float = 2362.
    b: float = 10.81
    a_f: float = 200370.
    b_f: float = 14.154
    a_s: float = 37245.
    b_s: float = 5.1645
    a_fs: float = 4108.
    b_fs: float = 11.3
    kappa: float = 5e6

    def __post_init__(self):
        if any(not isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError('paper H-O parameters must be positive and finite CGS values')


def paper_energy(F, fiber, sheet, p):
    """Eq. 81 W only; NOT a potential for the complete corrected eq. 82 PK1."""
    _, I1, _, _, _, I4f, I4s, I8 = invariants(F, fiber, sheet)
    ef, es = (I4f-1).clamp_min(0), (I4s-1).clamp_min(0)
    return (p.a/(2*p.b)*torch.exp(p.b*(I1-3))
        + p.a_f/(2*p.b_f)*torch.expm1(p.b_f*ef.square())
        + p.a_s/(2*p.b_s)*torch.expm1(p.b_s*es.square())
        + p.a_fs/(2*p.b_fs)*torch.expm1(p.b_fs*I8.square()))


def paper_pk1(F, fiber, sheet, p):
    """Eq. 82: derivative of RAW-I1 W plus normal-stress and log(I3) terms."""
    J, I1, _, Ff, Fs, I4f, I4s, I8 = invariants(F, fiber, sheet)
    invT = cofactor(F)/J[..., None, None]
    matrix = p.a*torch.exp(p.b*(I1-3))
    P = matrix[..., None, None]*(F-invT)
    outer = lambda u, v: u[..., :, None]*v[..., None, :]
    ef, es = (I4f-1).clamp_min(0), (I4s-1).clamp_min(0)
    P = P+(2*p.a_f*ef*torch.exp(p.b_f*ef.square()))[..., None, None]*outer(Ff, fiber)
    P = P+(2*p.a_s*es*torch.exp(p.b_s*es.square()))[..., None, None]*outer(Fs, sheet)
    P = P+(p.a_fs*I8*torch.exp(p.b_fs*I8.square()))[..., None, None]*(outer(Ff, sheet)+outer(Fs, fiber))
    return P+(2*p.kappa*torch.log(J))[..., None, None]*invT


@dataclass(frozen=True)
class InflationLoads:
    target_mmhg: float = 8.
    ramp_seconds: float = .8

    def __post_init__(self):
        if not isfinite(self.target_mmhg) or self.target_mmhg < 0 or not isfinite(self.ramp_seconds) or self.ramp_seconds <= 0:
            raise ValueError('invalid inflation pressure/ramp time')

    def at(self, time):
        if not isfinite(time) or time < 0:
            raise ValueError('finite nonnegative load time required')
        # Reconstructed from Fig. 24. No periodic reset and no active tension.
        return self.target_mmhg*1333.22387415*min(time/self.ramp_seconds, 1.), 0.


@dataclass(frozen=True)
class PaperLVConfig:
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
    solid_degree: int = 5
    interaction_degree: int = 2
    time: TimeConfig = field(default_factory=lambda: TimeConfig(1e-4, 1.5))
    fluid: FluidConfig = field(default_factory=lambda: FluidConfig((128,)*3, (13.,)*3, rho=1., mu=.04))
    material: PaperHOParameters = field(default_factory=PaperHOParameters)
    loads: InflationLoads = field(default_factory=InflationLoads)
    flow: BEFlowOptions = field(default_factory=BEFlowOptions)
    pressure_solver: MGOptions = field(default_factory=MGOptions)
    mass_solver: SolverOptions = field(default_factory=mass_options)
    interaction_quadrature: InteractionQuadratureOptions = field(default_factory=lambda:
        InteractionQuadratureOptions(mode='adaptive', rule_family='xiao-gimbutas',
            transfer_backend='fused', stencil_backend='shared', prepare_backend='triton', reuse_stencil_buffers=True))
    execution: LVExecutionConfig = field(default_factory=lambda: LVExecutionConfig(
        execution_backend='fused', pressure_backend='graph', mass_backend='graph',
        coupling_backend='optimized', warm_start=True))
    nonlinear_solver: str = 'jfnk'
    support_backend: str = 'vertices'
    nonlinear: NewtonOptions = field(default_factory=lambda: NewtonOptions(rtol=1e-6, atol=1e-9,
        max_iterations=15, linear=GMRESOptions(rtol=1e-3, atol=1e-11, max_iterations=240),
        linear_tolerance_fraction=.05))
    anderson: AndersonOptions = field(default_factory=AndersonOptions)
    max_courant: float = 1.
    output: OutputConfig = field(default_factory=lambda: OutputConfig(100, 1000, 200, True))

    def __post_init__(self):
        for name in ('fiber_files', 'sheet_files', 'basal_center_cm'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if len(self.fiber_files) != 3 or len(self.sheet_files) != 3:
            raise ValueError('three fiber and sheet components required')
        if self.source_units not in ('cm', 'mm', 'm') or len(self.fluid.shape) != 3:
            raise ValueError('3D fluid and valid source units required')
        if len({self.endo_tag, self.epi_tag, self.base_tag}) != 3 or any(type(t) is not int or t <= 0 for t in (self.endo_tag, self.epi_tag, self.base_tag)):
            raise ValueError('distinct positive endo/epi/base tags required')
        if len(self.basal_center_cm) != 2 or not all(isfinite(v) for v in self.basal_center_cm):
            raise ValueError('two finite basal centre coordinates required')
        if not isfinite(self.beta) or self.beta < 0 or self.solid_degree < 5 or self.interaction_degree < 2:
            raise ValueError('invalid basal coefficient or quadrature degree')
        if self.nonlinear_solver not in ('jfnk', 'newton', 'anderson-newton'):
            raise ValueError('nonlinear solver must be jfnk, newton or anderson-newton')
        if self.support_backend not in ('points', 'vertices'):
            raise ValueError('support backend must be points or vertices')
        if not isfinite(self.max_courant) or self.max_courant <= 0:
            raise ValueError('positive characteristic tracing CFL screen required')
        if not self.output.fluid_fields:
            raise ValueError('paper LV output requires paired solid/fluid fields')


class PaperLVSolid(RealLVSolid):
    """Reuse imported P1 assembly, follower pressure and blocked geometry checks."""
    def __init__(self, mesh, config):
        super().__init__(mesh, config)
        self.reference_energy = paper_energy(self.element_gradient(mesh.X), mesh.fiber, mesh.sheet, config.material)

    def force_from_geometry(self, x, F, endo_area, loads):
        P = paper_pk1(F, self.mesh.fiber, self.mesh.sheet, self.parameters)
        local = -self.volumes[:, None, None]*torch.einsum('eiJ,eaJ->eai', P, self.gradients)
        force = torch.zeros_like(x).index_add(0, self.mesh.cells.reshape(-1), local.reshape(-1, 3))
        pressure = bd._scatter(torch.einsum('q,qa,bqi->bai', self.endo.quadrature_weights,
                                           self.endo.values, -loads[0]*endo_area), self.endo)
        return force+pressure+self.basal_force(x)

    def diagnostics(self, x):
        from .mechanics import determinant3
        J = determinant3(self.element_gradient(x))
        return dict(cavity_volume_ml=p1.cavity_volume(x, self.endo_faces, self.rim).item(),
            wall_volume_cm3=(J*self.volumes).sum().item(), minimum_detF=J.min().item(), maximum_detF=J.max().item(),
            max_basal_constraint_cm=torch.linalg.vector_norm(self.basal_constraint(x), dim=-1).max().item(),
            max_total_displacement_cm=torch.linalg.vector_norm(x-self.mesh.X, dim=-1).max().item())

    def tangent_factory(self, chunk_size=2048):
        return PaperHOTangent(self, chunk_size)


class PaperHOTangent(HOTangentAssembler):
    def _volume(self, F, fiber, sheet, gradients, volumes, tension):
        stress = lambda F, f, s: paper_pk1(F, f, s, self.model.parameters)
        D = torch.vmap(torch.func.jacrev(stress, argnums=0))(F, fiber, sheet)
        return -torch.einsum('e,eaJ,eiJkL,ebL->eaibk', volumes, gradients, D, gradients)


def imported_model(config, device='cpu'):
    folder = Path(config.source_dir)
    mesh = read_solid_mesh(folder/config.mesh_file, units=config.source_units,
        boundaries=folder/config.boundary_file,
        fiber_components=[folder/f for f in config.fiber_files],
        sheet_components=[folder/f for f in config.sheet_files],
        tag_map={config.endo_tag: 1, config.epi_tag: 2, config.base_tag: 3}, device=device)
    return PaperLVSolid(mesh, config)
