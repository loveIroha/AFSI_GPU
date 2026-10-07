"""Public, serializable settings for generated MAC demos (cm-g-s units)."""
from dataclasses import asdict, dataclass, field, fields, replace
import json
from math import isfinite, prod
from pathlib import Path
from .afsi337 import AFSI337Loads, geometry_config
from .afsi340 import ValveConfig
from .geometry import LVConfig
from .materials import GuccioneParameters
from .mac.multigrid import MGOptions
from .fluid.solvers import SolverOptions


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')


@dataclass(frozen=True)
class TimeConfig:
    dt: float = 5e-5
    end_time: float = 2.

    def __post_init__(self):
        positive(self.dt, 'dt'); positive(self.end_time, 'end_time')
        steps = round(self.end_time/self.dt)
        if steps < 1 or abs(steps*self.dt-self.end_time) > 1e-12:
            raise ValueError('end time must be an integer multiple of dt')


@dataclass(frozen=True)
class FluidConfig:
    """Cell counts, box lengths/origin in cm, density g/cm³, viscosity g/(cm s)."""
    shape: tuple = (64, 64, 64)
    lengths: tuple = (5., 5., 5.)
    origin: tuple = (0., 0., 0.)
    rho: float = 1.
    mu: float = 1.

    def __post_init__(self):
        for name in ('shape', 'lengths', 'origin'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        dimension = len(self.shape)
        if dimension not in (2, 3) or any(type(n) is not int or n < 4 for n in self.shape):
            raise ValueError('fluid shape requires two or three integer cell counts >=4')
        if len(self.lengths) != dimension or len(self.origin) != dimension:
            raise ValueError('fluid shape, lengths and origin must have the same dimension')
        for v in self.lengths:
            positive(v, 'fluid length')
        if any(isinstance(v, bool) or not isfinite(v) for v in self.origin):
            raise ValueError('fluid origin must be finite')
        positive(self.rho, 'rho'); positive(self.mu, 'mu')
        shape = self.shape
        while min(shape) > 4 and all(n % 2 == 0 for n in shape):
            shape = tuple(n//2 for n in shape)
        if prod(shape) > 512:
            raise ValueError('fluid shape must coarsen to <=512 cells; prefer powers of two')

    @property
    def spacing(self):
        return tuple(L/n for L, n in zip(self.lengths, self.shape))

    def validate_time(self, time):
        if time.dt*self.mu/self.rho*sum(1/h**2 for h in self.spacing) > .25:
            raise ValueError('explicit viscous time step too large; reduce dt')


@dataclass(frozen=True)
class OutputConfig:
    """Step intervals; fluid_fields controls optional fluid export in the 2D valve."""
    log_every: int = 200
    checkpoint_every: int = 1000
    output_every: int = 400
    write_vtk: bool = True
    fluid_fields: bool = True

    def __post_init__(self):
        if any(type(n) is not int or n < 1 for n in
               (self.log_every, self.checkpoint_every, self.output_every)):
            raise ValueError('output intervals must be positive integers')
        if type(self.write_vtk) is not bool or type(self.fluid_fields) is not bool:
            raise ValueError('output switches must be bool')


@dataclass(frozen=True)
class LVExecutionConfig:
    execution_backend: str = 'torch'
    pressure_backend: str = 'torch'
    solid_backend: str = 'reference'
    mass_backend: str = 'pcg'
    coupling_backend: str = 'reference'
    warm_start: bool = False
    ib_backend: str = 'reference'

    def __post_init__(self):
        choices = dict(execution_backend=('torch', 'fused'), pressure_backend=('torch', 'fused', 'workspace', 'graph'),
                       solid_backend=('reference', 'pointwise'), mass_backend=('pcg', 'graph'),
                       coupling_backend=('reference', 'optimized'), ib_backend=('reference','cuda'))
        for name, values in choices.items():
            if getattr(self, name) not in values:
                raise ValueError(f'invalid {name}')
        if type(self.warm_start) is not bool:
            raise ValueError('warm_start must be bool')
        if self.ib_backend=='cuda' and self.execution_backend!='fused':
            raise ValueError('native MAC IB requires fused execution')
        if self.execution_backend != 'fused' and (self.solid_backend != 'reference' or
                self.mass_backend != 'pcg' or self.coupling_backend != 'reference'):
            raise ValueError('pointwise/graph mass/optimized coupling require fused execution')


@dataclass(frozen=True)
class ValveExecutionConfig:
    execution_backend: str = 'optimized'
    pressure_backend: str = 'auto'
    mass_backend: str = 'graph'
    fused: bool = True
    warm_start: bool = True
    ib_backend: str = 'reference'

    def __post_init__(self):
        if (self.execution_backend not in ('reference', 'optimized') or
                self.pressure_backend not in ('auto', 'reference', 'workspace', 'graph') or
                self.mass_backend not in ('pcg', 'graph') or self.ib_backend not in ('reference','cuda')):
            raise ValueError('invalid valve execution backend')
        if type(self.fused) is not bool or type(self.warm_start) is not bool:
            raise ValueError('fused/warm_start must be bool')


@dataclass(frozen=True)
class InletConfig:
    """u_x = amplitude * (sin(2*pi*time/period) + offset) * y*(H-y)."""
    amplitude: float = 5.
    period: float = 1.
    offset: float = 1.1

    def __post_init__(self):
        positive(self.period, 'inlet period')
        if any(isinstance(v, bool) or not isfinite(v) for v in (self.amplitude, self.offset)):
            raise ValueError('inlet amplitude/offset must be finite')


def mass_options():
    return SolverOptions(rtol=1e-12, atol=1e-13, max_iterations=500, check_every=4)


@dataclass(frozen=True)
class FEMFluidConfig(FluidConfig):
    """Q2/Q1 element counts; implicit viscosity has no MAC viscous-step guard."""
    shape: tuple = (32,32,32)

    def __post_init__(self):
        for name in ('shape','lengths','origin'):
            object.__setattr__(self,name,tuple(getattr(self,name)))
        if len(self.shape)!=3 or any(type(n) is not int or n<2 for n in self.shape):
            raise ValueError('FEM requires three integer element counts >=2')
        if len(self.lengths)!=3 or len(self.origin)!=3:
            raise ValueError('FEM lengths/origin require three coordinates')
        for length in self.lengths:
            positive(length,'fluid length')
        if any(isinstance(v,bool) or not isfinite(v) for v in self.origin):
            raise ValueError('FEM origin must be finite')
        positive(self.rho,'rho'); positive(self.mu,'mu')


@dataclass(frozen=True)
class LVFEMSimulationConfig:
    time: TimeConfig = field(default_factory=TimeConfig)
    fluid: FEMFluidConfig = field(default_factory=FEMFluidConfig)
    geometry: LVConfig = field(default_factory=geometry_config)
    material: GuccioneParameters = field(default_factory=GuccioneParameters)
    loads: AFSI337Loads = field(default_factory=AFSI337Loads)
    beta: float = 5e5
    solver: SolverOptions = field(default_factory=lambda: SolverOptions(
        max_iterations=4000,recompute_every=200,check_every=8))
    backend: str = 'csr'
    history_every: int = 20
    output: OutputConfig = field(default_factory=lambda: OutputConfig(100,200,200,True))
    ib_backend: str = 'reference'

    def __post_init__(self):
        if self.ib_backend not in ('reference','cuda'):
            raise ValueError('IB backend must be reference or cuda')
        if not self.output.fluid_fields:
            raise ValueError('3D LV output contains both solid and fluid fields; fluid_fields must be True')
        if self.geometry.long_axis!='x':
            raise ValueError('generated AFSI ellipsoid fibers require long_axis=x')
        if isinstance(self.beta,bool) or not isfinite(self.beta) or self.beta<0:
            raise ValueError('beta must be finite and nonnegative')
        if self.backend not in ('csr','quadrature'):
            raise ValueError('FEM backend must be csr or quadrature')
        if type(self.history_every) is not int or self.history_every<1:
            raise ValueError('history_every must be a positive integer')


@dataclass(frozen=True)
class LVSimulationConfig:
    time: TimeConfig = field(default_factory=TimeConfig)
    fluid: FluidConfig = field(default_factory=FluidConfig)
    geometry: LVConfig = field(default_factory=geometry_config)
    material: GuccioneParameters = field(default_factory=GuccioneParameters)
    loads: AFSI337Loads = field(default_factory=AFSI337Loads)
    beta: float = 5e5
    interaction_degree: int | None = None
    pressure_solver: MGOptions = field(default_factory=MGOptions)
    mass_solver: SolverOptions = field(default_factory=mass_options)
    execution: LVExecutionConfig = field(default_factory=LVExecutionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self):
        if not self.output.fluid_fields:
            raise ValueError('3D LV output contains both solid and fluid fields; fluid_fields must be True')
        if len(self.fluid.shape) != 3:
            raise ValueError('LV requires a 3D fluid grid')
        if self.geometry.long_axis != 'x':
            raise ValueError('generated AFSI ellipsoid fibers require long_axis=x')
        if isinstance(self.beta, bool) or not isfinite(self.beta) or self.beta < 0:
            raise ValueError('beta must be finite and nonnegative')
        if self.interaction_degree is not None and (type(self.interaction_degree) is not int or self.interaction_degree < 4):
            raise ValueError('interaction degree must be >=4')
        self.fluid.validate_time(self.time)
        validate_graph_options(self)


@dataclass(frozen=True)
class ValveSimulationConfig:
    time: TimeConfig = field(default_factory=lambda: TimeConfig(1/16000, 3.))
    fluid: FluidConfig = field(default_factory=lambda: FluidConfig((256,64), (8.,1.61), (0.,0.), mu=.1))
    solid: ValveConfig = field(default_factory=ValveConfig)
    inlet: InletConfig = field(default_factory=InletConfig)
    pressure_solver: MGOptions = field(default_factory=MGOptions)
    mass_solver: SolverOptions = field(default_factory=mass_options)
    execution: ValveExecutionConfig = field(default_factory=ValveExecutionConfig)
    output: OutputConfig = field(default_factory=lambda: OutputConfig(160,1600,160,True,False))

    def __post_init__(self):
        if len(self.fluid.shape) != 2 or self.fluid.origin != (0.,0.):
            raise ValueError('valve channel requires a 2D grid with origin (0,0)')
        if self.fluid.lengths[1] != self.solid.height:
            raise ValueError('fluid height must equal solid.height (wall-attached leaflets)')
        if self.solid.x_right >= self.fluid.lengths[0] or self.solid.x_right <= self.solid.width:
            raise ValueError('valve leaflets must be inside the channel')
        self.fluid.validate_time(self.time)
        validate_graph_options(self)


def validate_graph_options(config):
    if config.execution.pressure_backend == 'graph' or (
            isinstance(config, ValveSimulationConfig) and config.execution.pressure_backend == 'auto'
            and config.execution.execution_backend == 'optimized'):
        if config.pressure_solver.check_every > 16:
            raise ValueError('pressure graph check_every must be <=16')
    if config.execution.mass_backend == 'graph' and config.mass_solver.check_every > 16:
        raise ValueError('mass graph check_every must be <=16')


def _decode(cls, data, path='config'):
    if not isinstance(data, dict):
        raise ValueError(f'{path} must be a JSON object')
    default = cls() if isinstance(cls, type) else cls
    unknown = set(data)-{f.name for f in fields(default)}
    if unknown:
        raise ValueError(f'unknown {path} fields: {sorted(unknown)}')
    values = {}
    for name, value in data.items():
        current = getattr(default, name)
        values[name] = _decode(current, value, path+'.'+name) if hasattr(current, '__dataclass_fields__') else value
    return replace(default, **values)


def load_config(path, cls=LVSimulationConfig):
    """Read a partial JSON config, reject misspelled keys, fill documented defaults."""
    with Path(path).open(encoding='utf-8') as stream:
        data=json.load(stream)
    if isinstance(data,dict) and 'demo' in data:
        names={'ideal-lv-mac':'LVSimulationConfig','ideal-lv-fem':'LVFEMSimulationConfig',
               'ideal-valve-mac':'ValveSimulationConfig','real-lv':'PaperLVConfig'}
        actual=cls.__name__ if isinstance(cls,type) else type(cls).__name__
        if names.get(data.pop('demo'))!=actual:
            raise ValueError('JSON demo does not match this entry point')
    return _decode(cls,data)


def save_config(path, config):
    from .cycle_checkpoint import atomic_json
    atomic_json(path, asdict(config))


def lv_grid(settings):
    """Read new rectangular domains or legacy cubic checkpoint settings."""
    from .mac.grid import MACGrid
    return MACGrid(tuple(settings['fluid_shape']) if 'fluid_shape' in settings else (settings['fluid_cells'],)*3,
                   tuple(settings['fluid_lengths']) if 'fluid_lengths' in settings else (settings['box_length'],)*3,
                   tuple(settings.get('fluid_origin', (0.,0.,0.))))
