"""Generated LV solid model for the explicit coupling smoke test, CGS units."""
from dataclasses import dataclass, replace
from math import isfinite
import torch
from . import solid, boundary as bd
from .fields import prepare_reference_fields
from .materials import GuccioneParameters, guccione_energy
from .mechanics import determinant3
from .geometry import ENDO, BASE, rule_based_fibers, prepare_cavity, cavity_volume
from .units import MMHG_TO_DYN_PER_CM2


@dataclass(frozen=True)
class RampLoads:
    pressure_mmhg: float = 8.
    tension: float = 1000.  # dyn/cm^2, prescribed numerical-test load
    ramp_time: float = .01  # s; not a physiological cycle

    def __post_init__(self):
        if (not all(isfinite(v) for v in (self.pressure_mmhg, self.tension, self.ramp_time)) or
                min(self.pressure_mmhg, self.tension) < 0 or self.ramp_time <= 0):
            raise ValueError('nonnegative loads and positive ramp time required')

    def at(self, time):
        if not isfinite(time) or time < 0:
            raise ValueError('nonnegative finite load time required')
        scale = min(time/self.ramp_time, 1.)
        return self.pressure_mmhg*MMHG_TO_DYN_PER_CM2*scale, self.tension*scale


@dataclass(frozen=True)
class PreloadedLoads:
    """Physical-time load schedule starting at the saved preload, not zero."""
    pressure_mmhg: float
    tension: float = 0.
    pressure_increment_mmhg: float = 0.
    hold_time: float = .0005
    ramp_time: float = .0005

    def __post_init__(self):
        values=(self.pressure_mmhg,self.tension,self.pressure_increment_mmhg,self.hold_time,self.ramp_time)
        if not all(isfinite(v) for v in values) or min(values)<0 or self.ramp_time==0:
            raise ValueError('finite nonnegative preloaded loads and positive ramp time required')

    def at(self,time):
        if not isfinite(time) or time<0:
            raise ValueError('nonnegative finite load time required')
        fraction=min(max((time-self.hold_time)/self.ramp_time,0.),1.)
        return ((self.pressure_mmhg+fraction*self.pressure_increment_mmhg)*MMHG_TO_DYN_PER_CM2,self.tension)


class LVSolid:
    def __init__(self, mesh, *, loads=None, beta=5e5, parameters=None):
        if not isfinite(beta) or beta < 0:
            raise ValueError('nonnegative finite beta required')
        self.mesh, self.beta = mesh, beta
        self.loads = RampLoads() if loads is None else loads
        self.parameters = GuccioneParameters() if parameters is None else parameters
        self.geometry = solid.prepare_p2(mesh.X, mesh.cells)
        self.fibers = rule_based_fibers(mesh.X, mesh.config)
        self.fields = prepare_reference_fields(self.geometry, self.fibers.fiber, self.fibers.sheet, 0.)
        self.endo = bd.prepare_surface(mesh.X, mesh.surface(ENDO))
        self.base = bd.prepare_surface(mesh.X, mesh.surface(BASE))
        self.cavity = prepare_cavity(mesh.X, mesh.surface(ENDO))

    def validate(self, x):
        solid.validate_deformation(x, self.geometry)
        bd.validate_surface(x, self.endo)
        bd.validate_surface(x, self.base)
        volume = cavity_volume(x, self.cavity)
        if not torch.isfinite(volume) or volume <= 0:
            raise ValueError('cavity volume must remain finite and positive')

    def force(self, x, time):
        pressure, tension = self.loads.at(time)
        fields = replace(self.fields, tension=torch.full_like(self.fields.tension, tension))
        return (solid.guccione_force(x, self.geometry, fields, self.parameters)+
                bd.pressure_force(x, self.endo, pressure)+bd.spring_force(x, self.base, self.beta))

    def diagnostics(self, x):
        F = solid.deformation_gradient(x, self.geometry)
        J = determinant3(F)
        W = guccione_energy(F, self.fields.fiber, self.fields.sheet, self.fields.normal, self.parameters)
        return dict(cavity_volume_ml=cavity_volume(x, self.cavity).item(),
            wall_volume_cm3=(J*self.geometry.weights).sum().item(),
            minimum_detF=J.min().item(), maximum_detF=J.max().item(),
            passive_energy_erg=(W*self.geometry.weights).sum().item(),
            spring_energy_erg=bd.spring_energy(x, self.base, self.beta).item(),
            max_total_displacement_cm=torch.linalg.vector_norm(x-self.mesh.X, dim=-1).max().item())
