"""Prescribed ideal-LV cycle, in CGS, following AFSI demo_337 load shapes.

PressureEndo's Gaussian rise/decay is expressed as a symmetric pulse. The
initial filling ramp is applied only once, so subsequent cycles do not jump
back to the unloaded pressure. This is an imposed-load experiment, not a
closed circulation model. No AFSI source code is imported or redistributed.
"""
from dataclasses import dataclass
from math import expm1, isfinite
from .units import MMHG_TO_DYN_PER_CM2


@dataclass(frozen=True)
class AFSICycleLoads:
    period: float = .8
    filling_end: float = .2
    systole_start: float = .5
    initial_pressure: float = 0.             # dyn/cm^2
    diastole_pressure: float = 8.*MMHG_TO_DYN_PER_CM2
    systole_pressure: float = 150000.         # demo_337 amplitude
    max_tension: float = 600000.              # demo_337 amplitude, dyn/cm^2
    pressure_width: float = .004             # s^2
    tension_width: float = .005              # s^2

    def __post_init__(self):
        if not all(isfinite(v) for v in vars(self).values()):
            raise ValueError('cycle parameters must be finite')
        if not 0 < self.filling_end <= self.systole_start < self.period:
            raise ValueError('require 0 < filling_end <= systole_start < period')
        if not 0 <= self.initial_pressure <= self.diastole_pressure <= self.systole_pressure:
            raise ValueError('require 0 <= initial <= diastolic <= systolic pressure')
        if self.max_tension < 0 or min(self.pressure_width, self.tension_width) <= 0:
            raise ValueError('nonnegative tension and positive pulse widths required')

    def at(self, time):
        if not isfinite(time) or time < 0:
            raise ValueError('nonnegative finite load time required')
        phase = time % self.period
        base = self.diastole_pressure
        if time < self.filling_end:
            base = self.initial_pressure + (base-self.initial_pressure)*time/self.filling_end
        distance = max(0., min(phase-self.systole_start, self.period-phase))
        pressure_pulse = -expm1(-distance*distance/self.pressure_width)
        tension_pulse = -expm1(-distance*distance/self.tension_width)
        return (base + (self.systole_pressure-self.diastole_pressure)*pressure_pulse,
                self.max_tension*tension_pulse)

