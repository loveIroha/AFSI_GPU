"""The user's H--O UFL law, including its specific volumetric/active terms.

Only I1 is isochorically modified. I4f/I4s use the unmodified C and a
tension-only positive part. DG0 directions are retained without normalization.
Parameters are CGS stresses, not values to be converted again from kPa.
"""
from dataclasses import dataclass
from math import isfinite
import torch
from .mechanics import determinant3


@dataclass(frozen=True)
class HOParameters:
    a: float = 2244.87
    b: float = 1.6215
    a_f: float = 24267.
    b_f: float = 1.8268
    a_s: float = 5562.38
    b_s: float = .7746
    a_fs: float = 3905.16
    b_fs: float = 1.695
    kappa: float = 5e6
    active_stretch_slope: float = 4.9

    def __post_init__(self):
        if any(not isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError('H-O parameters must be positive and finite')


def cofactor(F):
    return torch.stack((torch.linalg.cross(F[..., 1, :], F[..., 2, :]),
                        torch.linalg.cross(F[..., 2, :], F[..., 0, :]),
                        torch.linalg.cross(F[..., 0, :], F[..., 1, :])), -2)


def invariants(F, fiber, sheet):
    J = determinant3(F)
    Ff = (F @ fiber[..., None]).squeeze(-1)
    Fs = (F @ sheet[..., None]).squeeze(-1)
    I1 = F.square().sum((-2, -1))
    return J, I1, I1*J.pow(-2/3), Ff, Fs, Ff.square().sum(-1), Fs.square().sum(-1), (Ff*Fs).sum(-1)


def ho_energy(F, fiber, sheet, parameters):
    """Passive W, including kappa*(ln J)^2; retain the UFL constant term."""
    p = parameters
    J, _, I1bar, _, _, I4f, I4s, I8 = invariants(F, fiber, sheet)
    ef, es = (I4f-1).clamp_min(0), (I4s-1).clamp_min(0)
    return (p.a/(2*p.b)*torch.exp(p.b*(I1bar-3))
            + p.a_f/(2*p.b_f)*torch.expm1(p.b_f*ef.square())
            + p.a_s/(2*p.b_s)*torch.expm1(p.b_s*es.square())
            + p.a_fs/(2*p.b_fs)*torch.expm1(p.b_fs*I8.square())
            + p.kappa*torch.log(J).square())


def active_energy(F, fiber, tension, parameters):
    """Potential at fixed prescribed tension; not passive stored energy."""
    stretch = torch.linalg.vector_norm((F @ fiber[..., None]).squeeze(-1), dim=-1)
    slope = parameters.active_stretch_slope
    return tension*((1-slope)/2*(stretch.square()-1)+slope/3*(stretch.pow(3)-1))


def ho_pk1(F, fiber, sheet, parameters, tension=0.):
    """Analytic diff(W,F) + kappa*ln(I3)*inv(F).T + prescribed active PK1."""
    p = parameters
    J, I1, I1bar, Ff, Fs, I4f, I4s, I8 = invariants(F, fiber, sheet)
    invT = cofactor(F)/J[..., None, None]
    isotropic = p.a*torch.exp(p.b*(I1bar-3))*J.pow(-2/3)
    P = isotropic[..., None, None]*(F-I1[..., None, None]/3*invT)
    ef, es = (I4f-1).clamp_min(0), (I4s-1).clamp_min(0)
    outer = lambda u, v: u[..., :, None]*v[..., None, :]
    P = P + (2*p.a_f*ef*torch.exp(p.b_f*ef.square()))[..., None, None]*outer(Ff, fiber)
    P = P + (2*p.a_s*es*torch.exp(p.b_s*es.square()))[..., None, None]*outer(Fs, sheet)
    P = P + (p.a_fs*I8*torch.exp(p.b_fs*I8.square()))[..., None, None]*(outer(Ff, sheet)+outer(Fs, fiber))
    P = P + (2*p.kappa*torch.log(J))[..., None, None]*invT
    active = tension*(1+p.active_stretch_slope*(torch.sqrt(I4f)-1))
    return P + active[..., None, None]*outer(Ff, fiber)


@dataclass(frozen=True)
class RealLVLoads:
    period: float = .8
    pressure_kpa: float = 1.067
    pressure_increment_kpa: float = 13.46
    tension_kpa: float = 84.26

    def __post_init__(self):
        if self.period != .8 or any(not isfinite(v) or v < 0 for v in vars(self).values()):
            raise ValueError('this piecewise waveform has the supplied 0.8 s period and nonnegative amplitudes')

    def at(self, time):
        from math import exp
        if not isfinite(time) or time < 0:
            raise ValueError('load time must be finite and nonnegative')
        # Remove binary roundoff at exact cycle boundaries, not a ramp reset
        # or smoothing of the discontinuity present in the supplied expression.
        cycle = round(time/self.period)
        phase = 0. if abs(time-cycle*self.period) <= 1e-12 else time % self.period
        if phase < .2:
            pressure, tension = self.pressure_kpa*phase/.2, 0.
        elif phase < .5:
            pressure, tension = self.pressure_kpa, 0.
        else:
            offset = phase-.5 if phase < .65 else .8-phase
            pressure = self.pressure_kpa+self.pressure_increment_kpa*(1-exp(-offset*offset/.004))
            tension = self.tension_kpa*(1-exp(-offset*offset/.005))
        return pressure*10000., tension*10000.
