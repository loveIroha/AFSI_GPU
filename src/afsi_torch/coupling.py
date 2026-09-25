"""Explicit, density-path IB/FEM stepping with afsi's lagged force ordering.

Bootstrap g=0. Step n uses g_n at x_n, solves u_(n+1), updates positions with
H(x_n)u_(n+1), then assembles g_(n+1)=force(x_(n+1), t_n). The source demo
samples prescribed pressure/tension at t_n, not t_(n+1); keep that lag explicit.
No clipping, fixed-solid projection, extra solid mass solve or pressure feedback.
"""
from dataclasses import dataclass
from math import isfinite
import torch
from . import ib


@dataclass(frozen=True)
class CoupledState:
    step: int
    time: float
    x: torch.Tensor
    velocity: torch.Tensor
    pressure: torch.Tensor
    force: torch.Tensor
    force_time: float | None


@dataclass(frozen=True)
class CoupledResult:
    state: CoupledState
    solid_velocity: torch.Tensor
    applied_density: torch.Tensor
    diagnostics: dict


class ExplicitIBStepper:
    def __init__(self, fluid, force, validate, *, max_grid_displacement=.25):
        if not isfinite(max_grid_displacement) or max_grid_displacement <= 0:
            raise ValueError('positive finite displacement limit required')
        self.fluid, self.force, self.validate = fluid, force, validate
        self.grid = fluid.op.mesh.velocity_grid
        self.max_grid_displacement = float(max_grid_displacement)

    @torch.no_grad()
    def initialize(self, x, velocity=None):
        """Source-compatible zero-force bootstrap; force is NOT evaluated here."""
        self.validate(x)
        ib.prepare_stencil(x, self.grid)
        X = self.fluid.op.mesh.velocity_coordinates
        if x.device != X.device or x.dtype != X.dtype:
            raise ValueError('solid and fluid must share device and dtype')
        velocity = torch.zeros_like(X) if velocity is None else velocity
        self.fluid.op._field(velocity, vector=True)
        if not torch.isfinite(velocity).all():
            raise ValueError('initial velocity must be finite')
        return CoupledState(0, 0., x.clone(), velocity.clone(),
            X.new_zeros(len(self.fluid.op.mesh.pressure_coordinates)), torch.zeros_like(x), None)

    @torch.no_grad()
    def initialize_equilibrium(self,x,*,force_tolerance):
        """Opt-in prescribed-traction equilibrium start, with u=0 and p=0.

        Retains actual total nodal force, including its small residual. The
        reference geometry remains owned by the force callback. p=0 is the
        background solver field, not the prescribed cavity traction pressure.
        The original initialize() continues to provide zero-force bootstrap.
        """
        if not isfinite(force_tolerance) or force_tolerance<=0:
            raise ValueError('positive finite equilibrium force tolerance required')
        if self.fluid.pressure_values.count_nonzero().item():
            raise ValueError('equilibrium initializer currently requires zero pressure gauge')
        state=self.initialize(x)
        force=self.force(state.x,0.)
        if (force.shape!=x.shape or force.dtype!=x.dtype or force.device!=x.device or
                not torch.isfinite(force).all()):
            raise ValueError('equilibrium force must be finite and match solid coordinates')
        norm=torch.linalg.vector_norm(force).item()
        if norm>force_tolerance:
            raise ValueError(f'preload is not balanced under the initial load: {norm:.6g} > {force_tolerance:.6g}')
        return CoupledState(0,0.,state.x,state.velocity,state.pressure,force.detach().clone(),0.)

    @torch.no_grad()
    def step(self, state, *, boundary_values=None):
        dt = self.fluid.dt
        if (not isinstance(state.step, int) or state.step < 0 or
                not isfinite(state.time) or abs(state.time-state.step*dt) > 1e-12*max(1., abs(state.time))):
            raise ValueError('state time/step inconsistent with this fixed dt')
        self.validate(state.x)
        old_stencil = ib.prepare_stencil(state.x, self.grid)
        if not torch.isfinite(state.force).all():
            raise ValueError('stored solid force must be finite')
        density = ib.spread_density(state.force, old_stencil)
        flow = self.fluid.step(state.velocity, density=density, boundary_values=boundary_values,
                               pressure_initial=state.pressure)
        solid_velocity = ib.interpolate(flow.velocity, old_stencil)
        displacement = dt*solid_velocity
        fraction = (displacement.abs()/displacement.new_tensor(self.grid.spacing)).max().item()
        if not isfinite(fraction) or fraction > self.max_grid_displacement:
            raise ValueError(f'explicit solid displacement {fraction:.6g} grid spacings exceeds '
                             f'{self.max_grid_displacement}; reduce dt (step not accepted)')
        x_new = state.x+displacement
        self.validate(x_new)
        new_stencil = ib.prepare_stencil(x_new, self.grid)
        new_force = self.force(x_new, state.time)
        if (new_force.shape != x_new.shape or new_force.device != x_new.device or
                new_force.dtype != x_new.dtype or not torch.isfinite(new_force).all()):
            raise ValueError('force callback must return finite integrated (N,3) nodal forces')
        # Validate the next spread before accepting the state. Do not cache the
        # old stencil: supports change whenever nodes cross lattice cells.
        next_density = ib.spread_density(new_force, new_stencil)
        if not torch.isfinite(next_density).all():
            raise ValueError('next force density is nonfinite')
        solid_power = (solid_velocity*state.force).sum()
        lattice_power = (flow.velocity*density).sum()*self.grid.cell_volume
        fe_power = (flow.velocity*self.fluid.op.density_load(density)).sum()
        diagnostics = dict(fluid=flow.diagnostics, time_s=(state.step+1)*dt,
            used_force_time_s=state.force_time, next_force_time_s=state.time,
            applied_force_norm_dyn=torch.linalg.vector_norm(state.force).item(),
            next_force_norm_dyn=torch.linalg.vector_norm(new_force).item(),
            max_displacement_cm=torch.linalg.vector_norm(displacement, dim=-1).max().item(),
            max_grid_displacement=fraction,
            solid_power_erg_per_s=solid_power.item(), lattice_power_erg_per_s=lattice_power.item(),
            lattice_power_error=abs((lattice_power-solid_power).item()),
            fe_power_erg_per_s=fe_power.item(), fe_minus_solid_power=(fe_power-solid_power).item(),
            spread_force_balance_max_abs=(density.sum(0)*self.grid.cell_volume-state.force.sum(0)).abs().max().item())
        new_state = CoupledState(state.step+1, (state.step+1)*dt, x_new, flow.velocity,
                                 flow.pressure, new_force, state.time)
        return CoupledResult(new_state, solid_velocity, density, diagnostics)
