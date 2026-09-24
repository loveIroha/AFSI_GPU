"""Non-incremental Chorin step with explicit convection and implicit viscosity.

All outer velocity DOFs are Dirichlet. Pressure has homogeneous Neumann data
and one pinned DOF as its gauge; net boundary flux must be compatible. This
matches the form of the selected afsi example, not an exact discrete Helmholtz
projection. No IB coupling, time-dependent BC interpolation or adaptive dt here.
"""
from dataclasses import dataclass, asdict
from math import isfinite
from numbers import Integral
import torch
from .solvers import SolverOptions, pcg, operator_diagonals


@dataclass(frozen=True)
class StepResult:
    velocity: torch.Tensor
    pressure: torch.Tensor
    tentative_velocity: torch.Tensor
    diagnostics: dict


class ChorinSolver:
    def __init__(self, operators, *, dt, rho=1., mu=1., pressure_dof=0,
                 pressure_value=0., options=None):
        for name, value in (('dt', dt), ('rho', rho), ('mu', mu)):
            if not isfinite(value) or (value < 0 if name == 'mu' else value <= 0):
                raise ValueError(f'invalid {name}')
        if (not isinstance(pressure_dof, Integral) or isinstance(pressure_dof, bool) or
                not 0 <= pressure_dof < len(operators.mesh.pressure_coordinates) or not isfinite(pressure_value)):
            raise ValueError('invalid pressure gauge')
        self.op, self.dt, self.rho, self.mu = operators, float(dt), float(rho), float(mu)
        self.options = SolverOptions() if options is None else options
        self.diagonals = operator_diagonals(operators)
        self.velocity_fixed = operators.mesh.velocity_boundary[:, None].expand(-1, 3)
        P = operators.mesh.pressure_coordinates
        self.pressure_fixed = torch.zeros(len(P), dtype=torch.bool, device=P.device)
        self.pressure_fixed[pressure_dof] = True
        self.pressure_values = P.new_zeros(len(P))
        self.pressure_values[pressure_dof] = pressure_value

    def tentative_action(self, u):
        return self.rho/self.dt*self.op.velocity_mass(u)+self.mu*self.op.velocity_stiffness(u)

    def divergence_l2(self, u):
        op = self.op
        d = torch.einsum('qaj,eaj->eq', op.dN, u[op.mesh.velocity_cells])
        return torch.sqrt(torch.einsum('eq,eq,q->', d, d, op.weights)).item()

    @torch.no_grad()
    def step(self, velocity, *, density=None, nodal_load=None, boundary_values=None,
             pressure_initial=None):
        """Advance one step; supply either Q2 force density or dual nodal load.

        density -> M f follows afsi's density path. nodal_load -> b enters
        directly with no extra mass multiplication. Default force/BC are zero.
        User supplies the NEW step's boundary values when time-dependent.
        """
        op = self.op
        def check(u):
            op._field(u, vector=True)
            if not torch.isfinite(u).all():
                raise ValueError('finite velocity/load data required')
        check(velocity)
        if density is not None and nodal_load is not None:
            raise ValueError('choose density OR nodal_load, never both')
        boundary_values = torch.zeros_like(velocity) if boundary_values is None else boundary_values
        check(boundary_values)
        load = torch.zeros_like(velocity)
        if density is not None:
            check(density)
            load = op.density_load(density)
        elif nodal_load is not None:
            check(nodal_load)
            load = nodal_load
        rhs1 = self.rho/self.dt*op.velocity_mass(velocity)-self.rho*op.convection(velocity)+load
        diag1 = self.rho/self.dt*self.diagonals['velocity_mass']+self.mu*self.diagonals['velocity_stiffness']
        star, info1 = pcg(self.tentative_action, rhs1, diag1, fixed=self.velocity_fixed,
                         values=boundary_values, initial=velocity, options=self.options)
        div_star = op.divergence(star)
        net_flux = div_star.sum().item()
        flux_tolerance = 1e-12+1e-10*div_star.abs().sum().item()
        if abs(net_flux) > flux_tolerance:
            raise ValueError(f'pressure Neumann compatibility violated: net boundary flux={net_flux:.6e}; '
                             'balance boundary velocities before using a single pressure gauge')
        rhs2 = -self.rho/self.dt*div_star
        pressure, info2 = pcg(op.pressure_stiffness, rhs2, self.diagonals['pressure_stiffness'],
            fixed=self.pressure_fixed, values=self.pressure_values, initial=pressure_initial, options=self.options)
        rhs3 = op.velocity_mass(star)-self.dt/self.rho*op.gradient(pressure)
        corrected, info3 = pcg(op.velocity_mass, rhs3, self.diagonals['velocity_mass'],
            fixed=self.velocity_fixed, values=boundary_values, initial=star, options=self.options)
        div_corrected = op.divergence(corrected)
        diagnostic = dict(solves={name: asdict(info) for name, info in
            [('tentative', info1), ('pressure', info2), ('correction', info3)]},
            pressure_full_residual_norm=torch.linalg.vector_norm(rhs2-op.pressure_stiffness(pressure)).item(),
            tentative_divergence_dual_norm=torch.linalg.vector_norm(div_star).item(),
            corrected_divergence_dual_norm=torch.linalg.vector_norm(div_corrected).item(),
            tentative_divergence_l2=self.divergence_l2(star), corrected_divergence_l2=self.divergence_l2(corrected),
            net_flux=net_flux, kinetic_energy=.5*self.rho*(corrected*op.velocity_mass(corrected)).sum().item())
        return StepResult(corrected, pressure, star, diagnostic)
