# Real-LV coupled Newton time stepping

The real-LV demo supports `--coupling implicit-newton`. This changes the time
discretization, rather than an execution optimization. Legacy checkpoints and
the generated LV/valve demos retain their existing explicit scheme by default.

## Equations

At each accepted position `x_n`, build the existing quadrature FE/MAC transfer
operators `S_n` and `J_n` using the same Peskin kernel and reference quadrature.
Both include the existing consistent CSR FE mass solve. Freeze these operators
during that time step. Solve

```text
u - u_n + dt*[N_h(u) - nu*L_h(u) + G_h*p/rho - S_n*b(x,t_new)/rho] = 0,
D_h*u = 0,
x - x_n - dt*J_n*u = 0.
```

Convection, viscosity, H-O deformation, pressure traction, active tension and
basal force all use the new state/time. The transfer geometry alone is frozen
at the old position: this is a geometry-semi-implicit strong coupling, not a
fully new-position delta-kernel linearization. The force sampling time is
`force_time=time` in accepted implicit checkpoints, rather than `time-dt`.

Eliminate x using the kinematic equation. Eliminate pressure using the existing
MAC Poisson projection `P`. The reduced Newton residual is

```text
R(u) = u - P[u_n + dt*(-N_h(u) + nu*L_h(u) + S_n*b(x_n+dt*J_n*u,t_new)/rho)].
```

Solving this residual couples the new velocity, deformation and force; pressure
is recovered from the eliminated equation. It is equivalent to solving the
above coupled discrete system to the linear/nonlinear solver tolerances.
Pressure elimination does not reduce the coupling to a one-pass force update.

## Assembled FE tangent and Newton solve

`ho_tangent.py` differentiates small H-O stress and follower face kernels with
`torch.func.jacrev`/`vmap`, contracts with P1 gradients/reference volumes and
assembles the nodal-force Jacobian `K=db/dx` into a cached CSR pattern. It includes
the supplied active stretch term, nonsymmetric open-surface follower-pressure
derivative and radial/z basal penalty. Assembly and CSR multiplication remain
on the tensor device, with bounded element chunks; CUDA compiles the local
derivative kernels. No dense global FE matrix is constructed.

The exact reduced Jacobian action is

```text
DR(u)*v = v - P[dt*(-DN_h(u)*v + nu*L_h(v) + S_n*K*(dt*J_n*v)/rho)].
```

MAC stencil actions and pressure elimination do not assemble a giant coupled
fluid/solid matrix. The solid FE tangent is explicitly assembled; iterative
mass/pressure solves are used as linear operators, rather than differentiated
through their stopping logic. Their actual residual checks remain enabled.

Newton uses restarted GMRES on the tensor device and Armijo backtracking, with
positive-J, surface/cavity, finite-value, IB support and solid displacement
checks on trial states. The final Newton velocity itself is rechecked against
the complete nonlinear residual; pressure, position and force are recovered
at that same velocity and new load time. Failed iterations preserve the previous
accepted simulation checkpoint and record the Newton failure history.

### Accept the Newton state without an additional fixed-point step

Writing `G(u)=P[right(u)]`, the nonlinear equation is `R(u)=u-G(u)=0`.
Once Newton has converged to a finite tolerance, replacing `u` by `G(u)` is
an additional undamped Picard iteration. It is not merely projecting `u` to
remove divergence. Newton convergence does not imply that this fixed-point
map is contractive. For example, `G(u)=b-20*u` gives `R(u)=21*u-b` and
`R(G(u))=-20*R(u)`: a residual of `5e-10` becomes `1e-8`, even though the
Newton state already met a `1e-9` target. A stiff solid/IB response can likewise
amplify a small terminal residual. The driver therefore preserves the Newton
velocity and independently verifies the equations at that unchanged state.

Continuity is checked as well. From `u=G(u)+R(u)`, the discrete MAC divergence
satisfies `||D u|| <= ||D G(u)|| + 2*sqrt(sum(h_c^-2))*nonlinear_tolerance`.
The first term includes the pressure solve's finite accuracy; a floating-point
roundoff allowance is added to the bound. Actual divergence, its acceptance
bound, the fresh momentum residual and all final inner-solve residuals are
recorded under `nonlinear.acceptance`. Failed final checks preserve the last
accepted checkpoint and write the same details to `failure.coupled_acceptance`.
This is an algebraic acceptance check, not a mesh-convergence or physical-error
estimate. The final nonlinear target and the time discretization are unchanged.

The mass and pressure inverses are iterative approximations. Their absolute
stopping tolerances can break numerical homogeneity when a unit Arnoldi vector
is replaced by a very small Newton correction. The implicit Jacobian action
therefore evaluates `||v|| * DR(u)[v/||v||]`, with an exact zero result for `v=0`.
This keeps inner-solve inputs at a common scale; it does not rescale the
nonlinear residual, alter the assembled tangent, or change the time equations.

By default the implicit Newton solver also sets each GMRES absolute target to
`max(linear.atol, 0.2 * nonlinear_tolerance)`. Its relative target still applies,
and GMRES still checks the true linear residual. For an outer target of `1e-9`,
the absolute floor is `2e-10`, rather than asking the inner solve to resolve
irrelevant corrections far below the final coupled tolerance. This floor is
strictly below the outer target and current unconverged residual. Armijo and
the final complete nonlinear residual check remain mandatory. Generic Newton
defaults keep this feature disabled (`linear_tolerance_fraction=0`).
This coordination follows the principle of avoiding oversolving in
[inexact Newton methods](https://sundials.readthedocs.io/en/v7.5.0/kinsol/Mathematics_link.html#stopping-criteria-for-iterative-linear-solvers);
it is not an implementation of the Eisenstat--Walker adaptive forcing formula.

The quadrature-based transfer and its power identity follow the unified weak
form described by [Griffith and Luo (2017)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5650596/).
This implementation uses first-order backward Euler and step-frozen IB
geometry; it does not claim to reproduce every time-integration choice in
that paper or to establish unconditional stability for this H-O case.

## Transport diagnostics and scope

The explicit predictor's CFL/D/A rejection bounds do not apply to this
backward-Euler transport solve. The implicit driver never executes the explicit
predictor, and does not relax its checks. C, A, D and cell Re are still reported
as diagnostic measures under policy `backward-euler-frozen-ib-newton-v1`.
The checkpoint inspector distinguishes implicit diagnostics from an actual
explicit-screen rejection.

Implicit time stepping addresses lagged elastic feedback and explicit transport
restrictions. It does not establish spatial convergence, remove the supplied
periodic load discontinuity, enforce exact incompressibility of the penalized
solid, or correct errors already accumulated before a scheme-switch checkpoint.
Large local Reynolds numbers still warrant inspection of spatial flow quality.
Finite dt introduces accuracy error; successful Newton convergence alone does
not certify the physical trajectory or a three-cycle steady state.

## Configuration and cost

The public `RealLVConfig.coupling` object/JSON section sets the scheme, Newton
tolerances, line search, GMRES restart/iterations and tangent chunk size. Example:

```json
{"coupling": {"scheme": "implicit-newton", "tangent_chunk_size": 2048,
  "newton": {"rtol": 1e-8, "atol": 1e-9, "max_iterations": 12,
    "linear_tolerance_fraction": 0.2,
    "linear": {"rtol": 0.01, "restart": 12, "max_iterations": 120}}}}
```

Every Newton/GMRES iteration incurs IB mass solves and pressure projection,
and each Newton iteration assembles a solid tangent. Each step therefore costs
more than a single explicit step. No full-scale speedup is claimed. On a 128³
grid, the default 12-vector GMRES restart allocates approximately 1.3 GB for
its two bases alone; FE assembly, state, transfer and solver storage add to it.
The optimized compact transfer, compiled kernels and mass/pressure CUDA graphs
are reused when configured. There is currently no coupled block preconditioner.

Reports/CSV record Newton iteration count, true residual and acceptance
tolerance. GMRES and line-search histories are in `last_solver_info.nonlinear`
or `failure.nonlinear`. Accepted GMRES history entries include configured and
effective absolute targets and the outer nonlinear target. Real-LV JSON or
checkpoint Newton sections without `linear_tolerance_fraction` inherit the
current coupling default `0.2`. An explicitly stored value is preserved;
set it to zero in a new-run JSON to disable the floor for a strict comparison.
GPU benchmark/trajectory validation is still required
on the actual real-LV mesh; local development verification uses CPU execution
plus full-graph tracing, with explicit CUDA test cases skipped when unavailable.

For running and switching checkpoints, see the [real-LV demo](../demo/real_lv_fsi/README.md).
