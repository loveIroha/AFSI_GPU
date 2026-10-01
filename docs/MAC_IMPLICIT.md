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
checks on trial states. The final projected state is rechecked against the
complete nonlinear residual; its position and force are consistent with its
accepted velocity and new load time. Failed iterations preserve the previous
accepted simulation checkpoint and record the Newton failure history.

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
or `failure.nonlinear`. GPU benchmark/trajectory validation is still required
on the actual real-LV mesh; local development verification uses CPU execution
plus full-graph tracing, with explicit CUDA test cases skipped when unavailable.

For running and switching checkpoints, see the [real-LV demo](../demo/real_lv_fsi/README.md).
