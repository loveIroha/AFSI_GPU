# CN–AB2 with midpoint finite-element/immersed-boundary coupling

This guide describes `coupling.scheme="cnab-midpoint"`, the explicit-force
comparison mode. The real-LV demo now defaults to the [implicit-elastic
extension](MAC_SEMIIMPLICIT.md), `cnab-semiimplicit`. The time
discretization follows Griffith–Luo, Appendix A, equations (71)–(81), and the
CN–AB treatment described by Gao et al. for immersed FE ventricular models.
This replaces the RK3-fluid/first-order-solid combination for this demo.
It is a semi-implicit fluid method with explicitly predicted structural force;
it does not perform a nonlinear coupled Newton solve.

## Equations and sampling times

Let `x` denote current FE coordinates, `u` MAC velocity, `p` the pressure
multiplier, `N` conservative momentum convection, `L` the no-slip velocity
Laplacian, and `D/G` the staggered divergence/gradient. `J(x)` and `S(x)` are
the existing adjoint quadrature interpolation and spreading operators, with
the assembled consistent FE mass matrix. Each ordinary step computes:

```text
x_half = x_n + dt/2 * J(x_n) u_n
F_half = F(x_half, t_n + dt/2)
f_half = S(x_half) F_half
N_half = 3/2 N(u_n) - 1/2 N(u_{n-1})

rho (u_new-u_n)/dt + rho N_half
    = -G p_half + mu L((u_new+u_n)/2) + f_half
D u_new = 0

x_new = x_n + dt J(x_half) ((u_new+u_n)/2)
```

Passive H–O stress, active stress, follower pressure and basal traction are
all evaluated at the predicted half-step configuration/time. Interpolation
and spreading use the same half-step IB geometry. Endpoint force is saved
for consistent checkpoint/output fields; it is not the force used to advance
this step. `last_solver_info.used_force_time_s` identifies the half-time force,
while CSV `force_time_s` identifies the saved endpoint force.

For startup, an old-force CN fluid predictor supplies `u_tilde`. The provisional
structural position is `x_tilde=x_n+dt J(x_n)u_n`, so its average with `x_n`
gives `x_half`. The corrected solve uses `N((u_n+u_tilde)/2)` and the half-step
force. This is the predictor/corrector startup in Appendix A.2. Subsequent
steps use AB2. No previous advection term is guessed from zero.

## No-slip CN Stokes solve

Set `alpha=mu*dt/(2*rho)` and `H=I-alpha*L`. Each fluid solve checks the actual
saddle equations:

```text
H u_new + dt/rho G p_half = b
D u_new = 0
b = (I+alpha L) u_n + dt*(-N_half + f_half/rho)
```

At no-slip walls, `H` and `G` do not commute. Projecting a Helmholtz predictor
once does not in general solve these momentum equations. The implementation
iterates on the pressure Schur residual using the existing Neumann geometric
multigrid solver and the approximate inverse `A0^-1 + alpha I`, where
`A0=-D G`. After each pressure correction, velocity is recomputed with the
actual wall Helmholtz operator. A fixed Jacobi polynomial approximates `H^-1`
to a conservative contraction bound. Both true momentum residual and true
divergence must pass their tolerances; a failed solve is not accepted.

This is a pressure-Schur Richardson iteration with multigrid preconditioning.
It implements the same CN velocity/pressure equations, but **does not reproduce
IBAMR's FGMRES Stokes solver**. Configuration limits and Helmholtz accuracy
are under `coupling.cnab`. The report records Schur iterations, all nested
Poisson cycles, and the true momentum/divergence residuals. Failure details
are stored under `failure.stokes`.

Pressure is the zero-mean **half-time multiplier**, not endpoint pressure.
VTK pairs it with the accepted endpoint state for visualization; its sampling
time is `t-dt/2` after an accepted step. Closed-box homogeneous Neumann pressure
and the existing no-slip velocity stencil are retained.

## Convection and GPU execution

The default `coupling.cnab.advection="ppm"` uses monotone parabolic reconstruction
and upwind conservative fluxes on the staggered momentum control volumes.
It limits reconstructed states at extrema/discontinuities. It is a
method-of-lines PPM reconstruction with AB2 time quadrature: **IBAMR's
characteristic tracing, PPM variants and AMR machinery are not replicated**.
`"centered"` remains available for controlled comparisons.

PyTorch compiles the tensor convection, Helmholtz and residual kernels on CUDA.
The existing compact IB kernels, consistent CSR mass solve with CUDA Graphs,
H–O P1 solid execution, and Triton/CUDA Graph pressure multigrid are reused.
Solver convergence decisions still synchronize small scalars. Startup takes
two CN fluid solves; ordinary steps use one CN solve and three FE mass solves
(old-velocity predictor, half-force spreading, average-velocity interpolation).
Each CN solve can require several nested pressure solves, which the benchmark
counts. Speedup over earlier methods must be measured on the target GPU.

## Time step, restart and scope

The requested real-LV defaults are `dt=1e-4 s`, fluid grid `128^3`, three
`0.8 s` loading periods: 24,000 accepted steps to `2.4 s`.
Materials, meshes, fiber/sheet fields, CSR mass, physical viscosity, loads,
and basal constraints are unchanged by this time-scheme selection.

Convection remains explicit. The existing conservative screen
`dt*sum(max(abs(u_i))/h_i) <= 0.25` is applied; `D`, `A` and cell Reynolds
number are monitors only for this scheme. CN removes the *explicit viscosity*
time-step restriction. Predicted elastic forces still have a stiffness/IB
stability restriction. A small convection CFL, successful Stokes solve or
time step used in another paper cannot prove full nonlinear FSI stability.
The supplied cycle pressure reset also remains discontinuous; second-order
temporal accuracy applies to smooth solutions/load intervals.

Checkpoints store `N(u_n)` as the previous term for the next AB2 step, protected
by the existing checksum. Same-scheme, same-dt restart preserves it. Scheme
changes and integer dt refinement deliberately reset it and perform a new
startup predictor/corrector. Old schemes/checkpoints remain supported.
Trial failures leave accepted coordinates, velocities and AB2 history intact.

## Verification and performance experiment

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_mac_cnab.py

CUDA_VISIBLE_DEVICES=0 python -u validation/benchmark_real_lv_schemes.py \
  --checkpoint results/real_lv_cnab_3cycles/checkpoint.npz \
  --schemes cnab-midpoint explicit-rk3 --device cuda --warmup 3 --steps 20 \
  --output results/real_lv_cnab_performance/report.json
```

The benchmark advances temporary branches without per-step files or changing
the source checkpoint. Different time schemes are not numerically equivalent;
this comparison measures cost, not full-cycle stability or solution accuracy.
Tests cover manufactured wall Stokes solutions (including an explicit-diffusion
limit violation), second-order CN–AB2 temporal convergence, PPM bounds,
reference/optimized FE/IB agreement and power pairing, failed-step state/history
preservation, restart, and nested-solve counts. CUDA cases run when available.

## References

- [Griffith & Luo: Hybrid finite difference/finite element immersed boundary method, Appendix A](https://arxiv.org/html/1612.05916).
- [Gao et al.: Dynamic finite-strain modelling of the human left ventricle in health and disease using an immersed boundary-finite element method (2014), §2.4.4](https://www.maths.gla.ac.uk/~xl/gao2014.pdf).
- [IBAMR explicit IB time integrator](https://github.com/IBAMR/IBAMR/blob/master/src/IB/IBExplicitHierarchyIntegrator.cpp).
- [IBAMR staggered Navier–Stokes time integrator](https://github.com/IBAMR/IBAMR/blob/master/src/navier_stokes/INSStaggeredHierarchyIntegrator.cpp).
