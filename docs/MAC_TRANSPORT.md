# Centered MAC transport checks

This page describes `explicit-lagged` stepping. The real-LV
`implicit-newton` option uses backward Euler for transport and new-time solid
forces; its C/D/A values are monitors, not explicit-screen vetoes. See
[coupled Newton stepping](MAC_IMPLICIT.md) for its equations and scope.

The 3D MAC backend retains its centered conservative momentum fluxes, explicit
physical viscosity, forward Euler predictor and pressure projection. This change
affects acceptance checks and diagnostics, not the update equations, IB kernel,
consistent FE mass, solid material, loads or fluid viscosity.

## Current policy

Let `U_i = max(abs(velocity_i))`, `nu = mu/rho`, and `h_i` be MAC spacing.
Before the predictor, require:

| Metric | Definition | Operational limit |
| --- | --- | --- |
| Courant number | `C = dt * sum(U_i/h_i)` | 0.25 |
| Viscous number | `D = nu * dt * sum(1/h_i^2)` | 0.25 |
| Advection/diffusion number | `A = dt * sum(U_i^2)/(2*nu)` | 0.25 |

The viscous bound also accommodates the larger boundary-adjacent diagonal from
odd tangential ghost values. Its previous limit is unchanged. Cell Reynolds
number `max(U_i*h_i/nu)` is monitored, with no hard cutoff at 1. It concerns
spatial resolution and possible oscillations; it is not a time-step bound.

The fused kernel calculates C, A and cell Re from the same three peak reductions
and transfers its scalar checks together. Successful fused steps add no extra
CPU/GPU synchronization. Extra peak diagnostics are collected only on rejection.

## Model analysis and safety margin

For constant-coefficient scalar advection/diffusion with periodic boundaries,
forward Euler and central differences in one dimension have the familiar
conditions `D <= 1/2` and `C^2 <= 2D`. See the primary
[BYU course notes](https://ignite.byu.edu/cbe541/lectures/advection_diffusion/).
The analysis is for the combined operator: explicit Euler with centered **pure
advection** is unstable, even at small CFL.

For the corresponding multidimensional frozen model, define
`D_i=nu*dt/h_i^2`, `C_i=U_i*dt/h_i`. The Fourier amplification factor is

```text
g(theta) = 1 - 4*sum(D_i*sin(theta_i/2)^2)
             - i*sum(C_i*sin(theta_i)).
```

Here is the multidimensional extension used to motivate the screen, rather than
a claim that the one-dimensional notes prove stability of this nonlinear MAC code.
Put `S=sum(D_i*sin(theta_i/2)^2)`, `D=sum(D_i)` and
`Q=sum(C_i^2/D_i)=dt*sum(U_i^2)/nu=2A`.
Weighted Cauchy–Schwarz and Jensen bounds give

```text
|imag(g)|^2 <= 4*Q*S*(1-S/D),
|g|^2 - 1 <= 4*S*((Q-2) + (4-Q/D)*S).
```

The expression in parentheses is linear in `S`, with endpoint values `Q-2`
at `S=0` and `4D-2` at `S=D`. Thus `D<=1/2`, `Q<=2` are sufficient for
`|g|<=1` in this frozen periodic model. The operational policy uses `D<=1/4`
and `A<=1/4`, leaving margin against the model's limits `D<=1/2`, `A<=1`.
The A margin is an engineering safety choice, not a derived sharp threshold
for the nonlinear closed-wall IB-FSI system.

## Limits of the check

Variable velocity gradients, nonlinear conservative momentum transport, wall
conditions, explicit elastic forcing and lagged IB coupling are beyond that
constant-coefficient Fourier argument. Passing the screen neither proves full
FSI stability nor establishes spatial accuracy, particularly at high cell Re.
Deformation, finite-value, IB-support, displacement and solver residual checks
remain active. Short runs with halved dt and inspection of local flow fields
are required before claiming recovery of a complete trajectory.

Real-LV reports and restart segments identify the policy as
`centered-explicit-advection-diffusion-v1`. New CSV rows report state-time C, A
and cell Re; step diagnostics describe the input state used by that step.
The checkpoint inspector evaluates the current policy but preserves the old
failure message and separately marks whether the retired Re<=1 rule would trip.

## Recovery experiment

`validation/compare_real_lv_transport.py` forks a common self-contained saved
state into original-dt and half-dt directories, preserving the input file.
The half-dt branch changes step numbering and resamples the initial force at
`start_time-half_dt` to remain consistent with the existing lagged load order.
It compares continuations from a common state, not full-history temporal
convergence. No source mesh import or repetition of previously accepted steps
is required. See the [real-LV demo](../demo/real_lv_fsi/README.md) for commands.

For a longer continuation, `run_mac.py --resume CHECKPOINT --resume-dt DT
--output NEW_DIRECTORY --cycles 3` refines the saved dt by an integer factor.
It preserves accepted x/u/p, resamples the lagged force at `current_time-DT`,
reindexes steps and preserves output spacing in seconds. The source directory
is untouched. The new report records restart provenance and elapsed time for
the new branch; its self-contained checkpoint supports ordinary further resumes.
The target is absolute simulation time, not an added duration. This option
neither relaxes the operational screen nor guarantees that future loading will
remain within it. A value just above 0.25 is a screen violation, not evidence
of crossing a derived nonlinear instability boundary.
