# Reduced solid-node CN–AB2 / implicit midpoint FE–IB coupling

`--coupling cnab-semiimplicit` is the real-LV demo default. It keeps the
CN viscosity, explicit AB2 convection, wall Stokes equations and predicted
midpoint IB geometry documented in [MAC_CNAB.md](MAC_CNAB.md). **Elastic force
is now evaluated at the solved structural midpoint**, rather than just at its
explicit prediction. This is an implicit-elastic extension of that scheme;
it is not presented as an exact reproduction of Gao, Griffith–Luo or IBAMR.

## Reduced nonlinear equations

`x_n` is the accepted current FE position. First predict the IB geometry:

```text
x_hat = x_n + dt/2 J(x_n) u_n
J_hat = J(x_hat), S_hat = S(x_hat)
N_half = 3/2 N(u_n) - 1/2 N(u_{n-1})
H = I - mu*dt/(2*rho) L
b0 = (I + mu*dt/(2*rho) L) u_n - dt N_half
```

Only the nodal midpoint correction `y` is an outer unknown. For each residual
evaluation, set `x_m=x_hat+y`, compute the actual nonlinear force and solve:

```text
H u(y) + dt/rho G p(y) = b0 + dt/rho S_hat F(x_m,t_n+dt/2)
D u(y) = 0
R(y) = x_m - x_n - dt/4 J_hat (u_n+u(y)) = 0
x_new = 2*x_m - x_n
```

This includes H–O passive/volumetric stress, active stress, open endocardial
follower pressure and basal radial traction. For the supplied mesh, `y` has
**80,667 entries**, compared with **6,340,608** velocity entries in the older
backward-Euler `implicit-newton` scheme. Fluid arrays still occupy the full
128³ grid; this reduction concerns the outer Krylov basis, not grid resolution.

Let `T=dF/dx` be the assembled CSR **nodal-force derivative** (restoring terms
have a negative sign). The exact reduced Jacobian action is:

```text
dforce = T v
H du + dt/rho G dp = dt/rho S_hat dforce,  D du = 0
DR(y) v = v - dt/4 J_hat du
```

Small element/face stress derivatives are summed into a cached global CSR
pattern on the tensor device. The follower tangent is nonsymmetric, so the
outer solve uses GMRES. No dense global tangent is formed; no autograd passes
through mass, pressure or Stokes iterations. The fluid response and Krylov
directions are normalized and rescaled to control absolute inner tolerances.

A right diagonal preconditioner in the CSR Newton path approximates local mobility using row-sum
mass and the magnitude of the tangent diagonal. **This does not lump the mass
in the discretization**: both `J_hat` and `S_hat`, every nonlinear residual,
and every Jacobian action still solve the assembled consistent FE mass system.
The preconditioner is an approximation; its benefit must be measured.

## Acceptance, history and stability

The demo defaults to `coupling.semiimplicit_solver="anderson-newton"` (CLI
`--nonlinear-solver anderson-newton`). This solves the **same nonlinear midpoint
residual** and retains the original acceptance tolerance. It first uses up to
six Anderson iterations with a four-column history. Each evaluation needs one
complete Stokes solve and two consistent FE mass solves, but no Jacobian action
or solid tangent assembly. A normalized, regularized small secant least-squares
system is solved on the tensor device; regularization changes the iterative
correction, not the physical equations.

The acceleration's intermediate probes may have bounded nonmonotone residuals
to learn a mildly noncontractive map. Invalid geometry, solve failure, excessive
residual growth or exhausted budget triggers CSR Newton from the best iterate.
These probes are **not accepted simulation states**. The original residual
target is preserved when falling back from a better initial guess; it is never
recomputed as a looser relative target. Settings are under `coupling.anderson`.

CSR Newton/GMRES uses residual-norm backtracking and assembles a tangent per
Newton iteration. It remains selectable with `--nonlinear-solver newton`.
Both paths can accept a sufficiently small initial residual without unnecessary
iterations. A single update is not advertised as a converged nonlinear solve.

The final residual is recomputed independently. The actual CN Stokes momentum
and divergence must also pass their existing tolerances. The solved midpoint
is retained; an extra unguarded Picard position update is not applied. The
endpoint kinematic residual is twice the checked midpoint residual.
The `nonlinear` report section records acceptance, unknown count, Anderson and
Newton history/iterations, fallback reasons, tangent assemblies, Jacobian
actions and actual Stokes calls. Logs and CSV show separate AA/Newton/GMRES
counts. The last fluid solve's MG count is not the total cost of a coupled step.

Startup uses the existing old-force fluid predictor to sample midpoint
convection. Normal steps preserve AB2 history. Same-dt/same-scheme restart
restores it; scheme changes and dt refinement reset it. These changes do not
erase errors already present in an old run. Saved nodal force remains the
endpoint force; the advancing force is evaluated at half time.

Convection remains explicit: `dt*sum(max(abs(u_i))/h_i) <= 0.25` is screened at
both the input state and the candidate accepted velocity. CN removes explicit
viscosity bounds; nonlinear elasticity is solved implicitly. Frozen IB
geometry, explicit convection, spatial resolution and nonlinear conditioning
still matter. `dt=1e-4` and a converged step are not unconditional guarantees
of three-cycle stability or physical accuracy. The supplied pressure reset
at each 0.8 s boundary is unchanged.

## Failure diagnosis and saved states

Accepted `x/u/p`, endpoint force and AB2 history are never replaced by a failed
trial. `checkpoint.npz` records the last accepted state on failure.
`recovery_checkpoint.npz` retains the most recent scheduled snapshot (initial
state and each `checkpoint_every` accepted steps); failure saves do not overwrite
it. It is a restart candidate, not a certificate that the state is healthy.

Failures after the reduced problem is prepared additionally record
`failure.coupled_acceptance.local`: accepted and trial velocity peaks and
coordinates, passive-without-volume/volume/active/follower/basal nodal-force
norms and peaks, force decomposition error, and nearby cell determinant and
reference volume. A trial is explicitly labelled as potentially rejected.
These extra diagnostics run **only on failure**. Normal output retains the
existing cadence and sampled power checks.

## GPU verification and staged experiment

Run from the repository root in the `afsi-torch` environment:

```bash
git pull --ff-only
python -m pip install -e ".[test,geometry,mesh,fused]"
CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/test_midpoint_solver.py tests/test_mac_semiimplicit.py tests/test_mac_cnab.py tests/test_mac_implicit.py
```

Tests compare the reduced CSR Jacobian against a finite difference of the
complete coupled residual at an active, deformed H–O state; independently
check wall momentum, divergence and midpoint kinematics; compare optimized
and reference paths; verify restart, recovery snapshots, transactional failure
and force decomposition. A stiff adjoint modal spring on the actual wall
Stokes grid grows with explicit midpoint force and remains energy bounded
with implicit midpoint force. It isolates temporal stiffness and does not
validate real-mesh IB resolution or full-cycle physiology.

First make a short startup checkpoint, then measure warmed solve cost:

```bash
CUDA_VISIBLE_DEVICES=0 python -u demo/real_lv_fsi/run_mac.py \
  --device cuda --coupling cnab-semiimplicit --dt 1e-4 --fluid-cells 128 \
  --end-time 0.005 --output results/real_lv_semiimplicit_startup

CUDA_VISIBLE_DEVICES=0 python -u validation/benchmark_real_lv_schemes.py \
  --checkpoint results/real_lv_semiimplicit_startup/checkpoint.npz \
  --schemes cnab-semiimplicit cnab-midpoint --device cuda --warmup 3 --steps 20 \
  --output results/real_lv_semiimplicit_performance/report.json
```

The benchmark reports total warmed time, **all** nested mass/pressure solves,
tangent assemblies and Jacobian actions, without modifying the checkpoint.
The modes solve different time-discrete equations: it is a cost comparison,
not numerical equivalence. Reducing outer unknowns is not a promised speedup;
extra fluid responses may remain the dominant cost. GPU performance and
long-horizon real-LV stability have to be measured on the target machine.

Continue the new startup run past the earlier 0.1685 s failure:

```bash
run_dir=results/real_lv_semiimplicit_startup
CUDA_VISIBLE_DEVICES=0 nohup /usr/bin/time \
  -f 'elapsed_seconds=%e exit_code=%x' -o "$run_dir/runtime_02s.txt" \
  python -u demo/real_lv_fsi/run_mac.py --device cuda \
  --resume "$run_dir/checkpoint.npz" --end-time 0.2 \
  > "$run_dir/run_02s.log" 2>&1 < /dev/null &
```

After inspecting this stage, continue the same checkpoint first across the
active contraction interval (e.g. `--end-time 0.8`), then to `--cycles 3`.
These horizons are supplied for continuation; they are not yet validated.

Implementation: `mac/semiimplicit.py`, `ho_tangent.py`,
`real_lv_diagnostics.py`, and the shared `mac/cnab.py`/transfer backends.

## Compare nonlinear solver cost on the same existing checkpoint

The earlier Newton implementation performs a full Stokes response and two
mass solves **per GMRES direction**, as well as complete residual evaluations.
Reducing the outer basis to solid nodes did not remove this nested work.
The new acceleration aims to avoid that cost in easy steps. Stiff steps may
still require Newton, so a fixed real-mesh GPU speedup is not guaranteed.
First-use kernel compilation belongs to setup/warmup time, not steady time.

Old checkpoints intentionally restore their saved solver, with missing solver
settings decoded as `newton`. To compare without changing the saved state:

```bash
CUDA_VISIBLE_DEVICES=0 python -u validation/benchmark_real_lv_schemes.py \
  --checkpoint results/real_lv_semiimplicit_02s/checkpoint.npz \
  --schemes cnab-semiimplicit --nonlinear-solvers newton anderson-newton \
  --device cuda --warmup 2 --steps 5 \
  --output results/real_lv_solver_performance/report.json
```

Both cases solve the same midpoint equations from the same accepted state.
The report separates setup time and warmed time and counts Stokes, pressure,
mass, Anderson, GMRES, tangent assemblies and fallback steps. GPU comparisons
should run without another simulation competing on the same GPU.

To switch an existing checkpoint to the accelerated solver while preserving
its AB2 time history, use a new branch directory:

```bash
CUDA_VISIBLE_DEVICES=0 python -u demo/real_lv_fsi/run_mac.py \
  --device cuda --resume results/real_lv_semiimplicit_02s/checkpoint.npz \
  --nonlinear-solver anderson-newton --end-time 0.2 \
  --output results/real_lv_semiimplicit_accelerated_02s
```

Solver-only changes preserve multistep history because the equations/dt are
unchanged. The source directory and checkpoint remain intact. Fresh demo
runs default to acceleration, or it may be requested explicitly. Local tests
cover the unchanged active H–O equations with fewer fluid/mass solves, growth
and invalid-probe fallback, stiff modal energy bounds, independent final
acceptance, restart and actual benchmark counts. A small-case success is not
a full real-mesh performance or three-cycle validation.
