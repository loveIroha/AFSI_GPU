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

A right diagonal preconditioner approximates local mobility using row-sum
mass and the magnitude of the tangent diagonal. **This does not lump the mass
in the discretization**: both `J_hat` and `S_hat`, every nonlinear residual,
and every Jacobian action still solve the assembled consistent FE mass system.
The preconditioner is an approximation; its benefit must be measured.

## Acceptance, history and stability

Newton/GMRES corrects the midpoint force feedback, with residual-norm
backtracking and geometry/support checks. A tangent is assembled per Newton
iteration. A small initial residual can accept without an unnecessary assembly.
There is no fixed single correction advertised as a converged nonlinear solve.

The final residual is recomputed independently. The actual CN Stokes momentum
and divergence must also pass their existing tolerances. The solved midpoint
is retained; an extra unguarded Picard position update is not applied. The
endpoint kinematic residual is twice the checked midpoint residual.
The `nonlinear` report section records acceptance, unknown count, all Newton
history, tangent assemblies and Jacobian actions.

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
  tests/test_mac_semiimplicit.py tests/test_mac_cnab.py tests/test_mac_implicit.py
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
