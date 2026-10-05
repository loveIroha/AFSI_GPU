# Owned IB workspaces and CFL-controlled coupled intervals

The real-LV driver can reuse large shared Peskin tables and subdivide a complete
CNAB/FE interval when convection exceeds the existing CFL screen. These controls
do not change the H-O law, loads, basal penalty, quadrature density, consistent
CSR mass, nonlinear tolerances, or pressure discretization.

## Memory ownership

`--reuse-ib-buffers` requires adaptive/shared/triton preparation. The workspace
retains at most two allocations, rounds large initial point capacities to 262,144-point
blocks (smaller blocks for small meshes), and grows a free allocation by at least 25% when necessary. It never
shrinks on every change in cell quadrature order. A weak stencil owner prevents
reuse while that stencil is alive. If external callers retain more stencils, extra
owned allocations are not cached. Different CUDA streams do not reuse one another's
slots. Keep the stencil object alive for as long as its base/phi views are needed.

The logical tables remain contiguous `(2,P,3)` and `(2,P,3,4)` arrays: capacity
is not a hidden lattice stride. Every selected quadrature point and reference
weight is retained. The semiimplicit driver releases its inherited initial cache
and the current-position table after predictor interpolation, before preparing
the nonlinear midpoint table. Later preparation cannot overwrite a live Jacobian's
frozen geometry.

At existing log/checkpoint cadence, allocator counters record actual and reserved
bytes, their peaks, inactive split bytes, allocation retries, and OOM counts.
They are host-side counter reads; no extra CUDA synchronization or per-step
`empty_cache()` is introduced. `interaction_quadrature.stencil_workspace` reports
capacity and reuse counters. These counters support diagnosis; the implementation
does not claim all previous OOM warnings were caused by fragmentation.

## Time stepping and acceptance

Use `--dt 5e-5 --adaptive-substeps --max-substep-levels 2`. The output/checkpoint
clock uses the macro interval, while the actual coupled steps may be `5e-5`,
`2.5e-5`, or `1.25e-5` s. A three-cycle run has 48,000 macro intervals. Internal
step count can be larger. Ordinary fixed-step execution remains available without
`--adaptive-substeps`; `--dt 1e-5` is also supported.

The controller estimates the required dyadic level from accepted face velocities,
using a target CFL of 0.20. Every internal step solves the same nonlinear midpoint
FE force and CN Stokes equations and checks candidate CFL against 0.25. A CFL
rejection rolls back the entire macro interval and retries at half the internal
step. It discards trial AB2 histories and clears transfer warm guesses. It does
not accept a failed trial, raise the CFL limit, freeze solid force over fluid-only
substeps, or swallow other nonlinear/Stokes/geometry failures. If the configured
minimum step still fails, the run stops and retains the macro input checkpoint.

AB2 extrapolation for a current interval h and previous accepted interval k is

```
N_half = (1 + h/(2*k))*N_current - h/(2*k)*N_previous.
```

This is the interval average of linear temporal extrapolation. Equal intervals
retain the original `1.5/-0.5` coefficients. Coarsening is limited to a factor two.
CN viscosity uses the actual internal step. Pressure/mass graphs are retained;
CN diagonals are updated in place. The time-control policy is an addition to the
previous fixed-step algorithm, not a claim that all numerical choices are unchanged.

Checkpoints store the previous accepted internal dt and the final substep pressure
time. Old checkpoints remain readable. A deliberate `--resume-dt` branch resets
AB2 history as before. New controls cannot be silently applied to an old checkpoint:
start a fresh run or explicitly configure a new branch workflow. VTK fluid FieldData
includes `pressure_time_s` when available; pressure belongs to the last substep
midpoint, while velocity/geometry belong to the displayed macro endpoint.

History contains internal dt/count, rejected macro attempts, maximum accepted
substep CFL, and maximum nonlinear residual/tolerance ratio. Solver counters sum
accepted substeps; total Stokes calls also include rejected work. The completion
checker validates sampled substep CFL and residual ratios. No completion checker
establishes time/mesh convergence or periodic steady state.

## Validation and a fresh three-cycle run

From the Linux host repository root:

```bash
git pull --ff-only
conda activate afsi-torch
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_mac_workspace_substeps.py

run_dir="$(pwd)/results/real_lv_workspace_3cycles_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_dir"
CUDA_VISIBLE_DEVICES=0 nohup /usr/bin/time \
  -f 'elapsed_seconds=%e exit_code=%x' -o "$run_dir/runtime.txt" \
  python -u demo/real_lv_fsi/run_mac.py \
  --mesh-dir /mnt/large2/gjh/realistic_left_ventricle \
  --device cuda --dt 5e-5 --fluid-cells 128 --cycles 3 \
  --kappa 5e6 --beta 5e6 \
  --coupling cnab-semiimplicit --nonlinear-solver anderson-newton \
  --interaction-quadrature adaptive --ib-rule-family xiao-gimbutas \
  --ib-point-density 2 --ib-transfer-backend fused \
  --ib-stencil-backend shared --ib-prepare-backend triton \
  --stokes-warm-start --reuse-validation --reuse-ib-buffers \
  --adaptive-substeps --max-substep-levels 2 --substep-courant-target .20 \
  --log-every 200 --checkpoint-every 2000 --output-every 400 \
  --output "$run_dir" > "$run_dir/run.log" 2>&1 < /dev/null &
echo "PID=$! output=$run_dir"
```

VTK output remains every 0.02 s and scheduled checkpoints every 0.1 s. Monitor
with `tail -f "$run_dir/run.log"`; exiting that monitor does not stop the solver.
After completion run `python validation/check_real_lv_cycles.py "$run_dir" --cycles 3`.
The local tests cover CPU/reference trajectories and interpreter kernels;
full-size GPU memory savings, runtime, and three-cycle stability require this run.
