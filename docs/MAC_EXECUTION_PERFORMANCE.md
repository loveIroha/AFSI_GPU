# MAC/IB/FEM execution backend

`--execution-backend fused` optimizes execution of the current quadrature
IB/consistent-mass/P2 solid model. `--pressure-backend fused` independently
selects the already implemented pressure V-cycle kernels. Both default to
`torch` until explicitly selected, and both choices persist in checkpoints.
Warm starts retain their existing independent setting.

## What changes

* Compact IB storage keeps only base coordinates and four one-dimensional
  Peskin weights on each staggered lattice. Triton gather/spread kernels form
  all 64 links in registers. For 243,502 points in FP64/int64 the stored stencil
  decreases from 748,038,144 to 87,660,720 bytes (8.53x smaller). No interaction
  points or quadrature weights are removed. Spread still uses GPU atomic sums.
* The assembled CSR consistent mass remains unchanged. The unconstrained IB
  Jacobi-PCG uses reusable x/r/d/Ad buffers and compiled vector operations.
  It keeps the same tolerances, restart/check intervals, warm starts, breakdown
  detection and true-residual acceptance. Returned coefficients are copies,
  never aliases of solver scratch storage. Static diagonal checks run at setup.
* Fluid predictor, pressure correction and residual evaluation use compiled
  tensor expressions. Finite-value, wall and stability checks are grouped
  into fewer host transfers. Centered advection, explicit viscosity and the
  pressure operator/boundary conditions retain the reference formulas.
* The LV execution wrapper reuses validated F and endocardial area vectors for
  force evaluation. A version-checked single-state cache avoids validating the
  same unmodified state twice. Inference tensors are not cache hits. Guccione
  strain/stress, P2 assembly and surface forces are compiled. The identity
  J*F^(-T)=cofactor(F) replaces the batched 3x3 inverse with cross products;
  the constitutive model is unchanged. Time-dependent loads are tensor inputs,
  so a new time value does not cause recompilation.

Model geometry, fibers and parameters are fixed for each driver. Rebuild the
driver if they change. Driver/solver scratch storage is not safe for concurrent
calls. This execution backend is forward-only, like the existing explicit MAC
driver; use the unchanged reference solid kernels for torch.func derivatives.

No node coupling, mass lumping, altered integration rule, precision reduction,
new preconditioner, relaxed tolerance or different time-stepping scheme is used.
Floating-point reduction order can change; equivalence is tolerance-based.
General compiled kernels keep automatic CUDA Graphs disabled to preserve
ownership of returned states. Optional `--mass-backend graph` explicitly
captures only PCG blocks in fixed workspaces and clones results to the caller.
Optional `--solid-backend pointwise` fuses scalar 3x3 constitutive algebra.
Both require fused execution and persist across checkpoint resume; see the
[demo experiment commands](../demo/ideal_lv_fsi/README.md).

## Validation and performance measurement

Local CPU tests cover transfer force/torque/power conservation, constant
velocity, anisotropic shifted grids, shared P2 nodes, quadrature refinement,
mass true residuals/restarts/breakdown, state ownership, stability guards,
solid force equivalence, tensor mutation invalidation and checkpoint resume.
A full-graph tracing test checks the optimized path without time-dependent
recompilation. A five-step CPU checkpoint replay passes field comparisons.
These checks do not establish CUDA code generation, correctness or speed:
the local development machine has CPU-only PyTorch. CUDA tests must run on
the Linux GPU machine. CPU compact kernels are reference emulation, not a
performance backend.

```bash
conda activate afsi-torch
git pull --ff-only
python -m pip install -e ".[test,geometry,fused]"
CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/test_mac.py tests/test_mac_multigrid_fused.py tests/test_mac_execution.py

CUDA_VISIBLE_DEVICES=0 python -u validation/benchmark_mac_execution.py \
  --checkpoint results/lv_mac_fused_2s/checkpoint.npz \
  --device cuda --warmup 3 --steps 20 \
  --output results/mac_execution/report.json
```

Use an existing checkpoint and a fresh output report path. Both benchmark
variants use fused pressure multigrid and warm-started mass solves; this
isolates the new execution changes. Warmup includes compilation; it can take
several minutes at first use and is excluded from throughput. Final diagnostics
and checkpoint I/O are also excluded. Nothing modifies the input checkpoint.
The 2 s checkpoint measures the held-load tail only.

`ib_mass_solves` sums both mass solves; `ib_gather_grid` and `ib_spread_grid`
measure pure transfer; `fe_force_evaluation` and `fe_velocity_assembly` measure
FE work outside PCG. These detail phases are nested inside the existing
spread/interpolate totals, and pressure is nested inside fluid total. Do not
add nested rows. CUDA phase events are read after the replay without added
per-phase synchronization. Production demos do not use these timing wrappers.

Inspect `passed`, state differences, mass iterations/true residuals, divergence,
power errors, peak allocated memory and `whole_step_speedup`. Report a failure
or regression rather than loosening tolerances to pass. Single short replays
do not determine full-horizon speedup.

After CUDA tests and replay pass, exercise the loading stage:

```bash
CUDA_VISIBLE_DEVICES=0 python -u demo/ideal_lv_fsi/run_mac.py \
  --device cuda --warm-start --pressure-backend fused --execution-backend fused \
  --end-time 0.005 --output results/lv_mac_execution_early
```

For 2 s use `--end-time 2.0` and a fresh output directory. Explicit
`--execution-backend torch` restores reference IB/solid/fluid execution while
leaving the selected pressure backend independent.

## Remaining costs

The next isolated four-way experiment targets solid small-matrix algebra and
PCG CUDA Graph blocks. See [solid/mass experiment](MAC_SOLID_MASS_EXPERIMENT.md)
for GPU checks, checkpoint replay commands and interpretation of detail timing.

Sparse mass matrix multiplies, global convergence reductions, force-scatter
atomics and checks required to accept a step remain. Compilation does not
guarantee fusion across sparse/library calls. Profiling on the target GPU is
needed to determine whether these remaining costs warrant further execution
optimization. This backend does not claim all kernels have reached hardware
limits or any measured CUDA speedup before that validation.
