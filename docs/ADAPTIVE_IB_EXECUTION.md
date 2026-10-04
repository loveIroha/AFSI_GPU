# Adaptive P1 FE/IB execution

The warmed real-LV comparison measured 1375.6 ms/step for conical quadrature,
1183.9 ms/step with final-response reuse, and 813.7 ms/step with compact
Xiao–Gimbutas rules and reuse on an RTX 4090. All sampled early filling from
one checkpoint; this does not measure active contraction or full-cycle stability.
The remaining work includes several million interaction points per preparation
and nonlinear response, plus nested mass and Stokes solves.

This experiment changes **execution only**, relative to the compact case.
It retains the point locations, weights, selected degrees, Peskin kernel,
reference consistent CSR mass, PCG acceptance, H–O constitutive model,
loads, dt and nonlinear midpoint equations. It does not lower quadrature
density, lump mass, alter precision or relax convergence checks.

## Paths

`interaction_quadrature.transfer_backend` selects:

| Value | Spread | Interpolation |
| --- | --- | --- |
| `reference` | Evaluate and weight a full point-force array, then spread | Gather a full point-velocity array, then assemble by quadrature groups |
| `fused` | Evaluate P1 force coefficients and weights inside each spread kernel | Gather fluid velocity, quadrature-integrate and assemble FE nodal RHS inside the kernel |
| `cell` | First reduce all same-cell contributions to each fluid node, then atomically add once | Same fused interpolation |

These paths all solve `M F=b` and `M U=B^T W K u`, and spread
`f=K^T W B F / cell_volume`. The paired transfer remains adjoint.
Floating-point summation order can change, so compare relative/absolute
errors and power balance rather than requiring bitwise equality.

The fused CUDA paths avoid full point-force and point-velocity intermediate
arrays. At 4,438,962 points, each FP64 three-component array occupies about
106.5 MB; eliminating it also removes write/read traffic on every invocation.
The separable stencil remains unchanged and is still a large allocation.

For a cell with selected order n, hmax <= n*dx_min/point_density.
The union of its four-point supports fits an axis-aligned cube of side
`4+ceil(n/point_density)`. Cell spread uses this bound without dropping links.
For the dominant n=4, 31-point rule, the old path performs 31*64=1984 grid
atomic adds per cell/component. The reduced path requires at most 6^3=216.
The actual mask often excludes some of these nodes. The reduction adds local
arithmetic; fewer atomics are **not** a promised whole-step speedup.

Dense groups are tiled during interpolation to bound register/shared-memory
use. Above 256 points per cell, interpolation retains the existing gather/
assembly path. Cell spread falls back to fused point spread above 256 points
or a support extent above eight. All fallbacks retain the full rule.
The CPU implementations are validation paths, not performance presets.

`reference` remains the default until target-GPU measurements support a
change. Old checkpoints missing the field also decode to `reference`.
Fresh runs can select `--ib-transfer-backend fused` or `cell`; normal restart
restores the saved value. No settings or fields are silently converted.

## Short, read-only GPU comparison

Run from the repository root in a clean afsi-torch environment:

```bash
git pull --ff-only
conda activate afsi-torch
python -m pip install -e ".[test,mesh,fused,quadrature]"
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_adaptive_cell_execution.py

CUDA_VISIBLE_DEVICES=0 python -u validation/benchmark_real_lv_schemes.py \
  --checkpoint results/real_lv_adaptive_early/checkpoint.npz \
  --schemes cnab-semiimplicit --nonlinear-solvers anderson-newton \
  --execution-variants compact fused cell \
  --device cuda --warmup 5 --steps 20 --profile \
  --output results/real_lv_cell_performance/report.json
```

All variants use compact quadrature and final-response reuse, begin at the
same saved physical state/AB2 history, and differ only in transfer execution.
`execution_comparisons` compares fused/cell to compact; the checkpoint remains
read only. Reports include field maxima, relative L2 errors (displacement,
velocity, pressure, force), and absolute endpoint differences. Performance
ratios use uninstrumented warmed wall time, excluding initial compilation.

`--profile` performs an additional subsequent replay with CUDA-event phase
timings. Its interval and actual nested-solve counts are recorded separately;
it is excluded from speedup measurements and does not save a continuation.
Mass timing is inside IB timings; pressure timing is inside Stokes. Do not
sum nested phases. A profile failure is reported separately from completed
performance measurements. Do not run a competing job on the same GPU.

Inspect the fastest passing candidate's total time and the remaining phase
costs before enabling it in a long run. No full-cycle acceleration or
stability claim follows from this startup benchmark.

## Validation and implementation

- `mac/adaptive_cell.py`: CPU cell reduction and device dispatch.
- `mac/_triton_adaptive.py`: fused spread, cell-reduced spread, tiled
  interpolation with direct FE RHS assembly.
- `mac/adaptive_transfer.py`: saved configuration, unchanged mass solves
  and paired frozen quadrature ownership.
- `tests/test_adaptive_cell_execution.py`: same-rule field comparison,
  force/torque/power, active coupled CSR tangent, accepted steps, read-only
  benchmark, phase separation, checkpoint restart and optional interpreter
  checks of the actual Triton kernels.

Local verification includes CPU numerical tests, Triton CPU interpreter
execution, and offline sm89 CUDA compilation. These do not establish target
GPU runtime correctness or performance; CUDA-parametrized tests and the
short checkpoint benchmark must be run on the target GPU. No Windows Triton
dependency is added to the project; Linux installs continue to use the
existing `fused` extra.
