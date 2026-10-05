# P1 validation execution

The fused execution path uses parallel support checks and pointwise affine
geometry. These changes preserve the CN/AB2 midpoint equations, material and
boundary forces, consistent CSR mass, interaction quadrature, solver
tolerances and rejection checks. They do not establish full-cycle stability.

## Support checks

The original compiled predicate can select one GPU block to scan the entire
point array. The new CUDA implementation assigns 1024 coordinate entries per
block, reduces their boolean results, then hierarchically reduces the partial
flags. It keeps one final host acceptance check; it does not synchronize every
block. Finite coordinates, inclusive lower bounds and strict upper bounds
retain their original meaning. FP32 uses rounded division and multiply/add
fusion is disabled for the bound calculation. CPU execution retains the
PyTorch predicate.

Adaptive affine P1 transfer checks each used mesh vertex once instead of
gathering four corners for every tetrahedron. Both arrays contain exactly the
same vertex set. The existing convex-hull support argument still applies;
interaction points, weights and Peskin links are unchanged. No invalid cell,
out-of-domain point or incomplete stencil is accepted by removing this
duplication.

## Geometry checks

P1 deformation gradients use a fixed sum of four coordinate/shape-gradient
outer products, allowing PyTorch compilation to fuse the operation instead
of invoking a batched small matrix product. Triangle area vectors use the
two affine edge tangents. Their constant values are expanded over the
original surface quadrature axis; surface integration is unchanged.

The shared validity predicates still reject nonfinite or nonpositive
determinants, collapsed boundary surfaces and invalid cavity volume. The
generic P1 solid keeps user-defined validity callbacks. Geometry caching
continues to track tensor identity and version and is invalidated when the
execution backend changes. Arithmetic ordering can cause roundoff differences
in forces, so comparison uses field errors and residual acceptance, not
bitwise equality.

The geometry helpers live in `src/afsi_torch/solids/p1_geometry.py` and are used
by both `P1Execution` and the supplied H-O execution adapter. The constitutive
callback and CSR tangent assembly are unchanged. Existing checkpoints select
the new path when their fused driver is constructed; saved physical state and
AB2 history are retained.

## Compare on the same checkpoint

Run from the Linux repository on an otherwise idle GPU:

```bash
conda activate afsi-torch
git pull --ff-only
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_validation_execution.py

CUDA_VISIBLE_DEVICES=0 python -u validation/benchmark_real_lv_schemes.py \
  --checkpoint results/real_lv_workspace_3cycles_20261005_210220/checkpoint.npz \
  --schemes cnab-semiimplicit \
  --nonlinear-solvers anderson-newton \
  --validation-backends reference blocked \
  --device cuda --warmup 5 --steps 20 --profile \
  --output results/real_lv_validation_performance/report.json
```

Replace the checkpoint path with an existing saved real-LV state if necessary.
Both modes use reference shared IB kernels and compiled torch CN velocity,
restore the same state and retain all physical settings. This comparison
requires adaptive quadrature and fused execution. The input checkpoint is
read only and no simulation checkpoint is written.

`validation_comparisons` reports warmed wall-time speedup and final-state
absolute/relative errors. `validation_execution` records the selected checks
and number of checked vertices. `--profile` performs a separate subsequent
replay and reports `ib_support_checks` and `solid_validation`; it is excluded
from the speedup measurement. These phases overlap their parent phases and
must not be added to them. Actual benefit must be measured on the target GPU.

The tests cover strict bounds, NaN/Inf, noncontiguous input, padded reduction
blocks, invalid geometries, force equivalence, active coupled steps, the CSR
tangent, cache invalidation and read-only benchmark input. Kernel tests run
natively on CUDA or with `TRITON_INTERPRET=1`; CPU geometry tests do not prove
GPU throughput.
