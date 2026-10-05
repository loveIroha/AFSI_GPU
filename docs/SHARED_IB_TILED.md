# Shared P1 IB execution alternatives

`interaction_quadrature.shared_execution` selects `reference`, `vector`, or
`reduced`. These choices require adaptive/shared/fused transfer, and operate
on the same frozen shared stencil and paired quadrature groups. They are
generic P1 tetrahedron-to-3D-MAC kernels, without left-ventricle-specific data.
The default is `reference`; new kernels remain opt-in until native GPU timings
and field comparisons demonstrate a benefit. Older checkpoints omit this key
and retain the existing implementation.

## Vector gather

The existing kernel launches separate component/quadrature tiles per cell.
`vector` assigns one CTA to each cell and processes bounded 16-point tiles in
a loop. It loads the cell's four node IDs once, reads shared shape/weight data
once per tile, and reuses six lattice base/weight tables for all three velocity
components. All 64 neighbors per component are included. Twelve cell-local
nodal RHS entries are accumulated in registers, then added once to the global
FE RHS. The same consistent CSR mass solver converts this RHS to nodal velocity.

Global nodal atomics decrease from
`12*sum(cell_count*ceil(points_per_cell/tile))` to `12*total_cells`.
No point-velocity array or cell-sized global intermediate is introduced.
Large positive conical fallback rules use the same bounded tile loop rather
than a quadrature-count-sized block. Fewer CTAs and atomics may help; longer
CTA lifetimes and register pressure still require actual GPU measurement.

## Local spread reduction

`reduced` includes vector gather and additionally assigns spread CTAs to four
quadrature points from the same cell. Their unchanged 64-link contributions
are computed once, sorted by destination grid ID with the original lane ID
as a tie-breaker, and combined with a segmented prefix sum. Only the last
entry of each equal-ID segment performs a global FP64 atomic add.

Padded points sort beyond the grid and never write. Small grids use safe
32-bit packed keys; larger grids use 64-bit keys. Group offsets are runtime
values, so changing cell memberships does not specialize new kernels. Sorting
and scanning incur work and can outweigh atomic savings; the separate vector
case isolates this tradeoff. Reduction across cells and a full-domain sort
are not required, and scratch remains block-local.

All quadrature points, reference weights and Peskin links are retained. Both
directions use the same weights, so the discrete adjoint power relation is
preserved to floating-point and linear-solve tolerances. The accumulation order
changes, as with any parallel reduction; bitwise equivalence is not expected.
The consistent mass matrix, H-O stress, CSR tangent, time scheme, force loading
and acceptance criteria are unchanged.

## Run a controlled comparison

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/test_shared_ib_tiled.py tests/test_shared_fe_fusion.py

CUDA_VISIBLE_DEVICES=0 python -u validation/benchmark_real_lv_schemes.py \
  --checkpoint "$checkpoint" \
  --schemes cnab-semiimplicit --nonlinear-solvers anderson-newton \
  --ib-shared-executions reference vector reduced \
  --device cuda --warmup 5 --steps 20 --profile \
  --output results/real_lv_shared_ib_performance/report.json
```

Use a saved checkpoint from the current load phase while the GPU is idle.
The benchmark reads the source only and uses compiled torch CN velocity in
every case, including when a source checkpoint explicitly selected graph.
All other execution choices and physical settings are inherited. The report
contains warmed full-step times, separate phase timings, solve counts, point
counts, final-state differences and relative norms. Mass timing is nested in
IB timing, and pressure/Helmholtz timing is nested in Stokes timing.

To select a measured winner in a fresh demo, keep all existing adaptive/shared/
fused settings and add `--ib-shared-execution vector` or `reduced`. Configuration
and checkpoints persist the selection; normal resume restores saved settings.
An execution-only branch may override these two kernels into a new output
directory while retaining x/u/p, force, AB2 advection history, previous dt and
pressure timestamp:

```bash
python -u demo/real_lv_fsi/run_mac.py --device cuda \
  --resume "$checkpoint" --helmholtz-backend torch \
  --ib-shared-execution vector --cycles 3 --output results/real_lv_vector_continuation
```

Replace `vector` with the measured winner. An override requires a fresh output
directory and preserves the source checkpoint. The example is conditional on
the preceding GPU correctness and performance comparison, not a claim that
the new implementation is already faster.
No new path is automatically made the default on the basis of CPU tests.

Tests cover dense/padded rules, nonzero group offsets, affine velocity, force,
torque, adjoint work, old-stencil lifetime, unchanged mass entries, active
coupled continuation, CSR tangent action, configuration, restart and read-only
comparison. Actual kernels also run in optional `TRITON_INTERPRET=1` mode.
Native CUDA execution, performance and full-cycle stability require GPU tests.
