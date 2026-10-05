# CN velocity workspace and graph execution

The CN wall Stokes solver uses the same fixed Jacobi polynomial for
`H = I - dt*mu/(2*rho)*L`. Its conservative contraction bound selects the
same iteration count as the tensor reference. Odd tangential ghosts and zero
normal wall values are retained, including on anisotropic boxes. The actual
wall Schur equation, pressure multigrid and final momentum/divergence acceptance
remain unchanged. This is an execution backend, not a change to CN--AB2,
midpoint FE/IB, H-O, quadrature, or nonlinear tolerances.

`coupling.cnab.helmholtz_backend` accepts `auto`, `torch`, `triton`, or `graph`.
`auto` selects the existing compiled torch execution on CUDA and eager torch
on CPU. A native RTX 4090 comparison at t=0.5 s measured 1389 ms/step for torch
and 1437 ms/step for graph, with the same iteration counts and tiny numerical
differences. The graph backend therefore remains an explicit experimental option.
Old checkpoints without this key decode to `auto`, so their physical settings and
AB2 history are retained while the measured faster torch execution is restored. Fresh real-LV
cases can select `--helmholtz-backend graph`, or `torch` for comparison. A
restart restores its settings; use the read-only benchmark to compare backends.

## GPU data path

- Initialize the right-hand side and diagonal initial guess with one fused
  kernel per component. If pressure is present, subtract its normal gradient
  directly, without storing three gradient arrays or three extra RHS arrays.
- Each sweep loads the six neighboring velocities, applies the existing odd
  ghost stencil and Jacobi update, and writes into the alternate buffer. No
  concatenated ghost arrays or intermediate Laplacians are allocated.
- Capture the fixed sweep chain in a CUDA Graph. The RHS, diagonal and two
  scalar coefficients have stable addresses. A time-step change updates these
  buffers; a changed iteration count selects a different graph. At most four
  graph counts are cached, with least-recently-used eviction.
- Stokes may borrow the scratch output until the next Helmholtz call. The
  continuity norm is consumed before scratch is reused, and every accepted
  flow result is copied into caller-owned tensors. Public `helmholtz()` also
  returns owned tensors, preventing later solves from overwriting stored states.

Scratch consists of one RHS and two full MAC velocity fields, approximately
145 MiB for an FP64 128-cubed grid. This bounded allocation replaces per-sweep
temporary allocation; more occupied VRAM is not itself a performance goal.
Each solver uses its original CUDA stream, as do its other mutable workspaces.
The capture initializer protects graph cleanup from occurring inside capture.

True convergence checks still synchronize with the host. They have not been
removed or relaxed. The nonlinear iteration count and number of complete
Stokes calls are also unchanged by this backend.

## Verification and measurement

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/test_mac_helmholtz.py tests/test_mac_cnab.py \
  tests/test_mac_semiimplicit.py tests/test_mac_workspace_substeps.py

CUDA_VISIBLE_DEVICES=0 python -u validation/benchmark_real_lv_schemes.py \
  --checkpoint "$checkpoint" \
  --schemes cnab-semiimplicit --nonlinear-solvers anderson-newton \
  --helmholtz-backends torch graph \
  --device cuda --warmup 5 --steps 20 --profile \
  --output results/real_lv_cn_stokes_performance/report.json
```

Set `checkpoint` to an actual saved real-LV checkpoint from the current load
phase. Run the benchmark while the GPU is idle; simultaneous full simulations
invalidate a controlled timing comparison. The checkpoint is read-only.
The report includes warmed wall times, Helmholtz solve/sweep counts, actual
backend and workspace statistics, and final-state differences. Profiling runs
separately after the timed replay; `velocity_helmholtz` and `pressure_solves`
are nested inside `stokes`, and must not be summed with their parent.

CPU tests cover the fixed polynomial, anisotropic walls, result ownership,
time-step changes and profiling registration. An optional
`TRITON_INTERPRET=1` test executes the actual initialization/sweep kernels on
CPU. Native CUDA tests cover graph capture/replay, bounded graph reuse and
active-load coupled continuation. CPU or interpreter passes do not establish
native CUDA correctness, speedup, or complete three-cycle stability.
