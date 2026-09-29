# Optional fused pressure multigrid

`--pressure-backend fused` uses Triton kernels on CUDA PyTorch tensors for
the seven-point Neumann stencil, weighted Jacobi update, restriction and
trilinear prolongation plus correction. Each solver preallocates and reuses
level buffers. Returned pressure is copied out so later solves cannot
overwrite previous states. One solver cannot run concurrent solves.

The default remains `torch`. Both retain the same V-cycle, boundary conditions,
mean-zero gauge, coarse solve, precision, residual tolerances and check frequency.
Convergence is checked with the original PyTorch operator. CUDA Graph capture
and relaxed tolerances are not part of this change.

On CPU, `fused` runs a buffered tensor reference (`buffered-cpu`) to test
workspace behavior. This does not test Triton compilation or performance.
Local validation: 20 CPU tests passed, 20 CUDA tests skipped. GPU validation
and measured speedup remain pending.

## Linux GPU validation

From the repository root in the `afsi-torch` environment:

```bash
git pull --ff-only
python -m pip install -e ".[test,geometry,fused]"
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_mac.py tests/test_mac_multigrid_fused.py
CUDA_VISIBLE_DEVICES=0 python -u validation/benchmark_mac_pressure.py \
  --checkpoint results/lv_mac_smoke/checkpoint.npz \
  --device cuda --warmup 3 --steps 20 \
  --output results/mac_pressure_fused/report.json
```

Use an existing checkpoint; the example path is the previous smoke test.
For a held-load comparison, use the actual completed 2 s checkpoint path.
A 2 s checkpoint samples only the held-load tail. Input checkpoints are never
modified. Choose a fresh output report path for repeat runs.

The comparison fixes the separable IB method and mass-solve warm starts,
changing only pressure backend. Setup/first-use compilation is excluded from
throughput and recorded separately. Final coordinates, forces, pressure and
velocity are compared. Failed comparisons are saved and return a nonzero exit
status; solver failures raise directly. Phase timings use a separate replay
with CUDA events, without added synchronization between phases. Original
convergence checks still synchronize. `pressure_solve` is inside `fluid_total`;
do not sum both. Inspect `passed`, `equivalence`, `pressure_speedup`,
`whole_step_speedup`, cycle counts and divergence. Short-replay speedup is not
a full-run estimate. First-use compilation can take longer than the replay.

## Demo selection

After CUDA tests and short replay pass, run a fresh short loading segment:

```bash
CUDA_VISIBLE_DEVICES=0 python -u demo/ideal_lv_fsi/run_mac.py \
  --device cuda --warm-start --pressure-backend fused --end-time 0.005 \
  --output results/lv_mac_fused_early
```

For 2 s use `--end-time 2.0` and a fresh output directory. Backend choice is
saved and restored on resume. Explicit `--pressure-backend torch` switches
back to the reference implementation. Demo runs do not enable phase profiling.
