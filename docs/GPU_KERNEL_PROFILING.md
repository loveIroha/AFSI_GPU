# GPU execution evidence

Use `validation/profile_real_lv_gpu.py` after a phase comparison identifies a
slow stage. This tool does not change production defaults, equations,
quadrature density, linear/nonlinear tolerances, or the source checkpoint.
It restores the saved AB2 history, previous dt and pressure time. It selects
the measured reference IB execution and compiled torch CN velocity unless an
IB alternative is explicitly requested.

The latest RTX 4090 comparison at t=1.0 s measured 1338.8 ms/step for reference,
1361.3 for vector and 1458.1 for reduced. Both alternatives remain opt-in.
The small vector difference needs repeatability evidence; reduced propagation
was substantially slower. Fewer atomics alone do not establish a speedup.

## Short coupled timeline

Run on an otherwise idle GPU, using an existing checkpoint. These commands
are Linux host commands in the `afsi-torch` environment, without Docker.
The timestamp prevents replacing previous profiling results.

```bash
cd /mnt/large2/gjh/AFSI_GPU
conda activate afsi-torch
git pull --ff-only
checkpoint=results/real_lv_workspace_3cycles_20261005_210220/checkpoint.npz
out=results/real_lv_gpu_profile_$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=0 python -u validation/profile_real_lv_gpu.py \
  --checkpoint "$checkpoint" --device cuda --scope coupled \
  --warmup 5 --steps 2 --batches 3 --output "$out/coupled"
```

This advances five warmup steps, three uninstrumented two-step batches, then
captures two subsequent steps. No simulation checkpoint is written. The
report separates uninstrumented batch timing from the instrumented replay.
Coupled batches advance time, so sample variation can reflect changing loads.
Normal demo runs have no new instrumentation or profiling overhead.

Artifacts:

- `report.json`: timing samples, range call counts, recorded GPU activity union,
  top kernels and host synchronization calls, quadrature and memory metadata.
- `trace.json`: Chrome trace with nested `afsi.*` ranges for IB, mass, Stokes,
  pressure, force and validation; view in Perfetto or a Chrome trace viewer.
- `operators.txt`: aggregated PyTorch operator timings. Parent/child GPU times
  and nested phases must not be added together.

The tool intentionally disables stack, shape and allocation recording to
limit capture overhead. One final synchronize closes the capture window; no
per-range synchronization or CUDA timing events are added. It computes a
union of recorded GPU intervals rather than adding overlapping streams.
Time without recorded activity is **not** automatically CPU overhead or GPU
idle time: dependencies, incomplete CUDA graph events and profiler coverage
must be examined. Missing CUDA activity is reported explicitly, not treated
as zero-cost GPU execution. Hardware bandwidth, occupancy, register spills
and atomic throughput cannot be obtained from this trace summary alone.

## Isolate the unchanged FE/IB kernels

After the coupled timeline, compare the exact same frozen checkpoint geometry,
velocity, coefficient and quadrature across execution alternatives:

```bash
for mode in reference vector reduced; do
  CUDA_VISIBLE_DEVICES=0 python -u validation/profile_real_lv_gpu.py \
    --checkpoint "$checkpoint" --device cuda --scope ib \
    --ib-shared-execution "$mode" --warmup 5 --repeats 10 --batches 3 \
    --steps 1 --output "$out/ib_$mode"
done
```

The consistent mass coefficient solve and stencil construction happen once
before timing. Pure `gather` assembles the FE velocity RHS; pure `spread` maps
the force coefficient to MAC force density. Neither timed operation includes
a mass solve, nonlinear iteration or prepare. Output allocation/zeroing stays
inside each call because it is part of the existing wrapper's actual cost.
Use `--operation gather` or `spread` to capture only one direction. All original
quadrature points, weights and 64 Peskin neighbors remain unchanged.

The report compares raw outputs with reference and supplies three timing
batches, rather than inferring performance from the number of atomics.
CUDA event pairs surround entire batches, not each launch. Event span can
include host enqueue gaps; use recorded kernel activity to distinguish it
from active GPU compute. The fixed stencil uses checkpoint x, so it isolates
execution but is not a replacement for coupled midpoint tests.

## Optional hardware profiling

The preceding commands need PyTorch only (plus existing case dependencies).
If NVIDIA Nsight tools are installed, `--capture external` supplies NVTX
ranges and CUDA profiler start/stop after warmup. It does **not** enable the
PyTorch/Kineto collector at the same time.

Nsight Systems reveals launch gaps, CUDA graph execution and synchronization:

```bash
mkdir -p "$out/nsight"
CUDA_VISIBLE_DEVICES=0 nsys profile \
  --trace=cuda,nvtx,osrt --sample=none --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o "$out/nsight/coupled" \
  python -u validation/profile_real_lv_gpu.py \
  --checkpoint "$checkpoint" --device cuda --scope coupled \
  --capture external --warmup 5 --steps 2 --batches 1 \
  --output "$out/nsight/coupled_metadata"
```

Nsight Compute collects hardware counters for a bounded number of **IB**
launches, without profiling pressure graphs or mass solves:

```bash
for mode in reference vector reduced; do
  CUDA_VISIBLE_DEVICES=0 ncu --profile-from-start off \
    --kernel-name-base function \
    --kernel-name 'regex:_gather_shared|_gather_cell|_spread_shared|_spread_reduced' \
    --launch-count 12 --set full -o "$out/nsight/ib_$mode" \
    python -u validation/profile_real_lv_gpu.py \
    --checkpoint "$checkpoint" --device cuda --scope ib --capture external \
    --ib-shared-execution "$mode" --warmup 2 --repeats 1 --batches 1 \
    --steps 1 --output "$out/nsight/ib_${mode}_metadata"
done
```

Counter replay can be expensive. Its kernel durations/cache state are not
comparable to the uninstrumented throughput measurements. Inspect each launch
configuration and quadrature group; a small first group need not represent
the dominant 31/57-point groups. If fewer than twelve matching launches exist,
only those launches are captured. Nsight requires compatible local tools and
hardware-counter access; an `ERR_NVGPUCTRPERM` result means no hardware-counter
evidence was collected. Do not change machine permissions automatically.

Inspect register count/spills and achieved occupancy for vector gather;
sorting/scanning instructions, cache behavior and atomic traffic for reduced
spread. Only select an optimization after both numerical equivalence and
repeatable uninstrumented timing demonstrate a benefit.

References: [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler),
[Nsight Systems capture ranges](https://docs.nvidia.com/nsight-systems/UserGuide/index.html),
[Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html).
