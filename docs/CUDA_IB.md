# Optional PyTorch C++/CUDA IB contraction

`--ib-csr-contraction-backend cuda` selects a custom CUDA implementation of
adaptive P1 quadrature contraction and hash insertion. It is registered through
PyTorch's dispatcher (`torch.ops.afsi_ib_cuda`) and consumes existing CUDA
tensors directly. This backend is experimental until measured on the target GPU;
`sites` remains the default. Compilation time is excluded from warmed benchmarks.

## Discrete operator

For each MAC velocity component, the extension computes exactly the existing
quadrature sum `B[a,i] = sum_q W[q] N[a,q] delta_h(i,X[q])`. One warp integrates
one tetrahedron/lattice pair and reuses each quadrature kernel value for all four
P1 nodes. Four lanes then accumulate the four nodal contributions. The kernel
does not allocate a global point-by-lattice-by-node contribution array.

The component and shared stencil layouts are supported, including nonzero group
offsets, arbitrary positive quadrature lengths, and 64-bit encoded sparse keys.
FP32 and FP64 are supported. Floating-point fusion and fast math are disabled;
parallel reduction/atomic addition orders can still differ from Triton.

Initial construction uses atomic compare-and-swap in a bounded hash table.
Cached updates read immutable keys and send missing contributions to the existing
bounded stream. A separate CUDA launch merges misses only after all lookup
readers have finished. Existing overflow, budget and snapshot rules apply.

CSR indices, numeric snapshots, the paired transpose, and consistent mass solves
still use the existing implementation. In particular:

- spreading is `B.T @ solve(M, force) / cell_volume`;
- interpolation is `solve(M, B @ fluid_velocity)`;
- the same numeric `B` defines both directions and their discrete power identity.

The BE-BE time scheme, loads, constitutive model, tolerances, nonlinear solver,
quadrature selection and point density do not change. AFSI's CPU C++ implementation
informs the kernel semantics, but is not copied as a host-vector wrapper: the
current MAC face locations and finite-element quadrature/mass operators are used.

## Build on Linux

Install the CUDA toolkit providing `nvcc`, a compatible C++ compiler, and Ninja.
The CUDA runtime packaged with a PyTorch wheel alone does not provide `nvcc`.
Use a toolkit compatible with `torch.version.cuda`; the CUDA version displayed
by `nvidia-smi` describes driver capability.

For a PyTorch `+cu130` build, install the CUDA 13.0 development toolkit into the
active environment rather than using a system CUDA 12 compiler. The existing
Ubuntu 24.04 GCC 13 compiler is supported by CUDA 13.0. NVIDIA documents Conda
toolkit installation in its [Linux installation guide](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-installation-guide-linux/index.html#conda-installation).

```bash
conda activate afsi-torch
conda install -c nvidia "cuda-toolkit=13.0" -y
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
hash -r
"$CUDA_HOME/bin/nvcc" --version
```

The build now checks the selected compiler's actual version and rejects a CUDA
major mismatch before invoking Ninja. This makes a stale `CUDA_HOME` or system
compiler visible immediately. Minor-version compatibility still depends on the
toolkit/PyTorch combination; matching the wheel's toolkit version is preferred.

```bash
conda activate afsi-torch
python -m pip install -e ".[test,cuda-ib]"
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvcc --version
g++ --version

# Set CUDA_HOME only if the installed toolkit is not automatically detected.
# export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="8.9"  # RTX 4090; change for other GPU architectures
export MAX_JOBS=2                 # compiler processes, not solver parallelism
CUDA_VISIBLE_DEVICES=0 python -m afsi_torch.mac.cuda_ib
```

The first build is cached by `torch.utils.cpp_extension`. Sources are shipped in
both editable installs and wheels. Build before CUDA Graph capture. Kernels use
PyTorch's current stream and device guard without downloading quadrature data.
An unavailable toolkit, incompatible device, or compilation error is reported;
the requested CUDA backend never silently runs the CPU oracle.

The construction operators mutate explicit scratch buffers and have no autograd
formula. They are used with frozen old geometry in the current BE problem.
Differentiating through the construction itself requires a separate derivative
implementation. This does not change the existing CSR response/tangent path.

## Correctness and performance

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/test_cuda_ib.py tests/test_csr_cached_assembly.py tests/test_csr_hash_assembly.py
```

GPU tests execute the actual extension and compare with an independent quadrature
oracle. They cover layouts, FP32/FP64, large keys, collision/overflow handling,
changed support/rules, old snapshot ownership, force/power consistency, nondefault
streams, CUDA Graph replay, and invalid input metadata. GPU tests are explicitly
skipped on CPU-only machines; a CPU test pass is not CUDA verification.

Use the same checkpoint for both variants. Give this process exclusive use of
the selected GPU. Select a checkpoint in the phase you want to measure: a finished
1.6 s checkpoint measures the beginning of the next cycle, not peak contraction.

```bash
checkpoint="/path/to/the/current/simulation/checkpoint.npz"
CUDA_VISIBLE_DEVICES=0 python -u validation/benchmark_paper_lv.py \
  --checkpoint "$checkpoint" \
  --device cuda --warmup 5 --steps 20 \
  --solvers anderson-newton --linear-check-intervals 5 \
  --ib-response-backends csr --csr-assembly-backends cached-hash \
  --csr-contraction-backends sites cuda \
  --profile --profile-steps 3 \
  --output results/paper_lv_cuda_ib/report.json
```

All other options are loaded from the checkpoint, preserving the successful
simulation's material, load and solver settings. Inspect end-state differences,
accepted residuals, total milliseconds per step, CSR assembly phases and peak
memory. A faster assembly kernel may have a small effect on total time if repeated
mass/fluid solves dominate. No total speedup is assumed from changing languages.

After the GPU checks and comparison, an existing simulation can select the new
backend without replacing its checkpoint configuration:

```bash
CUDA_VISIBLE_DEVICES=0 python -u demo/real_lv_fsi/run_mac.py \
  --resume "$checkpoint" --device cuda --end-time 1.6 \
  --ib-response-backend csr --ib-csr-assembly-backend cached-hash \
  --ib-csr-contraction-backend cuda
```

Resume must start before the requested end time. Use the original configuration
for a new run; add the three IB backend options above. Retain `sites` if the CUDA
variant is slower or does not pass the equivalence checks.
