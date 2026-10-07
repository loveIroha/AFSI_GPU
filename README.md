# AFSI_GPU

**GPU fluid–structure interaction with PyTorch, nonlinear finite elements, and immersed-boundary coupling.**

[Description](#description) · [Documentation](#documentation) · [Installation](#installation) · [Quick start](#quick-start) · [Configuration](#configuration) · [PyTorch implementation](#pytorch-implementation) · [GPU performance](#gpu-performance)

## Description

AFSI_GPU is a research implementation of immersed-boundary fluid–structure interaction (IB-FSI) built around PyTorch. It couples a deformable Lagrangian finite-element solid to an Eulerian fluid grid, with geometry generation, transient simulation, checkpoint restart, and ParaView output in a single workflow.

The project brings together three ideas: the cardiac and valve examples in [AFSI](https://github.com/loveIroha/afsi), the use of PyTorch tensors and sparse operators for GPU finite elements demonstrated by [torchcor](https://github.com/sagebei/torchcor), and a staggered-grid fluid solver suited to GPU stencil computation. The implementation provides:

- **Nonlinear finite-element solids:** P1/P2 tetrahedra in 3D and quadratic triangles in 2D; anisotropic constitutive models, boundary tractions, and spring constraints.
- **Two fluid discretizations:** MAC staggered finite differences with geometric multigrid, and a 3D Q2/Q1 finite-element Chorin solver with assembled CSR operators.
- **GPU IB coupling:** PyTorch C++/CUDA is the primary IB execution backend in all GPU demo presets: adaptive P1 CSR contraction, compact P2 MAC transfer, reflected 2D transfer and nodal FEM transfer. MAC FE coupling retains its consistent mass matrix and paired operators; tensor/Triton reference paths remain available.
- **Execution optimizations:** compiled tensor kernels, Triton stencil and transfer kernels, reusable workspaces, CUDA Graph replay, and warm-started iterative solves.
- **Reusable case configuration:** Python configuration objects, JSON files, command-line overrides, recorded effective parameters, and restartable simulations.
- **Replaceable 3D solid adapters:** material-independent force, assembled CSR tangent and validity interfaces; a composable P1 material/boundary implementation and a small runnable extension example.

### Included demos

| Demo | Solid | Fluid | Default grid | Simulated time |
| --- | --- | --- | --- | --- |
| Ideal left ventricle, MAC | P2 tetrahedra; Guccione material, active tension, endocardial pressure | 3D MAC + geometric multigrid | 64 × 64 × 64 | 2 s; 40,000 steps |
| Ideal left ventricle, FEM | Same generated LV model | Q2/Q1 FEM + Chorin projection | 32 × 32 × 32 elements | 2 s; 40,000 steps |
| Ideal two-leaflet valve | P2 triangles; FRH material and root springs | 2D MAC + geometric multigrid | 256 × 64 | 3 s; 48,000 steps |
| Imported real left ventricle (Ma et al. 2024) | P1 tetrahedra; paper H–O eq81/82, DG0 fiber/sheet | Open-box 3D MAC, BE–BE, semi-Lagrangian convection | 128 × 128 × 128 | 1.5 s; 15,000 steps |

The generated ideal left-ventricle demos follow the **loading-and-holding protocol** of AFSI `demo_337`: pressure and active tension rise linearly over 1.5 s and remain constant until 2 s. The valve demo follows the geometry, material, and periodic inlet of AFSI `demo_340`. These cases generate geometry with Gmsh and do not require external patient meshes or fiber files.

The [real-LV demo](demo/real_lv_fsi/README.md) now targets the **passive inflation benchmark** in section V.F of [Ma et al. (2024)](https://eprints.gla.ac.uk/333577/), DOI 10.1063/5.0225605. It imports a user-supplied P1 tetrahedral mesh and DG0 fiber/sheet directions, applies pressure rising to 8 mmHg over 0.8 s and then held, and uses the paper's raw-I1 H–O law with its printed stress correction. The fluid box is 13 cm cubed, with homogeneous Neumann velocity diffusion and zero Dirichlet pressure; density is 1 g/cm³ and viscosity is 4 cP. The active_cycle.json preset also provides the user-defined 0.8 s pressure/active-stress extension with separate material parameters; see the current real-LV run guide. Patient data are not distributed.

The coupled **BE–BE** residual evaluates solid force at the new configuration while freezing both dual IB operators at the old configuration. Backward-Euler diffusion and a MAC projection are combined with semi-Lagrangian convection. The default nonlinear solver is finite-difference JFNK with unpreconditioned BiCGSTAB and line search. Optional Anderson acceleration with assembled-CSR Newton fallback solves the same BE–BE residual. Compiled P1 force and geometry kernels, fused shared IB transfers, consistent CSR mass, Graph PCG, and a separate Dirichlet pressure multigrid with graph replay retain the project's GPU finite-element workflow.

The reproduction guide records the remaining differences: the user-approved radial basal constraint, fixed time steps instead of the published adaptive sequence, reconstructed characteristic tracing/interpolation, and explicit quadrature/tolerance choices. The paper's energy theorem does not certify arbitrary loaded open-boundary H–O runs or guarantee nonlinear convergence. Full GPU trajectory and agreement with the published curves still require validation. Legacy CN–AB2, midpoint, RK3 and coupled-Newton library solvers and their diagnostics remain available; their old checkpoints cannot be resumed by this new paper demo.

Lengths, time, density, viscosity, and stress use **cm–g–s units**. The 2D case assumes unit out-of-plane thickness. Mesh generation runs on the CPU; with `--device cuda`, solid force evaluation, sparse mass solves, IB transfer, and fluid advancement use GPU tensors. Logging, mesh generation, convergence decisions, and file output still involve the host.

## Documentation

**Start here:** [All demos: installation, commands and numerical methods](docs/DEMO_GUIDE.md).
Each demo has a current run guide covering short/full/background runs, parameters,
spatial and temporal discretization, IB coupling, restart and output:
[ideal LV MAC/FEM](demo/ideal_lv_fsi/RUN_GUIDE.md),
[2D valve](demo/ideal_valve_fsi/RUN_GUIDE.md),
[real LV passive/active](demo/real_lv_fsi/RUN_GUIDE.md).
For the optional native GPU kernel, see [CUDA IB installation and environment variables](docs/CUDA_IB.md).

| Topic | Guide |
| --- | --- |
| Public configuration API, parameter units, and project structure | [Configuration guide](docs/CONFIGURATION.md) |
| Solid interfaces, replaceable P1 materials/boundaries and migration limits | [Solid API](docs/SOLID_API.md) |
| External XDMF/HDF5 meshes, DOLFIN boundary tags, and cellwise fibers | [Mesh input API](docs/MESH_INPUT.md) |
| Real-LV passive inflation and active cycles, BE–BE, inputs and background execution | [Real LV run guide](demo/real_lv_fsi/RUN_GUIDE.md) |
| CN–AB2 fluid, midpoint FE/IB, wall Stokes residuals, startup and restart | [CN–AB2 method](docs/MAC_CNAB.md) |
| CN velocity buffers, fused Jacobi stencils and fixed-sweep CUDA Graphs | [CN–Stokes execution](docs/CN_STOKES_EXECUTION.md) |
| Shared P1 IB vector gather and local atomic reduction | [Shared IB execution](docs/SHARED_IB_TILED.md) |
| Adaptive FE/IB quadrature, compact positive rules, and alignment limits | [Gao/FE alignment](docs/GAO_FE_ALIGNMENT.md) |
| Fused adaptive P1 FE/IB transfers, cell-local spread reduction and profiling | [Adaptive IB execution](docs/ADAPTIVE_IB_EXECUTION.md) |
| Optional PyTorch C++/CUDA IB contraction, build, correctness and performance comparison | [CUDA IB extension](docs/CUDA_IB.md) |
| Shared MAC face/center tables and same-step Stokes pressure guesses | [Shared stencil/pressure experiment](docs/MAC_SHARED_PRESSURE.md) |
| Direct P1 coordinates/Peskin template preparation and phase profiling | [IB preparation experiment](docs/MAC_PREPARE.md) |
| Shared-table FE/IB fusion, point-array traffic and checked geometry reuse | [FE/IB memory traffic](docs/SHARED_FE_FUSION.md) |
| Parallel support reduction and pointwise P1 geometry checks | [Validation execution](docs/VALIDATION_EXECUTION.md) |
| Bounded IB buffers, allocator monitoring and CFL-controlled coupled substeps | [Workspace and time control](docs/MAC_WORKSPACE_SUBSTEPS.md) |
| Left-ventricle MAC/FEM demos, restart, and VTK output | [Ideal LV run guide](demo/ideal_lv_fsi/RUN_GUIDE.md) |
| Two-dimensional valve demo and boundary conditions | [Ideal valve run guide](demo/ideal_valve_fsi/RUN_GUIDE.md) |
| AFSI `demo_337` material, loading, geometry, and reference differences | [AFSI337 alignment](docs/AFSI337_ALIGNMENT.md) |
| Finite-element constitutive formulation | [Guccione model](docs/GUCCIONE.md) |
| Compact IB kernels and tensor execution | [MAC execution design](docs/MAC_EXECUTION_PERFORMANCE.md) |
| Fused pressure multigrid | [Multigrid implementation](docs/MAC_MULTIGRID_PERFORMANCE.md) |
| Solid force and consistent-mass optimization | [Solid/mass implementation](docs/MAC_SOLID_MASS_EXPERIMENT.md) |
| Real-LV implicit coupling, CSR solid tangent and Newton equations | [Implicit MAC/FE coupling](docs/MAC_IMPLICIT.md) |
| Real-LV explicit RK3 fluid, stability screen and warmed scheme benchmark | [RK3 MAC/FE coupling](docs/MAC_RK3.md) |
| Three-dimensional pressure graphs and coupling caches | [LV execution design](docs/LV_MAC_GRAPH_EXECUTION.md) |
| Two-dimensional GPU execution | [Valve execution design](docs/VALVE_GPU_EXECUTION.md) |

Detailed guides currently use Chinese. Some contain historical experiment records and test counts; the configuration files shipped with each demo define the current runnable presets.

## Installation

### Requirements

The optimized GPU path targets **Linux with an NVIDIA CUDA-capable GPU**. Development GPU runs have used an RTX 4090. Use Python **3.12** for the documented setup; the package metadata permits Python 3.10 or newer. CPU execution is available for small examples and verification. CUDA Graph performance and the Linux Triton path require a compatible GPU environment.

Install a working NVIDIA driver first and verify that `nvidia-smi` recognizes the GPU. Select a compatible PyTorch build using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/). The CUDA build installed with PyTorch and the maximum CUDA version displayed by `nvidia-smi` need not have identical version numbers.

### 1. Install system dependencies

On Ubuntu/Debian, the following system packages provide Git, native compilation tools, the GLU library used by Gmsh wheels, and the runtime utility used in the background-run example:

```bash
sudo apt-get update
sudo apt-get install -y git build-essential libglu1-mesa time
```

On a managed cluster, use the corresponding installed modules or ask the administrator for missing system libraries.

### 2. Clone the repository

The HTTPS URL works without configuring a GitHub SSH key:

```bash
git clone https://github.com/loveIroha/AFSI_GPU.git
cd AFSI_GPU
```

### 3. Create an isolated Python environment

Install [Miniforge](https://github.com/conda-forge/miniforge#install) if Conda is not already available, then create the environment:

```bash
conda create --override-channels -c conda-forge -n afsi-torch python=3.12 pip -y
conda activate afsi-torch
python -m pip install --upgrade pip
```

### 4. Install PyTorch and AFSI_GPU

Install the CUDA-enabled PyTorch wheel before installing the project. For example, for a machine compatible with the official CUDA 12.8 wheels:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[test,geometry,fused,cuda-ib]"
```

Choose a different official PyTorch wheel index when required by your driver or GPU. Let pip resolve the Triton version compatible with PyTorch; avoid replacing it independently. `torchvision` and `torchaudio` are not needed by these demos.

GPU demo presets use the native C++/CUDA IB extension. Install a CUDA Toolkit compatible with `torch.version.cuda`, configure it following the [CUDA IB guide](docs/CUDA_IB.md), then run `python -m afsi_torch.mac.cuda_ib`. The PyTorch wheel alone does not supply the required compiler. The editable install makes changes to the Python source available immediately. The package is named `afsi-torch`, and its Python import name is `afsi_torch`.

| Dependency / extra | Purpose |
| --- | --- |
| `torch`, `numpy` | Tensor computation, sparse linear algebra, numeric checkpoint storage |
| `geometry` | Gmsh 4.15.2 and meshio 5.3.5 for generated geometry and visualization files |
| `fused` | Triton on Linux for optimized GPU kernels |
| `test` | pytest |
| `io` | meshio and Matplotlib for additional I/O and plotting tools |
| `mesh` | h5py and meshio for external XDMF/HDF5 input and mesh inspection |
| `quadrature` | Basix 0.10.0 for compact positive tetrahedral reference tables in the real-LV demo |
| `reference` | Optional Basix dependency for reference checks |
| `cuda-ib` | Ninja for the C++/CUDA IB extension used by GPU demo presets; matching CUDA Toolkit must be installed separately |

The native PyTorch demos run without a FEniCSx, PETSc, AFSI, Docker, or Taichi installation. Separate native-AFSI comparison scripts require their own reference environment.

For the imported real-LV demo, install `python -m pip install -e ".[test,mesh,fused,quadrature,cuda-ib]"`. Its compact Xiao–Gimbutas rules preserve the selected polynomial degree while reducing interaction-point count. Basix constructs small reference tables during preparation; FE assembly, consistent-mass solves and IB transfer remain PyTorch/GPU computations. The endpoint coupled solver reuses an unchanged converged response while retaining final acceptance checks. Different quadrature rules can change IB kernel sampling; use the [same-checkpoint comparison](docs/GAO_FE_ALIGNMENT.md) to measure both numerical differences and speed.

### 5. Verify the environment

```bash
python - <<'PY'
import torch
import gmsh
import meshio
import afsi_torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA is unavailable in this Python environment"
print("GPU:", torch.cuda.get_device_name(0))
import triton
print("Triton:", triton.__version__)
print("Gmsh:", gmsh.__version__, "meshio:", meshio.__version__)
PY

python -m pytest -q \
  tests/test_simulation_config.py tests/test_mac_output.py
```

For the full test suite, run `python -m pytest -q`. GPU tests are skipped when CUDA is unavailable, so an entirely CPU test run does not establish GPU correctness. First use of compiled kernels and CUDA Graphs can take longer than subsequent steps.

If Python imports NumPy or other packages from a different FEniCSx/Spack environment, start a clean shell and activate `afsi-torch`; check `python -c "import sys, numpy; print(sys.executable); print(numpy.__file__)"`. A missing `libGLU.so.1` indicates a missing Gmsh system library.

### CPU setup

Follow the repository and environment steps above. For CPU-only use, replace the CUDA installation commands with the following, omitting the fused extra:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test,geometry]"

python demo/ideal_lv_fsi/run_mac.py \
  --device cpu --fluid-cells 16 --mesh-size 0.4 \
  --ib-backend reference --execution-backend torch --pressure-backend torch \
  --solid-backend reference --mass-backend pcg --coupling-backend reference \
  --end-time 0.0001 --no-vtk --output results/lv_cpu_smoke
```

This small run explicitly selects the reference execution backends. GPU demo defaults require CUDA and the native IB extension.

## One JSON per demo

```bash
# In the activated CUDA/toolkit environment; commands inherit your GPU selection.
python demo/run.py demo/ideal_lv_fsi/configs/mac_gpu.json
python demo/run.py demo/ideal_lv_fsi/configs/fem.json
python demo/run.py demo/ideal_valve_fsi/configs/mac_gpu.json
python demo/run.py demo/real_lv_fsi/configs/diastole_cuda.json
python demo/run.py demo/real_lv_fsi/configs/active_cuda.json
```

Choose one command. Each JSON declares its demo and all case parameters; the runner
prints a fresh timestamped output directory. Add `--end-time 0.005` for a short run
or `--output results/my_run` to choose the destination. Real-LV presets require your
mesh/fiber/sheet files and use the user H–O coefficients, with passive inflation
or two active 0.8 s cycles. Install/build the [CUDA IB extension](docs/CUDA_IB.md)
first. Its new P2/2D/nodal kernels require target-GPU validation; the measured P1
speedup does not establish performance of every demo.

## Quick start

Run commands from the repository root with `afsi-torch` activated. Use a new output directory for a fresh simulation.

### Left ventricle: optimized MAC fluid

The GPU preset enables fused execution, pointwise solid kernels, graph-based mass and pressure solves, optimized coupling, and warm starts:

```bash
python -u demo/ideal_lv_fsi/run_mac.py \
  --config demo/ideal_lv_fsi/configs/mac_gpu.json \
  --end-time 0.005 --output results/lv_mac_smoke
```

This advances 100 steps. Continue the same trajectory to the full 2 s horizon:

```bash
python -u demo/ideal_lv_fsi/run_mac.py \
  --resume results/lv_mac_smoke/checkpoint.npz --end-time 2.0
```

A fresh full run uses the preset's 2 s end time:

```bash
python -u demo/ideal_lv_fsi/run_mac.py \
  --config demo/ideal_lv_fsi/configs/mac_gpu.json \
  --output results/lv_mac_2s
```

### Valve: optimized two-dimensional MAC fluid

```bash
python -u demo/ideal_valve_fsi/run_mac.py \
  --config demo/ideal_valve_fsi/configs/mac_gpu.json \
  --end-time 0.005 --fluid-fields --output results/valve_mac_smoke

python -u demo/ideal_valve_fsi/run_mac.py \
  --resume results/valve_mac_smoke/checkpoint.npz \
  --end-time 3.0 --fluid-fields
```

### Left ventricle: finite-element fluid

```bash
python -u demo/ideal_lv_fsi/run_fem.py \
  --config demo/ideal_lv_fsi/configs/fem.json \
  --end-time 0.005 --output results/lv_fem_smoke
```

### Background execution and elapsed time

This example starts a fresh 2 s MAC LV run. Create the output directory before redirecting its log:

```bash
run_dir="results/lv_mac_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_dir"

nohup /usr/bin/time \
  -f 'elapsed_seconds=%e exit_code=%x' -o "$run_dir/runtime.txt" \
  python -u demo/ideal_lv_fsi/run_mac.py \
  --config demo/ideal_lv_fsi/configs/mac_gpu.json \
  --output "$run_dir" > "$run_dir/run.log" 2>&1 < /dev/null &

echo "PID=$! output=$run_dir"
tail -f "$run_dir/run.log"
```

The process continues after the terminal closes. `runtime.txt` is written on exit. The report's cumulative elapsed time can include earlier restart segments, whereas `/usr/bin/time` measures this process invocation.

## Configuration

Each demo main program contains a `CONFIG` object. Edit it directly, load a JSON file, or override common parameters on the command line:

**Demo defaults → JSON fields → explicit command-line arguments.**

Export the optimized LV preset as an editable configuration file without starting a simulation:

```bash
python demo/ideal_lv_fsi/run_mac.py \
  --config demo/ideal_lv_fsi/configs/mac_gpu.json --write-config lv.json
```

The main sections cover `time`, `fluid`, solid geometry/material/loading, solver tolerances, execution backends, and output intervals. For example, the following fields specify time and space resolution:

```json
{
  "time": {"dt": 0.00005, "end_time": 2.0},
  "fluid": {
    "shape": [64, 64, 64],
    "lengths": [5.0, 5.0, 5.0],
    "origin": [0.0, 0.0, 0.0],
    "rho": 1.0,
    "mu": 1.0
  },
  "geometry": {"mesh_size": 0.1},
  "output": {"write_vtk": true, "output_every": 400}
}
```

A partial JSON file inherits unspecified fields from the demo's Python `CONFIG`. Keep the `execution` section when editing an exported GPU preset to retain its optimizations. Unknown keys are rejected. `configuration.json` in the result directory records the effective settings and can seed another fresh run.

The public Python entry points are `afsi_torch.simulation.lv_mac.run`, `afsi_torch.simulation.lv_fem.run`, and `afsi_torch.simulation.valve_mac.run`, accepting the corresponding configuration type from `afsi_torch.config`. This lets other applications reuse the runners without importing example scripts.

A restart restores the saved mesh, material, loading, and time step; omit `--config` when using `--resume`. Changes to these physical settings require a fresh output directory. MAC grids must satisfy the multigrid coarsening and explicit time-step limits. The [configuration guide](docs/CONFIGURATION.md) explains rectangular grids, FEM element counts, units, and supported overrides.

## PyTorch implementation

### Nonlinear solid finite elements

The solid stores reference coordinates, connectivity, shape functions, reference gradients, quadrature weights, fiber fields, and the evolving coordinates as tensors. Each force evaluation computes deformation gradients and constitutive stresses at quadrature points, integrates element contributions, and assembles the nodal force by indexed tensor accumulation.

The LV model uses anisotropic Guccione elasticity, a volumetric penalty, active fiber stress, endocardial pressure traction, and basal springs. The valve model uses the FRH constitutive law and root springs. PyTorch automatic differentiation supports independent checks of constitutive derivatives and tangent actions; the production demos use their implemented force kernels. The transient coupled stepper is an explicit, partitioned forward solver, with nonlinear stress evaluated at the updated geometry.

### Fluid solvers

For MAC fluid, pressure is cell-centered and velocity components are stored on cell faces. The solver uses centered conservative convection, explicit viscosity, and a pressure projection with the discrete operator `-D G`. Geometric multigrid applies local stencils, smoothing, restriction, coarse correction, and prolongation directly to structured tensors. The 3D closed box uses homogeneous Neumann pressure conditions and a zero-mean gauge; the 2D channel has prescribed inflow, no-slip walls, and a pressure outlet.

The alternative FEM fluid path uses Q2 velocity/Q1 pressure and Chorin splitting. Fixed finite-element operators are assembled into sparse CSR tensors and reused in GPU iterative solves; its tentative-velocity solve treats viscosity implicitly. The MAC and FEM paths therefore have different fluid discretizations and boundary/operator details.

### Quadrature-based IB coupling for MAC

The MAC coupling follows the weak-form, quadrature-based approach described by [Griffith and Luo](https://doi.org/10.1002/cnm.2888). Solid interaction points move with the FE deformation, and the four-point Peskin kernel connects them to the staggered fluid grid.

Let `B` evaluate FE coefficients at interaction quadrature points, `W` contain their reference integration weights, and `K` interpolate grid velocity to those points. Let `b` be the integrated nodal force and `V_E` the Eulerian cell volume, or area in 2D. The implemented operations are:

```text
M = Bᵀ W B                          consistent reference FE mass matrix
M F = b                             recover force-density coefficients
f = V_E⁻¹ Kᵀ W B F                  spread force to the fluid grid
M U = Bᵀ W K u                      project interpolated velocity to FE nodes
```

The same interaction points and kernel weights are used in both directions. This gives the discrete power relation `bᵀU = V_E fᵀu` up to linear-solve and floating-point error. Boundary extensions are included consistently in the 2D transfer. The mass matrix is assembled once as CSR; both mass solves execute on the selected PyTorch device.

A MAC time step spreads the stored solid force, advances and projects the fluid velocity, interpolates velocity to the solid, updates coordinates, and evaluates the next nodal force. The LV implementation preserves the AFSI-style lagged loading order: the new force uses updated coordinates and the preceding state time. Initial fluid velocity and stored force are zero. The FEM-fluid comparison path retains its existing nodal IB transfer, so it should not be confused with the MAC quadrature transfer above.

## GPU performance

The implementation addresses GPU launch overhead, temporary tensors, repeated global-memory access, and host/device synchronization alongside sparse arithmetic.

| Component | Implementation | Intended benefit |
| --- | --- | --- |
| FE operators and IB mass matrix | Assemble COO contributions, coalesce, convert to CSR, and reuse fixed matrices | Avoid rebuilding matrices and repeating element-wise mass actions during each solve |
| Solid constitutive kernels | `torch.compile` / Inductor; fused scalar 3×3 Guccione algebra in the pointwise backend | Reduce small batched-matrix calls and intermediate tensors |
| IB transfer | Compact separable four-point stencils and Triton gather/spread kernels | Reduce expanded index/weight storage and repeated tensor passes |
| Pressure multigrid | Fused five-/seven-point stencils, residual/restriction and prolongation/correction kernels | Reduce kernel launches and intermediate fine-grid arrays |
| CN velocity Helmholtz | Fused pressure/RHS initialization, fixed-buffer Jacobi stencils and captured sweep chains | Reduce repeated velocity-field allocation, ghost/Laplacian arrays and host submissions |
| Shared P1 IB alternatives | Cell-owned vector gather; four-point-tile spread reduction | Reuse cell/shape data and reduce repeated global atomics; opt-in pending native measurements |
| Iterative solver storage | Preallocated workspaces and fixed-address double buffers | Reuse allocations and support repeatable execution graphs |
| CUDA Graph replay | Capture pressure V-cycle blocks and CSR/Jacobi-PCG iteration blocks between convergence checks | Reduce Python and CUDA submission overhead |
| Warm starts and geometry reuse | Reuse prior mass-solve coefficients; cache accepted geometry and IB stencils with invalidation | Reduce iterations and duplicate evaluation |
| Diagnostics | Combine acceptance flags and sample expensive diagnostics at configured intervals | Reduce frequent host reads while retaining numerical acceptance checks |

These execution backends retain FP64, the chosen quadrature, consistent mass matrices, solver tolerances, and true-residual checks. They do not replace the mass solve with diagonal lumping. Pressure multigrid uses structured stencil actions, while FE mass and FEM fluid operators use assembled sparse matrices. First-run compilation/capture and scheduled visualization still contribute to total run time.

### Example measurements

The following are development-run wall times reported on an **RTX 4090**, useful as scale estimates rather than portable benchmark guarantees:

| Case | Problem size and horizon | Reported elapsed time |
| --- | --- | --- |
| 3D ideal LV, optimized MAC | 64³ fluid cells; 28,840 P2 solid nodes; 243,502 interaction points; 40,000 steps / 2 s | 931.79 s, approximately 15.5 min |
| 2D ideal valve, optimized MAC | 256 × 64 fluid cells; 1,986 P2 solid nodes; 48,000 steps / 3 s | 333.48 s, approximately 5.6 min |

The LV timing predates regular VTK time-series export; the current GPU preset enables that output. Runtime depends on output frequency, host CPU, software versions, compilation, load phase, mesh, and solver iterations. These figures are not a matched-discretization speedup comparison with native AFSI. Larger meshes and different GPUs require measurement, especially because FP64 throughput and memory traffic both matter.

For a controlled comparison, replay an existing checkpoint using the supplied benchmark tools. For example:

```bash
python -u validation/benchmark_lv_mac_graph.py \
  --checkpoint results/lv_mac_2s/checkpoint.npz \
  --device cuda --warmup 10 --steps 100 --profile \
  --output results/lv_mac_benchmark/report.json
```

Use an existing checkpoint path. These benchmarks compare state equivalence as well as speed. Warmup/compilation is separated from throughput, and optional phase profiling uses a separate replay. A short replay at the end of loading does not predict the entire transient runtime; nested phase timings should not be added together.

For execution bottlenecks, the [GPU profiling guide](docs/GPU_KERNEL_PROFILING.md) provides short coupled traces and frozen-stencil FE/IB kernel comparisons. It separates uninstrumented timing from profiling overhead, preserves checkpoint history, and includes optional Nsight commands for register, cache and atomic-traffic evidence. Normal demo runs do not enable these diagnostics.

## Results and visualization

Runs produce `configuration.json`, `report.json`, `history.csv`, and `checkpoint.npz`. The checkpoint preserves the restart state; CSV contains sampled scalar diagnostics.

The 3D MAC LV demo also writes:

- `vtk/solid.pvd`: deformed P2 tetrahedral meshes (`.vtu`), displacement, nodal force, reference fibers, and element `J_min/J_max`.
- `vtk/fluid.pvd`: structured fluid fields (`.vti`) containing pressure, display velocity, and divergence.
- `vtk/fields.json`: field definitions and units.

Open the two `.pvd` files in ParaView and apply the readers to animate the solution. Solid coordinates are already deformed. MAC display velocity is reconstructed at cell centers; the checkpoint retains the original staggered velocity arrays. For point-based flow visualization, apply **Cell Data to Point Data** as needed.

With the LV preset, `output_every=400` and `dt=5e-5` give a 0.02 s frame interval. The 2D valve writes its collections under `fields/`; use `--fluid-fields` to include its fluid output. See the individual demo guides for FEM visualization and single-checkpoint export.

## Project structure

```text
src/afsi_torch/
  config.py          Public case configuration and JSON handling
  mesh_io.py         Static XDMF/HDF5 meshes and DOLFIN cell/facet fields
  simulation/        Reusable LV MAC, LV FEM, and valve runners
  geometry/          Solid geometry, boundary tags, and fiber generation
  mac/               3D MAC fluid, multigrid, FE/IB transfer, GPU execution
  mac2d/             2D channel fluid, multigrid, FE/IB transfer
  fluid/             Q2/Q1 FEM fluid, sparse operators, linear solvers
  solid.py           Solid finite-element quadrature and assembly
  materials.py       Constitutive models
  afsi337.py         Ideal-LV reference settings and model construction
  afsi340.py         Ideal-valve geometry and solid model

demo/                Runnable cases, editable CONFIG objects, JSON presets
examples/            Small examples and compatibility entry points
validation/          Numerical comparisons, performance tools, data export
scripts/             Optional native-AFSI/container workflows
tests/               Numerical, restart, output, and execution tests
docs/                Detailed methods and experiment documentation
```

## Validation and scope

Tests cover FE and constitutive identities, derivative checks, fluid operators and projection, IB transfer/power consistency, iterative-solver convergence, reference/optimized equivalence, checkpoint restart, configuration propagation, and visualization layout. Selected reference comparisons use NumPy or native FEniCSx/AFSI outside the main runtime.

The included LV and valve simulations have completed their prescribed horizons in development runs. Mesh/time-step convergence and agreement with a particular native AFSI trajectory still need to be established for each chosen case. The generated LV mesh/fibers are not asserted identical to external AFSI input files. The 2 s LV loading-and-holding example does not include a closed-loop circulation or a physiological cardiac cycle, and the valve example has no contact model. End-to-end differentiation of a transient FSI run is not currently exposed.

The current demos use one selected GPU. `CUDA_VISIBLE_DEVICES=0` selects a device; it does not enable multi-GPU domain decomposition.

When reporting an issue, include the command/configuration, commit, Python/PyTorch/Triton versions, GPU and driver, and the relevant report or traceback. Include output settings when comparing runtime.

## References and acknowledgements

- [AFSI](https://github.com/loveIroha/afsi): reference formulations and the ideal-LV `demo_337` and ideal-valve `demo_340` cases.
- [torchcor](https://github.com/sagebei/torchcor): inspiration for GPU finite-element computation using PyTorch tensors, assembly, and sparse solvers; its cardiac electrophysiology solver is a separate project.
- Griffith, B. E., and Luo, X. (2017). [Hybrid finite difference/finite element immersed boundary method](https://doi.org/10.1002/cnm.2888). *International Journal for Numerical Methods in Biomedical Engineering*, 33, e2888.
- [MAC-taichi](https://github.com/houkensjtu/MAC-taichi): reference for MAC predictor/correction design. AFSI_GPU implements its operators in PyTorch/Triton; see the retained [third-party notice](docs/MAC_TAICHI_LICENSE.txt).
