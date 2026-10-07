# PyTorch C++/CUDA IB：安装、运行与实现指南

[全部 demo](DEMO_GUIDE.md) · [真实 LV 完整运行](../demo/real_lv_fsi/RUN_GUIDE.md)

## 适用范围与源码位置

C++/CUDA 是全部 GPU demo 预设的主要 IB 后端。三维 P1 真实 LV 使用自适应积分 CSR 组装；理想 P2 左心室使用紧凑 MAC 插值/传播，二维 P2 瓣膜使用带壁面反射的 indexed 传递，FEM 流体使用节点 indexed 传递。根据各自离散选择内核，统一通过 PyTorch dispatcher 调用。
在仓库根目录下：

| 文件 | 作用 |
| --- | --- |
| [ib.cu](../src/afsi_torch/mac/csrc/ib.cu) | CUDA 积分权重、哈希插入、缓存更新内核 |
| [ib.cpp](../src/afsi_torch/mac/csrc/ib.cpp) | P1 组装输入检查及算子注册 |
| [transfer.cu](../src/afsi_torch/mac/csrc/transfer.cu) / [transfer.cpp](../src/afsi_torch/mac/csrc/transfer.cpp) | P2 MAC、二维反射及 FEM 节点传递内核/注册 |
| [cuda_ib.py](../src/afsi_torch/mac/cuda_ib.py) | 工具链检查、JIT 编译/缓存和调用入口 |
| [cached_transfer_assembly.py](../src/afsi_torch/mac/cached_transfer_assembly.py) | 缓存 CSR 支撑结构，更新数值 |
| [assembled_transfer.py](../src/afsi_torch/mac/assembled_transfer.py) | 用组装矩阵执行双向 IB 响应 |

## 为什么以前命令有多个 export

`export` 让当前 shell 的变量传给 Python、Ninja、nvcc 等子进程。这些变量配置工具链，不改变数值方法，也不指定 GPU 求解线程数。

| 变量 | 作用 | 是否必须手写 |
| --- | --- | --- |
| `CUDA_HOME` | CUDA 开发工具链根目录；PyTorch 用它定位 nvcc/头文件/库 | 自动检测正确可省略；本项目曾误用系统 CUDA 12，Conda 中装 CUDA 13 时建议固定 |
| `PATH` | 决定终端优先找到哪个 nvcc/编译器 | `conda activate` 通常已加入环境 bin；确认 `command -v nvcc` 后不必重复加 |
| `TORCH_CUDA_ARCH_LIST=8.9` | 为 RTX 4090 的架构生成内核 | 可省略，PyTorch 通常从可见 GPU 推导；固定它可稳定编译目标。其他卡不要照抄 8.9 |
| `MAX_JOBS=2` | 限制 Ninja 编译进程数，降低 CPU/RAM 峰值 | 可省略；不会把 GPU 计算限制为两个线程 |
| `CUDA_VISIBLE_DEVICES=0` | 运行时选择可见 GPU | 可按机器情况设置；不是编译器路径，也不会启用多 GPU 分解 |

普通 `export` 只在当前 shell 和后续子进程中生效。不能认为“编译过一次，以后不需要工具链”：当前 `build()` 在**每个新 Python 进程**首次使用时仍检查 nvcc，并调用 PyTorch 的缓存加载器；命中缓存通常不重新编译。源代码、编译参数、PyTorch 或工具链变化可能触发重建。

### 当前 Conda + RTX 4090 环境的一次性设置

仅当 Toolkit 确实安装在该 Conda 环境中时使用下面的 `CUDA_HOME`。对于系统 `/usr/local/cuda-*` 或集群 module，请使用其实际目录。

```bash
conda activate afsi-torch
# 仅首次缺少工具链时安装；这里针对 torch.version.cuda == '13.0'
conda install -c nvidia "cuda-toolkit=13.0" -y
python -m pip install -e ".[test,mesh,fused,quadrature,cuda-ib]"

# 写入这个 Conda 环境的设置，之后无需每次复制 export
conda env config vars set CUDA_HOME="$CONDA_PREFIX" TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS="2"
conda deactivate
conda activate afsi-torch
command -v nvcc
"$CUDA_HOME/bin/nvcc" --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -m afsi_torch.mac.cuda_ib
```

已经安装且编译成功的环境可跳过安装和验证，保留一次性环境设置。不要为此替换已经能工作的 C/C++ 编译器。未固定架构时也可通过 `torch.cuda.get_device_capability()` 查看当前设备能力。

工具链版本应与 **`torch.version.cuda`** 匹配；`nvidia-smi` 显示的是驱动能力。当前扩展会拒绝 CUDA 大版本不匹配。PyTorch 2.14 的编译日志使用 C++20，不能只按“C++14 以上”推断旧 nvcc 一定可用。参考 [PyTorch 扩展文档](https://docs.pytorch.org/docs/2.14/cpp_extension.html) 和 [CUDA 13 安装说明](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-installation-guide-linux/index.html)。

## 环境启动文件

对于已在 Conda 内安装 Toolkit 的两张 RTX 4090 机器，启动文件可写为：

```bash
conda activate afsi-torch
export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS=2
```

在已经能使用 conda 的 shell 中 `source` 这个文件，变量才会保留在当前终端；用 `bash 文件名.sh` 启动的子 shell 不会修改父终端。若该文件由批处理作业 source，需先按本机 Conda 安装方式初始化 conda。这里不额外覆盖 `CC/CXX` 或 `LD_LIBRARY_PATH`。选择物理 GPU 1 后进程内部的 `cuda:0` 就是它；运行命令不要再前缀 `CUDA_VISIBLE_DEVICES=0`。上面的启动文件与前面的 Conda 持久配置二选一即可。

## JSON 直接运行

```bash
python demo/run.py demo/real_lv_fsi/configs/active_cuda.json
python demo/run.py demo/ideal_lv_fsi/configs/mac_gpu.json
python demo/run.py demo/ideal_valve_fsi/configs/mac_gpu.json
python demo/run.py demo/ideal_lv_fsi/configs/fem.json
```

每行是单独实验。参数、材料、网格与时间都可改 JSON；无需重复手写后端开关。C++ 源文件新增后首次加载会增量编译。服务器上应先执行新增 GPU 接入测试：

```bash
python -m afsi_torch.mac.cuda_ib
python -m pytest -q tests/test_cuda_ib_demos.py tests/test_cuda_ib.py
```

新增的 P2/2D/FEM 测试覆盖独立传递对照、功率配对、壁面反射、非默认流与 Graph、加载状态推进和检查点。CPU 环境会跳过 CUDA 测试，不能当作新 CUDA 内核已在 GPU 验收。本次此前的 1.20 倍测速只属于 P1 CSR 路径，P2/2D 路径应另做性能比较。

## 原入口：显式选择后端

```bash
conda activate afsi-torch
mesh_dir="/mnt/large2/gjh/realistic_left_ventricle"
python -u demo/real_lv_fsi/run_mac.py \
  --config demo/real_lv_fsi/active_cycle.json --mesh-dir "$mesh_dir" \
  --device cuda --dt 1e-4 --cycles 2 \
  --nonlinear-solver anderson-newton --anderson-policy legacy \
  --anderson-max-iterations 6 --newton-preconditioner none --linear-policy inexact \
  --ib-response-backend csr --ib-csr-assembly-backend cached-hash \
  --ib-csr-contraction-backend cuda --ib-max-order 22 --ib-max-points 12000000 \
  --output results/real_lv_active_cuda
```

这条命令前台运行；完整后台版、纯舒张、VTK、续算见 [真实 LV 指南](../demo/real_lv_fsi/RUN_GUIDE.md)。选项 `csr`、`cached-hash`、`cuda` 分别控制响应方式、CSR 构建方式和积分权重内核。`cuda` 也可搭配 `hash`，不能搭配 `coalesce`。

日志应显示 CSR/cached-hash/cuda；最终配置中的 `ib_csr_contraction_backend` 应为 `cuda`，组装诊断执行名称为 `cpp-cuda-warp-sites`。该选项不会把所有 IB 操作改写为 C++：质量矩阵求解、CSR 乘法等继续走已有 PyTorch GPU 路径。

## 已测性能与适用边界

RTX 4090、FP64、128³，纯舒张检查点后 `1.6005–1.6025 s` 的同状态对照（预热 5 步、测量 20 步）：`sites` 781.17 ms/步，`cuda` 649.81 ms/步，总体 1.20 倍加速、耗时减少 16.8%。单独 3 步采样中，数值积分从 281.47 降至 152.22 ms/步，CSR 总组装从 310.91 降至 182.03 ms/步。分项有嵌套，不能相加。

两者均为每步 11 次流体响应、22 次质量求解；终态最大位置差 2.13e-14 cm、速度差 2.20e-10 cm/s、压力差 5.12e-7 dyn/cm²，均通过各自接受检查。此结果说明该短段的执行优化有效，不代表整个主动收缩周期已验证，也不是普适加速比。纯舒张结束后的检查点继续保持压力，主动周期结束后的检查点才进入下一周期。

## 实现与开发参考

`--ib-csr-contraction-backend cuda` selects a custom CUDA implementation of
adaptive P1 quadrature contraction and hash insertion. It is registered through
PyTorch's dispatcher (`torch.ops.afsi_ib_cuda`) and consumes existing CUDA
tensors directly. GPU demo presets select CUDA; library/reference defaults retain
`sites` for compatibility. The following section describes the adaptive P1 CSR path.
Compilation time is excluded from warmed benchmarks.

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
python -m afsi_torch.mac.cuda_ib
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
python -m pytest -q \
  tests/test_cuda_ib.py tests/test_csr_cached_assembly.py tests/test_csr_hash_assembly.py
```

GPU tests execute the actual extension and compare with an independent quadrature
oracle. They cover layouts, FP32/FP64, large keys, collision/overflow handling,
changed support/rules, old snapshot ownership, force/power consistency, nondefault
streams, CUDA Graph replay, and invalid input metadata. GPU tests are explicitly
skipped on CPU-only machines; a CPU test pass is not CUDA verification.

Use the same checkpoint for both variants. Give this process exclusive use of
the selected GPU. Select a checkpoint in the phase you want to measure: a finished
1.6 s passive checkpoint measures the held-pressure phase, while an active
checkpoint at that time enters the next cycle. Neither samples peak contraction.

```bash
checkpoint="/path/to/the/current/simulation/checkpoint.npz"
python -u validation/benchmark_paper_lv.py \
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
python -u demo/real_lv_fsi/run_mac.py \
  --resume "$checkpoint" --device cuda --end-time 1.6 \
  --ib-response-backend csr --ib-csr-assembly-backend cached-hash \
  --ib-csr-contraction-backend cuda
```

Resume must start before the requested end time. Use the original configuration
for a new run; add the three IB backend options above. Retain `sites` if the CUDA
variant is slower or does not pass the equivalence checks.
