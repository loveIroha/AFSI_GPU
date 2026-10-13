# 理想左心室：对齐原生 AFSI 的 PyTorch FEM/IB 验证

本验证使用 `demo/ideal_lv_fsi/run.py`（转发到 `run_fem.py`），不使用 MAC。
配置为 `demo/ideal_lv_fsi/configs/afsi337_fem.json`，从 AFSI 容器导入同一份网格、
P2 fiber/sheet 系数、表面标签和安装版本的积分规则。通用生成几何配置仍为 `configs/fem.json`。

## 数值格式

| 部分 | 本验证采用的格式 |
|---|---|
| 固体 | P2 四面体；Guccione，C=20000、bf=8、bt=2、bfs=4、kappa=5e5 |
| 主动应力 | `Ta (F f0) ⊗ f0`，与 demo_337 相同 |
| 基底 | 三方向参考位移弹簧，beta=5e5；不是零位移强约束 |
| 外载 | 压力与张力同时在 0–1.5 s 线性升至 150000 和 600000 dyn/cm²，随后保持到 2 s |
| 流体 | 32³ 六面体，Q2 速度/Q1 压力；盒子 [0,5]³ cm，rho=1，mu=1 |
| 时间 | dt=5e-5 s，40000 步；显式对流、隐式黏性，非增量 Chorin |
| 速度暂定步 | `(rho/dt Mv + mu Kv) u* = rho/dt Mv un - rho C(un) + Mv f` |
| 压力与修正 | `Kp p = -rho/dt D u*`，`Mv u_new = Mv u* - dt/rho G p` |
| 流体边界 | 六面无滑移；压力 Neumann，原点单自由度定标 |
| IB | 固体 P2 节点上的四点 Peskin 核；规则 Q2 速度节点网格；epsilon=h=5/64 cm |
| 传播 | 已积分节点力 b → `f=Hᵀb/h³` →流体弱载荷 `Mv f` |
| 插值 | `U=H u_new`；没有固体质量矩阵逆，也没有 MAC 积分点 IB |
| 耦合顺序 | 初始力为零；用旧坐标和旧力推进流体，再更新坐标，最后用旧时刻外载计算下一步力 |
| GPU 执行 | PyTorch 批量固体积分/组装、流体 COO→CSR、GPU 稀疏乘法和 PCG；IB 用 PyTorch C++/CUDA 算子 |

原生 AFSI 压力方程和速度修正没有显式 rho 因子，在这个 rho=1 算例上与上式相同。
改变 rho 后不应再宣称逐式复现该原生实现。线性求解器也有明确区别：原生采用 PETSc
BCGS/HYPRE 与 CG/SOR，这里用 GPU Jacobi-PCG；比较收敛后的场和真实残差，不比较迭代次数。

这里的流体黏性隐式不意味着整体 FSI 隐式。固体位置与力依旧按原 AFSI 的显式滞后顺序推进。
程序仍保留负体积、非有限数、IB 支撑域和位移步幅检查。

## 准备原生数据

在 Linux 主机的 AFSI_GPU 仓库执行，容器必须正在运行，且原 demo_337 的输入已经存在：

```bash
conda activate afsi-torch
bash scripts/prepare_afsi337_fem_reference.sh results/afsi337_reference
```

默认容器 `afsi_dev_ljy`，可用 `AFSI_CONTAINER` 修改。脚本只导出数据和独立的小型原生单步参考，
不启动原生完整心室模拟、不上传 SwanLab、不覆盖已有参考。无需在容器安装 PyTorch。
`solid.npz` 包含求解所需的全部原生固体数据；配置中的路径相对于运行工作目录。

首次使用 C++/CUDA IB 前，按 [CUDA IB 指南](CUDA_IB.md) 配置与 PyTorch 匹配的 CUDA Toolkit，
并运行 `python -m afsi_torch.mac.cuda_ib`。不要把驱动显示的 CUDA 版本当作编译器版本。

## 独立对照与回归

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/test_afsi337_fem_alignment.py tests/test_afsi_native_step.py \
  tests/test_afsi337.py tests/test_fluid.py tests/test_ib.py tests/test_coupling.py tests/test_csr.py

CUDA_VISIBLE_DEVICES=0 python validation/compare_afsi337_force.py \
  --solid-input results/afsi337_reference/solid.npz \
  --reference results/afsi337_reference/force.npz \
  --output results/afsi337_fem_alignment/solid_force.json

for state in zero advecting; do
  CUDA_VISIBLE_DEVICES=0 python validation/compare_afsi_native_step.py \
    --reference "results/afsi337_reference/$state.npz" \
    --device cuda --backend csr --ib-backend cuda \
    --output "results/afsi337_fem_alignment/${state}_cuda.json"
done
```

固体对照直接调用容器中的 GuccioneMaterial，并核对零载及非均匀变形加载状态的完整节点力。
流体对照从容器中的实际 Chorin/C++ IB 导出结果，逐阶段检查传播密度、有限元载荷、暂定速度、
压力、修正速度、结构速度和更新坐标。非零初速度案例覆盖对流项，避免零初速度掩盖对流错误。
这些检查通过只证明所测输入的阶段一致性，不能代替完整 2 s 的曲线对照和收敛分析。

## 运行

短运行仍使用完整空间网格及同一输入：

```bash
CUDA_VISIBLE_DEVICES=0 python -u demo/ideal_lv_fsi/run.py \
  --config demo/ideal_lv_fsi/configs/afsi337_fem.json \
  --device cuda --end-time 0.005 --output results/afsi337_fem_early
```

完整 2 s 后台运行：

```bash
run_dir="results/afsi337_fem_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_dir"
CUDA_VISIBLE_DEVICES=0 nohup /usr/bin/time \
  -f 'elapsed_seconds=%e exit_code=%x' -o "$run_dir/runtime.txt" \
  python -u demo/ideal_lv_fsi/run.py \
  --config demo/ideal_lv_fsi/configs/afsi337_fem.json \
  --device cuda --output "$run_dir/simulation" \
  > "$run_dir/run.log" 2>&1 < /dev/null &
echo "PID=$! output=$run_dir"
```

修改 JSON 可设置网格、时间步长、材料、载荷及输出频率；偏离参考配置后属于新的实验。
`--solid-input` 可以覆盖 JSON 路径。导入原生固体时拒绝 `--mesh-size`，防止误以为重新剖分生效。
检查点已嵌入网格和纤维，续算不依赖原生输入文件：

```bash
CUDA_VISIBLE_DEVICES=0 python -u demo/ideal_lv_fsi/run.py \
  --resume results/你的运行目录/simulation/checkpoint.npz --device cuda --end-time 2
```

`history.csv` 同时记录腔体积、壁体积、位移与流速；与原 AFSI 日志中的 `volume` 对比时应选择壁体积。
VTK 输出使用实际当前固体坐标，不要重复 Warp。`report.json` 记录离散格式、输入文件指纹、
实际后端与原生线性求解器差异，不能用旧 MAC 结果代替本次 FEM 验证结果。

## 服务器验收记录（2026-10-13）

相关 CPU/CUDA 回归 104 项通过；加载固体力相对误差为 3.41e-12、1.34e-12，fiber/sheet 系数逐项完全一致。零与非零初速度下，CSR + 参考/CUDA IB 的七个阶段均通过原生对照。完整空间网格已续算至 0.02 s，实际构型指标见 [机器可读验收记录](validation/afsi337_fem_20261013.json)。完整 2 s 的验证仍需等待该轨迹完成。
