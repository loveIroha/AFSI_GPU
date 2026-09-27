# AFSI 原生 IB/Chorin 与 PyTorch GPU 的单步对照（0.23.0）

此实验直接运行 [AFSI 的 Chorin 类](https://github.com/loveIroha/afsi/blob/main/afsic/src/afsic/euler/ChorinSolver.py)与[实际编译的 3D IB C++ 扩展](https://github.com/loveIroha/afsi/blob/main/afsic/src/coupling/IBMesh3D.h)。参考端使用 DOLFINx/PETSc；对照端使用本项目的 PyTorch 流体及 IB 算子。两端均在 4 cm 盒、每方向 4 个 Q2 单元、`dt=5e-5 s`、`rho=mu=1`、全壁零速度和一个零压力规范自由度下，从零流速推进一步。IB 使用 AFSI 的四点 Peskin 核，核尺度等于 Q2 速度节点间距 **0.5 cm**。

一个小型 P1 固体网格只提供相同位置上的**给定积分节点力**。本实验逐项比较力密度、Q2 弱载荷、暂定速度、压力、修正速度、插值所得固体速度以及更新后的坐标，并记录第一个超出容差的阶段。这里不重做固体 Guccione 有限元装配；这部分已有独立的 DOLFINx 固体参考检验。参考脚本还核对其逐阶段插桩与未修改的 AFSI `solve_one_step()` 最终场一致。当前本地环境没有 DOLFINx/afsic，原生结果尚未运行；不能把本地比较器测试当作跨项目数值吻合。

在能导入 `dolfinx`、`petsc4py`、`mpi4py`、`afsic`（含其 C++ 扩展）的 **AFSI CPU 环境**中，从 AFSI_GPU 仓库目录运行：

```bash
python validation/export_afsi_native_step.py \
  --output results/afsi_native_step/reference.npz
```

然后切换到已安装本项目的 GPU 环境，从同一仓库目录运行：

```bash
conda activate afsi-torch
CUDA_VISIBLE_DEVICES=0 python validation/compare_afsi_native_step.py \
  --reference results/afsi_native_step/reference.npz \
  --device cuda --output results/afsi_native_step/report.json
```

即使比较失败，脚本也先保存 `report.json`，其中的 `first_mismatch` 表示最早超过容差的阶段。若为 `fluid_density`，优先查 IB C++ 节点映射与核权重；若为 `fluid_weak_load`，查 PyTorch 的 Q2 质量/弱式装配；若首次出现在后三个流体阶段，查 Chorin 弱形式、边界规范或线性求解残差。比较器按物理坐标匹配 DOLFINx 和 PyTorch 的自由度编号，不依赖两端编号恰好相同。请保留参考 NPZ 与报告以供复核。

本实验也参照 [torchcor 单元组装](https://github.com/sagebei/torchcor/blob/main/torchcor/core/assemble.py)的批量张量计算和 [CG 求解器](https://github.com/sagebei/torchcor/blob/main/torchcor/core/solver.py)的 GPU 迭代思路。本项目已有批量 Q2/Q1 单元积分、`index_add` 全局散加和 GPU PCG，并对非线性固体重算单元力；因此继续使用当前矩阵自由算子。torchcor 的公开组装器主要面向低阶电生理矩阵，部分 COO 构造调用 `.tolist()`；直接替换当前 3D Q2/Q1 与非线性固体算子会改变离散实现，并可能把数据搬回 CPU。此阶段不需要安装 `torchcor` 包。后续若评估稀疏 CSR 组装，须先与当前矩阵自由算子在同一网格上比较作用、显存和耗时。

这一步检验的是**同一小规模离散输入下的实现对齐**。AFSI 理想左室算例使用外部网格和纤维文件、较细流体网格及大得多的压力/主动张力；这些条件不包含在该单步参考里。单步吻合后才能把仍存在的长时间差异归因于时间步、载荷、几何或空间分辨率。
