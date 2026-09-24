# Q2/Q1 流体有限元算子（0.8.0）

本阶段提供规则、轴对齐六面体上的算子作用，不组装全局稠密矩阵。速度每单元 27 个 Q2 节点，每节点 3 分量；压力每单元 8 个 Q1 节点，每节点 1 分量。网格坐标使用 cm，节点按 x 最快、然后 y、z 编号，与既有 IB 速度格点顺序一致。

## 运行与状态

无需增加依赖，沿用 0.7.0 的环境：

```bash
git pull --ff-only
conda activate afsi-torch
python -m pip install -e ".[test,geometry]"
CUDA_VISIBLE_DEVICES=0 python examples/fluid_patch.py --device cuda
CUDA_VISIBLE_DEVICES=0 python -m pytest -q
```

示例应输出 `status: passed`，完整 CPU/CUDA 测试为 155 项。本地无 CUDA 时为 88 passed、67 skipped。测试使用小型制造场，示例中的盒子不是未来左室的流体域。当前没有施加本质边界条件、压力基准，也没有线性求解、Chorin 时间步或完整流固耦合。

## 数学定义

记速度试函数 v、压力试函数 q，M 为一致质量矩阵。以下均返回**组装后的弱式载荷向量**，不是点值微分结果，也未乘 rho、mu 或 dt：

| 接口 | 定义 |
| --- | --- |
| `velocity_mass(u)` | ∫ v·u dV |
| `velocity_stiffness(u)` | ∫ grad(v):grad(u) dV |
| `pressure_mass(p)` | ∫ q p dV |
| `pressure_stiffness(p)` | ∫ grad(q)·grad(p) dV |
| `divergence(u)` | D u = ∫ q div(u) dV |
| `gradient(p)` | G p = ∫ v·grad(p) dV |
| `divergence_transpose(p)` | Dᵀp = ∫ p div(v) dV |
| `convection(u, advector=a)` | C(a,u) = ∫ v·((a·grad)u) dV，默认 a=u |
| `density_load(f)` | M f = ∫ v·f_h dV，f_h 是 Q2 插值力密度 |

分部积分给出 `vᵀG p + pᵀD v = ∫boundary p(v·n) dS`。在未约束边界上，**G 不能直接替换成 −Dᵀ**。测试检查内部速度自由度上的负转置关系以及非零边界通量。黏性采用原 AFSI Chorin 示例的分量 Laplacian 形式，而非自动改成对称应变形式；对流采用 advective 形式，尚未实现稳定化。

每个参考单元使用 4×4×4 Gauss 积分。Q2 对流三重乘积在单一坐标方向的最高次数可到 6，三点 Gauss 不足；因此接口拒绝少于四点。测试还与五点积分结果比较。核使用 PyTorch gather、einsum 和 index_add，支持 CUDA 和自动微分；只在设置积分规则时使用 NumPy。没有依赖 SciPy/PETSc 做运行时算子计算，也尚未优化为和因子分解或分块处理，大网格显存及性能需要后续测量。

## 与 IB 的接口

若背景每轴 n 个单元，Q2 速度格点数为 2n+1，间距为单元边长的一半。`mesh.velocity_grid` 提供兼容的 IB 描述，但四点核要求每轴至少两个单元，固体位置仍需满足完整支撑条件。

两条载荷路径不可混用：

1. `ib.spread_density(g, stencil)` 得到 `f = Hᵀg/dV_lattice`；按照原 AFSI 将 f 视为 Q2 力密度时，流体右端是 **M f**，通过 `density_load` 计算。
2. `ib.spread_load(g, stencil)` 得到 **b = Hᵀg**；若选择直接对偶载荷耦合，则直接进入弱式右端，不能再乘 M。

一致 M 不等于 `dV_lattice I`。第二条路径具有 `uᵀb=(Hu)ᵀg` 的代数功率恒等式；第一条路径一般不具备同一个恒等式。本轮用实际质量算子验证这一差别，尚未替后续耦合默认选择路径，也不声称原密度路径满足完全离散能量守恒。

## 独立 DOLFINx 对照

[本版 Linux CPU 验证已通过](https://github.com/loveIroha/AFSI_GPU/actions/runs/36021971483)，40 组向量最大绝对差 2.43e-15。目标 CUDA 外部对照仍需下述命令。

`export_fluid_dolfinx.py` 在独立 CPU 环境用 UFL 装配，不导入 PyTorch 或本项目流体模块；在带非零原点、各向异性尺寸的 2/16 单元网格上分别导出仿射和一般节点场。压力、速度顺序通过物理坐标匹配，包含完整边界自由度。每个场比较 9 个算子以及对流项的 UFL 方向导数，共 40 组向量；参考采用更高的 8 次积分精度。

已有 `afsi-reference` 环境时：

```bash
conda run --no-capture-output -n afsi-reference python validation/export_fluid_dolfinx.py
conda run --no-capture-output -n afsi-reference python validation/export_fluid_dolfinx.py --subdivisions 2 --output validation/results/fluid_fine.npz
conda activate afsi-torch
CUDA_VISIBLE_DEVICES=0 python validation/compare_fluid_dolfinx.py --device cuda
CUDA_VISIBLE_DEVICES=0 python validation/compare_fluid_dolfinx.py --device cuda --reference validation/results/fluid_fine.npz
```

如尚未创建 CPU 参考环境，使用 `conda env create -f validation/environment-dolfinx.yml`。也可从本版 Actions 的 `dolfinx-reference` 附件下载两个 fluid NPZ，放入 `validation/results/`，直接运行 CUDA 比较，无需在 GPU 环境安装 FEniCSx。

单位遵循 cm–g–s：速度 cm/s、压力 dyn/cm²、密度 g/cm³、动力黏度 g/(cm·s)、体力密度 dyn/cm³。几何积分会引入对应的 cm 次幂，调用方负责物理系数。

方程依据：[AFSI Chorin 示例](https://github.com/npuheart/afsi/blob/99df0ffba795fa05043ba874ad00353dcb986466/afsic/src/afsic/euler/ChorinSolver.py)。本项目独立实现这些数学算子，没有复制原求解器代码。
