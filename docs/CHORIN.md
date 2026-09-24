# PyTorch GPU 流体三步求解（0.9.0）

本版从“计算算子作用”进入“求解未知流体速度和压力”。实现非增量 Chorin 三步法，沿用所选 AFSI 示例的显式对流、隐式分量黏性、单点压力基准与速度校正形式。当前只求解规则背景盒内的流体，尚未连接固体运动。

## 运行

沿用现有环境，无新依赖：

```bash
git pull --ff-only
conda activate afsi-torch
python -m pip install -e ".[test,geometry]"
CUDA_VISIBLE_DEVICES=0 python examples/chorin_box.py --device cuda --cells 4 --steps 5 --output results/chorin_box.json
CUDA_VISIBLE_DEVICES=0 python -m pytest -q
```

完整测试应为 174 项；本地无 CUDA 时为 100 passed、74 skipped。示例报告每步的三次求解迭代数、真实残量、压力完整方程残量、散度、净通量和动能。`status: passed` 表示此短程测试的求解和有限性检查通过，不是时间/空间收敛或完整心室算例验收。

## 三步方程与边界

使用 [FLUID](FLUID.md) 中的 M、K、D、G 和 C：

1. `(rho/dt M + mu K) u* = rho/dt M u_n - rho C(u_n,u_n) + b_f`。
2. `K_p p = -rho/dt D u*`。
3. `M u_(n+1) = M u* - dt/rho G p`。

压力在每一步重新求解，不累加到旧压力；旧压力只可作为迭代初值。速度在整个外边界上取给定 Dirichlet 值，默认零；时间变化时调用者传入新时间层的值。每步预测和校正都执行非零边界提升，保持自由自由度子系统对称。压力采用齐次 Neumann 条件和一个节点值基准，默认第 0 节点 p=0。

纯 Neumann 压力问题要求净边界流量兼容；程序在固定压力节点前检查 `sum(D u*)`，不满足时明确报错，不通过压力钉点隐藏质量源。当前不提供开放出口、周期边界或混合压力边界接口。

`density=f` 将 Q2 节点力密度变为 `b_f=M f`，对应 AFSI 的力密度路径；`nodal_load=b` 则直接进入右端。二者不能同时传入。该区分保留用于下一阶段 IB 耦合。

长度 cm、时间 s、速度 cm/s、压力 dyn/cm²、rho 为 g/cm³、mu 为 g/(cm·s)、力密度为 dyn/cm³。烟雾测试的 1 cm 盒子和给定体力用于数值验证，不是左室几何或生理标定。

## 求解器与性能边界

`solvers.py` 提供 Jacobi 预条件共轭梯度：各算子的真实装配对角线仅准备一次，矩阵向量乘由已有 PyTorch 单元积分执行，不生成全局稠密矩阵、不调用 CPU 线性求解。指定 CUDA 时向量、算子和预条件操作均在 GPU；收敛判断每轮读取少量标量，会发生主机同步。

停止阈值为 `max(atol, rtol*||b_free-A_free,fixed g||)`，默认 rtol=1e-10、atol=1e-12。每 40 轮及候选收敛时重新计算真实自由残量；重算后必要时重启搜索方向。NaN、非正曲率、超出迭代上限均抛出异常，不能视为成功。报告中的不同子步残量具有各自单位，不能仅按数值大小互相比较。

本版是 FP64 前向求解基线。迭代求解显式禁用 autograd，尚不支持对完整时间步求导；已有材料/算子的自动微分测试保持有效。

本地 CPU 三步烟雾测试中，每轴 2/4/8 个单元的压力迭代数分别为 7–9 / 24 / 42–44。Jacobi 提供初始预条件，尚无网格无关收敛保证；尚未实现多重网格，不能据此宣称大规模 GPU 性能已达标。细网格将依据实测迭代数和显存决定后续优化。

## 散度诊断不是零误差投影

本离散中的压力 Laplacian 与速度质量矩阵消元得到的 Schur 补不同，且校正后重新施加速度边界。因此 Chorin 分裂并非严格离散 Helmholtz 投影，不能保证 `D u_(n+1)=0`，也不能保证每步的点值散度 L2 都下降。

本地默认强制流例每轴 4 单元时，第一步散度 L2 从约 0.01270 降到 0.00915，第三步从约 0.01749 变成 0.01984。每轴 8 单元时第三步约从 0.01438 降到 0.00980。该现象明确保留在报告中，并通过独立 DOLFINx 同方程求解比较。这里只建立与参考算法一致的可运行基线；完整左室验收仍须进行时间步/网格收敛、质量/泄漏和长时间稳定性检查。不会为了得到更小散度而未说明地改用另一种压力投影。

## 独立参考验证

`export_chorin_dolfinx.py` 用实际 UFL/DOLFINx 装配、PETSc LU 直接求解，包含独立的边界提升和压力基准。8/64 单元网格、静止和匀速平移边界、每种情况连续 3 步，逐项比较预测速度、压力、校正速度，并比较积分散度，共 36 组解场及 12 组散度诊断。

```bash
conda run --no-capture-output -n afsi-reference python validation/export_chorin_dolfinx.py
conda run --no-capture-output -n afsi-reference python validation/export_chorin_dolfinx.py --subdivisions 2 --output validation/results/chorin_fine.npz
conda activate afsi-torch
CUDA_VISIBLE_DEVICES=0 python validation/compare_chorin_dolfinx.py --device cuda
CUDA_VISIBLE_DEVICES=0 python validation/compare_chorin_dolfinx.py --device cuda --reference validation/results/chorin_fine.npz
```

也可从本版 Actions 的 `dolfinx-reference` 附件提取两个 `chorin_*.npz`，放入 `validation/results/` 后直接比较。GPU 运行环境无需安装 FEniCSx/PETSc。

单位测试额外验证 PCG 与小型直接解的一致性、时变解析剪切流、压力基准不改变速度、非兼容边界流量拒绝、两种载荷接口、求解失败和 CPU/CUDA 对照。

方程依据：[所选 AFSI Chorin 示例](https://github.com/npuheart/afsi/blob/99df0ffba795fa05043ba874ad00353dcb986466/afsic/src/afsic/euler/ChorinSolver.py)。独立参考接口依据：[DOLFINx 0.10 LinearProblem 官方文档](https://docs.fenicsproject.org/dolfinx/v0.10.0.post0/python/generated/dolfinx.fem.petsc.html)。代码独立实现，未复制原求解器源码。
