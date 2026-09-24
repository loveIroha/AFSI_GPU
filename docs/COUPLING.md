# 显式 IB/FEM 耦合与生成左室短程运行（0.10.0）

本版首次把非线性固体节点力、IB 力密度散布、Chorin 流体求解、速度插值和固体坐标更新连接成完整时间步。左室入口直接生成厘米制椭球网格和纤维，位移来自计算得到的速度，不再人为指定。它仍是短程数值验证，不是已验收的生理心动周期。

## 运行

沿用 0.7.0 以后安装的环境，无新增依赖：

```bash
git pull --ff-only
conda activate afsi-torch
python -m pip install -e ".[test,geometry]"
CUDA_VISIBLE_DEVICES=0 python examples/coupled_lv.py --device cuda --steps 10 --dt 1e-4 --output results/coupled_lv
CUDA_VISIBLE_DEVICES=0 python -m pytest -q
```

本版完整 CPU/CUDA 测试为 188 项；本地无 CUDA 时为 108 passed、80 skipped。CUDA 不可用会报错，不静默回退。Gmsh 网格生成、拓扑预处理和结果输出在 CPU；固体力、IB 插值/散布、流体迭代线性代数及坐标更新使用所选 device。迭代停止与数值检查读取少量标量，会发生主机同步。

当前 IB 使用 PyTorch CUDA 张量算子。后续允许根据实测瓶颈引入自定义 CUDA/C++ 扩展，但须保留参考路径，并重新通过核函数、单位缩放、合力/功率、时间推进和 CPU/GPU 对照；本版未新增编译器或 CUDA Toolkit 依赖。

## 与原示例一致的时间顺序

源算例先初始化流体力为零，再进入以下循环。为避免改变启动过程与载荷采样时刻，本版显式保留这一滞后：

1. 在 x_n 重建 H_n，以已保存的 g_n 散布 `f_n=H_nᵀg_n/dV_lattice`。
2. 用 `M f_n` 作为体力右端，三步求解 u_(n+1)、p_(n+1)。
3. 在**旧位置**插值 `U_s=H_n u_(n+1)`，显式更新 `x_(n+1)=x_n+dt U_s`。
4. 检查新构形和 IB 支撑，在**新位置**计算 `g_(n+1)=g(x_(n+1),t_n)`，供下一轮使用。

这里固体坐标时刻是 t_(n+1)，而新力的压力/主动张力参数采样时刻是 t_n，与选定源示例的循环一致。状态保存 `force_time`，报告同时标明已用力与下一轮力的时刻。初始 g_0=0；零初速时第一步不运动，零起点载荷斜坡还会带来额外启动滞后。没有悄悄改成预加载初力或另一种半隐式耦合。

`ExplicitIBStepper` 不改变输入状态张量；失败时不返回新状态。每步检查积分点 det(F)>0、表面正则性（左室模型）、腔体正体积、完整 IB 支撑和有限数据。单步固体位移默认不得超过每轴 IB 格距的 0.25 倍；超过则要求减小 dt，不裁剪位移。该限值只是异常步长保护，不是稳定性证明，也不保证单元全域无翻转。

## 载荷、单位与功率

默认左室仍采用 Guccione C=20000、kappa=500000 dyn/cm²，基底参考面积弹簧 beta=500000 dyn/cm³，所有三个位移分量受弹簧约束。载荷在 0.01 s 内线性升到 8 mmHg 和 1000 dyn/cm² 的给定主动张力；这些只为短程启动验证设定，未经生理标定。规定内膜压力是固体表面载荷，流体求出的压力通过流体速度影响固体，不再被重复加到内膜载荷中。虚拟腔体封口只用于测量，不施加压力。

背景盒长宽高各 12 cm，原点 (-6,-6,-8) cm，默认每轴 6 个 Q2 单元，IB 速度格距 1 cm。盒子保留了完整四点核支撑空间；不能把原示例的较小背景盒直接用于当前 7 cm 外径的左室。流体 rho=1 g/cm³、mu=1 g/(cm·s)，整个外边界零速度、单点压力基准；尚无出口/瓣膜/循环系统。固体和腔体均浸入背景流体，没有额外求解一套固体质量矩阵或加入重复惯性。

本版明确采用原密度路径 **M(Hᵀg/dV_lattice)**。不把它替换为直接对偶载荷 Hᵀg。报告区分：

- 固体节点功率 `g·H u`。
- 格点功率 `dV_lattice f·u`，应与上式在舍入误差内一致。
- 一致有限元体力功率 `u·M f`，一般与上两者不同，差值如实记录。

最后一项差值不是求解失败，也不能被忽略或称作能量守恒；本版不声称原密度路径具有严格的离散总能量守恒。报告的被动能量和弹簧能量不包括给定主动张力或开口压力的所谓储能。

## 默认运行结果与输出

本地 CPU 默认 10 步、dt=1e-4 s，共 0.001 s：腔体体积从 82.7580923 变为约 82.7589427 mL，最大总位移约 1.51e-5 cm，最终最小 det(F) 约 0.9999779。这是很短的小载荷响应，不应解释为心室收缩曲线或已达到平衡。

输出包括：

- `solid.pvd`、`fluid.pvd`：在 ParaView 中打开时间序列。
- `solid_*.vtu`：真实当前位置、位移和供下一步使用的节点力。
- `fluid_*.vtu`：速度节点值及 Q1 压力插值；为显示把 Q2 单元分成 8 个线性六面体，渲染不等同于高阶单元内部精确场。
- `history.csv`：体积、det(F)、位移和能量随时间变化。
- `report.json`：参数、每步线性残量、散度、载荷时刻、功率与合力诊断。
- `final_state.npz`：最终数组归档；本版尚未提供完整重启加载接口。
- `geometry/`：程序生成的参考网格、标签和纤维。

流体 Chorin 分裂误差仍存在，校正后的积分散度不保证每步下降；参见 [CHORIN](CHORIN.md)。一次短程运行通过，不代表大时间步、长周期或网格加密后必然稳定。

## 独立耦合参考

[本版 Linux CPU 验证已通过](https://github.com/loveIroha/AFSI_GPU/actions/runs/36027419539)：15 组数组全部通过，位置最大差 2.22e-16 cm、速度 8.49e-13 cm/s、压力 2.97e-11 dyn/cm²。完整回归 108 passed、80 CUDA skipped。目标 CUDA 验证仍需下述命令。

`export_coupled_dolfinx.py` 在独立 CPU 环境执行小型弹性体：UFL 装配 Neo-Hookean 被动力、给定主动张力、压力及基底项；纯 NumPy 逐格点计算 Peskin 核；DOLFINx/PETSc LU 求解流体；按同一滞后顺序推进三步。参考不导入 PyTorch，不复用生产 IB 索引/散布，也不执行原 afsi C++ 二进制。它验证整条耦合链；左室 Guccione 单项由已有独立固体对照覆盖。

```bash
conda run --no-capture-output -n afsi-reference python validation/export_coupled_dolfinx.py
conda activate afsi-torch
CUDA_VISIBLE_DEVICES=0 python validation/compare_coupled_dolfinx.py --device cuda
```

也可从本版 Actions 的 `dolfinx-reference` 附件下载 `coupled_patch.npz` 放入 `validation/results/`。每步比较位置、固体力、流体速度、压力和所用力密度，共 15 组数组。

剩余验收工作包括目标 GPU 回归、时间步与网格收敛、长程稳定性、质量/泄漏诊断、载荷时程与边界设定、性能评估及重启。当前不把这些标为完成。

顺序依据：[AFSI demo_337 主循环](https://github.com/npuheart/afsi/blob/99df0ffba795fa05043ba874ad00353dcb986466/afsic/demo/demo_337/fsi_paralell_fibers_contraction.py)。代码独立实现，没有复制原 C++ 或求解器源码。
