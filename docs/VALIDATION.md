# 验证记录

## 0.11.0：时间步、网格与时长研究

本地完整回归 **120 passed、81 CUDA skipped**，共 201 项。新增验证检查相同物理时刻、单变量比较、节点编号一致性、用响应归一化误差、近零响应拒绝伪造收敛，以及失败时保存最后有效状态。

11 组本地 CPU 运行均完成，但时间/流体网格敏感性未通过 5% 诊断筛选。dt 最后两级的体积响应差约 5.60%，节点位移观测阶约 0.986；流体 8³/10³ 差异在 dt 减至 2.5e-5 s 后仍约 40.36%。到 0.01 s 时最小 det(F)≈0.98326，壁体体积变化约 −0.2851%。这些是启动过程结果，不能视为完整周期稳定性或收敛证明。详见 [STABILITY](STABILITY.md)。本版 GPU 仍待执行。

## 0.10.0：显式 IB/FEM 耦合

[Linux 自动验证 36027419539](https://github.com/loveIroha/AFSI_GPU/actions/runs/36027419539) 已全部通过，代码提交 `b6381ac9edb866233a82ce66269b2b93ca134b2b`。完整回归 **108 passed、80 CUDA skipped**。独立 UFL 固体 + NumPy 全格点 IB + PETSc LU 耦合参考连续推进 3 步，15 组数组全部通过；最大绝对差：位置 2.22e-16 cm、节点力 5.93e-15 dyn、速度 8.49e-13 cm/s、压力 2.97e-11 dyn/cm²、力密度 1.98e-15 dyn/cm³。默认生成左室 10 步、体积/位移输出也通过；[完整报告](coupled-reference-results-0.10.0.json)。本版目标 GPU 尚待执行。

本地 CPU 完整回归 **108 passed、80 CUDA skipped**，共 188 项。新增测试覆盖零力启动和载荷滞后、旧位置插值/新位置受力、跨格点移动重建支撑、小型非线性弹性体、失败不修改旧状态、CPU/CUDA 对照、生成左室短程推进及 ParaView 输出。默认左室 10 步运行通过，详见 [COUPLING](COUPLING.md)。本版尚待目标 GPU 验证，实际 DOLFINx/NumPy 耦合对照由 Linux CI 执行并单独记录。

## 0.9.0：Chorin 三步求解

[Linux 自动验证 36024695559](https://github.com/loveIroha/AFSI_GPU/actions/runs/36024695559) 已通过，代码提交 `21695e89619531f5be7f9de3693d4869525a0980`。完整回归 **100 passed、74 CUDA skipped**。实际 DOLFINx/PETSc LU 在 8/64 单元网格、静止/平移边界上连续推进 3 步，共 36 组解场和 12 组散度诊断比较通过；解场最大绝对差分别为 **9.86e-11 / 1.71e-10**。旧固体、IB、流体算子与几何回归也通过。[完整报告和参考数据 SHA-256](chorin-reference-results-0.9.0.json)。目标 GPU 本版尚待执行。

本地 Windows CPU **100 passed, 74 CUDA skipped**，共 174 项；新增 12 passed、7 skipped。验证非零边界提升、真实残量、压力基准、时变解析剪切、载荷接口、失败检测和短程流动。每轴 2/4/8 单元三步测试中压力迭代数约 7–9/24/42–44。Jacobi 尚无网格无关性能保证，粗网格校正后散度并非每步下降，详见 [CHORIN](CHORIN.md)。Linux 实际 DOLFINx/PETSc LU 比较单独记录；本版 GPU 待执行。

## 0.8.0：Q2/Q1 流体有限元算子

用户随后反馈目标 GPU 环境 **155 passed**，本版完整 pytest 验证通过；不代表完整心室耦合或外部参考 CUDA 对照已执行。

[Linux 自动验证 36021971483](https://github.com/loveIroha/AFSI_GPU/actions/runs/36021971483) 已全部通过，代码提交 `e8542cf37c63b693abaefff29a2f9f9da7bfe133`。Linux 完整回归 **88 passed、67 CUDA skipped**。新增真实 DOLFINx 流体对照在 2/16 单元网格、两种场上共比较 40 组向量；最大绝对差分别为 **2.43e-15 / 1.60e-15**。原固体 DOLFINx 对照、IB 对照和左室示例也通过。[完整流体误差报告及参考数据 SHA-256](fluid-reference-results-0.8.0.json)。本结果仅为 CPU；目标 GPU 本版尚待执行。

本地 Windows CPU 回归 **88 passed, 67 CUDA skipped**，共 155 项。新增流体测试为 14 passed、10 CUDA skipped。覆盖解析张量积质量/刚度矩阵、多单元组装、制造场、边界通量与负转置关系、非线性对流积分和切线、IB 两种载荷路径区别。CPU 制造场示例通过；目标 GPU 本版尚待执行。实际 DOLFINx 对照由 Linux Actions 执行并单独记录。

## 0.7.0：厘米制生成左心室

用户随后反馈目标 Linux RTX 4090 环境完整回归 **131 passed、无跳过，8.18 秒**。保留一次 PyTorch 内部 TorchScript 弃用警告。该反馈确认 pytest 的 CPU/CUDA 测试，不等于完整耦合算例或 GPU 上的 DOLFINx 外部参考对照。

[Linux 自动验证 36018932241](https://github.com/loveIroha/AFSI_GPU/actions/runs/36018932241) 已通过，代码提交 `a16b378afa75f360e8d6884a2acc51ef9c68d30c`。Linux 回归 74 passed、57 CUDA skipped；两个网格的实际 DOLFINx 对照、IB 独立对照、左室生成/固体力示例和预览均成功。Linux 默认腔体体积同为 82.75809228453295 mL，保守力/能量梯度最大差 3.27e-11 dyn。运行附件包含左室 VTU、NPZ、报告及预览。

本地 Windows CPU 完整回归 **74 passed, 57 skipped**，跳过项全部依赖 CUDA；安装 geometry 依赖且 GPU 可用时应执行 **131 项**。Gmsh 4.15.2，meshio 5.3.5。保留一次 PyTorch JVP 弃用警告。此处为本地记录；目标 GPU 后续反馈见本节开头。

默认 h=1.2 cm 的本地网格有 616 个四面体、1276 个 P2 节点，最小参考单元体积 0.0533734 cm³；内膜/外膜/基底面数为 149/250/39。解析腔体体积 87.2665 mL，离散值 82.7581 mL（几何误差约 5.17%）；h=0.9 cm 时为 84.4422 mL。给定变形下最小 det(F)=1.029996，保守节点力与能量梯度最大差约 2.00e-11 dyn。网格数量可能随平台变化，不作为跨平台断言。

标签、闭合壁体拓扑、三轴椭球、体积及其导数、心尖正则化、单位换算和 VTU 顺序测试通过。已生成并查看网格预览。示例不包含未知位移求解、流体或时间推进；详见 [几何说明](GEOMETRY.md)。

## 0.6.0：IB 速度插值和力散布

本地 Windows CPU 完整回归 **61 passed, 52 skipped**，跳过项全部依赖 CUDA；GPU 可用时应执行 **113 项**。新增 IB 测试为 13 passed、8 skipped。测试覆盖核矩条件、线性速度再现、合力/力矩、离散功率、转置与自动求导、非零原点、位置更新以及固体能量变化率。PyTorch 内部 JVP 的一次弃用警告仍保留。

| 检查 | 本地 CPU 结果 |
| --- | --- |
| 独立全格点 NumPy 参考：速度最大差 | 2.3592239273284576e-16 |
| 独立参考：散布力密度最大差 | 7.105427357601002e-15 |
| 完整固体力 + IB 示例：合力最大差 | 3.979039320256561e-12 |
| 示例：力矩最大差 | 2.219557870830613e-12 |
| 示例：功率绝对差 | 4.547473508864641e-13 |
| 示例：功率相对差 | 1.9134082443219367e-16 |

独立参考只验证相同核数学定义，未执行原 afsi C++ 二进制。瞬时格点功率恒等式不等于一致流体 FE 质量矩阵下的全耦合稳定性证明。本版尚未在目标 RTX 4090 上执行。Linux Actions 已加入新 IB 对照与示例，运行结果按对应提交记录。

后续 Linux CPU 自动验证已全部通过：[运行 35993548123](https://github.com/loveIroha/AFSI_GPU/actions/runs/35993548123)，代码/工作流提交 `74fb9d2658c84bd85326397a112de53102a12131`。完整回归 **61 passed, 52 skipped**；两种网格的实际 DOLFINx 固体力/切线对照、IB NumPy 对照和完整固体力 IB 示例均通过。Linux 独立参照速度最大差 2.78e-16、力密度最大差 7.11e-15；示例合力最大差 2.96e-12、力矩最大差 2.90e-12，功率差在本次浮点计算中为 0。

首次运行因云主机 UCX 自动选择不支持的 RDMA 设备，在 MPI 初始化阶段失败；将单进程参考任务限定为 TCP/self/shared-memory 后通过。未改动数值公式或放宽测试容差。

## 0.5.0：实际 DOLFINx 装配对照程序

本地 Windows CPU 回归：**48 passed, 44 skipped**；跳过项全部依赖 CUDA。语法编译检查通过。新增外部积分规则、全局自由度置换、坐标歧义拒绝和边界标记传递测试。CUDA 可用时应执行 **92 项**。

本地没有 DOLFINx/Linux 运行环境，以上回归不等于实际 DOLFINx 装配验证。随后在 GitHub Linux Actions **实际执行并通过**真实 UFL/DOLFINx 导出与两种网格的 CPU PyTorch 对照：[运行 35985614102](https://github.com/loveIroha/AFSI_GPU/actions/runs/35985614102)，代码提交 `1228f0d1185181dded4679538eef4f1a90ed5d4d`。

| 网格 | 节点 | 所有力项/构形中节点力最大绝对差 | 切线作用最大绝对差 |
| --- | --- | --- | --- |
| 6 个四面体 | 27 | 5.456968210637569e-10 | 1.418811734765768e-10 |
| 48 个四面体 | 125 | 3.7562131183221936e-10 | 1.375610736431554e-10 |

每种网格比较两种构形、五个力项及其方向导数，共 40 组向量比较通过。Linux 回归同为 **48 passed, 44 skipped**；跳过项全部依赖 CUDA。环境：Ubuntu 24.04、DOLFINx 0.10.0、Basix 0.10.0、UFL 2025.2.1、PyTorch 2.14.0+cpu、NumPy 2.5.3。[完整误差报告](reference-results-0.5.0.json) 从该运行日志提取，包含参考数据 SHA-256 和版本。本版目标 GPU 对照仍需用户运行。[运行说明](DOLFINX.md)。

## 0.4.0：随动压力与基底弹簧

本地环境为 Windows CPU，版本与下述基线一致。完整 pytest 为 **39 passed, 42 skipped**，跳过项全部依赖 CUDA；目标 GPU 可用时应执行 **81 项**。本版尚未在目标 Linux GPU 上验证。

| 检查 | 结果 |
| --- | --- |
| 可编辑安装 | afsi-torch 0.4.0 成功 |
| 完整固体节点力示例 | 1 个 P2 四面体，10 节点，4 个外表面 |
| 基底弹簧解析力/能量梯度最大差 | 1.7763568394002505e-15 |
| 完整残量切线作用/有限差分相对误差 | 1.1929863471576661e-10 |
| Basix 体单元余子式参照：压力节点力最大差 | 2.2737367544323206e-13 |
| 独立参照：基底弹簧节点力最大差 | 9.237055564881302e-14 |
| 独立参照：基底弹簧能量差 | 1.2434497875801753e-14 |

```bash
python -m pytest -q
python examples/boundary_patch.py --device cpu
python validation/compare_boundary.py
```

新增测试覆盖法向与内部面剔除、平面/曲面压力、闭合面的合力和力矩、开放受压面的非对称切线、参考面积弹簧，以及材料与边界项组合后的残量切线。独立对照使用 Basix 的完整四面体基函数导数构造 F 和 cof(F)N，与主实现的表面切向量叉积比较。它尚不等于实际 DOLFINx 全局装配对照。PyTorch 内部 JVP 路径仍产生一次 TorchScript 弃用警告。

## 0.3.0：Guccione 与给定主动张力

本地环境仍为下述 Windows CPU 环境。完整 pytest 为 **25 passed, 27 skipped**，所有 skipped 均为 CUDA 测试；没有将跳过项记为通过。GPU 可用时应执行 **52 项**。

| 检查 | 结果 |
| --- | --- |
| 新增材料/场测试 | 9 passed, 9 skipped |
| 原 P1/P2 回归 | 16 passed, 18 skipped |
| 可编辑安装 | afsi-torch 0.3.0 成功 |
| Guccione 示例固定张力势函数 | 3232.1858758290914 |
| 示例最小 det(F) | 1.0156138610542116 |
| 示例解析组装力/自动求导力最大差 | 2.1827872842550278e-11 |
| 示例合力范数 | 1.942149220127487e-11 |
| 示例切线作用/有限差分相对误差 | 9.358497294235228e-11 |
| Basix + NumPy 复步长 PK1 最大差 | 2.35741026699543e-09 |
| 独立参照总势函数差 | 1.0459189070388675e-11 |
| 独立参照组装力最大差 | 1.382431946694851e-10 |
| 原 Neo-Hookean Basix 对照 | 通过 |

```bash
python -m pytest -q
python examples/guccione_patch.py --device cpu
python validation/compare_guccione.py
python validation/compare_basix.py
```

本轮未在 Linux GPU 上执行，也没有声称完成完整 DOLFINx/afsi 对照。一次 TorchScript 弃用警告仍来自 PyTorch 的 JVP 路径。

后续用户反馈：Linux RTX 4090 上 0.3.0 的 **52 项测试全部通过**，对应提交 `b85202f`。这是用户在目标机器运行和反馈的结果。

## 0.2.0：P2 基线

环境：Windows 11；Python 3.12.14；torch 2.14.0+cpu；numpy 2.5.3；pytest 9.1.1；可选 fenics-basix 0.10.0。

| 检查 | 结果 |
| --- | --- |
| 完整 pytest | 16 passed, 18 skipped；跳过项全部为 CUDA 相关 |
| 可编辑包安装 | afsi-torch 0.2.0 安装成功 |
| 两单元 P2 示例 | 2 单元，14 节点，每单元 64 个积分点 |
| 示例总能量 | 0.0698944071186671 |
| 示例积分点最小 det(F) | 1.0313891317561248 |
| 示例自动求导/解析组装力最大差 | 1.1102230246251565e-16 |
| 示例合力范数 | 1.316562550716263e-16 |
| 示例能量方向导数/有限差分误差 | 1.4469980769149515e-10 |
| Basix 形函数最大差 | 5.828670879282072e-16 |
| Basix 形函数导数最大差 | 1.5543122344752192e-15 |
| Basix/NumPy 总能量差 | 2.7755575615628914e-17 |
| Basix/NumPy 组装力最大差 | 1.0547118733938987e-15 |

运行命令：

```bash
python -m pytest -q
python examples/p2_patch.py --device cpu
python validation/compare_basix.py
```

存在一条 PyTorch 内部 TorchScript 弃用警告，来自 torch.func JVP 路径；测试通过，未屏蔽该警告。

后续用户反馈：Linux RTX 4090 上 0.2.0 的 **34 项测试全部通过**，并完成首次推送，基线提交 `82dadab`。该 GPU 结果由用户在目标机器运行和反馈；没有测量性能或双卡并行。
