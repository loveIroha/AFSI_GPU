# 真实左心室：P1 H–O 固体 + MAC/IB，三个周期

读取外部真实左心室网格、边界和 DG0 fiber/sheet，采用用户提供的 H–O UFL、
底部径向约束及 0.8 s 周期压力/主动张力。默认总时长 **2.4 s，24,000 步**。
本目录不分发真实网格或材料方向文件；它们保留在用户的数据目录。

## 默认设置

默认时间推进为兼容既有结果的 `explicit-lagged`；
`--coupling implicit-newton` 启用新时刻强耦合 Newton，见下述运行方法。

| 项目 | 设置 |
| --- | --- |
| 长度、时间、应力 | cm、s、dyn/cm² |
| 输入路径 | `/mnt/large2/gjh/realistic_left_ventricle` |
| 固体 | 原始 P1 四面体；本次参考网格 26,889 节点、135,430 单元 |
| 材料方向 | 原始 DG0 fiber/sheet；不进行节点平均或归一化 |
| 固体体积分/底部面积分 | degree=5 正权重规则 |
| IB 相互作用积分 | 默认 degree=2、4 点/四面体；541,720 相互作用点 |
| 质量矩阵 | 一致 P1 质量矩阵，一次组装 CSR；不使用集中质量 |
| 流体 | MAC + 几何多重网格；128³ 压力单元 |
| 流体盒 | `[0,15]³ cm`；`h=0.1171875 cm` |
| 时间步 | `dt=1e-4 s` |
| 周期 | `0.8 s`，重复 3 次 |
| 流体密度/黏度 | 沿用当前框架默认 `rho=1 g/cm³`、`mu=1 g/(cm s)`，可修改 |
| 体积惩罚/底部系数 | `kappa=5e6 dyn/cm²`；`beta=5e6 dyn/cm³` |
| 底部中心 | `(7.5,7.5) cm`，与所给 UFL 一致 |
| 输出 | 日志每 100 步；检查点每 1000 步；VTK 每 200 步，即 0.02 s |

源边界已确认为 **1=心外膜、2=心内膜、3=底部**。
输入时显式映射为内部 1=内膜、2=外膜、3=底部。
UFL 的 4096/4097/4098 是其原始标识，在本 demo 中对应这些相同的物理表面；
不要求把输入 XML 的数值标签改成 4096 等。

## 安装与输入

在 AFSI_GPU 仓库根目录运行：

```bash
git pull --ff-only
conda activate afsi-torch
python -m pip install -e ".[test,mesh,fused]"
```

保持以下文件的相对路径：

```text
realistic_left_ventricle/
  mesh_scale.xdmf
  mesh_scale.h5
  boundaries.xml
  fibers_0.xml
  fibers_1.xml
  fibers_2.xml
  sheets_0.xml
  sheets_1.xml
  sheets_2.xml
```

文件名或单位不同时，可以修改主程序顶部的 `CONFIG` 或 JSON 配置。
本 demo 使用 P1，**不要先调用 `to_p2()`，也不读取导入检查的 VTK 文件作为源网格**。
读取接口直接保留原始 26,889 个顶点。

## GPU 测试与短时检查

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/test_real_lv.py tests/test_mesh_io.py \
  tests/test_mac_output.py tests/test_mac_lv_graph.py

CUDA_VISIBLE_DEVICES=0 python -u demo/real_lv_fsi/run_mac.py \
  --mesh-dir /mnt/large2/gjh/realistic_left_ventricle \
  --device cuda --end-time 0.005 \
  --output results/real_lv_smoke
```

短时检查采用与完整实验相同的 128³ 网格、材料、时间步和执行后端，推进 50 步。
首次使用会编译 H–O/流体张量核并捕获压力、质量迭代 CUDA Graph。
测试包含参考/优化路径对照，另从收缩期 0.6 s 的已知加载状态检查主动应力路径，
因此测试不只覆盖初始零载荷。

此实现已在小网格上检验本构导数、边界力、一致质量、功率配对、保存/续算，
并在完整真实固体网格上完成 16³ 流体、3 步 CPU 启动检查。
**128³ GPU 的 2.4 s 完整轨迹尚待实际运行验证。**

## 后台运行三个周期

短时检查通过后，在新输出目录进行完整实验：

```bash
run_dir="results/real_lv_3cycles"
mkdir -p "$run_dir"

CUDA_VISIBLE_DEVICES=0 nohup /usr/bin/time \
  -f 'elapsed_seconds=%e exit_code=%x' -o "$run_dir/runtime.txt" \
  python -u demo/real_lv_fsi/run_mac.py \
  --mesh-dir /mnt/large2/gjh/realistic_left_ventricle \
  --device cuda --cycles 3 --dt 1e-4 --fluid-cells 128 \
  --output "$run_dir" \
  > "$run_dir/run.log" 2>&1 < /dev/null &

echo "PID=$! output=$run_dir"
```

终端退出后程序继续运行。监测：

```bash
tail -f results/real_lv_3cycles/run.log
```

完整实验完成后查看 `report.json` 和 `runtime.txt`。
同一输出目录已有报告、历史或检查点时，新运行会拒绝覆盖。

也可以从短时检查直接续算，避免重新生成初始文件：

```bash
CUDA_VISIBLE_DEVICES=0 python -u demo/real_lv_fsi/run_mac.py \
  --device cuda --resume results/real_lv_smoke/checkpoint.npz --cycles 3
```

续算使用原目录，恢复 H–O 参数、基底中心、DG0 方向、原始网格、流体设置和时间步。
检查点自包含，不依赖再次读取原始数据文件。
普通续算仅允许更改设备与目标时长；更改物理参数或执行设置需要新运行。
时间步减小使用下述显式 `--resume-dt` 接口，保存到新目录。

## 对流检查停止时

本节的 CFL/D/A 停止条件适用于 `explicit-lagged` 路径。
隐式 Newton 路径的对流、黏性和固体力在新时刻共同求解，CFL/A 作为监测量。
方法公式、CSR 切线、几何冻结范围与性能代价见 [隐式耦合说明](../../docs/MAC_IMPLICIT.md)。

### 使用新时刻强耦合 Newton

显式滞后力可能引入额外弹性稳定性限制和载荷相位误差；
反复触发 A 并不单独证明这些误差已发生。新路径使用后向 Euler，
将流体对流、黏性、当前位置的 H–O/主动力、随动压力和基底力同时纳入 Newton。
压力通过多重网格消元，固体切线实际组装为 CSR。
IB 几何在每步开始的位置冻结；插值和传播始终采用同一套算子。

质量矩阵和压力采用迭代求解，其绝对容差在很小的 Newton 修正量上可能成为误差底限。
隐式 Jacobian 作用现在先按单位输入范数计算再恢复尺度，零向量严格返回零。
默认 `coupling.newton.linear_tolerance_fraction=0.2` 协调内外层容差：
外层目标为 `1e-9` 时，GMRES 绝对容差下限为 `2e-10`。
GMRES 真实残差、线搜索及最终完整耦合残差检查全部保留，最终外层目标没有放宽。
真实左心室旧 JSON/检查点中未保存此字段时继承当前默认值 0.2；显式保存的值会保留。
修改该字段可通过新运行的 JSON 配置完成。

隐式终态直接保留 Newton 求得的速度，在同一状态恢复压力、位移和力并复核残差。
不再在 Newton 结束后额外执行一次 `u <- G(u)` 固定点更新；刚性问题中该更新
可能放大已达标的残差。动量与实际散度检查仍然执行。
复核失败时，`report.json` 的 `failure.coupled_acceptance` 保存 Newton 返回残差、
重新计算的残差、目标容差、散度及压力/质量矩阵求解残差，供区分收敛与精度问题。

先验证 GPU 路径，并从最新失败状态继续到 0.532 s，检查新方法的实际收敛和耗时。
此例假定最新状态位于 `real_lv_3cycles_dt25us` 且早于 0.532 s。
源状态晚于该时刻时，选择更晚的 `--end-time`，并保持它为 dt 的整数倍。
首次执行会编译新的切线/动量核；总耗时包括编译和 CSR 模式准备。

```bash
git pull --ff-only
conda activate afsi-torch
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_mac_implicit.py

source_checkpoint="results/real_lv_3cycles_dt25us/checkpoint.npz"
run_dir="results/real_lv_implicit_newton"
mkdir -p "$run_dir"

CUDA_VISIBLE_DEVICES=0 nohup /usr/bin/time \
  -f 'elapsed_seconds=%e exit_code=%x' -o "$run_dir/runtime_smoke.txt" \
  python -u demo/real_lv_fsi/run_mac.py \
  --device cuda --resume "$source_checkpoint" --coupling implicit-newton \
  --end-time 0.532 --output "$run_dir" \
  > "$run_dir/run_smoke.log" 2>&1 < /dev/null &

echo "PID=$! output=$run_dir"
```

该命令保留原检查点的 dt=2.5e-5 与 x/u/p，将力时间改为当前状态时间，
不重复已接受的时间段，不改写原结果；这是一条从显式历史转入隐式方法的混合轨迹。
已有显式历史的误差不会因切换而消失。方法切换必须使用新目录。
初始力与每一步接受状态均按新配置保存，下一次普通续算自动恢复隐式路径。

```bash
tail -f results/real_lv_implicit_newton/run_smoke.log
```

日志与 CSV 新增 Newton 迭代次数、真实残差和容差。
完整检查点报告也包含 Newton/GMRES 历史。失败时保存上一个接受状态，
Newton 不收敛不能被标记为成功时间步。每步多次质量/压力求解会增加耗时，
全尺寸 GPU 效率与完整三个周期尚待实际运行验证。

检查收敛后，可在同一目录直接继续到 2.4 s：

```bash
run_dir="results/real_lv_implicit_newton"
CUDA_VISIBLE_DEVICES=0 nohup /usr/bin/time \
  -f 'elapsed_seconds=%e exit_code=%x' -o "$run_dir/runtime_3cycles.txt" \
  python -u demo/real_lv_fsi/run_mac.py \
  --device cuda --resume "$run_dir/checkpoint.npz" --cycles 3 \
  > "$run_dir/run_3cycles.log" 2>&1 < /dev/null &
echo "PID=$!"
```

从零开始使用隐式路径时：

```bash
CUDA_VISIBLE_DEVICES=0 python -u demo/real_lv_fsi/run_mac.py \
  --coupling implicit-newton --dt 1e-4 --cycles 3 \
  --output results/real_lv_implicit_from_zero
```

隐式方法也有时间精度、空间分辨率和非线性收敛要求；不能凭通过旧 A 阈值
或某段 Newton 收敛宣称全部三个周期稳定。

MAC 中心对流采用与显式黏性项联合的时间步检查，令 `U_i=max|u_i|`、`nu=mu/rho`：

- `CFL = dt*sum(U_i/h_i) <= 0.25`。
- `D = nu*dt*sum(1/h_i²) <= 0.25`，保留原来的黏性时间步限制。
- `A = dt*sum(U_i²)/(2*nu) <= 0.25`，新增对流–扩散联合时间尺度检查。

原来的 `cell_Re<=1` 停止条件已移除；cell_Re 继续记录为网格分辨率指标。
这不是简单地将其上限改为更大的常数。减小 dt 会同时降低 CFL 和 A，
但不会改变同一速度场的 cell_Re。大 cell_Re 仍可能有空间振荡和分辨率问题。
公式依据、安全裕量与适用范围见 [MAC transport](../../docs/MAC_TRANSPORT.md)。
这些检查不是完整的非线性 FSI 稳定性证明；小散度也不证明力学耦合稳定。

发生此异常时，程序保存最后接受状态到原运行目录的 `checkpoint.npz`，
并写出 `report.json` 的失败信息。错误输出分别记录 CFL、A、cell_Re 和触发项，
失败报告同时保存 `failure.transport_guard`；旧版失败检查点也可以直接分析：

```bash
python validation/diagnose_real_lv_guard.py \
  --checkpoint results/real_lv_3cycles/checkpoint.npz \
  > results/real_lv_3cycles/guard_diagnosis.json
```

请使用实际失败的运行目录。此命令只在 CPU 读取保存的速度与配置，
不导入原始网格、不编译 GPU 内核、不推进或改写检查点；它不做完整校验和验证。
输出包括 CFL、A、cell_Re、速度峰值及其交错网格坐标，
分别显示当前策略的 `triggered` 与旧阈值的 `legacy_cell_re_gt_one`。
新版本可继续旧版本因 cell_Re 略大于 1 而停止的检查点；其他检查仍然有效。

### 减小时间步，从失败状态继续三个周期

若当前检查点触发 A 或 CFL，可从最后接受状态直接用更小的固定时间步继续。
例如 `dt=1e-4` 时 A=0.25028，保持相同速度改用 `dt=5e-5` 会将 A 减半到约 0.12514。
0.25 是当前检查的安全裕量，并非非线性 FSI 的精确失稳边界；略超限不能单独证明解发散。
此接口保持检查阈值、中心对流、材料、黏度和所有耦合离散不变。
后续加载使速度继续增大时，仍可能再次触发检查。

在仓库根目录运行以下命令。`source_checkpoint` 应指向实际失败的运行目录，
不要使用更早时刻的 `half_dt` 检查点替代最近状态：

```bash
git pull --ff-only
conda activate afsi-torch

source_checkpoint="results/real_lv_transport_recovery/original_dt/checkpoint.npz"
run_dir="results/real_lv_3cycles_dt5e5"
mkdir -p "$run_dir"

CUDA_VISIBLE_DEVICES=0 nohup /usr/bin/time \
  -f 'elapsed_seconds=%e exit_code=%x' -o "$run_dir/runtime.txt" \
  python -u demo/real_lv_fsi/run_mac.py \
  --device cuda --resume "$source_checkpoint" --resume-dt 5e-5 \
  --cycles 3 --output "$run_dir" \
  > "$run_dir/run.log" 2>&1 < /dev/null &

echo "PID=$! output=$run_dir"
```

`--cycles 3` 表示总目标时刻 2.4 s，而非从检查点额外运行 2.4 s。
总步编号相应变为 48,000，当前步编号也按原时间重新计算，已接受的时间段不会重跑。
接口保留当前固体位置、流体速度和压力；按既有滞后力时序在 `current_time-new_dt`
重算初始固体力。仅支持原 dt 的整数细分，如减半、三等分。
日志、VTK 和检查点的步数间隔同时乘细分倍数，保持其物理时间间隔不变。

减小 dt 必须指定新输出目录；原报告、CSV、VTK 和检查点均保留。
新目录的 CSV/VTK 从续算时刻开始，`restart_from` 记录来源、前次失败、时间步和重编号信息。
新目录的 `elapsed_seconds` 只统计本次分支及其后续普通续算，前次耗时单独保存在
`restart_from.source_elapsed_seconds`。保存的新检查点可再次普通续算，原始网格文件无需重新读取。
`run.log`、`runtime.txt` 等后台启动文件允许预先存在，但已有计算结果的目录不能覆盖。

```bash
tail -f results/real_lv_3cycles_dt5e5/run.log
```

接口测试可独立运行；无需重复已经完成的时间步对照：

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_real_lv.py
```

也可做局部时间步对照。以下是从早于 0.05 s 的共同状态继续两个分支的示例；
检查点晚于 0.05 s 时，须设置更晚的目标时刻：

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/test_real_lv_guard.py tests/test_real_lv.py tests/test_mac_execution.py

CUDA_VISIBLE_DEVICES=0 python -u validation/compare_real_lv_transport.py \
  --checkpoint results/real_lv_3cycles/checkpoint.npz \
  --device cuda --end-time 0.05 \
  --output results/real_lv_transport_recovery
```

此命令不改写输入检查点，不重跑已接受的 282 步。
`original_dt/` 和 `half_dt/` 保存各自的报告、CSV、检查点和 VTK，
顶层 `report.json` 保存是否到达目标时刻以及最终坐标、速度、腔体积差异。
两分支保持相同的初始坐标、流速和压力；减半时间步分支重新编号时间步，
并按现有滞后力时序在 `start_time-half_dt` 重新采样初始力。
因此这是从共同状态继续的局部时间步对照，不是整段历史的时间收敛证明。
若某分支失败，仍会尝试另一分支并记录失败状态；命令最终返回非零退出码。
输出目录必须为空；需要重跑对照时使用新的目录。

新的 CSV 记录状态时刻的 CFL、A、cell_Re。
`power_error` 在未采样时写为空白、JSON 中为 null，并用 `power_error_sampled` 标记；
续算时保留旧 CSV 的原有数值。短时对照通过后再验证收缩阶段与完整周期。

## 主程序与 JSON 参数接口

`run_mac.py` 顶部的 `CONFIG` 显式列出材料、标签、流体域、时间步和输出。
运行优先级是 CONFIG → JSON → 显式命令行参数。

```bash
python demo/real_lv_fsi/run_mac.py --write-config real_lv.json
```

编辑 `real_lv.json` 后：

```bash
CUDA_VISIBLE_DEVICES=0 python demo/real_lv_fsi/run_mac.py \
  --config real_lv.json --output results/real_lv_custom
```

常用覆盖包括 `--mesh-dir`、`--dt`、`--fluid-cells`、`--fluid-lengths`、
`--fluid-origin`、`--rho`、`--mu`、`--kappa`、`--beta`、`--cycles`、
输出间隔和 `--no-vtk`。`--reference` 使用 torch/PCG 参考路径，
适用于小网格验证。自定义文件名、积分阶数和求解器容差通过 CONFIG/JSON 设置。
修改周期时长需要新的分段载荷函数；本次按用户原式固定 0.8 s。

## 与用户 UFL 的对应

采用 `x` 存储当前坐标；`u=x-X_ref` 是位移。
因此 `grad(x)` 恰好对应用户的 `I+grad(u)`，不会重复加单位阵。

- `I1_bar = J^(-2/3)*tr(C)`，只修正 I1。
- 纤维、片层使用未修正的 `C`，分别将 `I4f/I4s` 截断到不小于 1。
- 纤维/片层耦合项使用 `I8fs²`。
- 原式体积应力 `kappa*ln(det(C))*F^(-T)` 对应能量 `kappa*(ln J)²`；
  没有替换成先前 Guccione 的 `(J-1)²` 惩罚。
- 主动应力保持 `T*(1+4.9*(sqrt(I4f)-1))*F*(f⊗f)`，没有额外截断该伸长因子。
- 不启用 UFL 中注释掉的 PN 消除项。
- 压力力为 `-p*cof(F)*N_ref`；只加载真实心内膜，没有额外阀口封盖压力。
- 底部约束保持原式：xy 中允许参考径向移动，惩罚切向位移；惩罚 z 位移。
  使用参考面积积分，并沿用 UFL 的力符号。
- 外膜为自然边界；没有额外添加外膜弹簧。

P1 位移、DG0 方向使 F 和材料应力在每个单元内为常量。
固体力仍采用 degree=5 积分权重，但只对每个单元计算一次 H–O 应力后求和，
避免在所有积分点重复计算相同的 3×3 代数。
IB 的核积分不是这个材料积分，单独使用可配置的相互作用积分规则。
默认 4 点规则对 P1 一致质量矩阵精确；它不证明正则化 IB 核已经积分收敛。

UFL 中 `inner(U,V)*dx` 对应质量矩阵。
在本项目中，固体弱形式先返回积分后的节点力，IB 传播再求解 `M F=b` 得到力密度系数，
因此不把质量项重复加入节点力，也不使用对角集中质量替代它。

周期压力和张力逐段照搬用户的 C++ 表达式，并只做一次 kPa→CGS 转换。
压力在每个周期末由约 1.067 kPa 重置到 0：这个跳变来自给定表达式，未自行平滑。
耦合沿用当前 AFSI 风格的滞后力时序；CSV 的 `force_time_s` 记录实际力采样时间，
载荷列则记录当前状态时刻的规定载荷。

固体采用有限体积惩罚，没有引入多孔介质压力/渗流未知量。
`explicit-lagged` 沿用原滞后力时序；`implicit-newton` 使用上述后向 Euler 强耦合求解。
三个规定载荷周期不等同于已达到周期稳态。

## 输出与代码位置

- `configuration.json`：实际配置。
- `history.csv`：压力/张力、腔体积、J 范围、底部约束误差和流体诊断。
- `report.json`：运行状态、离散/材料/边界说明和最后一个接受状态。
- `checkpoint.npz`：自包含 P1/DG0/H–O 检查点。
- `vtk/solid.pvd`：变形后的 P1 固体、位移、节点力、逐单元 fiber/sheet、J。
- `vtk/fluid.pvd`：流体压力、显示速度和散度。

腔体积用真实心内膜加一个平均开口顶点的虚拟三角扇封口计算。
封口只用于测量，不参与力学加载；非平面开口的体积依赖这个明确约定。

实现位置：`real_lv.py`（配置、固体和编译执行）、`holzapfel_ogden.py`（材料/载荷）、
`p1.py`（积分和形函数）、`simulation/real_lv_mac.py`（运行）、
`real_lv_checkpoint.py`（续算）。流体、IB 和 CSR/Graph 质量求解复用共享 `mac/` 模块。
