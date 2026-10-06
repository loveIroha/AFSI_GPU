# 真实左心室：Ma 等（2024）的被动充盈算例

本 demo 现以 *An unconditionally stable scheme for the immersed boundary
method with application in cardiac mechanics* 的 V.F 节为复现目标。
[论文信息及作者接受稿](https://eprints.gla.ac.uk/333577/)，DOI: 10.1063/5.0225605。
默认是被动充盈；`active_cycle.json` 在同一 BE–BE 框架上加入用户指定的
周期压力和主动收缩，作为单独的拓展算例，不属于论文 V.F 被动算例的复现。
输入真实网格和 fiber/sheet 属于用户数据，不随仓库分发。

## 对齐设置

| 项目 | 当前设置 |
| --- | --- |
| 固体 | 导入四面体网格，P1 位移，原样保留 DG0 fiber/sheet |
| 网格标签 | 源标签 1=外膜、2=内膜、3=底部 |
| 本构 | 论文式 (81)–(82)，原始 I1=tr(C)，含应力修正及 log(I3) 惩罚 |
| 参数（dyn/cm²） | a=2362，af=200370，as=37245，afs=4108，体积惩罚=5e6 |
| 指数参数 | b=10.81，bf=14.154，bs=5.1645，bfs=11.3 |
| 流体 | MAC 128³，13×13×13 cm³，rho=1 g/cm³，mu=0.04 g/(cm·s) |
| 外部流体边界 | 黏性速度求解采用零 Neumann；压力为零 Dirichlet |
| 压力加载 | 据图 24 重建：0–0.8 s 从零线性增至 8 mmHg，此后保持 |
| 主动收缩 | 无 |
| 时间推进 | BE–BE：新位置固体力、旧位置 IB 模板，BE 黏性＋投影 |
| 对流 | 半拉格朗日：一阶回溯，三线性 MAC 插值 |
| 模拟时间 | 1.5 s，固定 dt=1e-4，15,000 步 |
| 非线性默认求解器 | JFNK／无预条件 BiCGSTAB，带线搜索 |

底部**保留用户之前指定的径向弹簧约束**：中心 x=y=7.5 cm，允许径向运动，
限制切向和 z 方向，beta=5e6 dyn/cm³。用户已明确同意保留。
论文真实 LV 节没有给出这项约束，不能称为完全相同的底部条件。

## 实现与复现范围

每一步冻结同一旧位置模板 Xn，并求解

`R(y) = y - dt * I(Xn, u_new(Xn+y)) = 0`。

内层先在新位置计算有限元弱力 b，使用一致 CSR 质量矩阵求解 `M F=b`，
将力传播到 MAC 网格；求解 BE 速度 Helmholtz 方程与压力投影，然后通过同一
IB 模板插值并求解 `M U=Bᵀ W K u`。残差、正 Jacobian、IB 完整支撑和线性
求解精度均检查；没有通过删掉输运报警来替代改变数值格式。

可选 `--nonlinear-solver anderson-newton` 保留相同 BE–BE 残差，通过 Anderson
加速和组装 CSR 固体切线的 Newton 回退求解。它属于执行/代数求解器替代，
**不是论文使用的 JFNK/BiCGSTAB**。纯 CSR Newton 用 `--nonlinear-solver newton`。
GPU 路径复用编译固体力、分块几何检查、共享融合 IB 模板与双向传播、CSR
质量矩阵和 Graph PCG。新的零 Dirichlet 压力使用独立几何多重网格，提供
编译平滑与 CUDA Graph；不能复用旧零均值 Neumann 压力工作区。

JFNK 缓存每个非线性残差的范数。BiCGSTAB 使用递推残差，在候选收敛、
每 `nonlinear.linear.check_every` 次迭代（默认 5）及迭代预算耗尽时检查真实残差。
只有真实残差满足原容差才接受解；候选被拒绝或递推残差明显漂移时，从真实
残差重新启动。点积结果合并读回 CPU；BE 黏性求解的右端范数每次求解只计算
一次。这些改动调整代数求解的执行和核验频率，保留 BE–BE 方程及收敛容差。

必须区分数值格式实现和论文结果复现。论文提供的真实 LV 网格计数与用户的
26,889 节点、135,430 单元一致，但计数不能证明几何和方向场完全一致。
没有作者原始曲线数据时，应将体积曲线、最终充盈量、变形和 J 与图 25/27
逐项比较，再判断复现程度；当前报告始终将 `published_results_reproduced`
标为 false，直到独立结果分析确认。

已知实现选择/差异：

- 上述用户底部约束。
- 固定时间步；论文真实算例展示自适应时间步，但没有完整控制器参数。
- 论文未公开半拉格朗日追踪/插值细节，本实现明确采用一阶追踪和三线性插值。
- 速度 Neumann 条件用于黏性子步；之后按式 43 做压力投影，不再覆盖法向速度。
  因而这里是分步开放边界处理，并不声称逐点满足一个整体 Stokes 边界离散。
- 正权 Xiao–Gimbutas 规则、密度 2 和非线性/内层容差属于明确记录的实现选择。
- 特征追踪 CFL 上限 1 为本实现的精度/越界筛查选择，不是论文证明的稳定界。
- 压力加载来自图 24 的重建；默认流体盒原点 (0,0,0) 适用于当前用户坐标。

论文能量证明的适用假设不能直接推广为此开放边界、外载荷、修正 H–O 模型
在任意 dt 下无条件收敛。隐式格式也不保证 JFNK 一定收敛。通过一次运行不等于
空间/时间收敛验证；程序不会将完成时长自动解释为论文结果复现。

## 周期压力与主动收缩拓展

`--config demo/real_lv_fsi/active_cycle.json` 选择此拓展，默认三个 0.8 s 周期。
采用用户指定的材料参数，保留当前 RAW-I1/应力修正形式、128³ 流体、dt=1e-4、
径向底部约束和 BE–BE 残差。原论文参数继续作为默认被动算例的参数。
采用实测较快的 Anderson–Newton、Helmholtz Graph、共享 IB reference 执行。
`reference` 在这里仍是融合 GPU 内核。其他参数可通过部分 JSON 或命令行修改。

| 参数 | 主动拓展的用户参数 |
| --- | --- |
| a, b | 2400 dyn/cm², 5.08 |
| a_f, b_f | 14600 dyn/cm², 4.15 |
| a_s, b_s | 8700 dyn/cm², 1.6 |
| a_fs, b_fs | 3000 dyn/cm², 1.3 |
| beta_s，即代码 kappa | 5e6 dyn/cm²，体积惩罚 |
| 底部 beta | 5e6 dyn/cm³，独立的径向约束系数 |

材料数值已是厘米制应力单位，不再乘 10000；只有 kPa 波形需要单位换算。
这是参数替换，不会自动改回旧 UFL 的等容 I1bar 形式。

总第一类 Piola 应力为 `P = P_passive + P_active`，其中

`P_active = T(t) * [1 + 4.9*(lambda_f - 1)] * (F*f0) ⊗ f0`，
`lambda_f = sqrt(f0ᵀ Fᵀ F f0)`。

这与用户之前提供的 UFL 一致。T 是主动应力系数，单位 dyn/cm²；不能直接把
标量 T 加到应力张量，也不把 T 当作已经归一化的 Cauchy 主动应力。该主动项
在新时刻、新试探位置求值，包含在编译有限元力与组装 CSR Newton 切线中。
默认不截断伸长因子；当 lambda_f < 1-1/4.9≈0.796 时它会变负，这是原式的性质。

相位 `tau = t mod 0.8`，以下值先按 kPa 求出，再乘 10000 转为 dyn/cm²：

| 相位（s） | 内膜压力 p（kPa） | 张力系数 T（kPa） |
| --- | --- | --- |
| 0–0.2 | 1.067*tau/0.2 | 0 |
| 0.2–0.5 | 1.067 | 0 |
| 0.5–0.65 | 1.067+13.46*(1-exp(-d²/0.004)), d=tau-0.5 | 84.26*(1-exp(-d²/0.005)) |
| 0.65–0.8 | 同上，d=0.8-tau | 同上，d=0.8-tau |

周期边界压力从约 1.067 kPa 重置到 0，保留原表达式的跳变。84.26 kPa 是
张力表达式的幅值系数，实际峰值约 83.324 kPa；压力峰值约 14.4785 kPa。
这些是给定的压力/激活载荷，没有瓣膜或闭环循环模型，不能把分段名称当作
已强制满足等容收缩、射血等生理约束。时变主动项会输入机械能，不能据此
宣称无外力系统的能量单调衰减定理适用。

主动测试覆盖应力、力、切线、加载时刻、真实耦合残差、检查点和输出：

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_paper_active.py
```

真实网格验证需要覆盖 t>0.5 s，0.005 s 短模拟尚未激活。先跑一个周期可用：

```bash
CUDA_VISIBLE_DEVICES=0 python -u demo/real_lv_fsi/run_mac.py \
  --config demo/real_lv_fsi/active_cycle.json \
  --mesh-dir /mnt/large2/gjh/realistic_left_ventricle \
  --device cuda --cycles 1 --output results/real_lv_active_1cycle
```

三个周期后台运行（24,000 步）：

```bash
run_dir="results/real_lv_active_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_dir"
CUDA_VISIBLE_DEVICES=0 nohup /usr/bin/time \
  -f 'elapsed_seconds=%e exit_code=%x' -o "$run_dir/runtime.txt" \
  python -u demo/real_lv_fsi/run_mac.py \
  --config demo/real_lv_fsi/active_cycle.json \
  --mesh-dir /mnt/large2/gjh/realistic_left_ventricle \
  --device cuda --cycles 3 --output "$run_dir/simulation" \
  > "$run_dir/run.log" 2>&1 < /dev/null &
echo $! > "$run_dir/launcher.pid"
echo "$run_dir"
```

从未变形网格、零流速开始；不会把被动充盈末态当作新的无应力参考构形。
`--resume` 只恢复原检查点的协议，禁止中途把被动载荷改为主动载荷。
主动检查点可用同一入口续算，例如 `--resume .../checkpoint.npz --end-time 2.4`。
`history.csv` 记录真实 T(t)，报告区分主动拓展和被动复现。
`--cycles` 要求 active-cycle 协议，不与 `--end-time` 同时使用。
若要先隔离材料变化对充盈的影响，可用同一预设加
`--load-protocol inflation --end-time 1.5`，仅运行 0–8 mmHg 被动充盈。

V 表示内膜与虚拟底部封口围成的腔内容积，cm³=mL，并非累计流入量。
论文图 25 的末期体积约 132–133 mL（读图估计）。完成时间推进不等于该曲线
已经复现；如果结果明显偏离，需先核对参考网格、fiber/sheet、边界和离散解。
添加主动应力本身不能修复被动充盈偏小。

## 安装与短模拟

在仓库根目录执行；实际求解不依赖 DOLFINx 或 Docker。

```bash
git pull --ff-only
conda activate afsi-torch
python -m pip install -e ".[test,mesh,fused,quadrature,geometry]"
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_paper_lv.py tests/test_mac_stokes_execution.py

CUDA_VISIBLE_DEVICES=0 python -u demo/real_lv_fsi/run_mac.py \
  --mesh-dir /mnt/large2/gjh/realistic_left_ventricle \
  --device cuda --dt 1e-4 --end-time 0.005 \
  --output results/real_lv_ma2024_early
```

输入文件：mesh_scale.xdmf 及其引用的 HDF5、boundaries.xml、fibers_0/1/2.xml、
sheets_0/1/2.xml。文件名和材料、底部、输出、非线性容差可在 `CONFIG` 或部分
JSON 配置中修改。`--write-config path.json` 保存有效配置后退出。

可用 `--reference --fluid-cells 8` 做适合小网格的 CPU 测试；真实网格不要用过粗
流体网格开展物理验证。`--no-convection` 仅用于无对流算子检查，不是默认复现。

## 完整 1.5 s 后台运行

先确认短模拟与 GPU 测试成功，再从未变形初态开始，**不使用旧 CNAB 检查点**。

```bash
conda activate afsi-torch
run_dir="results/real_lv_ma2024_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_dir"
CUDA_VISIBLE_DEVICES=0 nohup /usr/bin/time \
  -f 'elapsed_seconds=%e exit_code=%x' -o "$run_dir/runtime.txt" \
  python -u demo/real_lv_fsi/run_mac.py \
  --mesh-dir /mnt/large2/gjh/realistic_left_ventricle \
  --device cuda --dt 1e-4 --end-time 1.5 \
  --output "$run_dir" > "$run_dir/run.log" 2>&1 < /dev/null &
echo $! > "$run_dir/launcher.pid"
echo "$run_dir"
```

默认 JFNK 便于对齐论文代数求解器。要比较更快的同残差 GPU 实现，把
`--nonlinear-solver anderson-newton` 加入新的运行命令，使用另一输出目录。
短阶段速度不能外推为全程稳定或承诺运行时间。

监控与续算：

```bash
tail -f "$run_dir/run.log"
cat "$run_dir/runtime.txt"  # 退出时生成
CUDA_VISIBLE_DEVICES=0 python -u demo/real_lv_fsi/run_mac.py \
  --device cuda --resume "$run_dir/checkpoint.npz" --end-time 1.5
```

保存 `history.csv`、`report.json`、`configuration.json`、带校验和的
`checkpoint.npz`。ParaView 打开 `vtk/solid.pvd` 与 `vtk/fluid.pvd`。
压力元数据说明为物理压力、盒壁零值，不再写成零均值规范。

## 预热后性能检查

从上述短模拟的检查点开始，对比每次线性迭代核验与每 5 次核验。
该脚本在内存中推进，不修改输入检查点，不输出 VTK；初始化和预热时间单独记录。
它保留最终真实残差核验，并记录流体求解、Jacobian 作用、真实残差检查次数和
两种频率下的最终场差异。这里比较的是核验频率，不是所有旧版本优化的总加速。

```bash
CUDA_VISIBLE_DEVICES=0 python -u validation/benchmark_paper_lv.py \
  --checkpoint results/real_lv_ma2024_early/checkpoint.npz \
  --device cuda --warmup 5 --steps 20 \
  --linear-check-intervals 1 5 \
  --output results/paper_lv_performance/report.json
```

降低完整算例的输出频率可加入 `--log-every 500 --output-every 1000`。
这不会降低求解器收敛检查频率；两类检查分别设置。

## 按耗时优先级选择执行方式

同一步内的试探位移、当前坐标和有效性检查按张量身份与版本复用，
Anderson 的前置检查与残差计算因此使用同一份几何。不同 FD 探针不会做完整
张量相等性比较；接受克隆的最终迭代点时才比较一次，保留最终残差核验。
输入或借出的试探坐标被原地修改后会使缓存失效。

`--support-backend vertices` 为默认支撑检查：P1 节点凸包足够远离盒壁时，
不生成积分点坐标。靠近壁面时按原积分点规则回退；不会因为一个顶点不满足
快速充分条件就拒绝原本合格的积分点。自适应阶数和点数预算仍检查。
`--support-backend points` 保留原检查用于对照。

`--helmholtz-backend workspace` 对 BE 黏性 Jacobi 使用固定双缓冲；
`graph` 在 CUDA 上进一步捕获一个原有检查间隔的扫掠块。每个块后仍检查真实
残差，投影输出和压力具有独立所有权。默认保留 `reference`，待目标 GPU
测量后选择更快方式。参考实现也已去除初值的多余三分量复制。

先比较耦合求解器与黏性执行，不改变材料、dt、流体网格、积分密度和收敛容差：

```bash
CUDA_VISIBLE_DEVICES=0 python -u validation/benchmark_paper_lv.py \
  --checkpoint results/real_lv_ma2024_early/checkpoint.npz \
  --device cuda --warmup 5 --steps 20 \
  --solvers jfnk anderson-newton --linear-check-intervals 5 \
  --support-backends vertices --helmholtz-backends reference graph \
  --profile --profile-steps 3 \
  --output results/paper_lv_priority_performance/report.json
```

常规测量与分项采样分开。分项使用末尾统一读取的 CUDA events，内部不增加
逐算子的同步；嵌套的质量求解、传播/插值、压力和流体总计不可相加。
events 区间可能包含 CPU 发射间隙，不等于独立的纯内核计算时间。
报告给出相同结束时刻的位置、压力、速度差异与真实残差接受比。

随后用上一对照中更合适的求解器和黏性后端，分别比较共享 IB 的
`reference/vector/reduced`，例如下面命令使用 Anderson–Newton 和黏性 Graph：

```bash
CUDA_VISIBLE_DEVICES=0 python -u validation/benchmark_paper_lv.py \
  --checkpoint results/real_lv_ma2024_early/checkpoint.npz \
  --device cuda --warmup 5 --steps 20 \
  --solvers anderson-newton --linear-check-intervals 5 \
  --helmholtz-backends graph --ib-shared-executions reference vector reduced \
  --profile --output results/paper_lv_ib_performance/report.json
```

三种 IB 执行保留所有积分点和 Peskin 邻居，区别为组织读写与局部归约方式，
`reference` 仍为 GPU 内核。浮点累加次序可能不同；应结合最终场差异判断。
Anderson–Newton 属于论文代数求解器替换，报告继续注明这项复现差异；
短阶段加速与收敛不能保证整个 1.5 s 的行为。

## 降低每步完整耦合求值次数

`fluid solves` 包含非线性残差求值与 Newton/GMRES 的流体响应。一轮通常包括
两次一致质量矩阵求解、IB 传播、BE 黏性/压力投影和 IB 插值。报告与历史现在
分别记录 AA、Newton、GMRES、切线组装、质量求解和总压力循环数。
这些计数只增加 CPU 整数，不增加逐算子的 GPU 同步或计时。

Anderson 回退到 Newton 时，复用已经计算过的最佳点真实残差，避免在相同
位置再次执行整轮耦合；Newton 的初始几何检查和最终 BE 残差检查继续保留。
复用的是完全相同的坐标和同一时间步的残差，不用上一时间步的残差代替。

`--anderson-policy legacy` 保留保存配置的基础 AA 预算，关闭额外迭代和停滞
判断；所有变体都包含上述残差复用。`adaptive` 最多额外允许 4 次 AA：
最近 3 次真实残差的几何平均每步下降率必须不大于 0.8，并且预测在剩余预算内达到原
容差。若最近 3 次最佳残差改善不足 5%，且已有至少 history_size 次 AA（默认 4），则提前回退。
每次新增试探仍检查几何及真实残差。原纸面 BE–BE 方程、dt、载荷、质量矩阵
和最终非线性容差不变。这是有界的代数求解策略，不保证每个时间段都更快。
JSON 的 `anderson` 对象可以单独设置这些控制值；默认额外预算为零。

`--newton-preconditioner solid-block` 提供可选右预条件器。从当前组装的固体
力 Jacobian 提取节点 3×3 块，使用 P1 参考节点体积近似局部流体惯性，构造
`B_i = I - dt²/(rho*m_i) * sym(K_ii)`。对其特征值设置下界 1，再在 GPU 上
应用块逆。CSR 块索引只构建一次，不构造稠密全局耦合矩阵。
节点体积近似只用于预条件器；真实 Jacobian 仍经过原一致质量逆、IB、黏性
及压力投影。最终检查原残差。默认 `none`，因为这一局部近似不包括流体
非局部性，可能帮助或拖慢特定变形状态，需测量后选择。

从本次较慢的约 0.3 s 检查点比较，而非使用 0.005 s 的早期检查点。
下面变量需替换为本次实际路径；基准不修改输入检查点或模拟输出：

```bash
checkpoint="本次输出目录/simulation/checkpoint.npz"
CUDA_VISIBLE_DEVICES=0 python -u validation/benchmark_paper_lv.py \
  --checkpoint "$checkpoint" --device cuda --warmup 5 --steps 20 \
  --solvers anderson-newton --linear-check-intervals 5 \
  --anderson-policies legacy adaptive \
  --newton-preconditioners none solid-block \
  --profile --profile-steps 3 \
  --output results/paper_lv_solver_performance/report.json
```

四种组合从同一检查点开始。报告含每步耗时、前 3 个测量步的完整残差历史、
Newton 回退比例和结束场差异，`fastest_variant` 指明本次最快配置。
分项计时含嵌套区间，不将父子项相加。GPU 正在运行其他模拟时，应先停止该
任务再做性能对照；并发任务会影响计时。

两项求解器配置允许在原检查点续算时覆盖，例如基准确认 adaptive/none 更快后：

```bash
CUDA_VISIBLE_DEVICES=0 python -u demo/real_lv_fsi/run_mac.py \
  --device cuda --resume "$checkpoint" --end-time 2.4 \
  --anderson-policy adaptive --newton-preconditioner none
```

续算保持材料、网格、时间步长和载荷，写回原模拟目录；旧 CSV 会补充新计数
字段，旧检查点中未保存的求解器设置使用兼容默认值。中途不能改变物理配置。
