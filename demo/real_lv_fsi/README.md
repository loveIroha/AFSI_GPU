# 真实左心室：Ma 等（2024）的被动充盈算例

本 demo 现以 *An unconditionally stable scheme for the immersed boundary
method with application in cardiac mechanics* 的 V.F 节为复现目标。
[论文信息及作者接受稿](https://eprints.gla.ac.uk/333577/)，DOI: 10.1063/5.0225605。
旧的主动收缩三周期预设已被替换；旧求解器仍保留在库中。
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
