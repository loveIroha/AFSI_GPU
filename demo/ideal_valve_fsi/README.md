# AFSI demo_340：二维理想双瓣叶 GPU 算例

入口为 `demo/ideal_valve_fsi/run_mac.py`，默认推进 **3 s、48000 步**。网格在 CPU 上用 Gmsh 生成；P2 固体有限元、FRH 应力/节点力组装、CSR 一致质量矩阵求解、IB 插值传播及二维 MAC 流体/压力多重网格在指定的 PyTorch 设备上运行。指定 `--device cuda` 时 CUDA 不可用会报错，不会退回 CPU。无需安装 FEniCSx、AFSI、Docker 或 Swanlab。

这是沿用本项目 MAC/积分点 IB 框架的新算例，不是原 AFSI 离散的逐项复刻。用户已完成旧版 GPU 的 3 s 运行（1094.143 s）；与原 AFSI 的完整轨迹一致性尚未建立。默认现启用二维固定压力工作区、Triton 融合模板、CUDA Graph、跨步 IB 缓存和合并检查；优化版 GPU 测试和性能仍待验证。[执行优化、已有检查点短程 A/B 与独立完整运行](../../docs/VALVE_GPU_EXECUTION.md)。

## 与原算例对齐的物理设置

参照 [原算例主程序](https://github.com/loveIroha/afsi/blob/main/afsic/demo/demo_340/fsi_paralell.py)、[双瓣叶网格生成](https://github.com/loveIroha/afsi/blob/main/afsic/demo/demo_340/generate_mesh-1.py) 和 [FRH 本构](https://github.com/loveIroha/afsi/blob/main/afsic/demo/demo_340/FRH.py)。本项目独立实现相应公式。

| 项目 | 本算例默认值 |
| --- | --- |
| 单位 | cm、g、s；二维力/能量按单位面外厚度 |
| 通道 | `[0,8] × [0,1.61]` cm |
| 下瓣叶 | `x∈[1.9788,2]`，`y∈[0,0.7]` cm |
| 上瓣叶 | `x∈[1.9788,2]`，`y∈[0.91,1.61]` cm |
| 初始间隙 | 0.21 cm |
| 固体网格 | Gmsh 尺寸 0.01 cm，P1 直边几何＋P2 位移/坐标；两瓣叶不相连 |
| 根部约束 | 下瓣叶 `y=0`、上瓣叶 `y=1.61` 的两分量弱弹簧，`beta=1e8`，没有替换成强制固定自由度 |
| 纤维 | 下瓣叶 `(1,1)/√2`，上瓣叶 `(1,-1)/√2` |
| FRH 参数 | `C0=2e5`，`C1=1e6`，`kappa=4e5` dyn/cm² |
| 流体 | `rho=1 g/cm³`，`mu=0.1 g/(cm·s)` |
| 入口 | `ux=5*(sin(2*pi*t)+1.1)*y*(1.61-y)` cm/s，`uy=0`；周期 1 s |
| 上下壁面 | 无滑移；MAC 切向速度用奇反射边界延拓 |
| 出口 | `p=0`，暂定速度法向导数为零；投影后保留出口速度修正以满足流量守恒 |
| 时间步/结束时间 | `dt=1/16000 s`，`T=3 s` |
| 额外载荷 | 无主动收缩、无额外入口压力波形、无显式接触模型 |

FRH 使用二维 `Fbar=J^(-1/2)F`、`I1bar=tr(FbarᵀFbar)`、`I4bar=f·FbarᵀFbar f`，能量为

```text
W = C0/2*(I1bar-3) + C1*(exp(I4bar-1)-I4bar)
    + kappa/2*((J²-1)/2-log(J))
```

应力采用该能量的解析导数，并用 PyTorch 自动求导独立检查。输出能量减去参考构形常数密度，使参考能量为零；应力和动力学不受这个常数影响。每个三角形用正权重六点、四次精确积分，根部用三点 Gauss 积分。默认 Gmsh 4.15.2 生成 **1986 个 P2 节点、846 个三角形、5076 个 IB 积分点**；这些计数不是自由度无关的物理参数，以报告实际值为准。

## 保留的离散差别

原 AFSI 流体是 **128×32 的 Q2/Q1 有限元**；本算例是 **256×64 的 MAC 有限差分**，间距分别为 `0.03125`、`0.02515625` cm，对齐原 Q2 速度节点间距。它不等同于原有限元压力空间，MAC 压力网格也更细。

原 Chorin 暂定速度使用显式对流、隐式黏性；这里使用中心守恒形式的显式对流与显式黏性，随后进行压力投影。程序检查黏性数、CFL 和中心显式输运的步长条件；这些保护不构成非线性长期稳定性证明。压力采用与 `-D G` 一致的混合边界算子，入口/壁面为齐次 Neumann、出口为 Dirichlet，不删除压力 RHS 的均值。

原 [C++ IB](https://github.com/loveIroha/afsi/blob/main/afsic/src/coupling/main.h) 在固体自由度坐标传递，并跳过网格外的核链接；这里沿用左室实现的积分点 IB：先解 CSR 一致质量矩阵，将弱节点力转换成力密度系数，再进行四点 Peskin 核传播；插值经一致质量矩阵投影回 P2 空间。靠近上下壁面使用同一套奇反射权重进行插值和传播，保持离散功率伴随关系。壁面承担反力，因此不能要求壁面附近传播力与固体力的合力简单相等。水平入口/出口附近不允许固体 IB 核越界。

耦合时序与源程序对应：旧固体力推动流体 → 插值速度 → 更新坐标 → 更新 IB 点 → 组装新固体力供下一步使用。入口在 `step*dt` 采样，接受状态标记为 `(step+1)*dt`。质量矩阵使用图捕获 PCG 和热启动，FP64；网格与材料不因执行优化而改变。

## Linux RTX 4090：安装与短程检查

在 AFSI_GPU 仓库根目录运行，先确认正在使用已清理路径污染的 `afsi-torch` 环境：

```bash
git pull --ff-only
conda activate afsi-torch
python -m pip install -e ".[test,geometry,fused]"
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_valve_mac.py tests/test_valve_execution.py
CUDA_VISIBLE_DEVICES=0 python -u demo/ideal_valve_fsi/run_mac.py \
  --device cuda --end-time 0.005 --fluid-fields \
  --output results/demo_ideal_valve/mac
```

该目录必须是新的；已有计算使用 `--resume`，不覆盖旧实验。短程运行是 80 步，会触发首次编译；不要用这段启动耗时估算完整 3 s 的速度。

本地 CPU 新测试及现有 MAC 回归为 **20 passed、18 CUDA skipped**，覆盖本构导数、几何/边界弱力、制造压力场、出口流量/散度、内点合力/力矩/功率、壁面伴随关系、耦合短程、断点续算、检查点校验及 VTU 输出。默认网格已推进至 **0.02 s、320 步**，`minJ≈0.97903`、散度 L2 约 `1.95e-11`、入口/出口流量误差在浮点精度内。CPU 通过不能代替上述 GPU 测试。

## 从短程检查继续完整 3 s：后台运行

确认 GPU 测试和短程检查通过，再运行：

```bash
mkdir -p results/demo_ideal_valve/mac
CUDA_VISIBLE_DEVICES=0 nohup /usr/bin/time \
  -f 'elapsed_seconds=%e exit_code=%x' \
  -o results/demo_ideal_valve/mac/runtime_3s.txt \
  python -u demo/ideal_valve_fsi/run_mac.py \
  --device cuda --resume results/demo_ideal_valve/mac/checkpoint.npz \
  --end-time 3.0 --fluid-fields \
  > results/demo_ideal_valve/mac/run_3s.log 2>&1 < /dev/null &
echo $!
```

`--resume` 从检查点恢复网格、材料、时间步、流体状态和旧固体力，不需要重新生成几何。不恢复 PCG/MG 的执行缓存，重新编译和前几次冷启动仅改变执行过程。可在中断后用同一条续算命令继续；确认旧进程已退出后再启动。

```bash
tail -f results/demo_ideal_valve/mac/run_3s.log
pgrep -af 'demo/ideal_valve_fsi/run_mac.py'
cat results/demo_ideal_valve/mac/runtime_3s.txt
```

默认每 160 步输出一帧和一行诊断、每 1600 步存检查点。`--field-every 0` 可关闭场文件；`--fluid-fields` 才输出流体场，固体场默认开启。完整运行约 300 个采样时点，另有初始帧和短程结束帧。

## 输出和结果判读

原版 AFSI 容器 `afsi_dev_ljy` 的相同固体网格对照、后台运行与计时见 [CPU 对照说明](../../docs/AFSI340_CPU_COMPARISON.md)。

- `report.json`：设置、实际步数、物理时间、求解器信息、累计运行时间和失败原因。
- `history.csv`：上下瓣尖位移、探针间隙、面积、`detF`、入口/出口流量、散度及采样时点的功率误差。上瓣尖探针沿用源程序的参考位置 `(1.9894,0.9101)`；探针间隙不是两瓣叶全局最短距离或接触判据。
- `checkpoint.npz`：可校验完整状态；失败时尽量保存最后接受的时间步，不跳过异常继续计算。
- `fields/solid.pvd`、`fields/fluid.pvd`：ParaView 时间序列；固体 P2 三角形节点已转换为 VTK 顺序。固体变形已写进当前坐标，直接播放即可；再次用位移 Warp 会重复施加变形。
- `run_3s.log`、`runtime_3s.txt`：后台日志和该进程墙钟时间；正常结束应为 `exit_code=0`。

分析时优先提供 `report.json`、`history.csv`、`runtime_3s.txt`、日志末尾。3 s 完成代表规定加载下的数值轨迹完成，不自动代表网格收敛、周期稳态或与原 AFSI 一致。
