# AFSI 原生非线性固体多步耦合对照

上一轮给定节点力的单步 AFSI C++ IB/Chorin 对照已通过。本轮用同一个 4 cm 流体盒、4×4×4 个 Q2 单元、`dt=5e-5 s`、CGS 单位和 AFSI 的四点 Peskin 核，改为让固体真正移动并逐步重算力。固体是盒内的小型 P2 四面体，不是左心室。原生端调用安装的 `afsic` IB 扩展和 `ChorinSolver`，以 DOLFINx/UFL 装配 Guccione 被动力、纤维主动应力、随动力压力及参考构形基底弹簧。PyTorch 端使用现有 P2 力装配、IB 和 Chorin 耦合驱动。两端都按 AFSI 示例的顺序：初始节点力为零，步 `n` 用上一轮力计算流体和固体速度，更新位置，再按 `t_n` 装配下一轮节点力。

每一步比较 IB 流体力密度、流体速度和压力、插值固体速度、**相对初始构形的固体位移**及新的非线性节点力。报告给出首个超出容差的时间步和阶段。位移比较不会因约 2 cm 的参考坐标而掩盖微小运动误差。两端线性求解器及浮点归约次序不同，因此使用相对/绝对容差；通过不等于逐位相同。本实验需要至少两步，默认六步。

在已启动、可导入 `dolfinx` 和 `afsic` 的 `afsi_dev_ljy` 中导出。若宿主机仓库未挂载到容器，用以下命令复制两个参考脚本并取回结果：

```bash
mkdir -p results/afsi_nonlinear_trajectory
docker cp validation/export_afsi_nonlinear_trajectory.py afsi_dev_ljy:/tmp/export_afsi_nonlinear_trajectory.py
docker cp validation/reference_io.py afsi_dev_ljy:/tmp/reference_io.py
docker exec afsi_dev_ljy python /tmp/export_afsi_nonlinear_trajectory.py \
  --output /tmp/afsi_nonlinear_trajectory.npz --steps 6
docker cp afsi_dev_ljy:/tmp/afsi_nonlinear_trajectory.npz \
  results/afsi_nonlinear_trajectory/reference.npz
```

然后在宿主机 GPU 环境中运行：

```bash
conda activate afsi-torch
CUDA_VISIBLE_DEVICES=0 python validation/compare_afsi_nonlinear_trajectory.py \
  --reference results/afsi_nonlinear_trajectory/reference.npz \
  --device cuda --output results/afsi_nonlinear_trajectory/report.json
```

`report.json` 无论比较是否通过都会先保存。请保留 NPZ 和 JSON；如果失败，看 `first_mismatch`，从对应时间步的前一轮力和位置追踪误差。固体采用与 AFSI 示例相同的 Guccione 参数，但使用常量纤维方向和人工短时载荷，目的是隔离耦合实现。它不能替代理想左室网格、纤维场、预载、整周期边界条件或网格/时间收敛验证。当前 Windows 本地环境没有 DOLFINx/afsic，原生导出需要在你的 Linux 容器验证。

相关原生实现：[AFSI Chorin](https://github.com/loveIroha/afsi/blob/main/afsic/src/afsic/euler/ChorinSolver.py)、[AFSI IB](https://github.com/loveIroha/afsi/tree/main/afsic/src/coupling)、[AFSI 理想左室示例](https://github.com/loveIroha/afsi/blob/main/afsic/demo/demo_337/fsi_paralell_fibers_contraction.py)。

