# 二维 MAC 瓣膜：GPU 执行优化

本次沿用 demo_340 的 FRH/P2 固体、积分点 IB、一致质量矩阵和 MAC 流体离散。默认仍为 256×64 流体单元、1986 个 P2 固体节点、846 个三角形、5076 个积分点、FP64、`dt=1/16000 s`、`T=3 s`。未修改载荷、边界、材料、积分规则、Jacobi 算法、收敛容差或检查间隔。

用户报告的旧版完整耗时：RTX 4090 GPU 为 **1094.143 s**，原 AFSI 8 MPI 进程为 **724.324 s**。两者流体压力空间和 IB 离散有差别，不能把这个比值当作相同离散算法的硬件加速比。本次新增的短程 A/B 实验使用同一检查点和同一数值设置比较执行效率；新的 GPU 加速幅度尚待测量。

## 实现内容

| 阶段 | 新执行方式 |
| --- | --- |
| 压力模板 | Triton FP64 二维相邻单元计算；连续数组、固定五点访问，避免用切片临时数组拼接压力算子 |
| 多重网格存储 | 每层预分配压力、右端、Jacobi 双缓冲；保持全局 Jacobi 更新顺序，奇数次平滑也正确 |
| 粗化/延拓 | 残差与 2×2 平均粗化在一个内核中完成；双线性延拓与细网格修正在一个内核中完成 |
| 混合边界 | 保留入口/壁面的偶反射 Neumann 和出口的奇反射 Dirichlet；保留非零均值 RHS，不增加压力钉扎 |
| CUDA Graph | 捕获两次 V-cycle 之间固定执行序列，在原每两次循环的残差检查位置重放；末尾不足两次时用对应短图 |
| 固体力 | 计算一次变形梯度，同时返回组装力及 `det(F)`，避免更新力前另算一次变形梯度 |
| IB 模板 | 接受新坐标时构建一次，下一步插值和传播复用；按状态对象与张量版本失效，续算重新构建 |
| 主机同步 | 合并输入有限值/CFL 检查、暂定速度边界检查，以及位移/正 Jacobian/IB 支撑/有限力检查的结果读取 |

质量矩阵仍用已有 CSR/图捕获 PCG 和热启动。压力解返回独立存储，后续图重放不会修改历史状态。所有失败检查保留，不以跳过检查换取速度。

默认入口启用 `--execution-backend optimized --pressure-backend auto`：CUDA 自动选 `triton-channel-graph`，CPU 选用于等价验证的缓冲实现。CPU 缓冲实现不是性能目标，仍可能产生 PyTorch 临时数组。`--execution-backend reference --pressure-backend reference` 恢复此前编译实现；`--pressure-backend workspace` 使用 Triton 固定工作区而不捕获压力图。

CUDA 需要匹配当前 PyTorch 的 Triton。指定 CUDA 不可用或缺少必要内核依赖时直接报错，不静默退回 CPU。

## 先测试，再读取已有检查点做短程 A/B

在 Linux 宿主机的 AFSI_GPU 根目录执行：

```bash
git pull --ff-only
conda activate afsi-torch
python -m pip install -e ".[test,geometry,fused]"
CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/test_valve_mac.py tests/test_valve_execution.py

CUDA_VISIBLE_DEVICES=0 python -u validation/benchmark_valve_mac.py \
  --checkpoint results/demo_ideal_valve/mac/checkpoint.npz \
  --device cuda --warmup 10 --steps 100 --profile \
  --output results/valve_mac_performance/report.json
```

检查点路径请使用已经存在的二维瓣膜检查点，不是左室检查点。已到 3 s 的检查点也可以使用：基准仅在内存中短程继续该状态，不写回检查点，也不覆盖旧 demo 结果。不需要重新生成网格或先重跑完整 3 s。

程序依次测量 `reference`、`workspace`、`graph`；每组从同一物理状态开始，推进相同的预热和测量步数。预热不计入墙钟结果。最终坐标、力、压力和两分量速度必须通过等价检查，才写出 `equivalence_passed=true` 的报告。

主要读取 `ms_per_step`、`measured_speedup`、压力循环数、质量矩阵迭代数、`stencil_builds` 及每个字段的 `equivalence`。优化路径的测量段预计构建 100 次 IB 模板，参考路径为 200 次。压力和质量矩阵循环数若不同，需要结合残差/状态差异解释，不将它们隐去。

`--profile` 另做一段相同长度的后续重放，记录 CUDA 事件并在该段结束后集中读取；阶段计时不污染主要墙钟测量。阶段轨迹的物理时间晚于主要测量段。`pressure_solve` 已包含在 `fluid_total`，不要把所有阶段相加；省略 `--profile` 可只测墙钟和等价性。完整 demo 不增加逐阶段计时。

## 优化版独立运行与完整 3 s

先做一个新目录的 80 步检查，保留此前对照结果：

```bash
CUDA_VISIBLE_DEVICES=0 python -u demo/ideal_valve_fsi/run_mac.py \
  --device cuda --end-time 0.005 --fluid-fields \
  --output results/demo_ideal_valve/mac_optimized
```

确认 GPU 测试和 A/B 等价检查通过，再从该短程检查点后台继续完整 3 s：

```bash
mkdir -p results/demo_ideal_valve/mac_optimized
CUDA_VISIBLE_DEVICES=0 nohup /usr/bin/time \
  -f 'elapsed_seconds=%e exit_code=%x' \
  -o results/demo_ideal_valve/mac_optimized/runtime_3s.txt \
  python -u demo/ideal_valve_fsi/run_mac.py \
  --device cuda \
  --resume results/demo_ideal_valve/mac_optimized/checkpoint.npz \
  --end-time 3.0 --fluid-fields \
  > results/demo_ideal_valve/mac_optimized/run_3s.log 2>&1 < /dev/null &
echo $!
```

监控和退出后的时间：

```bash
tail -f results/demo_ideal_valve/mac_optimized/run_3s.log
cat results/demo_ideal_valve/mac_optimized/runtime_3s.txt
```

上述完整运行沿用旧版输出频率，便于比较端到端时间。`report.json` 的累计时间包含先前短程段和本次初始化/输出，`runtime_3s.txt` 是本次进程墙钟，两者口径不同。短程 A/B 墙钟排除了初始化、编译和文件输出，不等同于完整运行的精确加速比例。

## 验证范围

测试覆盖制造解、单个 V-cycle 与旧算法等价、奇数/偶数 Jacobi 双缓冲、出口延拓、非零均值 RHS、图执行的检查间隔和末尾部分循环、返回存储独立性、无效数值拒绝、耦合短程/功率伴随关系、缓存失效及旧检查点切换到新执行方式的续算。

本地相关回归为 **31 passed、24 CUDA skipped、1 warning**（二维原有/新执行测试、原 MAC 回归及原 AFSI 容器运行器测试）。默认 256×64/1986 节点网格已完成优化路径 80 步并续算至 160 步、0.01 s，`minJ≈0.983346`、散度 L2 约 `2.25e-11`。已有检查点的 CPU 20 步 A/B 中全部状态字段最大差为零，IB 模板构建由 40 次减为 20 次。

本地只有 CPU，可验证缓冲工作区和耦合等价性；Triton 与 CUDA Graph 的测试必须在目标 RTX 4090 上完成。性能报告不预设加速倍数；二维问题规模小，优化后仍可能受固体/质量求解与主机调度限制。
