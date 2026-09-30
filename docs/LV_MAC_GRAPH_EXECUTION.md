# 三维左心室：压力图重放与合并耦合检查

本次将二维瓣膜验证有效的执行方式适配到三维左心室。二维完整运行已由约 1094 s 降到 333 s；三维的收益必须独立测量，不能直接套用二维加速倍数。

默认物理设置保持 AFSI demo_337 的厘米制生成左室、Laplace 纤维、Guccione 本构、内膜压力、主动张力和基底弹簧。前 1.5 s 加载至 150000/600000 dyn/cm²，保持至 2 s；`dt=5e-5 s`、64³ MAC、默认 0.1 cm 固体网格、P2、14 点积分、FP64、积分点 IB 与 CSR 一致质量求解均不变。完整 2 s 为 40000 步。

## 执行改动及三维边界

- 三维已有融合七点压力模板、紧凑 IB、热启动、固体几何缓存和质量求解图。这次不替换这些数值方法。
- `pressure_backend=workspace` 合并残差与 2×2×2 平均粗化，省去中间细网格残差数组。延拓与修正沿用既有融合内核。
- `pressure_backend=graph` 在新工作区上捕获 V-cycle 序列。默认每两次循环读取真实残差，循环上限、容差与检查位置保持原值；最后不足两次使用短图。
- Jacobi 双缓冲使用固定地址，保持全局迭代顺序，包括奇数次平滑；每层压力与粗网格 RHS 的零均值处理保留。
- 三维背景盒是全 Neumann 压力边界，存在常数零空间。继续拒绝净通量不相容的 RHS，保留逐层去均值及原粗网格零空间处理。没有引入二维的零压力出口。
- `coupling_backend=optimized` 将位移限值、正 Jacobian、内膜/基底非退化、正腔体积、IB 支撑和有限力的接受结果合并读取。固体已有几何缓存继续使用，失败几何不进入缓存。
- 接受新状态时计算一次交互点并准备下一步模板。三维旧路径本来每步只构建一次模板，这次减少的是交互坐标场求值与支撑检查的重复次数。预热后交互坐标求值由每步两次变为一次。
- 状态对象与坐标张量版本决定缓存失效。断点续算、换状态或坐标原地修改会重新准备。推理模式张量不复用跨步缓存。

保持原加载采样次序：旧力推动流体，插值更新坐标，以旧状态时间计算下一步力。所有非法状态保护保留。返回的压力使用独立存储，不会被后续图重放修改。

旧 `pressure_backend=fused`、`coupling_backend=reference` 保留用于对照或回退。旧检查点缺少耦合键时仍恢复参考行为，不静默改变执行选择。新选项已接入 `examples/lv_mac.py` 与 `demo/ideal_lv_fsi/run_mac.py`，支持续算覆盖与保存。demo 报告的 IB 构建/求值计数属于当前进程或续算段；基准计数仅属于预热后的测量段。

## Linux RTX 4090：测试与只读短程对照

在 AFSI_GPU 根目录执行：

```bash
git pull --ff-only
conda activate afsi-torch
python -m pip install -e ".[test,geometry,fused]"
CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/test_mac_lv_graph.py tests/test_mac_multigrid_fused.py \
  tests/test_mac_demo_backends.py
```

用已经完成的三维 MAC 左室检查点进行对照，不必先重跑完整 2 s：

```bash
CUDA_VISIBLE_DEVICES=0 python -u validation/benchmark_lv_mac_graph.py \
  --checkpoint results/lv_mac_solid_mass_2s/checkpoint.npz \
  --device cuda --warmup 10 --steps 100 --profile \
  --output results/lv_mac_graph_performance/report.json
```

检查点必须是三维 MAC 左室的文件。若此前输出在另一目录，只改 `--checkpoint` 路径。可列出已有检查点：

```bash
rg --files --no-ignore results -g checkpoint.npz
```

三组均使用前一版本的 fused 执行基线，保持检查点的固体后端、质量后端和热启动选项。组间只切换压力执行及耦合接受方式：

| 报告组 | 压力后端 | 耦合执行 |
| --- | --- | --- |
| reference | 既有 fused | reference |
| workspace | 融合粗化与固定地址 | optimized |
| graph | workspace＋CUDA Graph | optimized |

各组从同一检查点推进相同的预热步数，再测量相同的 100 步。预热后的 PCG 缓存保持连续，不在测量前重置。主墙钟排除初始化、编译、诊断及文件输出；`--profile` 另做后续重放，事件统一在该段结束后读取。`pressure_solve` 包含于 `fluid_total`，`ib_mass_solves` 包含于传播/插值，不将嵌套阶段相加。完整 demo 不增加逐阶段计时。

报告包含墙钟、压力循环/质量迭代、图数量、工作区内存、交互点求值次数及状态最大差/相对 L2 差。比较容差沿用已有三维执行对照：坐标/力/压力 `rtol=1e-7, atol=1e-8`，速度 `rtol=1e-7, atol=1e-9`。这些是结果比较容差，不改变求解器容差。等价性未通过时仍写报告并以非零状态退出。

检查点只读，不覆盖旧结果。已到 2 s 的检查点只检验保持载荷末段，尚需下述加载初期检查。CPU 仅比较 reference/workspace；不代替 CUDA 编译、图捕获与性能验证。

## 加载初期与完整运行

先用新目录推进 0.005 s、100 步：

```bash
CUDA_VISIBLE_DEVICES=0 python -u demo/ideal_lv_fsi/run_mac.py \
  --device cuda --warm-start \
  --execution-backend fused --solid-backend pointwise --mass-backend graph \
  --pressure-backend graph --coupling-backend optimized \
  --end-time 0.005 --output results/lv_mac_graph_early
```

确认 GPU 测试、A/B 等价性和加载初期运行通过后，可以从该状态继续完整 2 s：

```bash
mkdir -p results/lv_mac_graph_early
CUDA_VISIBLE_DEVICES=0 nohup /usr/bin/time \
  -f 'elapsed_seconds=%e exit_code=%x' \
  -o results/lv_mac_graph_early/runtime_2s.txt \
  python -u demo/ideal_lv_fsi/run_mac.py \
  --device cuda --resume results/lv_mac_graph_early/checkpoint.npz \
  --end-time 2.0 --log-every 200 --checkpoint-every 1000 \
  > results/lv_mac_graph_early/run_2s.log 2>&1 < /dev/null &
echo $!
```

续算继承图/耦合选择及网格、积分、时间步和载荷，不再生成几何。回退只需在续算时加 `--pressure-backend fused --coupling-backend reference`。使用新结果目录保留旧完整轨迹。

## 已完成与待验证

本地相关回归：**70 passed、60 CUDA skipped、1 warning**，包含原 MAC/固体质量求解/demo、二维回归和新增三维图测试。小规模 CPU 左室使用 16³、1133 节点/548 四面体，优化路径已启动并续算至 100 步、0.005 s，`minJ≈0.999547`、散度 L2 约 `3.72e-12`。同一旧检查点 12 步 CPU 对照通过，交互坐标求值由 24 次降为 12 次。CPU 工作区用于验证，不以其墙钟推断 GPU 加速。

新增测试覆盖各层零均值、相容 RHS、制造解、单 V-cycle、奇数 Jacobi、图执行的末尾部分循环、返回存储独立性、缓存失效、失败状态不进入缓存、载荷时序、旧检查点切换、demo 续算/回退及基准文件只读。目标 GPU 的 Triton 新内核、CUDA Graph 和性能仍需按上述命令验证。
