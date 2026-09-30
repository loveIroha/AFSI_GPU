# 原版 AFSI demo_340 容器对照与计时

使用现有 `afsi_dev_ljy`，从 Linux **宿主机**启动，无需进入容器、激活 afsi-torch 或安装 PyTorch 到容器。脚本调用容器原有 Python、DOLFINx 和 AFSI。不改动上游仓库源码，不请求远程实验编号，也不上传 Swanlab。

## 运行

先完成 GPU demo_340 的短程检查，使 `results/demo_ideal_valve/mac/checkpoint.npz` 存在。转换只取**参考坐标 X 和三角形连接**；即使检查点已到 3 s，CPU 仍从未变形固体、静止流体开始。原 AFSI 用相同 P1 几何建立自己的 P2 空间，不需要重新生成 Gmsh 网格或匹配 P2 节点坐标。

在宿主机的 AFSI_GPU 根目录执行：

```bash
git pull --ff-only
bash scripts/run_afsi340_container.sh \
  results/demo_ideal_valve/mac/checkpoint.npz
```

默认容器源码位置为 `/root/afsi/afsic/demo/demo_340`，需要 `fsi_paralell.py`、`FRH.py`、`NeoHookean.py` 已存在。脚本检查环境，将同一网格写成原版 XDMF/HDF5 和标签并读回检查，之后才用 `docker exec -d` **后台启动**完整 3 s、48000 步。每次使用独立目录，不覆盖结果。启动消息不保证求解已成功，需检查日志。

8 个 MPI 进程并行运行：

```bash
AFSI340_NP=8 bash scripts/run_afsi340_container.sh \
  results/demo_ideal_valve/mac/checkpoint.npz
```

每个 MPI 进程的库线程数仍是 1，避免启动 8×8 个线程。脚本在当前容器执行同样进程数的 IB 自检，`mpi_check.json` 必须 `passed=true` 后才启动完整模拟；不会仅因发现 MPI 可用就直接开始长时间运行。自检核对全部流体节点唯一 ownership、插值、传播，覆盖原版核在边界处丢弃网格外链接的行为。

已核对 [AFSI 的 C++ Python 绑定](https://github.com/loveIroha/afsi/blob/main/afsic/src/afsic_ext.cpp)：它使用 MPI Gatherv 收集 owned 数据，在 rank 0 执行原版 IB，再用 Scatterv 发回 owned 数据。因此保留原版 C++ IB 和数值公式，不改成新的 IB 算法；IB 核仍是串行瓶颈，有限元组装及 PETSc 求解可分布式运行。入口补上 IB 返回后的 solid velocity/force ghost 同步、速度最大值的 MPI 归约，以及 rank 0 独占日志/报告写出。KSP 未收敛时报错，不放宽容差。

本地 5 项测试通过，但无 DOLFINx/MPI 容器，8 进程实际验证由上述启动自检完成；尚未宣称完整并行轨迹与串行轨迹一致。

本地没有目标 Docker 环境；离线源码转换和输入校验测试已通过，DOLFINx 写出及原版求解仍需在目标容器验证。

## 监控、停止与结果

脚本会打印 `/root/afsi-data/afsi340_runlog/<运行ID>` 的确切路径及监控命令。替换下面的 `<运行ID>`：

```bash
docker exec afsi_dev_ljy tail -f \
  /root/afsi-data/afsi340_runlog/<运行ID>/run.log
docker top afsi_dev_ljy | grep '[o]ffline_runner.py'
docker exec afsi_dev_ljy cat \
  /root/afsi-data/afsi340_runlog/<运行ID>/runtime.txt
```

`runtime.txt` 在进程退出时写出，如 `elapsed_seconds=123.456789 exit_code=0`。同目录有 `config.json`、`history.csv`、`report.json`，原版场文件在 `fields/`，输入及来源信息在 `input/`。

停止使用容器 PID，不使用 `docker top` 显示的宿主机 PID：

```bash
docker exec afsi_dev_ljy bash -c \
  'kill -INT $(cat /root/afsi-data/afsi340_runlog/<运行ID>/launcher.pid)'
```

中断后通常 `exit_code=130`，不是完成结果。原版没有增加断点续算；再次运行是从头开始。容器 `/root` 已映射到宿主机 `/mnt/large2/qwer`，所以结果也可直接从下面的宿主机路径读取：

MPI 失败时某个进程可能无法执行 Python 的结束记录，因此另有 `mpi_runtime.txt` 记录整个 mpirun 作业时间和退出码，包含 MPI 启动/导入开销。正常结束的高精度 `runtime.txt` 为各进程计算计时的最大值；比较时也请提供 `mpi_check.json` 和 `mpi_runtime.txt`。旧版单进程脚本启动的任务仍使用其 `python.pid` 停止。

```text
/mnt/large2/qwer/afsi-data/afsi340_runlog/<运行ID>/
```

也可以复制分析文件：

```bash
mkdir -p results/afsi340_cpu/<运行ID>
docker cp afsi_dev_ljy:/root/afsi-data/afsi340_runlog/<运行ID>/report.json results/afsi340_cpu/<运行ID>/
docker cp afsi_dev_ljy:/root/afsi-data/afsi340_runlog/<运行ID>/history.csv results/afsi340_cpu/<运行ID>/
docker cp afsi_dev_ljy:/root/afsi-data/afsi340_runlog/<运行ID>/runtime.txt results/afsi340_cpu/<运行ID>/
docker cp afsi_dev_ljy:/root/afsi-data/afsi340_runlog/<运行ID>/run.log results/afsi340_cpu/<运行ID>/
```

## 比较与时间口径

- 固体参考网格、FRH、纤维、弹簧、入口、`dt=1/16000 s` 和 `T=3 s` 对齐。原版流体仍为 128×32 Q2/Q1 FEM，GPU 为 256×64 MAC；IB 分别为原版节点传递和本项目积分点传递。
- 默认 1 个 MPI 进程，设置 `AFSI340_NP=8` 则为 8 个 MPI 进程；OMP/OpenBLAS/MKL 各为 1 并记录。报告同时记录实际 MPI 进程数；不要把旧单进程结果与新并行结果混称为同一基线。
- 高精度墙钟计时包含原版设置/装配、推进、每步诊断和 XDMF 输出，排除输入转换、Docker 启动及 Python 导入/离线源码解析。GPU 报告包含网格生成/编译，续算也有启动开销，比较时应注明时间范围差别。
- 保留原版每步诊断及原有输出频率。总耗时比是各自完整实现的效率比，不能直接视为同一算法的纯 CPU/GPU 加速。
- CSV 保存原版上瓣尖 x/y 位移和面积 `volume`。源程序时间是 `step*dt`，但坐标已更新一步，另存 `accepted_time_s=source_time_s+dt` 与 GPU 对齐。原版二维 `volume` 对应 GPU `solid_area_cm2`。
- 正常完成需要 `report.json` 中 `completed=true` 和 `runtime.txt` 中 `exit_code=0`；失败或中断的耗时不作为完整运行成绩。

分析请提供两边 `history.csv`、`report.json`、`runtime*.txt` 及 CPU 日志末尾，先比较瓣尖运动和面积，再解释耗时差。
