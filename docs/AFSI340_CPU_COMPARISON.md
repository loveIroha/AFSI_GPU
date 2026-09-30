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
  'kill -INT $(cat /root/afsi-data/afsi340_runlog/<运行ID>/python.pid)'
```

中断后通常 `exit_code=130`，不是完成结果。原版没有增加断点续算；再次运行是从头开始。容器 `/root` 已映射到宿主机 `/mnt/large2/qwer`，所以结果也可直接从下面的宿主机路径读取：

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
- 原版用 **1 个 MPI 进程**，OMP/OpenBLAS/MKL 线程数设为 1 并记录。这是受控串行 CPU 基线，不代表 AFSI 最佳多核性能；原版节点 IB 的多进程一致性不在本实验验证范围内。
- 高精度墙钟计时包含原版设置/装配、推进、每步诊断和 XDMF 输出，排除输入转换、Docker 启动及 Python 导入/离线源码解析。GPU 报告包含网格生成/编译，续算也有启动开销，比较时应注明时间范围差别。
- 保留原版每步诊断及原有输出频率。总耗时比是各自完整实现的效率比，不能直接视为同一算法的纯 CPU/GPU 加速。
- CSV 保存原版上瓣尖 x/y 位移和面积 `volume`。源程序时间是 `step*dt`，但坐标已更新一步，另存 `accepted_time_s=source_time_s+dt` 与 GPU 对齐。原版二维 `volume` 对应 GPU `solid_area_cm2`。
- 正常完成需要 `report.json` 中 `completed=true` 和 `runtime.txt` 中 `exit_code=0`；失败或中断的耗时不作为完整运行成绩。

分析请提供两边 `history.csv`、`report.json`、`runtime*.txt` 及 CPU 日志末尾，先比较瓣尖运动和面积，再解释耗时差。
