# 理想左心室：32³ MAC / FEM / grad-div FEM 对照

原生 AFSI demo_337 设置 Nx=Ny=Nz=32、velocity_order=2、pressure_order=1，
在 [0,5]³ cm 上使用 Q2/Q1 六面体有限元。
32³ FEM 元素的速度节点间距是 5/64=0.078125 cm；
32³ MAC 单元的间距是 5/32=0.15625 cm。
本实验对齐背景单元数量，不能称作相同速度分辨率或同离散验证。
此前 64³ MAC 对齐的是 Q2 速度节点间距。

## 三组设置

| 算例 | 流体与 IB | 背景网格 | grad-div |
|---|---|---|---|
| mac32 | CN–AB2/xsPPM7、FGMRES Stokes、中点自适应积分点 IB/FE | 32³ MAC | 无 |
| fem32 | 隐式黏性、显式对流、Chorin、AFSI 节点 IB | 32³ Q2/Q1 FEM | alpha=0 |
| fem32_grad_div | 同上，预测速度方程加入隐式散度惩罚 | 32³ Q2/Q1 FEM | alpha=1，gamma=mu |

MAC 配置 `demo/ideal_lv_fsi/configs/mac_gl2017_32.json` 只将当前优化
`mac_gl2017.json` 的 `fluid.shape` 改为 [32,32,32]。
保留 point_density=3、P2 CUDA 融合传递、质量矩阵 Graph 求解及 FGMRES 工作区。
流体间距增大后，自适应积分阶数和点数会按原有公式重新选择；这也是本次网格变化的影响。

三组均读取同一原生 `results/afsi337_reference/solid.npz`，使用相同 P2 固体、
纤维、Guccione 材料及三方向基底弹簧。dt=5e-5 s，0→2 s，40000 步。
压力/张力在 1.5 s 线性达到 150000/600000 dyn/cm²，随后保持。
这是 2 s 加载保持实验。

## 后台运行三组

在已安装本项目及 CUDA IB 的环境、仓库根目录执行：

```bash
conda activate afsi-torch
bash scripts/run_lv_grid32_comparison.sh
```

默认 GPU 0 运行 MAC；GPU 1 先运行 FEM，结束后运行 grad-div FEM。
同一卡不同时运行两组。两个队列均脱离终端，exit 后继续运行。
即使普通 FEM 失败，grad-div FEM 仍会启动。
可通过 `AFSI_MAC_GPU=0 AFSI_FEM_GPU=1` 选择两张物理 GPU。
脚本继承当前环境的编译器配置；对只在 RTX 4090 使用的环境，
可在首次编译时指定 `TORCH_CUDA_ARCH_LIST=8.9`。

也可指定一个尚不存在的总目录及已有固体输入：

```bash
AFSI_SOLID_INPUT=results/afsi337_reference/solid.npz \
  bash scripts/run_lv_grid32_comparison.sh results/lv_grid32_my_run
```

脚本在启动前检查三组共同物理参数、配置合法性及输入文件，
保存配置快照、固体 SHA256、代码提交、GPU 分配和排队顺序。
每组 log_every=200、checkpoint_every=1000、output_every=400，输出固体和流体场。

## 查看进度、耗时和结果

将 batch 替换为启动时打印的总目录：

```bash
batch=results/lv_grid32_my_run
cat "$batch"/{mac32,fem32,fem32_grad_div}/status.txt
tail -n 5 "$batch/mac32/run.log"
tail -n 5 "$batch/fem32/run.log"
cat "$batch"/{mac32,fem32,fem32_grad_div}/runtime.txt
```

`queued` 表示未开始，`running` 表示运行中，`completed` 表示成功结束；
失败会记录退出码。未开始的组尚无运行日志/耗时文件。
每组目录保存 `config.json`、`run.log`、`runtime.txt` 和 `simulation/`。
模拟结果包括 `history.csv`、`report.json`、`configuration.json`、检查点及 VTK。
MAC 的集合位于 `simulation/vtk/solid.pvd` / `fluid.pvd`；
FEM 位于 `simulation/solid.pvd` / `fluid.pvd`。
三组固体输出均已使用当前坐标，不再叠加 Warp By Vector。

两张卡并行时仍共享 CPU、内存和存储；这里的完整耗时是此调度条件下的记录。
比较结果时应继续使用共同构型、共同采样位置，并报告 MAC/FEM 的分辨率与离散差异。
