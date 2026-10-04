# 真实左心室 H–O/P1 + MAC/IB 算例

读取外部真实网格与 fiber/sheet，使用 PyTorch GPU 有限元固体、MAC 有限差分
流体和积分点 IB 耦合。默认采用 `cnab-semiimplicit`：保留 CN 黏性、AB2 对流和
预测中点 IB 几何，同时通过 CSR 固体切线求解非线性中点力反馈。外层未知量仅为
固体节点位置修正，流体仍使用完整 MAC 网格；运动学、动量和散度均验收。
默认使用 Anderson 加速求解同一隐式残差；困难步回退到 CSR Newton。可用
`--nonlinear-solver newton` 对照。日志分别记录 AA、Newton、GMRES 和 Stokes 调用数。
这是 CN–AB2/半步格式的隐式弹性扩展，不声称逐项复制 Gao、Griffith–Luo 或 IBAMR。
[公式、验收条件与实现范围](../../docs/MAC_SEMIIMPLICIT.md)。

新运行默认采用按变形后单元尺寸选择的自适应 Gaussian IB 积分点，参考 Gao/IBTK
的积分阶数准则，保留用户本构、主动应力、载荷和边界参数。
默认用正权 Xiao–Gimbutas 规则减少积分点；收敛点完全一致时复用已计算的耦合响应，
继续验收残差、动量和散度。旧自适应检查点保留原 conical 规则。
[积分点选择、PyTorch/CSR 实现与适用范围](../../docs/GAO_FE_ALIGNMENT.md)。
旧检查点仍恢复原积分规则；可用 `--interaction-quadrature fixed` 做独立初态对照。

这是不同于生成理想左心室 `demo_337` 的外部网格算例。
患者数据不随仓库发布；完整 GPU 三周期轨迹仍需实际运行验证。

## 默认物理与离散设置

| 参数 | 设置 |
| --- | --- |
| 单位 | cm–g–s；应力 dyn/cm² |
| 固体 | P1 四面体，输入示例 26,889 顶点、135,430 单元 |
| 方向 | 原始 DG0 fiber 和 sheet，不归一化、不重排单元 |
| H–O 参数 | a=2244.87, b=1.6215, a_f=24267, b_f=1.8268, a_s=5562.38, b_s=0.7746, a_fs=3905.16, b_fs=1.695 |
| 体积/基底惩罚 | kappa=5e6，beta=5e6 |
| 基底约束 | 以 (7.5,7.5) 为参考径向中心，允许 xy 径向移动，惩罚切向和 z 位移 |
| 边界源标签 | 外膜=1，内膜=2，基底=3；内部映射为内膜=1、外膜=2、基底=3 |
| 流体 | 128³，15×15×15 cm³，原点 (0,0,0)，rho=mu=1 |
| 时间 | dt=1e-4 s，周期 0.8 s，3 周期至 2.4 s，共 24,000 步 |
| 积分 | 固体 degree=5；一致质量 degree=2；IB 按变形单元/流体间距选择 Gaussian 阶数，密度参数 2 |
| 执行 | fused 张量核、优化 IB、CSR 一致质量/CUDA Graph、Triton/CUDA Graph 压力多重网格 |
| 输出 | 日志每 100 步，检查点每 1000 步，VTK 每 200 步（0.02 s） |

H–O 对应用户 UFL：只对 I1 做等容修正；I4f/I4s 不小于 1，I8fs 不做等容修正；
体积能为 `kappa*(ln J)^2`；主动应力为
`T*(1+4.9*(sqrt(I4f)-1))*F*(f⊗f)`。未启用注释中的 PN 消除项。
内膜加载随动压力，外膜自然边界，没有额外外膜弹簧或封盖压力。
质量项由一致 FE 质量求解处理，不重复加入节点力。

周期压力和张力保留给定 C++ 分段表达式，包括周期末压力由约 1.067 kPa
重置到 0 的跳变。半步力使用半步载荷；CSV 的载荷列表示当前端点时刻。
有限体积惩罚不等同于严格不可压约束，也没有增加多孔渗流未知量。

## 输入与安装

在主机的 `AFSI_GPU` 仓库根目录运行；不需要 AFSI Docker/FEniCS 环境。

```bash
git pull --ff-only
conda activate afsi-torch
python -m pip install -e ".[test,geometry,mesh,fused,quadrature]"
```

默认目录 `/mnt/large2/gjh/realistic_left_ventricle` 包含：

```text
mesh_scale.xdmf
mesh_scale.h5
boundaries.xml
fibers_0.xml  fibers_1.xml  fibers_2.xml
sheets_0.xml  sheets_1.xml  sheets_2.xml
```

sheet 文件名不同可在 `CONFIG.sheet_files` 或 JSON 中修改。
读取接口支持 XDMF/HDF5 和 DOLFIN XML，见[网格输入说明](../../docs/MESH_INPUT.md)。

## GPU 验证与完整后台运行

只运行本次改动的检查：

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_mac_compact_quadrature.py
```

先统计真实网格初始所需积分点；此命令不启动流体模拟：

```bash
python -u validation/inspect_real_lv_interaction.py \
  --mesh-dir /mnt/large2/gjh/realistic_left_ventricle \
  --output results/real_lv_adaptive_plan.json
```

确认 `within_budget=true` 后运行短段：

```bash
CUDA_VISIBLE_DEVICES=0 python -u demo/real_lv_fsi/run_mac.py \
  --device cuda --interaction-quadrature adaptive --dt 1e-4 \
  --ib-rule-family xiao-gimbutas \
  --end-time 0.005 --output results/real_lv_compact_early
```

先完成新路径的 0.005 s 启动和预热性能实验，再推进至 0.20 s，跨过此前 0.1685 s
失败位置；通过后继续收缩阶段，再跑三个周期。具体分阶段命令见
[验证指南](../../docs/MAC_SEMIIMPLICIT.md#gpu-verification-and-staged-experiment)。

以下是从未变形初态推进三个周期的后台命令模板；完整 GPU 三周期仍待验证，
不要把模板本身当作已经验证通过。使用新输出目录：

```bash
run_dir="results/real_lv_semiimplicit_3cycles"
mkdir -p "$run_dir"

CUDA_VISIBLE_DEVICES=0 nohup /usr/bin/time \
  -f 'elapsed_seconds=%e exit_code=%x' -o "$run_dir/runtime.txt" \
  python -u demo/real_lv_fsi/run_mac.py \
  --mesh-dir /mnt/large2/gjh/realistic_left_ventricle \
  --device cuda --coupling cnab-semiimplicit --dt 1e-4 \
  --fluid-cells 128 --cycles 3 --output "$run_dir" \
  > "$run_dir/run.log" 2>&1 < /dev/null &

echo "PID=$! output=$run_dir"
```

退出终端后仍继续运行。`tail -f results/real_lv_semiimplicit_3cycles/run.log` 查看进度。
完成后 `runtime.txt` 写入墙钟耗时/退出码。
如需先检查新方案能否跨过此前 0.169 s 的失败位置，将 `--cycles 3` 改为
`--end-time 0.2`，并使用独立新目录；其余设置保持一致。

日志记录真实动量残差、散度、Stokes 修正次数和 MG 周期数。
CN 黏性不再使用显式 D/A 停止条件，弹性力在中点隐式反馈；显式对流仍检查 CFL。
日志还记录非线性残差和迭代数。隐式力反馈不代表完整 FSI 无条件稳定。
`dt=1e-4` 是本次指定的时间步，不是三个周期必然稳定或已收敛的证明。

## 保存与续算

```bash
CUDA_VISIBLE_DEVICES=0 python -u demo/real_lv_fsi/run_mac.py \
  --device cuda --resume results/real_lv_semiimplicit_3cycles/checkpoint.npz --cycles 3
```

普通续算使用原目录，恢复原配置、x/u/p、端点力与 AB2 上一时刻对流项。
旧检查点会恢复旧的 Newton 求解器；切换为 `--nonlinear-solver anderson-newton`
须给出新输出目录，保留原检查点和 AB2 历史。短程同方程耗时对比见
[性能实验命令](../../docs/MAC_SEMIIMPLICIT.md#compare-nonlinear-solver-cost-on-the-same-existing-checkpoint)。
检查点自包含，无需再访问原始网格文件。三周期表示总目标时刻 2.4 s。
异常保存最后接受状态；失败试探不会写入 AB2 历史。
切换方案必须使用新目录，并重置多步历史、执行启动步；这不消除已有历史误差。
旧 `explicit-lagged`、`explicit-rk3`、`implicit-newton` 和显式力 `cnab-midpoint`
保留用于对照；默认新运行使用 `cnab-semiimplicit`。旧检查点续算保留其原方案，完整新实验应从头运行上述命令。
自适应积分目前支持 P1 的两种 CNAB 方案；新运行使用其他时间方案时须同时指定
`--interaction-quadrature fixed`。切换积分规则应从独立初态实验开始。

## 配置、结果与性能

主程序顶部 `CONFIG` → `--config` JSON → 显式命令行覆盖。

```bash
python demo/real_lv_fsi/run_mac.py --write-config real_lv.json
CUDA_VISIBLE_DEVICES=0 python demo/real_lv_fsi/run_mac.py \
  --config real_lv.json --output results/real_lv_custom
```

流体网格/域、dt、材料、积分阶数、求解器和输出间隔均可设置。
`coupling.cnab.advection` 默认为 `ppm`；`centered` 可用于对照。
本实现的 PPM 为单调抛物重构配合 AB2，不逐项复制 IBAMR 的特征追踪算法。

- `configuration.json`：实际参数；`history.csv`：体积、J、载荷和求解诊断。
- `report.json`：真实残差、迭代数和失败时局部力分量/速度峰值诊断。
- `checkpoint.npz`：最后接受状态和 AB2 历史；`recovery_checkpoint.npz`：最近一次定期快照，失败保存不覆盖它。快照仍需检查是否适合续算。
- `vtk/solid.pvd`：固体位移、节点力、fiber/sheet、J。
- `vtk/fluid.pvd`：端点速度与半步压力（压力时刻为显示时间减 dt/2）。

腔体积使用内膜加虚拟开口三角扇；封口只用于测量，不加载力。
运行完成不代表已达到周期稳态或已建立网格收敛性。
无需新增长周期计时，可从已有检查点测量预热后的推进效率：

```bash
CUDA_VISIBLE_DEVICES=0 python -u validation/benchmark_real_lv_schemes.py \
  --checkpoint results/real_lv_semiimplicit_3cycles/checkpoint.npz \
  --schemes cnab-semiimplicit cnab-midpoint --device cuda --warmup 3 --steps 20 \
  --output results/real_lv_semiimplicit_performance/report.json
```

非线性中点修正位于 `mac/semiimplicit.py`，CSR 切线位于 `ho_tangent.py`；
流体 CN/AB2 位于 `mac/cnab.py`，PPM 位于 `mac/ppm.py`；
`real_lv.py`/`holzapfel_ogden.py` 管理固体与载荷，
`simulation/real_lv_mac.py` 管理输出与运行，`real_lv_checkpoint.py` 管理续算。

用已有的短段检查点分别测量两项改动，无需重跑网格生成或长周期：

```bash
CUDA_VISIBLE_DEVICES=0 python -u validation/benchmark_real_lv_schemes.py \
  --checkpoint results/real_lv_adaptive_early/checkpoint.npz \
  --schemes cnab-semiimplicit --nonlinear-solvers anderson-newton \
  --execution-variants baseline reuse compact \
  --device cuda --warmup 5 --steps 20 \
  --output results/real_lv_compact_performance/report.json
```

三组分别为原积分/重算、原积分/复用、紧凑积分/复用。报告记录点数、耗时、
求解次数及最终状态差异，原检查点保持只读。紧凑积分保持多项式精度，但改变
IB 核采样位置；其完整轨迹仍需验证，不能把点数减半当作整体速度翻倍。
