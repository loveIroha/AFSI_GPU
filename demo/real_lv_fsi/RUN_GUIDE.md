# 真实左心室：纯舒张与主动收缩运行指南

[全部 demo](../../docs/DEMO_GUIDE.md) · [CUDA IB 安装](../../docs/CUDA_IB.md) · [网格读取](../../docs/MESH_INPUT.md)

入口 `demo/real_lv_fsi/run_mac.py` 使用 Ma 等（2024）被动充盈问题启发的 BE–BE 实现，并提供用户定义的周期主动应力拓展。当前入口不再运行历史 CNAB/RK3 配置。以下所有命令在仓库根目录执行。

## 推荐：JSON 直接运行

```bash
python demo/run.py demo/real_lv_fsi/configs/diastole_cuda.json
python demo/run.py demo/real_lv_fsi/configs/active_cuda.json
```

每行对应一个独立实验。运行继承环境启动文件选择的 GPU，默认使用原生 CUDA IB；可追加 `--end-time 0.005` 短跑。统一入口打印独立输出目录，也支持 `--output`。

## 1. 输入与安装

```bash
conda activate afsi-torch
python -m pip install -e ".[test,mesh,fused,quadrature,cuda-ib]"
mesh_dir="/mnt/large2/gjh/realistic_left_ventricle"
```

该目录需要：

```text
mesh_scale.xdmf       # 及 XDMF 引用的 mesh_scale.h5，保留相对路径
mesh_scale.h5
boundaries.xml
fibers_0.xml  fibers_1.xml  fibers_2.xml
sheets_0.xml  sheets_1.xml  sheets_2.xml
```

默认长度单位 cm，源边界 1=外膜、2=内膜、3=基底。fiber/sheet 为按原四面体单元顺序读取的 DG0 三分量；不自动将节点 P1 数据当成 DG0。文件名不同可通过 JSON 的 `fiber_files`/`sheet_files` 修改。患者网格不随仓库发布。已用网格为 26889 顶点、135430 四面体，固体位移使用 P1，不升为 P2。

选择自定义 CUDA IB 前完成 [CUDA IB 指南](../../docs/CUDA_IB.md) 的编译。已经在相同环境编译并通过测试，可直接运行下面的命令。

## 2. 空间离散、本构与边界

固体采用三维 P1 四面体，参考有限元质量矩阵以一致 CSR 形式保存。H–O 应力通过单元变形梯度、DG0 纤维和 sheet 计算，边界弱式组装随形内膜压力和基底约束。默认固体积分配置 degree=5；IB 相互作用积分另行自适应，不能混为本构积分。

当前被动律使用原始 `I1=tr(FᵀF)`，基质应力为 `a*exp(b*(I1-3))*(F-F^(-T))`，纤维/sheet 项仅在各自 I4>1 时激活，包含纤维-sheet 剪切项及 `2*kappa*ln(J)*F^(-T)`。这对应代码的论文 eq81/82 路径，**不是早期 UFL 的等容 I1 版本**。完整公式见 [paper_lv.py](../../src/afsi_torch/paper_lv.py)。

| 参数 | 无 config：论文参数预设 | active_cycle.json：用户参数 |
| --- | --- | --- |
| a, b | 2362, 10.81 | 2400, 5.08 |
| a_f, b_f | 200370, 14.154 | 14600, 4.15 |
| a_s, b_s | 37245, 5.1645 | 8700, 1.6 |
| a_fs, b_fs | 4108, 11.3 | 3000, 1.3 |
| kappa | 5e6 | 5e6 |
| 基底 beta | 5e6 | 5e6 |

a 类与 kappa 的单位为 dyn/cm²，b 类无量纲；基底 beta 为 dyn/cm³。基底保留绕 `(x,y)=(7.5,7.5) cm` 的径向自由度，惩罚切向和 z 向运动。这是用户指定、论文真实 LV 部分没有明确给出的复现差异，换几何时应重新配置基底中心。

流体采用 13×13×13 cm 的规则 MAC 盒，原点 (0,0,0)，默认 128³ 单元；速度位于面、压力位于单元中心。rho=1 g/cm³、mu=0.04 g/(cm·s)，即 4 cP。外盒速度扩散为齐次 Neumann，投影压力为零 Dirichlet，允许通量穿过外盒。预设心内膜压力是固体边界载荷，不等于每个流体单元的压力值。

## 3. 时间格式、IB 与非线性求解

固定 `dt=1e-4 s`，一阶 BE–BE：每步冻结旧几何 `x_n` 的插值/传播算子与半拉格朗日出发速度，使用新构形 `x_(n+1)`、新时刻载荷计算固体力。每次流体响应先解 BE 黏性 Helmholtz，再解 Dirichlet 压力 Poisson 并投影。Helmholtz 使用带残差检查的迭代平滑，压力采用几何多重网格；GPU 预设用工作区与 CUDA Graph。

耦合未知量是节点增量 `y`，求解 `R(y)=y-dt*I(x_n,u_new(y))=0`。默认 `jfnk` 用有限差分 Jacobian 作用及 BiCGSTAB；下面运行命令用 `anderson-newton`，先 Anderson，必要时转为带线搜索的 Newton–GMRES。Newton 组装固体 CSR 切线，耦合切线作用仍包含 IB 与流体响应，不形成全局流固块矩阵。`inexact` 根据非线性残差调整线性求解目标，最终非线性接受标准仍保留。

IB 使用四点 Peskin 核及自适应四面体积分，按变形后单元最大边长与流体最小间距选择阶数，密度默认 2。1–8 阶使用实现支持的规则（这里选 Xiao–Gimbutas），高阶使用正权重 conical 回退；阶数上限 22、积分点预算 1200 万是资源上限，不表示每个单元都用 22 阶。构建后在该步非线性迭代内复用。

记 `B[a,i]=sum_q w_q*N_a(q)*delta_h(x_q-x_i)`（这里 delta_h 表示实现中的无量纲核权重），传播为 `Bᵀ M_s^(-1) f / cell_volume`，插值为 `M_s^(-1) B u`。IB 支撑要求位于外盒内部；外盒边界面在流体功率诊断中使用半体积权重，内部 IB 力为零的边界不会改变上述功率配对。两方向使用相同 B。质量方程采用 GPU PCG。选择 `cached-hash/cuda` 加速 B 的积分与哈希构建，不改本构、积分选择或时间残差。

论文的无条件能量稳定性结论不自动覆盖这里带外压、主动应力、基底弹簧及投影实现的所有轨迹，也不保证任意 dt 都准确或 Newton 必收敛。半拉格朗日追踪/插值、固定步长、约束和求解器差异均应在复现说明中记录。

## 4. 两种载荷协议

**纯舒张 `inflation`：** 0–0.8 s 线性升至 8 mmHg，之后保持，主动张力为零。跑至 1.6 s 是升压后保持，不是重复两次加载。无 `--config` 时采用上表论文参数；若需用户材料的纯舒张，传 `active_cycle.json` 并显式覆盖 `--load-protocol inflation`。

**`active-cycle`：** 0.8 s 重复周期，`tau=t mod 0.8`。压力在 0–0.2 s 由零线性升至 1.067 kPa，0.2–0.5 s 保持；0.5–0.65 s 按 `1.067+13.46*(1-exp(-(tau-0.5)^2/0.004))` 上升，之后用 `0.8-tau` 对称下降。周期边界按原公式重置，压力存在跳变，不额外平滑。

主动张力在 0–0.5 s 为零；上升段为 `84.26*(1-exp(-(tau-0.5)^2/0.005)) kPa`，下降段使用 `0.8-tau`。乘 10000 转 dyn/cm²。加入第一 Piola 应力：

```text
P_active = T(t) * [1 + 4.9*(lambda_f - 1)] * (F*f0) outer f0
lambda_f = |F*f0|
```

系数未截断；强缩短下可能为负，日志会记录。这是用户主动扩展，不是论文 V.F 的被动基准。两周期完成不自动表示周期稳态。

## 5. CUDA IB 短运行

```bash
python -u demo/real_lv_fsi/run_mac.py \
  --config demo/real_lv_fsi/active_cycle.json --mesh-dir "$mesh_dir" \
  --device cuda --dt 1e-4 --end-time 0.005 \
  --nonlinear-solver anderson-newton --anderson-policy legacy \
  --anderson-max-iterations 6 --linear-policy inexact --newton-preconditioner none \
  --ib-response-backend csr --ib-csr-assembly-backend cached-hash \
  --ib-csr-contraction-backend cuda --ib-max-order 22 --ib-max-points 12000000 \
  --output results/real_lv_cuda_short
```

## 6. 两个主动周期后台运行

```bash
run_dir="results/real_lv_active_cuda_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_dir"
nohup /usr/bin/time \
  -f 'elapsed_seconds=%e exit_code=%x' -o "$run_dir/runtime.txt" \
  python -u demo/real_lv_fsi/run_mac.py \
  --config demo/real_lv_fsi/active_cycle.json --mesh-dir "$mesh_dir" \
  --device cuda --dt 1e-4 --cycles 2 --fluid-cells 128 \
  --nonlinear-solver anderson-newton --anderson-policy legacy \
  --anderson-max-iterations 6 --linear-policy inexact --newton-preconditioner none \
  --ib-response-backend csr --ib-csr-assembly-backend cached-hash \
  --ib-csr-contraction-backend cuda --ib-max-order 22 --ib-max-points 12000000 \
  --log-every 100 --output-every 1000 --checkpoint-every 1000 \
  --output "$run_dir/simulation" \
  > "$run_dir/run.log" 2>&1 < /dev/null &
echo $! > "$run_dir/launcher.pid"
echo "$run_dir"
```

共 16000 步，周期 0.8 s，总长 1.6 s。`--cycles` 不与 `--end-time` 同传。

## 7. 纯舒张保持至 1.6 s 后台运行

下面明确使用**用户材料参数**，方便与上述主动算例比较。若要论文参数预设，删除 `--config demo/real_lv_fsi/active_cycle.json`，保留 `--load-protocol inflation`。

```bash
run_dir="results/real_lv_diastole_cuda_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_dir"
nohup /usr/bin/time \
  -f 'elapsed_seconds=%e exit_code=%x' -o "$run_dir/runtime.txt" \
  python -u demo/real_lv_fsi/run_mac.py \
  --config demo/real_lv_fsi/active_cycle.json --mesh-dir "$mesh_dir" \
  --load-protocol inflation --device cuda --dt 1e-4 --end-time 1.6 --fluid-cells 128 \
  --nonlinear-solver anderson-newton --anderson-policy legacy \
  --anderson-max-iterations 6 --linear-policy inexact --newton-preconditioner none \
  --ib-response-backend csr --ib-csr-assembly-backend cached-hash \
  --ib-csr-contraction-backend cuda --ib-max-order 22 --ib-max-points 12000000 \
  --log-every 100 --output-every 1000 --checkpoint-every 1000 \
  --output "$run_dir/simulation" \
  > "$run_dir/run.log" 2>&1 < /dev/null &
echo $! > "$run_dir/launcher.pid"
echo "$run_dir"
```

同一卡建议顺序运行；两张卡同时运行需分别指定 `CUDA_VISIBLE_DEVICES=0/1`，各进程仍是单 GPU。日志变量 `run_dir` 每次都会改为新实验目录，记录打印的路径。

## 8. 续算、输出、性能

```bash
# 将 run_dir 指向实际实验目录；checkpoint 时刻须小于 1.6 s
python -u demo/real_lv_fsi/run_mac.py \
  --device cuda --resume "$run_dir/simulation/checkpoint.npz" --end-time 1.6 \
  --ib-response-backend csr --ib-csr-assembly-backend cached-hash \
  --ib-csr-contraction-backend cuda
```

续算自动恢复材料、网格、dt、载荷及输出设置，不再传 `--config`、`--mesh-dir` 或 `--cycles`；只能使用入口允许的求解器/资源覆盖项。缺失或已完成的检查点不能用于未完成时段的续算。

`simulation/` 内保存 `configuration.json`、`report.json`、`history.csv`、`checkpoint.npz` 及 `vtk/solid.pvd`、`vtk/fluid.pvd`。每 1000 步一帧对应 0.1 s，可用 `--output-every 200` 输出每 0.02 s，增加磁盘与写出开销。`--no-vtk` 禁用场输出。日志包括体积、minJ、压力/主动张力、迭代与流体响应次数；外层 `runtime.txt` 统计整个进程墙钟时间。

正式算例不启用 profiler。性能对照命令、源码位置和等价性解释见 [CUDA IB 指南](../../docs/CUDA_IB.md)。扩展使用 PyTorch CUDA 张量与同一流执行，CSR 质量求解与有限元主体仍在 PyTorch/GPU 中。
