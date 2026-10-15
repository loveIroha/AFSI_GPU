# Griffith–Luo 2017：理想左心室 MAC/IB–FE 路线

方法来源：[Hybrid finite difference/finite element immersed boundary method](https://doi.org/10.1002/cnm.2888)，第 3 节、附录 A。
本入口对齐其统一弱形式和时间/流体求解结构。固体继续采用 AFSI337 的 P2 四面体、Guccione 本构、基底三方向弹簧和 1.5 s 线性加载，5 cm 流体盒。
这些算例条件与论文第 5.4 节的 Q1 六面体、4 cm 盒和准静态测试存在差异，因此这里是**方法对齐、AFSI337 条件的实验**。

## 方法与实现

| 部分 | 实现 |
|---|---|
| 流体空间 | 均匀 MAC：压力在单元中心，速度/力密度在面中心；二阶差分梯度、散度、黏性 |
| 对流 | `xsppm7.py`：七点重构、映射 WENO5、median 单调化、迎风面值、迎风速度重置，再计算 advective `u.grad(u)`；AB2 时间外推 |
| 黏性 | Crank–Nicolson，`H=I-dt*mu/(2*rho)*L`；无显式黏性步长限制 |
| 流体求解 | 联立 `H u + dt/rho Gp = b, D u=0`，FGMRES；压力自由投影预条件，几何多重网格 Poisson 子求解、Helmholtz 固定精度子求解 |
| IB 力 | 参考一致质量矩阵 `M F=b_s`；在当前构型积分点计算 FE 力密度，再用四点 Peskin 核传播 |
| IB 速度 | 在相同积分点插值，组装参考积分权重，解 `M U=B^T W K u`；和传播互为质量/网格体积内积下的伴随 |
| 结构时间 | 旧速度预测半步构型；半步载荷/非线性力；相同半步构型传递平均流速更新位置 |
| 首步 | 附录 A.2 两步预测校正；不以零力滞后一步替代 |
| 重启 | 保存旧对流、旧步长、半时间压力的时间戳；直接续算，不重新启动 AB2 |

具体公式：预测 `x_half=x_n+dt/2*J(x_n)u_n`；在 `x_half,t_half` 组装、投影并传播力。
正常步对流为 `1.5*N(u_n)-0.5*N(u_(n-1))`；首步先解临时 CN–Stokes，再在平均速度上计算对流。
最终 `x_(n+1)=x_n+dt*J(x_half)*(u_n+u_(n+1))/2`。
结构力在预测半步显式计算；论文该格式不要求对新时刻流固未知量做联合 Newton，也不具有无条件弹性稳定性。

FGMRES 求解整个速度/压力鞍点系统，投影只用于预条件。
既检查真实动量残差，也检查原交错场散度；估计 Krylov 残差小不能单独判为收敛。预条件的 Poisson 子求解默认相对容差 0.01；这不改变外层物理残差目标，FGMRES 允许此类不精确预条件。`stokes_preconditioner_rtol` 可配置，独立于报告中的真实方程验收容差。
内部变量对压力和连续方程作网格尺度缩放，不改变物理方程；压力零均值定标。
大向量、CSR 质量矩阵、预条件子求解和 Krylov 正交化均在张量所属 GPU 上。

### 自适应 P2 交互积分

有限元内力/本构积分保持原 AFSI337 规则；**交互积分单独自适应**。
参考网格限定为直边的仿射 P2 四面体，当前构型可以是二次弯曲单元。
每次按 P2 Bernstein 控制点凸包的直径 `d` 与最小流体间距选择
`n=max(3,ceil(point_density*d/dx_min))`，Gaussian 精度 `2*n-1`。
P2 最小阶数 3 保证参考一致质量矩阵的四次乘积精确积分。
默认 `point_density=3`；这是一项明确的保守密度策略，不声称等同于 libMesh 的逐点表或二维 3×3 网在三维四面体上的唯一推广。
默认采用紧凑的正权 Xiao–Gimbutas Gaussian 表，所选多项式精确度不变但积分点更少，需要 quadrature 安装项；阶数超过 8 时使用同精度的正权 conical Gauss。也可显式选择 conical 全部使用该表。`max_order/max_points` 为资源上限，达到上限时失败并保存检查点，不截断密度。

一份 stencil 拥有自己的积分点/权重分组，后续 prepare 不改变它；同次力和速度传递使用相同 stencil。
参考质量矩阵保持不变。默认使用 PyTorch 注册的 P2 C++/CUDA 算子融合坐标/权重准备、散播/插值与有限元积分组装，CSR/Graph-PCG 仍在 PyTorch 中求解。参考执行路径仍可显式选择，见 [P2 CUDA 执行指南](P2_CUDA_EXECUTION.md)。
`transfer_backend=reference` 表示通用 FE 积分组装，不意味着在 CPU 上运行；`execution.ib_backend=cuda` 仍调用原生 CUDA IB。

### xsPPM7 来源与边界

依据 IBAMR 的 `godunov_xsPPM7_extrapolate3d`、`WENO5_interp`、`monotonize`、`navier_stokes_reset_adv_velocity3d` 和 `advect_derivative3d` 移植。
官方说明：[INSStaggeredPPMConvectiveOperator](https://ibamr.github.io/IBAMR-docs/ibamr/classIBAMR_1_1INSStaggeredPPMConvectiveOperator.html)。
完整版权与 BSD-3-Clause 条款见 [IBAMR_COPYRIGHT](../third_party/IBAMR_COPYRIGHT)。
原 Fortran 标量重构例程作为独立测试 oracle 保存在 `tests/reference/xsppm7_oracle.f90`，仿真不调用 Fortran。
与旧 `ppm` 选项的四阶简化抛物重构不同；新预设使用 `xsppm7`。
无滑移盒使用关于实际边界位置的奇反射：法向速度有边界节点，切向速度在半格点，两者反射索引不同。
没有引入 IBAMR 的 AMR/MPI 网格设施；不声称和任意 IBAMR 输入文件的边界外推逐位一致。

## 安装、验证与运行

在仓库根目录：

```bash
conda activate afsi-torch
git pull --ff-only
python -m pip install -e ".[test,geometry,fused,cuda-ib,quadrature]"
python -m afsi_torch.mac.cuda_ib
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_gl2017_alignment.py
```

Fortran oracle 测试需要 `gfortran`；缺失时跳过该对照，运行本身不需要它。
CUDA 工具链使用已配置的 conda CUDA 环境，见 [CUDA IB 指南](CUDA_IB.md)。

```bash
CUDA_VISIBLE_DEVICES=0 python -u demo/run.py \
  demo/ideal_lv_fsi/configs/mac_gl2017.json \
  --end-time 0.005 --output results/lv_gl2017_early
```

与现有 FEM 的固体逐点比较时，额外追加 `--solid-input results/afsi337_reference/solid.npz`。
该文件可选；它保存原生 AFSI 固体、fiber/sheet 和内力积分规则，入口会保留这些系数。
`material/loads/beta/basal_constraint` 仍由本次 JSON 明确指定。

完整 2 s 后台运行（使用新目录，不覆盖旧结果）：

```bash
run_dir="results/lv_gl2017_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_dir"
CUDA_VISIBLE_DEVICES=0 nohup /usr/bin/time \
  -f 'elapsed_seconds=%e exit_code=%x' -o "$run_dir/runtime.txt" \
  python -u demo/run.py demo/ideal_lv_fsi/configs/mac_gl2017.json \
  --end-time 2 --output "$run_dir" \
  > "$run_dir/run.log" 2>&1 < /dev/null &
echo "PID=$! output=$run_dir"
```

续算：`python demo/ideal_lv_fsi/run_mac.py --device cuda --resume "$run_dir/checkpoint.npz" --end-time 2`。
旧 MAC 检查点保留其旧数值格式；不能通过给旧检查点增加一个开关将其解释为新 CN–AB2 历史。
旧方案独立配置为 `configs/mac_legacy.json`；`run_mac.py` 与 `mac_gpu.json` 的新运行默认用本方案。

## 输出与可比性

`configuration.json` 保存全部有效参数；`report.json` 记录时间/对流/Stokes 格式、自适应积分和仍保留的算例差异。
`history.csv` 增加 Stokes 迭代数和动量残差。VTK 中 `pressure_time_s` 标明 CN 半时间压力；速度和固体位置为整时间。
固体 VTU 已使用当前坐标，ParaView 不再叠加 Warp By Vector。
本配置仍为 AFSI337 的 2 s 加载保持，不是多个心动周期；结束时刻不自动等于静力平衡。
应通过步长/网格/交互积分收敛、心肌材料体积及应变等指标评估精度，不能只靠程序跑完或散度小。
新方法增加了积分点与线性代数工作，旧 MAC 的约 20 min 完整耗时不能当作本配置的实测性能。

积分表更换会改变非多项式 IB 核的采样，因此多项式精确度相同不代表耦合误差完全相同。验证记录：[2026-10-15](validation/griffith_luo2017_20261015.json)。

执行优化、相同检查点性能对照、参考后端和续算切换说明：[P2 CUDA 执行指南](P2_CUDA_EXECUTION.md)。
