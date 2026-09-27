# 0.26.0：固定流体 CSR 算子和诊断采样

本轮按用户要求优化现有 PyTorch 实现，不安排 AFSI CPU 对照实验。参考 [torchcor 的组装](https://github.com/sagebei/torchcor/blob/main/torchcor/core/assemble.py)和 [CSR 求解路径](https://github.com/sagebei/torchcor/blob/main/torchcor/core/solver.py)，独立实现适合本项目 Q2/Q1 的固定算子组装。固体非线性力与对流项继续随当前状态更新，材料参数、载荷、IB 核、力滞后顺序和离散方程均保持原定义。

## 实现

- `CSRFluidOperators` 在网格所在设备用单元矩阵生成 COO，合并重复项后转成 CSR。保存速度质量/刚度、压力质量/刚度及梯度/散度矩阵。速度三分量共用一个标量矩阵，避免三份块对角存储。梯度和散度各自按弱式组装，不假定二者互为负转置。
- Chorin 缓存 `rho/dt*M + mu*K`。矩阵在固定网格、固定 dt/rho/mu 下复用；需要改变这些设置时重新创建求解器。迭代中使用 PyTorch 稀疏乘法，Dirichlet 提升和压力基准仍按原规则处理。启动组装与存储会增加峰值内存；没有建立稠密全局矩阵。
- `SolverOptions.check_every` 默认 API 值为 1；周期入口设为 8。迭代中的曲率、有限性和递推残差在设备上计算，每组迭代才传回主机。中途已达到递推容差的状态在设备上冻结，防止额外迭代发生 0/0。每次返回前重新计算真实自由自由度残差，仍使用 `rtol=1e-10, atol=1e-12`，不将固定迭代次数当作收敛。迭代计数包括最多 7 次冻结期间的空更新。仍有初始化、组检查和每步有效性检查所需的同步，不能称为完全无同步。
- 周期入口调用 `driver.step(..., diagnostics=False)`，关闭 IB 功率/力平衡及流体全诊断。腔容积、壁体积、能量、流速、CFL 和散度只在 history/log/checkpoint/VTK 输出或终点采样。默认这些间隔对齐后每 20 步采样一次。其他历史验证脚本保持默认全诊断接口。
- 每步真实线性残差、非有限场检查、采样构形有效性、腔体正体积和 IB 支持/位移限制仍然保留；这类失败保护没有当作可选诊断移除。报告体积、det(F)、速度/CFL 极值现在是采样极值，不能当成全部 16000 步的精确极值。线性残差、迭代次数和 IB 步长极值仍逐步累计。失败保存时补算最后已接受状态，防止报告落后于检查点。

CSR 是本次前向时间推进路径；原积分算子仍可用于现有验证和自动微分工作。没有新增包或自定义 CUDA 编译依赖。

## Linux / RTX 4090 运行

先拉取、更新安装并运行测试：

```bash
git pull --ff-only
conda activate afsi-torch
python -m pip install -e ".[test,geometry,io]"
CUDA_VISIBLE_DEVICES=0 python -m pytest -q
```

在新目录运行与上次相同空间/时间设置的完整周期，便于观察耗时变化；原结果保留：

```bash
CUDA_VISIBLE_DEVICES=0 python examples/lv_cycle.py \
  --device cuda --end-time 0.8 --dt 5e-5 \
  --mesh-size 1.2 --fluid-cells 24 --box-length 12 \
  --backend csr --check-every 8 \
  --output results/lv_cycle_csr_080
```

需要先检查启动速度时，将同一命令的终点改为 `--end-time 0.01`，然后从同目录继续至 0.8 s：

```bash
CUDA_VISIBLE_DEVICES=0 python examples/lv_cycle.py \
  --device cuda --resume results/lv_cycle_csr_080/checkpoint.npz \
  --end-time 0.8 --backend csr --check-every 8
```

兼容 0.25.0 检查点，恢复物理状态后重建 CSR；执行分段记录保存 backend/check_every 和采样设置。`--backend quadrature --check-every 1` 可选回原算子路径，但新版周期入口仍采用诊断采样。默认仍生成 VTK，若关闭 VTK 应在比较耗时时注明 I/O 设置差别。

## 已完成验证

Windows CPU：194 passed、106 CUDA skipped、2 warnings（TorchScript 弃用和 PyTorch CSR beta 提示）。新测试覆盖非立方网格全部固定算子、组合矩阵、非零边界的多步 Chorin、非线性固体 IB 轨迹、检查间隔内提前收敛、失败/真实残差，以及采样间隔内失败的报告状态。已有续算、载荷和其他回归全部通过。

另用生成左室 mesh-size=1.8 cm、6³ 流体网格、dt=5e-5 s 跑完 CPU 200 步到 0.01 s（无 VTK）。报告内耗时 8.37 s，终点腔容积 77.558550933 mL、最小 det(F)=0.999139299。[先导报告](lv-cycle-csr-cpu-pilot-0.26.0.json)。这个小网格 CPU 结果仅验证入口和运行，不能用它宣称 RTX 4090 的加速倍数。目标 GPU 全周期耗时与显存尚待用户运行。

