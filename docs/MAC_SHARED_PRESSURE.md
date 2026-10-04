# 共享 MAC 模板与同一步压力热启动

这两项优化针对真实左心室自适应 P1 FE/IB 的执行开销。保留 CN–AB2 流体、隐式
中点弹性、Anderson/CSR Newton、一致质量矩阵、原多重网格与所有验收容差。
用户 H–O 本构、主动应力、载荷、边界、网格、时间步和 IB 积分规则均不改变。

## 选择依据

已有 RTX 4090 短段报告中，compact 为 770.373 ms/步，fused 为 835.012 ms/步，
cell 为 1108.190 ms/步。后两者没有加速，因此本次继续使用 compact 的原传播/插值核。
compact 的独立后续剖析中，模板准备约 223 ms/步，Stokes 约 271 ms/步；
压力求解包含在 Stokes 内，不能把这些嵌套阶段直接相加。
这些数据来自启动附近，不代表整个心动周期的性能。

## 共享一维模板

三个 MAC 速度分量分别沿自身轴位于面上、沿其他轴位于中心。
原来每个分量保存三个轴的模板；现在每个轴只保存面和中心两套模板。
CUDA 核直接选择对应轴表，避免在传播/插值前重新展开。

| 单个积分点的存储 | 分量模板 | 共享模板 |
| --- | --- | --- |
| int64 邻居基址 | 9 个 | 6 个 |
| float64 一维权重 | 36 个 | 24 个 |
| 总字节数 | 360 | 240 |

4,438,962 个积分点时，每个模板对象约从 1.598 GB 降为 1.065 GB，减少三分之一。
这是模板存储量，不是整个程序的显存；一个时间步可能同时保留多个模板。
未降低索引/权重精度，未削减积分点，未改变 Peskin 核、求和或质量求解。
CPU 验证路径会展开模板，优化主要面向 CUDA。

目前共享模板支持 `mode=adaptive`、`transfer_backend=reference`；
不能与实验性的 `fused`/`cell` 传播后端组合。

## 同一步 Stokes 压力热启动

每个非线性时间步第一次残差求解仍使用上一时刻压力；后续非线性残差使用本步
上一次成功求解的压力作为初值。缓存使用独立副本，防止图工作区或试探覆盖。
如果热启动求解抛出求解异常，仅重试一次原初值；仍按真实动量和散度验收。
失败求解不会更新缓存。

CSR 切线/GMRES 的线性响应继续使用原来的线性求解初值，不使用非线性压力缓存。
非线性方程与最终验收条件不改变，但浮点迭代路径可能产生容差内的差异。
压力复制也有成本，能否提速取决于减少了多少压力修正调用与 MG 周期。
该选项只用于 `cnab-semiimplicit`。

## 四组独立性能对照

默认仍是分量模板和原压力初值。使用同一检查点、相同规则和求解器分开测量两项收益：

```bash
git pull --ff-only
conda activate afsi-torch
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_mac_shared_pressure.py
CUDA_VISIBLE_DEVICES=0 python -u validation/benchmark_real_lv_schemes.py \
  --checkpoint results/real_lv_adaptive_early/checkpoint.npz \
  --schemes cnab-semiimplicit --nonlinear-solvers anderson-newton \
  --execution-variants compact shared pressure-warm shared-warm \
  --device cuda --warmup 5 --steps 20 --profile \
  --output results/real_lv_shared_warm_performance/report.json
```

- `compact`：原分量模板、原压力初值。
- `shared`：仅共享模板。
- `pressure-warm`：仅同一步压力热启动。
- `shared-warm`：两项同时启用。

报告记录 `stencil_storage_bytes`、压力调用/MG 周期、热启动次数/回退次数和终态
位移、速度、压力、力的相对差异。总步耗时排除预热编译和独立后续剖析。
检查点只读；四组从相同保存状态开始。

## 在新 demo 中启用

确认 GPU 结果和耗时后，新运行可加：

```bash
--ib-stencil-backend shared --stokes-warm-start
```

Python 配置字段：

```python
interaction_quadrature=InteractionQuadratureOptions(
    mode='adaptive', rule_family='xiao-gimbutas',
    transfer_backend='reference', stencil_backend='shared')
coupling=MACCouplingOptions(
    scheme='cnab-semiimplicit', semiimplicit_solver='anderson-newton',
    stokes_warm_start=True)
```

新检查点保存这些设置；普通续算恢复原设置。旧检查点缺少字段时仍恢复
`component`/`False`，不会自动改变执行路径。正常续算不能用这两个 CLI 开关覆盖配置；
独立对照使用上面的只读 benchmark。

## 验证范围

测试覆盖非等距网格、非零原点、旧/新模板字段、力/力矩/功率、压力失败回退、
缓存所有权、非线性验收、CSR 切线差分和检查点往返。
无 CUDA 的开发环境可做 CPU 等价性检查与 Triton 解释器/离线编译检查；
这些检查不能代替 GPU 测试，也不证明完整三周期稳定或具体加速倍数。
