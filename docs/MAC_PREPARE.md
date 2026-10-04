# 自适应 P1 FE/IB 模板准备

保留用户 H–O 本构、CN–AB2/隐式中点弹性、Anderson/CSR Newton、原积分阶数
选择和正权规则、一致质量、Peskin 四点核及验收容差，仅优化模板准备执行。
该接口属于通用的自适应 P1/MAC transfer，不依赖真实左心室坐标或材料参数。

## 为什么继续优化准备阶段

RTX 4090 的同检查点短段报告中，共享模板加压力热启动为 674.03 ms/步。
独立后续剖析中 `ib_prepare` 仍为 216.55 ms/步，高于 Stokes 的 182.01 ms/步。
这说明准备是剩余的重要开销；这些时间来自启动附近，不代表完整周期。

原执行链：

1. 每个积分规则组计算 P1 积分点坐标。
2. 拼接完整点云并检查支撑范围。
3. 广播面/中心坐标、距离数组，计算 Peskin 权重。
4. 写入共享邻居基址与权重表。

新 `prepare_backend='triton'` 将这些操作合并为每规则组一次 CUDA 核调用。
每个积分点/轴线程直接读取四个 P1 节点及同一形函数表，生成面和中心模板。
组单元数、总点数与写入偏移是运行时参数，避免它们随变形变化而反复编译。
4,438,962 个积分点时，不再分配约 106.5 MB 的完整坐标点云，也避免相关拼接
和广播临时数组。最终模板仍为同样的 int64 基址和 float64 权重；后续传播、
插值、质量求解没有变化。模板返回独立张量，不覆写仍被非线性/切线使用的旧模板。

## 同一 Peskin 核的平方根共享

记一维坐标为 s，原基址为 `floor(s-1)`，令 `f=s-floor(s-1)-1`。
四个相邻权重中的根号项均可表示为：

```text
R = sqrt(1 + 4*f - 4*f*f)
w0 = (3 - 2*f - R)/8
w1 = (3 - 2*f + R)/8
w2 = (1 + 2*f + R)/8
w3 = (1 + 2*f - R)/8
```

因此每个轴/格点位置只需一次平方根。除以 8 用精确的二进制系数 0.125 相乘。
这仍是原 Peskin 四点核，没有改成近似多项式、降低精度、归一化权重或减少积分点。
关闭 FMA，使用双精度运算；离线 sm89 编译检查确认平方根/除法为
`sqrt.rn.f64`/`div.rn.f64`。代数重排和 FE 累加仍可能产生微小浮点差异。

所有实际积分点仍检查原来的两网格支撑距离与有限值；核写入错误标记，主机验收
后才能传播/插值。仍检查变形单元尺寸并更新积分阶数，不冻结或降低积分密度。

## GPU 测试和两组性能对照

在仓库根目录运行：

```bash
git pull --ff-only
conda activate afsi-torch
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_mac_prepare.py
CUDA_VISIBLE_DEVICES=0 python -u validation/benchmark_real_lv_schemes.py \
  --checkpoint results/real_lv_adaptive_early/checkpoint.npz \
  --schemes cnab-semiimplicit --nonlinear-solvers anderson-newton \
  --execution-variants shared-warm prepare-warm \
  --device cuda --warmup 5 --steps 20 --profile \
  --output results/real_lv_prepare_performance/report.json
```

两组都启用共享模板、同一步压力热启动、相同 XG 积分规则和验收容差。
`shared-warm` 保留原 PyTorch 准备；`prepare-warm` 只替换准备执行。
检查点只读。报告比较总耗时和终态字段差异，并记录实际准备后端、积分点、
压力/质量调用数、迭代数。

剖析额外拆分 `ib_rule_selection`、`ib_point_coordinates`、`ib_table_generation`、
`ib_direct_prepare`。未调用的阶段不会出现在结果中；它们嵌套在 `ib_prepare` 内。
剖析是测量完成后继续推进的独立窗口，不能把它的阶段时间直接与测量总耗时相加。

默认配置仍使用原 `torch` 准备；实际 CUDA 正确性与加速幅度需要这次实验确认。
CPU 会明确回退至原 PyTorch 路径，报告 `prepare_execution='torch'`，
不能用 CPU 回退计时声称 GPU 内核速度。

## Demo 和配置接口

确认目标 GPU 结果后，新运行可以加入：

```bash
--ib-stencil-backend shared --ib-prepare-backend triton --stokes-warm-start
```

或设置：

```python
InteractionQuadratureOptions(
    mode='adaptive', rule_family='xiao-gimbutas',
    transfer_backend='reference', stencil_backend='shared',
    prepare_backend='triton')
```

直接准备要求共享模板及自适应 P1 参考传播路径；不与实验性 fused/cell 传播组合。
新检查点保存准备后端；普通续算恢复配置，不能用命令行覆盖该字段。
旧检查点缺少 `prepare_backend` 时仍使用 `torch`。

测试覆盖多积分规则组、尾部掩码、非等距网格/非零原点、整数/半整数附近的位置、
非有限值/越界、冻结模板、传播/插值、功率、主动阶段耦合验收与检查点/命令行。
无 GPU 开发机的 CPU 回归、Triton 解释器和离线编译不代替真实 CUDA 与完整周期验证。
