# 减少 FE/IB 的显存读写与重复检查

这是执行层优化：保持当前 CN–AB2/PPM、非线性中点弹性、积分点和权重、
一致 CSR 质量矩阵、共享 Peskin 模板、Stokes 算法与原验收容差。
新路径支持 3D affine P1 自适应积分，与材料模型无关。

## 共享模板融合

`--ib-transfer-backend fused --ib-stencil-backend shared` 使用
`mac/_triton_shared_fe.py`，并兼容 `--ib-prepare-backend triton`。

传播在一个核内读取单元节点系数、计算 FE 点值、乘参考积分权重，随后向
MAC 网格传播；插值在核内收集 64 个邻点，直接乘 FE 形函数与权重并组装
节点 RHS。两条路径均不写出完整的 `(interaction_points,3)` 点数组。
每个非线性评估仍各做一次一致质量矩阵求解。

444 万点的 FP64 三分量数组约为 107 MB；省去它的一次写出和读回约为
213 MB 逻辑显存流量。典型一步三次传播、四次插值时，约减少 1.5 GB 的
这种中间数组流量。该估算不包含原有网格访问、原子累加、缓存命中或
质量矩阵流量，不是硬件带宽实测，也不能直接换算成整体加速倍数。

组大小、总点数和分组偏移使用运行时参数，不按每次自适应成员变化重新
特化内核。积分阶数仍特化；首次使用新的阶数会编译。高阶规则使用固定
小 tile 覆盖全部点，不裁剪积分密度，也不退回完整点速度数组。
传播仍使用原子累加，因此数值一致性使用严格浮点容差验证，不要求位级相同。

CPU 使用独立的展开模板参考实现；CPU 耗时不代表此 CUDA 内核的性能。
`cell` 单元支撑域归约路径仍使用 component 模板，未混入本次新路径。

## 检查结果复用

`--reuse-validation` 对一个非线性中点问题保存已经完全通过检查的中点、
端点、几何结果与私有试探快照，避免验证后算力时再次计算相同几何，
并对最终完全相同的克隆解复用已有检查。缓存同时校验：

- 试探张量身份和版本，或最终解与私有快照的精确相等；
- 预测位置、旧状态坐标的身份和版本；
- 已检查中点/端点的版本，以及形状、设备、dtype。

新试探失败后立即清除资格；无版本计数的 inference 张量不复用。
准备后的网格、参考几何和有效性规则在驱动生命周期内保持不变；替换这些
规则应重建驱动。扩展固体可选提供 `checked_geometry(x)`；不提供时仍可
复用完整状态验证，但无法复用执行器几何。

最终非线性残差、Stokes 动量/散度、CFL、节点力有限性与端点有效性仍按
原条件验收。没有降低检查频率或容差。Anderson 还复用刚计算出的残差范数，
减少对同一未修改残差的重复归约和 host 同步。
报告新增 `validation_evaluations` 与 `validation_reuses`；这些是完整状态
验证次数及缓存命中数，不等同于全部内核或同步次数。

## 真机验证与分项对照

本地已完成 CPU 等价性、失败/修改缓存失效、Triton 解释器及 sm89 离线编译
检查。GPU 内核启动、寄存器/原子竞争及实际加速需要在目标机器实测。
原默认执行路径和旧检查点默认保留 `reuse_validation=false`。

当前三周期进程不会自动启用新增选项。请在它结束后进行短段对照，避免同一
GPU 的两个模拟互相影响计时；也可使用更早已有的完整检查点。

```bash
git pull --ff-only
conda activate afsi-torch
CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/test_shared_fe_fusion.py tests/test_midpoint_solver.py

CUDA_VISIBLE_DEVICES=0 python -u validation/benchmark_real_lv_schemes.py \
  --checkpoint /实际三周期输出目录/checkpoint.npz \
  --schemes cnab-semiimplicit --nonlinear-solvers anderson-newton \
  --execution-variants prepare-warm shared-fused-warm shared-fused-checked \
  --device cuda --warmup 5 --steps 20 --profile \
  --output results/real_lv_shared_fe_performance/report.json
```

三组依次是原共享模板/Triton 准备、融合传递、融合传递加检查复用。
它们使用相同初始检查点、积分规则和容差；报告记录耗时、嵌套求解计数、
状态误差与额外短段的阶段计时。输入检查点只读，无需重新生成网格。
确认求解次数、残差和状态一致性后才能解释速度差异；缓存命中不减少必要
的 Stokes 或质量矩阵求解。

## 新运行启用

在之前三周期命令中把 `--ib-transfer-backend reference` 改成 `fused`，
并增加 `--reuse-validation`。其余物理与数值参数保持原值，例如短段验证：

```bash
CUDA_VISIBLE_DEVICES=0 python -u demo/real_lv_fsi/run_mac.py \
  --mesh-dir /mnt/large2/gjh/realistic_left_ventricle \
  --device cuda --dt 1e-4 --fluid-cells 128 --end-time 0.005 \
  --coupling cnab-semiimplicit --nonlinear-solver anderson-newton \
  --interaction-quadrature adaptive --ib-rule-family xiao-gimbutas \
  --ib-transfer-backend fused --ib-stencil-backend shared \
  --ib-prepare-backend triton --stokes-warm-start --reuse-validation \
  --output results/real_lv_shared_fe_early
```

新选项写入配置与检查点。普通 resume 恢复其原执行设置，不能在原输出目录
覆盖这些选项；比较优化应使用上述只读 benchmark 或独立新运行。
