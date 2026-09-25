# 流体/IB 网格敏感性定位（0.12.0）

本轮完成固定几何、固定载荷的单步对照，不运行完整周期，不替换生产 IB、Chorin 或时间推进。替代方法只在 `validation/diagnose_ib.py` 中，分开检查核宽度、载荷路径和投影的影响。

## 设置与运行

- 同一个 h=1.2 cm 的 P2 左室网格、同一纤维与节点编号，位置固定为参考构形。
- 在载荷斜坡 t=0.001 s 取一次固体节点力并固定，流体初速度为零。不包含原时间推进的零力启动和载荷滞后，也不推进固体位置。
- dt=2.5e-5 s，rho=1 g/cm³，mu=1 g/(cm·s)，背景为 12³ cm³，零外边界速度。
- 流体每轴 6、12、18 个 Q2 单元；速度格点间距为 1、0.5、1/3 cm。
- 原核 epsilon=h_grid，支撑半径 2 epsilon；固定宽度核 epsilon=1 cm，支撑半径始终 2 cm。
- 选择整数倍加密，使 epsilon/h_grid=1、2、3。按子格点求和保持常数和一次矩恒等式，不重新归一化、不截断边界。

```bash
git pull --ff-only
conda activate afsi-torch
python -m pip install -e ".[test,geometry]"
CUDA_VISIBLE_DEVICES=0 python -m pytest -q
CUDA_VISIBLE_DEVICES=0 python validation/diagnose_ib.py --device cuda --output results/ib_diagnosis
```

完整测试共 214 项。本地 CPU 为 127 passed、87 CUDA skipped。0.11.0 的用户 GPU 结果为 201 passed，不能代替新增 CUDA 测试。

可选绘图：

```bash
python -m pip install -e ".[io]"
python scripts/plot_ib_diagnosis.py --source results/ib_diagnosis/diagnosis.json --output results/ib_diagnosis/diagnosis.png
```

默认 12 组配置 = 3 档网格 × 2 种核 × 2 种载荷路径；最粗档两种核相同，作一致性对照。每组记录预测、Chorin 和 Schur 三个响应。`--levels 6 12` 可缩短运行，但不能替代三档研究。固定宽度核邻居数为 (4m)³，增加显存开销；诊断实现尚未优化性能。

## 数学对照的含义

令 H 为 IB 插值矩阵，g 为积分后的固体节点力，dV 为速度格点体积，M 为流体一致质量矩阵。

- 生产路径 `density`：f=Hᵀg/dV，再将 M f 加入弱方程，保持当前选定 AFSI 示例的语义。
- 诊断路径 `dual`：直接将 b=Hᵀg 加入弱方程，满足 uᵀb=(Hu)ᵀg。

一般 M 不等于 dV I，两条路径不是同一种离散。格点功率守恒不等于一致有限元内积下的功率守恒。对照不能被解释为已证明 AFSI 原方程错误，也不能直接换成 `dual` 而仍声称是原方案。

Chorin 用压力 Laplace 算子校正，通常不会严格满足 D u=0。Schur 参考采用 S=D M_free⁻¹Dᵀ，求 S lambda=D u* 后令 u=u*−M_free⁻¹Dᵀlambda。外边界速度固定为零，检查包含压力基准位置在内的完整散度残量。lambda 是投影乘子，不作为生理压力输出。

Schur 在容差内满足 Q1 弱无散约束，不保证每个积分点 div(u)=0。M_free⁻¹ 利用盒子 Q2 质量矩阵的张量积结构，在当前 CPU/GPU 上做三个小型一维 Cholesky 求解，不组装稠密三维全局矩阵、不回退 CPU 求解。另以小网格独立稠密约束求解、质量逆作用、梯度消除和正交性测试核验。

## 本地与 Linux CI CPU 实测

[Linux 自动验证 36143192861](https://github.com/loveIroha/AFSI_GPU/actions/runs/36143192861) 已全部通过：127 passed、87 CUDA skipped，12 组诊断完成，结果与本地一致。代码提交 `f96a3951c16c5abae24b80a8ad179f42e34cb26b`；[完整数值报告](ib-diagnosis-results-0.12.0.json) 包含绝对/相对差、求解残量、代码指纹和参数。节点速度数组与结果图在该次 CI 验证附件中。本版 GPU 仍待执行。

下表比较 12³→18³，同一固体节点速度向量差除以较细结果的节点欧氏范数；它不是腔体体积误差，也不是体积加权 L2 误差。

| 核与载荷路径 | 压力校正前 | Chorin 后 | Schur 后 |
| --- | ---: | ---: | ---: |
| 原核 + density | 36.82% | 55.83% | 65.26% |
| 固定物理宽度 + density | 0.116% | 27.82% | 18.27% |
| 原核 + dual | 35.62% | 38.29% | 38.84% |
| 固定物理宽度 + dual | 3.64% | 3.00% | 4.07% |

1. **核宽度影响该冻结探针的网格敏感性。** 固定宽度后 density 预测速度差明显减小。原来只加密流体也改变了平滑尺度，不能当作同一正则化问题的纯网格收敛研究。
2. **单独换投影不足以解决问题。** 原核 + density 使用 Schur 后仍相差 65.26%。Chorin 留下的弱散度约为预测值的 4.3%–11.0%，Schur 后约为 1e-12；更好的弱无散不自动等于固体速度网格无关。
3. **载荷路径与投影有明显相互影响。** 固定宽度 + dual 趋势较好，但属于另一种离散，不能仅因差异较小就替换生产方案。固定宽度 + density 校正后速度比预测速度小得多，相对误差也会放大；须结合绝对响应、腔体体积变化率和速度数组判读。

固定宽度 + density 的 Chorin 腔体体积变化率在 6³、12³、18³ 上为约 0.20829、0.07950、0.05213 mL/s；固定宽度 + dual 为约 0.82156、0.78547、0.81046 mL/s。两者响应本身也不同，不能只比较百分比而忽略求解对象已变。

最大常数矩误差约 4.4e-16，一次矩/仿射插值误差约 2.7e-15 cm；本探针未发现邻居索引或漏掉支撑造成的常数/一次矩破坏。

这些是冻结单步响应的证据，尚不能把上一轮轨迹约 40% 的体积响应差完全归因于某一项，也未证明固体网格、局部无散或全耦合解收敛。

## 输出与后续

`diagnosis.json` 保存设置、代码指纹、各组残量、核矩、三种响应和相邻网格比较；`solid_probe.npz` 保存同一几何、力和体积导数；每组 `.npz` 保存三种固体节点速度。失败保留此前结果并报错，不暗中放宽容差。完成标志只表示实验完成，`full_cycle_ready` 仍为 false。

后续应做功率一致性与网格相位的独立制造解对照，并在短程耦合中复核趋势；再进行固体/流体联合加密，避免核变窄却一直用同一粗固体节点集表示载荷。若采用 dual 或新投影，须作为新算法分支验收，保留原 AFSI 路径对照。固定 epsilon=1 cm 也不代表物理精度足够，最终还需研究 epsilon 趋小的联合极限。
