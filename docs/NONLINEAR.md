# GPU 非线性固体平衡（0.13.0）

新增真正的 Newton–GMRES 迭代：求解 P2 固体残量 R(x)=0，而不只是计算力或切线。它用于独立固体平衡与预加载研究，当前显式 IB/FEM 仍按原有顺序推进。每个流固时间步改为静力平衡会改变问题，故没有这样接入。

## 算法

残量为内力、给定主动张力、随动压力与基底弹簧力之和的负值。Newton 求 J(x) delta=-R(x)，再回溯更新 x+alpha delta。

- `torch.func.jvp` 对整个残量求精确方向导数，包括随动压力；不组装全局 Jacobian、不使用有限差分切线。
- 重启 GMRES 支持非对称切线，右预条件采用节点 3×3 Guccione/主动张力/弹簧块。压力项只从预条件块中省略，实际 JVP 保留其完整导数。预条件每轮 Newton 重建。
- 向量、JVP、块逆、Krylov 基以及小型 Hessenberg QR/三角求解都在指定设备；没有生产 CPU 线性求解回退。收敛判断读取少量标量，会同步主机。
- 线性停止条件检查原始算子的真实残量；非线性停止条件为 max(atol, rtol×初始自由残量范数)。固定自由度的反力不计入平衡残量。
- Dirichlet 值是绝对坐标，初值提升后迭代增量在固定自由度为零。左室示例通过基底弹簧约束，不额外锁定基底节点。
- Armijo 回溯检查残量下降，先验证试探构形，再计算其残量。积分点 det(F)>0 检查不等于全域无自交证明。
- 失败抛出 `NonlinearFailure` 并携带最后接受迭代；不修改输入、不静默放宽容差。GMRES 耗尽迭代、奇异系统、回溯失败不会伪报收敛。

迭代求解当前为前向求解，不支持穿过完整求解流程的自动反向传播。残量、材料和 JVP 本身仍可微。节点块预条件尚未证明具有网格无关迭代数；本次左室每轮线性迭代可达数百次，需要后续优化，不能据此宣称已有 GPU 加速比。

## 运行

```bash
git pull --ff-only
conda activate afsi-torch
python -m pip install -e ".[test,geometry]"
CUDA_VISIBLE_DEVICES=0 python -m pytest -q
CUDA_VISIBLE_DEVICES=0 python examples/nonlinear_patch.py --device cuda --output results/nonlinear_patch
CUDA_VISIBLE_DEVICES=0 python examples/lv_equilibrium.py --device cuda --mesh-size 1.8 --pressure-mmhg 0.2 --load-steps 2 --output results/lv_equilibrium
```

本地 CPU 回归为 138 passed、94 CUDA skipped，共 232 项。目标 GPU 本版尚待执行。无需新增依赖；独立 DOLFINx 参考仅在 CI 专用环境运行。

[Linux 自动验证 36147461337](https://github.com/loveIroha/AFSI_GPU/actions/runs/36147461337) 已全部通过，代码提交 `26d1c532f0d6476a45143cb55fe01db7c1f46911`。回归 138 passed、94 CUDA skipped；与独立 UFL/DOLFINx 平衡解最大坐标差 1.77e-15 cm；两个小块算例和左室两级加载均收敛。见 [完整结果与残量历程](nonlinear-results-0.13.0.json)。原固体、流体、IB 和耦合参考检查也通过。

`nonlinear_patch.py` 使用 6 个 P2 四面体、27 个节点，包含两例：

1. Guccione 仿射平衡：8% 伸长与剪切，外载由解析常量 PK1 牵引的三角形积分给出，没有直接用内部装配力的负值构造右端。解与已知仿射坐标比较。
2. 单侧随动压力：1000 dyn/cm²，一侧固定。Newton–GMRES 与独立全 Jacobian/直接求解的 Newton 比较；另以 UFL/DOLFINx 非线性残量与完整切线生成独立平衡参考。

本地仿射例 6 次 Newton 收敛，最大坐标误差约 1.56e-15 cm；随动压力例 3 次收敛，最终自由残量约 3.02e-10 dyn。

`lv_equilibrium.py` 默认使用生成的 h=1.8 cm 左室（554 节点、260 单元），Guccione 参数和基底弹簧沿用原模型；压力分两级 0.1→0.2 mmHg，主动张力为零。载荷比例不是时间，不是心动周期。局部收敛状态作为下一级初值；不做自动步长调整。

本地最终最大位移约 0.01638 cm。这个小载荷测试验证迭代流程，**不是生理舒张预加载、无应力几何反演或完整周期初始状态的验收**。改变压力、张力或网格可能需要更强预条件和继续验证。

Linux 复核中两个加载级各需 4 次 Newton；最终最小 det(F)=0.99918974，自由残量约 6.78e-8 dyn，小于 1.04e-5 dyn 的停止阈值。线性迭代为每轮 305–825 次，不能宣称预条件已经适合大规模网格。

## 输出与下一步

- 每个示例的 `report.json`：残量历程、线性迭代数、回溯步长、参数、变形诊断。
- 小块示例 `affine.npz`、`follower.npz`：参考/当前坐标与单元拓扑。
- 左室 `last_converged.npz`：最后一个收敛载荷级的状态。若某级失败，另外写 `failed_load_last_iterate.npz`，不可将其当作收敛状态。

新增求解器没有消除先前的 IB 空间敏感性。后续需要继续核验功率一致性及短程耦合；预加载状态如何与背景流体、初始压力及加载时序一致，也须专门设计。暂不将静力平衡坐标直接代入原零力启动并宣称已完成初始化。
