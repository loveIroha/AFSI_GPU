# 预加载左室短程耦合的投影对照（0.21.0）

[RTX 4090 单步归因报告](projection-gap-gpu-results-0.20.0.json)已确认：18/24 cm 流体盒内，Chorin 的压力解经精确 Q2 质量矩阵逆重建后几乎得到同一速度，而参考 Schur 速度仍使左室节点响应相差 21.72%/21.17%。这一差异来自压力刚度算子 `K` 与离散 Schur 算子 `D M_free⁻¹ Dᵀ` 的选择。本轮检验它在**耦合时间推进**中怎样影响左室，并继续观察扩大流体盒后的变化。

`validation/compare_coupled_projection.py` 在同一已收敛的 0.2 mmHg 预加载左室上运行 Chorin 与参考 Schur 两条轨迹，每条默认 20 步、`dt=5e-5 s`，总时长 1 ms。0.02 mmHg 腔压增量按前半程保持、随后四分之一程线性升压、最后四分之一程保持的相同日程施加；固体力与速度更新沿用 [AFSI 顺序](../src/afsi_torch/coupling.py)。实验固定 IB 核物理宽度 1 cm、Q2 速度节点间距 0.5 cm、密度载荷 `M_f Hᵀg/ΔV`、左室网格和时间步。默认比较 18/24 cm 流体盒，流体盒只改变外壁距离，起始 IB 支撑物理节点和权重一致。

参考 Schur 分支先调用未修改的 Chorin 流体步骤，取得同一暂定速度；再以精确自由 Q2 质量逆和迭代 Schur 求解替换**修正后的速度**。这样两条轨迹每步使用同一个暂定方程，但由于上一时刻耦合状态不同，从第二步起暂定速度不再必然相同。Schur 分支存储的 `pressure` 是辅助 Chorin 压力，仅作为下一步 Poisson 初始猜测；它不是 Schur 乘子，也不是左室腔压，且不会反馈到固体力。参考分支额外计算未使用的 Chorin 修正，因此当前速度不是性能基准。

报告保存每步腔容积、最大增量位移、最小 `det F`、流体速度、离散散度及求解残差；终点同时比较两种投影的左室节点位移与腔容积增量，并对每种投影比较 18→24 cm 盒子差。`corrected_divergence_dual_l2` 是 Q1 弱散度约束的范数，`corrected_divergence_l2` 是积分意义的点态散度范数；前者近零不要求后者近零。相对差报告明确列出分母，微小信号时返回 `null`，不应把大比例直接解释为生理效应。

本地 CPU [12 cm、20 步先导报告](coupled-projection-cpu-pilot-0.21.0.json)全部完成。Chorin 的腔容积增量 `3.3677891e-6 mL`、最大增量位移 `9.4191631e-8 cm` 与 [0.17.0 GPU 报告](coupled-ib-gpu-results-0.17.0.json)中相同盒子、核及载荷路径的数值一致。Schur 与 Chorin 的终点左室节点位移向量差相对 Chorin 为 6.94%，腔容积增量差相对 Chorin 为 11.78%；Schur 弱散度残差低于 `5e-14`。12 cm 盒的 IB 支撑距离外壁较近，因此先导结果不能代表 18/24 cm 盒。由于本地检查点哈希与 GPU 检查点不同，正式结论需要目标 GPU 对照。本地完整回归：**175 passed、101 CUDA skipped、1 warning**。

```bash
git pull --ff-only
conda activate afsi-torch
python -m pip install -e ".[test,geometry]"
CUDA_VISIBLE_DEVICES=0 python -m pytest -q
CUDA_VISIBLE_DEVICES=0 python validation/compare_coupled_projection.py \
  --preload results/lv_equilibrium --device cuda --levels 18 24 --steps 20 \
  --output results/coupled_projection
```

请保留 `results/coupled_projection/report.json` 和各分支 `last_accepted.npz`，后者用于复核节点位移比较。实验完成不等于完整心动周期，也不构成流体/固体网格及时间步联合收敛证明。参考 Schur 分支是数值对照，生产 Chorin 和 AFSI 的原有耦合次序未改变。
