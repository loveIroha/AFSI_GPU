# 固定格距的流体边界与投影对照（0.19.0）

0.18.0 的 [RTX 4090 报告](ib-weak-reference-gpu-results-0.18.0.json)验证了固定 1 cm 核宽度下的 Q2 弱式载荷：`M_f H^T g/ΔV` 对独立积分参考的误差随网格加密从 10.24% 降至 0.40%，而直接 `H^T g` 的误差保持在 45%–51%。同一预加载左室、同一 0.5 cm 流体速度节点间距下，12 cm→18 cm 流体盒的 Chorin 节点速度差为 16.67%；同盒 Chorin/参考 Schur 差为 25.64%/21.72%。GPU 检查点哈希为 `f1b21532517fc0c5da41326c31a873fbfc329793a98d368223e7904f3d74fc09`。

本轮只研究这一**物理参考更一致的密度载荷**。`validation/compare_ib_box_projection.py` 对 12/18/24 cm 三个立方流体盒运行相同的单步增量力试验。每轴 Q2 流体单元数分别为 12/18/24，速度节点间距恒为 0.5 cm，中心和 IB 网格相位相同；源为同一预加载左室在 0.02 mmHg 腔压增量下的节点力，时间步为 `5e-5 s`，初始流体速度为零，核宽度固定 1 cm。脚本检查每盒 IB 支撑的**物理节点坐标及权重**一致，并记录支撑距外壁的最小距离、密度载荷合力和求解残差。

每盒先用同一个 Chorin 暂定速度，再将它分别作 Chorin 修正和 `D M_free^{-1} D^T` 参考 Schur 投影。对 12→18、18→24 和 12→24，报告两个投影下左室全部节点速度向量的绝对差及相对小盒/大盒范数的比例。`adjacent_change_ratio < 1` 表示本次单步响应的相邻盒子差减小，不能单独证明边界收敛；应同时查看 18→24 的相对差和 Chorin/Schur 差。参考 Schur 只严格满足离散 Q1 散度约束，不能据此视为完整 FSI 的真解。原 0.18.0 报告的 16.67% 使用小盒范数作分母，可与本报告 `chorin.12_to_18.relative_to_small` 对照。

24 cm 的 Chorin 压力求解在默认 1000 次迭代上限内未达到原定容差。本诊断把 Chorin 和参考 Schur 的迭代上限设为 4000、真残差重算间隔设为 200；**相对/绝对容差及离散方程均不变**。设置与实际残差比例写入报告，生产求解器默认值不受影响。

本地 CPU 的 [三盒先导报告](ib-box-projection-cpu-results-0.19.0.json)使用旧检查点 `81ffd2ba26b0aabec58d59c0dc45b70e6179348b77cb41e7568b044ba73a8311`：

| 左室节点速度向量差（相对小盒范数） | 12→18 cm | 18→24 cm |
| --- | ---: | ---: |
| Chorin | 16.67% | 3.33% |
| 参考 Schur | 22.07% | 4.13% |

同盒 Chorin/Schur 差相对 Schur 范数在 12/18/24 cm 分别为 25.64%/21.72%/21.17%。相邻盒子差明显下降，但 24 cm 仍有投影差；这只描述冻结增量力的单步响应。完整本地回归：**170 passed、99 CUDA skipped、1 warning**。

```bash
git pull --ff-only
conda activate afsi-torch
python -m pip install -e ".[test,geometry]"
CUDA_VISIBLE_DEVICES=0 python -m pytest -q
CUDA_VISIBLE_DEVICES=0 python validation/compare_ib_box_projection.py \
  --preload results/lv_equilibrium --device cuda --levels 12 18 24 \
  --output results/ib_box_projection/report.json
```

请保留 `report.json` 和同目录 `responses.npz`。前者包含对照指标及检查点哈希，后者保存每盒 Chorin/Schur 的原始左室节点速度，便于复核向量差。报告 `completed=true` 仅表示单步诊断完成，`full_cycle_ready=false`；没有进行耦合时间轨迹、时间步或固体网格收敛验证。用户提供的 [RTX 4090 三盒报告](ib-box-projection-gpu-results-0.19.0.json)使用原 GPU 预加载检查点，与 CPU 先导的各项盒子和投影差异在显示精度内一致；后续压力投影归因见 [0.20.0 实验](PROJECTION_GAP.md)。
