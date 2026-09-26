# 左室单步压力投影差异归因（0.20.0）

[0.19.0 RTX 4090 三盒报告](ib-box-projection-gpu-results-0.19.0.json)显示：固定 0.5 cm Q2 速度节点间距、1 cm IB 核宽度和同一预加载左室时，18→24 cm 盒子的 Chorin/参考 Schur 左室速度差分别只有 3.33%/4.13%，但 24 cm 同盒两种投影仍相差 21.17%。因此本轮只改变**投影的代数实现**，继续使用独立 Q2 弱式参考支持的密度载荷 `M_f H^T g/ΔV`。

`validation/diagnose_projection_gap.py` 对 18/24 cm 两盒用同一暂定流速 `u*` 构造三个结果：

1. 现有 Chorin 修正：压力刚度矩阵 `K` 求压力 `p`，速度质量矩阵 `M` 作修正。
2. 拉普拉斯重建：取 `λ_K = -(dt/ρ)p`，用精确的自由 Q2 质量逆计算 `u_K = u* - M_free⁻¹ Dᵀ λ_K`。它检验 Chorin 的弱梯度 `G`、边界处理及质量矩阵求解是否额外造成差异。
3. 参考 Schur：解 `(D M_free⁻¹ Dᵀ) λ_S = D u*`，得到 `u_S = u* - M_free⁻¹ Dᵀ λ_S`。这里只要求离散 Q1 散度约束，不把它当作完整 FSI 真解。

零外壁速度下的自由速度自由度应满足 `G p = -Dᵀ p`。脚本检查此等式、`K λ_K = D u*` 的求解残差、`u_K` 与 Chorin 的全部流体速度及左室节点速度差，并记录把 `λ_K` 代入 Schur 方程的相对残差。若重建不接近原 Chorin，脚本报错，不把差异误归因于 `K` 和 Schur 算子。每盒同样检查预加载、IB 支撑和力的合力；报告不修改生产求解器。

本地 CPU [先导报告](projection-gap-cpu-results-0.20.0.json)使用旧检查点 `81ffd2ba26b0aabec58d59c0dc45b70e6179348b77cb41e7568b044ba73a8311`。18/24 cm 的自由梯度转置等式相对差均约 `1.5e-15`，Chorin/重建左室速度相对差分别为 `2.15e-10`/`1.15e-10`，全部流体速度相对差小于 `1e-8`；但重建/Schur 左室速度差仍为 21.72%/21.17%。`λ_K` 的压力刚度方程相对残差约 `8e-11`，其 Schur 方程相对残差约 6.63%。这表明本单步试验中的主要投影差来自压力刚度算子 `K` 与实际离散 Schur 算子 `D M_free⁻¹ Dᵀ` 的不一致。CPU/GPU 检查点哈希不同，须在目标 GPU 上复核。

```bash
git pull --ff-only
conda activate afsi-torch
python -m pip install -e ".[test,geometry]"
CUDA_VISIBLE_DEVICES=0 python -m pytest -q
CUDA_VISIBLE_DEVICES=0 python validation/diagnose_projection_gap.py \
  --preload results/lv_equilibrium --device cuda --levels 18 24 \
  --output results/projection_gap/report.json
```

保留 `report.json` 与同目录 `responses.npz`。`completed=true` 表示冻结预加载左室的一次增量力诊断完成，不表示耦合时间轨迹、流体/固体联合加密或完整周期已验收。本地完整回归：**172 passed、100 CUDA skipped、1 warning**。

AFSI 的 [3D IB 耦合代码](https://github.com/loveIroha/afsi/blob/main/afsic/src/coupling/IBMesh3D.h)明确了四点核、节点映射、插值及体积缩放的力散布；这些是 PyTorch 实现的重要数值参考。此处观察到的约 21% 差异位于流体压力投影，直接把整个耦合层移植成 CUDA 不会改变该离散方程。IB 插值与散布若在性能分析中成为主要耗时，可再做局部 CUDA 扩展并与现有 PyTorch 算子逐项比对。用户提供的 [RTX 4090 投影归因报告](projection-gap-gpu-results-0.20.0.json)重现了本地数值；短程耦合对照见 [0.21.0 说明](COUPLED_PROJECTION.md)。
