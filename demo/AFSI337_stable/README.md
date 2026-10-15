# AFSI337_stable

理想左心室 IB/FEM 算例：固体 P2 四面体、流体 Q2/Q1 六面体，
在 AFSI demo_337 对齐版本的预测速度方程中加入隐式 grad-div 稳定项。
全部有限元组装、CSR 运算、PCG 求解和 CUDA IB 传递在选定 GPU 上进行。
Gmsh 网格生成、配置读取、收敛决策及结果写出由 CPU 执行。

## 数值格式

在原预测速度方程左侧增加

\[
\gamma\int_{\Omega_f}\operatorname{div}u^*\operatorname{div}v\,dx,
\qquad \gamma=\alpha_{gd}\mu.
\]

`fluid.grad_div_alpha` 是无量纲系数，默认 **1.0**；
`gamma` 的 CGS 单位为 g/(cm·s)，与动力黏度相同。
预测速度矩阵为 `rho/dt * M + mu * K + gamma * B_gd`。
`B_gd[(a,i),(b,j)] = integral(d_i N_a * d_j N_b)`，保留全部分量交叉项。
固定网格上一次组装九个 CSR 块；块共享标量速度矩阵的稀疏索引，
逐块组装以限制临时 COO 内存。每步使用 GPU 稀疏矩阵乘法和 PCG，
Jacobi 预条件器包含准确的、随速度分量变化的 grad-div 对角线。

对流仍为显式 advective form；黏性和 grad-div 隐式。
压力泊松、原点压力规范和一致质量矩阵速度修正沿用 Chorin 分步法。
IB 为 AFSI 节点 Peskin4 的 density path，主后端为 PyTorch C++/CUDA。
固体仍采用 Guccione、原三方向基底弹簧和显式滞后力更新。
此改动不保证显式对流/弹性/IB 耦合无条件稳定，也不保证最终速度严格无散。
`gamma * ||div(u_star)||²` 才是本预测步对应的稳定耗散；最终速度的同类指标不等于该步耗散。

默认参数与原 demo 一致：5 cm 立方域，32³ 流体单元，rho=mu=1，dt=5e-5 s；
固体网格尺寸 0.1 cm，Guccione C=20000、bf=8、bt=2、bfs=4、kappa=5e5；
基底三方向弹簧 beta=5e5。压力与主动应力分别线性升至 150000、600000 dyn/cm²，
1.5 s 达峰，保持至 2 s。长轴为 x。有限弹簧允许基底位移。

这是带额外稳定化的实验，不能再标为与原 AFSI 完全相同离散。
建议分别比较 alpha=0、0.1、1，检查最终速度散度、腔室与壁体积、J 范围、
流速/位移、PCG 次数与实际耗时；默认 1.0 尚不是经完整轨迹验证的最优系数。
参考：[grad-div 与投影方法](https://www.sciencedirect.com/science/article/abs/pii/S0045782514000784)。

## 安装与短算例

从仓库根目录运行；环境及扩展编译参见 [CUDA IB 指南](../../docs/CUDA_IB.md)。

```bash
conda activate afsi-torch
python -m pip install -e ".[test,geometry,cuda-ib]"
python -m afsi_torch.mac.cuda_ib

CUDA_VISIBLE_DEVICES=0 python -u demo/AFSI337_stable/run.py \
  --config demo/AFSI337_stable/config.json \
  --device cuda --end-time 0.005 \
  --output results/AFSI337_stable_early
```

JSON 可设置流体网格、物性、固体材料、时间步长和输出间隔，
显式命令行参数覆盖 JSON。生成几何的配置不依赖 Docker 或外部网格。
如需严格复用原生 AFSI 网格/纤维，在 JSON 的 `solid_input` 中填写导出 NPZ 路径，
或添加 `--solid-input results/afsi337_reference/solid.npz`，参见
[原生数据对齐](../../docs/AFSI337_FEM_ALIGNMENT.md)。

## 完整 2 s 后台运行

```bash
conda activate afsi-torch
out="results/AFSI337_stable_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$out"
CUDA_VISIBLE_DEVICES=0 nohup /usr/bin/time \
  -f 'elapsed_seconds=%e exit_code=%x' -o "$out/runtime.txt" \
  python -u demo/AFSI337_stable/run.py \
  --config demo/AFSI337_stable/config.json \
  --device cuda --output "$out" \
  > "$out/run.log" 2>&1 < /dev/null &
echo "PID=$! output=$out"
```

`tail -f "$out/run.log"` 查看进度；结束后读取 `runtime.txt`。
`history.csv` 保存散度、腔室/壁体积、J 范围、流速和求解信息，
`report.json` 与 `configuration.json` 记录实际 alpha/gamma、算法及对齐差异。
`solid.pvd` / `fluid.pvd` 用于 ParaView；固体坐标已为当前构型，不要再次 Warp。
检查点保存同一 alpha，续算自动恢复；不允许在续算命令中改 alpha。

```bash
CUDA_VISIBLE_DEVICES=0 python -u demo/AFSI337_stable/run.py \
  --device cuda --resume "$out/checkpoint.npz" --end-time 2
```

`--grad-div-alpha 0` 可运行无稳定项对照，此时不组装 grad-div 块。
旧检查点缺少系数时按 0 处理，不因使用 stable 入口而自动改变算法。

## 小规模验证

```bash
python -m pytest -q tests/test_grad_div.py tests/test_afsi337_stable.py
```

测试覆盖 GPU/CPU 的 CSR 与积分形式、对称性/半正定性、分量耦合、准确对角线、
非零边界下独立稠密预测步参考解、零系数兼容性、JSON/CLI、检查点及续算一致性。
稠密矩阵仅用于小网格测试，实际 demo 使用稀疏 GPU 求解。
