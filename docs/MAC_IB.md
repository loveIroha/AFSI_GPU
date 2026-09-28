# MAC、几何多重网格与有限元固体耦合

这是独立的实验后端：PyTorch 三维 MAC 流体、几何多重网格压力求解、
Griffith–Luo unified weak form 的积分点传递。现有 `lv_afsi337.py` 保留为 FEM 流体基线。
新入口为 `examples/lv_mac.py`。本阶段完成数值单元检查和短程耦合，不宣称已验证完整
2 s 收缩过程、网格收敛或实际 GPU 加速比。

## 参考核对

1. [MAC-taichi, commit 30fe9c0](https://github.com/houkensjtu/MAC-taichi/tree/30fe9c0535cc7ec97b05f21ef903c76cdba65d5f)
   的 `mac.py`：二维 MAC、中心动量通量、显式黏性、预测与压力修正。
   压力仅执行固定 30 次迭代，**没有多重网格，也没有残量收敛判据**。
   `mac-benchmark.py` 只测试速度预测内核，不能当完整 NS 性能数据。
   本项目借鉴离散结构，使用 PyTorch 重写三维算子，不引入 Taichi 运行时。
   来源授权见 [MAC-taichi notice](MAC_TAICHI_LICENSE.txt)。
2. Griffith & Luo (2017), [Hybrid finite difference/finite element immersed boundary method](https://doi.org/10.1002/cnm.2888)，
   第 3.2–3.3 节：一致质量矩阵、积分点传递、功率伴随关系。
   本实现采用 unified force 投影，不是文中的 partitioned surface-force 方案。
3. [Meng 等 GPU 无矩阵 FAS 多重网格论文](https://arxiv.org/abs/2510.11152)：
   参考 GPU 网格模板运算及多重网格思路。本实现是线性 Poisson 几何 V-cycle，
   不是该论文的完整 FAS 或 X-MCGS 复现。

## 流体离散

张量按 `(x,y,z)` 排列：

| 场 | 形状 | 所在位置 |
|---|---|---|
| p | `(nx,ny,nz)` | 单元中心 |
| u | `(nx+1,ny,nz)` | x 面中心 |
| v | `(nx,ny+1,nz)` | y 面中心 |
| w | `(nx,ny,nz+1)` | z 面中心 |

三种速度不共享同一个节点坐标数组。闭盒外边界法向速度为零，切向速度通过奇反射
ghost 值满足壁面零速度；压力采用齐次 Neumann 条件与零均值规范。

压力算子严格由 `A=-D G` 定义。求解 `A p=-(rho/dt) D u_star` 后，
`u_new=u_star-(dt/rho) G p`。因此压力线性残量直接控制离散散度。
不通过固定若干次迭代假定流体已不可压缩。

多重网格使用二倍粗化、体积平均限制、三线性延拓、加权 Jacobi 前后平滑。
最粗层通过带常数模态约束的小矩阵逆作用求解；逆在设置时构造，运行时保留在选择的设备上。
每隔若干 V-cycle 核对真实残量。非零净通量 RHS 会报错，不能靠减去任意非零均值掩盖边界错误。

首版时间推进为一阶显式对流/黏性预测与压力投影。它与 AFSI 的隐式黏性不同。
启用了黏性时间步及低网格 Reynolds 数保护；这是初始低 Reynolds 数算例后端，
不能作为任意高 Reynolds 数流动的通用稳定算法。默认 `h=5/64 cm`、`dt=5e-5 s`、
`rho=mu=1` 的黏性数为 0.024576，小于当前保守限值 0.25。

64³ MAC 的速度标量自由度为 798720、压力单元为 262144；原 32³ Q2/Q1 的速度自由度
为 823875、压力节点为 35937。选择 64³ 是为了与原速度节点间距 0.078125 cm 接近，
不表示二者精度完全相同，也不能用更粗 32³ MAC 的耗时冒充等精度加速比。

## 耦合推导与单位

令 `b` 为固体弱式已经组装的节点力（dyn），`F` 为参考体积上的力密度系数
（dyn/cm³），`B` 为节点到单元积分点的形函数矩阵，`W` 为参考构形物理积分权重
（cm³），`M=B^T W B` 为一致固体质量矩阵。

对每个速度分量分别构造 Peskin 四点核矩阵 `K_c`。它采用该分量面中心的真实坐标，
权重是三个一维 phi 的乘积，不含额外 `h^-3`。在完整支撑下：

```
M F_c = b_c
f_c = (1/cell_volume) K_c^T W B F_c

q_c = K_c u_c
M U_c = B^T W q_c
```

现有 Guccione、主动张力、内膜随动压力和基底弹簧组装产生的总 `b` 一起进入该投影。
压力/表面载荷在 unified formulation 下通过体积 FE 空间表示，没有再次单独散布表面力。

在质量矩阵求解误差范围内，满足：

```
sum_c U_c^T b_c = cell_volume * sum_c u_c^T f_c
```

因此结构输出的功率等于流体接收的功率。两个方向必须使用同一构形、同一批积分点、
同一核权重。`W` 是参考体积，只乘一次，不再乘 `det(F)`；MAC 动量方程直接使用
`f/rho`，也不乘旧的 Q2 流体质量矩阵。

一致质量矩阵在参考构形上预组装为 CSR，力密度与节点速度的投影由 GPU PCG 求解。
**不能对当前 P2 质量矩阵直接做行和集总**：四面体顶点的行和可以为负。

支撑进入外壁会显式报错，不裁剪、不重归一化，避免悄悄破坏合力和功率关系。
固体保持 AFSI 已核对的滞后时间顺序：用旧力推进流体，在旧位置插值更新固体，
再按旧时间采样载荷组装新力。这不等同于论文附录的完整二阶时间积分方案。

## 积分点密度与剩余验证

默认使用模型的 14 点 degree-4 四面体规则。可用 `--interaction-degree 4` 改为
64 点 Duffy 规则，`--interaction-degree 6` 为 125 点规则，固体本构积分规则不随之改变。
这用于独立检查传递积分精度。增加积分点会增加 IB 时间和显存。

本阶段使用固定规则，**尚未实现论文按变形动态加密交互点的策略**。
守恒测试通过不意味着界面已经不漏流；高拉伸或固体网格较粗时，仍须检查交互点密度、
壁体积、腔体积与流体分辨率。不能仅凭压力散度很小宣称 FSI 体积守恒。

## 验证与运行

新增 `tests/test_mac.py` 覆盖独立标量 Neumann 算子、分部积分、制造压力解、
投影散度和动能、黏性特征模态、中心对流通量、梯度体力、
曲面 P2/共享节点的合力、力矩、功率、常速度再现，以及 LV 断点续跑。
测试同时参数化 CPU/CUDA；CUDA 不可用时明确跳过。

在 Linux 项目目录运行：

```bash
git pull --ff-only
conda activate afsi-torch
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_mac.py
CUDA_VISIBLE_DEVICES=0 python -u examples/lv_mac.py \
  --device cuda --end-time 0.005 --output results/lv_mac_smoke
```

默认仍生成 mesh-size=0.1 cm 的固体，使用 64³ MAC 网格，100 个真实时间步。
报告包含多重网格残量/周期数、质量矩阵残量、功率差、散度、体积和 detF。
CPU 开发冒烟可用 `--mesh-size .4 --fluid-cells 16 --end-time .001`，其结果
不是原分辨率的物理对照。

每 200 步及结束保存 `checkpoint.npz`；`report.json`、`history.csv` 记录汇总。
当前输出为数值结果，尚未接入原 FEM 流体的 VTK 写出器。
仅能续跑本后端的 checkpoint，不能直接加载 FEM 流体状态：

```bash
CUDA_VISIBLE_DEVICES=0 python -u examples/lv_mac.py \
  --device cuda --resume results/lv_mac_smoke/checkpoint.npz --end-time .01
```

先核对目标 GPU 短程结果与传递积分精度，再决定完整 2 s 实验；当前不提供已经获得加速比的结论。
