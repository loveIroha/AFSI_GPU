# 理想左心室：MAC 与 FEM 完整运行指南

[全部 demo](../../docs/DEMO_GUIDE.md) · [AFSI337 对齐范围](../../docs/AFSI337_ALIGNMENT.md)

本目录两个入口共用生成的椭球壳 P2 固体、Guccione 本构和 AFSI `demo_337` 加载。Gmsh 在 CPU 生成几何，随后在 GPU 执行有限元力、IB 和流体计算。无需外部心脏网格。以下命令均在仓库根目录执行。

## 推荐：JSON 直接运行

```bash
python demo/run.py demo/ideal_lv_fsi/configs/mac_gpu.json
python demo/run.py demo/ideal_lv_fsi/configs/fem.json
```

每行对应一个独立实验。运行继承环境启动文件选择的 GPU，默认使用原生 CUDA IB；可追加 `--end-time 0.005` 短跑。统一入口打印独立输出目录，也支持 `--output`。

## 参数与边界

| 参数 | 值 |
| --- | --- |
| 内/外椭球半轴 | (0.7,0.7,1.7) / (1,1,2) cm |
| 底面截取、中心、长轴 | 0.5 cm；(3.5,2.5,2.5) cm；x |
| 固体 mesh_size | 0.1 cm，P2 四面体；实际数量以日志为准 |
| 流体盒 | 原点 (0,0,0)，5×5×5 cm |
| 密度、动力黏度 | 1 g/cm³、1 g/(cm·s) |
| Guccione C、bf、bt、bfs、kappa | 20000 dyn/cm²、8、2、4、500000 dyn/cm² |
| 基底约束、beta | `basal_constraint="spring"`；500000 dyn/cm³；三个方向均施加位移恢复力 |
| 压力/主动张力 | 同时从 0 线性升至 150000 / 600000 dyn/cm²；1.5 s 后保持 |
| 时间范围 | dt=5e-5 s，0→2 s，40000 步 |

纤维由 Laplace 跨壁坐标与椭球螺旋角程序生成，内外膜角度为 +90°/−90°。心内膜承受随形压力，外膜自由；外部流体盒无滑移。生成的网格/纤维并不保证与 AFSI 外部文件逐项相同。

**基底采用 AFSI demo_337 的原三方向弹簧。** 在参考边界积分点计算位移 `u=x-X`，边界节点力为 `-∫N beta u dA0`，弹簧能为 `beta/2 ∫|u|² dA0`。三个方向均产生恢复力；有限刚度不等于强制固定位移，所以不能据此保证基底完全不动。

两份 GPU JSON 以及两个入口的 `CONFIG` 均选择 spring、beta=5e5。真实 LV 的径向约束保持独立设置。如需另作径向试验，可在理想 LV 的 JSON 设置 `basal_constraint="radial"`、相应 `beta`；径向模式的参考方向取 `geometry.center` 与 `geometry.long_axis`，允许径向位移而惩罚轴向和切向位移。

参考、fused 和 pointwise 固体力路径均调用同一个基底力实现。`history.csv` 的 `spring_energy_erg` 记录弹簧能；可选 radial 模式另外记录 `max_basal_constraint_cm`（轴向/切向约束误差）与 `basal_constraint_energy_erg`。

### ParaView 朝向与基底位移

几何先在局部 z 方向构造长轴，再作 `(x_local,y_local,z_local) -> (z_local,x_local,y_local)` 的正旋转和平移，得到世界坐标长轴 x。中心为 (3.5,2.5,2.5) cm，基底位于 x=4 cm，心尖朝负 x。网格、纤维及基底力使用同一坐标系；VTK 直接输出当前世界坐标。因此横向显示是配置的朝向，不能把画面向下直接等同于长轴位移。查看坐标轴和 displacement_cm 的 x/y/z 分量再判断运动方向。仅希望竖直显示时调整相机，或对所有显示数据统一使用 ParaView Transform 旋转；不要因此单独修改受力方向。配套纤维生成当前要求 long_axis='x'，直接改为 z 不能作为此 demo 的完整旋转方案。

## 离散与求解

**固体共用部分：** P2 Lagrange 四面体，通过体积分点计算变形梯度、非线性第一 Piola 应力及内力，组装节点力；边界积分组装压力与基底力。这两个 demo 不在每步联合 Newton 求解流体与固体。

**MAC 入口 `run_mac.py`：** 均匀交错网格，压力在单元中心、速度在面中心；中心差分对流和黏性项采用显式一阶预测，压力 Poisson 用几何多重网格，随后梯度修正速度。封闭盒压力为 Neumann 条件并处理常数零空间。IB 使用有限元积分点、四点 Peskin 核、参考一致质量矩阵 CSR 及 PCG。先传播已存固体力，推进流体，插值新速度，再作 `x_new=x_old+dt*U`，计算下一步固体力；外载取旧步时刻。首步存储力为零。这是滞后力的显式耦合。

**FEM 入口 `run_fem.py`：** 流体六面体采用 Q2 速度/Q1 压力。Chorin 分三步：解 `(rho/dt*M+mu*K)u*=rho/dt*M*u-rho*N(u)+Mf`；压力 Laplace 求解；一致质量矩阵速度修正。对流显式、黏性隐式，线性系统用对角预条件 PCG 和 CSR 算子。速度边界为 Dirichlet，压力 Neumann 加固定一个自由度定标。该压力 Laplace 分裂不等同于精确离散 Schur 投影。

FEM 入口通过规则速度节点上的 Peskin 核作节点插值/传播：已积分固体节点力除以格点体积得到密度，再由流体质量矩阵形成载荷。它不是 MAC 的积分点/一致固体质量矩阵传递；不能直接将其格点功率恒等式认作流体 FE 质量内积下的功率恒等式。固体位置与载荷采样顺序仍是显式滞后更新。

因此 `32³` FEM 元素和 `64³` MAC 单元不是相同自由度、投影或耦合方法。比较耗时必须同时记录这些差异。

## 安装与短运行

```bash
conda activate afsi-torch
python -m pip install -e ".[test,geometry,fused,cuda-ib]"

python -u demo/ideal_lv_fsi/run_mac.py \
  --config demo/ideal_lv_fsi/configs/mac_gpu.json \
  --device cuda --end-time 0.005 --output results/ideal_lv_mac_short

python -u demo/ideal_lv_fsi/run_fem.py \
  --config demo/ideal_lv_fsi/configs/fem.json \
  --device cuda --end-time 0.005 --output results/ideal_lv_fem_short
```

两个入口都使用 `--ib-backend cuda`（GPU JSON 已设置）。MAC 的 P2 积分点传播/插值使用紧凑 C++/CUDA 内核，FEM 的节点传播/插值使用同一扩展中的 indexed 内核；均保留各自原离散。MAC 配置还启用 fused、pointwise、质量矩阵/压力 Graph、optimized coupling 和热启动；FEM 配置启用 CSR 流体。旧后端通过 `--ib-backend reference` 对照。真实 LV 的 `--ib-csr-contraction-backend` 是另一条自适应 P1 组装路径的参数。

## 完整后台运行

MAC：

```bash
run_dir="results/ideal_lv_mac_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_dir"
nohup /usr/bin/time \
  -f 'elapsed_seconds=%e exit_code=%x' -o "$run_dir/runtime.txt" \
  python -u demo/ideal_lv_fsi/run_mac.py \
  --config demo/ideal_lv_fsi/configs/mac_gpu.json \
  --device cuda --end-time 2 --output "$run_dir/simulation" \
  > "$run_dir/run.log" 2>&1 < /dev/null &
echo $! > "$run_dir/launcher.pid"
echo "$run_dir"
```

FEM（同一卡建议等上一个结束再启动）：

```bash
run_dir="results/ideal_lv_fem_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_dir"
nohup /usr/bin/time \
  -f 'elapsed_seconds=%e exit_code=%x' -o "$run_dir/runtime.txt" \
  python -u demo/ideal_lv_fsi/run_fem.py \
  --config demo/ideal_lv_fsi/configs/fem.json \
  --device cuda --end-time 2 --output "$run_dir/simulation" \
  > "$run_dir/run.log" 2>&1 < /dev/null &
echo $! > "$run_dir/launcher.pid"
echo "$run_dir"
```

## 改参数、续算与结果

`--dt`、`--mesh-size`、`--fluid-shape NX NY NZ`、`--fluid-lengths LX LY LZ`、`--rho`、`--mu` 可覆盖配置。`--fluid-cells N` 表示三个方向均为 N；勿与 `--fluid-shape` 同用。改本构/加载数值可编辑 JSON 的 `material`、`loads`。细化显式 MAC 网格时通常需要更小 dt。

基底模式和 beta 会写入有效配置及检查点。旧检查点缺少该字段时恢复原三方向弹簧；已保存的 radial 检查点仍按 radial 续算。要切回 spring、beta=5e5，使用当前 JSON 和新的结果目录从初态启动，不传 `--resume`。

```bash
# 检查点尚未到达 2 s 时；FEM 换成 run_fem.py
python -u demo/ideal_lv_fsi/run_mac.py \
  --device cuda --resume "$run_dir/simulation/checkpoint.npz" --end-time 2
```

续算不要再传 `--config`。`simulation/` 保存 `configuration.json`、`history.csv`、`report.json` 和 `checkpoint.npz`。MAC 的 ParaView 集合在 `simulation/vtk/solid.pvd` 与 `fluid.pvd`；FEM 集合在 `simulation/solid.pvd` 与 `fluid.pvd`。固体坐标已变形，不要再次按位移 Warp。用 `--output-every` 调整帧间隔，`--no-vtk` 禁用可视化；日志与检查点频率分别是 `--log-every`、`--checkpoint-every`。

完成 2 s 表示加载保持试验完成，不代表稳态心动周期，也没有瓣膜/循环系统模型。
