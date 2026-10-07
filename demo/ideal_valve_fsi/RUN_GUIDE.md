# 二维理想瓣膜运行与方法指南

[全部 demo](../../docs/DEMO_GUIDE.md) · [二维 GPU 实现](../../docs/VALVE_GPU_EXECUTION.md)

入口为 `demo/ideal_valve_fsi/run_mac.py`，对应 AFSI `demo_340` 的双瓣叶通道问题。以下命令在仓库根目录执行。

## 推荐：JSON 直接运行

```bash
python demo/run.py demo/ideal_valve_fsi/configs/mac_gpu.json
```

每行对应一个独立实验。运行继承环境启动文件选择的 GPU，默认使用原生 CUDA IB；可追加 `--end-time 0.005` 短跑。统一入口打印独立输出目录，也支持 `--output`。

## 设置与数值方法

- CPU 用 Gmsh 生成两片矩形瓣叶，宽 0.0212 cm、长 0.7 cm、右侧 x=2 cm，网格尺寸 0.01 cm；固体用六节点 P2 三角形，单位面外厚度。纤维方向分别为 `(1,1)/sqrt(2)` 与 `(1,-1)/sqrt(2)`。
- FRH 超弹性：`C0=2e5`、`C1=1e6`、`kappa=4e5 dyn/cm²`。用 `I1=tr(FᵀF)/J`、`I4=|F f0|²/J`，能量为 `C0/2*(I1-3)+C1*(exp(I4-1)-I4)+kappa/2*((J²-1)/2-ln J)`。三角形采用六点四次精确积分，根部边界采用弹簧 `beta=1e8`，参考线积分（单位厚度）。无瓣叶接触模型。
- 流体域 8×1.61 cm、MAC 网格 256×64，速度在面、压力在中心；密度 1 g/cm³、动力黏度 0.1 g/(cm·s)。中心差分对流与黏性显式预测，再解压力 Poisson 并修正速度。压力采用通道几何多重网格。
- 入口 `u_x(y,t)=5*[sin(2*pi*t/1s)+1.1]*y*(1.61-y)`，`u_y=0`；这里 5 是代码的抛物线系数，并非峰值速度。上下壁无滑移，出口压力为零，预测速度出口使用零法向导数。
- IB 用四点 Peskin 核、P2 积分点及参考一致质量矩阵 CSR，质量方程用 PCG/Graph。壁面附近使用实现中的边界传递规则。固体与流体的双向传递共享积分权重。
- 每步先传播已存固体力，推进流体，插值新速度并显式更新位置，再计算新的非线性固体力。初始化速度与存储力为零。没有联合 Newton 迭代。
- `dt=1/16000=6.25e-5 s`，3 s 共 48000 步，对应三个入口载荷周期。显式对流、黏性、弹性和 IB 均可能限制时间步。

## 安装与短运行

```bash
conda activate afsi-torch
python -m pip install -e ".[test,geometry,fused]"
python -u demo/ideal_valve_fsi/run_mac.py \
  --config demo/ideal_valve_fsi/configs/mac_gpu.json \
  --device cuda --end-time 0.005 --output results/ideal_valve_short
```

GPU 预设的 `execution.ib_backend="cuda"` 启用 C++/CUDA 插值/传播，保留反射壁面负权重、重复索引和一致质量矩阵。模板准备及 FE 求值/装配继续使用 PyTorch/Triton。另启用 optimized、Graph 质量求解、压力 auto（CUDA 下选择 Graph）与热启动。可以用 `--ib-backend reference` 对照；真实 LV 的 CSR contraction 参数不适用于本入口。

## 完整 3 s 后台运行

```bash
run_dir="results/ideal_valve_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_dir"
nohup /usr/bin/time \
  -f 'elapsed_seconds=%e exit_code=%x' -o "$run_dir/runtime.txt" \
  python -u demo/ideal_valve_fsi/run_mac.py \
  --config demo/ideal_valve_fsi/configs/mac_gpu.json \
  --device cuda --end-time 3 --fluid-fields \
  --output "$run_dir/simulation" \
  > "$run_dir/run.log" 2>&1 < /dev/null &
echo $! > "$run_dir/launcher.pid"
echo "$run_dir"
```

`--fluid-fields` 加入流体输出；预设只输出固体，做速度对比时应保持双方输出设置一致。

## 参数与续算

`--nx`、`--ny` 改流体单元数；`--dt`、`--mesh-size`、`--rho`、`--mu` 改对应参数。通道高度必须与 `solid.height` 一致；材料、入口参数与几何通过 JSON 修改。网格变密后不能默认原 dt 仍稳定。

```bash
# checkpoint 必须尚未到达 3 s，续算恢复已有物理参数
python -u demo/ideal_valve_fsi/run_mac.py \
  --device cuda --resume "$run_dir/simulation/checkpoint.npz" --end-time 3
```

`--field-every 160` 每 0.01 s 输出一次；`--field-every 0` 禁用场输出。此入口使用 `--field-every`，不是 LV 的 `--output-every`。

## 结果

`simulation/` 下有配置、`report.json`、`history.csv`、`checkpoint.npz`；ParaView 打开 `fields/solid.pvd`，启用流体输出后还有 `fields/fluid.pvd`。标量历史记录瓣叶位移、Jacobian、流场及求解诊断。日志/墙钟时间位于外层 `run.log`、`runtime.txt`。

本例的流体是 MAC 有限差分，原 AFSI 比较运行采用其 FEM 流体；耗时差异不是相同离散系统的纯硬件加速比。完整 3 s 也不证明接触正确性或网格收敛。
