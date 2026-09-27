# 生成理想左心室的 0.8 s GPU IB/FEM demo

入口是 `examples/lv_cycle.py`。默认在 CPU 上生成厘米制椭球左室网格、标签和纤维，将张量送往指定设备，然后用 PyTorch 的 P2 非线性固体力、四点 IB、Q2/Q1 Chorin 和显式坐标更新推进 0.8 s。`--device cuda` 不会回退到 CPU。该算例不需要输入左室网格或纤维文件；保存输出和检查点时才将张量复制到 CPU。

这是**规定压力和主动张力的理想左室 FSI demo**。默认第一周期包含从无载参考构形开始的充盈过程，不是已经达到周期稳态的循环模型。背景流体占据整个盒子，左室基底开口没有瓣膜。`full_cycle_completed` 仅表示接受的时间步覆盖了一个 0.8 s 载荷周期，不表示生理标定、网格收敛或周期稳态已经验收。

从 0.26.0 起，入口默认 `--backend csr --check-every 8`，固定算子组装一次后复用。昂贵诊断改为输出采样，报告的体积、det(F)、速度和 CFL 极值是**采样极值**；真实线性残差、非有限值、构形有效性和 IB 支持/步长限制仍逐步检查。详见 [CSR 运行说明](CSR_RUNTIME.md)。

## 与 AFSI demo_337 的对应

已核对 [原生示例](https://github.com/loveIroha/afsi/blob/main/afsic/demo/demo_337/fsi_paralell_fibers_contraction.py)（文件 blob `283b23f5155dbc57043edd2aa7280d61c3c8e985`）和 [PressureEndo.py](https://github.com/loveIroha/afsi/blob/main/afsic/demo/demo_337/PressureEndo.py)（blob `8bf2cce74346522f2d10139765ecec0781f1d2a5`）。原示例当前启用的是 1.5 s 线性加载，总时长 2 s；本 demo 按用户要求采用辅助文件中的 0.8 s 周期形状。代码独立表达相同的数学加载关系，不依赖安装 AFSI。

| 项目 | 本 demo |
| --- | --- |
| 材料 | 非等容分解 Guccione：C=20000、bf=8、bt=2、bfs=4、kappa=500000，与原示例一致 |
| 主动应力 | Ta(F f0)⊗f0；张力幅度 600000 dyn/cm²，取原示例配置 |
| 基底 | 三方向参考面积弹簧 beta=500000 dyn/cm³，与原示例一致 |
| 压力 | ENDO 随动牵引 −p cof(F)N；EPI 无规定牵引 |
| 流体 | rho=1 g/cm³、mu=1 g/(cm·s)，Q2/Q1、Chorin、外盒全壁零速度、角点压力规范 |
| IB | 四点 Peskin；核尺度等于速度格点间距；积分节点力 → 节点力密度 → Q2 一致质量弱载荷 |
| 时间步 | dt=5e-5 s，与原示例一致；0.8 s 共 16000 步 |
| 推进顺序 | 零力启动；用保存的 g_n 推进流体，在 x_n 插值速度并更新 x，按 t_n 装配下一轮节点力 |
| 几何/纤维 | 程序生成椭球壳、ENDO/EPI/BASE 和规则纤维，替代原示例外部数据；不声称几何相同 |

固体在每一时间步按当前变形重算完整非线性力，与 AFSI 显式耦合次序一致。每步没有额外求一个静态 Newton 平衡；已有 Newton–GMRES 仅在选择预加载流程时使用。

默认压力时程为：0～0.2 s 从 0 升至 8 mmHg；0.2～0.5 s 保持；0.5～0.65 s 升压，0.65～0.8 s 回落。主动张力在 0.5 s 前为零，在收缩段升高并回落。两者沿用 AFSI 辅助函数的 Gaussian 上升/回落形状，宽度平方分别为 0.004 s² 和 0.005 s²。舒张压 8 mmHg 取辅助函数默认值；收缩压力幅度 150000 dyn/cm²（约 112.51 mmHg）和张力幅度取示例配置。原示例配置中的 `diastole_pressure=100000` 没有进入其当前启用的线性加载表达式。本周期没有把它误用为舒张压。

沿用的指数波形没有做峰值归一化，所以 0.65 s 的实际峰值略低于幅度参数。后续周期保持舒张基线，不重复第一次从无载压力开始的充盈坡道，从而避免 0.8 s 接缝压力跳变。`loads.csv` 给出完整的规定时程；`history.csv` 同时区分当前时刻的规定压力、已施加节点力的时间戳和下一轮节点力时间戳。

## 空间设置

默认固体目标网格尺度 1.2 cm。生成左室内半轴为 (2.5,2.5,4.5) cm，外半轴为 (3.5,3.5,5.5) cm，基底 z=1.5 cm。流体盒为 12 cm，每方向 24 个 Q2 单元：单元边长 0.5 cm，速度节点间距及 IB 核尺度 0.25 cm。盒子原点默认 (−6,−6,−8) cm。

AFSI 原示例为 5 cm 盒、32 个单元/方向，单元边长 0.15625 cm、速度间距 0.078125 cm。原盒装不下当前宽约 7 cm 的生成左室，因此不能原样复用盒尺寸。**本次默认完整周期运行采用较粗流体网格，并不宣称与 AFSI 空间步长相同。** 若需同样的绝对流体间距，可以在新输出目录设置 `--box-length 12.5 --fluid-cells 80`；这明显增加显存、线性迭代和计算量，当前尚未对该大规模设置做性能验收。报告记录实际间距，不以相同单元数代替相同分辨率。

## 运行完整周期

在 Linux 的 AFSI_GPU 仓库目录执行，CUDA 计算使用一张 4090：

```bash
git pull --ff-only
conda activate afsi-torch
python -m pip install -e ".[test,geometry,io]"
CUDA_VISIBLE_DEVICES=0 python -m pytest -q
CUDA_VISIBLE_DEVICES=0 python examples/lv_cycle.py \
  --device cuda --end-time 0.8 --dt 5e-5 \
  --mesh-size 1.2 --fluid-cells 24 --box-length 12 \
  --output results/lv_cycle_080
```

该命令从新生成的无载左室出发，直接以 0.8 s 为终点，不自动缩短运行、不自动减小载荷，也不改用 Schur 投影。每 100 步打印进度、腔容积、min det(F)、载荷和按实测速率估算的剩余时间；每 200 步存检查点和 VTK。初始完整输出在开始推进前保存。用户可用 `--no-vtk` 仅保留数值报告、CSV 和检查点。

支持 `--diastole-pressure-mmhg`、`--systole-pressure-mmhg` 和 `--max-tension` 覆盖幅度，实际参数会写入报告。改变参数请使用新输出目录，以免覆盖已有轨迹。

如需从已经收敛的预载开始，在新运行中加 `--preload results/lv_equilibrium`，并去掉 `--mesh-size`。该选项保留原始参考 X、预载坐标、材料、纤维及残余节点力，并从保存的预载压力进入充盈坡道；它不会把已有 0.2 mmHg 测试预载重新标成 8 mmHg，也不会重新求解平衡。

## 续算、输出与结果判读

正常中断后，用相同目录的检查点继续：

```bash
CUDA_VISIBLE_DEVICES=0 python examples/lv_cycle.py \
  --device cuda --resume results/lv_cycle_080/checkpoint.npz --end-time 0.8
```

续算恢复网格、纤维、材料、载荷、x/u/p/g 和载荷时间戳，无需重生成网格或重新加载原预载目录。物理设置从检查点读取，因此不要同时传入 dt、网格、预载或载荷覆盖参数。CSV/PVD 会恢复到保存的有效步，去除检查点之后尚未提交的输出记录。检查点采用临时文件替换，包含校验和；数值失败也会保留最后有效状态并以非零退出码结束。若失败来自 det(F)、IB 支持越界或求解器不收敛，保留报告分析原因；原参数直接续算一般会再次遇到同一问题。

运行后生成曲线：

```bash
python examples/plot_lv_cycle.py --input results/lv_cycle_080
```

- `report.json`：是否完成、已接受步数、达到时间、采样 det(F)/速度/CFL 极值、逐步线性残量极值及运行配置；`diagnostic_scope` 说明覆盖范围。
- `history.csv`：位移、壁体积、腔体积、能量、散度、载荷和线性迭代记录；默认每 20 步，并保存终点。
- `loads.csv`：规定的完整压力/张力时程。
- `cycle_curves.png`：压力、主动张力、腔体积时间曲线和规定压力—腔容积轨迹。
- `solid.pvd`、`fluid.pvd`：ParaView 动画入口，默认每 0.01 s 一帧。
- `checkpoint.npz`：最近的可续算状态。

压力—容积图使用**规定的内膜牵引压力**。VTK 中的压力是背景 Chorin 压力，两者不得混为同一场。完成首个 0.8 s 周期不要求末态回到无载初态，也不把该曲线命名为闭环循环系统预测。

## 本地验证范围

Windows CPU 全量回归：189 passed、102 CUDA skipped、1 warning。新增测试覆盖 AFSI 周期各阶段与接缝、不中断/续算轨迹一致、保留滞后节点力时间、输出历史恢复、检查点篡改拒绝以及失败时保存最后有效状态。

实际生成左室 CPU 先导采用 mesh-size=1.8 cm、6³ 流体网格、同一 dt=5e-5 s，完成 200 步至 0.01 s，包含一次中途续算。腔容积从 77.521791 增至 77.558551 mL，最大节点位移 0.000772515 cm，min det(F)=0.9991393，线性残量均在设置容差内。这仅验证长程入口和充盈启动。随后用户在 RTX 4090 上完成 0.8 s、16000 步正式运行；[0.25.0 GPU 结果](LV_CYCLE_GPU_0.25.0.md)与[0.26.0 CSR GPU 结果](LV_CYCLE_CSR_GPU_0.26.0.md)记录了耗时、腔容积及数值问题。

