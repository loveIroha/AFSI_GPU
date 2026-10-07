# AFSI demo_337 对齐算例（0.27.0）

当前 `demo/ideal_lv_fsi` 的两个正式入口及 GPU JSON 均采用原始 AFSI demo_337 的三方向位移弹簧，beta=5e5，与下表一致。理想 LV 长轴为 x，基底平面 x=4 cm；横向显示并非坐标输出错误。径向模式仍为可选配置，已有检查点保持保存的边界而不受新默认影响。当前边界细节及 ParaView 显示说明见 [运行指南](../demo/ideal_lv_fsi/RUN_GUIDE.md)。

基准是 [fsi_paralell_fibers_contraction.py](https://github.com/loveIroha/afsi/blob/main/afsic/demo/demo_337/fsi_paralell_fibers_contraction.py)，核对的文件 blob 为 `283b23f5155dbc57043edd2aa7280d61c3c8e985`。以实际执行语句为准；不采用被注释的周期函数，也不混用目录内其他试验脚本或静力 Pulse 基准的边界条件。

## 实际启用的物理与离散设置

| 项目 | 新入口 `examples/lv_afsi337.py` |
| --- | --- |
| 时长 | 2 s，共 40000 步；不是重复心动周期 |
| 时间步 | 0.00005 s |
| 压力 | `p=150000*min(t/1.5,1)` dyn/cm²，最终 15 kPa，约 112.5 mmHg |
| 主动张力 | `Ta=600000*min(t/1.5,1)` dyn/cm²，最终 60 kPa |
| 加载方式 | 从零同时线性增加压力、张力；1.5–2 s 保持；不是先被动充盈再主动收缩 |
| 被动力学 | 非等容分解 Guccione；C=20000、bf=8、bt=2、bfs=4、kappa=500000 |
| 体积惩罚 | `Wvol=kappa*(J-1)^2`，不是 `kappa/2`，也不是混合严格不可压缩固体 |
| 主动应力 | 第一 Piola 应力 `Ta*(F*f0) outer f0` |
| 基底 | 参考面积上的三方向弹簧，beta=500000；没有固定全部基底自由度 |
| 内外膜 | 内膜随动压力；外膜自由 |
| 固体单元 | 直边 P2 四面体；当前构形可为曲边 P2 |
| 固体积分 | Basix 默认 degree 4：四面体 14 点、三角形 6 点；避免只对齐阶数却使用不同积分点 |
| 流体域 | `[0,5]^3` cm，32×32×32 六面体 |
| 流体网格间距 | 单元边长 0.15625 cm；Q2 节点格距 0.078125 cm |
| 流体空间与参数 | Q2 速度/体力，Q1 压力；rho=1 g/cm³，mu=1 g/(cm·s) |
| 流体边界 | 外部六面零速度；原点压力定标为零 |
| 流体推进 | Chorin，保留已对齐的弱形式；CSR 矩阵与 PyTorch Jacobi-PCG |
| IB | 四点 Peskin，epsilon 等于 Q2 节点格距；积分固体力经 IB 转为流体力密度，再施加一致质量矩阵 |
| 耦合时序 | 零力启动；流体推进→旧位置插值→更新固体→用 t_n 组装新力，下一步使用 |

原配置的 `diastole_pressure=100000` 在该脚本的当前加载语句中没有使用，不能据此设置一个额外的舒张压阶段。AFSI 虽然导入 IPCS，当前实际实例化的是 Chorin。

纤维和片层先按 P2 插值至积分点，再取 `n0=cross(s0,f0)`，不在积分点重新归一化。原例中的 `volume` 是壁体积 `∫J dX`，应与 `wall_volume_cm3` 对比；本项目另外提供的 `cavity_volume_ml` 是腔体积。原例 `u_max` 是最大分量，与本项目最大速度模不同。

## 几何与纤维：明确可对齐的范围

公开脚本从仓库外的 `~/afsi-data/337_ideal_left_ventricle` 读取网格及纤维，无法从公开代码确认其固体单元数量、实际网格尺寸和每个系数。

默认仍满足“不读外部网格、直接生成理想左室”的要求。参数取自同目录的 [配套静力基准几何生成代码](https://github.com/loveIroha/afsi/blob/main/afsic/demo/demo_337/plot/bench-ilv-contraction.py)：

- 内膜半短轴 0.7 cm、半长轴 1.7 cm；外膜分别 1 cm、2 cm。
- 长轴沿 x，心尖朝负 x；局部基底截面 x=0.5 cm。
- 平移 `(3.5,2.5,2.5)` cm，世界坐标基底 x=4 cm。
- Gmsh 目标网格尺寸默认 **0.1 cm**，这是明确选择的新生成参数，不能称为原始外部网格的尺寸。
- 纤维采用 [cardiac-geometries 椭球规则](https://github.com/ComputationalPhysiology/cardiac-geometriesx/blob/main/src/cardiac_geometries/fibers/lv_ellipsoid.py)：P1 Laplace 内膜0、外膜1、基底自然边界，将标量插值到 P2 节点，内膜螺旋角 +90°、外膜 −90°，椭球切向纤维及片层。采用该数学构造，未调用 FEniCS。
- 精确心尖处的切向方向有坐标奇点，选取 theta=0 的极限框架；不使用旧示例的径向逃逸。外部投影纤维与此生成场仍可能不同。

配套静力基准用于确认生成几何与纤维配方，**没有照搬其严格不可压缩材料和固定基底**。新入口使用上表中动态 AFSI 脚本的惩罚材料与弹簧条件。

因此默认模式报告 `solid_mesh_and_fibers_matched=false`，这不是运行失败。需要逐节点一致时，可使用下文导入模式。无论哪种模式，本项目都不会把加载一致等同于解已经与 AFSI 验证一致。

## Linux / RTX 4090 运行

在原 `AFSI_GPU` 仓库目录内：

```bash
git pull --ff-only
conda activate afsi-torch
python -m pip install -e ".[test,geometry]"
CUDA_VISIBLE_DEVICES=0 python -m pytest -q
CUDA_VISIBLE_DEVICES=0 python examples/lv_afsi337.py \
  --device cuda --output results/lv_afsi337
```

默认生成新几何、采用 32³ 流体网格与 2 s 时长。请使用新输出目录，不能从旧 0.8 s 波形的检查点接着改载荷。此次几何和域大小改变，不能将旧运行的容积、耗时直接视为相同算例的对照。

续算同一对齐算例：

```bash
CUDA_VISIBLE_DEVICES=0 python examples/lv_afsi337.py \
  --device cuda --resume results/lv_afsi337/checkpoint.npz
```

可用 `--end-time 2.5` 延长同一恒载保持段；不要凭此提前宣称已达平衡。`--mesh-size`、`--fluid-cells`、`--dt` 是显式实验覆盖项，报告记录实际设置和流体/时间步是否仍匹配；续算保留检查点的物理设置。核心计算依旧在 PyTorch/GPU 上，Gmsh 在 CPU 生成网格。保存、可视化和标量日志需要 CPU。

输出有 `report.json`、`history.csv`、`loads.csv`、检查点以及 VTK/PVD。判断保持段是否趋于平衡，查看 1.5–2 s 的速度、动能、腔/壁体积及连续帧位移变化；仅完成 40000 步不构成稳态证据。完成后提供 `report.json` 与 `history.csv` 即可分析。

## 可选：使用原始固体网格和 P2 纤维

这只是格式导出，不要求重新跑一套 AFSI CPU 对照实验。需要容器中确实具有原始 `mesh.xdmf` 及引用的 HDF5、`markers.json`、`f0.txt`、`s0.txt`、`cdm.txt`。导出脚本独立运行，只依赖容器已有 DOLFINx/Basix/numpy。

```bash
docker start afsi_dev_ljy
docker cp validation/export_afsi337_solid.py afsi_dev_ljy:/tmp/export_afsi337_solid.py
docker exec afsi_dev_ljy python /tmp/export_afsi337_solid.py \
  --data-root /root/afsi-data/337_ideal_left_ventricle \
  --output /tmp/afsi337_solid.npz
mkdir -p results/afsi337_input
docker cp afsi_dev_ljy:/tmp/afsi337_solid.npz results/afsi337_input/solid.npz
CUDA_VISIBLE_DEVICES=0 python examples/lv_afsi337.py \
  --device cuda --solid-input results/afsi337_input/solid.npz \
  --output results/lv_afsi337_native
```

若实际数据放在别处，修改 `--data-root`。导出保留原始 P2 系数，不重新生成或归一化；先按 AFSI 的原始毫米坐标哈希关联纤维，然后将坐标 `/10+(3.5,2.5,2.5)`，同时导出容器 Basix 的积分规则。它会拒绝哈希歧义、缺失系数、非直边参考单元或不完整边界，避免静默替换。

报告保留来源文件 SHA256 和导出包 SHA256，续算保存完整纤维与积分点。导入时 `solid_config` 是配套 CAD 的参考参数；真实网格以包内 X/cells 为准，不会依据 `mesh_size` 重新划分。

## 在没有原始外部文件时运行 AFSI 原版

目录中的 `plot/bench-ilv-contraction.py` 会程序生成椭球几何和纤维，但它用于另一个 Pulse 静力基准，输出 `lv_ellipsoid-problem3` 与 `Quadrature_6` 字段；FSI 主脚本读取的是 `~/afsi-data/337_ideal_left_ventricle` 的 XDMF + P2 纤维文本。因此它不能直接作为 FSI 的数据准备命令。

本项目提供一个双阶段桥接：在 GPU 主机使用当前生成算例写数值包，在已有 DOLFINx 容器内写成 AFSI 期待的 `mesh.xdmf`、HDF5、`markers.json`、`f0.txt`、`s0.txt`、`cdm.txt`。这是一套**新生成的、两边可共享的理想左室数据**，不冒称是原来未公开的外部文件。默认 0.1 cm 网格有 17393 个四面体和 28840 个 P2 节点；AFSI 的弱坐标哈希在这个测试网格上无冲突，写出脚本也再次检查。

在 **GPU 主机** 的 `AFSI_GPU` 目录：

```bash
git pull --ff-only
conda activate afsi-torch
python validation/generate_afsi337_source.py --output results/afsi337_input/source.npz
docker start afsi_dev_ljy
docker cp results/afsi337_input/source.npz afsi_dev_ljy:/tmp/afsi337_source.npz
docker cp validation/write_afsi337_native_inputs.py afsi_dev_ljy:/tmp/write_afsi337_native_inputs.py
docker exec afsi_dev_ljy python /tmp/write_afsi337_native_inputs.py \
  --source /tmp/afsi337_source.npz \
  --output /root/afsi-data/337_ideal_left_ventricle
```

生成器拒绝覆盖已有目录。若你已拥有真正的 `337_ideal_left_ventricle` 文件，保留它，使用上一节导入模式即可。容器写入脚本会用 DOLFINx 读回网格与标签并核对计数，确保交给原版主脚本的文件存在且可读。

然后在 **容器内**：

```bash
cd /root/afsi/afsic/demo/demo_337
mkdir -p /root/gjh/results/afsi337_cpu
nohup bash -c '
  start=$(date +%s)
  python -u fsi_paralell_fibers_contraction.py
  code=$?
  end=$(date +%s)
  printf "elapsed_seconds=%s exit_code=%s\n" "$((end-start))" "$code" > /root/gjh/results/afsi337_cpu/runtime.txt
  exit "$code"
' > /root/gjh/results/afsi337_cpu/run.log 2>&1 < /dev/null &
echo $! > /root/gjh/results/afsi337_cpu/pid
```

若源码实际位置不同，替换 `cd` 行。`runtime.txt` 仅在进程退出后出现，`exit_code=0` 才表示正常完成。AFSI 源脚本还依赖 SwanLab/项目编号服务及其原有输出路径，相关启动问题应查看 `run.log`。原版每步计算额外全局积分和高频文件输出；计时包含这些开销。

要使 PyTorch 计算使用**转换后的同一份 DOLFINx 固体网格及 P2 系数**，可按上一节运行 `export_afsi337_solid.py` 后，以 `--solid-input` 启动 GPU 算例；仅再次调用 Gmsh 不能保证两个程序跨环境逐节点相同。

## 验证边界

本地验证涵盖加载关键时刻、Basix 积分点一致性、椭球解析方向、独立 P1 Laplace 残差、非零固体力、短程耦合及断点续算、原始系数保留和错误边界拒绝。CPU 环境无法验证 CUDA；需运行上面的 GPU 测试。没有 DOLFINx 的本地环境，仍须在你的容器内验证 XDMF 和纤维文本的实际写出；此前只验证了桥接数值包、坐标映射和错误拒绝。尚未在此环境运行默认网格的完整 2 s 长程模拟。

本地完整回归：**203 passed、109 skipped、2 warnings**。Gmsh 4.15.2 默认 0.1 cm 网格初始化得到 17393 个四面体、28840 个 P2 节点；初始腔体积约 2.48153 mL、壁体积约 3.23236 cm³。这是小尺寸理想几何基准，不是此前约百毫升的演示几何。单元数量可能随 Gmsh 平台/版本变化，实际以报告为准。
