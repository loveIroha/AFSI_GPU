# 真实左心室：P1 H–O 固体 + MAC/IB，三个周期

读取外部真实左心室网格、边界和 DG0 fiber/sheet，采用用户提供的 H–O UFL、
底部径向约束及 0.8 s 周期压力/主动张力。默认总时长 **2.4 s，24,000 步**。
本目录不分发真实网格或材料方向文件；它们保留在用户的数据目录。

## 默认设置

| 项目 | 设置 |
| --- | --- |
| 长度、时间、应力 | cm、s、dyn/cm² |
| 输入路径 | `/mnt/large2/gjh/realistic_left_ventricle` |
| 固体 | 原始 P1 四面体；本次参考网格 26,889 节点、135,430 单元 |
| 材料方向 | 原始 DG0 fiber/sheet；不进行节点平均或归一化 |
| 固体体积分/底部面积分 | degree=5 正权重规则 |
| IB 相互作用积分 | 默认 degree=2、4 点/四面体；541,720 相互作用点 |
| 质量矩阵 | 一致 P1 质量矩阵，一次组装 CSR；不使用集中质量 |
| 流体 | MAC + 几何多重网格；128³ 压力单元 |
| 流体盒 | `[0,15]³ cm`；`h=0.1171875 cm` |
| 时间步 | `dt=1e-4 s` |
| 周期 | `0.8 s`，重复 3 次 |
| 流体密度/黏度 | 沿用当前框架默认 `rho=1 g/cm³`、`mu=1 g/(cm s)`，可修改 |
| 体积惩罚/底部系数 | `kappa=5e6 dyn/cm²`；`beta=5e6 dyn/cm³` |
| 底部中心 | `(7.5,7.5) cm`，与所给 UFL 一致 |
| 输出 | 日志每 100 步；检查点每 1000 步；VTK 每 200 步，即 0.02 s |

源边界已确认为 **1=心外膜、2=心内膜、3=底部**。
输入时显式映射为内部 1=内膜、2=外膜、3=底部。
UFL 的 4096/4097/4098 是其原始标识，在本 demo 中对应这些相同的物理表面；
不要求把输入 XML 的数值标签改成 4096 等。

## 安装与输入

在 AFSI_GPU 仓库根目录运行：

```bash
git pull --ff-only
conda activate afsi-torch
python -m pip install -e ".[test,mesh,fused]"
```

保持以下文件的相对路径：

```text
realistic_left_ventricle/
  mesh_scale.xdmf
  mesh_scale.h5
  boundaries.xml
  fibers_0.xml
  fibers_1.xml
  fibers_2.xml
  sheets_0.xml
  sheets_1.xml
  sheets_2.xml
```

文件名或单位不同时，可以修改主程序顶部的 `CONFIG` 或 JSON 配置。
本 demo 使用 P1，**不要先调用 `to_p2()`，也不读取导入检查的 VTK 文件作为源网格**。
读取接口直接保留原始 26,889 个顶点。

## GPU 测试与短时检查

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/test_real_lv.py tests/test_mesh_io.py \
  tests/test_mac_output.py tests/test_mac_lv_graph.py

CUDA_VISIBLE_DEVICES=0 python -u demo/real_lv_fsi/run_mac.py \
  --mesh-dir /mnt/large2/gjh/realistic_left_ventricle \
  --device cuda --end-time 0.005 \
  --output results/real_lv_smoke
```

短时检查采用与完整实验相同的 128³ 网格、材料、时间步和执行后端，推进 50 步。
首次使用会编译 H–O/流体张量核并捕获压力、质量迭代 CUDA Graph。
测试包含参考/优化路径对照，另从收缩期 0.6 s 的已知加载状态检查主动应力路径，
因此测试不只覆盖初始零载荷。

此实现已在小网格上检验本构导数、边界力、一致质量、功率配对、保存/续算，
并在完整真实固体网格上完成 16³ 流体、3 步 CPU 启动检查。
**128³ GPU 的 2.4 s 完整轨迹尚待实际运行验证。**

## 后台运行三个周期

短时检查通过后，在新输出目录进行完整实验：

```bash
run_dir="results/real_lv_3cycles"
mkdir -p "$run_dir"

CUDA_VISIBLE_DEVICES=0 nohup /usr/bin/time \
  -f 'elapsed_seconds=%e exit_code=%x' -o "$run_dir/runtime.txt" \
  python -u demo/real_lv_fsi/run_mac.py \
  --mesh-dir /mnt/large2/gjh/realistic_left_ventricle \
  --device cuda --cycles 3 --dt 1e-4 --fluid-cells 128 \
  --output "$run_dir" \
  > "$run_dir/run.log" 2>&1 < /dev/null &

echo "PID=$! output=$run_dir"
```

终端退出后程序继续运行。监测：

```bash
tail -f results/real_lv_3cycles/run.log
```

完整实验完成后查看 `report.json` 和 `runtime.txt`。
同一输出目录已有报告、历史或检查点时，新运行会拒绝覆盖。

也可以从短时检查直接续算，避免重新生成初始文件：

```bash
CUDA_VISIBLE_DEVICES=0 python -u demo/real_lv_fsi/run_mac.py \
  --device cuda --resume results/real_lv_smoke/checkpoint.npz --cycles 3
```

续算使用原目录，恢复 H–O 参数、基底中心、DG0 方向、原始网格、流体设置和时间步。
检查点自包含，不依赖再次读取原始数据文件。
续算仅允许更改设备与目标时长；更改物理参数或执行设置需要新运行。

## 对流检查停止时

MAC 中心对流当前保留两个保守检查：
`CFL = dt*sum(max|u_i|/h_i) <= 0.25` 和
`cell_Re = max(max|u_i|*h_i/(mu/rho)) <= 1`。
这些检查不是完整的非线性 FSI 稳定性证明；小散度也不证明力学耦合稳定。
默认网格与流体参数下，单分量速度超过 `8.533333 cm/s` 就会触发 cell_Re 限制。
减小 dt 只能降低 CFL，不会降低同一速度场的 cell_Re。
不要仅为继续运行而关闭检查或改变黏度；先检查实际速度、局部峰值与力学时间推进。

发生此异常时，程序保存最后接受状态到原运行目录的 `checkpoint.npz`，
并写出 `report.json` 的失败信息。新增错误输出分别记录 CFL、cell_Re 和触发项，
失败报告同时保存 `failure.transport_guard`；旧版失败检查点也可以直接分析：

```bash
python validation/diagnose_real_lv_guard.py \
  --checkpoint results/real_lv_3cycles/checkpoint.npz \
  > results/real_lv_3cycles/guard_diagnosis.json
```

请使用实际失败的运行目录。此命令只在 CPU 读取保存的速度与配置，
不导入原始网格、不编译 GPU 内核、不推进或改写检查点；它不做完整校验和验证。
输出包括 CFL、cell_Re、速度峰值及其交错网格坐标。
诊断和恢复尚未完成时，直接续算相同检查点会再次触发相同限制。

## 主程序与 JSON 参数接口

`run_mac.py` 顶部的 `CONFIG` 显式列出材料、标签、流体域、时间步和输出。
运行优先级是 CONFIG → JSON → 显式命令行参数。

```bash
python demo/real_lv_fsi/run_mac.py --write-config real_lv.json
```

编辑 `real_lv.json` 后：

```bash
CUDA_VISIBLE_DEVICES=0 python demo/real_lv_fsi/run_mac.py \
  --config real_lv.json --output results/real_lv_custom
```

常用覆盖包括 `--mesh-dir`、`--dt`、`--fluid-cells`、`--fluid-lengths`、
`--fluid-origin`、`--rho`、`--mu`、`--kappa`、`--beta`、`--cycles`、
输出间隔和 `--no-vtk`。`--reference` 使用 torch/PCG 参考路径，
适用于小网格验证。自定义文件名、积分阶数和求解器容差通过 CONFIG/JSON 设置。
修改周期时长需要新的分段载荷函数；本次按用户原式固定 0.8 s。

## 与用户 UFL 的对应

采用 `x` 存储当前坐标；`u=x-X_ref` 是位移。
因此 `grad(x)` 恰好对应用户的 `I+grad(u)`，不会重复加单位阵。

- `I1_bar = J^(-2/3)*tr(C)`，只修正 I1。
- 纤维、片层使用未修正的 `C`，分别将 `I4f/I4s` 截断到不小于 1。
- 纤维/片层耦合项使用 `I8fs²`。
- 原式体积应力 `kappa*ln(det(C))*F^(-T)` 对应能量 `kappa*(ln J)²`；
  没有替换成先前 Guccione 的 `(J-1)²` 惩罚。
- 主动应力保持 `T*(1+4.9*(sqrt(I4f)-1))*F*(f⊗f)`，没有额外截断该伸长因子。
- 不启用 UFL 中注释掉的 PN 消除项。
- 压力力为 `-p*cof(F)*N_ref`；只加载真实心内膜，没有额外阀口封盖压力。
- 底部约束保持原式：xy 中允许参考径向移动，惩罚切向位移；惩罚 z 位移。
  使用参考面积积分，并沿用 UFL 的力符号。
- 外膜为自然边界；没有额外添加外膜弹簧。

P1 位移、DG0 方向使 F 和材料应力在每个单元内为常量。
固体力仍采用 degree=5 积分权重，但只对每个单元计算一次 H–O 应力后求和，
避免在所有积分点重复计算相同的 3×3 代数。
IB 的核积分不是这个材料积分，单独使用可配置的相互作用积分规则。
默认 4 点规则对 P1 一致质量矩阵精确；它不证明正则化 IB 核已经积分收敛。

UFL 中 `inner(U,V)*dx` 对应质量矩阵。
在本项目中，固体弱形式先返回积分后的节点力，IB 传播再求解 `M F=b` 得到力密度系数，
因此不把质量项重复加入节点力，也不使用对角集中质量替代它。

周期压力和张力逐段照搬用户的 C++ 表达式，并只做一次 kPa→CGS 转换。
压力在每个周期末由约 1.067 kPa 重置到 0：这个跳变来自给定表达式，未自行平滑。
耦合沿用当前 AFSI 风格的滞后力时序；CSV 的 `force_time_s` 记录实际力采样时间，
载荷列则记录当前状态时刻的规定载荷。

这是有限惩罚、显式分区 IB–FSI 运行，不是严格混合不可压约束、隐式 Newton 耦合，
也不包含多孔介质压力/渗流未知量。三个规定载荷周期不等同于已达到周期稳态。

## 输出与代码位置

- `configuration.json`：实际配置。
- `history.csv`：压力/张力、腔体积、J 范围、底部约束误差和流体诊断。
- `report.json`：运行状态、离散/材料/边界说明和最后一个接受状态。
- `checkpoint.npz`：自包含 P1/DG0/H–O 检查点。
- `vtk/solid.pvd`：变形后的 P1 固体、位移、节点力、逐单元 fiber/sheet、J。
- `vtk/fluid.pvd`：流体压力、显示速度和散度。

腔体积用真实心内膜加一个平均开口顶点的虚拟三角扇封口计算。
封口只用于测量，不参与力学加载；非平面开口的体积依赖这个明确约定。

实现位置：`real_lv.py`（配置、固体和编译执行）、`holzapfel_ogden.py`（材料/载荷）、
`p1.py`（积分和形函数）、`simulation/real_lv_mac.py`（运行）、
`real_lv_checkpoint.py`（续算）。流体、IB 和 CSR/Graph 质量求解复用共享 `mac/` 模块。
