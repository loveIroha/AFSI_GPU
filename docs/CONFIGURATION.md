# 项目结构与算例配置

长度使用 cm，时间使用 s，密度使用 g/cm³，动力黏度使用 g/(cm·s)，
材料应力、内膜压力和主动张力使用 dyn/cm²。二维瓣膜按单位面外厚度计算。

## 代码结构

```text
src/afsi_torch/
  config.py              # 公共配置类型、JSON 读写及参数验证
  paper_lv.py            # Ma2024 真实 LV 物理配置、式81/82本构与固体适配器
  simulation/
    lv_mac.py            # 理想左室 MAC 耦合推进、检查点和结果记录
    lv_fem.py            # 左室 Q2/Q1 FEM 耦合推进及旧周期入口
    valve_mac.py         # 二维瓣膜 MAC 耦合推进
    paper_lv_mac.py      # 真实 LV BE–BE 被动充盈、检查点及 VTK 输出
    cli.py               # 三个 demo 共享的命令行处理
  mac/                   # 三维 MAC、压力多重网格、IB 及执行优化
  mac2d/                 # 二维通道 MAC、压力多重网格及 IB
  fluid/                 # Q2/Q1 FEM 流体、CSR 算子与线性求解
  geometry/              # 固体网格、纤维、边界
demo/
  ideal_lv_fsi/
    run_mac.py           # 主程序中集中定义 CONFIG
    run_fem.py           # 主程序中集中定义 CONFIG
    configs/mac_gpu.json # 当前优化的三维 GPU 配置
    configs/fem.json     # 原 FEM 默认配置
  ideal_valve_fsi/
    run_mac.py
    configs/mac_gpu.json
  real_lv_fsi/
    run_mac.py           # Ma2024 被动充盈；PaperLVConfig，默认1.5 s
examples/                # 保留旧命令和 import 的兼容入口
validation/              # 独立对照、性能测量及导出工具
tests/                   # 正确性与兼容性检查
results/                 # 用户运行生成的数据
```

demo 负责选择参数，可复用的运行代码已移入安装包。新算例可以在主程序中
导入配置和 runner，直接调用，无需从 examples 导入，也无需修改求解器。
当前提供的生成模型是理想左室和双瓣叶；其他拓扑的固体需要相应的网格、
材料/载荷模型和初始化代码，配置文件本身不会自动构造任意几何。

真实 LV 的当前入口使用 `PaperLVConfig`，定义 BE–BE／半拉格朗日开放盒流体。
它与旧 `RealLVConfig`／CN–AB2 库接口及检查点属于不同方案；详情及完整命令见
[论文真实左心室复现](../demo/real_lv_fsi/README.md)。

## 在 demo 主程序中设置

打开对应 `run_mac.py` 或 `run_fem.py`，修改文件顶部的 `CONFIG`。
例如三维 MAC 的主要字段：

```python
from afsi_torch.config import (
    LVSimulationConfig, TimeConfig, FluidConfig, LVExecutionConfig,
)
from afsi_torch.simulation.lv_mac import run

config = LVSimulationConfig(
    basal_constraint='spring', beta=5e5,
    time=TimeConfig(dt=5e-5, end_time=2.0),
    fluid=FluidConfig(
        shape=(64,64,64), lengths=(5.,5.,5.), origin=(0.,0.,0.),
        rho=1., mu=1.,
    ),
    execution=LVExecutionConfig(
        execution_backend='fused', pressure_backend='graph',
        solid_backend='pointwise', mass_backend='graph',
        coupling_backend='optimized', warm_start=True,
    ),
)
run(case_config=config, device='cuda', output='results/my_lv')
```

Python API 配置类型为 `LVSimulationConfig`、`LVFEMSimulationConfig`、
`ValveSimulationConfig`；对应 runner 为 `simulation.lv_mac.run`、
`simulation.lv_fem.run`、`simulation.valve_mac.run`。

理想左心室的 `basal_constraint` 可取 `spring` 或 `radial`：前者在三个方向施加位移弹簧，与 AFSI demo_337 一致；后者允许基底径向运动，以 `beta` 惩罚轴向/切向位移。正式 MAC/FEM demo 及库配置均默认 spring、beta=5e5。可选径向模式的约束平面随 `geometry.long_axis` 转动，横截面中心取 `geometry.center`；三方向弹簧无需投影。这两个字段记录在模型检查点中，续算不会依据新 demo 默认改写旧边界。beta 单位为 dyn/cm³。当前生成纤维的配套例程要求 long_axis='x'，改变计算几何朝向还需同步适配纤维生成，不能只改这个字段。

| 字段 | 内容 |
| --- | --- |
| `time` | `dt`、`end_time` |
| `fluid` | `shape`、`lengths`、`origin`、`rho`、`mu` |
| 左室 `geometry` | 内外半轴、基底高度、中心、固体 `mesh_size` |
| 左室 `material`、`loads`、`beta` | Guccione 参数、内膜压力、主动张力、加载斜坡时间、基底弹簧 |
| 左室 `basal_constraint` | radial 径向约束或 spring 三方向弹簧 |
| 瓣膜 `solid` | 几何、FRH 参数、根部弹簧 |
| 瓣膜 `inlet` | 周期入口的 `amplitude`、`period`、`offset` |
| MAC `pressure_solver` | MG 容差、平滑次数、最大循环次数、检查间隔 |
| MAC `mass_solver` | IB 一致质量矩阵 PCG 容差、最大迭代、重算与检查间隔 |
| MAC `execution` | 执行、压力、质量求解等后端及热启动 |
| FEM `solver`、`backend` | 流体 PCG 参数及 CSR/积分点算子 |
| `output` | 日志、检查点、VTK 的步数间隔和开关 |
| FEM `history_every` | CSV 诊断采样间隔 |

`mesh_size` 是 Gmsh 的目标尺寸，不保证每个单元具有相同边长。
流体每轴空间步长是 `lengths[i]/shape[i]`；FEM 的 `shape` 是 Q2/Q1
单元数量，速度节点间距为单元边长的一半。左室生成纤维沿用原 Laplace
椭球 ±90° 配方和 x 长轴；瓣膜沿用原 ±45° 方向与固定积分规则。
新增参数接口没有更改这些离散方法。

## JSON 配置与命令行

优先级为：**主程序 CONFIG < JSON 指定字段 < 显式命令行参数**。
JSON 可以只写需要修改的字段；其余继承该 demo 的 CONFIG。
拼错字段名会报错，避免参数未生效却继续运行。

```bash
git pull --ff-only
conda activate afsi-torch
python -m pip install -e ".[test,geometry,fused]"
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_simulation_config.py

# 输出主程序的配置模板，立即退出，不启动模拟。
python demo/ideal_lv_fsi/run_mac.py --write-config lv.json
python demo/ideal_lv_fsi/run_fem.py --write-config lv_fem.json
python demo/ideal_valve_fsi/run_mac.py --write-config valve.json
```

`--write-config` 写主程序 CONFIG，或与 `--config` 合并后的模板；它不应用
用于实际运行的其他命令行覆盖。每次运行保存的 `configuration.json` 才是
所有覆盖后的实际参数，可直接作为同类 demo 下次新运行的 `--config`。

例如自行编写 `lv_custom.json`：

```json
{
  "time": {"dt": 0.000025, "end_time": 2.0},
  "fluid": {"shape": [64,64,64], "lengths": [5.0,5.0,5.0], "rho": 1.0, "mu": 1.0},
  "geometry": {"mesh_size": 0.1},
  "output": {"output_every": 800}
}
```

参数配置需要满足当前求解器限制；模板读入时会验证完整配置。命令行保留
`--fluid-cells`（三轴相同）及二维 `--nx/--ny`，新增三维 `--fluid-shape NX NY NZ`、
`--fluid-lengths`、`--fluid-origin`、`--rho`、`--mu`。材料和完整求解器选项使用
Python CONFIG 或 JSON。

```bash
# 三维 MAC：已验证优化路径；以下仅运行初期 0.005 s。
CUDA_VISIBLE_DEVICES=0 python -u demo/ideal_lv_fsi/run_mac.py \
  --config demo/ideal_lv_fsi/configs/mac_gpu.json \
  --end-time 0.005 --output results/lv_config_smoke

# 同一配置跑 2 s；终止时间来自 JSON，无需重复后端参数。
CUDA_VISIBLE_DEVICES=0 python -u demo/ideal_lv_fsi/run_mac.py \
  --config demo/ideal_lv_fsi/configs/mac_gpu.json \
  --output results/lv_config_2s

# FEM：采用 Q2/Q1 单元计数，而非 MAC 单元计数。
CUDA_VISIBLE_DEVICES=0 python -u demo/ideal_lv_fsi/run_fem.py \
  --config demo/ideal_lv_fsi/configs/fem.json \
  --end-time 0.005 --output results/lv_fem_config_smoke

# 二维瓣膜：3 s 配置，命令行可以覆盖空间和时间分辨率。
CUDA_VISIBLE_DEVICES=0 python -u demo/ideal_valve_fsi/run_mac.py \
  --config demo/ideal_valve_fsi/configs/mac_gpu.json \
  --nx 256 --ny 64 --dt 0.0000625 \
  --end-time 0.005 --output results/valve_config_smoke
```

三维主程序的 Python CONFIG 仍保留原参考执行默认；提供的 `mac_gpu.json`
显式选择此前优化过的 fused/pointwise/graph/optimized 路径。
GPU graph 配置需 CUDA；CPU 检查可选择三维 `pressure_backend='workspace'`
或 `torch`。二维 `auto` 会根据设备选择原有压力后端。

## 参数验证、输出和续算

MAC 网格必须能逐层粗化到最多 512 个粗网格单元，建议使用 2 的幂。
非立方网格已接入三维求解、结果导出和性能工具；单元尺度长宽比过大时，
原平滑器可能收敛较慢。二维流体原点固定 `(0,0)`，通道高度必须等于
`solid.height`，因为瓣叶根部连接上下壁面。

这里的生成理想 LV/瓣膜默认 MAC 黏性项显式推进，配置会检查黏性时间步上限；
细化网格时可能需要减小 dt。真实 LV 的 CNAB 方案采用 CN 黏性和显式 AB2
对流，不使用显式黏性限制；显式对流及弹性/IB 的限制仍需检查。
对流、IB 支持区域、单步位移和固体变形检查仍在推进中执行。
FEM 的黏性项使用原隐式求解，不套用 MAC 的黏性时间步限制。
终止时间需要为 dt 的整数倍。

输出频率以步数设置，实际帧间隔为 `output_every*dt`。例如 dt=5e-5、
output_every=400 对应 0.02 s；dt 减半后需要 output_every=800 才保持相同帧间隔。
二维旧开关 `--field-every 0` 关闭 VTK，三维可用 `--no-vtk`。
`output.fluid_fields` 是二维瓣膜的可选流体输出开关；三维左室输出固定包含
固体和流体两类空间场，该字段需保持 true。

新运行保存 `configuration.json`、`report.json` 和 `checkpoint.npz`。
报告中的 `configuration` 使用公共配置结构；检查点保留实际网格、材料、
载荷、流体参数、求解器选项和执行后端。为兼容旧工具，三维 settings 中的
`fluid_cells/box_length` 保留为第一轴值；新代码使用 `fluid_shape/fluid_lengths`
和完整坐标，不依据旧标量推断非立方网格。

```bash
# 从同一目录续算；物理配置和 dt 来自检查点，不再指定 --config。
CUDA_VISIBLE_DEVICES=0 python -u demo/ideal_lv_fsi/run_mac.py \
  --device cuda --resume results/lv_config_smoke/checkpoint.npz --end-time 2.0
```

续算可以修改终止时间、输出采样与兼容的执行后端；不能替换几何、网格、
dt、材料或载荷。改变这些物理设置需要新运行目录。旧版没有新增字段的
检查点仍恢复原立方网格、原入口波形和原容差。已有 examples 命令及
`from examples.lv_mac import run` 等导入保留，新增代码建议使用包内 runner。

## 材料与边界扩展

配置负责选择参数；添加新的物理模型时使用
[统一固体接口](SOLID_API.md)。共享 3D 耦合求解器只依赖节点力、CSR 切线和
有效性检查，不限定真实左心室或 H–O。本构、边界力与算例载荷分别可替换；
可组合 P1 实现和 `examples/custom_solid_mac.py` 展示具体接入。
任意新回调的保存/恢复需要其自己的配置重建逻辑，现有心室/瓣膜检查点
并不自动序列化任意新材料。
