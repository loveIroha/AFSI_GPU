# 程序生成理想左心室（0.7.0）

入口 `examples/ideal_lv.py` 不读取已有网格或纤维文件。Gmsh 在 CPU 上构造外椭球减内椭球、按基底平面截断并生成四面体；转换为共享边 P2 自由度后，可将张量传到 CUDA，计算纤维场、体积和给定变形下的固体节点力。

## 单位与默认尺寸

本例采用 cm–g–s，与选定 AFSI 示例的压力换算一致。有限元内核本身不自动换算单位。

| 量 | 单位 / 默认值 |
| --- | --- |
| 内椭球半轴 | (2.5, 2.5, 4.5) cm |
| 外椭球半轴 | (3.5, 3.5, 5.5) cm |
| 中心 | (0, 0, 0) cm |
| 基底平面 | 中心上方 1.5 cm，保留下方部分，心尖朝 −z |
| 网格目标尺度 | 1.2 cm |
| 体积 | cm³ = mL |
| 压力、材料应力、主动张力 | dyn/cm²；1 mmHg = 1333.22368421 dyn/cm² |
| 节点力、能量 | dyn、erg |
| 基底弹簧系数 | dyn/cm³ |

这些是数值验证参数，并非原始 AFSI 网格尺寸或患者标定。外径为 7 cm，后续流体域必须覆盖它并留足 IB 核支撑范围，不能直接沿用原例的 5 cm 背景盒。

## 运行

```bash
conda activate afsi-torch
python -m pip install -e ".[test,geometry]"
CUDA_VISIBLE_DEVICES=0 python examples/ideal_lv.py --device cuda --mesh-size 1.2 --output results/ideal_lv
CUDA_VISIBLE_DEVICES=0 python -m pytest -q
```

若 Linux 导入 Gmsh 报缺少 `libGLU.so.1`，Ubuntu/Debian 可安装 `sudo apt-get install libglu1-mesa`。无 GPU 时明确指定 `--device cpu`；指定 CUDA 不会自动回退 CPU。安装 geometry 依赖且 CUDA 可用时，完整测试为 131 项；本地 CPU 为 74 passed、57 CUDA skipped。

尺寸可通过 Python 配置：

```python
from afsi_torch.geometry import LVConfig, generate_lv, rule_based_fibers
config = LVConfig(inner_axes=(2.5, 2.5, 4.5), outer_axes=(3.5, 3.5, 5.5),
                  base_height=1.5, mesh_size=0.9)
mesh = generate_lv(config, device="cuda")
fields = rule_based_fibers(mesh.X, config)
```

输出包括 `reference.vtu`、`surfaces.vtu`、`prescribed_deformation.vtu`、`generated.npz` 和 `report.json`。VTU 可用 ParaView 查看，表面标签为 ENDO=1、EPI=2、BASE=3；所有法向均为固体外法向，ENDO 指向腔内。NPZ 是生成结果归档，不是生成入口的输入。报告记录单位、几何参数、Gmsh 版本和数值检查。

可选静态预览：

```bash
python -m pip install -e ".[io]"
python scripts/preview_lv.py --source results/ideal_lv/generated.npz --output results/ideal_lv/preview.png
```

## 离散与限制

参考网格为直边四面体，P2 边节点取中点，不投影到椭球曲面，以匹配现有固体核。默认解析腔体体积为 87.2665 mL；本地 h=1.2 cm 网格为 82.7581 mL，h=0.9 cm 为 84.4422 mL。加密减小曲面近似误差，但这些粗网格尚不能作为几何收敛结果。

跨壁螺旋角默认从内膜 +60° 变到外膜 −60°。心尖采用平滑正则化，纤维在局部允许离开切平面，避免环向方向奇点；片层是数值上的正交补全，并非经过生理标定的片层模型。积分点仍先插值 fiber、sheet，再计算 `normal=cross(sheet,fiber)`，不额外归一化改变已有材料定义。

腔体体积使用内膜加虚拟基底封口。封口由当前开口边界和中心点构成 P2 三角扇，保留曲边边界；基底非平面运动时，体积依此明确约定定义。封口仅用于测量，既不是固体单元，也不承受内膜压力。开口移动时，体积梯度不能直接当成仅内膜的压力节点力。

示例施加给定仿射变形，检查正体积、det(F)、保守力与能量梯度、体积变换，并输出被动、给定主动张力、随动压力和基底弹簧合力。它尚未求解静力平衡、流体或心动周期。

测试另外覆盖标签/边界拓扑、三轴椭球和平移、网格加密、纤维无退化、压力虚功、体积导数、CGS/SI 力与能量换算、VTK 节点顺序和 CPU/CUDA 一致性。GPU 项仍须在目标机器实际执行。

几何 API 依据：[Gmsh 官方手册](https://gmsh.info/doc/texinfo/gmsh.html)。
