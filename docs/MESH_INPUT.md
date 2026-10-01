# 外部四面体网格与纤维读取接口

`afsi_torch.mesh_io.read_solid_mesh` 读取静态 P1 四面体 XDMF，
包括 FEniCS 写出的 XDMF/HDF5，以及旧版 DOLFIN 的 XML 边界和逐单元纤维数据。
不需要安装 FEniCS/DOLFINx，不调用网格生成程序，不改动输入文件。

本接口完成数据读取、校验、可选 P2 提升和材料积分点字段准备。
它本身不启动真实左心室 FSI；生成几何的 demo 不会自动切换到外部网格。
已经提供独立的 [真实左心室 H–O/P1 demo](../demo/real_lv_fsi/README.md)，
包含确认后的边界映射、片层导入、流体设置以及 DG0 检查点和可视化。
其他真实算例仍需明确自己的材料、边界和载荷，不能由一个网格文件自动确定。

## 安装与文件组织

```bash
conda activate afsi-torch
python -m pip install -e ".[test,mesh]"
```

`mesh` extra 包含 `h5py` 和 `meshio`；只有 XDMF/HDF5 读取需要 h5py，
检查脚本的 VTK 输出需要 meshio。PyTorch 与 NumPy 是项目基础依赖。

保持 XDMF 和它引用的 HDF5 文件相对路径不变。例如：

```text
data/real_lv/
  mesh_scale.xdmf
  mesh_scale.h5
  boundaries.xml
  fibers_0.xml
  fibers_1.xml
  fibers_2.xml
```

`boundaries.txt` 如果内容也是 DOLFIN XML，可以替代 `boundaries.xml`；
二者不用同时传入。输入数据不随项目分发，也不用上传 GitHub。

## Python 接口

```python
from pathlib import Path
from afsi_torch.mesh_io import read_solid_mesh

folder = Path("data/real_lv")
mesh = read_solid_mesh(
    folder / "mesh_scale.xdmf",
    units="cm",  # 必须确认原始单位；也支持 mm、m，并转换到 cm
    boundaries=folder / "boundaries.xml",
    fiber_components=[folder / f"fibers_{i}.xml" for i in range(3)],
    # 只有确认源标签 1=外膜、2=内膜、3=底部后才采用下面的映射：
    tag_map={1: 2, 2: 1, 3: 3},
    device="cpu",
)

# mesh.X: (N,3)，厘米、float64
# mesh.cells: (E,4)，int64，原始单元及局部顶点顺序
# mesh.faces: (B,3)，朝固体外侧的边界三角形
# mesh.facet_tags: (B,)，映射后的边界标签
# mesh.boundary_cells / boundary_local_facets: 原始所属单元/局部面编号
# mesh.fiber: (E,3)，原始 DG0 纤维，未平均、未归一化
# mesh.sheet: None，除非显式提供 sheet_components

# 不改变四面体数量，不改变逐单元方向；每条共享边只添加一个中点。
p2 = mesh.to_p2().to("cuda")
# p2.cells: (E,10)，v0,v1,v2,v3,e01,e02,e03,e12,e13,e23
```

`units` 为必填参数，接口不从文件名或几何大小猜测单位。
`translation_cm=(dx,dy,dz)` 在单位转换后平移网格，默认不平移。
不执行隐式缩放、旋转或重新生成纤维。

没有 `tag_map` 时保留原始标签。项目内部左心室使用 **1=内膜、2=外膜、3=底部**；
外部文件的数字意义必须由用户确认，不能仅凭数字自动匹配。
`surface(tag)` 选择对应的外边界。

## 命令行检查与 ParaView 输出

以下命令保留源边界标签。将 `--units cm` 改为实际原始单位：

```bash
python -u examples/read_lv_mesh.py \
  --mesh data/real_lv/mesh_scale.xdmf --units cm \
  --boundaries data/real_lv/boundaries.xml \
  --fibers data/real_lv/fibers_0.xml data/real_lv/fibers_1.xml data/real_lv/fibers_2.xml \
  --p2 --output results/imported_real_lv
```

如果已经确认标签意义，可以额外传入：

```text
--endo-tag 2 --epi-tag 1 --base-tag 3
```

三个标签参数必须同时提供且互不相同。省略 `--p2` 时保留 P1 网格。
本脚本在 CPU 上读取和输出，不启动流体求解、力学迭代或耦合时间推进。

输出：

- `report.json`：源文件、单位变换、节点/单元数量、边界计数、包围盒、壁体积与纤维范数。
- `solid.vtu`：参考网格、逐单元纤维、原始单元编号和单元体积。
- `boundary.vtu`：有向外边界、标签、所属单元和局部面编号。

可在 ParaView 中查看边界标签与纤维（Cell Data），先确认内外膜和底部。
若只想检查原始文件的拓扑和坐标，可用 `read_xdmf(path)`；它返回未经单位转换的
NumPy 坐标和连接数组。完整网格校验由 `read_solid_mesh` 完成。

## 逐单元纤维与现有有限元组装

本次参考文件的三个 `fibers_*.xml` 是 `dim=3` 的标量 MeshValueCollection，
每个分量通过 `cell_index` 定位，合并后是 **DG0 向量 (E,3)**。
它们不是节点 P1/P2 函数，也不是未知排序的全局 DOF 数组。

边界文件是 `dim=2` 的 MeshValueCollection，通过 `(cell_index, local_entity)`
定位四面体面。DOLFIN 的局部面编号 `i` 表示与局部顶点 `i` 相对的面。
接口保留原始单元顺序，用这个约定读取标签，再将外边界定向为朝固体外侧。
参考：[FEniCS MeshValueCollection](https://olddocs.fenicsproject.org/dolfin/1.4.0/cpp/programmers-reference/mesh/MeshValueCollection.html)、
[DOLFIN 四面体局部实体定义](https://github.com/live-clones/dolfin/blob/2019.1.0/dolfin/mesh/TetrahedronCell.cpp)、
[FEniCS XDMFFile](https://olddocs.fenicsproject.org/dolfin/2017.1.0/python/programmers-reference/cpp/io/XDMFFile.html)。

Guccione 模型还需要独立的片层方向。可以传入三份 DG0
`sheet_components` XML，或通过明确选择的规则准备 `(E,3)` 的 sheet：

```python
from afsi_torch import solid
from afsi_torch.materials import GuccioneParameters

geometry = solid.prepare_p2(p2.X, p2.cells)
# sheet_per_cell 应由用户提供或按明确的材料方向规则生成。
fields = p2.reference_fields(geometry, sheet=sheet_per_cell, tension=0.0)
force = solid.guccione_force(p2.X, geometry, fields, GuccioneParameters())
```

`reference_fields` 将同一单元的方向原值广播到该单元的积分点，
沿用现有 `cross(sheet,fiber)` 法向约定；不做节点平均、归一化或正交化。
没有 sheet 时会明确报错，不会假造患者片层方向。
现有 `LVSolid` 的节点 FiberField 接口和 checkpoint 不应直接接收这些 DG0 数组。

## 当前支持范围与校验

- 静态、单个网格的 P1 四面体 XDMF，XYZ 几何；DataItem 支持 XML 或 HDF5。
- 多网格文档需要 `grid_name`；时间序列、曲线/P2 输入几何及其他单元暂不支持。
- DOLFIN XML MeshValueCollection：逐面整数标签、完整逐单元浮点分量。
- 边界必须覆盖外表面；一般数据读取可显式设 `require_full_boundary=False`。
- 未知全局实体/DOF 排序的 MeshFunction、Function XML 不会被猜测性解释。
- 检查尺寸、索引、重复单元、退化单元、非流形面、缺失/重复方向分量、
  共享面标签冲突和误标的内部面。纤维数值与范数保持源值。

```bash
python -m pytest -q tests/test_mesh_io.py
```

测试涵盖 XDMF/HDF5 与内嵌 XML、乱序的纤维记录、两种四面体方向、单位转换、
标签映射、P2 共享中点和面顺序、DG0 积分点字段以及现有 Guccione 力/能量导数一致性。
GPU 字段/组装测试在有 CUDA 的环境中运行，否则跳过。
