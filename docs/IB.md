# 四点 IB 插值与力散布

0.6.0 实现规则三维速度节点格点上的 Peskin 四点核。沿每个方向取 4 个邻点，每个固体节点涉及 64 个流体速度节点。核计算、gather 和 index_add 均为 PyTorch 张量运算，可在 CPU/CUDA 上执行。

## 定义与单位

令 x_a 是当前固体节点，y_i 是背景速度格点，h 是三个方向的格点间距，dV=h_x h_y h_z。定义无量纲权重：

```text
H[a,i] = phi((x_a[0]-y_i[0])/h_x)
       * phi((x_a[1]-y_i[1])/h_y)
       * phi((x_a[2]-y_i[2])/h_z)

U = H u                  # interpolate: 固体节点速度
b = H^T g                # spread_load: 流体节点对偶载荷
f = H^T g / dV           # spread_density: 规则格点上的力密度
```

这里 g 是已有固体弱式装配得到的**积分节点力**。它已经包含体积/面积积分，不能再乘一次固体体积或积分权重。若外部输入的是固体力密度，则须先明确积分权重并转为积分力。

`phi` 是偶函数，r=|distance|：

```text
0 <= r < 1: (3 - 2r + sqrt(1 + 4r - 4r^2))/8
1 <= r < 2: (5 - 2r - sqrt(-7 + 12r - 4r^2))/8
r >= 2:     0
```

完整支持域下，权重满足零阶与一阶矩条件，因此再现常量/线性速度，保持总力与力矩，并有：

```text
sum_i u_i dot b_i = sum_a U_a dot g_a
dV * sum_i u_i dot f_i = sum_a U_a dot g_a
```

这是瞬时离散功率恒等式，不是完整时间推进稳定性的证明。例子使用给定速度场，不求解 Navier–Stokes。

## 与 afsi 的对应及流体质量矩阵

核宽度 4、插值加权求和，以及散布时除以 h_x h_y h_z，对应原 afsi 的 [IBMesh3D](https://github.com/npuheart/afsi/blob/99df0ffba795fa05043ba874ad00353dcb986466/afsic/src/coupling/IBMesh3D.h) 和 [Peskin 核](https://github.com/npuheart/afsi/blob/99df0ffba795fa05043ba874ad00353dcb986466/afsic/src/coupling/spatial/kernel_expression.h)。原程序把固体装配向量传入 `solid_to_fluid`，其粒子权重设为 1；本实现同样不重复乘固体积分权重。代码根据数学公式独立编写，未复制其 C++ 实现。

必须区分以下两种流体 RHS：

| 接口 | 数学对象 | 接入方式 |
| --- | --- | --- |
| spread_load | b=H^T g | 已是弱式节点载荷，直接加入 RHS |
| spread_density | f=H^T g/dV | 若作为 FE Function 系数装配，弱式 RHS 为 M_f f |

原 [ChorinSolver](https://github.com/npuheart/afsi/blob/99df0ffba795fa05043ba874ad00353dcb986466/afsic/src/afsic/euler/ChorinSolver.py) 使用 `inner(f,v)*dx`，对应第二行。一般一致质量矩阵 M_f 不等于 dV I，因此不能据本轮功率测试声称整个原始 FE 耦合已经能量一致。后续流体阶段将显式验证 M_f、两种 RHS 的差别，并记录默认模式；不能静默替换离散。若希望一致质量内积下 adjoint density，需要解 M_f f=H^T g。

## 网格与边界约定

- `UniformGrid.shape=(nx,ny,nz)` 是速度节点数量。Q2 背景网格每方向有 n 个单元时，速度节点数为 2n+1，间距为 L/(2n)，不是 L/n。
- 格点索引为 `i + nx*(j + ny*k)`，x 方向最快；三分量速度存为 `(Nf,3)`。导入流体 FE 自由度时必须显式建立此映射。
- 支持不同轴向间距和非零原点，统一用 `(x-origin)/spacing`。
- 当前要求每轴 4 个 stencil 节点都在格点范围内；精确检查区间为 `1 <= (x-origin)/h < shape-2`。靠边界时直接报错，不截断或重新归一化。实际左室与背景箱边界应保留足够缓冲并考虑最大变形。
- 当前没有周期边界、墙面核修正、MPI 或多 GPU。不得把邻近边界的支持域丢失解释为物理边界条件。
- 固体坐标变化后必须重新调用 `prepare_stencil`。旧 stencil 不会自动追踪坐标更新。

## 使用与验证

```python
from afsi_torch import ib

grid = ib.UniformGrid((65, 65, 65), (.001, .001, .001), (-.032, -.032, -.032))
stencil = ib.prepare_stencil(current_solid_coordinates, grid)
solid_velocity = ib.interpolate(fluid_velocity, stencil)
fluid_density = ib.spread_density(integrated_solid_force, stencil)
# 若流体离散直接接受弱式载荷，使用 ib.spread_load。
```

```bash
CUDA_VISIBLE_DEVICES=0 python examples/ib_patch.py --device cuda
CUDA_VISIBLE_DEVICES=0 python validation/compare_ib.py --device cuda
CUDA_VISIBLE_DEVICES=0 python -m pytest -q
```

当前完整 pytest 在 CUDA 可用时应为 **113 passed**。独立 NumPy 对照遍历全部流体节点、直接计算标量核，不复用 PyTorch stencil 或 scatter；它不运行原 afsi C++ 二进制，因此不能称为原程序端到端对照。

每个固体节点保存 64 个 int64 索引与 64 个权重；FP64 下两者合计约 1024 字节/节点，不含构建临时量和 gather 临时量。大型网格后续需要分块。预处理中的有效性检查会产生少量主机同步；尚未做吞吐优化、性能基准或跨设备传输优化。GPU index_add 的浮点累加顺序可能不同，按误差容限比较，不要求逐位相等。
