# 固体接口与算例迁移

3D MAC 耦合驱动依赖固体的力、切线与有效性接口。耦合求解器不再选择
H–O 材料或限定 `RealLVSolid` 类型；材料与边界属于固体模型，载荷、网格、
求解参数属于算例。真实左心室只是其中一个适配器。

## 项目职责

| 层 | 文件/目录 | 职责 |
| --- | --- | --- |
| 算例入口 | `demo/*/run_mac.py` | 配置网格、材料、载荷、dt、流体域与输出 |
| 配置 | `config.py`、算例的配置类 | 参数验证、JSON 与命令行覆盖 |
| 网格输入 | `mesh_io.py`、`geometry/` | 外部网格/方向导入或 CPU 几何生成 |
| 固体契约 | `solids/contracts.py` | 节点力、CSR 切线、有效性、可选执行与诊断接口 |
| 可组合 P1 固体 | `solids/p1_model.py` | 可替换 PK1 回调、单元字段、边界力和检查条件 |
| 通用 P1 组装 | `solids/csr.py` | 单元局部自动微分、缓存稀疏模式、GPU CSR 组装 |
| 真实左心室适配器 | `real_lv.py`、`holzapfel_ogden.py`、`ho_tangent.py` | 用户 H–O 本构、方向、内膜压力与底部约束 |
| 理想左心室适配器 | `lv_model.py` | Guccione/P2 固体及对应执行内核 |
| 3D 流体/IB/耦合 | `mac/` | MAC、CN–AB2、质量矩阵、传播/插值与非线性推进 |
| 2D 流体/IB/耦合 | `mac2d/` | 二维瓣膜使用的独立执行路径 |
| 实验运行与保存 | `simulation/`、各算例 checkpoint 模块 | 时间循环、输出、恢复与输入身份校验 |

现有模块路径保留，旧 demo、配置与检查点无需迁移。真实 LV 的专用运行文件
负责输入、腔体积诊断和保存，而不是决定共享 3D 耦合器可以支持什么材料。

## 固体模型的最低要求

模型提供：

- `mesh.X`：参考坐标 `(n,3)`；`mesh.cells`：设备上的 int64 连接关系。
- `geometry`：准备好的参考有限元几何，供相同的质量矩阵和 IB 传递使用。
- `force(x,time)`：当前**坐标**和秒制时刻对应的积分节点力 `(n,3)`。
- `validate(x)`：拒绝负 J、非有限值或模型特有的无效状态。
- `tangent_factory(chunk_size)`：隐式弹性所需的组装器，其
  `assemble(x,time)` 返回 `d force/d x` 的 `(3n,3n)` CSR，
  `diagonal(CSR)` 返回同一节点/分量顺序的对角线。

内部力采用 `-∫P:∇v dV`；切线是这一节点力及全部边界力的导数。
不能把正号刚度、材料张量或质量加权力密度直接作为该切线。
全局固体切线以 CSR 组装；自动微分只用于小单元材料/边界核。
耦合中的压力消元与 Krylov 作用仍通过现有 Stokes/IB 响应执行。

`execution_backend='torch'` 使用普通模型回调。选择 `fused` 时模型提供
`execution_factory()`，返回 `force/validate` 执行器，并保留 `model`。
`coupling_backend='optimized'` 还需要 `force_with_geometry` 与
`remember_geometry`，几何元组最后一项是有效性布尔标志。
可选 `pointwise_execution_factory()` 支持专用点内核；缺少能力时明确报错，
不会隐式改成不同算法。可选 `failure_diagnostics(grid,state,problem)`
提供模型特有诊断，缺省仅报告通用接受状态与计数。

可选 `checked_geometry(x)` 仅返回同一张量、同一版本下已通过验证的几何，
否则返回 `None`；共享耦合器可用它减少重复计算，未实现时正常回退。
网格、参考字段和有效性规则在准备后的驱动生命周期内应保持不变。

## 添加另一个材料和边界

可直接使用 `P1Solid`，无需从心室类继承。
[可运行示例](../examples/custom_solid_mac.py) 使用 St. Venant–Kirchhoff 本构
和独立弹簧/节点载荷，通过与真实 LV 相同的 3D CN–AB2 非线性中点耦合器推进：

```bash
python -u examples/custom_solid_mac.py --device cpu --steps 3
CUDA_VISIBLE_DEVICES=0 python -u examples/custom_solid_mac.py \
  --device cuda --steps 10 --dt 1e-4 --fluid-cells 16
```

替换示例中的 `material(F,fields,time)` 返回单个单元的 PK1 应力 `(3,3)`；
`cell_fields` 每个张量第一维必须是单元数，可携带材料系数和 DG0 方向。
使用 `BoundaryForce` 注册局部边界节点力回调和边界字段。回调返回**积分后的
节点力**，如需面积积分、表面 Gaussian 规则或 follower 载荷，应在回调内
实现或编写对应适配器。示例的指定节点力不是压力面积积分。

回调必须使用纯 tensor 运算，不应 `.item()`、转 NumPy 或依据 Tensor 值
执行 Python 分支；P1 实现通过 `vmap`、局部 `jacrev` 与 `torch.compile`
运行材料和边界核。所有字段与坐标使用相同设备和浮点类型。

`validity_checks` 和边界的 `validity` 可以补充纯 Tensor 检查，例如有效
表面/腔体积。P1 默认仅保证单元 J 正且有限，不假设固体一定是心室。
新固体应确认所有边界力都包含在切线中，并检查传递支撑域位于流体壁内。

## 支持范围与不变的数值方法

通用可组合实现目前支持 **3D affine P1 四面体、单元常量字段**。
更高阶位移、空间变化方向或混合不可压材料需要相应几何、积分与切线适配器，
不能仅替换回调。现有 P2 理想 LV/2D 瓣膜保持原实现；接口并不表示它们
已经自动支持新的 P1 切线或三维隐式弹性方案。

真实 LV 保留原 H–O 力、CSR 切线、follower 压力、径向底部约束和执行内核。
CN 黏性、AB2/PPM 对流、预测中点冻结 IB 几何、非线性中点固体力、
Anderson/CSR Newton、质量矩阵、积分规则、容差及检查点内容均不改变。
这次架构调整只将模型选择改成工厂接口。

任意 Python 回调不能自动安全序列化。现有检查点恢复仍由相应算例 codec
重建已知模型；新材料/边界应提供自己的参数配置和重建逻辑，不使用 pickle
保存任意执行对象。通用模型也需要自己的输出字段/几何诊断适配，不能直接
借用专用心室的 cavity、fiber/sheet 字段。

## 验证

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/test_solid_api.py tests/test_cycle_completion.py \
  tests/test_mac_semiimplicit.py tests/test_mac_implicit.py
```

接口测试覆盖另一材料/边界的 CSR–JVP–有限差分一致性、两种隐式耦合器、
优化/普通执行轨迹与原 H–O 组装器一致性。完整真实 LV 三周期需用实际
患者输入执行，[三周期运行与检查](../demo/real_lv_fsi/README.md#完整三周期测试)
提供命令；短段兼容测试不能代替完整周期稳定性验证。
