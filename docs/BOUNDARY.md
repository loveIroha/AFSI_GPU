# 第四步：P2 表面压力与基底约束

## 与 afsi 的对应

选定 afsi demo_337 收缩例子的固体节点力包含三部分：

$$g_a=-\int_{\Omega_0}P\nabla_XN_a\,dV_0
-\int_{\Gamma_{BASE,0}}\beta N_a(x-X)\,dA_0
-\int_{\Gamma_{ENDO,0}}N_a p\,\operatorname{cof}(F)N_0\,dA_0.$$

本步加入后两项。P 已包含 Guccione 与给定主动应力。示例代码中称为 circum_constraint 的向量实际上包括 x、y、z 三个坐标差，因此这里实现三方向弹簧，并非只限制圆周分量，也不是强制 Dirichlet 固定。

这是给定变形下的完整固体弱式节点力计算。尚未求解未知位移、更新流体或进行耦合时间推进。

## 外表面提取与标记

`extract_boundary(X,cells)` 对每个 P2 四面体的四个面按顶点全局编号去重：相邻单元的共享面剔除，剩余面为固体边界，包括内腔表面。根据所属四面体的对顶点检查方向并翻转三角形顺序，得到固体外法向。

每个面为六个节点：三个顶点后跟边 01、02、12 的中点。共享边若使用不同中点自由度，会拒绝该非协调网格；重复单元和非流形面也会报错。网格仍假设没有单元重叠等几何缺陷，此处不是完整网格质量认证。

返回的面行可以按参考坐标或未来导入的 facet tags 选择，分别准备 BASE、ENDO。示例只用几何规则生成测试标记，不是心脏真实边界标记导入。`prepare_surface` 拒绝空区域、重复面、退化参考三角形和非中点参考边节点。传入的面必须是上述提取器返回的有向面；直接给任意连接关系时，预处理无法推断哪一侧属于固体。

## 压力：随动方向与面积

面上坐标插值为 x(r,s)=sum_a N_a(r,s)x_a，定义有向面积向量

$$a(x)=\partial_rx\times\partial_sx.$$

Nanson 关系给出

$$a(x)\,dr\,ds=\operatorname{cof}(F)N_0\,dA_0.$$

因此压力直接按

$$g_a^{pressure}=-\sum_qw_qN_a(q)p_q a(x_q)$$

组装。既不归一化 a，也不再乘一次当前或参考面积。这样自然保留 P2 当前曲面的法向与面积变化，避免单位法向替代面积映射。

正 p 沿固体外法向的反方向作用。内腔面固体外法向指向腔内，故 -pN 把固体推离腔内。压力方向不是通过“正 z”一类全局约定指定。

压力幅值是给定的参考标量场，保持独立于当前 x；可以随时间重新赋值，但尚未实现流体压力反馈。

## 基底弹簧：参考面积

$$E_{base}=\tfrac12\int_{\Gamma_{BASE,0}}\beta|x-X|^2dA_0,
\qquad g_a^{base}=-\int_{\Gamma_{BASE,0}}\beta N_a(x-X)dA_0.$$

弹簧使用参考面积，不会随着当前表面拉伸额外改变积分测度。它的节点力可独立用能量求导核对。beta 的单位为应力/长度；p 的单位为应力。beta=0 表示无弹簧，beta 有限时不等同于严格固定边界。

`prepare_coefficient(surface,value)` 接受标量、全局节点值 (N,) 或积分点值 (B,Q)。对刚度使用 `nonnegative=True`，包括检查插值后的值；P2 插值可能过冲，不会通过静默裁剪改变场。调用纯力核时应传入这些已验证值。

## 积分和切线

默认三角形 degree=4 Duffy/Gauss 规则有 9 个正权积分点。对直边参考三角形、二次当前坐标、常量 p/beta，压力和弹簧节点力的被积式最多四次，因此该规则足够精确。若 p 或 beta 是 P2 空间场，节点力最多六次，应使用 degree=6（16 点）。

开放受压面上的随动力一般没有可直接套用的标量势函数，其切线也可能不对称。因此总残量应由节点力构造并直接求 JVP：

```python
import torch
from afsi_torch import boundary as bd, solid

# X,cells,geometry,fields,parameters already prepared for the same P2 mesh.
faces = bd.extract_boundary(X, cells)
base_mask = (X[faces[:, :3], 2].abs() < 1e-12).all(-1)
load_mask = (X[faces[:, :3]].sum(-1) > 0.9).all(-1)  # Unit-tetra TEST tag only.
base = bd.prepare_surface(X, faces[base_mask])
loaded = bd.prepare_surface(X, faces[load_mask])
pressure = bd.prepare_coefficient(loaded, 1200.)
beta = bd.prepare_coefficient(base, 5000., nonnegative=True)

def residual(x):
    g = solid.guccione_force(x, geometry, fields, parameters)
    g = g + bd.pressure_force(x, loaded, pressure)
    g = g + bd.spring_force(x, base, beta)
    return -g

solid.validate_deformation(x, geometry)
bd.validate_surface(x, loaded)
bd.validate_surface(x, base)
_, Kv = torch.func.jvp(residual, (x,), (direction,))
```

闭合、有一致方向的受压表面在常量 p 下可以使用 p*体积的势函数验证；测试中使用这一独立恒等式，但不把它套到开放 ENDO 区域。未来隐式解法不能默认完整切线对称正定，因此也不能无条件采用 CG。

`validate_surface` 只在积分点检查表面是否退化；它不证明固体体积未翻转、表面无自交或全局单射。还需单独执行固体变形验证。现阶段没有生产级网格修复或时间步控制。

## 测试及独立对照

- 三角形节点插值、多项式与梯度再现、解析单项式积分。
- 外边界提取、相反四面体编号、不协调边和非流形检查。
- 平面均匀压力：P2 顶点载荷为零，三个边节点各承受合力的三分之一。
- 仿射变形的余子式面积映射，以及二次曲面的解析压力合力。
- 闭合面力/力矩平衡和体积梯度恒等式。
- 开放压力面的差分、JVP 和非对称切线。
- 弹簧刚体平移、参考面积缩放与能量梯度。
- Guccione+主动应力+压力+弹簧总残量的切线差分，以及 CPU/GPU 对照。

`validation/compare_boundary.py` 使用 Basix 的完整 P2 四面体基函数在面上制表，通过体积变形梯度计算 `det(F)*inv(F).T*N0`，独立核对表面迹的叉乘实现。包含变系数压力/刚度及多面共享节点组装。它仍共享输入网格拓扑和积分点，不是实际 DOLFINx 全局自由度/边界标记的端到端验证。

## 来源

- [afsi 的三项固体弱式，固定提交](https://github.com/npuheart/afsi/blob/99df0ffba795fa05043ba874ad00353dcb986466/afsic/demo/demo_337/fsi_paralell_fibers_contraction.py)
- [Basix 0.10 创建和制表 API](https://docs.fenicsproject.org/basix/v0.10.0/python/_autosummary/basix.html)
