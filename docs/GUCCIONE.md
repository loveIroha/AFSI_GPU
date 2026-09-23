# 第三步：Guccione、纤维场与主动应力

## 对齐范围

对齐 afsi `demo_337` 收缩例子的 `deviatoric=False` 被动材料与给定主动张力。代码按数学公式独立实现，不包含复制的 afsi 源文件。尚未运行完整 DOLFINx/afsi 求解器，也未加入表面压力、基底约束、主动张力时间演化、流体和 IB。

本实现不提供 deviatoric 参数，避免把原示例未启用的分支与常见等体积修正混淆。参数对应如下：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| C | 20000 | 应力尺度 |
| bf | 8 | 纤维向非线性系数 |
| bt | 2 | 横向非线性系数 |
| bfs | 4 | 纤维相关剪切系数 |
| kappa | 500000 | 体积惩罚系数 |

C、kappa 和 Ta 使用相同应力单位；其他系数无量纲。

## 被动材料

当前坐标为 x，参考坐标为 X；F=grad_X(x)，不是 I+grad_X(x)。定义

$$E=\tfrac12(F^TF-I),\qquad J=\det F,\qquad E_{ab}=a_0^TEb_0,$$

$$Q=b_fE_{ff}^2+b_t(E_{ss}^2+E_{nn}^2+2E_{sn}^2)+2b_{fs}(E_{fs}^2+E_{fn}^2),$$

$$W_p=\frac C2(\exp Q-1)+\kappa(J-1)^2.$$

计算用 expm1(Q) 减少 Q 接近零时的消减误差。体积项没有额外 1/2 系数，与选定 afsi 例子一致。

令 A=[f0,s0,n0]，E_local=A^T E A，B 的对角为 (bf,bt,bt)，纤维剪切位置为 bfs，其余片层-法向剪切位置为 bt。解析应力由

$$S=C\exp(Q)A(B\odot E_{local})A^T,$$

$$P_p=FS+2\kappa(J-1)JF^{-T}$$

计算。能量自动求导和解析应力组装是两个验证路径；额外使用 NumPy 标量公式的复步长导数建立独立参照。

## 纤维场与张力场

`prepare_reference_fields` 接受常量方向 `(3,)` 或 P2 节点方向 `(N,3)`，张力接受标量或 P2 节点值 `(N,)`。它把数据转到几何所在 device/dtype，生成积分点数据 `(E,Q,3)` 和 `(E,Q)`：

$$f_q=\sum_aN_a(q)f_a,\qquad s_q=\sum_aN_a(q)s_a,\qquad n_q=s_q\times f_q.$$

先插值，再叉乘；不能替换成节点叉乘的插值。这里不归一化、不正交化，也不裁剪 P2 张力插值的可能过冲，以保留所给离散场。零方向、平行方向和非有限值在预处理中报错。有效性仅在积分点检查。

常量 Ta 与 afsi 示例相同；节点 Ta 是额外的给定空间场能力。本步不会求解激活方程。张力或参考方向改变后，应重新准备相应 fields；求 x 的导数时保持这些输入固定。fields 必须与同一份 geometry 配套使用。

## 主动应力和势函数的适用条件

$$P_a=T_a(Ff_0)\otimes f_0.$$

当 Ta 与参考方向相对于当前 x 固定时，可以构造瞬时势函数

$$W_a=\tfrac12T_a\big((Ff_0)\cdot(Ff_0)-f_0\cdot f_0\big),\qquad \partial W_a/\partial F=P_a.$$

因此可以对 `sum(w*(Wp+Wa))` 求导验证总节点力。Wa 在参考态为零，但主动应力通常非零；不能用“刚体运动零总应力”作为主动材料的验收标准。

这个势函数不是整个激活周期的储能定律。若 Ta 依赖长度、速度、钙动力学或当前构形，必须明确内部变量与线性化策略，不能直接沿用固定 Ta 的能量解释。

## 有限元接口

```python
import torch
from afsi_torch import solid
from afsi_torch.fields import prepare_reference_fields
from afsi_torch.materials import GuccioneParameters

# X and cells are P2 reference data already on the target device.
geometry = solid.prepare_p2(X, cells, degree=4)
fields = prepare_reference_fields(
    geometry, fiber=[1., 0., 0.], sheet=[0., 1., 0.], tension=1000.)
parameters = GuccioneParameters()
x = X.clone()
solid.validate_deformation(x, geometry)

potential = lambda y: solid.guccione_energy(y, geometry, fields, parameters)
residual = torch.func.grad(potential)          # R_int = +dPi/dx
g_ad = -residual(x)
g_pk1 = solid.guccione_force(x, geometry, fields, parameters)
direction = torch.sin(x)
_, Kv = torch.func.jvp(residual, (x,), (direction,))
```

组装仍为 `g=-sum(w*P*grad(N))`，结果是积分后的节点力，不是加速度，也不能再次乘节点体积。材料层处理 `(E,Q,3,3)` 张量，方向插值及几何预处理一次完成；每次力计算没有 Python 单元循环或 NumPy 数据转换。

## 验证与局限

新增测试覆盖：被动参考态和旋转、单轴解析应力与各向异性、非正交方向下本构导数、主动应力的大小和收缩符号、方向插值顺序、节点力和切线有限差分、CPU/GPU 一致性。

独立脚本 `validation/compare_guccione.py` 使用 Basix 制表及 NumPy 标量应变分量公式，借助复步长求 PK1，再逐单元组装。双方使用相同积分点，隔离积分规则差异。这是公式与体积分验证，不能替代 DOLFINx 自由度映射和完整弱式验证。

默认 degree=4 的 Duffy 规则有 64 个积分点，与 afsi/Basix 的默认 degree=4 规则并非同一规则。Guccione 的指数积分一般不精确，后续逐值对照必须统一积分点或进行积分收敛研究。大变形时指数可能溢出；当前核不会通过裁剪 Q、J 或应力改变本构来掩盖失效。尚无时间步控制、Newton 线搜索或生产规模性能结论。

## 来源

- [afsi Guccione 公式，固定提交](https://github.com/npuheart/afsi/blob/99df0ffba795fa05043ba874ad00353dcb986466/afsic/demo/demo_337/Guccione.py)
- [afsi 纤维收缩例子，固定提交](https://github.com/npuheart/afsi/blob/99df0ffba795fa05043ba874ad00353dcb986466/afsic/demo/demo_337/fsi_paralell_fibers_contraction.py)
- [PyTorch JVP](https://docs.pytorch.org/docs/2.14/generated/torch.func.jvp.html)
