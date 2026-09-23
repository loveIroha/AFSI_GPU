# 验证记录

## 0.3.0：Guccione 与给定主动张力

本地环境仍为下述 Windows CPU 环境。完整 pytest 为 **25 passed, 27 skipped**，所有 skipped 均为 CUDA 测试；没有将跳过项记为通过。GPU 可用时应执行 **52 项**。

| 检查 | 结果 |
| --- | --- |
| 新增材料/场测试 | 9 passed, 9 skipped |
| 原 P1/P2 回归 | 16 passed, 18 skipped |
| 可编辑安装 | afsi-torch 0.3.0 成功 |
| Guccione 示例固定张力势函数 | 3232.1858758290914 |
| 示例最小 det(F) | 1.0156138610542116 |
| 示例解析组装力/自动求导力最大差 | 2.1827872842550278e-11 |
| 示例合力范数 | 1.942149220127487e-11 |
| 示例切线作用/有限差分相对误差 | 9.358497294235228e-11 |
| Basix + NumPy 复步长 PK1 最大差 | 2.35741026699543e-09 |
| 独立参照总势函数差 | 1.0459189070388675e-11 |
| 独立参照组装力最大差 | 1.382431946694851e-10 |
| 原 Neo-Hookean Basix 对照 | 通过 |

```bash
python -m pytest -q
python examples/guccione_patch.py --device cpu
python validation/compare_guccione.py
python validation/compare_basix.py
```

本轮未在 Linux GPU 上执行，也没有声称完成完整 DOLFINx/afsi 对照。一次 TorchScript 弃用警告仍来自 PyTorch 的 JVP 路径。

## 0.2.0：P2 基线

环境：Windows 11；Python 3.12.14；torch 2.14.0+cpu；numpy 2.5.3；pytest 9.1.1；可选 fenics-basix 0.10.0。

| 检查 | 结果 |
| --- | --- |
| 完整 pytest | 16 passed, 18 skipped；跳过项全部为 CUDA 相关 |
| 可编辑包安装 | afsi-torch 0.2.0 安装成功 |
| 两单元 P2 示例 | 2 单元，14 节点，每单元 64 个积分点 |
| 示例总能量 | 0.0698944071186671 |
| 示例积分点最小 det(F) | 1.0313891317561248 |
| 示例自动求导/解析组装力最大差 | 1.1102230246251565e-16 |
| 示例合力范数 | 1.316562550716263e-16 |
| 示例能量方向导数/有限差分误差 | 1.4469980769149515e-10 |
| Basix 形函数最大差 | 5.828670879282072e-16 |
| Basix 形函数导数最大差 | 1.5543122344752192e-15 |
| Basix/NumPy 总能量差 | 2.7755575615628914e-17 |
| Basix/NumPy 组装力最大差 | 1.0547118733938987e-15 |

运行命令：

```bash
python -m pytest -q
python examples/p2_patch.py --device cpu
python validation/compare_basix.py
```

存在一条 PyTorch 内部 TorchScript 弃用警告，来自 torch.func JVP 路径；测试通过，未屏蔽该警告。

后续用户反馈：Linux RTX 4090 上 0.2.0 的 **34 项测试全部通过**，并完成首次推送，基线提交 `82dadab`。该 GPU 结果由用户在目标机器运行和反馈；没有测量性能或双卡并行。
