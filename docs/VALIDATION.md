# 0.2.0 验证记录

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

本轮尚未运行 Linux/CUDA，也未测量性能或双卡并行。此前用户在 RTX 4090 上的 9 passed 只对应 0.1.0 的 P1 测试。GPU 可用时，0.2.0 完整套件应执行 34 项。
