# AFSI_GPU：PyTorch P1/P2 非线性有限元验证

目标是逐步把 afsi 的 IB/FEM 计算移到 PyTorch。0.3.0 在三维 P1/P2 与 Neo-Hookean 基础上加入 Guccione 被动材料、P2 纤维/片层场和给定主动张力，支持节点力及切线作用。P2 支持直边参考网格上的二次变形；尚未实现边界压力、流体求解和 IB 耦合。详见 [P2 数学说明](docs/P2.md) 与 [Guccione/主动应力说明](docs/GUCCIONE.md)。

## GitHub 协作

仓库为 [loveIroha/AFSI_GPU](https://github.com/loveIroha/AFSI_GPU)。仓库名称为 AFSI_GPU，Python 包名仍为 afsi_torch，Conda 环境名仍为 afsi-torch。

在 Linux 上首次获取代码：

```bash
git clone git@github.com:loveIroha/AFSI_GPU.git
cd AFSI_GPU
conda activate afsi-torch
python -m pip install -e ".[test]"
CUDA_VISIBLE_DEVICES=0 python -m pytest -q
```

后续获取已经测试并提交的更新：

```bash
git pull --ff-only
python -m pip install -e ".[test]"
CUDA_VISIBLE_DEVICES=0 python -m pytest -q
git rev-parse --short HEAD
```

反馈测试结果时附上提交号，避免把不同版本的测试结果混在一起。测试记录区分本地 CPU、Basix 对照与目标 Linux GPU；一次 CPU 测试通过不表示 GPU 测试已完成。若本地有修改导致 pull 失败，先保留和整合修改，不使用强制覆盖。

## 已有环境：运行第三步

无需重建已经通过 P2 测试的环境。在 AFSI_GPU 项目目录执行：

```bash
conda activate afsi-torch
git pull --ff-only
python -m pip install -e ".[test]"
CUDA_VISIBLE_DEVICES=0 python examples/guccione_patch.py --device cuda
CUDA_VISIBLE_DEVICES=0 python -m pytest -q
```

GPU 可用时，本版完整测试应为 **52 passed**。若出现 skipped，请先运行环境检查，不能将跳过项当作 GPU 验证成功。

可选的 Basix 独立对照（只需轻量的 Basix，无需安装完整 FEniCSx）：

```bash
python -m pip install -e ".[reference]"
python validation/compare_basix.py
python validation/compare_guccione.py
```

Basix 仅用于 CPU 参考验证，不是 GPU 计算依赖。该对照在相同积分点上比较基函数、导数、总能量和多单元节点力；尚未覆盖 DOLFINx 全局自由度、边界项或 afsi 完整算例。

## Linux 环境

目标硬件：两张 RTX 4090，NVIDIA 驱动 580.126.09。建议 Python 3.12，固定 PyTorch 2.14.0 的 CUDA 13.0 构建。官方包索引已核对存在 Python 3.12 / Linux x86_64 wheel；该 wheel 要求 glibc >= 2.28。

如果已有 conda，在 Linux 终端执行：

```bash
conda create -n afsi-torch python=3.12 pip --override-channels -c conda-forge -y
conda activate afsi-torch
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install numpy==2.5.3 pytest==9.1.1
```

也可用已有 Python 3.12 创建 venv，替代上面两条 conda 命令：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

无需同时创建 conda 和 venv。无需为这些纯 PyTorch 测试另装系统 CUDA Toolkit、cuDNN、torchvision 或 torchaudio；pip 会安装该 PyTorch 构建所需的运行库。以后编译自定义 CUDA 扩展时，再配置相应编译工具链。

将本项目文件夹复制到 Linux 并进入该文件夹后执行（不要复制 Windows 的 .venv）：

```bash
python -m pip install -e ".[test]"
python scripts/check_environment.py --device cuda:0
python scripts/check_environment.py --device cuda:1
CUDA_VISIBLE_DEVICES=0 python -m pytest -q
python -m pip freeze > requirements-local-linux-cu130.txt
```

检查脚本实际运行 FP64 矩阵运算、自动求导及稀疏矩阵乘法，并核对解析结果。指定 CUDA 后无法使用时会报错，不会悄悄转到 CPU。两次检查仅分别验证两张卡，并未实现双卡并行。pytest 在没有 CUDA 的机器上会明确跳过 GPU 测试；因此先运行检查脚本是必要的。

本地 Windows 验证环境使用 Python 3.12.14、PyTorch 2.14.0+cpu、NumPy 2.5.3、pytest 9.1.1。它不能证明目标 Linux/CUDA 环境已通过测试。

历史验证：用户反馈 Linux RTX 4090 上 P1 的 **9 项**及 0.2.0 P2 的 **34 项**测试均通过；P2 基线提交为 `82dadab`。0.3.0 的 GPU 验证仍需在目标机器运行。PyTorch 内部在 JVP 测试中会发出 TorchScript 弃用提示。3x3 行列式使用标量三重积实现，使参考构形处的二阶导数通过有限差分检验。

0.2.0 本地验证：**16 passed, 18 skipped**（全部跳过项为 CUDA 测试）。两单元示例通过，解析力/自动求导力最大差 1.11e-16。Basix 0.10.0 独立对照通过，形函数最大差 5.83e-16、导数最大差 1.55e-15、节点力最大差 1.05e-15。详见 [验证记录](docs/VALIDATION.md)。

0.3.0 本地验证：**25 passed, 27 skipped**。Guccione 独立 Basix + NumPy 复步长对照通过，PK1 最大差 2.36e-9、组装力最大差 1.38e-10；CUDA 验证待目标机器执行。

## 哪些包需要安装

| 阶段 | 包 | 用途 |
| --- | --- | --- |
| 当前必需 | torch、numpy、pytest | GPU 张量/自动求导，数据准备，数值验证 |
| 网格导入及结果输出 | meshio | 读写网格与场数据 |
| 绘图与三维检查 | matplotlib、pyvista | 收敛曲线、网格和变形场 |
| 生成网格 | gmsh 或 pygmsh | 只在自行生成几何/网格时需要 |
| CPU 对照和监控 | scipy、nvidia-ml-py | 独立数值基准、显存/利用率记录 |
| 后续 afsi 对照环境 | FEniCSx/DOLFINx 及匹配的 Basix/UFL/FFCx/PETSc | 保留原始离散作对照，单独环境管理 |

torchcor 的完整依赖还包含 pandas、wfdb、seaborn、scikit-learn。当前力学测试无需心电信号处理依赖，也不需要把 torchcor 安装为本项目依赖。若完整运行 torchcor，应遵循其声明：torch>=2.0、numpy>=1.26、matplotlib>=3.8、pygmsh>=7.1、nvidia-ml-py>=12、pyvista>=0.45、scipy>=1.13、pandas>=2.2.3,<3、wfdb>=4.3.0,<5、seaborn>=0.13、scikit-learn>=1.4。

## 当前代码结构与下一步

```text
src/afsi_torch/mechanics.py      P1 几何、变形梯度、能量、PK1 节点力
src/afsi_torch/tetrahedron.py    P2 形函数、参考节点、共享边自由度
src/afsi_torch/quadrature.py     四面体积分点与权重（预处理）
src/afsi_torch/materials.py      Neo-Hookean、Guccione、给定张力的主动应力
src/afsi_torch/fields.py         参考方向/张力场插值，积分点法向
src/afsi_torch/solid.py          P2 参考几何、积分及节点力组装
tests/test_mechanics.py         刚体运动、解析能量、力组装、差分、切线及 CPU/GPU 对照
tests/test_p2.py                P2 多项式再现、解析变形、积分收敛及切线验证
tests/test_guccione.py          各向异性、主动应力、方向场、节点力与切线验证
examples/p2_patch.py            可在 CPU/CUDA 运行的两单元例子
examples/guccione_patch.py      afsi 材料参数与变纤维方向的两单元例子
validation/compare_basix.py     可选的 Basix/NumPy 独立对照
validation/compare_guccione.py  Basix + NumPy 复步长应力/力独立对照
scripts/check_environment.py    运行环境和必要张量运算检查
pyproject.toml                 包与可选依赖定义
requirements-test.txt          已验证的 NumPy/pytest 版本
```

首个数值链路为 X -> 单元形函数梯度/体积 -> F=grad_X(x) -> W(F) -> E -> g=-dE/dx。解析 PK1 组装独立核对自动求导，torch.func.jvp 核对切线矩阵与向量乘积。使用 float64 建立精度基线。

下一步加入表面压力与基底约束，并使用相同网格、积分点和材料场与 DOLFINx 节点力逐项比较；随后加入 IB 插值/散布，最后连接流体时间步。afsi 当前选定示例以显式更新固体坐标为主，因此不把完整 Newton 求解器设为首阶段必需项。

## 来源

- [torchcor 依赖声明](https://github.com/sagebei/torchcor/blob/b02daee87d4283888981cbfc3a6b00f57789a820/pyproject.toml)
- [afsi 收缩示例](https://github.com/npuheart/afsi/blob/99df0ffba795fa05043ba874ad00353dcb986466/afsic/demo/demo_337/fsi_paralell_fibers_contraction.py)
- [PyTorch CUDA 13.0 官方包索引](https://download.pytorch.org/whl/cu130/torch/)
- [PyTorch 安装说明](https://pytorch.org/get-started/locally/)
- [NVIDIA 驱动兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [nvidia-smi 中 CUDA 版本的含义](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
