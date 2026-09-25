# AFSI_GPU：PyTorch IB/FEM 流固耦合

目标是在 GPU 上实现 AFSI 的非线性固体、背景流体与 IB 耦合，算例采用程序生成的厘米制理想左心室。当前已具备 P2 固体、Q2/Q1 流体、Chorin 求解及显式耦合时间步，并通过独立 DOLFINx/NumPy 对照；完整心动周期、收敛和长期稳定性仍待验收。

## 第十二步：定位流体/IB 网格敏感性

当前版本 **0.12.0** 增加冻结左室的核宽度、载荷路径与压力投影对照，生产耦合算法保持不变。已发现多项离散选择共同影响响应，不能只缩小时间步或只换投影。详见 [实验设置、实测结果与限制](docs/IB_DIAGNOSIS.md)。

```bash
git pull --ff-only
conda activate afsi-torch
python -m pip install -e ".[test,geometry]"
CUDA_VISIBLE_DEVICES=0 python -m pytest -q
CUDA_VISIBLE_DEVICES=0 python validation/diagnose_ib.py --device cuda --output results/ib_diagnosis
```

本地 CPU **127 passed、87 CUDA skipped**，完整目标 GPU 测试共 **214 项**；本版 GPU 尚待执行。用户已提供 0.11.0 的 **201 passed** 和 RTX 4090 的完整 11 组研究报告，CPU/GPU 数值结论一致，见 [GPU 记录](docs/lv-study-gpu-results-0.11.0.json)。

[0.12.0 Linux 自动验证已通过](https://github.com/loveIroha/AFSI_GPU/actions/runs/36143192861)：原独立参考对照、完整回归和 12 组新诊断均完成；[数值报告](docs/ib-diagnosis-results-0.12.0.json)。完成诊断不等于全耦合解已经收敛。

## 第十一步：时间步、网格和时长研究

版本 **0.11.0** 增加 11 组受控研究、失败状态保存和汇总图。**研究发现当前流体网格敏感性仍明显，尚不能宣称完整周期可靠。** 保持已有物理方程和耦合顺序不变，见 [研究设置与实测结论](docs/STABILITY.md)。

```bash
git pull --ff-only
conda activate afsi-torch
python -m pip install -e ".[test,geometry]"
CUDA_VISIBLE_DEVICES=0 python -m pytest -q
CUDA_VISIBLE_DEVICES=0 python validation/study_lv.py --device cuda --output results/lv_study
```

本地 CPU 测试为 **120 passed、81 CUDA skipped**，完整 GPU 环境应执行 **201 项**。研究的运行完成与响应收敛分别记录；应阅读 `study.json` 的 `assessment`，不能把退出码 0 当作收敛证明。本版目标 GPU 尚待验证。

[0.11.0 Linux 自动验证已完成](https://github.com/loveIroha/AFSI_GPU/actions/runs/36116534841)：120 passed、81 CUDA skipped，独立参考对照通过，11 组研究全部运行完成；时间和流体网格敏感性检查仍未通过。见 [可复核的数值报告](docs/lv-study-results-0.11.0.json)。

## 第十步：完整显式耦合与生成左室短程运行

0.10.0 连接固体力、IB 密度散布、流体求解、速度插值与坐标更新。左室位移现在来自耦合计算，默认运行 10 步、共 0.001 s，尚不是完整心动周期。见 [耦合顺序、单位、输出与验收限制](docs/COUPLING.md)。

```bash
git pull --ff-only
conda activate afsi-torch
python -m pip install -e ".[test,geometry]"
CUDA_VISIBLE_DEVICES=0 python examples/coupled_lv.py --device cuda --steps 10 --dt 1e-4 --output results/coupled_lv
CUDA_VISIBLE_DEVICES=0 python -m pytest -q
```

本地 CPU **108 passed、80 CUDA skipped**；目标 GPU 环境应执行 **188 项**。可用 ParaView 打开输出的 `solid.pvd` 和 `fluid.pvd`；数值曲线在 `history.csv`，残量、散度与功率差在 `report.json`。沿用现有依赖。本版 GPU 验证尚待执行。

[0.10.0 Linux 自动验证已通过](https://github.com/loveIroha/AFSI_GPU/actions/runs/36027419539)：三步完整耦合的 15 组数组与独立参考一致，生成左室的 10 步运行也通过。

## 第九步：GPU 流体三步求解

0.9.0 增加 Jacobi-PCG、非零速度边界提升、单点压力基准及完整 Chorin 三步法。现在可求解流体速度和压力，但尚未接入固体运动。详见 [求解器、边界条件与验证限制](docs/CHORIN.md)。无需新增依赖。

```bash
git pull --ff-only
conda activate afsi-torch
python -m pip install -e ".[test,geometry]"
CUDA_VISIBLE_DEVICES=0 python examples/chorin_box.py --device cuda --cells 4 --steps 5 --output results/chorin_box.json
CUDA_VISIBLE_DEVICES=0 python -m pytest -q
```

本地 CPU **100 passed, 74 CUDA skipped**，目标 GPU 环境应为 **174 passed**。上一版 0.8.0 已由用户反馈 **155 passed**。此版本尚待目标 GPU 验证。三步分别检查真实残量，并报告校正前后散度；方程收敛不表示严格无散或完整耦合稳定。

[0.9.0 Linux 自动验证已通过](https://github.com/loveIroha/AFSI_GPU/actions/runs/36024695559)：36 组实际 DOLFINx/PETSc 三步解场对照通过，最大绝对差 1.71e-10，散度诊断同样通过。目标 GPU 本版仍需执行上述命令。

## 第八步：Q2/Q1 流体算子

0.8.0 增加规则六面体流体网格、一致质量、黏性、压力 Laplacian、梯度、散度和非线性对流的 PyTorch 算子。支持 CUDA 和自动微分，详见 [流体数学、IB 载荷接口及独立验证](docs/FLUID.md)。此阶段尚未求解流体方程或推进时间。

```bash
git pull --ff-only
conda activate afsi-torch
python -m pip install -e ".[test,geometry]"
CUDA_VISIBLE_DEVICES=0 python examples/fluid_patch.py --device cuda
CUDA_VISIBLE_DEVICES=0 python -m pytest -q
```

本地 CPU 回归 **88 passed, 67 CUDA skipped**；目标 GPU 环境完整测试应为 **155 passed**。新增实际 DOLFINx 对照入口为 `validation/export_fluid_dolfinx.py` 和 `validation/compare_fluid_dolfinx.py`，可复用现有 CPU 参考环境。上一版 0.7.0 已由用户反馈在 RTX 4090 环境完成 **131 passed、无跳过**。

[0.8.0 Linux 自动验证已通过](https://github.com/loveIroha/AFSI_GPU/actions/runs/36021971483)：两种六面体网格上共 40 组实际 DOLFINx 对照通过，最大绝对差 2.43e-15；原固体和 IB 回归仍通过。

## 第七步：厘米制理想左心室几何

0.7.0 自动生成带基底开口的椭球壳、ENDO/EPI/BASE 标签、P2 网格与规则纤维场，接入已有固体节点力计算。长度采用 **cm**，体积采用 **mL**，压力采用 **dyn/cm²**。详见 [几何说明](docs/GEOMETRY.md)。

```bash
git pull --ff-only
conda activate afsi-torch
python -m pip install -e ".[test,geometry]"
CUDA_VISIBLE_DEVICES=0 python examples/ideal_lv.py --device cuda --mesh-size 1.2 --output results/ideal_lv
CUDA_VISIBLE_DEVICES=0 python -m pytest -q
```

Gmsh 在 CPU 生成网格，后续张量计算可运行于 CUDA。结果包含可用 ParaView 打开的网格、表面标签和给定位移/节点力。本地 CPU **74 passed, 57 skipped**；安装 geometry 且 CUDA 可用时应执行 **131 项**。当前示例验证给定变形下的力，尚未求解平衡或流固耦合时间步。

0.7.0 的 [Linux 自动验证已通过](https://github.com/loveIroha/AFSI_GPU/actions/runs/36018932241)，包含实际 DOLFINx 回归和左室生成示例。目标 GPU 验证仍需执行上述命令。

## 第六步：GPU IB 传递验证

无需安装新依赖，使用已有 afsi-torch 环境：

```bash
git pull --ff-only
conda activate afsi-torch
python -m pip install -e ".[test]"
CUDA_VISIBLE_DEVICES=0 python examples/ib_patch.py --device cuda
CUDA_VISIBLE_DEVICES=0 python validation/compare_ib.py --device cuda
CUDA_VISIBLE_DEVICES=0 python -m pytest -q
```

两个程序应输出 `status: passed`；0.6.0 在 GPU 可用时完整测试为 **113 passed**。例子使用给定流体速度，验证固体完整节点力通过 IB 的瞬时功率、合力与力矩；不代表已求解 Navier–Stokes。独立参考遍历全部格点计算标量核，未运行原 afsi C++。

0.6.0 的 [Linux 自动验证已通过](https://github.com/loveIroha/AFSI_GPU/actions/runs/35993548123)：61 passed、52 CUDA 项跳过，实际 DOLFINx 回归和新增 IB 独立对照均通过。目标 GPU 验证仍需上述命令。

实际 Linux CPU 对照已经通过：[GitHub Actions 结果](https://github.com/loveIroha/AFSI_GPU/actions/runs/35985614102)。6/48 单元两种网格上共 40 组向量比较通过，所有节点力最大绝对差 5.46e-10、切线作用最大绝对差 1.42e-10。目标 RTX 4090 对照仍待执行。

## 第五步：实际 DOLFINx 对照

更新代码后，首次创建独立 CPU 参考环境并导出数据：

```bash
git pull --ff-only
conda env create -f validation/environment-dolfinx.yml
conda run --no-capture-output -n afsi-reference python validation/export_dolfinx.py
conda activate afsi-torch
python -m pip install -e ".[test]"
CUDA_VISIBLE_DEVICES=0 python validation/compare_dolfinx.py --device cuda --output validation/results/gpu.json
CUDA_VISIBLE_DEVICES=0 python -m pytest -q
```

比较应输出 `status: passed`。0.5.0 pytest 为 92 项，0.6.0 为 113 项，0.7.0（含 geometry）为 131 项，0.8.0 为 155 项，0.9.0 为 174 项，0.10.0 为 188 项，当前 0.11.0 为 **201 项**。已经创建过 `afsi-reference` 时跳过环境创建。原有 CUDA 环境保持独立。加密网格对照、误差解释及从 Actions 下载参考数据的方法见 [详细说明](docs/DOLFINX.md)。

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

## 已有环境：运行第四步

无需重建已经通过 Guccione 测试的环境。在 AFSI_GPU 项目目录执行：

```bash
conda activate afsi-torch
git pull --ff-only
python -m pip install -e ".[test]"
CUDA_VISIBLE_DEVICES=0 python examples/boundary_patch.py --device cuda
CUDA_VISIBLE_DEVICES=0 python -m pytest -q
```

0.4.0 的完整测试为 **81 项**，0.5.0 为 92 项，0.6.0 为 113 项，0.7.0（含 geometry）为 131 项，0.8.0 为 155 项，0.9.0 为 174 项，0.10.0 为 188 项，当前 0.11.0 为 **201 项**。若出现 skipped，请先运行环境检查，不能将跳过项当作 GPU 验证成功。

可选的 Basix 独立对照（只需轻量的 Basix，无需安装完整 FEniCSx）：

```bash
python -m pip install -e ".[reference]"
python validation/compare_basix.py
python validation/compare_guccione.py
python validation/compare_boundary.py
```

Basix 仅用于 CPU 参考验证，不是 GPU 计算依赖。对照在相同积分点上比较基函数、导数、体积/表面积分及节点力；边界压力使用独立的体单元导数与余子式计算。尚未覆盖真实 DOLFINx 全局自由度和边界标记导入，也未运行 afsi 完整算例。

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

历史验证：用户反馈 Linux RTX 4090 上 P1 的 **9 项**、0.2.0 P2 的 **34 项**和 0.3.0 Guccione 的 **52 项**测试均通过；Guccione 基线为 `b85202f`。0.4.0 的 GPU 验证仍需在目标机器运行。PyTorch 内部在 JVP 测试中会发出 TorchScript 弃用提示。3x3 行列式使用标量三重积实现，使参考构形处的二阶导数通过有限差分检验。

0.2.0 本地验证：**16 passed, 18 skipped**（全部跳过项为 CUDA 测试）。两单元示例通过，解析力/自动求导力最大差 1.11e-16。Basix 0.10.0 独立对照通过，形函数最大差 5.83e-16、导数最大差 1.55e-15、节点力最大差 1.05e-15。详见 [验证记录](docs/VALIDATION.md)。

0.3.0 本地验证：**25 passed, 27 skipped**。Guccione 独立 Basix + NumPy 复步长对照通过，PK1 最大差 2.36e-9、组装力最大差 1.38e-10；随后用户完成目标 GPU 验证。

0.4.0 本地验证：**39 passed, 42 skipped**。完整固体节点力示例通过，残量切线作用/有限差分相对误差 1.19e-10；独立 Basix 体单元余子式对照中，压力节点力最大差 2.27e-13、基底弹簧节点力最大差 9.24e-14。所有跳过项均依赖 CUDA，等待目标机器验证。

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
src/afsi_torch/coupling.py       显式 IB/FEM 耦合状态和时间步
src/afsi_torch/lv_model.py       左室非线性固体力及载荷斜坡
examples/coupled_lv.py          程序生成左室的真实耦合短程运行
src/afsi_torch/fluid/            Q2/Q1 算子、PCG 及 Chorin 三步法
examples/chorin_box.py          短程流体求解与残量/散度诊断
examples/fluid_patch.py         制造场算子检查
src/afsi_torch/geometry/         椭球壳、纤维、腔体积及结果输出
src/afsi_torch/units.py          CGS 单位与 mmHg 换算
examples/ideal_lv.py            程序生成左室与给定变形固体力验证
scripts/preview_lv.py           网格表面与剖视预览
src/afsi_torch/mechanics.py      P1 几何、变形梯度、能量、PK1 节点力
src/afsi_torch/tetrahedron.py    P2 形函数、参考节点、共享边自由度
src/afsi_torch/quadrature.py     四面体积分点与权重（预处理）
src/afsi_torch/materials.py      Neo-Hookean、Guccione、给定张力的主动应力
src/afsi_torch/fields.py         参考方向/张力场插值，积分点法向
src/afsi_torch/solid.py          P2 参考几何、积分及节点力组装
src/afsi_torch/triangle.py       P2 三角形迹、表面积分规则
src/afsi_torch/boundary.py       外表面提取、随动压力、参考基底弹簧
src/afsi_torch/ib.py             规则三维格点上的 Peskin 插值/节点载荷与力密度散布
tests/test_ib.py                核矩条件、合力/力矩、功率及 CPU/GPU 对照
examples/ib_patch.py            完整固体节点力与给定速度场的 IB 传递验证
validation/compare_ib.py        全格点 NumPy 标量核独立对照
tests/test_mechanics.py         刚体运动、解析能量、力组装、差分、切线及 CPU/GPU 对照
tests/test_p2.py                P2 多项式再现、解析变形、积分收敛及切线验证
tests/test_guccione.py          各向异性、主动应力、方向场、节点力与切线验证
tests/test_boundary.py          法向、面积映射、边界力、完整残量切线验证
examples/p2_patch.py            可在 CPU/CUDA 运行的两单元例子
examples/guccione_patch.py      afsi 材料参数与变纤维方向的两单元例子
examples/boundary_patch.py      体积分+压力+基底弹簧的完整固体节点力示例
validation/compare_basix.py     可选的 Basix/NumPy 独立对照
validation/compare_guccione.py  Basix + NumPy 复步长应力/力独立对照
validation/compare_boundary.py  Basix 体单元/余子式的边界力独立对照
scripts/check_environment.py    运行环境和必要张量运算检查
pyproject.toml                 包与可选依赖定义
requirements-test.txt          已验证的 NumPy/pytest 版本
```

首个数值链路为 X -> 单元形函数梯度/体积 -> F=grad_X(x) -> W(F) -> E -> g=-dE/dx。解析 PK1 组装独立核对自动求导，torch.func.jvp 核对切线矩阵与向量乘积。使用 float64 建立精度基线。

0.5.0 的对照入口为 `validation/export_dolfinx.py` 和 `validation/compare_dolfinx.py`，环境定义为 `validation/environment-dolfinx.yml`。0.6.0 本地回归 **61 passed, 52 skipped**，其中 CUDA 项尚待目标机器验证；见 [VALIDATION](docs/VALIDATION.md)。0.7.0 已完成生成左室网格与纤维，本地回归 74 passed、57 CUDA skipped。0.8.0 已实现流体 FEM 算子，后续三个里程碑是 GPU 流体求解、耦合时间推进和完整算例验收。afsi 当前选定示例以显式更新固体坐标为主，因此不把完整 Newton 求解器设为首阶段必需项。

## 来源

- [torchcor 依赖声明](https://github.com/sagebei/torchcor/blob/b02daee87d4283888981cbfc3a6b00f57789a820/pyproject.toml)
- [afsi 收缩示例](https://github.com/npuheart/afsi/blob/99df0ffba795fa05043ba874ad00353dcb986466/afsic/demo/demo_337/fsi_paralell_fibers_contraction.py)
- [PyTorch CUDA 13.0 官方包索引](https://download.pytorch.org/whl/cu130/torch/)
- [PyTorch 安装说明](https://pytorch.org/get-started/locally/)
- [NVIDIA 驱动兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [nvidia-smi 中 CUDA 版本的含义](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
