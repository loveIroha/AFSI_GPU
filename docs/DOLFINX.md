# DOLFINx 与 PyTorch 的直接装配对照

0.5.0 增加实际 UFL/DOLFINx 参考数据导出和 PyTorch 比较程序。它是小网格的离散一致性验证，尚未导入原 afsi 心室网格或运行 IB/流体时间步。

## 两个环境，各自运行

保留现有 `afsi-torch` CUDA 环境。DOLFINx 使用独立 CPU 环境，导出普通数值 NPZ；比较程序不依赖 DOLFINx、PETSc 或 Basix。

在 Linux 仓库根目录执行一次：

```bash
conda env create -f validation/environment-dolfinx.yml
```

环境固定 Python 3.12、DOLFINx/Basix 0.10.x，使用 conda-forge、实数 PETSc、MPICH 和 C 编译器。没有向现有 CUDA 环境安装这些依赖。已有 `afsi-reference` 时无需重复创建。

导出实际装配结果：

```bash
conda run --no-capture-output -n afsi-reference python validation/export_dolfinx.py
conda run --no-capture-output -n afsi-reference python validation/export_dolfinx.py --subdivisions 2 --output validation/results/fine.npz
```

第一次运行会编译 UFL 形式。输出 `status: exported` 仅表示参考数据导出成功，不代表 PyTorch 对照通过。

若某些云主机在 `MPI_Init_thread` 阶段出现 UCX/RDMA 设备初始化错误，可在导出命令前设置 `UCX_TLS=tcp,self,sm`。这里是单 MPI 进程参考计算，不需要 RDMA。GitHub Actions 已明确配置此变量，避免云端可见但不可用的网络设备被自动选中；不修改用户机器的全局 MPI 设置。

在已有 GPU 环境比较：

```bash
conda activate afsi-torch
python -m pip install -e ".[test]"
CUDA_VISIBLE_DEVICES=0 python validation/compare_dolfinx.py --device cuda --output validation/results/gpu.json
CUDA_VISIBLE_DEVICES=0 python validation/compare_dolfinx.py --device cuda --reference validation/results/fine.npz --output validation/results/fine-gpu.json
CUDA_VISIBLE_DEVICES=0 python -m pytest -q
```

两次比较应各输出 `status: passed`；0.5.0 单元测试为 92 项，当前 0.6.0 在 CUDA 可用时应为 **113 passed**。比较失败会明确报出构形、力项及 force/tangent，并以非零状态退出。指定 CUDA 后不会退回 CPU。当前 pytest 本身不运行实际 DOLFINx；只有执行上述导出和比较才能验证两个框架的装配一致性。

## 比较范围与排序

默认单位立方体由 6 个四面体构成，P2 场有 27 个节点；加密一次为 48 个四面体、125 个节点。参考几何为 P1 直边，位移/当前坐标、纤维、片层、主动张力、压力和弹簧系数均使用连续 P2 插值。

比较参考构形和给定二次变形构形下的五项：Guccione 被动力、给定主动张力、随动压力、参考面积基底弹簧、总力。每项同时比较节点力和沿给定 P2 方向的导数。DOLFINx 直接使用 `ufl.diff` 构造被动 PK1，使用 `ufl.derivative` 构造力的方向导数；PyTorch 使用解析 PK1 组装和 `torch.func.jvp`。这里导数符号为 **dg/dx**；若以后定义残量 R=-g，切线符号也应相反。

导出保留 DOLFINx 向量空间的全局节点编号，显式核对块大小为 3。局部单元节点通过 DOLFINx 几何映射与坐标匹配转换到项目顺序 `v0,v1,v2,v3,e01,e02,e03,e12,e13,e23`。标量空间另行匹配，不能假设它与向量空间的编号相同。坐标匹配遇到缺失或歧义会报错；这是小验证网格的预处理，不是大型网格通用导入器。

压力面标记为 1（参考 x=1），弹簧面标记为 2（参考 z=0），未标记面不施加边界载荷。这些面的位置只是验证算例定义，并不代表真实心室边界。PyTorch 根据导出的三顶点编号匹配外表面，保留标签并确定固体外法向，不重新按坐标分类。

## 积分与误差

非线性体积分采用明确传入 FFCx 的 Basix 六阶积分点与权重；相同数组导出给 PyTorch。DOLFINx 参考顶点顺序也保留，因此每个单元的物理积分点一致，避免非多项式 Guccione 能量的积分差异。

表面积分两侧均使用 Basix 默认六阶三角形规则。局部面的定向可以置换顶点；当前二次坐标、P2 压力与 P2 弹簧使压力和弹簧节点力的多项式次数不超过 6，六阶规则足以精确积分这些项。切线同样满足此界。不能把这个结论直接用于曲参考面或任意非多项式压力。

逐分量容差：节点力 `atol=1e-8, rtol=3e-10`；切线作用 `atol=2e-7, rtol=3e-10`。报告同时给出最大绝对误差、以 `max(1, ||reference||)` 归一化的 L2 误差、参考文件 SHA-256 和软件版本。接近零的被动力主要按绝对容差判断。

参考数据通过 `allow_pickle=False` 读取；数组形状、有限值、连接关系和 schema 都要检查。输入输出保存在被 git 忽略的 `validation/results/`。

## 自动验证与限制

GitHub Actions 的 `DOLFINx reference comparison` 在 Linux 上生成两种网格的实际参考数据，再在独立 CPU PyTorch 环境运行比较和回归测试。它不执行 CUDA。参考 NPZ 和 JSON 报告作为 `dolfinx-reference` artifact 保留 14 天；可下载相同 NPZ，在本地 GPU 环境执行比较。

0.5.0 已完成 [首次实际 Linux CPU 对照](https://github.com/loveIroha/AFSI_GPU/actions/runs/35985614102)，两种网格全部通过。[参考数据下载](https://github.com/loveIroha/AFSI_GPU/actions/runs/35985614102/artifacts/10801408461) 需要登录 GitHub，受上述保留期限限制。解压后将 `coarse.npz`、`fine.npz` 放入 `validation/results/`，即可在 `afsi-torch` 中分别指定 `--reference validation/results/coarse.npz` 和 `--reference validation/results/fine.npz` 比较；这条路径无需在本地安装 DOLFINx。永久误差记录见 [验证记录](VALIDATION.md) 和 [JSON](reference-results-0.5.0.json)。

支持单 MPI 进程、实数 FP64、直边四面体和固定参考材料场。拒绝多进程导出，以免把局部分区误当完整全局装配。尚未验证分布式幽灵节点、多 GPU、曲参考几何、真实心室网格或积分收敛；两个网格对照是离散一致性检查，不是物理解的收敛证明。

## API 依据

- [DOLFINx 0.10 有限元 API](https://docs.fenicsproject.org/dolfinx/v0.10.0.post0/python/generated/dolfinx.fem.html)
- [DOLFINx 0.10 网格 API](https://docs.fenicsproject.org/dolfinx/v0.10.0.post0/python/generated/dolfinx.mesh.html)
- [FEniCS 官方安装说明](https://fenicsproject.org/download/)
- [UCX 通信方式配置](https://openucx.readthedocs.io/en/master/faq.html)
- [原 afsi 收缩示例](https://github.com/npuheart/afsi/blob/99df0ffba795fa05043ba874ad00353dcb986466/afsic/demo/demo_337/fsi_paralell_fibers_contraction.py)
