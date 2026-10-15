# P2 自适应 IB：相同离散下的 CUDA 执行优化

适用于理想左心室的 `cnab-midpoint` / CN–AB2、xsPPM7、P2 四面体及自适应积分点传递。
这些改动只改变执行方式。网格、dt、积分密度与阶数选择、全部积分点、Peskin4、参考一致质量矩阵、耦合阶段、FP64 和原有残差接受条件保持不变。

## 执行路径

- 原生 `p2_prepare_` 融合 P2 坐标评价和一维 Peskin 表生成，使用相同的整数/半整数格点偏移。三个速度分量共享六套一维表，存储比原来九套表减少三分之一。
- `p2_gather_assemble_` 将网格插值、参考积分和 10 节点 P2 单元贡献累加融合，避免生成整个积分点速度数组。
- `p2_spread_` 在单元块内生成力点值并累加重叠网格贡献，减少全局原子写入。局部包围盒放不进有界共享内存时使用相同点贡献的全局累加，保留完整支撑。
- 这些算子通过 PyTorch dispatcher 调用，使用 PyTorch 当前 CUDA stream；一致质量投影仍由 PyTorch CSR/Graph-PCG 求解。
- stencil 使用有界缓冲池，并由活跃 stencil 独占。旧构型插值和启动阶段传播结束后才允许回收旧表；后续 prepare 不覆盖仍被引用的表。
- FGMRES 复用 V/Z 工作区并融合向量算子；保留原来的重启长度、两次正交化、预条件器、真实残差和物理动量/连续性检查。
- 质量求解的有限性与 RHS 范数在同一个控制点一并读取，不改变迭代、重启或残差检查时刻。

并行归约顺序可能产生 FP64 舍入差。测试比较实际算子、力/力矩/功关系、构型、速度、压力和保存/续算，不要求并行累加逐位相同。

## 配置

新的 `demo/ideal_lv_fsi/configs/mac_gl2017.json` 和 `mac_gpu.json` 默认使用：

```json
{
  "interaction_quadrature": {
    "mode": "adaptive",
    "point_density": 3.0,
    "max_order": 12,
    "max_points": 12000000,
    "rule_family": "xiao-gimbutas",
    "transfer_backend": "fused",
    "stencil_backend": "shared",
    "prepare_backend": "cuda",
    "reuse_stencil_buffers": true
  },
  "coupling": {
    "cnab": {"stokes_vector_backend": "workspace"}
  }
}
```

`mac_gl2017_reference.json` 保留相同数值格式及参数，使用之前的参考执行路径，供性能与结果对照。它与旧一阶数值格式 `mac_legacy.json` 不同。

已有检查点默认继续使用保存的 IB 执行设置。可用 `--p2-execution cuda` 显式切换执行后端；该选项不重新解释或清除 AB2 历史。`--p2-execution reference` 切回参考 P2 路径。显式选择 `--ib-backend reference` 时，也会切回对应的 P2 参考执行设置，因此 CPU 对照入口保持兼容。

## 安装和验证

在仓库根目录执行，使用已配置好 CUDA Toolkit 的环境：

```bash
conda activate afsi-torch
git pull --ff-only
python -m pip install -e ".[test,geometry,fused,cuda-ib,quadrature]"
python -m afsi_torch.mac.cuda_ib
CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/test_p2_cuda_execution.py tests/test_stokes_vector_execution.py \
  tests/test_gl2017_alignment.py tests/test_mac_solid_mass.py
```

若安装环境将 `TORCH_CUDA_ARCH_LIST` 设置为很多架构，首次编译会较慢。只在 RTX 4090 上使用时，可在编译命令前指定 `TORCH_CUDA_ARCH_LIST=8.9`；这只影响编译目标，不影响计算精度。

## 同检查点性能对照

此命令不会覆盖模拟结果，三种变体保留相同的数值参数和接受条件：

```bash
CUDA_VISIBLE_DEVICES=0 python -u validation/benchmark_p2_mac_execution.py \
  --checkpoint results/lv_gl2017_20261016_001105/checkpoint.npz \
  --variants reference p2 optimized --warmup 5 --steps 30 --profile \
  --output results/p2_mac_execution/report.json
```

- `reference`：原 P2 component/torch 执行及原 FGMRES 向量路径。
- `p2`：新的 P2 CUDA 准备、插值组装与传播；FGMRES 向量路径不变。
- `optimized`：P2 CUDA 加上复用的 FGMRES 工作区与融合向量操作。

输出包含相同起点推进后的 x、force、pressure、三个速度分量差，以及同一阶段耗时。子阶段已包含在父阶段中，不能重复相加。编译和预热不计入稳态每步耗时；短检查点回放不等于完整时程性能或收敛验证。

## 完整 2 s 算例

保留与现有 FEM 比较所用的原生固体输入：

```bash
run_dir="results/lv_gl2017_cuda_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_dir"
CUDA_VISIBLE_DEVICES=0 nohup /usr/bin/time \
  -f 'elapsed_seconds=%e exit_code=%x' -o "$run_dir/runtime.txt" \
  python -u demo/run.py demo/ideal_lv_fsi/configs/mac_gl2017.json \
  --solid-input results/afsi337_reference/solid.npz \
  --end-time 2 --output "$run_dir" \
  > "$run_dir/run.log" 2>&1 < /dev/null &
echo "PID=$! output=$run_dir"
```

如不需要与该原生网格逐点比较，可以省略 `--solid-input`，按配置生成理想几何。
这仍是原来的 2 s 斜坡加载/保持实验；优化不会将它改成周期性压力、其它耦合方程或其它心室材料。

## 已测结果

RTX 4090、FP64、末态 3882811 个交互积分点：预热 5 步、测量 30 步，参考路径 132.49 ms/步，P2 CUDA 99.23 ms/步，P2 CUDA + FGMRES 工作区 91.91 ms/步（1.44 倍）。GPU 已分配显存峰值分别约 2.00 / 1.58 / 1.55 GiB。初态到 0.02 s 的 400 步另与原完整运行保存帧对比，最大坐标差约 1.29e-14 cm，腔室容积差约 1.78e-15 mL。修改后针对性回归为 66 passed、1 skipped。

这些是短回放与启动轨迹验证，尚未重新计时完整 2 s。由既有 5754.86 s 和短回放速度比例推算约 67 min，仅作估计。原始数值及限制见 [验证记录](validation/p2_cuda_execution_20261016.json)。
