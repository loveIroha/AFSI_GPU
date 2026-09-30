# 固体力与一致质量矩阵的执行优化实验

本实验比较四组运行：`reference`（现有 fused 执行）、`pointwise`（固体小矩阵融合）、
`pcg_graph`（分段 CUDA Graph PCG）、`combined`（两者同时启用）。所有组使用相同
checkpoint、热启动、紧凑积分点 IB、MAC 流体与 fused 压力多重网格。

## 实现范围

固体实验把 Guccione 本构中的 3×3 批量矩阵乘法改写为显式三项标量收缩，交给
PyTorch Inductor 融合，减少小矩阵库调用和中间数组。它继续调用原来的 P2 单元
积分和节点组装；本阶段没有宣称已经实现专用的单元积分/节点组装 CUDA 内核。

质量矩阵仍为已组装的 CSR 一致质量矩阵。PCG 仍使用 Jacobi 预条件，原来的
容差、精度、热启动和真实残差验收。捕获区间严格在原来的检查、残差重算或
最大迭代次数边界结束，支持重算间隔不是检查间隔倍数的情况。捕获区间中途
达到递推残差容差时，原来的 active 标志冻结更新；区间结束后仍检查真实残差。
矩阵和工作区地址固定，求解结果复制出工作区，下一次求解不会覆盖上次返回值。
只支持每 1–16 次迭代检查的实验配置，目前 IB 配置为每 4 次检查。

CUDA Graph 采用侧流预热、固定输入/输出缓冲区和显式 capture/replay，依据
[PyTorch CUDA Graph 文档](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)。
CPU 执行同样的分段递推用于验证，但不能验证 GPU 捕获或速度。

实验优化已接入完整 demo 的可选配置：`--solid-backend pointwise`
与 `--mass-backend graph`，两者要求 `--execution-backend fused`。
默认仍为既有固体执行与 PCG；检查点保存配置，续算继承或显式覆盖。
从旧检查点恢复时，缺失字段按原实现解释。

## Linux GPU 命令

在 AFSI_GPU 仓库根目录运行：

```bash
git pull --ff-only
conda activate afsi-torch
python -m pip install -e ".[test,geometry,fused]"
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_mac_solid_mass.py

CUDA_VISIBLE_DEVICES=0 python -u validation/benchmark_mac_solid_mass.py \
  --checkpoint results/lv_mac_fused_2s/checkpoint.npz \
  --device cuda --warmup 3 --steps 20 \
  --output results/mac_solid_mass/report.json
```

使用实际存在的 checkpoint；若文件位于其他目录，替换 `--checkpoint`。输入
checkpoint 只读，输出报告必须使用新路径。首次编译与捕获属于预热阶段，可能
耗时几分钟，单独记录而不计入整步性能。测试包含一个粗网格短程四组 GPU 对照。

## 如何判读

优先看 `passed`、`equivalence` 与 `repeat_equivalence`。检查 `final_force_mass`
和 `final_velocity_mass` 的真实残差不超过容差，并查看散度、功率误差和固体
诊断。任何失败都应报告，不通过放宽容差处理。

`speedups` 基于不加分项计时的 replay 总时间。`phase_timings` 给出固体力、
两次质量矩阵求解和流体等分项；嵌套分项不能相加。

`mass_instrumented_details` 在另外一轮 reference/pointwise replay 中测量
CSR 乘法 (`mass_action`)、更新/残差归约 (`mass_advance`)、预条件和方向更新
(`mass_direction`) 与重启 (`mass_restart`)。每次迭代的事件记录会扰动调度，
这些分项用于定位，不能据此替代不加计时的整步速度；也不包含所有主机检查。
Graph 组不插入这些内部事件，避免改变已捕获图。

`solid_isolated_details` 分开测量本构和组装，额外物化应力数组，不能直接加和
或从完全融合的固体力耗时中扣除。它用于判断本构与组装哪个更值得下一轮优化。

2 s checkpoint 测量的是保持载荷末段，不能代表加载段或完整 2 s 的收益。
四组 4090 对照已通过，固体力约 12.94→1.33 ms/步，质量矩阵求解
约 12.17→8.45 ms/步，整步约 42.24→21.84 ms；这是 20 步保持载荷末段的
实测结果。加载初段、完整 2 s 后台实验和续算命令见
[demo 说明](../demo/ideal_lv_fsi/README.md#本次固体质量矩阵优化实验)。
