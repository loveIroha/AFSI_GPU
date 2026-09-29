# MAC 耦合算例的短程性能检查

2 s / 5e-5 s = 40,000 步。总耗时 2.5 小时对应平均 225 ms/步，但包括固体、IB、流体及输出，不能仅凭总耗时判定压力求解器最慢。

当前默认几何有约 243,502 个交互积分点。三个交错速度分量各有 64 个四点核邻居，因此每步要处理约 46,752,384 条 IB 连接。旧版 `FETransfer.prepare()` 在 `(积分点,64,3)` 数组上计算一维 Peskin 核，每个轴的四个值重复计算 16 次。

本次使用核的可分离性，先在 `(积分点,3,4)` 上计算，再做张量积。保留原来的索引排列、64 点支持、积分权重、一致质量矩阵、float64、时间步长及求解容差。只减少核函数的重复计算和大临时数组；最终的连接索引和权重仍然显式保存。

## 更新代码

在 Linux 上的 `AFSI_GPU` 根目录更新代码：

```bash
cd /mnt/large2/gjh/AFSI_GPU
git pull --ff-only
conda activate afsi-torch
python -m pip install -e ".[test]"
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_mac.py
```

## 从已有检查点比较新旧实现

不需要重新生成网格，不需要重新跑 2 s。将下面的检查点路径换成已完成 MAC 算例的实际路径：

```bash
CUDA_VISIBLE_DEVICES=0 python -u validation/benchmark_mac.py \
  --checkpoint results/demo_ideal_lv/mac/checkpoint.npz \
  --device cuda --warmup 3 --steps 20 \
  --output results/mac_performance/report.json
```

如果原来从 `results/lv_mac_smoke` 续算，检查点就是该目录中的 `checkpoint.npz`。输入只读，所有推进均在内存中进行，不改变原来的结果或检查点。脚本分别运行旧的 expanded 核和新的 separable 核，每个版本先预热，再做无阶段计时的吞吐测试和带阶段事件的测试。所有测试都从同一个输入状态开始。

2 s 终点检查点会在内存中推进至 2.001 s（20 步）；此时载荷已保持恒定。这只代表加载末期的性能，不能代表整个加载阶段。若保留有中间检查点，可另选中间状态重复检查。

输出包含：

- `ms_per_step`：无阶段计时的实际整步耗时，含原程序必要的校验，不含网格构造、打印、文件保存及最终物理诊断。
- `phase_timings`：IB 核构造、力传递、速度插值、流体总计、压力多重网格、固体力和固体有效性检查。
- `pressure_solve` 已包含在 `fluid_total` 内，不能重复相加。质量矩阵 PCG 分别包含在力传递及速度插值时间内。
- CUDA 阶段计时只在整段结束后读取事件，不在每个阶段新增同步；事件间隔可能包含 CPU 发射空隙，不等于 GPU 内核纯执行时间。
- `instrumentation_ratio`：带事件与不带事件的墙钟耗时之比，用于判断测量扰动。
- 压力循环数、质量矩阵迭代数、显存峰值、新旧最终状态差异、散度和功率伴随误差。

新旧状态超出比较容差时脚本会报错，而非只报告加速结果。运行完成后提供 `results/mac_performance/report.json` 即可定位下一处热点。

## 当前验证与下一步

本地 CPU：MAC 测试 9 passed，CUDA 测试 9 skipped（本地无 CUDA）。一个 16³、7,672 个交互积分点的旧检查点回放 3 步，新旧位移、力、压力和速度逐项相同。本地一次测量约 218→87 ms/步，IB 核构造约 146→13 ms/步。该小规模 CPU 结果只证明此处的优化有效，不能作为 4090 或完整 2 s 的加速比。

收到目标 GPU 的短程报告后，再依据占比选择：压力多重网格的融合 stencil／CUDA Graph、IB spreading 的融合与原子累加优化，或固体计算与质量矩阵迭代优化。保留物理参数与误差控制，先确认实施热点。
