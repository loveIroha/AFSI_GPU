# 理想左心室 2 s IB/FEM 流固耦合 demo

MAC 新增可选的 `--pressure-backend fused`，以 Triton 融合压力多重网格内核并复用工作数组；默认仍为 `torch`。请先完成 [GPU 正确性与短程性能对照](../../docs/MAC_MULTIGRID_PERFORMANCE.md)，再用于完整 2 s 计算。后端选择支持断点恢复，生产运行不启用分阶段计时。

两个入口共享 AFSI `demo_337` 对齐的程序生成左心室、纤维、Guccione 本构、主动张力、内膜压力和基底弹簧。压力与主动张力在前 **1.5 s** 线性升至 150000 和 600000 dyn/cm²，随后保持到 **2 s**。这是加载—保持算例，不是 0.8 s 心动周期，也不含瓣膜和循环系统。

| 入口 | 流体空间离散与求解 | 默认流体网格 | 现有验证状态 |
| --- | --- | --- | --- |
| `run_fem.py` | PyTorch Q2/Q1 有限元、CSR、Chorin 投影 | 32³ 六面体，速度节点间距 0.078125 cm | 已有 2 s GPU 运行记录；不代表网格收敛或生理验证 |
| `run_mac.py` | PyTorch MAC 交错网格有限差分、几何多重网格压力投影 | 64³ 单元，间距 0.078125 cm | 16 项 CPU/CUDA 测试和 0.005 s GPU 耦合短程通过；**完整 2 s 待本次实验验证** |

两者默认固体网格尺度为 0.1 cm，`dt=5e-5 s`，因此 2 s 需要 **40,000 步**。MAC 通过 Griffith–Luo 积分点传递和一致固体质量矩阵与有限元固体耦合；默认每个四面体有 14 个交互积分点。有限元流体和 MAC 流体是两种不同离散，虽然速度格距相同，结果和性能都不能直接视为同精度。

MAC 完整算例若需定位运行耗时，可从已有检查点做 20 步短程回放，比较 IB 核优化前后的结果与各阶段耗时；见 [MAC 性能检查说明](../../docs/MAC_PERFORMANCE.md)。

MAC 的 `--warm-start` 可让两次 IB 一致质量矩阵求解使用上一步结果作为迭代初值；实测 4090 加载末期短程回放约从 103.7 降到 87.2 ms/步。开关默认为关闭，选项和检查点一起保存，续算时可以用 `--warm-start` 或 `--no-warm-start` 改变它。长时间加载阶段的性能及累计数值差异仍需单独验证。

先检查加载初期的 0.005 s（100 步），使用新的输出目录，不覆盖已完成的 2 s 结果：

```bash
CUDA_VISIBLE_DEVICES=0 python -u demo/ideal_lv_fsi/run_mac.py \
  --device cuda --warm-start --end-time 0.005 \
  --output results/lv_mac_warm_early
```

## 环境与前台运行

在仓库根目录运行：

```bash
conda activate afsi-torch
python -m pip install -e ".[test,geometry]"
CUDA_VISIBLE_DEVICES=0 python -u demo/ideal_lv_fsi/run_mac.py --device cuda
```

有限元流体的入口：

```bash
CUDA_VISIBLE_DEVICES=0 python -u demo/ideal_lv_fsi/run_fem.py --device cuda
```

默认结果分别写入 `results/demo_ideal_lv/mac` 和 `results/demo_ideal_lv/fem`。MAC 输出 `history.csv`、`report.json` 和 `checkpoint.npz`，目前没有 VTK 场输出；FEM 还会按原实现写出 VTK。程序运行时打印进度，报告中的 `completed` 只表示达到指定终点，仍需检查腔体积、壁体积、`det(F)`、散度和求解残量。

## MAC 完整实验：后台运行

若已经运行了 0.005 s 的短程算例，建议直接从原检查点续算：

```bash
mkdir -p results/lv_mac_smoke
CUDA_VISIBLE_DEVICES=0 nohup /usr/bin/time \
  -f 'elapsed_seconds=%e exit_code=%x' \
  -o results/lv_mac_smoke/runtime_2s.txt \
  python -u demo/ideal_lv_fsi/run_mac.py \
  --device cuda --resume results/lv_mac_smoke/checkpoint.npz \
  > results/lv_mac_smoke/run_2s.log 2>&1 < /dev/null &
```

这里无需同时指定 `--output`、`--dt` 或网格参数：续算会从检查点恢复它们，并把新结果写回同一目录。若希望从零开始、使用 demo 的独立结果目录：

```bash
mkdir -p results/demo_ideal_lv/mac
CUDA_VISIBLE_DEVICES=0 nohup /usr/bin/time \
  -f 'elapsed_seconds=%e exit_code=%x' \
  -o results/demo_ideal_lv/mac/runtime.txt \
  python -u demo/ideal_lv_fsi/run_mac.py --device cuda \
  > results/demo_ideal_lv/mac/run.log 2>&1 < /dev/null &
```

两个命令只选一个。若结果目录已有运行，使用其 `checkpoint.npz` 续算；新启动不能覆盖已有报告。不要同时用同一张 GPU 跑两个 2 s 算例。

```bash
tail -f results/lv_mac_smoke/run_2s.log
cat results/lv_mac_smoke/runtime_2s.txt
```

完成后检查 `report.json` 中 `status=completed`、`accepted_steps=40000` 和 `reached_time_s=2.0`，并检查 `history.csv` 曲线。较长加载阶段的稳定性、界面漏流和空间收敛尚未由 0.005 s 短程结果证明。[MAC 数值方法与限制](../../docs/MAC_IB.md)。

FEM 若要后台从零运行，同样先建好结果目录：

```bash
mkdir -p results/demo_ideal_lv/fem
CUDA_VISIBLE_DEVICES=0 nohup /usr/bin/time \
  -f 'elapsed_seconds=%e exit_code=%x' \
  -o results/demo_ideal_lv/fem/runtime.txt \
  python -u demo/ideal_lv_fsi/run_fem.py --device cuda \
  > results/demo_ideal_lv/fem/run.log 2>&1 < /dev/null &
```
