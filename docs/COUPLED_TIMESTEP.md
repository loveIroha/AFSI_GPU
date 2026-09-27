# 预加载左室耦合轨迹的时间步对照（0.22.0）

AFSI 的 [理想左室收缩算例](https://github.com/loveIroha/afsi/blob/main/afsic/demo/demo_337/fsi_paralell_fibers_contraction.py)设定 `dt=1/20000 s = 5e-5 s`，流体域为 `5×5×5 cm³`，每方向 32 个六面体单元，故 Q2 单元边长 `5/32 = 0.15625 cm`、规则 Q2 节点间距 `0.078125 cm`。另一个 [早期理想左室算例](https://github.com/loveIroha/afsi/blob/main/afsic/demo/demo_337/fsi_paralell.py)也采用 5 cm、32 单元，但时间步为 `1e-3 s`。本项目目前的 24 cm 流体盒为每方向 24 个 Q2 单元，单元边长 **1 cm**、速度节点间距 **0.5 cm**。两者的几何、壁面距离、网格和材料加载条件不同，不能把 AFSI 的 32 单元直接视作本项目的同等分辨率。若保持 24 cm 盒并照搬 `0.15625 cm` 单元边长，每方向约需 154 单元，当前实现与单张 RTX 4090 的资源预算尚未验证。

这一步以 AFSI 收缩算例的 `5e-5 s` 为粗时间步，在**固定 24 cm 盒、1 cm 流体单元、0.5 cm Q2 节点间距、1 cm IB 核、同一左室网格**的条件下，将时间步减半为 `2.5e-5 s`。粗轨迹 20 步、细轨迹 40 步，均到 1 ms；腔压基线、0.02 mmHg 增量及按绝对物理时间定义的保持/升压日程相同。分别比较 Chorin 与参考 Schur 投影，并在每个共同物理时间点保存腔容积、最大位移、实际施压值、最小 `det F` 和弱散度。终点另以快照比较完整固体节点位移向量及腔容积增量。

AFSI 式显式耦合使用前一步固体力，因而粗、细轨迹在同一个*终点*可能取到不同的上一时刻压力值；这是时间离散差异的一部分。程序不强行令逐步施压值相等，但要求连续时间加载函数、预加载检查点、固体参考、IB/流体空间设定一致。参考 Schur 仍是投影诊断分支，不能由本研究直接推断完整心动周期稳定性或空间收敛。

在先前 GPU 结果和 `last_accepted.npz` 快照仍位于 `results/coupled_projection` 时，只需运行新的 40 步轨迹：

```bash
git pull --ff-only
conda activate afsi-torch
python -m pip install -e ".[test,geometry]"
CUDA_VISIBLE_DEVICES=0 python -m pytest -q
CUDA_VISIBLE_DEVICES=0 python validation/study_coupled_timestep.py \
  --preload results/lv_equilibrium --device cuda \
  --reuse-coarse results/coupled_projection/report.json \
  --output results/coupled_timestep
```

如旧快照已丢失，去掉 `--reuse-coarse` 即可在 `results/coupled_timestep/coarse` 重跑粗轨迹，再运行细轨迹。主报告在 `results/coupled_timestep/report.json`；完整细步历史及两条终点快照在 `results/coupled_timestep/fine`。发送主报告、细步报告及必要的快照后，才能分析时间收敛趋势。单次减半仅能检查敏感性，不能证明时间收敛阶；如差异显著，再以相同 1 ms 终点增加第三个时间步。
