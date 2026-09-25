# 预加载左室的流体/IB 网格敏感性（0.15.0）

使用上一步已收敛的理想左室平衡结果；原参考构形、材料、纤维和约束保持一致。研究中的给定腔压从 0.20 mmHg 小幅增至 0.22 mmHg，属于数值扰动，不是生理心动周期。不会重算既有非线性平衡。

```bash
git pull --ff-only
conda activate afsi-torch
python -m pip install -e ".[test,geometry]"
CUDA_VISIBLE_DEVICES=0 python -m pytest -q
CUDA_VISIBLE_DEVICES=0 python validation/study_preloaded_ib.py \
  --preload results/lv_equilibrium --device cuda \
  --output results/preloaded_ib_study
```

目录 `results/preloaded_ib_study/frozen/diagnosis.json` 是**固定预加载位置**的单步试验，`report.json` 是短时耦合试验和总览。每个 `fluid_N` 目录保存最后接受的原始参考坐标、预加载坐标、更新坐标以及流体和节点力数组，以便复查位移范数。

固定试验在同一预加载构形上施加 *增量力* `g(0.22 mmHg)-g(0.20 mmHg)`；因此旧有约 1e-7 dyn 的平衡残量不会掩盖扰动响应。使用流体每轴 6、12、18 个 Q2 单元，分别比较四点 IB 核随格距缩放与保持 1 cm 物理宽度的诊断核。每种核又分别使用原有密度载荷路径与直接弱式载荷路径，并比较压力校正前、Chorin 后和参考 Schur 投影后的固体速度。除原核+密度+Chorin 外，这些都只是诊断分支，不代表生产算法更改或已通过完整 FSI 验收。

耦合试验使用同一 P2 预加载固体与原 IB 核、密度载荷路径、Chorin 求解器和既有载荷滞后。流体每轴 6、8、10 个单元；每组时间步为 5e-5 s、20 步，运行到相同物理时间 1 ms。前 0.5 ms 保持原压力，接下来 0.25 ms 增压，末 0.25 ms 保持目标压力。比较的位移是 `x_final-x_preload`，腔体积响应是 `V_final-V_preload`，不是包括原预加载变形的绝对位移或腔体积。报告还包括最小 `det(F)`、壁体体积变化、流体散度、真实线性残量与流固功率差。

最终相邻两档以较细网格的**响应**归一化，并列出绝对差。响应接近零返回 `inconclusive`，不伪称零误差；相对差的 5% 只是预先声明的诊断筛选线，不是医学精度或渐近收敛证明。`completed=true` 只表示计划算例跑完；即使差异大，研究仍应保留真实数据并将 `all_refinement_screens_met=false`。失败保存已接受的状态和错误，不自动放宽容差。

本轮只改变流体网格，故原 IB 核的物理支撑也改变。固定位置试验用于辨认核、弱式载荷及压力投影的影响；耦合试验验证这些差异是否传递到短程轨迹。下一阶段必须在时间误差受控后做固体与流体**联合加密**，并检查流体网格相对固体位置的相位敏感性。现有腔压是固体内表面的给定随动牵引，背景流体压力不等于生理腔压；整个研究仍不构成完整周期验证。

## 首次本地结果

本地 CPU 12 组固定位置试验和 3 组耦合轨迹全部完成，线性求解残量满足容差，完整回归 158 passed、96 CUDA skipped。原核+密度+Chorin 的 6³→12³、12³→18³ 固体速度差分别为 71.65% 和 72.30%；固定 1 cm 核+直接弱式载荷+Chorin 分别为 18.01% 和 3.43%，但它改变了离散。生产耦合 8³→10³ 的增量腔体积差 38.61%、最大增量位移差 45.65%、节点增量位移向量差 65.22%；5% 筛选未通过。[完整对照记录](preloaded-ib-results-0.15.0.json)。

[Linux 自动验证](https://github.com/loveIroha/AFSI_GPU/actions/runs/36189135178)也已完成，158 passed、96 CUDA skipped，三档耦合的筛选结论和数值与本地一致；详细结果已并入[同一数值记录](preloaded-ib-results-0.15.0.json)。目标 GPU 仍待运行。

用户后续提交的 RTX 4090 两份报告也复现了上述差异，三档耦合均完成，筛选仍未通过；见 [GPU 数值记录](preloaded-ib-gpu-results-0.15.0.json)。随附文件没有 pytest 计数。
