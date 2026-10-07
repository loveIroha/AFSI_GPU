# Demo 使用与数值方法总览

本文覆盖 `demo/` 下四个算例入口及统一 JSON 启动器。命令在 Linux 的仓库根目录执行；先 `conda activate afsi-torch`。所有例子使用 cm–g–s、默认 FP64，`--device cuda` 选择 GPU。网格读取/生成、文件输出和迭代控制仍涉及 CPU。

## 选择算例

| 入口 | 固体离散 | 流体离散与时间推进 | 耦合 | 完整运行 |
| --- | --- | --- | --- | --- |
| [理想 LV MAC](../demo/ideal_lv_fsi/RUN_GUIDE.md) `run_mac.py` | P2 四面体，Guccione | 64³ MAC；显式对流、显式黏性、压力投影 | 滞后力的显式 IB/FE | dt=5e-5，2 s |
| [理想 LV FEM](../demo/ideal_lv_fsi/RUN_GUIDE.md) `run_fem.py` | 同上 | 32³ 六面体 Q2/Q1；Chorin，显式对流、隐式黏性 | 节点 IB，显式固体位置更新 | dt=5e-5，2 s |
| [二维瓣膜](../demo/ideal_valve_fsi/RUN_GUIDE.md) `run_mac.py` | P2 三角形，FRH | 256×64 MAC；显式对流/黏性、压力投影 | 积分点 IB/FE，显式更新 | dt=6.25e-5，3 s |
| [真实 LV](../demo/real_lv_fsi/RUN_GUIDE.md) `run_mac.py` | 导入 P1 四面体，H–O，DG0 fiber/sheet | 128³ 开放盒 MAC；半拉格朗日对流、BE 黏性、压力投影 | 旧几何双向 IB + 新时刻固体力，非线性 BE–BE | dt=1e-4；纯舒张或主动周期 |

理想 LV 的 2 s 是压力与张力同时升高后保持，不是周期性心动。真实 LV 主动周期为 0.8 s；两个周期是 1.6 s。纯舒张升至 8 mmHg 后保持至 1.6 s 是两个周期的**时长**，不重复压力波形。

## 安装

先按 [主安装说明](../README.md#installation) 安装 NVIDIA 驱动、CUDA 版 PyTorch 和 Python 3.12 环境。然后选择项目依赖：

```bash
# 已经在 AFSI_GPU 仓库根目录、afsi-torch 环境内
python -m pip install -e ".[test,geometry,mesh,fused,quadrature,cuda-ib]"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

`geometry` 用于生成理想几何；`mesh` 用于读取真实网格；`quadrature` 的 Basix 只生成参考积分表；`cuda-ib` 安装 Ninja，不安装 CUDA Toolkit。原生运行不依赖 FEniCSx、AFSI、Docker 或 Taichi。

只有选择自定义 CUDA IB 后端时还需按 [CUDA IB 指南](CUDA_IB.md) 安装匹配的 `nvcc`。所有 GPU demo 预设现在使用 C++/CUDA IB：真实 P1 采用自适应积分 CSR 内核，理想 P2 MAC 采用紧凑传递内核，二维 P2 采用带反射权重的传递内核，FEM 流体采用节点传递内核。有限元基函数求值、质量矩阵求解和模板准备仍可使用 PyTorch/Triton。低层库保留 reference 后端作为 CPU 和等价性对照。

## 一个 JSON 启动一个 demo

```bash
# 激活已配置好的环境后运行；GPU 选择继承环境启动文件
python demo/run.py demo/ideal_lv_fsi/configs/mac_gpu.json
python demo/run.py demo/ideal_lv_fsi/configs/fem.json
python demo/run.py demo/ideal_valve_fsi/configs/mac_gpu.json
python demo/run.py demo/real_lv_fsi/configs/diastole_cuda.json
python demo/run.py demo/real_lv_fsi/configs/active_cuda.json
```

以上是五个独立选择，不要一次复制启动所有长任务。真实 LV 两个预设均为 1.6 s、用户 H–O 材料，主动例为两个周期。修改 JSON 的 `source_dir` 可定位自己的网格；其他文件名、标签、基底中心也要与数据匹配。

`demo` 字段选择已知入口，不能执行任意 Python 文件；其余 JSON 字段由对应配置类型校验。统一入口自动创建带时间戳的结果目录，也可追加 `--output results/my_case`。例如追加 `--end-time 0.005` 做短运行。JSON 不是 Python 源码，正确形式是 `python demo/run.py config.json`，不能用 `python config.json`。

GPU JSON 默认 IB 为 cuda。CPU 小规模验证需显式回退：理想 LV MAC 加 `--device cpu --ib-backend reference --pressure-backend torch`；二维瓣膜加 `--device cpu --ib-backend reference --pressure-backend reference`；FEM 加 `--device cpu --ib-backend reference`。真实 LV 加 `--device cpu --reference` 并使用适合 CPU 的小网格配置。请求 cuda 后端不会静默转为 CPU。

## 配置与运行习惯

1. 先运行对应指南的短例，再运行完整时间范围；已有同版本成功测试可直接复用。
2. 理想算例配置优先级为入口 `CONFIG` < `--config` JSON < 命令行参数。真实 LV 使用 `PaperLVConfig`，支持部分 JSON 合并。
3. `--write-config` 只导出配置而不计算。真实 LV 导出有效覆盖配置；共享理想算例 CLI 导出所选 JSON/默认模板，其他 CLI 覆盖在实际运行时生效。以结果里的 `configuration.json` 为运行记录。
4. 新实验使用新输出目录。续算使用原目录 `checkpoint.npz`，只传该入口允许的续算参数。旧 CNAB 真实 LV 检查点不能用于当前 BE–BE demo。
5. `CUDA_VISIBLE_DEVICES=0` 仅选择一张卡；项目没有多 GPU 域分解。修改网格、时间步或容差后需要重新验证。CUDA Graph 和编译不会取消显式稳定性或非线性收敛要求。

## 后台运行、监控、结果

各算例指南给出完整后台命令，均先创建日志目录，再运行 `nohup /usr/bin/time ... &`。`exit` 退出普通 SSH 终端后任务可继续，前提是服务器或调度系统没有另行清理会话任务。

```bash
# run_dir 为启动命令打印的目录；重新登录后需重新赋值为该实际路径
tail -f "$run_dir/run.log"
cat "$run_dir/runtime.txt"     # 进程退出后才有最终结果
```

`Ctrl+C` 退出 `tail` 不会停止模拟。`runtime.txt` 的 `exit_code=0` 表示进程正常退出，还应核对 `report.json` 的最终时刻。日志中的 `V` 是心腔容积（mL），`minJ` 是最小变形梯度行列式。结果包含标量历史、检查点及按配置输出的 VTK 时间序列；文件位置见各指南。

吞吐量比较用 `validation/` 下对应 benchmark，预热后计时，同一检查点、同一物理设置、同一空闲 GPU。首次编译、网格初始化和 VTK 写出可能影响完整墙钟时间。嵌套性能分项不能直接相加。

## 方法边界与扩展

MAC 压力使用几何多重网格；固体是有限元。这不意味着整个耦合系统都组装成一个全局矩阵。真实 LV 可组装固体 CSR 切线和 IB CSR，耦合切线作用仍通过流体响应完成。

更换材料、载荷或网格见 [固体接口](SOLID_API.md)、[网格接口](MESH_INPUT.md)、[配置接口](CONFIGURATION.md)。`examples/` 是小型演示和兼容入口，`validation/` 是诊断工具，均不替代这里的四个正式 demo 入口。

完整运行成功与网格收敛、周期稳态、实验/文献定量一致是不同的验证目标。真实 LV 中保留的径向基底约束以及用户主动加载属于已注明的复现差异。
