# T06 DhlA 1 ps 验收：从原始产物重算

日期：2026-10-05（Asia/Tokyo）。

## 产物

位置：`/home/kasuga/openmm-orca-runs/dhla_r2scan3c_np40/`（kasuga01 本地盘），运行时间 2026-10-01 19:12–22:18。方案 A（14 QM 原子 + 1 link H），r2SCAN-3c，nprocs 40，2000 步 × 0.5 fs。该运行早于 `run.json`，没有逐步 SCF 循环数、稳定键与反应键的分列，也没有机器可读的验收文件。

```text
077c10679e919e9e00559e5d94e08d52ceabe8a0de9f16e67c84417ff2ea900f  equilibrated.xml
1db3ad9a3cb42ed3b6a6a2aebe038808cd8d747f26685c9faddaf5c847d6e11a  prepared.pdb
41433cce91e740ab59bb66397e491aa07fa7004e524c46995759e038013fe45b  run.log
feee4924196782799456f0fc6929a56d58249afb194facb264ad3dc630ece401  steps.csv
d975350298759ba65c47a00025301fb7e511364c306d49a1b2e1c2ce525b4457  trajectory.dcd
```

## 命令

```bash
/home/ruigengji/miniforge3/envs/openmm_dev/bin/python examples/analyze_enzyme_run.py \
  /home/kasuga/openmm-orca-runs/dhla_r2scan3c_np40 --legacy-region A \
  --json docs/validation/2026-10-05-dhla-r2scan3c-report.json
```

退出码 0，无不一致项。完整报告：[2026-10-05-dhla-r2scan3c-report.json](2026-10-05-dhla-r2scan3c-report.json)。

## 重算结果与运行日志对照

| 项目 | 重算 | `run.log` |
|---|---|---|
| 步数 | 2000/2000，步号连续 | 2000 (1.000 ps) |
| 末 500 步温度 | 298.57 ± 1.13 K | 298.6 ± 1.1 K |
| 最大 QM 键偏离（`steps.csv`） | 15.90 % | 15.9 % |
| 最大 QM 键偏离（DCD 200 帧独立重算） | 15.87 % | — |
| t_orca 均值／p95（不含首步） | 5.454 s / 5.643 s | 5.454 s / 5.643 s |
| 嵌入组均值；跨越步数 | 152；286 | 152；286 |
| fresh SCF 重试 | 未逐步记录 | 0 |
| Task 22 结论 | PASS | PASS |

DCD 每 10 步一帧，独立重算的偏离不超过 `steps.csv` 对应步的累计最大值，帧上最大值 15.87%，低于逐步记录的 15.90%。QM 键由 `prepared.pdb` 按方案 A 选区重建，参考键长取 `equilibrated.xml` 起始坐标（该运行未做 QM/MM 最小化）。

结论范围：Task 22 的实现门槛（运行长度、温度、QM 键稳定性）。不涉及反应准确性或平衡采样。

## 2026-10-05 新运行（T07，当前代码）

位置：`/home/kasuga/openmm-orca-runs/dhla_r2scan3c_np32_2026-10-05/`。方案 A，r2SCAN-3c，ORCA 6.1.1（本地 `/home/kasuga/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2`），nprocs 32（核 4–35），CUDA，2000 步，checkpoint 间隔 100。2026-10-04T18:37:27Z 至 21:36:04Z，MD 墙钟 10717 s。准备与平衡按新 manifest 重新生成（MM 平衡 18 s，盒 6.818 nm）。

`analyze_enzyme_run.py`（`analysis.json` 在同一目录）退出码 0，与 `acceptance.json` 一致：

| 项目 | 结果 |
|---|---|
| 步数 | 2000/2000，1 次尝试，`run.json` 状态 completed，checkpoint 2000 |
| 末 500 步温度 | 298.6 ± 1.4 K |
| 最大 QM 键偏离 | 17.3 %（`steps.csv` 与 DCD 200 帧一致） |
| t_orca 均值／p95 | 5.273 s / 5.877 s |
| SCF 循环 | 均值 9.5，最大 14；fresh 重试 0 |
| 嵌入组均值；跨越步数 | 154；312 |
| Task 22 | PASS |

`run.json` 中 git 提交为 null：以 kasuga 运行时 git 因仓库属主不同拒绝访问。中断续跑未在该运行中实测。
