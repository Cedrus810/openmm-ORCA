# C2 有限匹配单点：三构型首批比较

来源：[C1 合同](../../docs/plans/2026-11-01-c1-comparison-contract.md) 与 [扫描状态记录](../../docs/validation/2026-11-06-current-status.md)。本实验复用现有 xTB 约束扫描，生成 9 帧候选清单，仅对候选 0、4、8 做首批方案 A 单点。已完成 15 个 PySCF、6 个 ORCA 单点；固定构型相对嵌入差 −10.26/−16.09 kJ/mol 超出筛查目标，当前停止扩大，详见 [结果记录](../../docs/validation/2026-11-07-c2-matched-pilot.md)。

## 输入与选帧

`prepare_pilot.py` 读取 `experiments/2026-11-01_c2_rc_scan/scan/` 及其原始 `prepared.pdb`，保存：

- `inputs/selection.json`：来源文件、零基帧号、保存步号、实际 RC/距离/角度、Trp125/175 接触距离、A/C 的 QM 原子和边界映射。
- `inputs/selected_frames.npz`：9 帧全体系坐标、盒和保存步号。
- `inputs/requests_A.npz`：从现有 `createMixedSystem` 回调捕获首批 3 帧的 S0/S2 输入，包含 link H、实际嵌入点电荷和盒；捕获过程返回零力，不执行 QM 或积分。

候选包括 w00 的首尾保存帧、w05 中实际 RC 最大帧、w06 的 3 个稀疏中央区间帧、w07 中实际 RC 最小帧及 w15 的首尾保存帧。首批 0、4、8 分别为 w00/step450、w06/step1200、w15/step1400。中央区间仅用于描述覆盖，不定义 TS；本批不计算样本标准误或窗口均值验收。

## 参考能量定义

- `S0` / `S2`：PySCF r2SCAN/def2-SVP，分子 QM + 相同点电荷；同时保存 QM 梯度供 ORCA 桥接。
- `ENV20` / `ENV30`：PySCF 周期 MM 环境，精确耦合半径为 20/30 Å，去掉 QM 自身周期镜像的势。`periodic_reference.py` 在 SCF 和总能量两处共用的 `get_qm_ewald_pot` 入口返回零多极势；包含周期 MM 场及其诱导的 QM 密度响应。
- `EW20`：原 B5 的 PySCF 周期定义，保留 QM 自身镜像项，仅作参考定义差异的诊断。

ENV 的移除发生在 SCF 泛函层面；对已经受自身镜像极化的密度事后减一个能量项，不用于本实验。原周期梯度另含 QM 镜像导数，因此 ENV 的解析梯度明确拒绝调用；此首批对 ENV 只比较能量。固定 QM 电荷、同盒条件下比较相对 R 的差分，保留 PySCF 本身的 Ewald 势约定。

`test_periodic_reference.py` 的控制包括：零外场恢复孤立分子的能量和密度、原周期参考产生非零自镜像效应的负对照、保留非零周期 MM 场且能量对密度的数值变分与 Fock 一致，以及拒绝未验证的解析梯度。零外场使用重合的等量异号点电荷，避免 PySCF 对全零电荷的 Ewald 网格退化。

## 执行入口

本次 181 夜间运行，CPU；ORCA 单进程、每个 PySCF 进程 2 线程，scratch 在本地 `/tmp`，未使用 GPU。以下命令由项目根目录执行；输入输出均保留，不复用其他批次结果。

```bash
PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
PSCF=/home/ruigengji/miniforge3/envs/pyscf-env/bin/python
PILOT=experiments/2026-11-06_c2_matched_pilot
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENMM_CPU_THREADS=1
export TMPDIR=/tmp PYSCF_TMPDIR=/tmp

# 首次准备；输出目录已存在时拒绝覆盖
$PY "$PILOT/prepare_pilot.py" --outdir "$PILOT/inputs"

# 小体系参考控制；不调用 ORCA
$PSCF -m pytest -q "$PILOT/test_periodic_reference.py"

# 三构型 ORCA 的 S0/S2 桥接，真实记录在 orca.log
export OPI_ORCA=/home/kasuga/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2
$PY -u "$PILOT/run_orca.py" "$PILOT/inputs/requests_A.npz" "$PILOT/orca"

# 主要比较与分子梯度桥接；真实记录在 pyscf.log
$PSCF -u "$PILOT/run_pyscf.py" "$PILOT/inputs/requests_A.npz" "$PILOT/pyscf" --schemes S2 ENV20 S0 --threads 2

# 同一三构型的参考自检；真实记录在 reference_checks.log
$PSCF -u "$PILOT/run_pyscf.py" "$PILOT/inputs/requests_A.npz" "$PILOT/pyscf" --schemes ENV30 EW20 --threads 2

$PY "$PILOT/analyze_pilot.py"
```

ORCA 和 PySCF 各保存每个完成单点的 JSON，已有结果拒绝覆盖。结果含实际几何、方法、输入路径和计时。`pilot_results.json` 为可重新生成的派生汇总，并列出缺失的必需结果；可选参考自检单独列出。重跑已有单点需选择新的输出目录，并在分析入口明确使用该批结果，不能通过删除已有结果规避身份检查。

## 验收范围

`analyze_pilot.py` 计算同几何的 `D = E_S2 − E_ENV20`，再取相对候选 0 的 `D(f) − D(0)`，对这两个固定构型对照 5 kJ/mol 目标，同时报告 ENV30−ENV20 自检。S0−S2 的 ORCA/PySCF 桥接逐构型报告能量差及伸缩/进攻力差，不把 3 点相关性写成方法普适性。

候选的标签为反应物、跨越区和产物样几何。它们尚无目标方法的极值认证；3 个单点不提供窗口统计误差或势垒认证。M3 的 A/C 反应区单点、本批之外的构型补充和 M4 方法参考未在此启动。
