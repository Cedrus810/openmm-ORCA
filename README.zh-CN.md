# openmmorca

[English](README.md) | 简体中文

OpenMM 驱动的 QM/MM：**OpenMM 负责 MD**（力场、积分器、温压控、轨迹），**ORCA 计算 QM 区**（电子结构 + 静电嵌入梯度），OPI（ORCA Python Interface）负责 ORCA 输入输出。接口风格与 `openmm-ml` / `openmm-pyscf` 一致。

各版本变更：[CHANGELOG.zh-CN.md](CHANGELOG.zh-CN.md)。设计规格：`openmm_orca_opi_design_plan.md`；实施计划：`docs/plans/2026-09-26-openmm-orca-implementation-plan.md`；ONIOM 计划：`docs/plans/2026-09-28-oniom.md`。

后续修复、验证与功能扩展见 [TODO 清单](TODO.md)（含优先级、证据和验收条件）。

**当前状态（v0.4.0）**：full-QM、QM/MM（电子嵌入，含 H link atom 共价边界、周期性 MM 的近似截断嵌入）、双层 ONIOM（QM:QM，非周期）、restart 与失败包诊断已可用（M0–M5 + ONIOM）。已在溶剂化酶体系上验证（DhlA，31,610 原子，`examples/enzyme_qmmm.py`）。

## 安装

```bash
# 环境：mamba env openmm_dev（Python ≥ 3.10，openmm ≥ 8.5，orca-pi ≥ 2.0，numpy）
/home/ruigengji/miniforge3/envs/openmm_dev/bin/python -m pip install -e /home/ruigengji/openmm-ORCA

# ORCA 路径（必设）
export OPI_ORCA=/home/ruigengji/ORCA611
# 并行（nprocs>1）需要 MPI；在没有系统 mpirun 的节点上（如登录节点）设：
export OPI_MPI=/home/apps/openmpi/5.0.7_gcc13.3.0
# 注意：PBS 任务内并行 ORCA 要求任务申请的核数 ≥ nprocs（OpenMPI/PRRTE 读取
# 调度器分配，"Not enough slots available" 就是核不够），例如：
#   qsub -I -l select=1:ncpus=40   # 跑 40 核基准
# 确实要超订时显式设 OMPI_MCA_rmaps_default_mapping_policy=:oversubscribe
```

ORCA 装在 NFS 上时，NFS 一拥堵，每步的启动开销会成倍增加（本机实测 0.4 s → 3.5 s/步）。跑 MD 前先把 ORCA 复制到本地盘或 tmpfs（`autoci_*` 用不到，可以不复制），再让 `OPI_ORCA` 指向副本：

```bash
D=/dev/shm/orca611-$USER; mkdir -p $D
cd /home/ruigengji/ORCA611 && cp -a lib datasets $D/ && ls | grep -v -e '^autoci_' -e '^lib$' -e '^datasets$' | xargs -I{} cp -a {} $D/
export OPI_ORCA=$D
```

kasuga01 本地已装有同一 ORCA 6.1.1 构建：`/home/kasuga/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2`，`OPI_ORCA` 直接指向它即可，无需复制。运行产物放在持久的本地盘（如 `/home/kasuga/openmm-orca-runs/`），不要放 `/tmp`。

OpenMPI 的 `mpirun` 不理会 `taskset`，总是从 0 号核开始绑定。要把并行 ORCA 限定在指定核上（比如和别的任务共用机器时），设 `PRTE_MCA_hwloc_default_cpu_list=4-35`。

## 最小示例

full-QM（一个水分子跑 20 步 MD，见 `examples/full_qm_water.py`）：

```python
from openmmorca import ORCAPotential

potential = ORCAPotential(method="HF", basis="def2-SVP", extra_keywords=("TightSCF",))
system = potential.createSystem(topology)
```

QM/MM（QM = 水 0，其余保持 MM，电子嵌入）：

```python
potential = ORCAPotential(method="HF", basis="def2-SVP", extra_keywords=("TightSCF",))
mixed = potential.createMixedSystem(topology, system, atoms=[0, 1, 2], forceGroup=0)
```

带 H link atom 的共价 QM/MM 边界（v0.3；经 link atom 的有限差分与 ORCA 解析力之差 < 0.05 kJ/mol/nm，见 `examples/link_atom_dipeptide.py`）。每条被切断的键声明为 `(q1, m1)`，q1 在 QM 区、m1 在 MM 区；每条边界在 `R_q1 + g (R_m1 − R_q1)` 处放一个 H，其受力按链式法则分给 q1/m1：

```python
mixed = potential.createMixedSystem(
    topology, system, atoms=qm_atoms,
    boundaryPairs=[(cb, ca)],      # 例如在 CB–CA 处切氨基酸侧链
    linkRatios=None,               # C–C（1.09/1.526）与 C–N（1.09/1.449）有默认 g
)
```

此时 `charge`/`multiplicity` 描述"QM 原子 + link H"。只支持单键，每个 q1 一个 link。m1 的电荷从嵌入中去掉、平均加到它的 MM 邻居上（**简化版** charge shift，不含文献方案中补偿键偶极的点电荷对）；OpenMM 中 MM–MM 静电不变。从力场残基中切出的 QM 区，每个被切开残基的 QM 部分力场电荷 x 通常不是整数；残差 x − round(x) 会加到该残基的 M2 原子上，使每个被切开残基剩下的 MM 部分（以及整个嵌入）都是整数电荷。某个残基的 QM 部分接近半整数（取整有歧义）、或 `charge` 与力场推出的 QM 区电荷不一致时发警告。

带截断嵌入的周期性 QM/MM（v0.4；酶体系示例 `examples/enzyme_qmmm.py`）。System 是周期性的（NonbondedForce 用 PME、LJPME 或 Ewald；`CutoffPeriodic` 会被拒绝）时，回调切换为截断嵌入：每步先把 QM 区（以及 link atom 的 m1）拼回同一个镜像，只有某个原子落在任一 QM 原子 `embeddingCutoff` 以内的 MM 残基才被整体嵌入，取离 QM 区最近的镜像：

```python
mixed = potential.createMixedSystem(topology, system, atoms=qm_atoms, embeddingCutoff=1.2 * unit.nanometer)
```

**这是近似**：截断以外的 QM–MM 静电被忽略（与 PME 不一致），残基进出截断球会让能量不连续，因此不保证严格的 NVE 能量守恒。`embeddingCutoff` 加上 QM 区尺寸必须小于最窄盒宽的一半（每步检查）。计时日志每步记录 `n_embed_groups` 与 `embed_changed`。

ONIOM（双层减法 QM:QM；这里水 0 用 HF/STO-3G，5 个水全部用 xTB——见 `examples/oniom_water_cluster.py`）：

```python
from openmmorca import ONIOMPotential, ORCAPotential

oniom = ONIOMPotential(
    high=ORCAPotential(method="HF", basis="STO-3G", extra_keywords=("TightSCF",)),  # model 区
    low=ORCAPotential(method="XTB"),                                                # 全体系
)
system = oniom.createONIOMSystem(topology, atoms=[0, 1, 2], forceGroup=0)
```

能量是标准减法组合 `E_high(model) + E_low(full) − E_low(model)`；层间机械耦合（QM 层之间不加点电荷嵌入）、model 区必须整分子、System 不含力场项。每步 3 次 QM 调用（1 高 2 低），走 3 个独立 backend，各 restart 链尺寸自洽。`high` 的 charge/multiplicity 描述 model 区，`low` 的描述全体系；低层 model 区计算使用 `high` 的 charge/multiplicity，因此 low-only 原子可以带电或开壳层。每次 `createONIOMSystem` 都会新建这 3 个 backend；`oniom.summarize_timings()` / `oniom.close()` 汇总两层。

QM/MM 团簇 NVE（1 ps，验证能量守恒）：`examples/qmmm_water_cluster_nve.py`；ONIOM 对应版本：`examples/oniom_water_cluster.py`（默认 400 步 = 0.1 ps，`--steps 4000` 为 1 ps）；link atom 二肽 NVT：`examples/link_atom_dipeptide.py`。`examples/analyze_nve.py DIR` 从 `DIR` 中的证据 CSV（见“运行测试”）重算 NVE 漂移／标准差与 restart 重现性差值，写出带输入哈希的 `summary.json`。

酶示例少于 2000 步（1 ps）时只报告冒烟检查，`Task 22 acceptance` 为 `NOT_EVALUATED`；单步运行也可正常汇总。`--reaction-bond I J` 用零起始 OpenMM 原子索引显式指定单独监控的反应键，可重复指定；稳定性检查针对其余 QM 键，原 Task 22 的全 QM 键门槛仍单独报告。退出码对应冒烟／所配置的稳定性检查，完整验收结果见 `acceptance.json`。`steps.csv` 每步写入，失败时保留已完成行并关闭后端。

`prepared.pdb` 与 `equilibrated.xml` 缓存现在需要对应的 `.manifest.json`，核对阶段参数、输入／产物哈希、版本、原子映射和盒信息。无 manifest 的旧缓存或配置不一致的缓存会被保留并拒绝复用；使用新的 `--outdir` 重新准备。改变后续 QM 方法或区域不会使 MM 缓存身份失效。QM/MM 运行不会静默覆盖已有运行：输出目录中已有运行产物（`run.json`、`steps.csv`、轨迹、`checkpoints/`、最终 State）时拒绝启动，除非用 `--archive-existing` 将其移至 `archive/run-<UTC 时间>/`。

`examples/analyze_enzyme_run.py OUTDIR [--json report.json]` 从运行产物重算运行长度、温度、键偏离、嵌入组变化、SCF 循环／重试与耗时；QM 键偏离另从 DCD 帧独立重算（需 mdtraj）并与 `steps.csv` 核对。脚本把重算的 Task 22 结论与 `acceptance.json` 对照，并附上 `run.json` 中的溯源信息（版本、主机、命令、git 提交）；发现任何不一致时退出码为 1。

Checkpoint 与续跑：每 `--checkpoint-interval` 步（默认 100，须为 10 步 DCD 间隔的整数倍）及最后一步写 OpenMM checkpoint 与可移植 State XML，连同 `steps.csv` 长度与哈希提交到 `run.json`。`run.json` 同时记录运行状态（running / completed / failed）和每次尝试；完成时写 `final_state.xml` 与 `final.pdb`。`--resume checkpoint` 在新进程中校验运行身份后续跑，恢复 Langevin 随机数状态，要求平台和 OpenMM 版本一致；确定性后端下与不中断的轨迹逐位一致。ORCA 续跑首步为 fresh SCF 初猜，只在 SCF 收敛精度内一致。`--resume state` 从 State 换新种子起跑，不重现原轨迹。checkpoint 之前的 `steps.csv` 逐字节保留，之后重算的行另存为 `steps.superseded.attempt-NNN.csv`；每次尝试写独立轨迹文件；增大 `--steps` 可延长已完成的运行。

## 参数（`ORCAPotential.__init__`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `method` | 必填 | ORCA simple keyword，如 `"HF"`、`"PBE0"`、`"XTB"`、`"r2SCAN-3c"` |
| `basis` | `None` | 基组；复合方法 / xTB 时为 None |
| `charge` / `multiplicity` | 0 / 1 | QM 区总电荷与多重度（v0.3 起按"QM 原子 + link H"计） |
| `nprocs` | 1 | >1 时写 `%pal` 并设置 MPI 环境变量（不覆盖用户已设值） |
| `maxcore_mb` | 2000 | 每个 MPI 进程内存（ORCA `%maxcore` 语义） |
| `extra_keywords` | `()` | 原样追加的 simple keywords（`RIJCOSX`、`def2/J`、`TightSCF`…）；任务类关键字（`SP`/`Opt`/`EnGrad`…）被拒绝 |
| `extra_blocks` | `()` | 原样追加的 `% block` 字符串；`%pointcharges`/`%pal`/`%maxcore`/`%moinp`/`%output` 被拒绝（后端自管） |
| `scratch_root` | `None` | 默认 `/tmp/openmmorca-<user>`（tmpfs） |
| `restart` | `True` | 用 `.gbw` 做 MORead 初猜（SCF 失败自动回退全新 SCF 一次） |
| `keep_failed` | `True` | 失败时保留失败包 |
| `max_fresh_retries` | 1 | MORead 失败后全新 SCF 重试次数 |
| `timeout_s` | `None` | 单次 ORCA 超时（秒） |
| `orca_path` | `None` | 默认读 `OPI_ORCA` 或 PATH |
| `backend_factory` | `None` | 注入自定义后端（测试用） |

**每次 `createSystem` / `createMixedSystem` / `createONIOMSystem` 都会新建后端实例（ONIOM 为 3 个）和独立 scratch 目录**；一个后端只服务一个 Context。

## scratch 目录与失败包

```text
<scratch_root>/<uuid>/
├── current/     每步覆盖：qm.inp qm.out qm.engrad qm.pcgrad qm.gbw pc.pc guess.gbw qm.property.json
├── restart/     last_good.gbw（MORead 初猜）
├── failures/    failure_step_NNNNNN/ ← 失败现场
└── timings.csv  每步耗时（write/orca/read/total、SCF 圈数、restart 使用）
```

失败时抛出 `ORCACalculationError` / `ORCAOutputError` / `ORCATimeoutError`，信息末尾附失败包路径。复现：

```bash
cd <scratch>/failures/failure_step_000123 && $OPI_ORCA/orca qm.inp > rerun.out
```

`ORCAPotential.summarize_timings()` 返回每个后端的 mean/P50/P95 耗时（自动跳过冷启动第一步）。

## 已知限制

- 周期体系只能通过上述近似截断嵌入；ONIOM 只支持非周期。QM/MM 区可以经 link atom 切断单键（见上文）；ONIOM 的 model 区仍须整分子。ONIOM 层间机械耦合（不带嵌入）。
- 包含 PythonForce 的 System 不能 XML 序列化（回调持有锁与临时目录）。
- 每步有进程启动固定开销（串行 ≈0.4 s）；QM 区很小时这不可忽略。
- NPT 下 `MonteCarloBarostat` 每次尝试额外触发 2 次 QM 计算（默认频率 25 时 QM 开销 +8%）。
- 嵌入电荷走 `%pointcharges` 文件；禁止 inline `Q`（会重复计算 MM–MM 静电且无 pcgrad）。
- SCF 失败绝不返回旧力：重试失败即硬报错并停 MD。
- 酶示例的 checkpoint 续跑只在确定性后端、Reference 平台上验证过逐位一致；尚未实际续跑过中断的 ORCA 运行。续跑首步为 fresh SCF 初猜，ORCA 续跑轨迹只在 SCF 收敛精度内与不中断运行一致。

## 运行测试

```bash
export OPI_ORCA=/home/ruigengji/ORCA611
$PY -m pytest                    # 默认：单元测试 + 快速 ORCA 测试（不含 slow）
$PY -m pytest -m orca -v         # 只跑需要 ORCA 的测试
$PY -m pytest -m slow -v         # NVE 与 restart 重现性（kasuga01 上约 36 分钟）

# 保存 slow 测试的机器可读证据（T06），再重算：
OPENMMORCA_EVIDENCE_DIR=$DIR $PY -m pytest -m slow -v     # 写 qmmm_nve.csv、restart.csv
$PY examples/oniom_water_cluster.py $DIR/oniom_nve.csv --steps 4000
$PY examples/analyze_nve.py $DIR
```

没有 ORCA 时 `orca` 标记的测试自动跳过（注意：桌面 Linux 上 `/usr/bin/orca` 可能是屏幕阅读器，conftest 只认 ELF 二进制）。`tests/test_platform_consistency.py` 在双精度 CUDA 与 Reference 之间对照 link atom、周期镜像与虚拟位点，无可用 CUDA 设备时跳过。`tests/test_packaging.py` 在安装元数据过期时失败，用 `pip install -e . --no-deps` 修复。`examples/analyze_enzyme_run.py` 的 DCD 核对需要 mdtraj，未安装时跳过该项。

当前验证状态与证据：[TODO.md](TODO.md)（状态表）与 [docs/validation/](docs/validation/)。

## 致谢

- [openmm-pyscf](https://github.com/Gallicchio-Lab/openmm-pyscf)（Gallicchio Lab）：本项目的 OpenMM 侧逻辑移植自它，并修正了其键合项删除规则与 O(N²) exception 问题。感谢 Gallicchio Lab 的工作。
