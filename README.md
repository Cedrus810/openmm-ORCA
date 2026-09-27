# openmmorca

OpenMM 驱动的 QM/MM：**OpenMM 负责 MD**（力场、积分器、温压控、轨迹），**ORCA 计算 QM 区**（电子结构 + 静电嵌入梯度），OPI（ORCA Python Interface）负责 ORCA 输入输出。接口风格与 `openmm-ml` / `openmm-pyscf` 一致。

设计规格：`openmm_orca_opi_design_plan.md`；实施计划：`docs/plans/2026-09-26-openmm-orca-implementation-plan.md`。

**当前状态（v0.2.0）**：非周期体系的 full-QM / QM/MM（电子嵌入）、restart 与失败包诊断已可用（M0–M3）；link atom（v0.3）与周期性 MM + 截断嵌入（v0.4）在计划中。

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

QM/MM 团簇 NVE（1 ps，验证能量守恒）：`examples/qmmm_water_cluster_nve.py`。

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

**每次 `createSystem` / `createMixedSystem` 都会新建一个后端实例和独立 scratch 目录**；一个后端只服务一个 Context。

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

## 已知限制（v0.4 前）

- 非周期体系、整分子 QM（周期性 + 截断嵌入在 M5）；QM/MM 边界需要 link atom 时在 M4。
- 包含 PythonForce 的 System 不能 XML 序列化（回调持有锁与临时目录）。
- 每步有进程启动固定开销（串行 ≈0.4 s）；QM 区很小时这不可忽略。
- 嵌入电荷走 `%pointcharges` 文件；禁止 inline `Q`（会重复计算 MM–MM 静电且无 pcgrad）。
- SCF 失败绝不返回旧力：重试失败即硬报错并停 MD。

## 运行测试

```bash
export OPI_ORCA=/home/ruigengji/ORCA611
$PY -m pytest                    # 默认：单元测试 + 快速 ORCA 测试（不含 slow）
$PY -m pytest -m orca -v         # 只跑需要 ORCA 的测试
$PY -m pytest -m slow -v         # NVE（约 25 分钟）、restart 重现性等长测试
```

没有 ORCA 时 `orca` 标记的测试自动跳过（注意：桌面 Linux 上 `/usr/bin/orca` 可能是屏幕阅读器，conftest 只认 ELF 二进制）。

## 致谢

- [openmm-pyscf](https://github.com/Gallicchio-Lab/openmm-pyscf)（Gallicchio Lab）：本项目的 OpenMM 侧逻辑移植自它，并修正了其键合项删除规则与 O(N²) exception 问题。感谢 Gallicchio Lab 的工作。
