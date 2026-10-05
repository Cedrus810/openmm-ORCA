# OpenMM–ORCA 后续 TODO

更新：2026-10-05（Asia/Tokyo）；审查基线：`main`，`d1ff991`（2026-10-05 历史重写后的哈希；本清单 T01–T07 所列改动已于当日拆分提交并推送），源码 v0.4.0。

P1 优先修复已确认的使用问题；P2 补齐可复现性、验证与维护；P3 为按应用需求选择的功能扩展。勾选任务必须附提交或结果路径；测试跳过、文档中的历史 PASS、SCF 收敛均不能代替相应验收。

## 当前状态（2026-10-05）

| 项目 | 状态 |
|---|---|
| T01–T05 | 完成 |
| T06 | DhlA 部分完成；NVE／ONIOM／restart 原始证据待归档 |
| T07 | MPI、slow 测试、CUDA 对照完成；ONIOM NVE 机器可读结果待做；ORCA 上的中断续跑未实测 |
| T08–T16 | 未开始 |

| 验证 | 结果 |
|---|---|
| 快速回归（`-m 'not slow'`，本地 ORCA 6.1.1） | 224 通过，无跳过（含 5 个 MPI、3 个 CUDA 用例） |
| slow 测试 | 2 通过（NVE 漂移；restart 开／关 50 步），38 min |
| 干净 venv 安装 wheel（OpenMM 8.6.1） | 非 ORCA 测试 179 通过 |
| DhlA 2026-10-01 运行（`/home/kasuga/openmm-orca-runs/dhla_r2scan3c_np40/`） | 从产物重算 PASS：298.6 ± 1.1 K，键偏离 15.9% |
| DhlA 2026-10-05 当前代码运行（`/home/kasuga/openmm-orca-runs/dhla_r2scan3c_np32_2026-10-05/`） | 从产物重算 PASS：298.6 ± 1.4 K，键偏离 17.3% |

验证记录：[P1 修复](docs/validation/2026-10-04-priority-fixes.md)，[T04／T05](docs/validation/2026-10-05-t04-t05.md)，[DhlA 重算与新运行](docs/validation/2026-10-05-t06-dhla-recompute.md)。

## 当前已有的能力

- M0–M5 与双层 QM:QM ONIOM 已有实现。full-QM、电子嵌入、link atom、截断周期嵌入、MORead restart、失败包不再列为待开发项。
- 原实施计划唯一仍未勾选的部分是可选 PySCF 交叉验证（附录 C）。其余下列事项包含审查发现与新建议，不代表 M0–M5 全部未完成。
- ORCA 运行使用本地 `/home/kasuga/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2`（与 NFS 上 `/home/ruigengji/ORCA611` 为同一 6.1.1 构建），运行产物放在 `/home/kasuga/openmm-orca-runs/`。

## P1：优先修复

### T01 — 修正 OutOfPlaneSite 支持声明与力回分

- [x] 修正 `openmmorca/force.py::_vsite_redistribution_map`。原先对该类型调用不存在的 `getWeight()`，已改用其实际权重接口。
- [x] OutOfPlaneSite 按坐标定义的 Jacobian 回分；保留平均位点支持，其他不支持的类型清晰拒绝，文档已同步。
- [x] 验收：面外、Two-/ThreeParticleAverageSite 与嵌套位点的父原子力通过原生 OpenMM 对照及有限差分。证据见本次修复记录。

依据：[force.py](openmmorca/force.py)，[虚拟位点测试](tests/test_qmmm_system.py)。

### T02 — 酶示例区分冒烟与完整验收

- [x] 短跑只报告冒烟结果，Task 22 为 NOT_EVALUATED；完整验收检查运行长度与样本数量。
- [x] 校验 `steps > 0`，避免零步时访问 `rows[0]`；单步耗时统计正常。
- [x] `--reaction-bond I J` 显式配置单独监控的反应键；稳定键检查与原 Task 22 全键门槛分别报告，默认仍检查所有 QM 键。
- [x] 验收：零步拒绝，短跑不宣称完整验收；完整门槛至少 2000 步且有末尾 500 步观测。验收判定、小体系单步运行和注入失败用例通过；2026-10-05 当前代码完整酶应用 PASS。

依据：[酶示例](examples/enzyme_qmmm.py)，[Task 22](docs/plans/2026-09-26-openmm-orca-implementation-plan.md#task-22酶体系应用与基准)（历史记录也指出反应键检查的局限）。

### T03 — 给准备和平衡缓存增加配置身份

- [x] 缓存增加阶段 manifest：结构／产物哈希、拓扑／原子顺序、力场及电荷模型、质子化、溶剂盒、平衡参数与软件版本；平衡记录有效 MM System 哈希。
- [x] 输入或参数改变时拒绝静默复用并报告不一致字段；保留旧文件，指示用新的 `--outdir` 重建。无 manifest 的旧缓存也拒绝直接采用。
- [x] 验收：配置失配、合法复用、产物损坏、粒子数与映射检查通过；真实小水体系的冷 MM 缓存路径通过。身份按阶段定义，后续 QM 配置不参与 MM 缓存身份。完整酶准备／平衡已在 2026-10-05 新运行中按新 manifest 重新生成。

依据：[酶示例的 prepare / equilibrate](examples/enzyme_qmmm.py)。

## P2：复现、验证与维护

### T04 — 持久保存应用日志，并支持中断续跑

- [x] `steps.csv` 每步写入并 flush；注入失败后仍保留已完成行，使用 `try/finally` 关闭 backend。`acceptance.json` 的运行／失败状态避免沿用旧 PASS。证据见本次修复记录。
- [x] 增加最终结构/State，以及周期 checkpoint 和运行状态（running / completed / failed）；明确覆盖、追加与续跑规则。`final_state.xml`/`final.pdb`；`--checkpoint-interval` 写 `.chk` + State XML，经 `run.json` 原子提交；已有运行产物时拒绝新运行，`--archive-existing` 归档；checkpoint 后的行另存 superseded；输出目录 flock。
- [x] 如实现续跑，重建 PythonForce 与 backend，并校验配置；区分 OpenMM checkpoint、可移植 State 与 `.gbw` 初猜的恢复能力。当前 SCF restart 不是跨进程 MD 续跑。`--resume checkpoint|state` 校验运行身份（起始 State、拓扑、MM System、QM 设置、配方）与 CSV 前缀哈希；`.gbw` 不跨进程保留，续跑首步为 fresh SCF（见 `examples/enzyme_qmmm.py` 模块说明）。
- [x] 验收：注入中途失败后仍能读取已完成日志和恢复文件；续跑不覆盖历史步骤。声称轨迹一致恢复时须实测积分器/RNG 状态，否则明确仅为 State 起跑。`tests/test_enzyme_workflow.py` T04 部分：checkpoint 续跑与不中断运行的最终坐标／速度及逐步物理量逐位一致（确定性 fake 后端、Reference）；State 续跑经测试确认与原轨迹不同。ORCA 与 CUDA 上的续跑未实测（归 T07）。证据：[T04／T05 验证记录](docs/validation/2026-10-05-t04-t05.md)。

依据：[酶示例](examples/enzyme_qmmm.py)，[scratch 清理](openmmorca/runtime/scratch.py)，[SCF restart](openmmorca/runtime/restart.py)。

### T05 — 统一安装版本并检查打包

- [x] 刷新可编辑安装元数据：当前 `openmmorca.__version__` / `pyproject.toml` 为 0.4.0，但 `importlib.metadata.version('openmm-orca')` 为 0.2.0。实际为 site-packages 0.0.1 + 仓库内旧 egg-info 0.2.0；`openmm_dev` 已重装为 0.4.0，旧 egg-info 已删除；版本改为单一来源（`dynamic`，取自 `openmmorca.__version__`）。
- [x] 验证 wheel 在干净环境能导入且版本一致；记录实际测试版本与依赖版本，避免沿用旧 `.egg-info`。干净 venv（OpenMM 8.6.1、numpy 2.5.3）非 ORCA 测试 179 通过。
- [x] 验收：源码、项目元数据、安装元数据一致；安装产物包含运行所需模块。此项不包括发布包或修改用户全局环境。`tests/test_packaging.py` 守护版本一致；wheel 含全部 21 个模块。证据：[T04／T05 验证记录](docs/validation/2026-10-05-t04-t05.md)。

依据：[pyproject.toml](pyproject.toml)，[包入口](openmmorca/__init__.py)。

### T06 — 归档历史验收证据

- [x] 定位 DhlA 原始 `steps.csv`、轨迹、准备结构、平衡 State、计时日志及完整命令；记录产物位置、哈希、提交、软件版本、硬件、QM 原子/边界、电荷/多重度、cutoff、步长与种子。2026-10-01 的产物位于 `/home/kasuga/openmm-orca-runs/dhla_r2scan3c_np40/`。新运行的 `run.json` 已自动记录运行身份（起始 State、拓扑、MM System 哈希、QM 原子／边界、电荷／多重度、cutoff、步长、种子）、各次尝试及其环境（软件版本、主机、CPU 数、命令行、git 提交与是否有未提交改动、`OPI_ORCA`）。 原始产物哈希与重算见 [DhlA 重算记录](docs/validation/2026-10-05-t06-dhla-recompute.md)；该运行未记录提交号与种子（早于 `run.json`）。
- [x] 用独立分析脚本从产物重算运行长度、温度、键偏离、嵌入组变化、SCF 重试和耗时，不仅复制终端摘要。`examples/analyze_enzyme_run.py`：不复用运行端判定代码；QM 键偏离另从 DCD 帧独立重算并核对 `steps.csv`；重算 Task 22 结论与 `acceptance.json` 对照，不一致时退出码 1。`steps.csv` 新增 `scf_cycles`、`restart_used`、`fresh_retries` 列。测试见 `tests/test_enzyme_workflow.py` T06 部分（完整运行、续跑、篡改 acceptance／CSV、CSV 低报键偏离）。
- [ ] 同样归档非周期 QM/MM NVE、ONIOM NVE 与 restart 重现性的原始证据；无法找到的保留“历史记录，原始数据未复核”，安排新运行。现状：保留为历史记录；新运行随 T07 执行并保存机器可读结果。
- [ ] 验收：每项结果能由命令和数据重算；README 中的“验证”措辞对应具体实现门槛，不扩大为反应准确性或平衡采样证明。DhlA 已由原始数据重算（`--legacy-region A`，与运行日志一致）；NVE／ONIOM／restart 部分随 T07 运行。

依据：[CHANGELOG](CHANGELOG.zh-CN.md)，[ONIOM 计划](docs/plans/2026-09-28-oniom.md)。

### T07 — 补跑当前版本的 MPI、CUDA 与长测试

- [x] 在允许 MPI 通信的执行环境运行 5 个被跳过的测试，记录环境、分配核数及结果；不要将 sandbox 跳过计为通过。2026-10-05 宿主机快速回归中 5 个 MPI 用例实际运行并通过（无 skip）。
- [x] 运行 2 个 slow 测试：1 ps 非周期 QM/MM NVE，以及同一初态下 restart 开/关的 50 步能量比较。2026-10-05 本地 ORCA、核 0–3：2 passed（38 min），日志 `/home/kasuga/openmm-orca-runs/slow_tests_2026-10-05/pytest.log`。
- [ ] 补一组 CUDA 与 Reference/CPU 的小体系能量/力对照，覆盖 link atom、周期镜像和虚拟位点；ONIOM NVE 示例保留机器可读结果。CUDA 对照已完成：`tests/test_platform_consistency.py`（RTX 2080 Ti，double；QM 力组差 < 1e-12 相对）。ONIOM NVE 机器可读结果待做。DhlA 新运行（当前代码，CUDA）PASS，见 [DhlA 记录](docs/validation/2026-10-05-t06-dhla-recompute.md)。
- [ ] 验收：非周期 QM/MM NVE 漂移 < 0.017 kJ/mol/ps、标准差 < 0.05 kJ/mol；restart 每步差 < 1e-6 Eh；其余沿用现有测试阈值。周期硬截断体系不套用严格 NVE 门槛。 NVE 与 restart 两项由 2026-10-05 slow 测试通过（测试内断言上述阈值）；ONIOM NVE 待做。

依据：[MPI 测试](tests/test_orca_mpi.py)，[NVE 测试](tests/test_nve.py)，[restart 对照](tests/test_restart_reproducibility.py)。

### T08 — 完成可选 PySCF 独立交叉验证

- [ ] 在独立环境用同一 H2O + 2 点电荷几何、HF/def2-SVP 比较能量、QM 梯度和点电荷梯度。
- [ ] 保存输入、版本、SCF/积分设置、数组及逐分量差异；记录单位和力/梯度符号。
- [ ] 验收：按原附录 C 的预期量级核对（能量约 1e-6 Eh，梯度约 1e-5 Eh/bohr）；超出时解释并排查设置，不自动放宽阈值。

依据：[实施计划附录 C](docs/plans/2026-09-26-openmm-orca-implementation-plan.md#附录-c可选与-pyscf-交叉验证)。该项明确尚未勾选。

### T09 — 测量截断与 QM 区选择的影响

- [ ] 在同一批固定快照上比较多个满足盒宽约束的 embedding cutoff，报告 QM/关键 MM 原子的力与能量差；分析组进出事件附近的跳变。
- [ ] DhlA 方案 C（65 原子、4 个 link）有选择代码，但未找到与方案 A 同等的完整验收记录；先做单点/短跑，再决定是否跑完整应用。
- [ ] 比较边界位置、QM 方法与区域选择对反应坐标的影响；涉及反应势垒或自由能时另立参考与采样设计。
- [ ] 验收：提供匹配快照的比较表、误差/差异指标及耗时；区分运行稳定性与科学结论。不同 QM 区不能直接比较绝对总能量并将其当作精度差。

依据：[截断实现](openmmorca/qmmm/embedding.py)，[酶示例](examples/enzyme_qmmm.py)，[spec §11](openmm_orca_opi_design_plan.md#11-周期性-mm--截断嵌入m5v04)。这是新增验证建议。

### T10 — 补齐电子态诊断与失败包溯源

- [ ] 为开壳层应用记录 `get_s2()` 可用结果、能量变化、SCF 循环与 restart 状态；阈值由具体电子态设置，缺失的诊断值须明确记录。
- [ ] 在失败包 metadata 中补实际 QM/MM OpenMM 索引映射、link 定义、盒向量和 ORCA/OPI/包版本；当前只存配置、元素、点电荷数量等，缺少 spec §9.3 所列的部分溯源字段。
- [ ] 验收：失败包能把每行梯度追溯到体系原子；至少一个开壳层对照展示 restart/fresh SCF 的能量及自旋诊断。诊断正常不等于确认正确电子态。

依据：[后端](openmmorca/backend/orca_opi.py)，[失败包](openmmorca/runtime/diagnostics.py)，[spec §9.3 / §16](openmm_orca_opi_design_plan.md)。

### T11 — 审查尚未覆盖的运行时边界

- [ ] 审查两个独立 backend 同时调用 OPI 时对进程全局 `os.environ` 的修改：现有锁仅限单实例。用受控交错复现后，决定进程级串行化、隔离或明确拒绝线程并行。
- [ ] 核对超时是否终止 ORCA 的全部子进程、保留现场且不更新 last-good；现有 timeout 接口不等于已验证所有进程清理情形。
- [ ] 补输入校验：QM/MM 坐标及电荷有限、MM 电荷严格一维、盒合法、重试数与 timeout 合法；防止 NaN 参数绕过数值比较。
- [ ] 验收：先记录复现结果，确认问题后加回归；并发、超时和无效输入不能污染其他实例、返回旧力或静默启动无效计算。前两项目前是审查候选，未在本次证明发生故障。

依据：[后端锁与环境管理](openmmorca/backend/orca_opi.py)，[请求校验](openmmorca/backend/base.py)，[嵌入校验](openmmorca/qmmm/embedding.py)。

### T12 — 增加持续检查与可复现环境说明

- [ ] 增加不依赖 ORCA 的 CI；ORCA/MPI/slow 测试使用可用的专用执行环境，分别报告通过、跳过和未运行。
- [ ] 记录 core/test 环境与酶示例的额外依赖（PDBFixer、OpenMMForceFields、OpenFF/NAGL、RDKit），给出环境导出或可复现安装步骤。
- [ ] 验收：新 checkout 可按文档运行非 ORCA 测试；酶示例依赖缺失时可清晰诊断；检查不把跳过数合并进通过数。

依据：[pyproject.toml](pyproject.toml)，[酶示例依赖](examples/enzyme_qmmm.py)；本次未找到 `.github` 工作流。

## P3：按需求选择的扩展

这些项目不属于 v0.4 的欠账，开始前须另定范围与成本。

- [ ] **T13 — ONIOM 的共价边界。** 在 high(model) 与 low(model) 使用一致 link 定义及力回分，low(full) 不加 link；验收包括 high=low 恒等式、边界有限差分和各层独立 restart。原准备文档 D4 明确推后。
- [ ] **T14 — 平滑截断嵌入。** 验证获得 QM 静电势的路径与成本；开关电荷时须包含电荷对坐标变化带来的能量导数。验收检查 cutoff 附近能量/力连续与有限差分；平滑本身不恢复远程静电或 PME 一致性。
- [ ] **T15 — 完整边界 charge shift。** 当前是简化电荷重分配，未实现补偿键偶极的点电荷对。先定义几何、能量及所有位置依赖的链式导数，再与简化方案做匹配结构对照。
- [ ] **T16 — 用户定义嵌入电荷组。** `CutoffEmbedding` 内部接受 groups，但公共 `createMixedSystem` 固定按残基分组。若应用需要，新增公开配置，校验分组、映射、边界电荷和周期完整性。

更远期选项：多层 ONIOM / QM:MM ONIOM、严格周期 QM–MM 静电、极化 MM、自适应 QM 区、MTS、异步 QM。原设计将其中多项列为非目标；不默认纳入本轮工作。

## 建议执行顺序与复核命令

T01–T05 已完成；T06／T07 余项为 NVE／ONIOM／restart 原始证据与 ONIOM NVE 机器可读结果。T08–T10 在科学应用扩大前完成。T11 的并发/超时审查可独立进行，T12 保持后续回归。T13–T16 按实际应用需求排期。

```bash
export PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
export OPI_ORCA=/home/ruigengji/ORCA611

# 快速纯软件回归
$PY -m pytest -m 'not orca and not slow' -q

# 真实 ORCA 快速回归；必须记录 skip 原因
$PY -m pytest -m 'orca and not slow' -q -rs

# 较长验证，安排可用计算资源后执行
$PY -m pytest -m slow -v -rs

# MPI 专项：在允许通信且分配足够核数的环境执行
$PY -m pytest tests/test_orca_mpi.py -m orca -v -rs
```

T05 的重装只在选定的开发环境执行；测试结果与运行产物按 T06 归档。本次仅执行本地软件回归与真实 ORCA 快速测试，未安装依赖或发布版本。
