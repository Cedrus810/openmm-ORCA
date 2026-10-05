# 更新记录

[English](CHANGELOG.md) | 简体中文

版本号对应 `docs/plans/2026-09-26-openmm-orca-implementation-plan.md` 的里程碑（M2 = v0.1，M3 = v0.2，M4 = v0.3，M5 = v0.4）。

## 未发布 — 2026-10-05

### 修复
- `OutOfPlaneSite` 的力回分改用坐标 Jacobian，支持依赖其他虚拟位点的位点；父原子力通过原生 OpenMM 对照和有限差分。

### 新增
- 酶示例（`examples/enzyme_qmmm.py`）：
  - 冒烟检查与完整 Task 22 验收分开；`--reaction-bond` 单独监控反应键。
  - 准备／平衡缓存带 manifest，旧格式或失配缓存拒绝复用。
  - `steps.csv` 每步 flush，新增 `scf_cycles`、`restart_used`、`fresh_retries` 列。
  - 周期写 OpenMM checkpoint 与 State XML，提交到 `run.json`；`run.json` 记录状态、各次尝试及溯源信息（版本、主机、命令、git 提交、`OPI_ORCA`）。运行完成时写 `final_state.xml`、`final.pdb`。
  - `--resume checkpoint|state` 校验运行身份与 CSV 前缀；`--archive-existing` 归档旧产物；新运行不覆盖已有运行产物；输出目录加锁。
- `examples/analyze_nve.py`：从 slow 测试（`OPENMMORCA_EVIDENCE_DIR`）与 `examples/oniom_water_cluster.py --steps` 写出的证据 CSV 重算 NVE 漂移／标准差与 restart 差值。
- `examples/analyze_enzyme_run.py`：从产物重算运行结果，DCD 独立核对键偏离，Task 22 结论与 `acceptance.json` 对照；`--legacy-region` 读取 `run.json` 之前的旧运行。
- 测试：`tests/test_platform_consistency.py`（CUDA 对 Reference）、`tests/test_packaging.py`（版本一致性）。

### 变更
- QM/MM Langevin 积分器显式设种子（`QMMM_RECIPE`）。
- 版本单一来源 `openmmorca.__version__`；构建依赖 `setuptools>=77`（PEP 639 字符串 license）。

### 验证（2026-10-05）
- 快速回归：**226 通过**，无跳过（MPI 用例在宿主机实际运行；CUDA 用例在可用 CUDA GPU 上运行）。
- slow 测试：2 通过（1 ps NVE 漂移；restart 开／关各 50 步）。原始证据随验证记录归档，用 `examples/analyze_nve.py` 重算：QM/MM NVE +0.0056 kJ/mol/ps、标准差 0.0163 kJ/mol；ONIOM NVE（1 ps）+0.0067、0.0231；restart 最大差 3.9e-8 Eh。与 2026-09-27／29 记录一致。
- 干净 venv 安装 wheel（OpenMM 8.6.1）：非 ORCA 测试 179 通过。
- DhlA Task 22，用 `analyze_enzyme_run.py` 从产物重算：
  - 2026-10-01 运行（已归档，见验证记录）：PASS，298.6 ± 1.1 K，最大 QM 键偏离 15.9%。
  - 2026-10-05 当前代码运行（已归档，见验证记录）：PASS，298.6 ± 1.4 K，17.3%。
- checkpoint 续跑：确定性后端在 Reference 上与不中断运行逐位一致；尚未用 ORCA 实测。
- 记录：[P1 修复](docs/validation/2026-10-04-priority-fixes.md)，[T04／T05](docs/validation/2026-10-05-t04-t05.md)，[T06／T07 运行证据](docs/validation/2026-10-05-t06-dhla-recompute.md)。

## 0.4.0 — 2026-10-01（M5：周期性 MM + 截断嵌入）

### 新增
- **带截断嵌入的周期性 QM/MM**（近似，与 PME 不一致）：PME/LJPME/Ewald 体系可用 `createMixedSystem(..., embeddingCutoff=1.2 * nanometer)`。每步把 QM 区与 link atom 的 m1 拼回同一镜像（`openmmorca.qmmm.imaging`：`qm_bond_graph`、`make_qm_whole`、`minimum_image`），截断内的 MM 残基整体以最近镜像嵌入（`openmmorca.qmmm.embedding`：`CutoffEmbedding`、`groups_from_topology`、`check_cutoff_against_box`、`perpendicular_widths`）。
- `QMRequest.diagnostics`；计时日志新增 `n_embed_groups`、`embed_changed` 两列；有组进出截断时写一条 DEBUG 日志。
- `make_python_force(..., periodic=False)`；`QMMMCallback(..., embedding=None, qm_graph=None)`。

### 变更
- **QM 区非整数电荷改为修正**，不再只警告：每个被边界切开的残基，把 δ = x − round(x)（x 为其 QM 部分的力场电荷）加到该残基自己的 M2 上，嵌入净电荷因此为整数。新增 `MixedSystemParts.qm_formal_charge`；残基接近半整数（取整有歧义）或 `charge` 与力场推出的电荷不一致时警告。0.3.0 的"non-integer"警告已去掉。
- 周期体系不再抛 `NotImplementedError`；`CutoffPeriodic` 抛 `ValueError`。

### 新增（应用）
- 示例 `examples/enzyme_qmmm.py`：卤代烷脱卤酶 DhlA + 1,2-二氯乙烷（原始结构 `examples/data/2DHC.pdb`），PDBFixer + ff14SB + DCE 用 OpenFF Sage/NAGL + TIP3P，MM 平衡，带 link atom 与截断嵌入的周期性 QM/MM NVT；QM 方案 A 与 C；结束时自动判定验收。

### 验证
- DhlA（PDB 2DHC，31,610 原子，PME），QM 方案 A（DCE + Asp124 侧链，15 原子含 1 个 link H，电荷 −1），r2SCAN-3c，40 个 MPI 进程，Langevin NVT 300 K、0.5 fs：2000 步（1 ps）无人工干预，5.53 s/步（ORCA 5.45 s；写 0.009 s、读 0.012 s），平均嵌入 152 个残基组（约 1,800 个点电荷），286 步有组进出截断，0 次 fresh SCF 重试；最后 500 步 298.6 ± 1.1 K；QM 键最大偏离 15.9%（来自 C1–Cl1：第 ~1907 步 O–C1 一度靠近到 0.232 nm 的 SN2 进攻尝试，随后回弹）。
- 周期体系 fake 后端有限差分 < 1e-2 kJ/mol/nm；整体平移一个晶格矢量、单个原子被包裹到盒子另一侧，ORCA 能量都不变（< 1e-7 Eh）。

## 0.3.0 — 2026-09-29（M4：link atom）

### 新增
- **带 H link atom 的共价 QM/MM 边界。** `ORCAPotential.createMixedSystem(..., boundaryPairs=[(q1, m1), ...], linkRatios=None)`：每条声明的键在 `R_q1 + g (R_m1 − R_q1)` 处放一个 H，其受力按链式法则分给 q1/m1（`F_q1 += (1 − g) F_L`，`F_m1 += g F_L`）。C–C（1.09/1.526）与 C–N（1.09/1.449）有默认 `g`，其他元素组合须显式给 `linkRatios`。`charge`/`multiplicity` 描述"QM 原子 + link H"。
- `openmmorca.qmmm.linkatoms`：`BoundaryPair`、`default_link_ratio`、`make_boundary_pairs`、`check_boundary_pairs`（q1 在 QM、m1 在 MM、二者成键、每个 q1 一个边界、无未声明的切断键）、`LinkAtomManager`。
- `openmmorca.qmmm.charges.ChargeShift`：m1 的电荷从嵌入中去掉，平均加到 m1 的 MM 邻居（M2）上；OpenMM 中 MM–MM 静电不变。这是**简化版** charge shift，不含文献方案中补偿键偶极的点电荷对。
- QM 区力场电荷之和不为整数（残差 > 0.05 e）时发 `UserWarning`，不做修正。
- `check_whole_molecules(..., allowed_bonds=())`、`build_mixed_system(..., boundary_pairs=())`、`MixedSystemParts.boundary_pairs`、`QMMMCallback(..., link_manager=None)`。
- 示例 `examples/link_atom_dipeptide.py`（ACE-ALA-NME，QM = ALA 甲基侧链，HF/def2-SVP，200 步 NVT）；测试数据 `tests/data/ace_ala_nme.pdb`。

### 变更
- 声明了边界时，`MixedSystemParts.mm_charges_e` 为 shift 后的嵌入电荷（无边界时与力场电荷相同）；原本电荷为 0 的 M2 会加入嵌入。
- `check_whole_molecules` 的报错信息改为指向 `boundaryPairs`。

### 验证
- fake 后端：经 1 个、2 个 link atom（C–C 与 C–N）的有限差分与解析力之差 < 1e-3 kJ/mol/nm。
- ORCA（HF/def2-SVP TightSCF）：Q1、M1、M2 的有限差分误差 ≤ 0.042 kJ/mol/nm（受力 300–870 kJ/mol/nm）；整体平移 1 nm 能量变化 5e-12 Eh；‖ΣF‖/max‖F_i‖ = 1.4e-10。二肽示例 200 步内 CA–CB 保持在 0.149–0.158 nm，720 ms/步。

### 已知限制
- 只支持单键边界、每个 q1 一个 link atom；QM 区非整数电荷残差只警告；ONIOM 的 model 区仍须整分子；仅非周期体系（M5）。

## 0.2.2 — 2026-09-29

### 修复
- **ONIOM：低层 model 区计算用了全体系的电荷/多重度。** `E_low(model)` 现在用 low 的方法、model 区（`high`）的 charge/multiplicity。此前 low-only 区带电或开壳层时，`E_low(model)` 会静默算错或被电子数校验拒绝；"low-only 原子净电荷必须为 0"的限制已取消。**行为变更。**

### 新增
- ONIOM 的 topology 没有任何键时发 `UserWarning`（整分子校验看不到被切断的分子）。
- `ORCAPotential._create_backend(config=None)`。

### 变更
- `high` 与 `low` 是同一个对象时，`ONIOMPotential.summarize_timings()` 不再把每个 backend 列两遍。

### 验证
- ONIOM NVE（5 水团簇，HF/STO-3G : xTB，0.25 fs，1 ps）：漂移 +0.0067 kJ/mol/ps，标准差 0.0231 kJ/mol，732 ms/步，在 QM/MM v0.1 阈值内。
- 新增带电 low-only 区（中性 model 水 + Na⁺）的 ORCA 恒等检验；异构双层测试逐 backend 检查 MO restart 链。

## 0.2.1 — 2026-09-29

### 新增
- **双层减法 ONIOM（QM:QM）**：`ONIOMPotential(high=..., low=...).createONIOMSystem(topology, atoms)`，`E = E_high(model) + E_low(full) − E_low(model)`，层间机械耦合，每个 System 3 个独立 backend。示例 `examples/oniom_water_cluster.py`。
- 公共辅助 `validate_atom_indices`、`topology_masses`、`make_python_force(name=...)`。

## 0.2.0 — 2026-09-27（M3：restart 与健壮性）

### 新增
- SCF restart 状态机（从上一个成功的 `.gbw` MORead，失败后重试一次全新 SCF，绝不返回旧力）、失败包、逐步计时日志与 `summarize_timings()`、`nprocs > 1` 时的 MPI 调优。

### 验证
- restart 与非 restart 轨迹前 50 步能量差 < 1e-6 Eh；连续 1000 步无人工干预。

## 0.1 — 2026-09-27（M2：非周期电子嵌入）

### 新增
- `ORCAPotential.createSystem`（full-QM）与 `createMixedSystem`（经 `%pointcharges` / `.pcgrad` 的电子嵌入 QM/MM），支持 TIP4P 类虚拟位点。

### 验证
- QM 水 + MM 水团簇 NVE（HF/STO-3G，0.25 fs，1 ps）：漂移 +0.0056 kJ/mol/ps，标准差 0.0163 kJ/mol。
