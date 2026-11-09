# S3 可选模式实施计划（V3）

建立：2026-11-07。状态：**已批准（用户 2026-11-07 对"批准这份计划、从 Task 0 开工吗？"答"可以"）；Task 0 完成，见 [Task 0 记录](../validation/2026-11-07-s3-v3-task0.md)。** 原状态：计划，未开工。用户 2026-11-07 在"要我写 V3 的实施计划吗（只写文档……不写代码）？"之后答"继续"。本文只写做什么、在哪个文件、接口、测试和完成标准，不含实现代码。源码改动在用户批准本计划后开始。

## 依据

- [设计稿](../../openmm_orca_opi_design_plan.md) §1.2（2026-11-07 起允许 S3 作为可选模式）、§5.2 分层规则、§6.4 单位、§11（S2 定义）、§13 测试分层。
- [S3 提案](2026-11-07-s3-ewald-correction-proposal.md)：S3 定义 §2，力的链式关系 §3.2，验证阶段 §4。
- [V1 记录](../validation/2026-11-07-s3-v1-pyscf.md)：2 阶修正后首批相对误差 ≤ 0.12 kJ/mol；赝电荷必须正则化（K = 50、R_s = 2.0 nm、λ = 1e-4 时 |Q| ≤ 0.62 e）。
- [V2 记录](../validation/2026-11-07-s3-v2-orca.md)：ORCA 接受追加的赝电荷；S3 相对修正量与 PySCF 差 ≤ 0.26 kJ/mol。

## 1. 本计划定下的设计选择

以下由 Claude 按提案和 V1/V2 结果定下；用户可以推翻。

1. **阶数取 2**（提案 §6 第 2 条）。依据：V1 中 1 阶在产物样帧残差 0.43 kJ/mol，2 阶 ≤ 0.12，3 阶 ≤ 0.005。
2. **修正系数用 r_c 处的解析 Taylor 系数，不用 V1 的区域拟合。** V1 的系数来自 QM 原子周围随机点上的最小二乘，系数随 QM 几何和随机点变化，没有解析导数。改为 `dV` 在展开中心的 0–2 阶导数（共 10 个系数），对 MM 坐标和 r_c 都有解析导数。这改变了 V1 的数值定义，Task 0 先重新核对能量残差。
3. **展开中心 r_c = QM 请求坐标（含 link H）的算术平均。**
4. **赝电荷：K = 50，球半径 R_s = 2.0 nm，岭回归 λ = 1e-4**（V1/V2 已用）。Q = W c，W 为常数矩阵（K × 10），在以 r_c 为原点的固定模板点集上拟合一次，不随步变化。
5. **MM 力所需的 ∂E/∂Q_k 取 QM 在赝电荷位置的静电势（ESP）。** 对变分 SCF，∂E/∂Q_k 正是该点的 QM 总静电势（核 + 电子）。ORCA 自带 `orca_vpot`（设计稿 §2.1、§11.3）。用多极矩代替只在赝电荷势恰为多项式时成立，会引入与拟合残差同量级的不一致，只作备选。
6. **只支持全盒嵌入（`embeddingCutoff=None`）下启用**；截断嵌入与 S3 组合时报 `ValueError`。默认不启用。

## 2. 能量与力的定义

记 QM 请求坐标 R_a（含 link H，N 个），S2 点电荷 {q_j, r_j}，赝电荷 {Q_k, s_k}：

- r_c = (1/N) Σ_a R_a；s_k = r_c + R_s u_k（u_k 为固定单位向量）
- c = C(r_c, {q_j, r_j})：`dV = V_Ewald − V_S2` 在 r_c 处的 0–2 阶 Taylor 系数；V_Ewald 为同一批 S2 电荷的周期 Ewald 和（均匀背景，无 QM 自身镜像），V_S2 为同一批电荷在其 S2 放置位置上的直接库仑和
- Q = W c；E = E_QM(R; {q_j, r_j} ∪ {Q_k, s_k})

力（φ_k 为 QM 在 s_k 处的 ESP，g_k 为赝电荷 pcgrad，J_j = ∂c/∂r_j 为 10 × 3）：

- m̃ = Wᵀ φ（10 维，即 ∂E/∂c）
- MM 原子 j：F_j = F_j(pcgrad) − m̃ᵀ J_j
- r_c：∂E/∂r_c = Σ_k g_k + m̃ᵀ ∂c/∂r_c，其中 ∂c/∂r_c = −Σ_j J_j（核函数只依赖 r_c − r_j；镜像选择逐段为常数；背景项与位置无关）
- QM 请求原子：F_a = F_a(engrad) − (1/N) ∂E/∂r_c，然后按现有方式把 link H 的力回分到 Q1/M1

整体平移时 Σ F = 0，作为测试项。

## 3. 文件与接口

| 文件 | 改动 |
|---|---|
| `openmmorca/qmmm/ewald_correction.py`（新建） | 经典部分：Taylor 系数、Jacobian、赝电荷映射；只依赖 numpy 和 `units.py` |
| `openmmorca/backend/base.py` | `QMRequest` 增加可选 `esp_points_nm`；`QMResult` 增加可选 `esp_kj_mol_e`；`check_result` 校验形状与有限性 |
| `openmmorca/backend/fake.py` | 按假电荷解析计算 ESP |
| `openmmorca/backend/orca_opi.py`、`orca_files.py` | 有 `esp_points_nm` 时，SCF 成功后调用 `orca_vpot`；写点文件、读结果、单位换算、失败处理 |
| `openmmorca/force.py` | `QMMMCallback` 接受 `ewald_correction`；组装请求和力（§2）；写诊断 |
| `openmmorca/potential.py` | `createMixedSystem(..., ewaldCorrection: bool = False)` |
| `openmmorca/runtime/diagnostics.py` | timings 增加 `t_ewald`、`s3_q_absmax`、`s3_qm_extent_nm`（QM 原子离 r_c 的最大距离） |
| 设计稿 §11.6（新增）、README（中英日）、CHANGELOG | S3 的定义、限制和验证记录 |

接口（产出）：

```text
class EwaldCorrection:
    def __init__(self, order: int = 2, n_pseudo: int = 50, radius_nm: float = 2.0, ridge: float = 1e-4,
                 alpha_nm: float = 1.2, kmax: int = 10)          # Task 0：α 1.2、kmax 10 误差 ~1e-5
    def coefficients(self, center_nm, mm_positions_nm, mm_charges_e, box_vectors_nm, mm_positions_ewald_nm=None)
        -> tuple[np.ndarray, np.ndarray]      # c (10,), J (n_mm, 10, 3)
    def pseudo_charges(self, center_nm, c) -> tuple[np.ndarray, np.ndarray]   # positions (K,3), Q (K,)
    def forces(self, J, m_tilde, n_qm) -> tuple[np.ndarray, np.ndarray]       # MM 附加力、r_c 附加梯度
    W: np.ndarray                             # (K, 10)，构造时算好

QMRequest.esp_points_nm: np.ndarray | None = None
QMResult.esp_kj_mol_e: np.ndarray | None = None
ORCAPotential.createMixedSystem(..., ewaldCorrection: bool = False)
```

依赖规则（设计稿 §5.2）：`qmmm/ewald_correction.py` 不 import backend；`orca_vpot` 的调用、文件名和格式只出现在 `orca_opi.py` / `orca_files.py`；常数只来自 `units.py`。

## 4. 任务

### Task 0：开工前核对（实验目录，不改包）

在 `experiments/2026-11-0X_s3_v3_prep/` 中：

1. **Taylor 定义的能量残差。** 用解析 Taylor 系数（2 阶）重做 V1 的 PySCF 比较（首批 3 帧，直接多项式势和赝电荷两种），报告相对 `E − E_ENV20`。若残差明显大于 V1 的拟合定义，说明 Taylor 截断在 QM 区边缘不够，回到本计划 §1 第 2 条重定。
2. **`orca_vpot` 的用法与精度。** 在一帧上对 3 个赝电荷，比较 `orca_vpot` 给出的 ESP 与 ORCA 能量对该赝电荷电荷的中心差分 ∂E/∂Q_k；确认它包含核贡献、使用的密度名（如 SCF 密度）以及对 r2SCAN-3c 的适用性；记录每次调用的耗时。
3. **经典部分耗时。** 31,952 个电荷时，Taylor 系数和 Jacobian 的耗时随 kmax 的变化，以及 kmax 对系数的影响（与 kmax 收敛值比较）。

产出：一份验证记录，写明三项实测值。任一项不可行（例如 `orca_vpot` 拿不到 ESP），停下报告，改用 §1 第 5 条的备选并更新本计划。

### Task 1：`ewald_correction.py`

要求：

- 实空间：对每个电荷，Ewald 实空间项（最小镜像到 r_c）减去其在 S2 放置位置上的 1/r 项，求到 3 阶导数（2 阶系数的 Jacobian 需要 3 阶）。倒易空间：一般三斜盒的 k 向量；结构因子用逐轴可分离的相位因子。背景项为常数。
- `W`：在以原点为中心、半径 0.6 nm 的固定模板点集上（Task 0 后由 0.4 改为 0.6 nm），用岭回归拟合 K 个赝电荷重现 10 个单项式（不含 2 阶迹的部分无法由外部电荷表示，按 V1 §3 的方式保留残差）。
- 不支持非周期体系。

测试（`tests/test_ewald_correction.py`，不需要 ORCA）：

| 用例 | 期望 |
|---|---|
| `test_taylor_matches_direct_dV` | 随机电荷、立方盒：Taylor 多项式在 r_c 周围 0.4 nm 内与直接计算的 `V_Ewald − V_S2` 之差，3 阶项量级（与 `experiments/2026-11-07_c2_error_decomp/classical.py` 对照） |
| `test_ewald_self_consistency` | 两组 (α, kmax)：系数除常数项外一致到数值精度 |
| `test_jacobian_fd` | J 与 c 对 r_j 的中心差分一致 |
| `test_center_derivative_is_minus_sum` | ∂c/∂r_c 的有限差分 = −Σ_j J_j |
| `test_lattice_translation` | 任一电荷平移一个晶格矢量（S2 放置不变）时 c 不变 |
| `test_triclinic_box` | 三斜盒：与直接 Ewald 求值一致 |
| `test_pseudo_charges_reproduce_polynomial` | 赝电荷势与多项式在模板区域内的差 ≤ V1 记录的量级；|Q| 有界 |
| `test_neutral_and_charged_mm` | MM 净电荷 0 与 +1 时都可计算（背景项只改常数） |

### Task 2：后端 ESP

- `QMRequest.esp_points_nm` / `QMResult.esp_kj_mol_e`：给了点就必须返回同长度、有限的 ESP；不给时行为与现在完全相同。
- `FakeBackend`：ESP = 假电荷的库仑势。
- ORCA：SCF 成功后在 `current/` 内写点文件（首行点数，x y z 为 bohr）并运行 `orca_vpot qm.gbw qm.scfp <点文件> <输出>`（Task 0 核实：密度名 `qm.scfp`，需要 `qm.densities` 与 `qm.densitiesinfo`；输出每行 x y z V，Eh/e）；结果为 Eh/e，经 `units.py` 换算；`orca_vpot` 失败或输出不符按设计稿 §12 抛 `ORCAOutputError` 并写失败包；不重试。timings 记录 `t_vpot`。

测试：

| 用例 | 需要 ORCA | 期望 |
|---|---|---|
| `test_request_esp_points_shape` | 否 | 形状错误 → `ValueError` |
| `test_check_result_requires_esp` | 否 | 请求了 ESP 而结果缺失 → `ValueError` |
| `test_fake_esp_is_coulomb` | 否 | 与解析值一致 |
| `test_orca_esp_matches_charge_derivative` | 是 | H2O + 2 点电荷（HF/def2-SVP，TightSCF）：在第 3 个点放一个小电荷 Q，ESP 与 ∂E/∂Q 的中心差分一致，容差取设计稿 §1.3 v0.1 的有限差分标准换算到电势 |
| `test_orca_no_esp_unchanged` | 是 | 不给点时输入文件和结果与现在逐位相同 |

### Task 3：回调与 API

- `createMixedSystem(..., ewaldCorrection=True)` 只在 `embeddingCutoff=None` 时允许；否则 `ValueError`。
- 回调流程：现有 S2 选组 → r_c → c, J → 赝电荷追加到请求，`esp_points_nm` = 赝电荷位置 → 后端 → 按 §2 组装力 → link 回分 → 虚拟位点回分（现有）。
- 赝电荷不对应 OpenMM 粒子，其 pcgrad 只进入 r_c 项。
- docstring 写明：S3 不计 QM 自身周期镜像；QM 电荷不进入 OpenMM PME；仍不是 PME 一致的 QM/MM。

测试（`tests/test_periodic_fake.py` 增补，FakeBackend，Reference 平台）：

| 用例 | 期望 |
|---|---|
| `test_s3_requires_whole_box` | 截断嵌入 + S3 → `ValueError` |
| `test_s3_fd_qm_link_mm` | 小周期体系（溶剂化二肽 + link，或水盒）：对 QM 原子、Q1、M1、近处和最远处 MM 原子做中心差分，误差标准同现有 `test_periodic_forces_consistent_without_crossings` |
| `test_s3_momentum` | ‖Σ F‖ 相对最大单原子力在数值精度内 |
| `test_s3_lattice_invariance` | 整体平移一个晶格矢量、单个 MM 分子平移一个晶格矢量，能量不变 |
| `test_s3_reimaging_continuity` | 构造一个组在盒面换镜像的位移：S3 的能量跳变小于 S2 的跳变（提案 §5 的预期，测试报告两者实测值） |
| `test_s3_off_by_default` | 不传 `ewaldCorrection` 时，请求与现在逐位相同 |

### Task 4：真实 ORCA 检查

- `tests/test_periodic_orca.py` 增补：2.5 nm 水盒、QM 水 HF/STO-3G（B4 §8 的体系）开启 S3，晶格平移不变性与有限差分（QM O/H、近处和最远处 MM 原子）。
- 酶单帧（实验脚本，不进测试套件）：首批候选 0，r2SCAN-3c，对 C1、Cl1、OD2、Q1、M1 和一个远处 MM 原子各一个分量做中心差分。约 12 个分量 × 2 个单点 × 约 1 min，超过 5 分钟，执行前按全局规则报计划。

### Task 5：经 API 重现 V1/V2

用 `prepare_pilot.py` 的捕获方式，经 `createMixedSystem(..., ewaldCorrection=True)` 取出首批 3 帧的完整请求（含赝电荷），与 Task 0 的 Taylor 定义结果比较：经典系数逐位一致；PySCF/ORCA 能量在 SCF 收敛精度内一致。

### Task 6：文档与回归

- 设计稿新增 §11.6；README（三种语言）参数表与已知限制；CHANGELOG；TODO 与路线图状态。
- 回归：`-m 'not orca and not slow'` 全套；与改动相关的 ORCA 测试（`test_orca_backend.py`、`test_periodic_orca.py`、`test_potential_orca.py`）。
- 完成后写 `.commit_drafts/` 草稿。

## 5. 不在本计划内

- V4（B5 反应物区力偏移复核）、V5（C1 窗口统计）、V6（MD 冒烟与 1 ps）：Task 6 完成后另报计划。
- 截断嵌入下的 S3、非周期体系、QM 自身周期镜像、改变默认嵌入。
- 方案 C 的应用。

## 未确认（Claude 推断）

- `orca_vpot` 对 r2SCAN-3c 使用的密度、是否含核贡献、每次调用耗时：Task 0 第 2 项核实。
- 经典部分每步耗时：按诊断脚本 325 个点每帧约 2 s 推算，单中心 + Jacobian 可能在 1 s 以内，未测。
- 解析 Taylor 系数与 V1 拟合系数的能量差别：Task 0 第 1 项核实。

## 进度（2026-11-07 收工时）

- **Task 0 完成**，见 [Task 0 记录](../validation/2026-11-07-s3-v3-task0.md)。
- **Task 1 完成**：`openmmorca/qmmm/ewald_correction.py`。与计划的差别：不返回逐原子 Jacobian（10×3×N 的倒易空间收缩太贵），改为 `prepare()` 给系数与赝电荷、`mm_forces(step, m)` 直接给 MM 修正力，r_c 梯度为其和；用 `math.erf` 避免新增 scipy 依赖。`tests/test_ewald_correction.py` 11 passed（独立暴力 Ewald、三斜/剪切晶胞、力和中心梯度的有限差分）。首批 3 帧上与 Task 0 原型的系数/力差 ≤ 9e-6（两者 Ewald 截断量级）；每步 `prepare` 0.33 s、MM 力 0.6 s（181，8 线程）。
- **Task 2 完成**：`QMRequest.esp_points_nm` / `QMResult.esp_kj_mol_e`；FakeBackend 解析 ESP；ORCA 后端在 SCF 后调用 `orca_vpot qm.gbw qm.scfp esp.xyz esp.out`（`orca_files.write_esp_points/read_vpot`）；timings 新增 `t_vpot`、`t_ewald`、`s3_q_absmax`、`s3_qm_extent_nm`；失败包含 esp 文件。新增测试含真实 ORCA 的 ESP 与 ∂E/∂q 有限差分对照。
- **Task 3 完成**：`QMMMCallback(ewald_correction=...)`，`createMixedSystem(..., ewaldCorrection=False)`；只允许周期体系 + `embeddingCutoff=None`。`t_ewald` 记录的是 `prepare` 耗时，力的耗时在 `callback.last_ewald_force_s`。新增 fake 测试 7 个：与独立暴力 Ewald 的能量修正一致（差 2e-5 kJ/mol）、带 link 的有限差分、动量、晶格不变、换镜像跳变小于 S2、默认关闭。
- **Task 4 完成**（2026-11-09，183，`experiments/2026-11-09_s3_v3_enzyme_fd/`）：首批候选 0，r2SCAN-3c VeryTightSCF，S3 经 API；C1、Cl1、OD2、Q1（Asp CB）、M1（Asp CA）、最远水 O（离锚点 5.68 nm）各 x、y 分量，h = 1e-4 nm。12 个分量的解析力与中心差分最大差 0.034 kJ/mol/nm（判据 0.5；力 0.27–914 kJ/mol/nm）；QM/MM 力组合力 1.5e-4 kJ/mol/nm（最大单原子力 2439）。说明：最小离面距离 9e-6 nm 小于 h，Q1 正是镜像锚点，其差分对之间是否有电荷组换镜像未记录；该两个分量差 ≤ 0.008。基准单点 t_orca 83 s（25 个并发）、t_vpot 0.17 s、t_ewald 0.47 s。
- 原 Task 4 记录：`tests/test_periodic_orca.py::test_s3_orca_invariance_and_finite_difference`（2.5 nm 水盒，HF/STO-3G）通过。酶单帧有限差分（>5 分钟）**未做**，执行前按规则报计划。
- **Task 5 完成**（`experiments/2026-11-07_s3_v3_api/`）：经 API 捕获的请求，真实点电荷部分与首批逐位相同，赝电荷与 Task 0 原型差 ≤ 3.5e-7 e；捕获请求的 PySCF 能量与 Task 0 差 ≤ 8.4e-7 kJ/mol。经 API 的真实 ORCA（r2SCAN-3c）每帧 1 次 QM 调用，`t_orca` 52–54 s、`t_vpot` 0.10–0.13 s、`t_ewald` 0.44–0.48 s；ORCA 的 S3 相对修正量 `+10.575 / +16.116`，PySCF `+10.309 / +16.155 kJ/mol`，差 `+0.27 / −0.04`（与 V2 一致）。
- 回归：`-m 'not orca and not slow'` 223 passed；`test_periodic_orca.py`、`test_orca_backend.py`、`test_potential_orca.py` 21 passed。slow 测试未跑。
- **Task 6 完成**（2026-11-09）：设计稿 §11.6、README（英/中/日）S2/S3 段落、CHANGELOG（英/中）、TODO/路线图状态、`.commit_drafts/` 草稿。未重跑测试（Task 6 只改文档）。
