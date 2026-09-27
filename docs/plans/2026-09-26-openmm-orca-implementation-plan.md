# OpenMM–ORCA QM/MM 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal：** 实现 `openmmorca` 包：OpenMM 通过 `PythonForce` 调用 ORCA（经 OPI）计算 QM 区，支持非周期电子嵌入（v0.1）、restart 与健壮性（v0.2）、link atom（v0.3）、周期性 MM + 截断嵌入（v0.4）。

**Architecture：** 四层：`qmmm/`（OpenMM 侧的体系改造，不知道 ORCA 存在）→ `force.py`（PythonForce 薄回调）→ `backend/`（"QM 原子 + 点电荷 → 能量 + 力"协议，含 ORCA 后端与解析假后端）→ `runtime/`（scratch、restart、诊断）。层间只通过 `backend/base.py` 中的数据类交互。

**Tech Stack：** Python ≥ 3.10，OpenMM 8.5.2，OPI（`orca-pi`）2.0.0，ORCA 6.1.1，numpy，pytest 9。

**Spec：** `/home/ruigengji/openmm-ORCA/openmm_orca_opi_design_plan.md`（下文写作"spec §x.y"）。**执行者必须先通读 spec §2（已验证的事实与坑）再开始任何任务。**

**本文档的写法：** 本计划只规定"做什么、在哪个文件、对外接口长什么样、用什么用例验证、做到什么程度算完成"，**不含实现代码**。接口签名是约束：后续任务依赖这些名字和类型，改名必须同步修改本计划。

**版本管理：** 由团队按自己的 git 流程处理；每个任务结束是一个自然的提交点。

---

## Global Constraints

每个任务都隐含遵守以下约束（数值取自 spec）：

- 包名 `openmmorca`，仓库根 `/home/ruigengji/openmm-ORCA`，测试目录 `tests/`。
- 开发环境：`/home/ruigengji/miniforge3/envs/openmm_dev/bin/python`；ORCA 路径通过环境变量 `OPI_ORCA=/home/ruigengji/ORCA611` 指定。
- 依赖下限：`openmm>=8.5`、`orca-pi>=2.0`、`numpy`。不引入 ASE。PySCF 不进入运行依赖。
- backend 边界单位：输入坐标 nm、电荷 e；输出能量 kJ/mol、力 kJ/mol/nm（spec §6.4）。
- 物理常数只在 `openmmorca/units.py` 中定义：`BOHR_IN_ANGSTROM = 0.529177210903`，`HARTREE_IN_KJ_MOL = 2625.4996394799`，`COULOMB_KJ_MOL_NM = 138.93545764438198`（已核对与 OpenMM 内部值相同）。
- `qmmm/` 不得 import `backend/orca_opi.py` 或 `opi`；`backend/` 不得 import `openmm`。
- ORCA 文件名、关键字、`%` block 只出现在 `backend/orca_opi.py` 和 `backend/orca_files.py`。
- 嵌入电荷只能写 `%pointcharges` 文件，**禁止**使用 OPI 的 `PointCharge` / inline `Q`（spec §2.3）。
- **禁止**在 SCF 失败时返回旧能量或旧力（spec §9.2）。
- **ORCA 返回码不可信**，异常终止时也返回 0（spec §2.6）。
- 不修改 `/home/ruigengji/openmm-pyscf`（只读参考）。
- pytest 标记：需要 ORCA 的测试标 `@pytest.mark.orca`，运行超过约 1 分钟的标 `@pytest.mark.slow`；默认 `pytest` 不跑 slow。

## Review Focus

以下是 spec 隐含、但最容易在真实使用中出问题的五类情况。每条都已在所属任务里加了对应测试：

1. **读到上一步残留的输出文件**：本步 ORCA 失败但目录里还有上一步的 `qm.engrad`，后端静默返回旧力。期望：每步先清空 `current/`，失败必须抛异常。→ Task 4 用例 `test_stale_engrad_is_never_read`。
2. **QM 原子索引无序或不连续**（例如 `atoms=[7, 2, 3]`）：力被加到错误的原子上。期望：力严格按用户给出的 OpenMM 索引回填。→ Task 8 用例 `test_unsorted_qm_atoms_map_forces_correctly`。
3. **零电荷 MM 原子和带电虚拟位点**（TIP4P 类水模型：O 电荷为 0，M 位点带电且无质量）：零电荷原子不应进入点电荷文件；作用在 M 位点上的力必须经 OpenMM 分配到母原子。→ Task 7 用例 `test_zero_charge_mm_atoms_are_not_embedded`，Task 10 用例 `test_tip4p_virtual_site_embedding`。
4. **电荷与多重度不匹配**（例如电子数为奇数却给 `multiplicity=1`）：不应等到 ORCA 报一个难懂的错误。期望：构造后端时报 `ValueError` 并说明原因。→ Task 4 用例 `test_impossible_multiplicity_rejected`。
5. **同一个 ORCAPotential 创建两个 System / 两个 Context**：两个后端共用 scratch 或 `.gbw`，导致串味。期望：每次 `createSystem`/`createMixedSystem` 各自新建一个后端和独立的 uuid 目录。→ Task 9 用例 `test_each_system_gets_its_own_backend`。

---

## 文件结构总览

```text
/home/ruigengji/openmm-ORCA/
├── pyproject.toml                         Task 0
├── README.md                              Task 15
├── openmm_orca_opi_design_plan.md         spec（已存在）
├── openmmorca/
│   ├── __init__.py                        Task 0（导出 ORCAPotential 在 Task 9 加入）
│   ├── units.py                           Task 1
│   ├── errors.py                          Task 1
│   ├── backend/
│   │   ├── __init__.py                    Task 2
│   │   ├── base.py                        Task 2   协议与数据类
│   │   ├── orca_files.py                  Task 3   engrad / pcgrad / 点电荷文件
│   │   ├── orca_opi.py                    Task 4, 13, 14   ORCA 后端
│   │   └── fake.py                        Task 6   解析假后端
│   ├── qmmm/
│   │   ├── __init__.py                    Task 7   build_mixed_system
│   │   ├── system.py                      Task 7   System 改造
│   │   ├── linkatoms.py                   Task 16
│   │   ├── charges.py                     Task 17
│   │   ├── imaging.py                     Task 19
│   │   └── embedding.py                   Task 20
│   ├── runtime/
│   │   ├── __init__.py                    Task 4
│   │   ├── scratch.py                     Task 4
│   │   ├── restart.py                     Task 12
│   │   └── diagnostics.py                 Task 12
│   ├── force.py                           Task 8, 16, 21
│   └── potential.py                       Task 9, 16, 21
├── tests/
│   ├── conftest.py                        Task 0
│   ├── data/h2o_2pc.{pc,engrad,pcgrad}    Task 3
│   ├── helpers.py                         Task 7   测试用的小体系构造函数
│   └── test_*.py                          各任务
└── examples/
    ├── full_qm_water.py                   Task 10
    ├── qmmm_water_cluster_nve.py          Task 11
    ├── bench_nprocs.py                    Task 14
    ├── link_atom_dipeptide.py             Task 18
    └── enzyme_qmmm.py                     Task 22
```

---

## 里程碑与任务索引

| 里程碑 | 任务 | 版本 | 状态 |
|---|---|---|---|
| M0 后端冒烟 | Task 0–5 | — | 完成 |
| M1 OpenMM 层 + fake 后端 | Task 6–9 | — | 完成 |
| M2 非周期电子嵌入 | Task 10–11 | v0.1 | 完成（2026-09-27） |
| M3 restart 与健壮性 | Task 12–15 | v0.2 | 完成（2026-09-27），发布 0.2.0 |
| M4 link atom | Task 16–18 | v0.3 | 未开始 |
| M5 周期性 MM + 截断嵌入 | Task 19–22 | v0.4 | 未开始 |

Task 14 的基准只测到 nprocs=32（本机 4 个物理核留给别的工作），结果见 spec §2.2.1。

依赖关系：M0 与 M1 可以两人并行（M1 只依赖 Task 2 的协议）。M2 需要 M0 和 M1 都完成。M4、M5 需要 M3 完成；M5 的 Task 19–20 可与 M4 并行。

---

# M0：后端冒烟

### Task 0：项目骨架与测试基础设施

**文件：**
- 新建：`pyproject.toml`、`openmmorca/__init__.py`、`tests/conftest.py`

**要求：**
- `pyproject.toml`：setuptools 构建；项目名 `openmm-orca`，版本 `0.0.1`；依赖见 Global Constraints；`[tool.setuptools.packages.find] include = ["openmmorca*"]`；pytest 配置 `testpaths = ["tests"]`，注册 `orca`、`slow` 两个 marker，`addopts = "-m 'not slow'"`。
- `openmmorca/__init__.py`：暂时只定义 `__version__ = "0.0.1"`。
- `tests/conftest.py`：
  - 判定 ORCA 是否可用：`OPI_ORCA` 指向的目录中存在可执行文件 `orca`，或 `shutil.which("orca")` 非空。
  - 在 `pytest_collection_modifyitems` 中，ORCA 不可用时给所有 `orca` 标记的测试加 skip，原因写明"ORCA not found: set OPI_ORCA"。
  - fixture `scratch_root(tmp_path)`：返回 `tmp_path / "scratch"`，供所有后端测试使用，避免写到 `/tmp/openmmorca-<user>`。

**步骤：**
- [x] 创建上述三个文件
- [x] 在 openmm_dev 环境中可编辑安装：`/home/ruigengji/miniforge3/envs/openmm_dev/bin/python -m pip install -e /home/ruigengji/openmm-ORCA`
- [x] 运行 `python -c "import openmmorca; print(openmmorca.__version__)"`，期望输出 `0.0.1`
- [x] 运行 `pytest`，期望 "no tests ran"，退出码 5（pytest 没有收集到测试时的正常退出码）

**完成标准：** 可编辑安装成功；`pytest` 能启动。

---

### Task 1：单位与异常

**文件：**
- 新建：`openmmorca/units.py`、`openmmorca/errors.py`
- 测试：`tests/test_units.py`

**接口（产出）：**
```text
units.ANGSTROM_PER_NM: float = 10.0
units.BOHR_IN_ANGSTROM: float = 0.529177210903
units.HARTREE_IN_KJ_MOL: float = 2625.4996394799
units.HARTREE_PER_BOHR_IN_KJ_MOL_NM: float   # = HARTREE_IN_KJ_MOL * 10 / BOHR_IN_ANGSTROM ≈ 49614.752589
units.COULOMB_KJ_MOL_NM: float = 138.93545764438198
units.nm_to_angstrom(x: array_like) -> np.ndarray
units.hartree_to_kj_mol(energy_eh: float) -> float
units.gradient_to_forces(gradient_eh_bohr: array_like) -> np.ndarray   # 取负号并换算到 kJ/mol/nm

errors.OpenMMORCAError(RuntimeError)
errors.ORCACalculationError(OpenMMORCAError)   # ORCA 非正常终止 / SCF 不收敛
errors.ORCAOutputError(OpenMMORCAError)        # 输出文件缺失、格式错、数量不符、NaN
errors.ORCATimeoutError(OpenMMORCAError)       # 超时
```
配置错误一律抛内置 `ValueError`，不另建异常类。

**测试用例（`tests/test_units.py`）：**

| 用例 | 输入 | 期望 |
|---|---|---|
| `test_force_conversion_factor` | — | `HARTREE_PER_BOHR_IN_KJ_MOL_NM` 与 49614.752589 的相对误差 < 1e-9 |
| `test_gradient_to_forces_flips_sign` | 梯度 `[[1.0, -2.0, 0.0]]` Eh/bohr | 结果为 `[[-49614.75…, 99229.50…, 0]]` |
| `test_nm_to_angstrom` | `[[0.1, 0.2, 0.3]]` | `[[1, 2, 3]]` |
| `test_coulomb_constant_matches_openmm` | OpenMM Reference 平台：2 个 +1 e 粒子相距 1 nm，NonbondedForce NoCutoff，sigma=0.1、epsilon=0 | Context 势能与 `COULOMB_KJ_MOL_NM` 之差 < 1e-9 kJ/mol |
| `test_error_hierarchy` | — | 三个具体异常都是 `OpenMMORCAError` 和 `RuntimeError` 的子类 |

**步骤：**
- [x] 写 `tests/test_units.py`，运行 `pytest tests/test_units.py -v`，确认因 import 失败而 FAIL
- [x] 实现 `units.py`、`errors.py`
- [x] 运行 `pytest tests/test_units.py -v`，全部 PASS

**完成标准：** 5 个用例通过。

---

### Task 2：backend 协议

**文件：**
- 新建：`openmmorca/backend/__init__.py`（空）、`openmmorca/backend/base.py`
- 测试：`tests/test_backend_base.py`

**接口（产出）：**
```text
@dataclass(frozen=True)
class QMRequest:
    qm_elements: tuple[str, ...]
    qm_positions_nm: np.ndarray            # (n_qm, 3)
    mm_positions_nm: np.ndarray | None = None   # (n_mm, 3)
    mm_charges_e: np.ndarray | None = None      # (n_mm,)
    step: int = 0
    n_qm: int      # property
    n_mm: int      # property；无点电荷时为 0

@dataclass(frozen=True)
class QMResult:
    energy_kj_mol: float
    qm_forces_kj_mol_nm: np.ndarray        # (n_qm, 3)
    mm_forces_kj_mol_nm: np.ndarray | None # (n_mm, 3)；n_mm > 0 时必须非 None
    timings_s: dict[str, float] = {}       # 默认空 dict（用 field(default_factory=dict)）

class QMBackend(Protocol):
    def evaluate(self, request: QMRequest) -> QMResult: ...
    def close(self) -> None: ...

def check_result(request: QMRequest, result: QMResult) -> None   # 不匹配时抛 ValueError
```

**要求：**
- `QMRequest.__post_init__` 把数组转成 float ndarray 并校验：QM 坐标形状必须是 `(len(qm_elements), 3)`；MM 坐标与 MM 电荷必须同时给或同时不给，且数量一致。
- `check_result` 检查：能量有限；QM 力形状正确且有限；`n_mm > 0` 时 MM 力非 None、形状正确且有限。

**测试用例（`tests/test_backend_base.py`）：**

| 用例 | 输入 | 期望 |
|---|---|---|
| `test_request_shapes` | 3 个元素 + (3,3) 坐标 + 2 个点电荷 | `n_qm == 3`，`n_mm == 2` |
| `test_request_rejects_wrong_qm_shape` | 3 个元素 + (2,3) 坐标 | `ValueError` |
| `test_request_requires_positions_and_charges_together` | 只给 `mm_positions_nm` | `ValueError` |
| `test_request_rejects_count_mismatch` | 2 个 MM 坐标 + 3 个电荷 | `ValueError` |
| `test_check_result_accepts_full_qm` | `n_mm == 0`，`mm_forces=None` | 不抛异常 |
| `test_check_result_requires_mm_forces` | `n_mm == 2`，`mm_forces=None` | `ValueError`，信息含 "MM forces" |
| `test_check_result_rejects_nan` | QM 力中含 NaN | `ValueError` |
| `test_check_result_rejects_wrong_shape` | MM 力形状 (3,3)，但 `n_mm == 2` | `ValueError` |

**步骤：**
- [x] 写测试，运行确认 FAIL
- [x] 实现 `base.py`
- [x] 运行 `pytest tests/test_backend_base.py -v`，全部 PASS

**完成标准：** 8 个用例通过。M1 的同事从这里开始可以并行。

---

### Task 3：ORCA 文件读写

**文件：**
- 新建：`openmmorca/backend/orca_files.py`
- 新建：`tests/data/h2o_2pc.pc`、`tests/data/h2o_2pc.engrad`、`tests/data/h2o_2pc.pcgrad`
- 测试：`tests/test_orca_files.py`

**接口（产出）：**
```text
write_pointcharges(path: Path, charges_e: array_like, positions_ang: array_like) -> None
read_engrad(path: Path, n_atoms: int) -> tuple[float, np.ndarray]     # (Eh, (n_atoms,3) Eh/bohr)
read_pcgrad(path: Path, n_charges: int) -> np.ndarray                  # (n_charges,3) Eh/bohr
```

**要求：**
- 文件格式见 spec §2.3–§2.5。点电荷文件每个数字用 `%.10f` 格式写出。
- 读取失败（文件不存在、解析错误、数量不符、NaN/Inf）一律抛 `ORCAOutputError`，信息中包含文件路径。
- `write_pointcharges` 的电荷数与坐标数不一致时抛 `ValueError`。

**测试数据（实测文件，按原样保存；计算条件：HF/def2-SVP EnGrad，H2O 几何 O(0,0,0)、H(0.96,0,0)、H(−0.24,0.93,0) Å）：**

`tests/data/h2o_2pc.pc`：
```text
2
 -0.834  3.000  0.000  0.000
  0.417  3.500  0.800  0.000
```

`tests/data/h2o_2pc.engrad`：
```text
#
# Number of atoms
#
 3
#
# The current total energy in Eh
#
    -75.977160671564
#
# The current gradient in Eh/bohr
#
      -0.003408773893
      -0.019449988281
      -0.000000000000
      -0.000347458111
       0.003871343602
       0.000000000000
      -0.005745074316
       0.017869707398
       0.000000000000
#
# The atomic numbers and current coordinates in Bohr
#
   8     0.0000000    0.0000000    0.0000000 
   1     1.8141371    0.0000000    0.0000000 
   1    -0.4535343    1.7574453    0.0000000 
```

`tests/data/h2o_2pc.pcgrad`：
```text
2
   0.012533896858  -0.001642805287  -0.000000000000
  -0.003032590537  -0.000648257432   0.000000000000
```

**测试用例（`tests/test_orca_files.py`）：**

| 用例 | 输入 | 期望 |
|---|---|---|
| `test_read_engrad_fixture` | `h2o_2pc.engrad`，n_atoms=3 | 能量 == −75.977160671564；梯度形状 (3,3)；第 0 行 == (−0.003408773893, −0.019449988281, 0) |
| `test_read_engrad_atom_count_mismatch` | 同上文件，n_atoms=4 | `ORCAOutputError` |
| `test_read_engrad_missing_file` | 不存在的路径 | `ORCAOutputError`，信息含该路径 |
| `test_read_engrad_nan` | 把 fixture 复制到 tmp_path，第一个梯度值改成 `nan` | `ORCAOutputError` |
| `test_read_pcgrad_fixture` | `h2o_2pc.pcgrad`，n_charges=2 | 第 0 行 == (0.012533896858, −0.001642805287, 0) |
| `test_read_pcgrad_count_mismatch` | 同上，n_charges=3 | `ORCAOutputError` |
| `test_write_pointcharges_format` | 电荷 [−0.834, 0.417]，坐标同 fixture | 第一行为 `2`；逐行解析出的数值与输入相同（容差 1e-10） |
| `test_write_pointcharges_count_mismatch` | 2 个电荷、3 个坐标 | `ValueError` |

**步骤：**
- [x] 建立 `tests/data/` 下三个文件（内容逐字符照抄上文）
- [x] 写测试，运行确认 FAIL
- [x] 实现 `orca_files.py`
- [x] 运行 `pytest tests/test_orca_files.py -v`，全部 PASS

**完成标准：** 8 个用例通过。

---

### Task 4：ORCA/OPI 后端（无 restart）与 scratch 目录

**文件：**
- 新建：`openmmorca/runtime/__init__.py`（空）、`openmmorca/runtime/scratch.py`
- 新建：`openmmorca/backend/orca_opi.py`
- 测试：`tests/test_scratch.py`、`tests/test_orca_backend.py`

**接口（产出）：**
```text
class ScratchDir:
    root: Path; current: Path; restart: Path; failures: Path    # 均为 resolve() 后的绝对路径
    def __init__(self, root: Path)                               # 创建三个子目录
    @classmethod
    def create(cls, scratch_root: str | Path | None = None) -> ScratchDir
        # 默认 scratch_root = tempfile.gettempdir()/openmmorca-<user>；在其下新建 uuid4().hex 子目录
    def reset_current(self) -> None      # 删除 current/ 下所有内容（保留目录本身）
    def cleanup(self) -> None            # 删除 current/ 和 restart/；保留 failures/ 与 timings.csv；幂等

@dataclass(frozen=True)
class ORCAConfig:
    method: str
    basis: str | None = None
    charge: int = 0
    multiplicity: int = 1
    nprocs: int = 1
    maxcore_mb: int = 2000
    extra_keywords: tuple[str, ...] = ()
    extra_blocks: tuple[str, ...] = ()
    scratch_root: str | None = None
    restart: bool = True              # Task 13 起生效
    keep_failed: bool = True          # Task 13 起生效
    max_fresh_retries: int = 1        # Task 13 起生效
    timeout_s: float | None = None
    orca_path: str | None = None      # None 时由 OPI 读取 OPI_ORCA 或 PATH

class ORCAOPIBackend:                 # 实现 QMBackend 协议
    BASENAME = "qm"
    config: ORCAConfig
    scratch: ScratchDir
    def __init__(self, config: ORCAConfig, *, check_version: bool = True)
    def evaluate(self, request: QMRequest) -> QMResult
    def close(self) -> None
```

**要求：**

1. **`ORCAConfig.__post_init__` 校验**（违规时抛 `ValueError`）：
   - `nprocs >= 1`，`maxcore_mb >= 1`，`multiplicity >= 1`。
   - `extra_keywords` 中不得出现任务类关键字（不区分大小写）：`SP`、`Opt`、`Freq`、`NumFreq`、`MD`、`EnGrad`、`NumGrad`、`MORead`。
   - `extra_blocks` 中每个字符串去掉首部空白后，不得以这些前缀开头（不区分大小写）：`%pointcharges`、`%pal`、`%maxcore`、`%moinp`、`%output`。
2. **构造**：创建 `ScratchDir`；`check_version=True` 时用 OPI `Runner(working_dir=scratch.current)` 调用一次 `check_version()`（`orca_path` 非 None 时先调用 `runner.set_orca_path(Path(orca_path))`）。Runner 在第一次需要时创建并缓存，保证 `check_version=False` 且机器上没有 ORCA 时也能构造后端（Task 13 的单元测试依赖这一点）。
3. **电荷与多重度检查**（Review Focus 4）：第一次 `evaluate` 时，用 QM 元素的核电荷总和减去 `charge` 算出电子数；电子数奇偶性与 `multiplicity` 不相容时抛 `ValueError`，信息中给出电子数和多重度。（也可以直接用 OPI `Structure.multiplicity_is_possible()`，但要保证报错信息可读。）
4. **每步流程**（spec §8.2，此任务不做 restart）：
   1. `scratch.reset_current()`
   2. 有点电荷时写 `current/pc.pc`（nm→Å）
   3. 用 OPI 构造输入并写出 `current/qm.inp`。OPI 用法与坑见 spec §2.7；要点：
      - `Calculator("qm", working_dir=scratch.current, version_check=False)`，并设 `json_via_input = False`
      - `Structure(atoms=[Atom(el, coordinates=(x, y, z)), ...], charge=..., multiplicity=...)`，坐标必须用关键字参数传
      - simple keywords（字符串）顺序：`method`、`basis`（若有）、`extra_keywords`、`EnGrad`
      - `input.memory = maxcore_mb`；`nprocs > 1` 时设 `input.ncores = nprocs`
      - arbitrary strings（必须传 str）：`%output jsonpropfile true end`；有点电荷时加 `%pointcharges "pc.pc"`；再依次加入 `extra_blocks`
   4. 运行 ORCA：`runner.run_orca(inpfile, timeout=...)`；`subprocess.TimeoutExpired` 转为 `ORCATimeoutError`
   5. 校验（spec §8.5）：`Output("qm", working_dir=..., version_check=False)`；`terminated_normally()` 为 False → `ORCACalculationError`；方法不是 xTB 时 `scf_converged()` 为 False → `ORCACalculationError`（xTB 的输出里是否有 SCF 的 `SUCCESS` 标志尚未验证，实现时先跑一次 `method="XTB"` 查看 `.out`，把结论写进代码注释和 spec §2.6）
   6. `read_engrad(current/qm.engrad, n_qm)`；有点电荷时 `read_pcgrad(current/qm.pcgrad, n_mm)`
   7. 交叉校验：`out.parse(do_create_gbw_json=False, read_gbw_json=False)` 之后 `abs(out.get_final_energy() - engrad 能量) < 1e-8`，否则抛 `ORCAOutputError`
   8. 单位换算后返回 `QMResult`；`timings_s` 至少包含 `write`、`orca`、`read`、`total` 四项
5. **`close()`**：调用 `scratch.cleanup()`；幂等。同时用 `weakref.finalize` 注册 cleanup，保证对象被回收或进程退出时也会清理。

**测试用例（`tests/test_scratch.py`，不需要 ORCA）：**

| 用例 | 期望 |
|---|---|
| `test_create_makes_uuid_dirs` | 在 `scratch_root` 下建出唯一子目录，`current`/`restart`/`failures` 都存在；连续创建两次得到不同目录 |
| `test_reset_current_removes_files_and_subdirs` | 在 current 下放一个文件和一个子目录，reset 后 current 为空但仍存在 |
| `test_cleanup_keeps_failures_and_timings` | failures 下有内容、root 下有 `timings.csv` 时，cleanup 后这两者保留，current 与 restart 被删除；再调一次 cleanup 不报错 |
| `test_cleanup_removes_empty_root` | 什么都没留下时 cleanup 删除整个 root |

**测试用例（`tests/test_orca_backend.py`）：** 除特别说明外均标 `@pytest.mark.orca`；几何为 Task 3 中的 H2O；配置 `method="HF", basis="def2-SVP"`，`scratch_root` 用 fixture。

| 用例 | 输入 | 期望 |
|---|---|---|
| `test_config_rejects_task_keywords`（不需要 ORCA） | `extra_keywords=("Opt",)`，以及 `("engrad",)` | 两者都抛 `ValueError` |
| `test_config_rejects_managed_blocks`（不需要 ORCA） | `extra_blocks=("%pal nprocs 4 end",)`，以及 `("  %PointCharges \"x\"",)` | 两者都抛 `ValueError` |
| `test_full_qm_matches_reference` | 无点电荷 | 能量换算回 Eh 后与 −75.960838761435 之差 < 1e-8；`mm_forces_kj_mol_nm is None`；`qm.inp` 中没有 `%pointcharges` |
| `test_point_charges_match_fixture` | Task 3 中的两个点电荷 | 能量与 −75.977160671564 之差 < 1e-8 Eh；MM 力等于 `gradient_to_forces(fixture pcgrad)`，容差为 1e-8 Eh/bohr 对应的 kJ/mol/nm |
| `test_input_file_contents` | 同上 | `current/qm.inp` 中包含 `EnGrad`、`%pointcharges "pc.pc"`、`jsonpropfile`；不包含 `jsongbwfile`；坐标块中没有以 `Q` 开头的行 |
| `test_scf_failure_raises` | `extra_blocks=("%scf maxiter 2 end",)` | 抛 `ORCACalculationError` |
| `test_stale_engrad_is_never_read`（Review Focus 1） | 同一个后端先成功算一步（此时 `current/qm.engrad` 存在）；然后 `backend.config = dataclasses.replace(backend.config, extra_blocks=("%scf maxiter 2 end",), restart=False)`，稍微改动几何再算一步 | 第二步抛 `ORCACalculationError`，而不是返回第一步的能量（`restart=False` 是必要的：Task 13 之后 MORead 两圈就能收敛，失败就制造不出来） |
| `test_impossible_multiplicity_rejected`（Review Focus 4） | H2O，`charge=0, multiplicity=2` | `evaluate` 在启动 ORCA 之前就抛 `ValueError`，信息含 "10 electrons" 和 "multiplicity 2" |
| `test_timings_reported` | 任一成功步 | `timings_s` 含 `write`、`orca`、`read`、`total`，且都 ≥ 0 |
| `test_close_removes_scratch` | 成功一步后 `close()` | `scratch.current` 不存在；再次 `close()` 不报错 |
| `test_xtb_runs` | `method="XTB", basis=None`，带 2 个点电荷 | 返回有限的能量、QM 力和 MM 力 |

**步骤：**
- [x] 写 `tests/test_scratch.py`，确认 FAIL；实现 `scratch.py`；确认 PASS
- [x] 写 `tests/test_orca_backend.py`，确认 FAIL
- [x] 实现 `ORCAConfig` 及其校验；确认两个不需要 ORCA 的配置用例 PASS
- [x] 实现 `ORCAOPIBackend`；设置 `export OPI_ORCA=/home/ruigengji/ORCA611` 后运行 `pytest tests/test_orca_backend.py -v -m orca`，全部 PASS
- [x] 如果 xTB 的 `scf_converged` 行为与假设不符，按实测修正实现，并更新 spec §2.6

**完成标准：** 上述用例全部通过；scratch 目录在测试结束后没有残留（`failures/` 除外）。

**坑：** spec §2.6、§2.7 的每一条都会在这个任务里遇到，动手前先读。

---

### Task 5：ORCA 后端有限差分

**文件：**
- 测试：`tests/test_orca_fd.py`（全部 `@pytest.mark.orca`）

**要求：** 这一步只加测试，不改实现。测试里自带一个小工具函数：给定后端和请求，对某一坐标做中心差分 `(E(+h) − E(−h)) / 2h`，单位统一换算为 Eh/bohr 后再比较。

**测试用例：** 配置 `HF/def2-SVP`，`extra_keywords=("TightSCF",)`，步长 h = 1e-3 Å。

| 用例 | 覆盖的坐标 | 期望 |
|---|---|---|
| `test_fd_qm_atoms_full_qm` | H2O 全部 9 个坐标，无点电荷 | 每个分量上，解析梯度与有限差分之差 < 1e-4 Eh/bohr |
| `test_fd_qm_atoms_with_charges` | H2O 全部 9 个坐标，带 2 个点电荷 | 同上 |
| `test_fd_point_charges` | 2 个点电荷的全部 6 个坐标 | 同上（spec §2.4 实测误差为 7e-8，远小于阈值） |
| `test_forces_sum_to_zero` | 带 2 个点电荷的一次计算 | ‖Σ F_QM + Σ F_MM‖ < 1e-3 × 所有原子中最大的 ‖F_i‖ |

**步骤：**
- [x] 写测试并运行 `pytest tests/test_orca_fd.py -v`（约 30 次 ORCA 调用，耗时 30 秒左右）
- [x] 全部 PASS。如果失败，说明 Task 4 的单位或符号有错，回到 Task 4 修正，不要放宽阈值

**完成标准：** 4 个用例通过。**M0 完成。**

---

# M1：OpenMM 层 + fake 后端

### Task 6：解析假后端

**文件：**
- 新建：`openmmorca/backend/fake.py`
- 测试：`tests/test_fake_backend.py`

**接口（产出）：**
```text
class FakeBackend:                           # 实现 QMBackend 协议
    def __init__(self, qm_charges_e: Sequence[float], k_bond: float = 1000.0)
    n_calls: int                             # evaluate 被调用的次数
    last_request: QMRequest | None
    def evaluate(self, request: QMRequest) -> QMResult
    def close(self) -> None                  # 空操作
```

**要求：** 势能按 spec §13.2 定义：

- 所有 QM 原子对之间加谐振子项 ½ k (r − r0)²，其中 r0 取**第一次** evaluate 时的原子间距。
- 每个 QM 原子带固定假电荷 `qm_charges_e[a]`，与所有嵌入电荷之间算库仑相互作用，库仑常数用 `units.COULOMB_KJ_MOL_NM`。
- 力是上述能量的精确负梯度，QM 力和 MM 力都要返回。
- `qm_charges_e` 的长度与 `request.n_qm` 不一致时抛 `ValueError`。
- 用 numpy 向量化实现，因为 Task 8 的测试会大量调用它。

**测试用例：**

| 用例 | 期望 |
|---|---|
| `test_equilibrium_has_zero_bond_energy` | 第一次调用时，谐振子部分的能量为 0（无点电荷时总能量为 0） |
| `test_fd_qm_and_mm` | 3 个 QM 原子、2 个点电荷，先用参考几何调用一次，再在扰动几何上对全部 15 个坐标做中心差分（h = 1e-6 nm）：误差 < 1e-5 kJ/mol/nm |
| `test_newton_third_law` | 任意几何下 ‖Σ F‖ < 1e-9 |
| `test_charge_count_mismatch` | 给 2 个假电荷，但请求中有 3 个 QM 原子 → `ValueError` |

**步骤：**
- [x] 写测试，确认 FAIL
- [x] 实现
- [x] 运行测试，全部 PASS

**完成标准：** 4 个用例通过。

---

### Task 7：OpenMM System 改造

**文件：**
- 新建：`openmmorca/qmmm/system.py`、`openmmorca/qmmm/__init__.py`
- 新建：`tests/helpers.py`（测试用的小体系构造函数）
- 测试：`tests/test_qmmm_system.py`

**接口（产出）：**
```text
# qmmm/system.py
SUPPORTED_FORCE_TYPES: tuple[type, ...]   # spec §6.2 白名单
copy_system(system: openmm.System) -> openmm.System            # XmlSerializer 往返深拷贝
check_supported_forces(system) -> None                          # 不在白名单的力 → ValueError（信息列出类名）
get_nonbonded_force(system) -> openmm.NonbondedForce            # 不是恰好一个 → ValueError
zero_qm_bonded_terms(system, qm_atoms: Sequence[int]) -> int    # 返回置零的项数；规则见 spec §6.3
apply_qm_embedding(nonbonded, qm_atoms) -> dict[int, float]     # spec §6.2；返回被置零的 QM 原电荷
remove_qm_constraints(system, qm_atoms) -> int                  # 删除两端都在 QM 区的约束，返回删除数
particle_charges(nonbonded) -> np.ndarray                       # 每个粒子的电荷（e）
check_whole_molecules(topology, qm_atoms) -> None               # 有一端 QM、一端 MM 的键 → ValueError

# qmmm/__init__.py
@dataclass(frozen=True)
class MixedSystemParts:
    system: openmm.System               # 改造后的拷贝（还没有加 QM 力）
    qm_atoms: tuple[int, ...]           # 保持用户给出的顺序
    qm_elements: tuple[str, ...]
    mm_atoms: tuple[int, ...]           # 只含 |q| > 1e-12 的 MM 粒子（包括带电的虚拟位点）
    mm_charges_e: np.ndarray            # 与 mm_atoms 一一对应，取自原始 System
    removed_qm_charges: dict[int, float]

build_mixed_system(topology, system, qm_atoms, remove_constraints=True) -> MixedSystemParts
```

**要求：**
- `build_mixed_system` 的顺序：校验索引（非空、无重复、无负数、不越界）→ `check_whole_molecules` → `copy_system` → `check_supported_forces` → 检查 NonbondedForce 没有参数 offset → 在改造前先记录全体粒子电荷 → `zero_qm_bonded_terms` → `apply_qm_embedding` → 若 `remove_constraints` 为 True 则 `remove_qm_constraints`，为 False 且 QM 区内部有约束时发出 `UserWarning`。
- **周期性体系在 M1–M3 中不支持**：`system.usesPeriodicBoundaryConditions()` 为 True 时，`build_mixed_system` 抛 `NotImplementedError`，信息写 "periodic QM/MM is planned for M5"。
- 键合项"删除"用置零实现（spec §6.3）：HarmonicBond/Angle 的 k=0，PeriodicTorsion 的 k=0，RBTorsion 的 c0..c5 全部为 0；CMAP 项改指向一张新加的全零 map。规则是**所有原子都在 QM 区才置零**。
- 元素符号取自 `topology`；有原子没有 element 时抛 `ValueError`。
- `tests/helpers.py` 提供以下函数：
  - `water_topology(n_waters)`：残基名 HOH，原子名 O/H1/H2，包含键
  - `water_positions(n_waters, spacing_nm=0.3)`：各水分子均处于力场平衡几何，沿 x 轴排列
  - `flexible_tip3p_system(topology)`：`ForceField("tip3p.xml")`，`rigidWater=False`，`constraints=None`，`NoCutoff`
  - `tip4pew_system(topology)`：`ForceField("tip4pew.xml")`，需要用 `Modeller.addExtraParticles` 加入 M 位点，`rigidWater=False`，`NoCutoff`

**测试用例：**

| 用例 | 体系 | 期望 |
|---|---|---|
| `test_bonded_rule_all_qm_only` | 手工搭的 4 粒子链 0–1–2–3：键 (0,1)、(1,2)、(2,3)；角 (0,1,2)、(1,2,3)；二面角 (0,1,2,3)；QM = {0,1,2} | 键 (0,1)、(1,2) 和角 (0,1,2) 的力常数为 0；键 (2,3)、角 (1,2,3)、二面角 (0,1,2,3) 不变；返回值为 3 |
| `test_rb_torsion_all_qm_zeroed` | 4 粒子 RBTorsion，全部为 QM | c0..c5 全为 0 |
| `test_qm_charges_zeroed_and_returned` | 2 个水，QM = 水 0 | 3 个 QM 粒子电荷为 0；返回的 dict 中是原始电荷（O −0.834，H 0.417） |
| `test_qm_qm_exceptions` | 同上 | 所有 QM–QM 原子对都有 chargeProd = 0、epsilon = 0 的 exception |
| `test_exception_count_scales_with_qm_only` | 50 个水，QM = 水 0 | 新增的 exception 数 ≤ 3（与 MM 原子数无关） |
| `test_qm_mm_lj_retained` | 2 个水 | 改造后的 System 中，QM O 与 MM O 之间的 LJ 能量等于手工计算的 4ε[(σ/r)¹² − (σ/r)⁶] |
| `test_mm_mm_unchanged` | 3 个水，QM = 水 0（处于平衡几何） | 把水 0 平移 100 nm 后，改造后 System 与原 System 的总能量之差 < 1e-6 kJ/mol（此时两者的差别只剩 QM 内部键合能（平衡几何下为 0）和 100 nm 外可忽略的 QM–MM 相互作用，所以任何对 MM–MM 项的误改都会暴露出来） |
| `test_unsupported_force_rejected` | 原 System 加一个 `CustomExternalForce` | `ValueError`，信息含 "CustomExternalForce" |
| `test_parameter_offsets_rejected` | NonbondedForce 加一个 particle parameter offset | `ValueError` |
| `test_split_molecule_rejected` | 2 个水，QM = [0, 1]（O 和 H1，漏掉 H2） | `ValueError`，信息含 "whole molecules" |
| `test_bad_indices_rejected` | 空列表、重复索引、负数、越界 | 全部 `ValueError` |
| `test_periodic_not_yet_supported` | 带盒向量、`PME` 的水体系 | `NotImplementedError` |
| `test_constraints_removed_inside_qm` | tip3p 刚性水（`rigidWater=True`），QM = 水 0 | 水 0 的 3 个约束被删除，水 1 的约束保留 |
| `test_constraints_kept_with_warning` | 同上，`remove_constraints=False` | 约束全部保留，并发出 `UserWarning` |
| `test_zero_charge_mm_atoms_are_not_embedded`（Review Focus 3） | TIP4P-Ew，2 个水，QM = 水 0 | `mm_atoms` 中不含 MM 水的 O（电荷 0），但含 M 位点；`mm_charges_e` 之和等于 MM 水的净电荷（0） |
| `test_qm_order_preserved` | QM = [2, 0, 1] | `parts.qm_atoms == (2, 0, 1)`，`qm_elements` 顺序与之对应 |

**步骤：**
- [x] 写 `tests/helpers.py`，并先用一个冒烟测试确认两个力场的 System 能正常创建
- [x] 写 `tests/test_qmmm_system.py`，确认 FAIL
- [x] 逐个实现 `system.py` 中的函数，每实现一个就跑一次对应用例
- [x] 实现 `build_mixed_system`
- [x] 运行 `pytest tests/test_qmmm_system.py -v`，全部 PASS

**完成标准：** 16 个用例通过；全程不需要 ORCA。

---

### Task 8：PythonForce 回调

**文件：**
- 新建：`openmmorca/force.py`
- 测试：`tests/test_force_fake.py`

**接口（产出）：**
```text
class QMMMCallback:
    def __init__(self, backend: QMBackend, qm_atoms: Sequence[int], qm_elements: Sequence[str],
                 n_particles: int, mm_atoms: Sequence[int] = (), mm_charges_e: Sequence[float] = ())
    step: int                                    # 已完成的回调次数
    def __call__(self, state: openmm.State) -> tuple[float, np.ndarray]
        # 返回 (能量 kJ/mol, 力数组 (n_particles, 3) kJ/mol/nm)，都是普通浮点数，不带 openmm.unit

def make_python_force(callback: QMMMCallback, force_group: int = 0) -> openmm.PythonForce
```

**要求：**
- `__call__` 的流程：`state.getPositions(asNumpy=True)` 转为 nm 数组 → 检查粒子数 → 构造 `QMRequest(step=self.step)` → `backend.evaluate` → `check_result` → 按索引把力填入一个全零数组（QM 用赋值，MM 用 `+=`）→ `step += 1`。
- 回调中不做任何单位换算以外的物理计算。
- `make_python_force` 设置 force group，并 `setName("ORCA QM/MM")`。
- 已知限制（写进 docstring）：包含该力的 System 不能用 `XmlSerializer` 序列化（回调对象持有锁和临时目录，无法 pickle）。实现时要实测确认 `openmm.Context` 的创建不需要 pickle；如果需要，就把这个结论写进 spec §2.8 并在这里改变设计。

**测试用例（全部使用 FakeBackend 和 Reference 平台）：**

| 用例 | 体系 | 期望 |
|---|---|---|
| `test_full_qm_energy_and_forces` | 1 个水，只有 PythonForce | Context 能量与力 == 直接调用 FakeBackend 的结果（容差 1e-10） |
| `test_mixed_energy_decomposition` | 2 个 flexible tip3p 水，QM = 水 0，水 0 处于平衡几何；`build_mixed_system` 后加 PythonForce（force group 1），假电荷用 tip3p 电荷 | ① group 1 的能量 == FakeBackend 能量；② 其余 group 的能量 == 原 System 总能量 − 手算的 QM–MM 库仑能（9 对，用 `COULOMB_KJ_MOL_NM`），容差 1e-6 kJ/mol。这一条同时证明 QM–MM 静电被干净地去掉、其他相互作用都没动 |
| `test_openmm_fd_total_energy` | 同上 | 对 QM O 的 x 坐标和 MM H1 的 y 坐标，在 Context 总能量上做中心差分（h = 1e-5 nm）：与 Context 给出的力之差 < 1e-3 kJ/mol/nm |
| `test_unsorted_qm_atoms_map_forces_correctly`（Review Focus 2） | 3 个水，QM = 水 1，按 `[5, 3, 4]` 的顺序传入 | Context 中粒子 5、3、4 上的力分别等于 FakeBackend 返回的 QM 力第 0、1、2 行 |
| `test_step_counter` | 调用 3 次 `getState(getForces=True)` | `callback.step == 3`，且 `backend.last_request.step == 2` |
| `test_particle_count_mismatch` | 回调声明 6 个粒子，但 Context 中有 3 个 | 抛 `ValueError` |

**步骤：**
- [x] 写测试，确认 FAIL
- [x] 实现
- [x] 运行测试，全部 PASS

**完成标准：** 6 个用例通过。

---

### Task 9：ORCAPotential 用户入口

**文件：**
- 新建：`openmmorca/potential.py`
- 修改：`openmmorca/__init__.py`（导出 `ORCAPotential`）
- 测试：`tests/test_potential_fake.py`

**接口（产出）：**
```text
class ORCAPotential:
    def __init__(self, method: str, basis: str | None = None, charge: int = 0, multiplicity: int = 1,
                 nprocs: int = 1, maxcore_mb: int = 2000,
                 extra_keywords: Sequence[str] = (), extra_blocks: Sequence[str] = (),
                 scratch_root: str | None = None, restart: bool = True, keep_failed: bool = True,
                 max_fresh_retries: int = 1, timeout_s: float | None = None, orca_path: str | None = None,
                 backend_factory: Callable[[], QMBackend] | None = None)
    config: ORCAConfig                   # 构造时就生成，因此配置错误在这里立即报出
    backends: list[QMBackend]            # 本对象创建过的所有后端
    def createSystem(self, topology, removeCMMotion: bool = True) -> openmm.System
    def createMixedSystem(self, topology, system, atoms, removeConstraints: bool = True, forceGroup: int = 0,
                          interpolate: bool = False, embedding: str = "electronic") -> openmm.System
    def getSupportedEmbeddings(self) -> list[str]        # ["electronic"]
    def summarize_timings(self) -> list[dict]            # Task 15 实现；此处先返回空列表
    def close(self) -> None                              # 关闭所有后端
```

**要求：**
- `backend_factory` 为 None 时，默认工厂是 `lambda: ORCAOPIBackend(self.config)`。**每次** `createSystem` / `createMixedSystem` 都调用一次工厂（Review Focus 5）。
- `createSystem`：粒子质量取自 topology 的元素；加 PythonForce；`removeCMMotion` 为 True 时加 `CMMotionRemover`。
- `createMixedSystem`：`interpolate=True` 抛 `NotImplementedError`；`embedding != "electronic"` 抛 `ValueError`；其余交给 `build_mixed_system` 和 `QMMMCallback`。
- `ORCAPotential` 自身不含任何 ORCA 相关逻辑，只负责拼装。

**测试用例（用 `backend_factory` 注入 FakeBackend）：**

| 用例 | 期望 |
|---|---|
| `test_create_system_full_qm` | 1 个水：System 有 3 个粒子，质量正确，包含 PythonForce 和 CMMotionRemover；Context 能量等于 FakeBackend 的能量 |
| `test_create_mixed_system` | 2 个水：结果与 Task 8 的 `test_mixed_energy_decomposition` 一致 |
| `test_each_system_gets_its_own_backend`（Review Focus 5） | 同一个 potential 调两次 `createMixedSystem`：`len(backends) == 2`，两个后端是不同对象 |
| `test_interpolate_rejected` / `test_embedding_rejected` | 分别抛 `NotImplementedError` / `ValueError` |
| `test_config_errors_raised_at_construction` | `ORCAPotential("HF", extra_keywords=("Opt",))` 在构造时就抛 `ValueError` |
| `test_close_closes_all_backends` | 用一个记录 close 调用的假后端：`close()` 后每个后端都被关闭 |

**步骤：**
- [x] 写测试，确认 FAIL
- [x] 实现 `potential.py`，导出 `ORCAPotential`
- [x] 运行完整的 `pytest`（不带 `-m orca`），全部 PASS

**完成标准：** 7 个用例通过。**M1 完成。**

---

# M2：非周期电子嵌入（v0.1）

### Task 10：ORCA 端到端集成

**文件：**
- 测试：`tests/test_potential_orca.py`（全部 `@pytest.mark.orca`）
- 新建：`examples/full_qm_water.py`

**要求：** 这一步只接上真实后端并验证，原则上不改代码；发现 bug 就回到对应任务修正并补测试。

**测试用例：** 方法用 `HF/def2-SVP` 加 `TightSCF`，Reference 平台。

| 用例 | 体系 | 期望 |
|---|---|---|
| `test_full_qm_water_openmm_equals_backend` | 1 个水，`createSystem` | Context 能量和力 == 直接调用 `ORCAOPIBackend.evaluate` 的结果（能量容差 1e-6 kJ/mol，力容差 1e-4 kJ/mol/nm） |
| `test_qmmm_dimer_decomposition` | 2 个 flexible tip3p 水，QM = 水 0，PythonForce 放在 force group 1 | ① group 1 的能量 == 直接后端调用的能量（请求中的点电荷为水 1 的 3 个 tip3p 电荷）；② 其余 group 的能量 == 原 System 中"水 1 的键合能 + QM–MM LJ"（水 0 放在平衡几何，使其 MM 键合能为 0） |
| `test_translation_invariance` | 同上，整体平移 (1, −2, 0.5) nm | 能量变化 < 1e-7 Eh（换算为 kJ/mol 后比较）；‖Σ F‖ < 1e-3 × max‖F_i‖ |
| `test_openmm_fd_mm_atom` | 同上 | 在 Context 总能量上，对 MM 水 O 的 x 坐标做中心差分（h = 1e-4 nm）：与力之差 < 0.5 kJ/mol/nm（约等于 1e-5 Eh/bohr） |
| `test_tip4p_virtual_site_embedding`（Review Focus 3） | TIP4P-Ew，2 个水，QM = 水 0 | ① 点电荷文件中有 3 个电荷（2 个 H + M 位点，没有 O）；② 在 Context 总能量上对 MM 水 O 的 x 坐标做中心差分，与 O 上的力之差 < 0.5 kJ/mol/nm（证明 M 位点上的力被正确分配到了母原子） |

`examples/full_qm_water.py`：1 个水，full-QM，HF/def2-SVP，Verlet 积分 0.5 fs，跑 20 步，打印每步的势能、动能、总能量，最后打印 `potential.backends[0].scratch.root`。

**步骤：**
- [x] 写测试，运行 `pytest tests/test_potential_orca.py -v -m orca`
- [x] 失败时按"先用 fake 后端复现"的原则定位：fake 能复现就是 OpenMM 层的问题，不能复现就是 ORCA 后端的问题
- [x] 运行示例，确认总能量没有明显漂移

**完成标准：** 5 个用例通过；示例可以运行。

---

### Task 11：NVE 验证

**文件：**
- 测试：`tests/test_nve.py`（`@pytest.mark.orca` + `@pytest.mark.slow`）
- 新建：`examples/qmmm_water_cluster_nve.py`

**要求：**
- 体系：QM = 1 个水，MM = 4 个 flexible tip3p 水，构成一个紧凑团簇（水分子中心间距约 0.3 nm）；`HF/STO-3G` 加 `TightSCF`（为了速度；如果每步耗时 > 1 s，改用 `HF-3c`，并把选择记录在测试注释中）。
- 先用 OpenMM `LocalEnergyMinimizer` 优化，再按 300 K 分配速度，然后 `VerletIntegrator(0.25 fs)`，不加恒温器，跑 4000 步（1 ps），每 10 步记录一次总能量。
- 漂移用总能量对时间做线性拟合的斜率表示，单位 kJ/mol/ps。
- 示例脚本做同样的事，把每步能量写入 CSV，并在结束时打印漂移和 `timings`。

**测试用例：**

| 用例 | 期望 |
|---|---|
| `test_nve_drift` | 漂移绝对值 < 0.1 kJ/mol/ps（spec §1.3 的目标值）；总能量的标准差 < 0.5 kJ/mol |

**步骤：**
- [x] 先运行示例脚本，获得实测的漂移和标准差
- [x] 如果实测值远小于目标，就把阈值收紧到实测值的 3 倍；如果达不到目标，**不要放宽阈值**，先用 fake 后端跑同样的 NVE 排除 OpenMM 层问题，再检查 SCF 收敛阈值，并与负责人讨论
- [x] 把最终阈值写回 spec §1.3（去掉"目标值"字样）
- [x] 运行 `pytest tests/test_nve.py -v -m slow`

**完成标准：** 测试通过，spec §1.3 已更新。**M2 完成，发布 v0.1。**

---

# M3：restart 与健壮性（v0.2）

### Task 12：restart 状态与诊断模块

**文件：**
- 新建：`openmmorca/runtime/restart.py`、`openmmorca/runtime/diagnostics.py`
- 测试：`tests/test_runtime.py`（不需要 ORCA）

**接口（产出）：**
```text
# runtime/restart.py
GUESS_NAME = "guess.gbw"
class RestartState:
    def __init__(self, restart_dir: Path, enabled: bool = True)
    last_good: Path                                   # restart_dir / "last_good.gbw"
    has_guess: bool                                   # property：enabled 且 last_good 存在
    def stage_guess(self, current_dir: Path) -> Path | None   # 复制为 current/guess.gbw 并返回路径；无初猜时返回 None
    def commit(self, gbw_path: Path) -> None          # 先写 .tmp 再 os.replace，原子替换；enabled=False 时什么都不做
    def invalidate(self) -> None

# runtime/diagnostics.py
TIMING_FIELDS = ("step", "t_write", "t_orca", "t_read", "t_total", "scf_cycles", "restart_used", "fresh_retries")
BUNDLE_FILES = ("qm.inp", "qm.out", "qm.err", "pc.pc", "qm.engrad", "qm.pcgrad", "guess.gbw")
class TimingLog:
    def __init__(self, path: Path)
    rows: list[dict]
    def append(self, row: dict) -> None               # 缺字段抛 ValueError；首次写入时写表头
    def summarize(self, skip_first: bool = True) -> dict[str, dict[str, float]]
        # 对 t_write/t_orca/t_read/t_total 各给出 {"mean", "p50", "p95", "n"}；默认跳过第一行（冷启动）
def write_failure_bundle(failures_dir: Path, step: int, current_dir: Path, qm_elements, qm_positions_ang,
                         metadata: dict) -> Path
    # 目录名 failure_step_{step:06d}；复制 BUNDLE_FILES 中存在的文件；写 geometry.xyz 和 metadata.json
```

**测试用例：**

| 用例 | 期望 |
|---|---|
| `test_restart_no_guess_initially` | 新建时 `has_guess` 为 False，`stage_guess` 返回 None |
| `test_restart_commit_and_stage` | commit 一个内容为 b"abc" 的假 gbw 后，`has_guess` 为 True；`stage_guess` 生成的 `current/guess.gbw` 内容为 b"abc" |
| `test_restart_commit_is_atomic` | commit 后 restart 目录中没有残留的 `.tmp` 文件 |
| `test_restart_disabled` | `enabled=False`：commit 之后 `has_guess` 仍为 False |
| `test_restart_invalidate` | invalidate 后 `has_guess` 为 False |
| `test_timing_log_csv` | append 2 行后，CSV 有表头和 2 行数据；列顺序与 `TIMING_FIELDS` 相同 |
| `test_timing_log_missing_field` | 缺少 `t_orca` 字段 → `ValueError` |
| `test_timing_summary_skips_cold_start` | `t_total` 依次为 [10, 1, 1, 1] → summary 中 mean 为 1，n 为 3 |
| `test_failure_bundle_contents` | current 下有 qm.inp、qm.out、pc.pc：bundle 中有这三个文件，外加 geometry.xyz（第一行是原子数）和 metadata.json（可被 json 解析，含传入的字段） |
| `test_failure_bundle_overwrites_same_step` | 同一个 step 写两次 bundle，不报错，且内容为第二次的 |

**步骤：**
- [x] 写测试，确认 FAIL
- [x] 实现
- [x] 运行测试，全部 PASS

**完成标准：** 10 个用例通过。

---

### Task 13：后端接入 restart 状态机与失败包

**文件：**
- 修改：`openmmorca/backend/orca_opi.py`
- 测试：`tests/test_orca_restart.py`

**接口变化：**
```text
class ORCAOPIBackend:
    restart_state: RestartState
    timings: TimingLog                      # 写到 scratch.root / "timings.csv"
    n_fresh_retries: int                    # 累计的"MORead 失败、全新 SCF 重试成功"次数
    def _run_step(self, request: QMRequest, use_guess: bool) -> _StepOutput
        # 单次 ORCA 运行：清空 current → 写输入（use_guess 时加 MORead 和 %moinp guess.gbw）
        # → 运行 → 校验 → 读取。失败时抛异常，不做重试。
        # 设计成独立方法，是为了让单元测试可以 monkeypatch 它。
    # _StepOutput：内部数据类，含 energy_eh、qm_gradient、pc_gradient、t_write、t_orca、t_read、scf_cycles
```

**要求：**
- `evaluate` 实现 spec §9.2 的状态机：
  - `restart_state.has_guess` 为 False（第一步或 restart 被禁用）：只尝试一次全新 SCF。
  - 为 True：先用 MORead 尝试；如果抛 `ORCACalculationError`，再做最多 `max_fresh_retries` 次全新 SCF；重试成功时 `n_fresh_retries += 1` 并发出 `RuntimeWarning`。
  - `ORCAOutputError` 和 `ORCATimeoutError` **不重试**。
  - 成功后调用 `restart_state.commit(current/qm.gbw)`，并往 timings 追加一行。
  - 最终失败：`keep_failed` 为 True 时调用 `write_failure_bundle`（metadata 至少包含 step、异常类型与信息、ORCAConfig 各字段、QM 元素、点电荷数），然后重新抛出原异常，并在异常信息末尾附上 bundle 路径。
- `scf_cycles`：从 `qm.out` 中用正则匹配 `SCF CONVERGED AFTER\s+(\d+)\s+CYCLES`；没匹配到时记为 −1。
- 重入保护：`evaluate` 用 `threading.Lock().acquire(blocking=False)`；拿不到锁时抛 `RuntimeError("ORCAOPIBackend.evaluate is not re-entrant; one backend serves one OpenMM Context")`。

**测试用例：**

| 用例 | 需要 ORCA | 期望 |
|---|---|---|
| `test_state_machine_first_step_fresh_only` | 否 | monkeypatch `_run_step` 使其总是抛 `ORCACalculationError`：第一步只被调用 1 次（`use_guess=False`），然后抛异常 |
| `test_state_machine_retry_after_moread_failure` | 否 | 先人为放一个 last_good；`_run_step` 在 `use_guess=True` 时抛异常、False 时成功：调用序列为 [True, False]，返回结果，`n_fresh_retries == 1`，发出 `RuntimeWarning` |
| `test_state_machine_both_fail` | 否 | 两种都失败：调用序列为 [True, False]，抛 `ORCACalculationError`，failures 下出现 `failure_step_000000` |
| `test_output_error_not_retried` | 否 | `_run_step` 抛 `ORCAOutputError`：只调用 1 次 |
| `test_reentrancy_rejected` | 否 | 在 `_run_step` 内部再次调用 `evaluate`：内层抛 `RuntimeError` |
| `test_second_step_uses_moread` | 是 | H2O 连续 3 步，每步把 H1 沿 x 方向移动 0.002 Å：第 1 行 timings 的 `restart_used` 为 False，第 2、3 行为 True；第 2、3 步的 `scf_cycles` < 第 1 步 |
| `test_restart_energy_matches_fresh` | 是 | 同一几何，restart 开和关各算一次：能量差 < 1e-6 Eh |
| `test_forced_scf_failure_bundle` | 是 | `extra_blocks=("%scf maxiter 2 end",)`：抛 `ORCACalculationError`；bundle 中有 qm.inp 和 qm.out，且 qm.out 中含 "SCF NOT CONVERGED" |
| `test_failed_run_does_not_clobber_last_good` | 是 | 先成功一步，记录 last_good 的字节数；再让下一步失败：last_good 的内容不变 |

**步骤：**
- [x] 写不需要 ORCA 的用例，确认 FAIL；实现状态机；确认 PASS
- [x] 写需要 ORCA 的用例，确认 PASS
- [x] 重新运行 Task 4、5、10 的全部测试，无回归

**完成标准：** 9 个用例通过，已有测试无回归。

---

### Task 14：MPI 并行与 nprocs 基准

**文件：**
- 修改：`openmmorca/backend/orca_opi.py`
- 测试：`tests/test_orca_mpi.py`（`@pytest.mark.orca`）
- 新建：`examples/bench_nprocs.py`

**要求：**
- 每次运行 ORCA 时，用一个上下文管理器临时设置环境变量，运行结束（包括异常）后恢复原值（spec §8.6）：
  - 总是设置 `OMP_NUM_THREADS=1`
  - `nprocs > 1` 时再设置 `OMPI_MCA_pml=ob1`、`OMPI_MCA_btl=self,sm`、`OMPI_MCA_mtl=^ofi`、`OMPI_MCA_osc=^ucx`
  - 用户已经在环境中设置过的同名变量不覆盖（便于在集群上自定义）
- 构造时检查：`nprocs` 大于物理核数（可用 `os.cpu_count() // 2` 近似，并在注释中说明本机有超线程）时发出警告；`maxcore_mb × nprocs` 超过物理内存的 75%（读 `/proc/meminfo`）时发出警告。
- `examples/bench_nprocs.py`：命令行参数为 xyz 文件、方法字符串和 nprocs 列表（默认 `1,2,4,8,16,32,40`）；对每个 nprocs 先预热一次，再计时 3 次，输出中位数墙钟时间的表格。附带一个 30–50 原子的示例 xyz，放在 `examples/data/`。

**测试用例：**

| 用例 | 期望 |
|---|---|
| `test_nprocs2_matches_serial` | H2O 加 2 个点电荷，`nprocs=2` 与 `nprocs=1` 的能量差 < 1e-8 Eh，梯度差 < 1e-6 Eh/bohr |
| `test_mpi_env_restored` | 运行结束后，`os.environ` 中没有新增的 `OMPI_MCA_*` 变量 |
| `test_user_mpi_env_not_overridden` | 预先设置 `OMPI_MCA_btl=self,tcp`：monkeypatch runner，在其内部读取到的值仍为 `self,tcp` |
| `test_nprocs2_is_not_pathologically_slow` | `nprocs=2` 一次 H2O 计算的墙钟时间 < 5 s（用来抓住 MCA 设置失效、退回约 10 s 的情况） |

**步骤：**
- [x] 写测试，确认 FAIL；实现；确认 PASS
- [x] 运行 `bench_nprocs.py`，结果表格贴进 spec §2.2.1（本机只测到 nprocs=32，原因见里程碑表下的说明）

**完成标准：** 4 个用例通过；spec §2.2 中有 30–50 原子体系的 nprocs 实测表格。

---

### Task 15：restart 重现性、计时汇总与 README

**文件：**
- 修改：`openmmorca/potential.py`（实现 `summarize_timings`）
- 测试：`tests/test_restart_reproducibility.py`（`@pytest.mark.orca` + `@pytest.mark.slow`）
- 新建：`README.md`

**要求：**
- `summarize_timings()` 返回列表，每个后端一项：`{"scratch": str(root), "n_fresh_retries": int, **timings.summarize()}`。
- README 内容：安装方法（环境、`OPI_ORCA`）；full-QM 与 QM/MM 的最小示例（直接引用 examples 目录下的文件）；参数表（与 `ORCAPotential` 的签名一致）；已知限制（非周期、整分子 QM、不能序列化 System、每步有固定开销）；scratch 目录布局与失败包用法；如何运行测试（默认、`-m orca`、`-m slow`）。

**测试用例：**

| 用例 | 期望 |
|---|---|
| `test_restart_trajectory_reproducible` | Task 11 的团簇，restart 开和关各跑 50 步 NVE（相同初速度）：逐步势能差 < 1e-6 Eh |
| `test_summarize_timings`（不需要 ORCA，用 fake 后端配合手工写入的 TimingLog） | 返回结构中含 `t_total` 的 mean、p50、p95 |

**步骤：**
- [x] 写测试；实现 `summarize_timings`
- [x] 运行完整测试 `pytest -m "orca or not orca"` 以及 `pytest -m slow`，全部 PASS
- [x] 写 README

**完成标准：** 用例通过；README 中的示例可以原样运行。**M3 完成，发布 v0.2。**

---

# M4：link atom（v0.3）

> **开始前必须与应用负责人确定**：目标酶体系，以及其中的 QM/MM 边界类型（spec §15 第 7 项）。本里程碑只支持单键 C–C（或 C–N）边界的 H link atom，边界电荷只用 charge-shift。

### Task 16：link atom 几何与力回分

**文件：**
- 新建：`openmmorca/qmmm/linkatoms.py`
- 修改：`openmmorca/qmmm/__init__.py`、`openmmorca/qmmm/system.py`（`check_whole_molecules` 放行用户声明的边界键）、`openmmorca/force.py`、`openmmorca/potential.py`
- 测试：`tests/test_linkatoms.py`

**接口：**
```text
@dataclass(frozen=True)
class BoundaryPair:
    q1: int                 # QM 边界原子的 OpenMM 索引
    m1: int                 # MM 边界原子的 OpenMM 索引
    g: float                # R_L = R_Q1 + g (R_M1 - R_Q1)

default_link_ratio(element_q1: str, element_m1: str) -> float
    # C–C 返回 1.09/1.526；C–N 返回 1.09/1.449；其他组合抛 ValueError，要求用户显式给出 g

check_boundary_pairs(topology, qm_atoms, pairs) -> None
    # q1 在 QM 区、m1 不在；二者在 topology 中成键；每个 q1 最多出现一次；
    # 除声明的边界键外，没有其他跨 QM/MM 的键

class LinkAtomManager:
    def __init__(self, pairs: Sequence[BoundaryPair])
    n_links: int
    def link_positions(self, positions_nm: np.ndarray) -> np.ndarray          # (n_links, 3)
    def redistribute(self, forces: np.ndarray, link_forces: np.ndarray) -> None
        # 原地修改：forces[q1] += (1 - g) F_L；forces[m1] += g F_L

# 接口扩展
ORCAPotential.createMixedSystem(..., boundaryPairs: Sequence[tuple[int, int]] | None = None,
                                linkRatios: Sequence[float] | None = None)
QMMMCallback.__init__(..., link_manager: LinkAtomManager | None = None)
    # QM 请求中的元素 = QM 元素 + n_links 个 "H"；坐标 = QM 坐标 + link 坐标
```

**要求：**
- 键合项规则保持 spec §6.3（所有原子都在 QM 区才置零）。这个任务要用测试确认 Q1–M1 键、跨边界的角和二面角都被保留。
- NonbondedForce 的处理不变（spec §10.5）：Q1 电荷置零，涉及 QM 原子的 exception 的 chargeProd 置零，LJ 保留。
- 用户给出的 `charge` 是"QM 原子 + link H"的总电荷。电子数检查（Task 4）要把 link H 算进去。

**测试用例：**

| 用例 | 期望 |
|---|---|
| `test_link_position_formula` | Q1 = (0,0,0)，M1 = (0.15,0,0) nm，g = 0.7 → L = (0.105,0,0) |
| `test_redistribute_chain_rule` | F_L = (1,2,3)，g = 0.7 → Q1 增加 (0.3,0.6,0.9)，M1 增加 (0.7,1.4,2.1) |
| `test_fd_with_link_atoms_fake_backend` | 丙醇（或 ACE-ALA-NME 的一段），切在一个 C–C 键上，用 FakeBackend（假电荷数 = QM 原子数 + 1）：对 Q1、M1 的全部 6 个坐标在 Context 总能量上做中心差分，误差 < 1e-3 kJ/mol/nm |
| `test_boundary_bonded_terms_kept` | 同上体系：Q1–M1 键、Q2–Q1–M1 角、跨边界二面角的力常数不为 0；所有原子都在 QM 区的项为 0 |
| `test_boundary_validation` | q1 不在 QM 区 / q1 与 m1 不成键 / 同一个 q1 出现两次 / 存在未声明的跨界键 → 全部 `ValueError` |
| `test_default_ratio_unknown_pair` | O–S 组合 → `ValueError` |

**完成标准：** 6 个用例通过。

---

### Task 17：charge-shift 边界电荷

**文件：**
- 新建：`openmmorca/qmmm/charges.py`
- 修改：`openmmorca/qmmm/__init__.py`（`build_mixed_system` 接收 `boundary_pairs`，输出的 `mm_charges_e` 使用 charge-shift 之后的电荷）
- 测试：`tests/test_charges.py`

**接口：**
```text
class ChargeShift:
    def embedding_charges(self, topology, mm_atoms: Sequence[int], mm_charges_e: np.ndarray,
                          boundary_pairs: Sequence[BoundaryPair]) -> tuple[tuple[int, ...], np.ndarray]
        # 返回新的 (mm_atoms, charges)：去掉 M1；M1 的原电荷平均加到与 M1 成键的 MM 原子（M2）上
```

**要求：**
- 只修改传给 ORCA 的嵌入电荷。OpenMM 中 MM–MM 静电不变（spec §10.4）。
- M1 没有 MM 邻居时抛 `ValueError`。
- 修改后如果某个 M2 的电荷变为 0，它仍然保留在嵌入中（不要误删）。

**测试用例：**

| 用例 | 期望 |
|---|---|
| `test_total_embedding_charge_conserved` | 嵌入电荷总和不变 |
| `test_m1_removed_and_m2_shifted` | M1 不在返回的 mm_atoms 中；每个 M2 增加 q_M1 / n_M2 |
| `test_m1_without_mm_neighbours` | 构造一个 M1 没有 MM 邻居的体系 → `ValueError` |
| `test_openmm_mm_electrostatics_unchanged` | 改造后 System 的 MM–MM 静电能量与原 System 相同（QM 区平移到远处后比较） |

**完成标准：** 4 个用例通过。

---

### Task 18：link atom 的 ORCA 端到端验证

**文件：**
- 测试：`tests/test_linkatoms_orca.py`（`@pytest.mark.orca`）
- 新建：`examples/link_atom_dipeptide.py`

**要求：**
- 体系：ACE-ALA-NME 二肽（OpenMM 自带的 `amber14-all.xml` 就有模板，不需要额外工具），真空，NoCutoff。QM = ALA 的 CA、HA、CB、HB1、HB2、HB3；边界为 CA–N 和 CA–C，共 2 个 link atom。QM 电荷为 0，多重度 1。方法 `HF/def2-SVP` 加 `TightSCF`。
- 示例脚本：能量最小化后跑 200 步 NVT（Langevin，300 K，0.5 fs），打印每步耗时并检查 QM 区结构没有散架（CA–CB 距离保持在 0.14–0.17 nm）。

**测试用例：**

| 用例 | 期望 |
|---|---|
| `test_link_fd_boundary_atoms` | 对 CA、N、C 三个原子的全部 9 个坐标，在 Context 总能量上做中心差分（h = 1e-4 nm）：误差 < 0.5 kJ/mol/nm |
| `test_link_translation_invariance` | 整体平移后能量变化 < 1e-7 Eh；‖Σ F‖ < 1e-3 × max‖F_i‖ |

**完成标准：** 2 个用例通过，示例可以运行。**M4 完成，发布 v0.3。**

---

# M5：周期性 MM + 截断嵌入（v0.4）

> 本里程碑引入的截断嵌入是**近似**（spec §11）。所有文档、docstring 和日志都必须写明这一点，不得宣称与 PME 兼容。

### Task 19：QM 区跨周期边界时拼回完整分子

**文件：**
- 新建：`openmmorca/qmmm/imaging.py`
- 测试：`tests/test_imaging.py`

**接口：**
```text
qm_bond_graph(topology, qm_atoms) -> dict[int, list[int]]        # 只包含 QM 内部的键
make_qm_whole(positions_nm: np.ndarray, qm_atoms: Sequence[int], graph: dict[int, list[int]],
              box_vectors_nm: np.ndarray) -> np.ndarray
    # 返回 (n_qm, 3)：以 qm_atoms[0] 为锚点做 BFS，把每个原子平移到离已放置的邻居最近的镜像；
    # 支持三斜盒（使用 OpenMM 约定的约化盒向量）
```

**测试用例：**

| 用例 | 期望 |
|---|---|
| `test_whole_molecule_unchanged` | 分子本来就完整时，输出与输入相同 |
| `test_split_across_x_boundary` | 盒长 3 nm 的正交盒，水分子的一个 H 被包裹到盒的另一侧 → 拼回后键长恢复为 0.09572 nm |
| `test_triclinic_box` | 截角八面体盒中跨边界的分子 → 键长恢复 |
| `test_disconnected_qm_region` | QM 区由两个互不成键的分子组成：每个连通分量各自以其第一个原子为锚点；两个分量之间取最近镜像（相对于第一个分量的锚点） |

**完成标准：** 4 个用例通过。

---

### Task 20：截断嵌入的电荷选择

**文件：**
- 新建：`openmmorca/qmmm/embedding.py`
- 测试：`tests/test_embedding.py`

**接口：**
```text
class CutoffEmbedding:
    def __init__(self, groups: Sequence[tuple[int, ...]], charges_e: np.ndarray, cutoff_nm: float = 1.2)
        # groups：每个残基中带电 MM 原子的索引；charges_e：按 OpenMM 粒子索引排列的全体电荷
    def select(self, qm_positions_nm: np.ndarray, positions_nm: np.ndarray, box_vectors_nm: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]
        # 返回 (选中的 OpenMM 索引, 最小镜像后的坐标, 电荷)。
        # 某个组中任一原子与任一 QM 原子的最小镜像距离 < cutoff，整组纳入；
        # 组内所有原子使用同一个平移矢量，保证组不被拆开
    last_n_groups: int                  # 最近一次选中的组数
    last_changed: int                   # 与上一步相比，进入和离开的组数之和
groups_from_topology(topology, mm_atoms) -> list[tuple[int, ...]]   # 默认按残基分组
```

**测试用例：**

| 用例 | 期望 |
|---|---|
| `test_group_inclusion_all_or_nothing` | 一个水分子只有 H 在截断内 → 3 个原子都被选中 |
| `test_minimum_image_positions` | QM 在盒的左边缘、MM 水在右边缘 → 返回的坐标是左侧的镜像，组内三个原子的相对几何不变 |
| `test_cutoff_excludes_far_groups` | 距离 2 nm 的组不被选中 |
| `test_change_counter` | 连续两次 select 之间把一个组移出截断 → `last_changed == 1` |
| `test_cutoff_larger_than_half_box_rejected` | cutoff ≥ 最短盒长的一半 → `ValueError` |

**完成标准：** 5 个用例通过。

---

### Task 21：周期性回调与 API 接入

**文件：**
- 修改：`openmmorca/force.py`、`openmmorca/potential.py`、`openmmorca/qmmm/__init__.py`（去掉 Task 7 中的 `NotImplementedError`）、`openmmorca/runtime/diagnostics.py`（给 TIMING_FIELDS 增加 `n_embed_groups`、`embed_changed`）
- 测试：`tests/test_periodic_fake.py`

**接口扩展：**
```text
ORCAPotential.createMixedSystem(..., embeddingCutoff: openmm.unit.Quantity = 1.2 * nanometer)
QMMMCallback.__init__(..., embedding: CutoffEmbedding | None = None, qm_graph: dict | None = None)
    # 有 embedding 时：PythonForce.setUsesPeriodicBoundaryConditions(True)；
    # 从 state.getPeriodicBoxVectors() 读盒向量 → make_qm_whole → embedding.select → 构造请求；
    # 力按原始 OpenMM 索引回填（平移不改变力）
```

**要求：**
- 周期体系必须使用 PME 或 Ewald 作为 NonbondedForce 的方法；`CutoffPeriodic` 抛 `ValueError`（反应场与嵌入的定义不一致）。
- 每步通过 timings 记录嵌入组数和进出组数。如果 `embed_changed > 0`，以 DEBUG 级别写日志。
- `createMixedSystem` 的 docstring 必须写明："QM–MM electrostatics beyond embeddingCutoff are neglected; this is not PME-consistent."

**测试用例（FakeBackend，Reference 平台）：**

| 用例 | 期望 |
|---|---|
| `test_periodic_forces_consistent_without_crossings` | 边长 2.5 nm 的水盒，QM = 1 个水，cutoff 1.0 nm；在一个没有组跨越截断的小位移下，对 QM O 的坐标在总能量上做中心差分：误差 < 1e-2 kJ/mol/nm |
| `test_qm_split_across_boundary` | 把 QM 水放在盒的角上，使它被 OpenMM 包裹拆开：FakeBackend 收到的是完整分子（键长正确） |
| `test_embedding_counts_logged` | timings 中有 `n_embed_groups` 列，且数值 > 0 |
| `test_cutoff_periodic_rejected` | NonbondedForce 使用 `CutoffPeriodic` → `ValueError` |

**完成标准：** 4 个用例通过。

---

### Task 22：酶体系应用与基准

**文件：**
- 新建：`examples/enzyme_qmmm.py`、`examples/data/<酶体系文件>`
- 修改：spec §2.2（追加实测耗时）、spec §15

**要求：**
- 使用 M4 开始前确定的酶体系：溶剂化、PBC、PME、link atom、截断嵌入。方法由应用负责人参照 Task 14 的基准结果选定。
- 流程：MM 平衡（由应用方提供，或在脚本中用纯 MM 完成）→ QM/MM 能量最小化 → 1 ps NVT（Langevin，300 K，0.5 fs）。
- 输出：每步耗时拆分（write/orca/read）、嵌入组数、`n_fresh_retries`、每 10 步一帧 DCD 轨迹。
- 验收：跑满 1 ps 无人工干预；最后 500 步的温度均值在 300 ± 10 K；QM 区中每个共价键长偏离初始值不超过 20%。

**步骤：**
- [ ] 与应用方确认体系文件和 QM 区选择，写入 `examples/enzyme_qmmm.py` 开头的注释
- [ ] 先用 xTB 跑 200 步冒烟，确认流程能走通，再切换到正式方法
- [ ] 跑满 1 ps，把耗时统计贴进 spec §2.2，把体系描述写进 spec §15

**完成标准：** 满足上述验收条件。**M5 完成，发布 v0.4。**

---

## 附录 C（可选）：与 PySCF 交叉验证

对应 spec §13.4。不属于任何里程碑的验收条件，建议在 M2 之后做一次：

- [ ] 新建一个独立环境（不要装进 `openmm_dev`），安装 `pyscf`
- [ ] 用 Task 3 的 H2O + 2 个点电荷几何，在 PySCF 中计算 HF/def2-SVP 的能量、QM 梯度和点电荷梯度（`pyscf.qmmm.mm_charge`，其 `grad_hcore_mm + grad_nuc_mm` 即为点电荷梯度，参见 `/home/ruigengji/openmm-pyscf/openmmpyscf/pyscfforce.py` 第 547–552 行）
- [ ] 与 Task 3 的 fixture 比较：能量差应在 1e-6 Eh 量级，梯度差在 1e-5 Eh/bohr 量级。把结果写进 spec §13.4

## 附录 A：常用命令

```bash
# 环境
PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
export OPI_ORCA=/home/ruigengji/ORCA611

# 安装
$PY -m pip install -e /home/ruigengji/openmm-ORCA

# 测试
$PY -m pytest                      # 默认：不含 slow；没有 ORCA 时 orca 测试会被跳过
$PY -m pytest -m orca -v           # 只跑需要 ORCA 的测试
$PY -m pytest -m slow -v           # NVE、重现性等长测试

# 手工运行一个 ORCA 输入（排查失败包时用）
cd <scratch>/failures/failure_step_000123 && $OPI_ORCA/orca qm.inp > rerun.out
```

## 附录 B：排错顺序

1. 先看 spec §2（已验证的事实与坑）。
2. OpenMM 层可疑 → 把后端换成 FakeBackend 复现。
3. ORCA 后端可疑 → 用失败包里的 `qm.inp` 手工运行 ORCA。
4. 数值可疑 → 有限差分（Task 5 / Task 8 中的工具函数）。**不要为了让测试通过而放宽阈值。**
