# OpenMM–ORCA/OPI QM/MM 接口：设计规格（Design Spec）

> 文档性质：**设计规格**，回答"做什么、为什么这样做"。
> 具体执行顺序、改哪些文件、每步怎么测，见实施计划 `docs/plans/2026-09-26-openmm-orca-implementation-plan.md`。
> 原始讨论稿备份：`openmm_orca_opi_design_plan.md.orig`。
> 最后更新：2026-09-27。

---

## 0. 读者须知

### 0.1 一句话目标

做一个 **OpenMM 驱动、ORCA 计算 QM 区** 的 QM/MM 力后端：OpenMM 负责 MD、MM 力场、积分器、约束、温压耦合和轨迹；ORCA 只负责电子结构和 QM–MM 静电嵌入；OPI（ORCA Python Interface）负责 ORCA 输入对象、执行与输出检查。

### 0.2 应用定位

- **第一目标（C）**：方法学验证。气相或水团簇、整分子 QM、非周期、无 link atom。证明"能量—QM 力—MM 反作用力"的数学闭环正确。
- **最终应用（B）**：酶活性位点 QM/MM。需要共价边界（link atom），且体系在水盒子里跑周期性边界（PBC）。

路线：C 做扎实 → 健壮性 → link atom → 周期性 MM + 截断嵌入 → 酶体系应用。

### 0.3 术语

| 术语 | 含义 |
|---|---|
| QM 区 / QM 原子 | 由 ORCA 计算的原子集合（用户给出的 OpenMM 原子索引） |
| MM 原子 | 其余原子，由 OpenMM 力场描述 |
| 嵌入电荷（embedding charges） | 传给 ORCA 的 MM 点电荷，极化 QM 电子密度 |
| pcgrad | ORCA 输出的点电荷梯度 ∂E/∂R_pc（`<basename>.pcgrad`） |
| engrad | ORCA 输出的能量与 QM 原子核梯度（`<basename>.engrad`） |
| Q1 / M1 / M2 | 共价边界上的 QM 边界原子 / 与之成键的 MM 原子 / 与 M1 成键的其他 MM 原子 |
| link atom (L) | 为饱和 Q1 悬挂键而加入 QM 计算的 H 原子，不是 OpenMM 粒子 |
| backend | 把"坐标 + 点电荷 → 能量 + 力"封装起来的对象，本项目中即 ORCA/OPI 后端 |

---

## 1. 目标、非目标与成功标准

### 1.1 目标

1. 提供与 `openmm-ml`、`openmm-pyscf` 风格一致的用户 API：`ORCAPotential(...).createSystem(topology)` 与 `.createMixedSystem(topology, system, atoms, ...)`。
2. 电子嵌入（electrostatic embedding）QM/MM，QM 原子力与 MM 点电荷反作用力都由 ORCA 解析梯度给出。
3. ORCA 特有能力（RIJCOSX、复合方法 r2SCAN-3c 等、开壳层、broken-symmetry、ECP、GFN2-xTB）可以原生透传，而不是被压缩成 `method + basis`。
4. 运行时健壮：SCF 失败有限重试后硬失败，绝不静默返回旧力；失败现场可复现。
5. 支持共价 QM/MM 边界（H link atom）和周期性 MM 环境下的截断嵌入，用于酶体系。

### 1.2 非目标（至少到 v0.4 都不做）

- 不再造一个 ASE，也不经 ASE 中转。
- 不做严格的周期性 QM/MM 静电（PME 与 QM 耦合、Ewald 嵌入）。
- 不做自适应 QM 区、极化 MM、多时间步（MTS）、异步 QM。
- 不自动识别化学边界：link atom 的边界原子对由用户显式给出。
- 不支持一个后端实例服务多个 OpenMM Context。
- 不追求 ORCA 全部关键字的类型化封装；高级选项以原生字符串/OPI 对象透传。

### 1.3 成功标准（按阶段）

| 阶段 | 可验证的成功标准 |
|---|---|
| v0.1（C 完成） | 有限差分：QM 原子力与点电荷力分量误差 < 1e-4 Eh/bohr（HF/def2-SVP，TightSCF）；非周期体系总力 ‖ΣF‖ < 1e-3 × max‖F_i‖；QM 水 + MM 水团簇 NVE 1 ps（0.25 fs 步长，HF/STO-3G TightSCF）：总能量漂移 < 0.017 kJ/mol/ps，总能量标准差 < 0.05 kJ/mol（2026-09-27 实测 +0.0056 kJ/mol/ps、0.0163 kJ/mol，阈值取约 3 倍；原目标 0.1 / 0.5）。初速度必须去掉质心平动：`setVelocitiesToTemperature` 生成的速度带质心运动，`ForceField.createSystem` 默认加的 `CMMotionRemover` 会在第一步把这部分动能（本团簇约 10.7 kJ/mol）直接删掉，造成一个假的能量跳变，未处理时拟合出的漂移为 −0.153 kJ/mol/ps、标准差 0.53 kJ/mol |
| v0.2 | 连续 1000 步无人工干预；restart 与非 restart 轨迹前 50 步逐步能量差 < 1e-6 Eh（两条轨迹必须从同一个初始状态出发：各自做能量最小化时，SCF 初猜不同带来的约 1e-9 Eh 噪声会让 L-BFGS 走不同路径，实测第 0 步就差 9e-5 Eh；2026-09-27 通过）；故意制造 SCF 失败时正确重试/报错并生成失败包 |
| v0.3 | 带共价边界的小体系（ACE-ALA-NME 二肽，QM = ALA 甲基侧链，边界 CB–CA）有限差分通过，力正确回分到 Q1/M1（2026-09-29 通过：HF/def2-SVP，Q1/M1/M2 有限差分误差 ≤ 0.042 kJ/mol/nm） |
| v0.4 | 溶剂化酶体系（PBC + PME）在 NVT 下稳定运行 ≥ 1 ps，每步耗时分解有记录（2026-10-01 通过：DhlA 2DHC，r2SCAN-3c，1 ps，最后 500 步 298.6 ± 1.1 K，见 §2.2.2 与 §15） |

---

## 2. 已验证的环境与事实（2026-09-26 实测）

本节内容全部经过实测，是后续设计决定的依据。交接后如环境变化，请重新验证本节。

### 2.1 软件与硬件

| 项目 | 值 |
|---|---|
| ORCA | 6.1.1，路径 `/home/ruigengji/ORCA611`（二进制与 `lib/` 同目录），自带 `otool_xtb`、`orca_vpot` |
| Python 环境 | mamba env `openmm_dev`，解释器 `/home/ruigengji/miniforge3/envs/openmm_dev/bin/python` |
| OpenMM | 8.5.2（平台：Reference、CPU、CUDA、OpenCL） |
| OPI | 2.0.0（pip 包名 `orca-pi`，import 名 `opi`；要求 ORCA ≥ 6.1.1） |
| pytest | 9.1.1 |
| PySCF | **未安装**在 `openmm_dev`（交叉验证时需另装） |
| MPI | 节点相关！spec 验证节点有系统 OpenMPI 5.0.11（`/usr/bin/mpirun`）；**登录节点（yayoi）没有任何 mpirun**，并行必须设 `OPI_MPI=/home/apps/openmpi/5.0.7_gcc13.3.0`（集群 `orca/6.1.1` 模块依赖的版本，提供 `orca_startup_mpi` 需要的 `libmpi.so.40`）。设了 `OPI_MPI` 后 OPI Runner 会自动把 `<mpi>/bin` 与 `<mpi>/lib` 加进子进程 PATH/LD_LIBRARY_PATH，后端无需额外处理 |
| CPU | 2 × Xeon Gold 6138：40 物理核 / 80 逻辑线程 |
| 内存 | 93 GB |
| GPU | RTX 2080 Ti 11 GB |
| `/tmp` | tmpfs（47 GB） |

### 2.2 ORCA 单次调用耗时（H2O，能量 + 梯度，热缓存）

| 配置 | 墙钟时间 |
|---|---|
| 串行 HF/def2-SVP | ≈ 0.45 s |
| 串行 + `MORead` | ≈ 0.45 s |
| 串行 HF/STO-3G | ≈ 0.6 s（首轮，含缓存预热） |
| GFN2-xTB（`! XTB`） | ≈ 0.2 s |
| `%pal nprocs 4`，默认 MPI 环境 | ≈ 10 s |
| `%pal nprocs 4`，MPI 限定共享内存传输 | ≈ 1.8 s |
| 经 OPI `Calculator` 调用（串行 HF/def2-SVP + 点电荷） | ≈ 0.5–0.6 s，解析 property JSON ≈ 3 ms |

结论：

1. **串行的固定开销约 0.4 s**，对真实 QM 区（几十原子的 DFT，每步十几秒以上）可以忽略。**首次调用（冷缓存）会慢数倍，基准测试必须丢弃第一次。**
2. **MPI 默认设置下每个模块的初始化开销巨大。** 后端在 `nprocs > 1` 时必须设置：
   ```
   OMPI_MCA_pml=ob1
   OMPI_MCA_btl=self,sm
   OMPI_MCA_mtl=^ofi
   OMPI_MCA_osc=^ucx
   OMP_NUM_THREADS=1
   ```
   调优后仍有约 1.4 s 的 MPI 额外开销，所以 **QM 区小的时候串行更快**，`nprocs` 应根据 QM 区大小实测选择（见实施计划 M3 基准任务）。
3. **ORCA 没有常驻/服务模式。** 每次计算都由 `orca` 主程序拉起各模块子进程。OPI 的 `CalcServer`/`OpiServer` 是反方向的常驻（让 Python 计算器常驻、由 ORCA 通过 ExtOpt 调用），不适用于本项目。
4. **ORCA 安装目录不要放在 NFS 上跑 MD（2026-09-27 实测）。** 本机 `/home/ruigengji` 是 NFS 挂载，ORCA 每步都要从安装目录启动多个静态链接的大程序（每个约 31 MB）。其他任务压满 NFS 后，同一个 H2O HF/STO-3G 步从 0.4 s 涨到 3.5 s（进程卡在 D 状态），1 ps NVE 从约 30 分钟变成 2 小时以上。做法：把 ORCA 复制到本地盘或 tmpfs，并让 `OPI_ORCA` 指向副本。不需要 `autoci_*`（13.5 GB，耦合簇/CI 的自动生成代码），其余约 3.8 GB，例如 `/dev/shm/orca611-<user>`。Python 环境（`miniforge3`）同样在 NFS 上，NFS 拥堵时解释器启动加 import 要几分钟。

### 2.2.1 30–50 原子体系的 nprocs 实测（Task 14）

`examples/bench_nprocs.py examples/data/water12.xyz`（12 个水，36 原子），能量 + 梯度，`TightSCF`，`restart=False`（每次全新 SCF）；每个 nprocs 先预热一次，再取 3 次的中位数。ORCA 在 tmpfs 上；MPI 进程用 `PRTE_MCA_hwloc_default_cpu_list=4-35` 限定在物理核 4–35（见 §8.6），同时另有 2 个串行 ORCA 任务占用核 0–2。2026-09-27，2× Xeon Gold 6138。

| nprocs | HF/def2-SVP (s) | 加速比 | PBE0/def2-SVP RIJCOSX def2/J (s) | 加速比 |
|---|---|---|---|---|
| 1 | 59.6 | 1.0 | 98.9 | 1.0 |
| 2 | 35.4 | 1.7 | 53.1 | 1.9 |
| 4 | 20.2 | 3.0 | 33.2 | 3.0 |
| 8 | 12.9 | 4.6 | 20.3 | 4.9 |
| 16 | 9.7 | 6.2 | 15.3 | 6.5 |
| 32 | 7.5 | 8.0 | 11.5 | 8.6 |

结论：对 36 原子体系，8 核以内接近线性，16 核之后收益明显变小（16→32 核只快约 1.3 倍）。几十原子的 QM 区用 8–16 核比较划算，剩余的核可以同时跑别的任务。

### 2.2.2 酶体系 QM/MM 单步耗时（Task 22，2026-10-01）

DhlA 方案 A（15 个 QM 原子含 1 个 link H，1.2 nm 截断内约 1,800 个嵌入点电荷），每步 ORCA 时间：

| 方法 | 核数 | ORCA 每步 |
|---|---|---|
| GFN2-xTB | 1 | 0.21 s |
| HF-3c | 1 | 3.0 s |
| HF/def2-SVP | 1 | 15.4 s |
| r2SCAN-3c | 1 / 16 / 32 / 40 | 17.9 / 5.4 / 4.8 / 5.45 s |
| B3LYP-D3BJ/def2-SVP | 1 | 34.6 s |

结论：QM 区小而点电荷多时，开销主要在点电荷积分与 pcgrad，不在 QM 原子数；r2SCAN-3c 在 16 核以后几乎不再加速（40 核的 1 ps 正式跑实测 5.45 s，与 32 核的短测相当）。OpenMM 侧（CUDA 上的 MM、回调中的镜像与选组）每步约 0.1 s。

### 2.3 点电荷：必须用 `%pointcharges` 文件，不能用 inline `Q`

同一个 H2O + 2 个点电荷体系，两种写法对比：

| 写法 | 总能量 (Eh) | 是否输出 `.pcgrad` |
|---|---|---|
| `%pointcharges "pc.pc"` 外部文件 | −75.977160671564 | **是** |
| 坐标块中 inline `Q` 行（即 OPI `PointCharge` 对象的写法） | −76.172238644025 | **否** |

两者能量差 −0.195 Eh，正好等于两个点电荷之间的库仑能（−0.834 × 0.417 / 1.78 bohr）。即 **inline `Q` 把点电荷之间的相互作用也算进总能量**。MM–MM 静电已由 OpenMM 计算，所以 inline `Q` 会重复计算，而且拿不到 pcgrad。

**决定：嵌入电荷一律写 `%pointcharges` 外部文件；不使用 OPI 的 `PointCharge` 类。**

点电荷文件格式（已验证）：

```text
<N>
<q_1> <x_1> <y_1> <z_1>
...
```

电荷单位 e，坐标单位 Å。

### 2.4 pcgrad 的定义与单位（已用有限差分验证）

`.pcgrad` 格式：第一行 N，随后 N 行 `gx gy gz`，单位 Eh/bohr，顺序与点电荷文件一致。

有限差分验证（第 1 个点电荷的 x 方向，步长 ±0.001 Å）：

```text
FD  dE/dx_pc1 = 0.012533970765 Eh/bohr
pcgrad x_pc1  = 0.012533896858 Eh/bohr
```

误差 7×10⁻⁸，确认 **pcgrad = +∂E/∂R_pc**（梯度，不是力），E 为 `FINAL SINGLE POINT ENERGY`（含 QM 核—点电荷、QM 电子—点电荷相互作用，不含点电荷—点电荷相互作用）。

### 2.5 `.engrad` 格式

注释行以 `#` 开头；依次为：原子数、总能量 (Eh)、3N 行梯度（Eh/bohr，按 x1 y1 z1 x2 … 顺序）、N 行 `Z x y z`（bohr）。

### 2.6 ORCA 运行行为（实测）

- **ORCA 异常终止时进程返回码仍为 0**（实测 `error termination in GUESS` 时 rc=0）。**不能用返回码判断成功**，必须检查输出中的 `ORCA TERMINATED NORMALLY` 和 SCF 收敛标志。
- **`MORead` 的初猜文件不能与 basename 同名。** ORCA 启动时会先覆盖 `<basename>.gbw`，导致 `No orbitals were found in the gbw file`。初猜文件必须改名（本项目用 `guess.gbw`）。
- MORead 有效：H2O HF/def2-SVP 从上一个几何的 `.gbw` 读初猜，SCF 从 11 圈降到 2 圈。
- 失败的运行也可能覆盖或清空 `.gbw`，所以 last-good `.gbw` 必须保存在 `current/` 之外（§9.1）。
- **SCF 不收敛时 ORCA 异常终止**（`SCF NOT CONVERGED` → `error termination in LEANSCF`，返回码 0），并在工作目录留下大量 `*.tmp`、`*.bas*` 等临时文件。每步运行前必须清空 `current/`。测试中可用 `%scf maxiter 2 end` 稳定地制造 SCF 失败。
- `%output jsonpropfile true end`（小写 true）有效，只生成 `<basename>.property.json`，不生成 gbw JSON。
- **xTB（`! XTB`）的输出中没有 SCF 的 `SUCCESS` 标志**（成功运行也没有，`Output.scf_converged()` 对 xTB 恒为 False；2026-09-27 实测）。后端对 xTB 只校验 `terminated_normally()`，对 SCF 方法才校验 `scf_converged()`（`orca_opi.py` 中 `_validate_output`）。
- OPI 的 `Input.moinp` 在**赋值时**就检查文件存在（不存在抛 `FileNotFoundError`），所以后端必须先把 `guess.gbw` 复制到 `current/` 再设置 `input.moinp`。

### 2.7 OPI 2.0.0 API 实测要点（交接时容易踩坑）

- ORCA 路径：设置环境变量 `OPI_ORCA=/home/ruigengji/ORCA611`（否则 OPI 用 `which orca` 查找）。OpenMPI 路径可用 `OPI_MPI`，本机系统 MPI 已在 PATH 中，不必设置。**坑：本机 `/usr/bin/orca` 是 GNOME 屏幕阅读器（Python 脚本），不是 ORCA**——不设 `OPI_ORCA` 时 OPI 会解析到它并报"Could not determine version of ORCA binary"。测试的 conftest 因此在用 `which` 判定时额外要求 ELF 二进制。
- `Atom` 必须用关键字传坐标：`Atom("O", coordinates=(0.0, 0.0, 0.0))`。位置参数写法 `Atom("O", (0,0,0))` 会报 `missing 1 required positional argument: 'coordinates'`。
- `Structure(atoms=[...], charge=0, multiplicity=1)`；每步更新坐标用 `structure.update_coordinates(array_ang)`，要求形状 `(N, 3)`。
- `calc.input.add_arbitrary_string('%pointcharges "pc.pc"')` 要传 **str**，传 `ArbitraryString` 对象会报 TypeError。
- 关键字对象：`from opi.input.simple_keywords import Method, BasisSet, Task`，例如 `Method.HF`、`BasisSet.DEF2_SVP`、`Task.ENGRAD`。
- `Calculator(basename, working_dir, version_check=False)`：`version_check=True` 每次构造都会调用 ORCA 查版本，应只在后端初始化时单独调用一次 `calc.check_version()`。
- `Calculator.json_via_input=True`（默认）会让 ORCA 同时写 `jsonpropfile` 和 `jsongbwfile`；gbw JSON 对 H2O 就有约 1 MB，体系大时会显著拖慢。**后端应关闭 gbw JSON，只保留 property JSON。** 具体做法：`json_via_input=False` 后手动添加 `%output jsonpropfile true end`（通过 `BlockOutput` 或 arbitrary string）。
- `Output`：`terminated_normally()` 与 `scf_converged()` 直接 grep `<basename>.out`，不依赖 JSON。`get_final_energy()`（Eh）、`get_gradient()`（扁平列表，Eh/bohr）、`get_s2()` 等需要先 `parse()`。**`parse()` 默认会在缺少 gbw JSON 时额外启动进程去生成它**，后端必须调用 `out.parse(do_create_gbw_json=False, read_gbw_json=False)`。实测 JSON 中的梯度与 `.engrad` 差约 1e-7 Eh/bohr（有效数字较少）。**能量与梯度的主数据源用 `.engrad`/`.pcgrad`；OPI Output 用于终止状态、SCF 收敛判定及诊断量。**
- OPI **没有** `.pcgrad` 读取器，也没有 `.engrad` 读取器；两者由本项目在 `backend/orca_files.py` 中实现（见 §8.4）。

### 2.8 OpenMM `PythonForce` 契约（8.5.2）

- `openmm.PythonForce(computation, globalParameters={})`：`computation(state)` 返回 `(energy, forces)`，energy 为标量（或带单位量），forces 为 NumPy 数组（或带单位量）。
- 有 `setUsesPeriodicBoundaryConditions(bool)`：设为 True 后，传入的 `State` 可取得周期盒向量（M5 需要）。
- `setForceGroup` / `setName` 可用。
- （2026-09-27 实测补充）`state.getPositions(asNumpy=True)` 返回带单位的 Quantity，回调内需 `.value_in_unit(unit.nanometer)` 剥离；回调抛出的 Python 异常穿过 C++ 层后以 `openmm.OpenMMException` 抛出（原始消息保留，类型不保留）；返回的力数组单位为 kJ/mol/nm。包含 PythonForce 的 System 不能 XML 序列化。
- （2026-09-27 实测补充，虚拟位点）OpenMM **会**把 PythonForce 返回数组中虚拟位点上的力按权重分配到母原子（Reference 平台实测）；但 **`Context.setPositions` 不重算虚拟位点坐标**（坐标只由积分器在 `step()` 时维护），所以能量/力的有限差分测试必须整体平移含虚拟位点的分子。为不依赖平台行为差异，回调（`force.py`）自行把虚拟位点力分配到母原子并把位点自身力清零，两种路径结果一致。

### 2.9 openmm-pyscf 参考实现中需要修正的两处

参考实现：`/home/ruigengji/openmm-pyscf/openmmpyscf/pyscfforce.py`。

1. **`_remove_bonded_terms`（第 718–744 行）删除所有"包含任一 QM 原子"的键合项。** 整分子 QM 时正确；有共价边界时会把 Q1–M1 键、跨边界的角和二面角也删掉，边界失去约束。新规则见 §6.3。
2. **`_apply_electrostatic_embedding`（第 747–774 行）对"QM 原子 × 所有原子"逐对添加 exception。** 50 个 QM 原子 × 5 万原子 ≈ 250 万个 exception，对溶剂化酶体系不可接受。新做法见 §6.2（把 QM 原子电荷置零，只对 QM–QM 对加 exception）。非周期下两者数学等价；PME 下新做法更干净，因为 QM 电荷根本不进入倒易空间。

此外，参考实现每步不复用 SCF 初猜（`_is_qm_model_cacheable` 返回 False），这也是 ORCA 后端可以用 `.gbw` restart 获得性能优势的地方。

---

## 3. 为什么用 OPI，以及 OPI 不管什么

### 3.1 OPI 的收益

OPI 的价值不是"让 ORCA 变快"（ORCA 仍是外部进程），而是把"ORCA 是外部程序"这件事封装在一个稳定的、懂 ORCA 的层里：

1. **结构化输入**：`Structure` 管坐标、电荷、多重度；关键字和 block 是类型化对象（`blocks/` 下有 scf、cpcm、casscf、basis、method、rel、mdci 等），拼写错误在 Python 端就报错。
2. **高级功能直接透传**：RIJCOSX、DLPNO、broken-symmetry、ECP、相对论、CASSCF 等都有 OPI 对象或可用 arbitrary string 表达。
3. **输出检查**：`terminated_normally()`、`scf_converged()`、`get_s2()`（自旋污染，开壳层诊断）、布居分析（Mulliken/CHELPG/Hirshfeld/MBIS）、偶极矩等，直接可用于错误处理和轨迹诊断。
4. **版本管理集中**：`check_version()`；ORCA 路径与 MPI 环境由 OPI Runner 统一设置。
5. **restart 入口**：`Input.moinp` + `MOREAD`、`%scf autostart`。
6. **边界清晰**：OpenMM 层完全不接触 `.inp`、`.gbw`、`%pal`、`%maxcore`。

### 3.2 OPI 不管、由本项目负责的部分

| 缺口 | 本项目的处理 |
|---|---|
| 无 `.pcgrad` 读取 | `backend/orca_files.py` 实现 |
| 无 `.engrad` 读取（仅有 JSON 梯度，精度较低） | 同上 |
| `PointCharge`（inline Q）语义不符 | 改写 `%pointcharges` 文件（§2.3） |
| 无 MPI 性能调优 | 后端设置 OMPI_MCA 环境变量（§2.2） |
| 无 SCF restart 状态机 | `runtime/restart.py` |
| 无常驻模式 | 接受每步进程启动开销（≈0.4 s 串行） |

---

## 4. ORCA 与 PySCF 后端对比

| 维度 | ORCA/OPI | PySCF（openmm-pyscf） |
|---|---|---|
| 每步固定开销 | 串行 ≈0.4 s；MPI 调优后再加 ≈1.4 s；xTB ≈0.2 s | ≈0（进程内） |
| 中等体系杂化 DFT 梯度 | 强：RIJCOSX，MPI 可用满 40 核 | CPU 上偏慢，OpenMP 扩展一般 |
| 过渡金属 / 开壳层 | 强：TRAH、SlowConv、level shift；broken-symmetry 成熟 | 可做，需更多手动调参 |
| 适合 MD 的低成本方法 | r2SCAN-3c、B97-3c 等复合方法；同一后端可跑 GFN2-xTB | 需另接 |
| 激发态 / 多参考梯度 | TDDFT、CASSCF 梯度成熟 | 有，成熟度较低 |
| ECP / 相对论 / 重元素 | 全面 | 可以，梯度支持较少 |
| 周期性嵌入 | 无现成方案（本项目 M5 用截断嵌入） | 已有 Ewald 实现（`pyscf.qmmm.pbc`） |
| 可定制性 | 闭源，只能拿到输出文件中的量 | 开源，中间量（密度矩阵等）随取 |
| 许可 | 学术免费，用户自装 | Apache，随包安装 |

**结论**：两者互补。PySCF 适合小 QM 区、高频调用、需要严格周期性、或需要改算法的场景。ORCA 适合 QM 区较大、杂化 DFT、金属/开壳层化学（正是酶活性位点 B 的主战场），且可以用 xTB 做低成本预采样。设计上保持 backend 协议与具体程序无关，为将来合并成 `openmm-qm` 留口子，但 v0.x 不为通用性提前抽象。

---

## 5. 架构

### 5.1 分层

```text
┌──────────────────────────────────────────────────────────────┐
│ potential.py  ORCAPotential（用户入口，风格同 openmm-ml）     │
├──────────────────────────────────────────────────────────────┤
│ force.py      QMMMForceCallback：State → numpy → backend      │
│               → 组装全体系力 → (energy, forces)               │
├──────────────────────────────────────────────────────────────┤
│ qmmm/         OpenMM 这一侧的处理（完全不知道 ORCA 存在）     │
│   system.py      复制 System、删除键合项、改 NonbondedForce   │
│   charges.py     提取 MM 电荷、边界电荷再分配                 │
│   linkatoms.py   link atom 几何与力回分（M4）                 │
│   embedding.py   嵌入电荷选择：全部 / 截断 + 最小镜像（M5）   │
│   imaging.py     QM 分子跨周期边界时拼回完整分子（M5）        │
├──────────────────────────────────────────────────────────────┤
│ backend/      QM 程序后端                                     │
│   base.py        QMBackend 协议、QMRequest、QMResult          │
│   fake.py        解析解假后端（测试 OpenMM 层用）             │
│   orca_opi.py    ORCA/OPI 后端                                │
│   orca_files.py  .engrad / .pcgrad / 点电荷文件的读写         │
├──────────────────────────────────────────────────────────────┤
│ runtime/      外部进程运行时                                  │
│   scratch.py     每个后端实例一个 scratch 目录                │
│   restart.py     .gbw restart 状态机                          │
│   diagnostics.py 失败现场打包、分段计时日志                   │
├──────────────────────────────────────────────────────────────┤
│ units.py      所有单位换算常数（唯一出处）                    │
└──────────────────────────────────────────────────────────────┘
```

包名：`openmmorca`（与 `openmmpyscf` 命名风格一致），仓库根目录 `/home/ruigengji/openmm-ORCA`。

### 5.2 依赖规则（必须遵守）

- `qmmm/` 不 import `backend/` 中任何 ORCA 相关模块；只通过 `backend/base.py` 的协议交互。
- `backend/` 不 import `openmm`（`units.py` 之外不使用 `openmm.unit`）。backend 只接收 numpy 数组，返回 numpy 数组。
- ORCA 文件名、关键字、`%` block 只出现在 `backend/orca_opi.py` 与 `backend/orca_files.py`。
- 单位换算常数只出现在 `units.py`。
- `runtime/` 不理解化学含义，只管目录、文件、计时和 restart 状态。

这样可以把问题清晰地切成三类：OpenMM 层 bug（用 fake 后端定位）、ORCA 后端 bug（脱离 OpenMM 单测）、OPI 集成 bug。

### 5.3 与 openmm-pyscf 的关系

- 独立仓库、独立包，**不 fork 后原地改**。从 openmm-pyscf 移植 OpenMM 侧逻辑（System 复制、约束处理、力组、电荷提取、单位处理），并按 §2.9 修正。
- 用户 API 形状保持一致，便于在两个后端之间切换做交叉验证。
- 不修改 `/home/ruigengji/openmm-pyscf`（别人的项目，只读参考）。

---

## 6. 能量与力的定义

### 6.1 能量分解（v0.1，整分子 QM，电子嵌入）

E_total = E_OpenMM^modified + E_ORCA^(QM+embedding)

| 相互作用 | 由谁计算 |
|---|---|
| MM–MM 键合 | OpenMM |
| MM–MM 静电、LJ | OpenMM |
| QM–MM LJ | OpenMM |
| QM 内部键合 | 删除（ORCA 覆盖） |
| QM–QM 静电、LJ | 删除（ORCA 覆盖） |
| QM–MM 经典静电 | 删除（ORCA 电子嵌入覆盖） |
| QM 电子 + 核能量 | ORCA |
| QM 核—MM 点电荷 | ORCA |
| QM 电子—MM 点电荷 | ORCA |
| MM 点电荷之间 | **不能**由 ORCA 算（否则重复），见 §2.3 |

力：

- F_Q = −∂E_ORCA/∂R_Q（来自 `.engrad`）
- F_M = −∂E_ORCA/∂R_M（来自 `.pcgrad`）

**`F_M` 是必需输出**。只让 MM 极化 QM 而不把反作用力返回给 MM，会破坏动量守恒、造成 NVE 漂移。接口中 `mm_forces` 允许为 `None` 只为 full-QM 模式。

### 6.2 NonbondedForce 改造（修正 O(N²) 问题）

对唯一的 `NonbondedForce`（多于一个则报错）：

1. 对每个 QM 原子 i：`setParticleParameters(i, charge=0, sigma_i, epsilon_i)`。QM–MM 静电因此自动消失，QM–MM LJ 保留。
2. 对每个已有 exception (i, j)：若 i、j 至少一个是 QM 原子，令 `chargeProd = 0`；若 i、j 都是 QM 原子，再令 `epsilon = 0`。其他参数不变。
3. 对每个没有 exception 的 QM–QM 原子对：`addException(i, j, 0, sigma_ij, 0)`，去掉 QM–QM LJ（sigma 取 Lorentz–Berthelot 组合，数值无关紧要，因为 epsilon=0）。

exception 数量从 O(N_QM × N) 降到 O(N_QM²)。

注意：

- PME 下，把 QM 电荷置零会让 MM 部分带净电荷，OpenMM 会自动加中和背景。只要 QM 区电荷在模拟中不变，这只是一个常数能量偏移，不影响力。文档与日志中要注明。
- 被移除的 QM 原子电荷要保存下来（`charges.py`），用于诊断和边界电荷再分配。
- 只允许以下力类型出现在输入 System 中：`HarmonicBondForce`、`HarmonicAngleForce`、`PeriodicTorsionForce`、`RBTorsionForce`、`CMAPTorsionForce`、`NonbondedForce`、`CMMotionRemover`、`MonteCarloBarostat`。其他力类型（例如 `CustomNonbondedForce`、`GBSAOBCForce`）**出现即报错**。v0.x 采取保守策略：通用地判断任意力是否涉及 QM 原子不可靠，所以不做这个判断。**不静默忽略未知力。**
- `NonbondedForce` 使用了参数 offset（`getNumParticleParameterOffsets() > 0` 或 `getNumExceptionParameterOffsets() > 0`，常见于自由能计算）时报错。
- 电荷为 0 的 MM 原子（例如 TIP4P 的 O）不写入点电荷文件，以减小 ORCA 输入；力映射按保留下来的原子索引进行。
- 虚拟位点（例如 TIP4P 的 M 位点）带电荷时照常作为点电荷。OpenMM 会把作用在虚拟位点上的力自动分配到它的母原子，这一点需要测试覆盖。

### 6.3 键合项删除规则（修正共价边界问题）

**规则：只删除所有原子都在 QM 区的键合项。** 至少有一个 MM 原子的键合项保留在 OpenMM 中。

实现上，"删除"用**把力常数置零**来完成（HarmonicBond/Angle 的 k=0，PeriodicTorsion 的 k=0，RBTorsion 的 c0..c5=0；CMAP 项改指向一张新加的全零 map）。OpenMM 的这些力类没有删除单项的 API，置零与删除在能量和力上完全等价，而且不改变项的索引，比 openmm-pyscf 的 XML 往返更简单。

- 整分子 QM（v0.1）：没有跨边界的键合项，新规则与 openmm-pyscf 的旧规则结果相同。
- 共价边界（v0.3）：Q1–M1 键、Q2–Q1–M1 角、Q3–Q2–Q1–M1 及 Q2–Q1–M1–M2 二面角等保留，由 MM 力场约束边界几何。这是 link atom 方案的标准做法。
- `CMAPTorsionForce`：同样按"所有原子都在 QM 区才删除"。部分在 QM 区的 CMAP 项保留。
- 约束：`removeConstraints=True` 时删除两端都在 QM 区的约束。跨边界的约束（例如 M1 上的 H-bond 约束，一端 MM）保留。`removeConstraints=False` 时保留 QM 区内部的约束（例如为了用更大步长保留 X–H 约束），但构造期发出警告：约束会改变 QM 势能面上的动力学，NVE 验证（测试 8）必须在 `removeConstraints=True` 下进行。

### 6.4 units 约定

backend 边界统一：

```text
输入：坐标 nm，电荷 e
输出：能量 kJ/mol，力 kJ/mol/nm
```

ORCA 适配器内部负责换算：

```text
nm → Å：×10
Å → bohr：÷0.529177210903
Eh → kJ/mol：×2625.4996394799
Eh/bohr → kJ/mol/nm：×2625.4996394799 × 10 / 0.529177210903  (= ×49614.752589)
```

以上常数只允许出现在 `units.py`；测试也从 `units.py` 引用。

---

## 7. Backend 协议

```python
@dataclass(frozen=True)
class QMRequest:
    qm_positions_nm: np.ndarray          # (N_qm_eff, 3)，含 link atom
    qm_elements: tuple[str, ...]         # 长度 N_qm_eff
    mm_positions_nm: np.ndarray | None   # (N_emb, 3)，已选好、已镜像的嵌入电荷位置
    mm_charges_e: np.ndarray | None      # (N_emb,)
    step: int                            # OpenMM 回调计数，用于日志和失败包

@dataclass(frozen=True)
class QMResult:
    energy_kj_mol: float
    qm_forces_kj_mol_nm: np.ndarray      # (N_qm_eff, 3)
    mm_forces_kj_mol_nm: np.ndarray | None  # (N_emb, 3)；有嵌入电荷时必须非 None
    timings_s: dict[str, float]          # 至少含 "total"；ORCA 后端再含 "write" "orca" "read"

class QMBackend(Protocol):
    def evaluate(self, request: QMRequest) -> QMResult: ...
    def close(self) -> None: ...
```

设计要点：

- **backend 不知道 OpenMM 原子索引。** 索引映射（哪些是 QM 原子、link atom 在哪、嵌入电荷对应哪些 OpenMM 原子）全部由 `qmmm/` 和 `force.py` 负责。backend 只看到"一组 QM 原子 + 一组点电荷"。
- **QM 电荷与多重度在构造 backend 时固定**，不随请求传入（QM/MM MD 中 QM 区电子态不变）。
- 协议极小，fake 后端和 ORCA 后端都实现它；将来 PySCF 后端也可以实现它。

---

## 8. ORCA/OPI 后端设计

### 8.1 用户配置

```python
ORCAPotential(
    method="PBE0",               # 放入 simple keywords 的方法名；也可为 "XTB"、"r2SCAN-3c" 等
    basis="def2-SVP",            # 复合方法 / xTB 时为 None
    charge=0,
    multiplicity=1,
    nprocs=1,                    # >1 时写 %pal 并设置 OMPI_MCA 环境变量
    maxcore_mb=2000,             # 每个 MPI 进程的内存（ORCA %maxcore 语义）
    extra_keywords=("RIJCOSX", "def2/J", "TightSCF"),  # 原样追加的 simple keywords
    extra_blocks=(),             # 原样追加的 % block 字符串，例如 "%scf maxiter 300 end"
    scratch_root=None,           # 默认 /tmp/openmmorca-<user>
    restart=True,                # 使用 .gbw restart
    keep_failed=True,            # 失败时保留失败包
    orca_path=None,              # 默认读 OPI_ORCA 或 PATH
)
```

原则：常用参数显式化；ORCA 高级参数以 `extra_keywords` / `extra_blocks` 原生透传。不设计"统一关键字语言"。

后端必须拒绝的用户输入（构造时报错）：

- `extra_keywords` 中出现任务类关键字（`SP`、`Opt`、`Freq`、`NumFreq`、`MD`、`EnGrad`、`NumGrad`，不区分大小写）——任务由后端固定为 `EnGrad`。
- `extra_blocks` 中出现 `%pointcharges`、`%pal`、`%maxcore`、`%moinp`——这些由后端管理。

### 8.2 每步流程

```text
QMRequest
  │
  ├─ 1. 坐标 nm→Å，structure.update_coordinates()
  ├─ 2. 写 <scratch>/current/pc.pc（N 行 q x y z，Å，%.10f）
  ├─ 3. 决定初猜：有可用 last-good .gbw → 复制为 current/guess.gbw 并加 MORead + %moinp
  ├─ 4. calc.write_input(); 运行 ORCA（OPI Runner），带超时
  ├─ 5. 校验（§8.5），失败则走 restart 状态机（§9.2）
  ├─ 6. 读 current/qm.engrad、current/qm.pcgrad（orca_files.py）
  ├─ 7. 换算单位，梯度取负得到力
  ├─ 8. current/qm.gbw → restart/last_good.gbw（原子替换：先写临时文件再 os.replace）
  └─ 9. 记录分段计时，返回 QMResult
```

固定 basename 为 `qm`，固定目录 `current/`，**每步覆盖，不为每步新建目录**。

### 8.3 输入文件示例（后端生成的目标形态）

```text
! PBE0 def2-SVP RIJCOSX def2/J TightSCF EnGrad MORead
%pal nprocs 8 end
%maxcore 2000
%output jsonpropfile true end
%moinp "guess.gbw"
%pointcharges "pc.pc"
* xyz 0 1
O  ...
H  ...
*
```

- 第一步或 restart 被禁用时去掉 `MORead` 与 `%moinp`。
- `nprocs == 1` 时不写 `%pal`。
- `%moinp` 的文件必须在 working_dir 内（OPI `write_input` 会检查 `moinp` 是否为工作目录子路径）。
- `%moinp` 文件不能与 basename 同名（§2.6 实测：ORCA 启动时先覆盖 `qm.gbw`，导致 `No orbitals were found in the gbw file`），所以初猜文件必须改名为 `guess.gbw`。

### 8.4 文件读写（`orca_files.py`）

- `write_pointcharges(path, charges_e, positions_ang)`：第一行 N，每行 `q x y z`；N=0 时不写文件、不加 `%pointcharges`。
- `read_engrad(path) -> (energy_eh, gradient_eh_bohr[N,3])`：跳过 `#` 注释行；校验原子数与期望一致。
- `read_pcgrad(path) -> gradient_eh_bohr[N,3]`：校验第一行 N 与写入的点电荷数一致。
- 所有读函数在文件缺失、行数不对、出现 NaN/Inf 时抛出带文件路径的 `ORCAOutputError`。

### 8.5 校验清单（任何一项失败都不能返回结果）

1. ORCA 进程未超时。（**返回码不可信**：ORCA 异常终止时也返回 0，见 §2.6。返回码非 0 视为失败，但返回码为 0 不代表成功。）
2. `Output.terminated_normally()` 为 True。
3. `Output.parse()` 成功，`scf_converged()` 为 True。
4. 结果文件必须来自本步：每步运行前清空整个 `current/`（§9.1），因此运行后存在的 `qm.engrad`/`qm.pcgrad` 一定是本步生成的。
5. engrad 原子数 = 请求的 QM 原子数（含 link atom）；pcgrad 行数 = 点电荷数。
6. 能量和所有梯度分量有限。
7. `.engrad` 能量与 `Output.get_final_energy()` 相差 < 1e-8 Eh（交叉校验两条读取路径）。
8. 最大单原子力超过阈值（默认 50000 kJ/mol/nm，可配置）时发出警告并写诊断，不自动中止。

### 8.6 MPI 与线程

- `nprocs > 1`：只在 ORCA 运行期间设置 §2.2 的 `OMPI_MCA_*` 与 `OMP_NUM_THREADS=1`，运行结束（包括异常）后恢复原值。OPI Runner 通过 `subprocess.run` 继承 `os.environ`，没有传 env 的参数，所以用一个上下文管理器临时修改 `os.environ` 并在 `finally` 中恢复。
- **OPI 的环境恢复有 bug（2026-09-27 实测，OPI 2.0.0）**：`_orca_environment`（装饰在 `Runner.run` 上，`check_version` 也经过它）在 `finally` 里执行 `os.environ = org_env`，其中 `org_env = os.environ.copy()` 是普通 dict。第一次调用 Runner 之后，`os.environ` 就不再是 `os._Environ`，后续修改不会调用 putenv，子进程收不到。后果：第一次 ORCA 调用是串行时，之后所有 nprocs>1 的调用都拿不到 MCA 设置，每次退回约 10 s 的 MPI 启动开销；而 OPI 自己在 PATH/LD_LIBRARY_PATH 前面追加的路径和我们设的变量，也永远留在进程环境里。后端的对策是 `_preserve_os_environ()`：每次调用 Runner（`check_version` 和 `run_orca`）前保存真正的 `os.environ` 对象和它的内容，调用后把这个对象放回 `os.environ`，再逐个变量恢复原值（`os._Environ` 的赋值和删除会调用 putenv/unsetenv）。测试在全新的 Python 进程里检查子进程实际继承到的环境（`tests/test_orca_mpi.py`），不能只看 `os.environ`，否则会被前面测试的状态掩盖。
- **CPU 绑定（2026-09-27 实测）**：OpenMPI 5 的 `mpirun` 不理会调用者的 `taskset` 亲和性，总是从 0 号核开始按核绑定进程。同一台机器上同时跑两个 nprocs>1 的 ORCA，或 ORCA 和别的程序共用机器时，会互相抢同一批核（实测 nprocs=2 比串行还慢）。要把 ORCA 限定在指定核上，用 `PRTE_MCA_hwloc_default_cpu_list=4-35`（`OMPI_MCA_` 前缀无效）。后端不自动设置它，由用户或作业脚本决定。
- **PBS 任务内的槽位限制（2026-09-27 实测）**：OpenMPI 5/PRRTE 会读取资源管理器分配；任务只分到 1 核时 `mpirun -np 2` 报 "Not enough slots available"，ORCA 在 Startup 异常终止。**并行 ORCA 要求任务申请的核数 ≥ nprocs**（`qsub -l select=1:ncpus=N`）；确实要超订时由用户显式设 `OMPI_MCA_rmaps_default_mapping_policy=:oversubscribe`（或 PRTE_MCA_ 前缀），后端绝不自动超订。测试的 `require_mpi` fixture 用真实 `mpirun -np 2 hostname` 探针判定。
- `nprocs` 上限默认取物理核数（本机 40），超过则警告。
- 推荐 OpenMM 使用 CUDA 平台（占 1 个 CPU 线程）。若使用 CPU 平台，建议设置 `OPENMM_CPU_THREADS` 为小值，避免与 ORCA 抢核。
- `maxcore_mb × nprocs` 超过物理内存 75% 时构造期警告。

---

## 9. 运行时：scratch、restart、诊断

### 9.1 scratch 目录

```text
<scratch_root>/<instance_uuid>/
├── current/        每步覆盖：qm.inp qm.out qm.engrad qm.pcgrad qm.gbw pc.pc guess.gbw qm.property.json
├── restart/        last_good.gbw
├── failures/       failure_step_<NNNNNN>/ …
└── timings.csv     step, t_write, t_orca, t_read, t_total, scf_cycles, restart_used
```

- 默认 `scratch_root=/tmp/openmmorca-<user>`（tmpfs，快）。用户可改到磁盘。
- 后端 `close()` 或进程正常退出时删除 `current/` 与 `restart/`，保留 `failures/` 与 `timings.csv`。
- 一个后端实例 = 一个 uuid 目录 = 一个 OpenMM Context。

### 9.2 SCF restart 状态机

```text
第一步：全新 SCF（无 MORead）
   │ 成功 → 保存 last_good.gbw
   ▼
第 n 步：MORead last_good.gbw
   ├─ 成功 → 替换 last_good.gbw
   └─ 失败（未收敛 / 异常终止）
        ▼
      全新 SCF 重试一次（无 MORead）
        ├─ 成功 → 替换 last_good.gbw，计数器 +1，写警告
        └─ 失败 → 写失败包，抛出 ORCACalculationError，MD 停止
```

**绝对禁止**：SCF 失败时返回上一步的能量或力。

可配置项：`restart=True/False`、`max_fresh_retries=1`。

### 9.3 失败包

```text
failures/failure_step_000123/
├── qm.inp  qm.out  pc.pc
├── qm.engrad qm.pcgrad（若存在）
├── guess.gbw（若使用）
├── geometry.xyz        QM 区（含 link atom）
└── metadata.json       step、QM 原子索引映射、嵌入电荷对应的 OpenMM 索引、配置、ORCA/OPI 版本、错误信息
```

用户进入该目录运行 `orca qm.inp` 即可复现。

### 9.4 分段计时

每步写一行 `timings.csv`。`ORCAPotential` 提供 `summarize_timings()`（均值、P50、P95）。这是调 `nprocs`、判断 restart 收益的依据。

### 9.5 并发

- 一个后端实例只服务一个 active Context；在实例上持有 `threading.Lock`，重入调用直接报错（而不是排队）。
- 两个 Context 需要两个后端实例（两个 uuid 目录），因此两个 `createMixedSystem` 调用各自生成独立的后端。

---

## 10. Link atom（M4，v0.3）

### 10.1 输入

用户显式给出边界对：`boundary_pairs=[(q1_index, m1_index), ...]`（OpenMM 原子索引）。约束：

- Q1 在 QM 区、M1 在 MM 区，二者在 topology 中成键。
- 只支持单键边界（通常 C–C）；每个 Q1 最多一个 M1。
- QM 区总电荷 `charge` 由用户按"QM 原子 + link H"给出。

### 10.2 link atom 几何

L 放在 Q1→M1 键上：

R_L = R_Q1 + g · (R_M1 − R_Q1)

g 为固定比例，默认 g = d_eq(Q1–H) / d_eq(Q1–M1)。对 C–C 边界取 1.09 / 1.526 ≈ 0.714。用户可按边界对单独设置。

（原讨论稿中的公式 `R_H = R_B + λ(R_B − R_A)` 方向写反了，L 会落在 Q1 远离 M1 的一侧，此处已更正。）

### 10.3 力回分（链式法则）

L 不是 OpenMM 粒子，其受力 F_L 必须分给 Q1 与 M1：

F_Q1 += (1 − g) · F_L
F_M1 += g · F_L

g 固定时这是精确的链式法则结果，能量守恒。

### 10.4 边界电荷处理（charge shift）

M1 的点电荷离 L 只有约 0.44 Å，直接放入嵌入会过度极化。v0.3 采用 charge-shift 方案：

1. 嵌入电荷中 M1 的电荷置为 0。
2. 原 M1 电荷平均分到与 M1 成键的 MM 原子（M2 集合）上，仅用于嵌入。
3. OpenMM 中 MM–MM 静电**不变**（只改传给 ORCA 的嵌入电荷）。

M2 上叠加了额外电荷，因此 pcgrad 给出的 M2 的梯度已包含这部分，力直接加到 M2 上，无需额外回分。M1 在嵌入中电荷为 0，只通过 link atom 力回分获得 QM 力。

以后可扩展 RCD（redistributed charge and dipole）等方案；`charges.py` 中以策略对象实现，v0.3 只提供 `ChargeShift`。

注意这是**简化版** charge shift：文献中的方案（Sherwood 等，ChemShell）还在 M2 附近加一对点电荷以补偿 M1–M2 键偶极，本版不加。docstring 与 README 必须写明。

QM 区取自力场残基的一部分时，QM 原子的 MM 电荷之和一般不是整数（例：ALA 的 CA、HA、CB、HB1–3 为 +0.1144 e），嵌入电荷整体因此带非整数净电荷；charge shift 不解决这个问题。v0.3 只在 |Σq − round(Σq)| > 0.05 e 时警告。

**M5 起（2026-10-01，prep D3）改为修正**：对每个被切开的残基 r，取其 QM 部分的力场电荷 x_r，把 δ_r = x_r − round(x_r) 平均加到 r 自己的 M2 原子上（r 中没有 M2 时加到 r 其余非 M1 的 MM 原子上），使每个被切开残基的 MM 剩余部分为整数电荷，嵌入总电荷也为整数。修正是局部的，与用户给的 QM 电荷无关；|δ_r| > 0.25（接近半整数，取整有歧义）或 Σ round(x_r) ≠ 用户 `charge` 时警告。DhlA 方案 A：Asp124 侧链 x = −0.858 → Asp 的 M2（N、HA、C）共加 +0.142 e。

### 10.5 与 §6 的交互

- 键合项：按 §6.3 规则，Q1–M1 相关项保留。
- NonbondedForce：Q1 电荷置零（它是 QM 原子）；Q1–M1、Q1–M2 等已有 exception 的 chargeProd 置零（§6.2 第 2 条自动覆盖）；LJ 保留。
- 嵌入电荷集合 = 所有 MM 原子（M5 前）减去 M1，加上 M2 的额外电荷。

---

## 11. 周期性 MM + 截断嵌入（M5，v0.4）

### 11.1 定义

- OpenMM 继续用 PME 计算 MM–MM 与 QM–MM LJ（QM 电荷已置零）。
- QM 区看到的嵌入电荷：以**残基（或用户定义的电荷组）为单位**，只要该组任一原子与任一 QM 原子的最小镜像距离 < R_emb（默认 12 Å），整组纳入。按组选择避免把中性基团切开。
- 嵌入电荷位置取相对于 QM 区的最小镜像；QM 区自身先由 `imaging.py` 拼回完整分子。
- 截断以外的 QM–MM 静电被忽略：**这是近似**，文档、日志与 API docstring 都要明确说明，不得宣称 PME 兼容。

### 11.2 QM 分子镜像

以 QM 区的第一个原子为锚点，沿 topology 键图做 BFS，把每个 QM 原子平移到与已放置邻居最近的镜像。力按原 OpenMM 原子索引返回（平移不改变力）。

### 11.3 截断的不连续性

硬截断下，电荷组进出截断球会造成能量跳变，NVE 不守恒。v0.4 的处理：

1. v0.4 只承诺 NVT/NPT 下稳定运行，不承诺严格 NVE。
2. 每步记录嵌入组数量与"进出组"事件，以便评估影响。
3. 平滑开关函数需要 ∂E/∂q_j，即 QM 在点电荷处产生的静电势（由 Hellmann–Feynman 定理，变分 SCF 能量对 q_j 的导数就是该静电势）。ORCA 自带 `orca_vpot` 可以计算指定点的静电势，但每步多一次调用。作为 v0.4 之后的可选改进，不进入 v0.4 范围。

### 11.4 PythonForce 设置

M5 中回调需要盒向量：`PythonForce.setUsesPeriodicBoundaryConditions(True)`，从 `state.getPeriodicBoxVectors()` 读取。非周期体系保持 False。

实测（2026-09-29，OpenMM 8.5.2）：

- 回调收到的坐标**不做包裹**（Reference/CPU/CUDA 相同）。§11.2 的拼接主要防输入结构按原子包裹；拼接必须把边界 M1 一起放到 Q1 的镜像里，否则 link atom 位置出错。
- `MonteCarloBarostat` 每次尝试额外触发 **2 次** PythonForce 调用，NPT 下 QM 开销增加 2/频率（默认 25 → +8%）。
- 截断判据是"组内任一原子与任一 QM 原子"的最小镜像距离，因此要求 R_emb + D < 最短盒宽 / 2（D 为 QM 区尺寸），每步校验（NPT 下盒子会变）。
- charge shift 是静态的：截断嵌入使用 shift 后的电荷，M1 不进入任何组。

---

## 12. 错误处理总则

QM/MM MD 最大的忌讳是"算错了还继续跑"。

| 情况 | 行为 |
|---|---|
| SCF 未收敛 / ORCA 异常终止 | 按 §9.2 重试一次全新 SCF；再失败则写失败包并抛 `ORCACalculationError` |
| 输出文件缺失、格式错、数量不匹配、NaN | 写失败包，抛 `ORCAOutputError`（不重试，这通常是配置或程序错误） |
| ORCA 超时 | 杀进程，写失败包，抛 `ORCATimeoutError` |
| 输入 System 有不支持的力涉及 QM 原子 | 构造期抛 `ValueError` |
| 边界对不成键、QM 区包含半个分子但未给边界对 | 构造期抛 `ValueError` |
| 力异常大 | 警告 + 诊断，不中止 |

所有异常类放在 `openmmorca/errors.py`，统一继承 `OpenMMORCAError`。

---

## 13. 验证方案

### 13.1 测试分层

| 层级 | 依赖 | 运行时间 | 用途 |
|---|---|---|---|
| unit | 无 ORCA | 秒级 | 文件读写、单位、System 改造、fake 后端下的 OpenMM 层 |
| orca | ORCA（HF/STO-3G、HF/def2-SVP、xTB） | 分钟级 | 后端正确性、有限差分 |
| slow | ORCA + 较大体系 | 十分钟到小时级 | NVE、restart 重现性、基准 |

pytest 标记：`@pytest.mark.orca`、`@pytest.mark.slow`；无 ORCA 时 `orca`/`slow` 自动跳过（依据 `OPI_ORCA` 或 `which orca`）。

### 13.2 fake 后端

解析势，便于精确验证 OpenMM 层：

E = Σ_{a<b ∈ QM} ½ k (|R_a − R_b| − r0_ab)² + Σ_{a ∈ QM, j ∈ emb} k_e · q_a^fake · q_j / |R_a − R_j|

其中 r0_ab 取初始几何，q_a^fake 为构造时给定的固定电荷，k_e 为 OpenMM 使用的库仑常数（138.935458 kJ·mol⁻¹·nm·e⁻²）。它有精确解析梯度，能同时测"QM 力放对了原子"、"MM 反作用力放对了原子"、"单位正确"、"静电没有重复计算"。

### 13.3 必须通过的测试

1. **文件读写**：engrad/pcgrad 读取对照 §2.4、§2.5 的实测文件内容（存为测试 fixture）。
2. **System 改造**：QM 原子电荷为 0；QM–QM 对 LJ 为 0；QM–MM LJ 保留；MM–MM 完全不变（逐项对比能量）；exception 数量为 O(N_QM²)。
3. **fake 后端 + OpenMM**：OpenMM Context 给出的总能量与力 = MM 部分 + fake 解析值；有限差分通过。
4. **ORCA 后端 full-QM H2O**：后端结果 = 直接运行 ORCA 的结果（能量差 < 1e-9 Eh）。
5. **ORCA 有限差分**：H2O 全部 9 个坐标 + H2O + 2 个点电荷全部 6 个点电荷坐标；中心差分 h = 1e-3 Å，TightSCF，误差 < 1e-4 Eh/bohr。
6. **平移不变性**：非周期 QM/MM 整体平移 1 nm，能量差 < 1e-7 Eh；‖ΣF‖ 接近 0（见 §1.3 标准）。
7. **QM 水 + MM 水二聚体**：OpenMM 总能量 = ORCA(QM + 嵌入) + MM(MM 水内部 + QM–MM LJ)，逐项核对，证明没有重复计算和遗漏。
8. **NVE**：QM 水 + 若干 MM 水团簇，0.25 fs，无恒温器，1 ps。
9. **restart**：同一轨迹 restart 开/关，前 50 步能量差 < 1e-6 Eh。
10. **link atom 有限差分**（M4）：对 Q1、M1 坐标做有限差分，验证 §10.3 回分。
11. **截断嵌入**（M5）：选组正确性（构造跨边界的组）；QM 分子镜像拼接正确。

### 13.4 与 PySCF 交叉验证（可选，推荐）

同一 QM 几何 + 同一组点电荷，HF/STO-3G 或 HF/def2-SVP，比较能量、QM 梯度、点电荷梯度。需在单独环境中安装 pyscf（不污染 `openmm_dev`）。差异应在 1e-6 Eh 量级（积分精度差异）。

---

## 14. 里程碑

| 里程碑 | 交付内容 | 验收 | 版本 |
|---|---|---|---|
| **M0 后端冒烟** | `units.py`、`orca_files.py`、`backend/base.py`、`orca_opi.py`（full-QM + 点电荷，无 restart）、测试 1/4/5 | 有限差分通过；每步耗时有记录 | — |
| **M1 OpenMM 层 + fake 后端** | `qmmm/system.py`、`qmmm/charges.py`、`backend/fake.py`、`force.py`、`potential.py`（仅 fake）、测试 2/3 | 全部 unit 测试通过，无需 ORCA | — |
| **M2 非周期电子嵌入** | `ORCAPotential` 接 ORCA 后端；测试 6/7/8；示例脚本 | §1.3 v0.1 标准 | **v0.1（C 完成）** |
| **M3 restart 与健壮性** | `runtime/`（scratch、restart、失败包、计时）；nprocs 基准脚本；测试 9 | §1.3 v0.2 标准 | **v0.2** |
| **M4 link atom** | `qmmm/linkatoms.py`、`ChargeShift`、键合规则切换到 §6.3、测试 10 | §1.3 v0.3 标准 | **v0.3（B 核心）** |
| **M5 周期性 + 截断嵌入** | `qmmm/embedding.py`、`qmmm/imaging.py`、PBC 回调、测试 11、酶体系示例 | §1.3 v0.4 标准 | **v0.4（B 应用）** |
| 以后 | xTB 预采样 / Δ-学习、MTS、平滑开关（orca_vpot）、RCD 边界电荷、严格周期嵌入、与 PySCF 合并为 openmm-qm | — | — |

---

## 15. 基准体系（按复杂度递增）

1. H2O full-QM
2. H2O + 外部点电荷
3. QM H2O + MM H2O（非周期）
4. QM H2O + MM 水团簇（非周期，NVE）
5. 共价边界小分子（ACE-ALA-NME 二肽，QM/MM 切在侧链 CB–CA 键上）
6. 溶剂化小分子（PBC，截断嵌入）
7. 酶活性位点（PBC，link atom，截断嵌入）——卤代烷脱卤酶 DhlA（起始结构 PDB 2DHC 的 DCE 复合物；2HAD 为游离酶），Asp124 对 1,2-二氯乙烷的 SN2（2026-09-29 确定，细节见 `docs/plans/2026-09-29-m4-m5-prep.md` D1）。体系：ff14SB + DCE 用 OpenFF Sage/NAGL + TIP3P，6.83 nm 立方盒，31,610 原子，17 Na⁺。2026-10-01 跑通 1 ps（方案 A，r2SCAN-3c）：DhlA（PDB 2DHC，31,610 原子，PME），QM 方案 A（DCE + Asp124 侧链，15 原子含 1 个 link H，电荷 −1），r2SCAN-3c，40 个 MPI 进程，Langevin NVT 300 K、0.5 fs：2000 步（1 ps）无人工干预，5.53 s/步（ORCA 5.45 s；写 0.009 s、读 0.012 s），平均嵌入 152 个残基组（约 1,800 个点电荷），286 步有组进出截断，0 次 fresh SCF 重试；最后 500 步 298.6 ± 1.1 K；QM 键最大偏离 15.9%（来自 C1–Cl1：第 ~1907 步 O–C1 一度靠近到 0.232 nm 的 SN2 进攻尝试，随后回弹）。

前 4 个验证数学正确性；5 验证边界；6、7 验证真实工作流。

---

## 16. 风险与未决问题

| 风险 / 问题 | 影响 | 应对 |
|---|---|---|
| ORCA 版本升级改变 `.engrad`/`.pcgrad` 格式或 JSON 结构 | 读取失败 | 读取函数严格校验并报错；§8.5 第 7 条双路径交叉校验；测试 fixture 覆盖 |
| MPI 环境在其他机器（集群）上表现不同 | 并行性能差或无法运行 | MCA 变量可配置；基准脚本（M3）在新机器上先跑 |
| 截断嵌入的不连续性对酶体系的影响大小 | 能量漂移、采样偏差 | v0.4 记录进出事件；以后用 orca_vpot 做平滑开关 |
| 酶体系 QM 区大（100+ 原子），每步几分钟 | ps 级模拟需要天量级时间 | M3 基准指导方法选择（r2SCAN-3c / xTB 预采样）；MTS 列入以后 |
| 开壳层/金属体系 SCF 在 restart 下收敛到错误态 | 能量不连续 | 记录 ⟨S²⟩（`get_s2()`）和能量跳变并报警；broken-symmetry 场景由用户在 `extra_blocks` 中配置 |
| OpenMM 输入 System 使用了本设计不支持的力 | 构造失败 | §6.2 白名单 + 清晰报错；按需扩展 |

---

## 17. 明确不做的事（v0.x）

- 为兼容 ASE 再加一层 ASE。
- 一开始支持所有 ORCA 关键字的类型化封装。
- 自适应 QM/MM、三斜盒 PME 嵌入。
- 用缓存的旧力掩盖 SCF 失败。
- 把 `.engrad`/`.pcgrad` 解析散落在项目各处。
- 让 backend 修改 OpenMM `System`，或让 OpenMM 层理解 ORCA scratch 文件。
- 在没有有限差分测试的情况下相信"力看起来差不多"。
- 修改 `/home/ruigengji/openmm-pyscf`。
