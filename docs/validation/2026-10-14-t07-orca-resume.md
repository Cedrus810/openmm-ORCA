# T07 遗留：ORCA + CUDA 上的中断续跑（2026-10-14）

结论：修复前，在新进程中续跑必然段错误。触发条件已经用纯 OpenMM 的最小复现确认（见下文）。修复之后，ORCA + CUDA 上的 `--resume checkpoint` 实测能完成，和不中断的运行只有 1e-6 nm 量级的差（见末节）。`--resume state` 只有子进程单元测试，没有在 ORCA 上实测。

## 运行

`experiments/2026-10-14_t07_orca_resume/resume_test_181.sh`：kasuga01，CUDA，r2SCAN-3c，nprocs 16（A 用核 4–19，B 用核 20–35），60 步，checkpoint 间隔 20。复用 2026-10-05 正式运行的准备和平衡缓存。

- A（不中断）：60/60 步完成，每步约 6.1 s。
- B：第 30 步之后 `kill -9`。`--resume checkpoint` 从第 20 步恢复，第 21–30 步按设计移到 `steps.superseded.attempt-001.csv`，`run.json` 记录了 attempt 2。**接着在续跑后的第一次力计算中段错误**，连续两次都一样（`B.resume.log`、`B.resume2.log`，后者带 faulthandler）。core dump 中崩溃的线程是 `OpenMM::ComputationWrapper::compute` ← `CommonCalcPythonForceKernel::executeOnWorkerThread`，SEGV_MAPERR，线程上没有 Python 帧。

A 与 B 在已有步数上的对照（`steps.csv` 及 superseded 文件）：第 1–30 步的 O–C1、C1–Cl1 距离、SCF 圈数、嵌入组数之差全部为 0；温度之差最大 3.9e-5 K，所以两个文件不是逐字节相同（CUDA 的 `DeterministicForces=false`，原因没有核实）。

## 最小复现（`experiments/2026-10-14_t07_orca_resume/repro_segv/`，不含 openmmorca 和 ORCA）

| 复现 | 结果 |
|---|---|
| 4 粒子 PythonForce，同一进程内 createCheckpoint → 新 Context → loadCheckpoint → step（新 Context 先 `setPositions`）；Reference/CPU/CUDA，周期性开或关 | 正常 |
| 3 nm PME 水盒 + 周期性 PythonForce，写 checkpoint 和加载在不同进程；CPU/CUDA | **段错误** |
| 同上，去掉 PythonForce（CPU） | 正常 |
| 4 粒子非周期 PythonForce，跨进程；Reference/CPU | **段错误** |
| 4 粒子，新进程中直接 `setState`（State XML） | **段错误** |
| 4 粒子，新进程中先 `setPositions` 再 `loadCheckpoint` 或 `setState` | 正常 |
| 4 粒子，新进程中 `loadCheckpoint` 后只调用 `getState(getForces=True)` | **段错误** |
| 4 粒子，新进程中先让**另一个** Context `setPositions`（不算力），再在本 Context 上 `loadCheckpoint`、算力 | 正常 |

触发条件（区分性对照已确认）：**进程中还没有任何 Context 调用过 `setPositions`，就开始计算 PythonForce。** 这是进程级的条件：任何一个 Context 调用过一次 `setPositions` 就够了，不必是同一个，也不必算过力。它和平台、体系大小、周期性、checkpoint 与 State 哪种加载方式都无关。看起来像是进程级的一次性初始化，但 OpenMM 内部的根因没有查（库是编译好的）。

T04 的测试没有发现这个问题，是因为测试在 pytest 的同一个进程里续跑，进程之前已经 `setPositions` 过。修复前，`examples/enzyme_qmmm.py` 的 `restore_checkpoint` 在新进程中直接 `loadCheckpoint` 或 `setState`，所以 `--resume checkpoint` 和 `--resume state` 都会崩。

## 修复（2026-10-14）

- `examples/enzyme_qmmm.py::restore_checkpoint`：校验通过后、加载之前，先 `simulation.context.setPositions(全零)`；随后的 checkpoint 或 State 会覆盖这些坐标。
- 回归测试 `tests/test_enzyme_workflow.py::test_restore_in_fresh_process_survives_pythonforce[checkpoint|state]`：在**子进程**中写 checkpoint，再在另一个子进程中用 `restore_checkpoint` 续跑并 step。临时去掉修复时，两个用例都失败；加上修复后通过。整个 `tests/test_enzyme_workflow.py` 共 50 个用例通过。
- 真实 ORCA + CUDA 上的续跑结果见下节。

## 修复后的实测（kasuga01，CUDA，r2SCAN-3c np16，核 20–35）

B 以修复后的代码再次执行 `--resume checkpoint --steps 60`（attempt 4，`B.resume3_fixed.log`），从第 20 步恢复，跑完第 21–60 步，`run.json` 状态 completed，没有 fresh SCF 重试。用 `compare_ab.py` 与 A（不中断）对照（`cuda181/compare_ab.json`）：

| 步数范围 | 温度差 max (K) | O–C1 / C1–Cl1 距离差 max (nm) | SCF 圈数差 max |
|---|---|---|---|
| 1–20（B 续跑前保留的前缀） | 2.3e-5 | 0 / 0 | 0 |
| 21–30（B 被 kill 前写的、已移到 superseded 的行） | 3.9e-5 | 0 / 0 | 0 |
| 21–60（B 续跑） | 3.1e-4 | 3.4e-6 / 3.2e-6 | 6 |

- 续跑第一步（第 21 步）按设计是全新 SCF（14 圈，`restart_used=0`）；A 在这一步用的是 MORead。
- 第 60 步最终状态：坐标差最大 4.8e-6 nm，RMS 8.8e-7 nm；速度差最大 2.0e-3 nm/ps。
- 续跑前两条独立运行在位置上逐步一致，续跑后出现 1e-6 nm 量级的差。这和"续跑首步改用全新 SCF、之后 SCF 圈数不同"一致，但没有用对照实验把原因坐实。
- `run.json` 中崩溃的 attempt 2、3 状态停在 `running`：段错误发生时进程来不及写失败状态。只有 attempt 4 是 completed。这一点没有修，记录在这里。
