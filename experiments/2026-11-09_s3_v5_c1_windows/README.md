# S3 提案 V5：C1 窗口统计（183）

依据 [C1 合同](../../docs/plans/2026-11-01-c1-comparison-contract.md) §4 与 [S3 提案](../../docs/plans/2026-11-07-s3-ewald-correction-proposal.md) §4 V5；结果见 [验证记录](../../docs/validation/2026-11-09-s3-v5-c1-windows.md)。

- `capture_windows.py`（NFS openmm_dev）：C2 扫描 w00、w05、w06、w07、w15 每窗 20 帧，零力捕获后端取 S2 请求，包内 `EwaldCorrection` 生成 S3 赝电荷 → `requests/wXX.npz`。
- `v5_pyscf.py`（183 本地 pyscf-env）：每帧 S2、S3、ENV20 能量，PySCF r2SCAN/def2-SVP，conv_tol 1e-10 → `out/wXX_fYY_<scheme>.json`（已有跳过）。
- `run_v5.sh`：捕获 5 进程；QM 10 worker × 4 线程（`SMOKE=1` 只算 w00 第 0 帧）。`analyze_v5.py` → `v5_summary.json`。

2026-11-09：冒烟 15:27–15:32；全量 15:33–15:50（300 个 SCF）。PID 见 `pids.txt`。
