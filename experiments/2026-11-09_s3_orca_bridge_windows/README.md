# S3：ORCA/PySCF 桥接扩到 V5 的 100 帧（183）

结果见 [验证记录](../../docs/validation/2026-11-09-s3-orca-bridge-windows.md)。输入为 V5 的 `requests/wXX.npz`（同一组 S2 点电荷与 S3 赝电荷）。

- `orca_bridge.py`（NFS openmm_dev，183 本地 ORCA）：每帧 ORCA r2SCAN-3c 的 S2、S3 能量，nprocs 1，fresh SCF，默认 SCF 设置 → `out/wXX_fYY_<S2|S3>.json`（已有跳过）。
- `run_bridge.sh`：`NWORKERS` 个 worker（默认 20；当时 183 上另有一个作业占约 20 个核）；`SMOKE=1` 只算 w00 第 0 帧。`analyze_bridge.py` → `bridge_summary.json`（PySCF 一侧取 V5 的 `out/`）。

2026-11-09：冒烟 17:21–17:26；全量 17:27–17:48（20 worker，200 个单点）。PID 见 `pids.txt`。
