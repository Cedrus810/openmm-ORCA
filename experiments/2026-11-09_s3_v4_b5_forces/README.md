# S3 提案 V4：B5 反应物区 20 帧的力偏移复核（183）

依据 [S3 提案](../../docs/plans/2026-11-07-s3-ewald-correction-proposal.md) §4 V4；结果见 [验证记录](../../docs/validation/2026-11-09-s3-v4-b5-forces.md)。

- `make_pseudo.py`（openmm_dev，181）：用包内 `EwaldCorrection` 为 B5/T09 的 20 帧生成 S3 赝电荷 → `pseudo_b5.npz`。
- `v4_pyscf.py`（183 本地 pyscf-env）：PySCF r2SCAN/def2-SVP，conv_tol 1e-10，设置同 B5 `pyscf_e1.py`。每帧：S3 能量 + 解析梯度、ENV20 能量（C2 首批的去 QM 自身镜像周期参考），以及 ENV20、S2、S3 沿伸缩/进攻方向的 ±0.002 Å 能量差分；第 0 帧另有 S2 复现、EW20 差分对照，`--s3fd` 为冒烟后加的 S3 差分诊断。结果 `out/frame_XX_<job>.json`，已有跳过。
- `run_v4.sh`（5 worker × 8 线程；`FRAMES=0` 为冒烟）、`run_s3fd.sh`；`analyze_v4.py` → `v4_summary.json`。

2026-11-09：冒烟 14:29–14:31（11 个 SCF）；S3 差分诊断 14:33（4 个）；全量 14:37–15:01（285 个 SCF）。PID 见 `pids.txt`。
