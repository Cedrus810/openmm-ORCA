# S3 V3 Task 5：经 API 重现首批 3 帧

`api_pilot.py capture`（openmm_dev）：`createMixedSystem(..., embeddingCutoff=None, ewaldCorrection=True)` + 零结果捕获后端 → `capture.npz`、`capture.json`。`api_pilot.py orca F`：同一路径接真实 ORCA（r2SCAN-3c，nprocs 1，fresh SCF）→ `orca_XX.json`。`pyscf_capture.py F`（pyscf-env）：捕获请求的 PySCF 能量 → `pyscf_XX.json`。2026-11-07 在 181 本机运行，结果见 `docs/plans/2026-11-07-s3-v3-implementation-plan.md` 的"进度"一节。
