# S3 V3 Task 4：酶单帧有限差分（183）

依据 [V3 实施计划](../../docs/plans/2026-11-07-s3-v3-implementation-plan.md) Task 4。首批候选 0（C2 反应物样帧），方案 A，r2SCAN-3c VeryTightSCF，nprocs 1，经 `createMixedSystem(..., embeddingCutoff=None, ewaldCorrection=True)`；只算 QM/MM 力组（31）。

- `fd_enzyme.py base` / `fd_enzyme.py fd ATOM AXIS ±1`：单点，结果 `out/*.json`，已有结果跳过（可续跑）。h = 1e-4 nm。
- `run_fd.sh`：183 上并行 25 个进程（基准 + 6 原子 × 2 分量 × ±）。`analyze_fd.py` → `fd_summary.json`。
- 2026-11-09 13:22 启动，13:25 全部完成（session-920 记录 3 min 17 s、44 min CPU、内存峰值 24.7 GB）。第一次启动写法 `cd … && setsid nohup … &` 让 ssh 不返回；改为 `cd …; setsid nohup … &`，第二次启动全部跳过已有结果。181 上一度看不到 `out/`（NFS 缓存），当时误判为作业被杀，见 `pids_notes.txt`。

```bash
ssh korakuen@192.168.0.183 'bash -c "cd /home/ruigengji/openmm-ORCA/experiments/2026-11-09_s3_v3_enzyme_fd; setsid nohup ./run_fd.sh > run.log 2>&1 < /dev/null & echo \$!"' >> pids.txt
ssh korakuen@192.168.0.183 'bash -c "cd /home/ruigengji/openmm-ORCA/experiments/2026-11-09_s3_v3_enzyme_fd && /home/ruigengji/miniforge3/envs/openmm_dev/bin/python analyze_fd.py"'
```
