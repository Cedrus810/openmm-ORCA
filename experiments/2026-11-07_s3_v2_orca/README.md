# S3 提案 V2：ORCA r2SCAN-3c 单点（首批 3 帧）

依据 [S3 提案](../../docs/plans/2026-11-07-s3-ewald-correction-proposal.md) §4 V2；结果见 [验证记录](../../docs/validation/2026-11-07-s3-v2-orca.md)。不改 `openmmorca/`。

- `make_pseudo.py`（pyscf-env）：用 V1 的同一函数、随机种子和岭回归（K = 50，R_s = 2.0 nm，λ = 1e-4）重新生成赝电荷，并与 V1 记录的 |Q| 最大值、总电荷逐位核对；输出 `pseudo_K50_R2.0_lam1e-4.npz`。
- `run_orca_s3.py`（openmm_dev）：一个帧的 ORCA S2 与 S3P2（S2 点电荷 + 赝电荷）单点，设置同首批 ORCA 桥接（r2SCAN-3c，nprocs 1，fresh SCF），scratch 在 `/dev/shm`。

```bash
PSCF=/home/ruigengji/miniforge3/envs/pyscf-env/bin/python
PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
D=experiments/2026-11-07_s3_v2_orca
(cd $D && $PSCF make_pseudo.py)
export OPI_ORCA=/home/kasuga/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2 TMPDIR=/dev/shm OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for f in 0 4 8; do $PY -u $D/run_orca_s3.py $f > $D/orca_$f.log 2>&1 & done; wait
```

2026-11-07 在 181 本机运行，三帧并行，每个单点 56–59 s。输出 `out/candidate_XX_{S2,S3P2}.json`、`orca_*.log`。
