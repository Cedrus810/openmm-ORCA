# S3 V3 Task 0：开工前核对

依据 [V3 实施计划](../../docs/plans/2026-11-07-s3-v3-implementation-plan.md) Task 0；结果见 [验证记录](../../docs/validation/2026-11-07-s3-v3-task0.md)。不改 `openmmorca/`。

- `taylor.py`：解析 Taylor 系数（`dV` 在 r_c 处 0–2 阶）与修正的 MM 力（给定 ∂E/∂c），立方盒，numpy 原型。
- `check_classical.py`（pyscf-env）：系数对有限差分、Taylor 多项式对精确 `dV`、MM 力对有限差分、Ewald (α, kmax) 收敛与耗时 → `check_classical.json`、`check_classical.log`。
- `check_taylor_energy.py`（pyscf-env）：首批 3 帧，Taylor 系数下的直接多项式势与固定模板赝电荷（模板半径 0.4/0.6/0.8 nm）的 `E − E_ENV20` → `out/`、`energy_*.log`。
- `check_vpot.py`（openmm_dev）：帧 0、S2 + V2 赝电荷，ORCA r2SCAN-3c VeryTightSCF；`base` 保留 gbw/densities 并调用 `orca_vpot`，`fd K ±1` 为对第 K 个赝电荷电荷 ±0.05 e 的单点 → `vpot/`、`vpot_*.log`。第一次 base 未拷 `qm.densitiesinfo`，记录保留为 `vpot/base_attempt1_missing_densitiesinfo.json`；第二次脚本中的密度名 `scfp` 不对，正确的名字 `qm.scfp` 在 `vpot/base_files/` 中手工运行得到 `pot_qm.scfp.out`。

```bash
PSCF=/home/ruigengji/miniforge3/envs/pyscf-env/bin/python
PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
cd experiments/2026-11-07_s3_v3_prep
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 $PSCF -u check_classical.py > check_classical.log 2>&1
export OPI_ORCA=/home/kasuga/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2 TMPDIR=/dev/shm PYSCF_TMPDIR=/dev/shm
(export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8; for f in 0 4 8; do $PSCF -u check_taylor_energy.py $f --threads 8 > energy_$f.log 2>&1 & done; wait)
(export OMP_NUM_THREADS=1; $PY -u check_vpot.py base > vpot_base.log 2>&1 &
 for k in 0 17 33; do for s in +1 -1; do $PY -u check_vpot.py fd $k $s > vpot_fd_${k}_${s}.log 2>&1 & done; done; wait)
cd vpot/base_files && $OPI_ORCA/orca_vpot qm.gbw qm.scfp points.xyz pot_qm.scfp.out
```

2026-11-07 在 181 本机运行。
