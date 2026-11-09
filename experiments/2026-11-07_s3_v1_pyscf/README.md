# S3 提案 V1：PySCF 原型（首批 3 帧）

依据 [S3 提案](../../docs/plans/2026-11-07-s3-ewald-correction-proposal.md) §4 V1；结果见 [验证记录](../../docs/validation/2026-11-07-s3-v1-pyscf.md)。不改 `openmmorca/`。

- `s3_v1.py`：对一个首批帧，用 `../2026-11-07_c2_error_decomp/classical.py` 计算 `dV = V_Ewald − V_S2` 并拟合 1–3 阶多项式；PySCF r2SCAN/def2-SVP 依次算 S2、S3_o1/o2/o3（多项式势直接加进单电子哈密顿量与核能）、S3_P2（2 阶多项式由球面赝电荷表示，追加到 S2 点电荷）。比较 `E − E_ENV20`（ENV20 取首批结果）。
- `pseudo_conditioning.py`：纯经典，扫描赝电荷数 K、球半径 R_s、岭回归系数 λ 下的电荷大小与拟合误差，输出 `pseudo_conditioning.json`。

```bash
PSCF=/home/ruigengji/miniforge3/envs/pyscf-env/bin/python
D=experiments/2026-11-07_s3_v1_pyscf
P=experiments/2026-11-06_c2_matched_pilot
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYSCF_TMPDIR=/dev/shm
for f in 0 4 8; do $PSCF -u $D/s3_v1.py $P/inputs/requests_A.npz $P/pyscf $D/out $f --threads 8 > $D/frame_$f.log 2>&1 & done; wait
(cd $D && $PSCF pseudo_conditioning.py)
for f in 0 4 8; do $PSCF -u $D/s3_v1.py $P/inputs/requests_A.npz $P/pyscf $D/out $f --threads 8 --K 50 --Rs 2.0 --lam 1e-4 --only-pseudo > $D/frame_${f}_P50.log 2>&1 & done; wait
```

2026-11-07 在 181 本机运行，三帧并行、每帧 8 线程；每个 SCF 12–14 s，第一组每帧约 70 s。输出 `out/candidate_XX.json`、`out/candidate_XX_P_K50_R2.0_lam0.0001.json`、`frame_*.log`。第一组的 S3_P2 为 K = 100、R_s = 2.0 nm、不加正则（`--lam 0`）。
