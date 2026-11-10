#!/usr/bin/env bash
# B5 E1b on 183 (pyscf-env, 40 threads): r2SCAN/def2-SVP, region A, T09 frames 0-19: S0, S2, EW20; EW30 on frames 0 10 19.
# Resumable: finished (frame, scheme) entries in pyscf_e1/frame_*.json are skipped.
set -euo pipefail
cd /home/ruigengji/openmm-ORCA/experiments/2026-10-29_b5_scope
export OMP_NUM_THREADS=40 MKL_NUM_THREADS=40 PYSCF_TMPDIR=/dev/shm
PY=/home/korakuen/micromamba/envs/pyscf-env/bin/python
$PY pyscf_e1.py requests_t09_A.npz pyscf_e1 --frames $(seq 0 19) --schemes S2 EW20 S0
$PY pyscf_e1.py requests_t09_A.npz pyscf_e1 --frames 0 10 19 --schemes EW30
echo "done $(date -Is)"
