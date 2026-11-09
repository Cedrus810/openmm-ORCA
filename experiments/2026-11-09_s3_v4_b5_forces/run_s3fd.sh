#!/usr/bin/env bash
# V4 diagnostic on 183: S3 energy differences on frame 0 (4 SCF, 4 workers x 8 threads).
set -euo pipefail
cd /home/ruigengji/openmm-ORCA/experiments/2026-11-09_s3_v4_b5_forces
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYSCF_TMPDIR=/dev/shm
PY=/home/korakuen/micromamba/envs/pyscf-env/bin/python
mkdir -p logs
for w in 0 1 2 3; do $PY -u v4_pyscf.py --worker $w --nworkers 4 --threads 8 --s3fd > logs/s3fd_$w.log 2>&1 & done
wait
echo "ALL DONE $(date -Iseconds)"
