#!/usr/bin/env bash
# V4 on 183: 5 PySCF workers x 8 threads. FRAMES env selects frames (default all 20). Resumable.
set -euo pipefail
cd /home/ruigengji/openmm-ORCA/experiments/2026-11-09_s3_v4_b5_forces
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYSCF_TMPDIR=/dev/shm
PY=/home/korakuen/micromamba/envs/pyscf-env/bin/python
FRAMES=${FRAMES:-$(seq -s ' ' 0 19)}
mkdir -p logs
for w in 0 1 2 3 4; do
  $PY -u v4_pyscf.py --worker $w --nworkers 5 --threads 8 --frames $FRAMES > logs/worker_$w.log 2>&1 &
done
wait
echo "ALL DONE $(date -Iseconds)"
