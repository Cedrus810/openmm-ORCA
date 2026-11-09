#!/usr/bin/env bash
# V5 on 183. Stage 1: capture 5 windows (5 processes, 1 thread). Stage 2: 10 PySCF workers x 4 threads = 40 cores.
# SMOKE=1: only w00 frame 0 in stage 2. Resumable (existing outputs skipped).
set -euo pipefail
cd /home/ruigengji/openmm-ORCA/experiments/2026-11-09_s3_v5_c1_windows
mkdir -p logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENMM_CPU_THREADS=1
PYO=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
for w in 0 5 6 7 15; do $PYO -u capture_windows.py $w > logs/capture_$w.log 2>&1 & done
wait
echo "CAPTURE DONE $(date -Iseconds)"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYSCF_TMPDIR=/dev/shm
PY=/home/korakuen/micromamba/envs/pyscf-env/bin/python
if [ "${SMOKE:-0}" = 1 ]; then
  for w in 0 1 2; do $PY -u v5_pyscf.py --worker $w --nworkers 3 --threads 4 --windows 0 --frames 0 > logs/smoke_$w.log 2>&1 & done
else
  for w in $(seq 0 9); do $PY -u v5_pyscf.py --worker $w --nworkers 10 --threads 4 > logs/worker_$w.log 2>&1 & done
fi
wait
echo "ALL DONE $(date -Iseconds)"
