#!/usr/bin/env bash
# ORCA bridge on 183: NWORKERS workers x ORCA nprocs 1 (default 20: another job holds ~20 cores on 183,
# 2026-11-09 17:20). SMOKE=1: only w00 frame 0. Resumable.
set -euo pipefail
cd /home/ruigengji/openmm-ORCA/experiments/2026-11-09_s3_orca_bridge_windows
export OPI_ORCA=/home/korakuen/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2
export TMPDIR=/dev/shm OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
N=${NWORKERS:-20}
mkdir -p logs
if [ "${SMOKE:-0}" = 1 ]; then
  for w in 0 1; do $PY -u orca_bridge.py --worker $w --nworkers 2 --windows 0 --frames 0 > logs/smoke_$w.log 2>&1 & done
else
  for w in $(seq 0 $((N - 1))); do $PY -u orca_bridge.py --worker $w --nworkers $N > logs/worker_$w.log 2>&1 & done
fi
wait
echo "ALL DONE $(date -Iseconds)"
