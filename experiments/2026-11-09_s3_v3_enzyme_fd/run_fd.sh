#!/usr/bin/env bash
# V3 Task 4 on 183: base + 24 central-difference single points in parallel (ORCA nprocs 1 each).
set -euo pipefail
cd /home/ruigengji/openmm-ORCA/experiments/2026-11-09_s3_v3_enzyme_fd
export OPI_ORCA=/home/korakuen/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2
export TMPDIR=/dev/shm OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENMM_CPU_THREADS=1
PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
mkdir -p logs
$PY -u fd_enzyme.py base > logs/base.log 2>&1 &
for atom in C1 Cl1 O Q1 M1 farO; do
  for axis in 0 1; do
    for sign in 1 -1; do
      $PY -u fd_enzyme.py fd $atom $axis $sign > logs/fd_${atom}_${axis}_${sign}.log 2>&1 &
    done
  done
done
wait
echo "ALL DONE $(date -Iseconds)"
