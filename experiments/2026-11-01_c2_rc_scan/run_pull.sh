#!/usr/bin/env bash
# C2 pull on 181: GFN2-xTB (ORCA, 8 OpenMP threads, cores 0-7), region A, S2, CUDA; 16 windows RC -0.10..+0.20 nm,
# 200 steps each, sequential. Resumable (finished windows skipped).
set -euo pipefail
cd /home/ruigengji/openmm-ORCA/experiments/2026-11-01_c2_rc_scan
export OPI_ORCA=/home/kasuga/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2 TMPDIR=/dev/shm
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
taskset -c 0-7 /home/ruigengji/miniforge3/envs/openmm_dev/bin/python rc_scan.py \
  ../2026-10-28_b4_s2_impl/md_a_s2 scan --mode pull
echo "done $(date -Is)"
