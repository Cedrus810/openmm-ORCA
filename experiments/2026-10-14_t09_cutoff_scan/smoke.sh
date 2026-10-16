#!/usr/bin/env bash
# T09 冒烟：1 帧 × cutoff 1.2 / 2.5 nm，r2SCAN-3c np32，测单点耗时（183，CPU 平台只算 QM 力组）。
set -euo pipefail
cd /home/ruigengji/openmm-ORCA/experiments/2026-10-14_t09_cutoff_scan
export OPI_ORCA=/home/korakuen/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2 TMPDIR=/dev/shm
export PRTE_MCA_hwloc_default_cpu_list=4-35 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENMM_CPU_THREADS=4
/home/ruigengji/miniforge3/envs/openmm_dev/bin/python cutoff_scan.py run_files frames.npz smoke \
  --cutoffs 1.2 2.5 --frames 0 --method r2SCAN-3c --nprocs 32
