#!/usr/bin/env bash
# B3-2 smoke on 183: 3 MD steps of region C, r2SCAN-3c np36, CUDA; checks the copied MM cache is reused and measures s/step.
set -euo pipefail
cd /home/ruigengji/openmm-ORCA/experiments/2026-10-16_b3_region_c
export OPI_ORCA=/home/korakuen/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2 TMPDIR=/dev/shm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PRTE_MCA_hwloc_default_cpu_list=0-35
/home/ruigengji/miniforge3/envs/openmm_dev/bin/python /home/ruigengji/openmm-ORCA/examples/enzyme_qmmm.py \
  --region C --method r2SCAN-3c --nprocs 36 --steps 3 --checkpoint-interval 100 --outdir md_c_smoke
