#!/usr/bin/env bash
# B4 S2 smoke on 181: region A, whole-box embedding (--cutoff none), r2SCAN-3c np32 (cores 4-35), CUDA, 10 steps.
# Start state = the 2026-10-05 region-A run (prepared.pdb / equilibrated.xml copied from B3 md_c, same files).
set -euo pipefail
cd /home/ruigengji/openmm-ORCA/experiments/2026-10-28_b4_s2_impl
export OPI_ORCA=/home/kasuga/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2 TMPDIR=/dev/shm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PRTE_MCA_hwloc_default_cpu_list=4-35
/home/ruigengji/miniforge3/envs/openmm_dev/bin/python /home/ruigengji/openmm-ORCA/examples/enzyme_qmmm.py \
  --region A --method r2SCAN-3c --nprocs 32 --cutoff none --steps 10 --checkpoint-interval 10 --outdir md_a_s2_smoke
