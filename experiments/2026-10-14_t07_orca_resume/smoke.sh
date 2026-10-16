#!/usr/bin/env bash
# T07 遗留：ORCA 上的中断续跑。冒烟：183，CPU 平台，r2SCAN-3c np32，20 步，测单步耗时。
set -euo pipefail
cd /home/ruigengji/openmm-ORCA
export OPI_ORCA=/home/korakuen/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2
export TMPDIR=/dev/shm            # ORCA scratch -> /dev/shm/openmmorca-$USER/<uuid>
export PRTE_MCA_hwloc_default_cpu_list=4-35
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENMM_CPU_THREADS=32
/home/ruigengji/miniforge3/envs/openmm_dev/bin/python examples/enzyme_qmmm.py \
  --method r2SCAN-3c --nprocs 32 --platform CPU --steps 20 --checkpoint-interval 10 \
  --outdir experiments/2026-10-14_t07_orca_resume/smoke
