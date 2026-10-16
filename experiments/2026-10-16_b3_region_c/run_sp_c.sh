#!/usr/bin/env bash
# B3-1a: region C single points on the 20 T09 frames, cutoff 1.2 nm, r2SCAN-3c; 181, two workers np18 (one per NUMA node).
# Resumable: rerunning skips finished frames (per-worker outdir sp_c/w0, sp_c/w1).
set -euo pipefail
cd /home/ruigengji/openmm-ORCA/experiments/2026-10-16_b3_region_c
T09=/home/ruigengji/openmm-ORCA/experiments/2026-10-14_t09_cutoff_scan
export OPI_ORCA=/home/kasuga/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2 TMPDIR=/dev/shm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENMM_CPU_THREADS=1
PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
PRTE_MCA_hwloc_default_cpu_list=0-17 $PY $T09/cutoff_scan.py $T09/run_files $T09/frames.npz sp_c/w0 --cutoffs 1.2 --region C --nprocs 18 --frames 0 2 4 6 8 10 12 14 16 18 > sp_c_w0.log 2>&1 &
p0=$!
PRTE_MCA_hwloc_default_cpu_list=20-37 $PY $T09/cutoff_scan.py $T09/run_files $T09/frames.npz sp_c/w1 --cutoffs 1.2 --region C --nprocs 18 --frames 1 3 5 7 9 11 13 15 17 19 > sp_c_w1.log 2>&1 &
p1=$!
echo "$(date -Is) run_sp_c.sh $$: w0 pid $p0, w1 pid $p1" >> pids.txt
wait $p0 $p1
echo "done $(date -Is)"
