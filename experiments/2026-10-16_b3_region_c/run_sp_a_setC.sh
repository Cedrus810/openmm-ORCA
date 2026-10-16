#!/usr/bin/env bash
# B3-1b: region A with region C's embedded residues, 20 T09 frames, r2SCAN-3c; 181, two workers np18.
# --reach 2.6: groups within 1.2 nm of the C side chains can lie > 2.0 nm from region A (bound: reach + A extent < half box 3.41 nm).
# Run after B3-1a. Resumable.
set -euo pipefail
cd /home/ruigengji/openmm-ORCA/experiments/2026-10-16_b3_region_c
: # B3-1a finished
T09=/home/ruigengji/openmm-ORCA/experiments/2026-10-14_t09_cutoff_scan
B2=/home/ruigengji/openmm-ORCA/experiments/2026-10-15_b2_crossing_events
export OPI_ORCA=/home/kasuga/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2 TMPDIR=/dev/shm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENMM_CPU_THREADS=1
PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
PRTE_MCA_hwloc_default_cpu_list=0-17 $PY $B2/b2_singlepoints.py jobs_a_setC.json $T09/frames.npz $T09/run_files sp_a_setC --region A --reach 2.6 --worker 0 --n-workers 2 --nprocs 18 > sp_a_setC_w0.log 2>&1 &
p0=$!
PRTE_MCA_hwloc_default_cpu_list=20-37 $PY $B2/b2_singlepoints.py jobs_a_setC.json $T09/frames.npz $T09/run_files sp_a_setC --region A --reach 2.6 --worker 1 --n-workers 2 --nprocs 18 > sp_a_setC_w1.log 2>&1 &
p1=$!
echo "$(date -Is) run_sp_a_setC.sh $$: w0 pid $p0, w1 pid $p1" >> pids.txt
wait $p0 $p1
echo "done $(date -Is)"
