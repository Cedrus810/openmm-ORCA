#!/usr/bin/env bash
# B2 step 2: 740 single points (r2SCAN-3c), two workers np18, one per NUMA node of 181 (cores 0-17, 20-37).
# Resumable: rerunning skips jobs already in sp/sp_w*.npz.
set -euo pipefail
cd /home/ruigengji/openmm-ORCA/experiments/2026-10-15_b2_crossing_events
export OPI_ORCA=/home/kasuga/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2 TMPDIR=/dev/shm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENMM_CPU_THREADS=1
PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
RUN=/home/kasuga/openmm-orca-runs/dhla_r2scan3c_np32_2026-10-05
PRTE_MCA_hwloc_default_cpu_list=0-17 $PY b2_singlepoints.py jobs.json frames.npz $RUN sp --worker 0 --n-workers 2 --nprocs 18 > sp_w0.log 2>&1 &
p0=$!
PRTE_MCA_hwloc_default_cpu_list=20-37 $PY b2_singlepoints.py jobs.json frames.npz $RUN sp --worker 1 --n-workers 2 --nprocs 18 > sp_w1.log 2>&1 &
p1=$!
echo "$(date -Is) run_sp.sh $$: worker0 pid $p0, worker1 pid $p1" >> pids.txt
wait $p0 $p1
echo "done $(date -Is)"
