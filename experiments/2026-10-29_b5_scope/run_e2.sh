#!/usr/bin/env bash
# B5 E2 on 181: S2 single points, regions A and C, on F_A (S2 region-A run) and F_C (B3 region-C run) frames.
# Two workers np18 (cores 0-17 / 20-37). Resumable: rerunning skips finished frames.
set -euo pipefail
cd /home/ruigengji/openmm-ORCA/experiments/2026-10-29_b5_scope
RUN=/home/ruigengji/openmm-ORCA/experiments/2026-10-28_b4_s2_impl/md_a_s2
export OPI_ORCA=/home/kasuga/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2 TMPDIR=/dev/shm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENMM_CPU_THREADS=1
PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
( export PRTE_MCA_hwloc_default_cpu_list=0-17
  $PY b5_sp.py $RUN frames_a_s2.npz sp_fa --region A && $PY b5_sp.py $RUN frames_a_s2.npz sp_fa --region C ) > e2_w0.log 2>&1 &
p0=$!
( export PRTE_MCA_hwloc_default_cpu_list=20-37
  $PY b5_sp.py $RUN frames_c_s0.npz sp_fc --region A && $PY b5_sp.py $RUN frames_c_s0.npz sp_fc --region C ) > e2_w1.log 2>&1 &
p1=$!
echo "$(date -Is) run_e2.sh $$: w0 pid $p0, w1 pid $p1" >> pids.txt
wait $p0 $p1
echo "done $(date -Is)"
