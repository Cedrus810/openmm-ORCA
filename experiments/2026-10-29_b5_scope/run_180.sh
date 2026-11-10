#!/usr/bin/env bash
# B5 on 180 (i9-9900K, 8 cores), moved from 181/183 on user request 2026-10-29. Resumable (finished frames skipped).
#   E2: region C S2 single points on F_A then F_C frames, ORCA np4 on cores 0-3.
#   E1: PySCF r2SCAN/def2-SVP S2/EW20/S0 on T09 frames 0-19 (+ EW30 on 0 10 19), 4 threads on cores 4-7.
set -euo pipefail
cd /home/ruigengji/openmm-ORCA/experiments/2026-10-29_b5_scope
RUN=/home/ruigengji/openmm-ORCA/experiments/2026-10-28_b4_s2_impl/md_a_s2
PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
PYSCF=/home/ruigengji/miniforge3/envs/pyscf-env/bin/python
( export OPI_ORCA=/home/canna/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2 TMPDIR=/dev/shm
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENMM_CPU_THREADS=1 PRTE_MCA_hwloc_default_cpu_list=0-3
  $PY b5_sp.py $RUN frames_a_s2.npz sp_fa --region C --nprocs 4 && $PY b5_sp.py $RUN frames_c_s0.npz sp_fc --region C --nprocs 4
) > e2_180.log 2>&1 &
p0=$!
( export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYSCF_TMPDIR=/dev/shm
  taskset -c 4-7 $PYSCF pyscf_e1.py requests_t09_A.npz pyscf_e1 --frames $(seq 0 19) --schemes S2 EW20 S0
  taskset -c 4-7 $PYSCF pyscf_e1.py requests_t09_A.npz pyscf_e1 --frames 0 10 19 --schemes EW30
) > e1_180.log 2>&1 &
p1=$!
echo "$(date -Is) 180 run_180.sh $$: e2 pid $p0, e1 pid $p1" >> pids.txt
wait $p0 $p1
echo "done $(date -Is)"
