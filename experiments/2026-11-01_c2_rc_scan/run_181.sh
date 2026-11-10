#!/usr/bin/env bash
# C2 window scan on 181 after hours (user 2026-11-01: "181 跑起来我下班了？"). GFN2-xTB (single-threaded in ORCA),
# region A, S2, CUDA (2080 Ti). Pull: remaining windows sequential, 200 steps each. Prod: 16 windows in parallel,
# 400 eq + 1000 steps, frame every 50 steps. Resumable: rerun skips finished windows.
set -euo pipefail
cd /home/ruigengji/openmm-ORCA/experiments/2026-11-01_c2_rc_scan
export OPI_ORCA=/home/kasuga/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2 TMPDIR=/dev/shm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
RUN=/home/ruigengji/openmm-ORCA/experiments/2026-10-28_b4_s2_impl/md_a_s2
$PY -u rc_scan.py $RUN scan --mode pull
pids=()
for w in $(seq 0 15); do
  $PY -u rc_scan.py $RUN scan --mode prod --windows $w > scan/prod_w$(printf %02d $w).log 2>&1 &
  pids+=($!)
done
echo "$(date -Is) prod pids ${pids[*]}" >> pids.txt
wait "${pids[@]}"
grep -h "^window" scan/prod_w*.log
echo "done $(date -Is)"
