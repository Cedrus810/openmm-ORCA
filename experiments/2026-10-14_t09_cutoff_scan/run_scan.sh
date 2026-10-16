#!/usr/bin/env bash
# T09 第 1 项：20 帧 × 7 个 cutoff，r2SCAN-3c，两个进程各 np16（183）。可续跑：重跑同一命令跳过已完成的帧。
set -euo pipefail
cd /home/ruigengji/openmm-ORCA/experiments/2026-10-14_t09_cutoff_scan
export OPI_ORCA=/home/korakuen/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2 TMPDIR=/dev/shm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENMM_CPU_THREADS=2
PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
PRTE_MCA_hwloc_default_cpu_list=4-19 $PY cutoff_scan.py run_files frames.npz scan --cutoffs 0.8 1.0 1.2 1.25 --nprocs 16 > scan_1.log 2>&1 &
p1=$!
PRTE_MCA_hwloc_default_cpu_list=20-35 $PY cutoff_scan.py run_files frames.npz scan --cutoffs 1.6 2.0 2.5 --nprocs 16 > scan_2.log 2>&1 &
p2=$!
echo "scan_1 pid $p1, scan_2 pid $p2" >> pids.txt
wait $p1 $p2
echo done
