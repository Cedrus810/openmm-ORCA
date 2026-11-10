#!/bin/zsh
# C2 contract §5: GFN2-xTB restrained windows along RC = d(C1–Cl1) − d(O–C1), region A, S2, CUDA.
# Pull: 16 windows (RC −0.10..+0.20 nm), 200 steps each, sequential (window 0 already done on 181).
# Prod: all 16 windows in parallel, 400 eq + 1000 steps, frame every 50 steps (20 frames per window).
# Smoke on 181 (2026-11-01): pull 0.78 s/step, prod 1 s/step. Resumable: rerun skips finished windows.
# 提交：qsub c2_scan.pbs.zsh
# ← -q：walltime > 72h 改成 week
#PBS -q default
# ← select：不指定 host；xTB 在 ORCA 里是单线程（OMP_NUM_THREADS=1,1），16 个窗口各占约 1 核，共用 1 块 GPU
#PBS -l select=1:ncpus=20:ngpus=1:mem=60gb
# ← walltime：pull 约 15×160 s ≈ 40 min，prod 约 1400 步 ×（1–1.5 s）≈ 25–35 min，留余量
#PBS -l walltime=04:00:00
#PBS -j oe
# ← -N：作业名
#PBS -N c2_rc_scan
test $PBS_O_WORKDIR && cd $PBS_O_WORKDIR
. /home/apps/Modules/init/profile.sh
export MODULEPATH=/home/ruigengji/modulefiles:$MODULEPATH

export MAMBA_EXE=/home/ruigengji/miniforge3/bin/mamba
export MAMBA_ROOT_PREFIX=/home/ruigengji/miniforge3
source /home/ruigengji/miniforge3/etc/profile.d/mamba.sh
mamba activate openmm_dev                                  # ← 环境名

cd /home/ruigengji/openmm-ORCA/experiments/2026-11-01_c2_rc_scan   # ← 项目目录
# ← xTB 单线程；每个进程 1 线程
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python    # ← 和环境名一起改
echo "=== $(date '+%F %T')  host $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ← ORCA 用集群模块（用户 2026-10-28 说明）
module load orca/6.1.1
export OPI_ORCA=${$(whence -p orca):h} TMPDIR=/dev/shm
echo "OPI_ORCA=$OPI_ORCA"

RUN=/home/ruigengji/openmm-ORCA/experiments/2026-10-28_b4_s2_impl/md_a_s2
$P -u rc_scan.py $RUN scan --mode pull                     # ← 运行命令（pull，串行）
pids=()
for w in {0..15}; do
  $P -u rc_scan.py $RUN scan --mode prod --windows $w > scan/prod_w$(printf %02d $w).log 2>&1 &
  pids+=($!)
done
wait $pids
grep -h "^window" scan/prod_w*.log
echo "=== done $(date '+%F %T') ==="
