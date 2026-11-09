#!/bin/zsh
# B4 S2: DhlA region A, whole-box minimum-image embedding (--cutoff none), r2SCAN-3c, 2000 steps (1 ps), CUDA.
# Smoke on 181 (md_a_s2_smoke.sh, 2026-10-28): 10 steps, 8.1 s/step at np32 -> ~4.5 h for 2000 steps on 181.
# Resumable: rerunning this script continues from the last checkpoint (every 100 steps).
# 提交：qsub md_a_s2.pbs.zsh
# ← -q：walltime > 72h 改成 week
#PBS -q default
# ← select：yayoi42（i7-12700K 12 核/20 线程（8P+4E）+ RTX 4080 Super，128 GB；用户 2026-10-28 指定）。
# mpiprocs=12：作业 19797 没写，ORCA 的 mpirun -np 12 报 not enough slots（推测 PBS 默认只给 1 个 MPI slot，未核实）
#PBS -l select=1:ncpus=20:mpiprocs=12:ngpus=1:mem=60gb:host=yayoi42
# ← walltime：预计 4.5 h（181 实测），留余量
#PBS -l walltime=12:00:00
#PBS -j oe
# ← -N：作业名
#PBS -N b4_s2_dhla_A
test $PBS_O_WORKDIR && cd $PBS_O_WORKDIR
. /home/apps/Modules/init/profile.sh
export MODULEPATH=/home/ruigengji/modulefiles:$MODULEPATH

export MAMBA_EXE=/home/ruigengji/miniforge3/bin/mamba
export MAMBA_ROOT_PREFIX=/home/ruigengji/miniforge3
source /home/ruigengji/miniforge3/etc/profile.d/mamba.sh
mamba activate openmm_dev                                  # ← 环境名

cd /home/ruigengji/openmm-ORCA/experiments/2026-10-28_b4_s2_impl   # ← 项目目录
# ← ORCA 用 MPI 进程并行（12 个 rank = 12700K 物理核数），OpenMP 必须是 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python    # ← 和环境名一起改
echo "=== $(date '+%F %T')  host $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ← ORCA 用集群模块（自动加载 gcc/13.3.0、openmpi/5.0.7_gcc13.3.0；用户 2026-10-28 说明）
module load orca/6.1.1
export OPI_ORCA=${$(whence -p orca):h} TMPDIR=/dev/shm
echo "OPI_ORCA=$OPI_ORCA  mpirun=$(whence -p mpirun)"

# 有检查点就续跑；有 run.json 但没有检查点（上次在第一个检查点前失败）就归档旧产物重新开始
RESUME=()
if [[ -f md_a_s2/run.json ]]; then
  if $P -c 'import json,sys; sys.exit(json.load(open("md_a_s2/run.json")).get("checkpoint") is None)'; then
    RESUME=(--resume checkpoint)
  else
    RESUME=(--archive-existing)
  fi
fi
echo "start mode: ${RESUME:-fresh}"
$P -u /home/ruigengji/openmm-ORCA/examples/enzyme_qmmm.py \
  --region A --method r2SCAN-3c --nprocs 12 --cutoff none --steps 2000 --checkpoint-interval 100 \
  --outdir md_a_s2 $RESUME                                  # ← 运行命令
echo "=== done $(date '+%F %T') ==="
