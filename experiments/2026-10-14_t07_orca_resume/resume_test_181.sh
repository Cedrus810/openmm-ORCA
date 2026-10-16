#!/usr/bin/env bash
# T07 遗留：ORCA + CUDA 上的中断续跑实测（181，CUDA，r2SCAN-3c）。
# A：不中断跑 STEPS 步。B：同一起点，跑过第一个 checkpoint 后 kill -9，再 --resume checkpoint 到 STEPS。
# A、B 同时跑，各 16 核（A: 4-19，B: 20-35），nprocs 相同，便于逐步对照。
# 复用 2026-10-05 正式运行（181，CUDA）的 prepared/equilibrated 缓存（manifest 校验合法复用）。
# 可续跑：A/B 目录已有产物时直接 --resume checkpoint 接着跑。
set -euo pipefail
cd /home/ruigengji/openmm-ORCA
E=experiments/2026-10-14_t07_orca_resume/cuda181
SRC=/home/kasuga/openmm-orca-runs/dhla_r2scan3c_np32_2026-10-05
mkdir -p $E
STEPS=${STEPS:-60} CKPT=${CKPT:-20} KILL_AFTER=${KILL_AFTER:-30}
PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
# 181 的 /tmp 是 tmpfs，ORCA scratch 默认就在内存里；不改 TMPDIR：OpenMM CUDA 的 TempDirectory
# 属于平衡缓存身份（10/05 记为 /tmp），改了会被 CacheMismatchError 拒跑。
export OPI_ORCA=/home/kasuga/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENMM_CPU_THREADS=4
common=(examples/enzyme_qmmm.py --method r2SCAN-3c --nprocs 16 --platform CUDA --checkpoint-interval "$CKPT")
for d in A B; do
  mkdir -p $E/$d
  for f in prepared.pdb prepared.pdb.manifest.json protein_xtal.pdb equilibrated.xml equilibrated.xml.manifest.json; do
    [ -e $E/$d/$f ] || cp $SRC/$f $E/$d/
  done
done
resume_flag() { [ -e "$1/run.json" ] && echo "--resume checkpoint" || true; }

PRTE_MCA_hwloc_default_cpu_list=4-19 $PY "${common[@]}" --steps "$STEPS" --outdir $E/A $(resume_flag $E/A) > $E/A.log 2>&1 &
pa=$!; echo "A pid $pa" | tee -a $E/pids.txt

if [ ! -e $E/B/run.json ]; then
  PRTE_MCA_hwloc_default_cpu_list=20-35 $PY "${common[@]}" --steps "$STEPS" --outdir $E/B > $E/B.attempt1.log 2>&1 &
  pb=$!; echo "B pid $pb (attempt 1, will be killed)" | tee -a $E/pids.txt
  until [ -s $E/B/steps.csv ] && [ "$(($(wc -l < $E/B/steps.csv) - 1))" -ge "$KILL_AFTER" ]; do
    kill -0 $pb 2>/dev/null || { echo "B attempt 1 exited before step $KILL_AFTER"; exit 1; }
    sleep 2
  done
  ps -o pid,etime,cmd -p $pb
  kill -9 $pb; wait $pb || true
  echo "killed B at $(($(wc -l < $E/B/steps.csv) - 1)) rows" | tee -a $E/B.attempt1.log
fi
PRTE_MCA_hwloc_default_cpu_list=20-35 $PY "${common[@]}" --steps "$STEPS" --outdir $E/B --resume checkpoint > $E/B.resume.log 2>&1 &
pr=$!; echo "B pid $pr (resume)" | tee -a $E/pids.txt
wait $pa $pr
echo done
