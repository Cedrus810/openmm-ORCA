#!/usr/bin/env bash
# T08 on 183: PySCF (pyscf-env) vs ORCA 6.1.1 (local). Serial, seconds.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OPI_ORCA=/home/korakuen/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2 TMPDIR=/dev/shm
/home/korakuen/micromamba/envs/pyscf-env/bin/python pyscf_side.py pyscf.json
PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
$PY orca_side.py orca_default.json orca_default
$PY orca_side.py orca_verytight_norijcosx.json orca_verytight_norijcosx VeryTightSCF NoRIJCOSX
$PY compare.py orca_default.json orca_verytight_norijcosx.json | tee comparison.md
