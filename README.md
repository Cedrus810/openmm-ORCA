# openmmorca

English | [简体中文](README.zh-CN.md)

OpenMM-driven QM/MM: **OpenMM runs the MD** (force field, integrator, thermostat/barostat, trajectories) while **ORCA computes the QM region** (electronic structure + electrostatic-embedding gradients), with OPI (ORCA Python Interface) handling ORCA input/output. The interface style follows `openmm-ml` / `openmm-pyscf`.

Design spec: `openmm_orca_opi_design_plan.md`; implementation plan: `docs/plans/2026-09-26-openmm-orca-implementation-plan.md` (both in Chinese).

**Current status (v0.2.0):** full-QM and QM/MM (electronic embedding) for non-periodic systems, plus restart and failure-bundle diagnostics, are working (M0–M3). Link atoms (v0.3) and periodic MM + cutoff embedding (v0.4) are planned.

## Installation

```bash
# Environment: mamba env openmm_dev (Python ≥ 3.10, openmm ≥ 8.5, orca-pi ≥ 2.0, numpy)
/home/ruigengji/miniforge3/envs/openmm_dev/bin/python -m pip install -e /home/ruigengji/openmm-ORCA

# ORCA path (required)
export OPI_ORCA=/home/ruigengji/ORCA611
# MPI is needed for nprocs > 1; on nodes without a system mpirun (e.g. login nodes) set:
export OPI_MPI=/home/apps/openmpi/5.0.7_gcc13.3.0
# Note: inside a PBS job, parallel ORCA requires ncpus ≥ nprocs (OpenMPI/PRRTE read the
# scheduler allocation; "Not enough slots available" means too few cores), e.g.:
#   qsub -I -l select=1:ncpus=40   # 40-core benchmark
# To oversubscribe deliberately, set OMPI_MCA_rmaps_default_mapping_policy=:oversubscribe
```

If ORCA lives on NFS, congestion multiplies the per-step startup overhead (measured here: 0.4 s → 3.5 s per step). Before running MD, copy ORCA to a local disk or tmpfs (`autoci_*` is not needed and can be skipped) and point `OPI_ORCA` at the copy:

```bash
D=/dev/shm/orca611-$USER; mkdir -p $D
cd /home/ruigengji/ORCA611 && cp -a lib datasets $D/ && ls | grep -v -e '^autoci_' -e '^lib$' -e '^datasets$' | xargs -I{} cp -a {} $D/
export OPI_ORCA=$D
```

OpenMPI's `mpirun` ignores the caller's `taskset` affinity and always binds processes starting from core 0. To pin parallel ORCA to specific cores (e.g. when sharing the machine with other jobs), set `PRTE_MCA_hwloc_default_cpu_list=4-35`.

## Minimal example

full-QM (one water molecule, 20 MD steps; see `examples/full_qm_water.py`):

```python
from openmmorca import ORCAPotential

potential = ORCAPotential(method="HF", basis="def2-SVP", extra_keywords=("TightSCF",))
system = potential.createSystem(topology)
```

QM/MM (QM = water 0, everything else stays MM, electronic embedding):

```python
potential = ORCAPotential(method="HF", basis="def2-SVP", extra_keywords=("TightSCF",))
mixed = potential.createMixedSystem(topology, system, atoms=[0, 1, 2], forceGroup=0)
```

QM/MM water-cluster NVE (1 ps, energy-conservation check): `examples/qmmm_water_cluster_nve.py`.

## Parameters (`ORCAPotential.__init__`)

| Parameter | Default | Description |
|---|---|---|
| `method` | required | ORCA simple keyword, e.g. `"HF"`, `"PBE0"`, `"XTB"`, `"r2SCAN-3c"` |
| `basis` | `None` | basis set; `None` for composite methods / xTB |
| `charge` / `multiplicity` | 0 / 1 | total QM-region charge and multiplicity (counted as "QM atoms + link H" from v0.3 on) |
| `nprocs` | 1 | >1 writes `%pal` and sets the MPI environment variables (user-set values are never overridden) |
| `maxcore_mb` | 2000 | memory per MPI process (ORCA `%maxcore` semantics) |
| `extra_keywords` | `()` | simple keywords appended verbatim (`RIJCOSX`, `def2/J`, `TightSCF`…); task-type keywords (`SP`/`Opt`/`EnGrad`…) are rejected |
| `extra_blocks` | `()` | `% block` strings appended verbatim; `%pointcharges`/`%pal`/`%maxcore`/`%moinp`/`%output` are rejected (managed by the backend) |
| `scratch_root` | `None` | default `/tmp/openmmorca-<user>` (tmpfs) |
| `restart` | `True` | use the `.gbw` file as an MORead guess (automatic fallback to one fresh SCF on failure) |
| `keep_failed` | `True` | keep the failure bundle when a step fails |
| `max_fresh_retries` | 1 | number of fresh-SCF retries after a failed MORead attempt |
| `timeout_s` | `None` | timeout for a single ORCA call (seconds) |
| `orca_path` | `None` | by default reads `OPI_ORCA` or `PATH` |
| `backend_factory` | `None` | inject a custom backend (for tests) |

**Every `createSystem` / `createMixedSystem` call creates a fresh backend instance with its own scratch directory**; one backend serves exactly one Context.

## Scratch directory and failure bundles

```text
<scratch_root>/<uuid>/
├── current/     overwritten every step: qm.inp qm.out qm.engrad qm.pcgrad qm.gbw pc.pc guess.gbw qm.property.json
├── restart/     last_good.gbw (MORead guess)
├── failures/    failure_step_NNNNNN/ ← captured failure
└── timings.csv  per-step timings (write/orca/read/total, SCF cycles, restart usage)
```

On failure an `ORCACalculationError` / `ORCAOutputError` / `ORCATimeoutError` is raised, with the failure-bundle path appended to the message. Reproduce with:

```bash
cd <scratch>/failures/failure_step_000123 && $OPI_ORCA/orca qm.inp > rerun.out
```

`ORCAPotential.summarize_timings()` returns mean/P50/P95 timings per backend (the cold-start first step is skipped automatically).

## Known limitations (before v0.4)

- Non-periodic systems, whole molecules as the QM region (periodic + cutoff embedding planned for M5); QM/MM boundaries needing link atoms arrive in M4.
- Systems containing the PythonForce cannot be XML-serialized (the callback holds a lock and a temp directory).
- Every step pays a fixed process-startup overhead (≈0.4 s serial); not negligible for small QM regions.
- Embedding charges go through a `%pointcharges` file; inline `Q` is forbidden (it double-counts MM–MM electrostatics and yields no pcgrad).
- Stale forces are never returned on SCF failure: when retries fail, it hard-errors and stops the MD.

## Running the tests

```bash
export OPI_ORCA=/home/ruigengji/ORCA611
$PY -m pytest                    # default: unit tests + fast ORCA tests (no slow)
$PY -m pytest -m orca -v         # only tests that need ORCA
$PY -m pytest -m slow -v         # NVE (~25 min), restart reproducibility and other long tests
```

Without ORCA, `orca`-marked tests are skipped automatically (note: on desktop Linux `/usr/bin/orca` may be the screen reader; conftest only accepts an ELF binary).

## Acknowledgments

- [openmm-pyscf](https://github.com/Gallicchio-Lab/openmm-pyscf) (Gallicchio Lab): the OpenMM-side system-modification logic of this project was ported from it, with fixes to its bonded-term removal rule and its O(N²) exception handling. Many thanks to Gallicchio Lab.
