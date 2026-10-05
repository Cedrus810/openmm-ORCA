# openmm-orca

English | [简体中文](README.zh-CN.md)

OpenMM-driven QM/MM: **OpenMM runs the MD** (force field, integrator, thermostat/barostat, trajectories) while **ORCA computes the QM region** (electronic structure + electrostatic-embedding gradients), with OPI (ORCA Python Interface) handling ORCA input/output. The interface style follows `openmm-ml` / `openmm-pyscf`.

Changes per version: [CHANGELOG.md](CHANGELOG.md). Design spec: `openmm_orca_opi_design_plan.md`; implementation plan: `docs/plans/2026-09-26-openmm-orca-implementation-plan.md`; ONIOM plan: `docs/plans/2026-09-28-oniom.md` (all in Chinese).

Follow-up fixes, validation and feature extensions: [TODO list](TODO.md) (Chinese; includes priorities, evidence and acceptance criteria).

**Current status (v0.4.0):** full-QM, QM/MM (electronic embedding, covalent boundaries with H link atoms, periodic MM with approximate cutoff embedding) and two-layer ONIOM (QM:QM, non-periodic), plus restart and failure-bundle diagnostics, are working (M0–M5 + ONIOM). Validated on a solvated enzyme (DhlA, 31,610 atoms, `examples/enzyme_qmmm.py`).

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

Covalent QM/MM boundaries with H link atoms (v0.3; finite differences through the link atom agree with ORCA analytic forces to < 0.05 kJ/mol/nm, see `examples/link_atom_dipeptide.py`). Declare every cut bond as `(q1, m1)` with q1 in the QM region and m1 in MM; each gets a hydrogen cap at `R_q1 + g (R_m1 − R_q1)` whose force is split onto q1/m1 by the chain rule:

```python
mixed = potential.createMixedSystem(
    topology, system, atoms=qm_atoms,
    boundaryPairs=[(cb, ca)],      # e.g. cut an amino-acid side chain at CB–CA
    linkRatios=None,               # default g for C–C (1.09/1.526) and C–N (1.09/1.449)
)
```

`charge`/`multiplicity` then describe the QM atoms plus link H. Single bonds only, one link per q1. The m1 charge is removed from the embedding and spread evenly over its MM neighbours (a *simplified* charge shift — no dipole-restoring point-charge pairs as in the literature scheme); OpenMM's MM–MM electrostatics are unchanged. A QM region cut out of force-field residues usually carries a non-integer force-field charge x per cut residue; the residual x − round(x) is added to that residue's M2 atoms, so every cut residue's MM remainder — and the whole embedding — carries an integer charge. A warning is issued when a residue's QM part is close to a half-integer charge (ambiguous rounding) or when `charge` differs from the charge the force field implies for the QM region.

Periodic QM/MM with cutoff embedding (v0.4; enzyme example `examples/enzyme_qmmm.py`). A periodic System (NonbondedForce with PME, LJPME or Ewald; `CutoffPeriodic` is rejected) switches the callback to cutoff embedding: each step the QM region (and any link-atom m1) is re-imaged into one piece, and only the MM residues with an atom within `embeddingCutoff` of a QM atom are embedded, whole, at their nearest image:

```python
mixed = potential.createMixedSystem(topology, system, atoms=qm_atoms, embeddingCutoff=1.2 * unit.nanometer)
```

**This is an approximation**: QM–MM electrostatics beyond the cutoff are neglected (not PME-consistent), and residues crossing the cutoff make the energy discontinuous, so strict NVE energy conservation is not expected. `embeddingCutoff` plus the QM region's extent must stay below half the narrowest box width (checked every step). The timing log records `n_embed_groups` and `embed_changed` per step.

ONIOM (two-layer subtractive QM:QM; here HF/STO-3G on water 0, xTB on all 5 waters — see `examples/oniom_water_cluster.py`):

```python
from openmmorca import ONIOMPotential, ORCAPotential

oniom = ONIOMPotential(
    high=ORCAPotential(method="HF", basis="STO-3G", extra_keywords=("TightSCF",)),  # model region
    low=ORCAPotential(method="XTB"),                                                # full system
)
system = oniom.createONIOMSystem(topology, atoms=[0, 1, 2], forceGroup=0)
```

The energy is the standard subtractive combination `E_high(model) + E_low(full) − E_low(model)`; layers couple mechanically (no point-charge embedding between QM layers), the model region must be whole molecules, and the System carries no force-field terms. Each step costs three QM evaluations (one high, two low) through three independent backends, so every restart chain stays correctly sized. `high`'s charge/multiplicity describe the model region, `low`'s the full system; the low-level model-region evaluation uses `high`'s charge/multiplicity, so the low-layer-only atoms may be charged or open-shell. Every `createONIOMSystem` call creates those three backends; `oniom.summarize_timings()` / `oniom.close()` aggregate over both layers.

QM/MM water-cluster NVE (1 ps, energy-conservation check): `examples/qmmm_water_cluster_nve.py`; ONIOM counterpart: `examples/oniom_water_cluster.py`; link-atom dipeptide NVT: `examples/link_atom_dipeptide.py`.

Enzyme runs shorter than 2000 steps (1 ps) report smoke checks and `Task 22 acceptance: NOT_EVALUATED`; a single step is supported. Repeat `--reaction-bond I J` to monitor reactive bonds separately using zero-based OpenMM atom indices. Configured stability checks use the remaining QM bonds; the original Task 22 all-bond gate is still reported separately. The exit code follows the smoke/configured stability checks; full acceptance is recorded in `acceptance.json`. Each `steps.csv` row is flushed during the run; failures retain completed rows and close the backend.

Preparation/equilibration caches now require matching `.manifest.json` sidecars, checking stage settings, input/artifact hashes, versions, atom mapping and box information. Legacy or mismatched caches are preserved and rejected; use a new `--outdir` to rebuild. Later QM method/region changes do not invalidate MM cache identity. Later QM/MM runs never overwrite an earlier run silently. An output directory holding run products (`run.json`, `steps.csv`, trajectories, `checkpoints/`, final State) is refused unless `--archive-existing` moves them to `archive/run-<UTC time>/`.

`examples/analyze_enzyme_run.py OUTDIR [--json report.json]` recomputes a finished run from its products: length, temperature, bond deviations, embedding changes, SCF cycles/retries and timings. It also recomputes the QM bond deviations independently from the DCD frames (needs mdtraj) and checks them against `steps.csv`. It compares the recomputed Task 22 verdict with `acceptance.json`, includes the provenance from `run.json` (versions, host, command, git commit), and exits with 1 on any inconsistency.

Checkpoints and resume: every `--checkpoint-interval` steps (default 100, a multiple of the 10-step DCD interval) and at the last step, the run writes an OpenMM checkpoint and a portable State XML. These are committed in `run.json` with the `steps.csv` length and hash. `run.json` also records the status (running / completed / failed) and every attempt, and a completed run writes `final_state.xml` and `final.pdb`. `--resume checkpoint` continues in a new process after checking the run identity, and restores the Langevin random-number state; it requires the same platform and OpenMM version. With a deterministic backend it reproduces the uninterrupted trajectory exactly. With ORCA, the first resumed step uses a fresh SCF guess, so agreement holds only within SCF convergence. `--resume state` restarts from the State with a new seed and does not reproduce the trajectory. The checkpointed `steps.csv` prefix is kept byte for byte; rows recomputed after the checkpoint are kept in `steps.superseded.attempt-NNN.csv`. Each attempt writes its own trajectory file, and a larger `--steps` extends a completed run.

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

**Every `createSystem` / `createMixedSystem` / `createONIOMSystem` call creates fresh backend instance(s) with their own scratch directory**; one backend serves exactly one Context.

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

## Known limitations

- Periodic systems only through the approximate cutoff embedding above; ONIOM is non-periodic. QM/MM regions may cut single bonds via link atoms (see above); the ONIOM model region must still be whole molecules. ONIOM layers couple mechanically (no embedding between QM layers).
- Systems containing the PythonForce cannot be XML-serialized (the callback holds a lock and a temp directory).
- Every step pays a fixed process-startup overhead (≈0.4 s serial); not negligible for small QM regions.
- Under NPT, each `MonteCarloBarostat` attempt triggers two extra QM evaluations (+8% QM cost at the default frequency of 25).
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
