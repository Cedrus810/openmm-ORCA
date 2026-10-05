"""Enzyme QM/MM benchmark: haloalkane dehalogenase DhlA with 1,2-dichloroethane (Task 22).

System: PDB 2DHC (DhlA–DCE Michaelis complex; 2HAD is the substrate-free
enzyme). QM region for the SN2 step (Asp124 attacking DCE C1, Cl1 leaving),
cut at side-chain CB–CA bonds with H link atoms (prep §5 in
docs/plans/2026-09-29-m4-m5-prep.md):

* region A — DCE + Asp124 side chain: 15 QM atoms incl. 1 link H, charge −1
* region C — A + Trp125, Trp175, His289 (HID) side chains: 65 atoms, 4 links, −1

Pipeline (each stage is cached in --outdir):

1. prepare: PDBFixer (pH 7, His289 forced to HID), amber14 (ff14SB) for the
   protein, OpenFF Sage 2.2 with NAGL AM1-BCC-like charges for DCE, TIP3P with
   1.0 nm padding, Na+ to neutralize.
2. MM equilibration on --platform: minimize, 20 ps NVT, 50 ps NPT (2 fs, HBonds).
3. QM/MM: PME + cutoff embedding (approximate, not PME-consistent), Langevin
   NVT (300 K, 0.5 fs) started from the MM-equilibrated positions *and
   velocities*, DCD every 10 steps. An optional QM/MM minimization
   (--min-iterations > 0) relaxes the whole solvated system and drains its
   thermal potential energy: fresh 300 K velocities then equipartition down to
   ~150 K within ~50 fs, which a 1/ps thermostat does not recover within 1 ps.

Acceptance (implementation plan Task 22): at least 2000 steps (1 ps) finish
without intervention; the last 500 temperature samples average 300 ± 10 K;
stable QM covalent bonds stay within 20 % of their starting lengths. Shorter
runs report smoke checks only. Explicit --reaction-bond pairs are monitored
separately; by default every QM covalent bond is treated as stable.

Preparation/equilibration caches require matching sidecar manifests. Legacy
or mismatched caches are preserved and rejected; use a new --outdir to rebuild.

QM/MM run products (run.json, steps.csv, trajectory*.dcd, checkpoints/,
final_state.xml, final.pdb) are never overwritten silently: a fresh run into
an outdir that holds them is refused unless --archive-existing moves them to
archive/run-<UTC time>/. Every --checkpoint-interval steps and at the last step
the run saves an OpenMM checkpoint (.chk) and a portable State (.xml), then
commits them in run.json together with the length and SHA-256 of steps.csv.
run.json records the run status (running / completed / failed) and each attempt.

--resume rebuilds the mixed System, PythonForce and a new ORCA backend in a new
process, after checking the run identity (start State, topology, MM System,
QM settings, recipe) against run.json; nprocs and platform may differ only with
--resume state. The checkpointed steps.csv prefix is kept byte for byte; rows a
failed attempt wrote after its checkpoint go to steps.superseded.attempt-NNN.csv
and are recomputed. Each attempt writes its own trajectory file, and run.json
lists the steps each one covers. --steps sets the total, so a completed run can
be extended.

* --resume checkpoint: OpenMM checkpoint, which restores positions, velocities,
  box, time, step count and the Langevin random-number state. It requires the
  same platform and OpenMM version. With a deterministic QM backend the result
  matches an uninterrupted run exactly (tested with the fake backend on
  Reference). ORCA starts the first resumed step from a fresh SCF guess (the
  last-good .gbw lives in the temporary scratch of the old backend), so real
  runs agree only within SCF convergence.
* --resume state: portable State XML (positions, velocities, box, time, step);
  a new Langevin seed (integrator_seed + attempt number) is used. This restarts
  from the State; it does not reproduce the uninterrupted trajectory.

    export OPI_ORCA=/path/to/orca
    python examples/enzyme_qmmm.py \\
        --method XTB --steps 200 --outdir dhla_xtb          # smoke test
    PRTE_MCA_hwloc_default_cpu_list=0-39 ... --method r2SCAN-3c --nprocs 40 \\
        --steps 2000 --outdir dhla_r2scan3c   # 1 ps, ~3 h (validated 2026-10-01: PASS)
    python examples/analyze_enzyme_run.py dhla_r2scan3c

DCE lies entirely in the QM region (both regions), where its force-field
charges are zeroed and its internal terms removed, so its parameters only
matter for the MM equilibration. Sage + NAGL charges need no AmberTools.
"""

from __future__ import annotations

import argparse
import csv
import errno
import fcntl
import hashlib
import io
import json
import math
import os
import platform as host_platform
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit

if __package__:
    from . import enzyme_workflow as workflow
    from .enzyme_workflow import (
        RunMismatchError, StageCache, acceptance_summary, atomic_write_bytes, atomic_write_text,
        json_digest, sha256_file,
    )
else:
    import enzyme_workflow as workflow
    from enzyme_workflow import (
        RunMismatchError, StageCache, acceptance_summary, atomic_write_bytes, atomic_write_text,
        json_digest, sha256_file,
    )

DATA = Path(__file__).resolve().parent / "data" / "2DHC.pdb"
BACKBONE = {"N", "H", "CA", "HA", "C", "O", "OXT"}
SIDE_CHAINS = {"A": (124,), "C": (124, 125, 175, 289)}

# These recipes describe only their own stage; changes to the later QM method,
# region or embedding cutoff do not invalidate MM preparation/equilibration.
# Bump recipe_version when a stage's algorithm changes.
PREPARATION_RECIPE = {
    "recipe_version": 1,
    "forcefield_files": ["amber14-all.xml", "amber14/tip3p.xml"],
    "ligand_smiles": "ClCCCl",
    "ligand_forcefield": "openff-2.2.0",
    "charge_model": "openff-gnn-am1bcc-0.1.0-rc.3.pt",
    "pH": 7.0, "histidine_289": "HID",
    "solvent_model": "tip3p", "padding_nm": 1.0,
    "neutralize": True, "positive_ion": "Na+", "negative_ion": "Cl-",
    "ionic_strength_molar": 0.0,
}
EQUILIBRATION_RECIPE = {
    "recipe_version": 1, "temperature_K": 300.0, "pressure_bar": 1.0,
    "friction_per_ps": 1.0, "step_fs": 2.0,
    "nvt_steps": 10_000, "npt_steps": 25_000,
    "velocity_seed": 2026, "integrator_seed": 2027, "barostat_seed": 2028,
    "barostat_frequency": 25, "min_tolerance_kj_mol_nm": 10.0, "min_iterations": 0,
}
# The QM/MM stage. Part of the run identity in run.json (--resume checks it).
QMMM_RECIPE = {
    "recipe_version": 1, "charge": -1, "multiplicity": 1,
    "temperature_K": 300.0, "friction_per_ps": 1.0, "step_fs": 0.5,
    "integrator_seed": 2029, "min_velocity_seed": 2026, "min_tolerance_kj_mol_nm": 10.0,
    "dcd_interval": 10,
}
DEFAULT_CHECKPOINT_INTERVAL = 100


def dependency_versions():
    versions = {"openmm": mm.__version__}
    for name in ("numpy", "pdbfixer", "rdkit", "openmmforcefields", "openff-toolkit", "openff-nagl"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def topology_digest(topology):
    """Preserve atom order and chemical connectivity, including residue IDs."""
    atoms = [
        [a.index, a.name, a.id, a.element.atomic_number if a.element else None,
         a.residue.index, a.residue.name, a.residue.id, a.residue.insertionCode,
         a.residue.chain.index, a.residue.chain.id]
        for a in topology.atoms()
    ]
    bonds = sorted(
        [min(a.index, b.index), max(a.index, b.index), str(bond.type), str(bond.order)]
        for bond in topology.bonds() for a, b in [bond]
    )
    return json_digest({"atoms": atoms, "bonds": bonds})


def box_values(box):
    if box is None:
        raise ValueError("periodic box is missing")
    values = np.asarray(box.value_in_unit(unit.nanometer), dtype=float)
    if values.shape != (3, 3) or not np.all(np.isfinite(values)) or np.linalg.det(values) <= 0:
        raise ValueError("periodic box must be finite and have positive volume")
    return values.tolist()


def prepared_observations(pdb):
    positions = np.asarray(pdb.positions.value_in_unit(unit.nanometer))
    if positions.shape != (pdb.topology.getNumAtoms(), 3) or not np.all(np.isfinite(positions)):
        raise ValueError("prepared positions do not match the topology or are not finite")
    return {
        "topology_sha256": topology_digest(pdb.topology),
        "n_particles": pdb.topology.getNumAtoms(),
        "box_vectors_nm": box_values(pdb.topology.getPeriodicBoxVectors()),
    }


def preparation_cache(outdir):
    return StageCache(outdir / "prepared.pdb", {
        "stage": "prepare", "source_sha256": sha256_file(DATA),
        "recipe": PREPARATION_RECIPE, "versions": dependency_versions(),
    })


def equilibration_cache(outdir, pdb, system, platform):
    return StageCache(outdir / "equilibrated.xml", {
        "stage": "equilibrate", "prepared_sha256": sha256_file(outdir / "prepared.pdb"),
        "prepared": prepared_observations(pdb), "recipe": EQUILIBRATION_RECIPE,
        "system_sha256": hashlib.sha256(mm.XmlSerializer.serialize(system).encode()).hexdigest(),
        "platform": {"name": platform.getName(), "properties": {
            name: platform.getPropertyDefaultValue(name) for name in platform.getPropertyNames()
        }},
        "versions": dependency_versions(),
    })


def state_observations(state, topology):
    n = topology.getNumAtoms()
    for label, values in (("positions", state.getPositions(asNumpy=True)),
                          ("velocities", state.getVelocities(asNumpy=True))):
        units = unit.nanometer if label == "positions" else unit.nanometer / unit.picosecond
        values = np.asarray(values.value_in_unit(units))
        if values.shape != (n, 3) or not np.all(np.isfinite(values)):
            raise ValueError(f"cached State {label} must be finite and match {n} topology atoms")
    return {
        "n_particles": n, "topology_sha256": topology_digest(topology),
        "box_vectors_nm": box_values(state.getPeriodicBoxVectors(asNumpy=True)),
    }


def forcefield_with_dce():
    from openff.toolkit import Molecule
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator

    recipe = PREPARATION_RECIPE
    dce = Molecule.from_smiles(recipe["ligand_smiles"])
    dce.assign_partial_charges(recipe["charge_model"])  # NAGL, no AmberTools
    forcefield = app.ForceField(*recipe["forcefield_files"])
    forcefield.registerTemplateGenerator(
        SMIRNOFFTemplateGenerator(molecules=dce, forcefield=recipe["ligand_forcefield"]).generator
    )
    return forcefield


def prepare(outdir: Path, forcefield):
    """Protonated, solvated, neutralized 2DHC; cached as prepared.pdb."""
    cache = preparation_cache(outdir)
    manifest = cache.load()
    if manifest is not None:
        pdb = app.PDBFile(str(cache.artifact))
        cache.check_observations(prepared_observations(pdb), manifest)
        return pdb
    from pdbfixer import PDBFixer
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    lines = DATA.read_text().splitlines()
    protein = [l for l in lines if l.startswith(("ATOM", "HETATM", "TER")) and l[17:20] != "DCE"]
    dce = [l for l in lines if l.startswith("HETATM") and l[17:20] == "DCE"]
    (outdir / "protein_xtal.pdb").write_text("\n".join(protein + ["END"]) + "\n")

    fixer = PDBFixer(filename=str(outdir / "protein_xtal.pdb"))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    modeller = app.Modeller(fixer.topology, fixer.positions)
    recipe = PREPARATION_RECIPE
    variants = [
        (recipe["histidine_289"] if r.name == "HIS" and int(r.id) == 289 else None)
        for r in modeller.topology.residues()
    ]
    modeller.addHydrogens(forcefield, pH=recipe["pH"], variants=variants)

    # DCE from the crystal heavy atoms (Cl1 is the leaving chlorine), H added by RDKit.
    heavy = {l[12:16].strip(): [float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in dce}
    order = ["CL1", "C1", "C2", "CL2"]
    mol = Chem.RWMol()
    for name in order:
        mol.AddAtom(Chem.Atom("Cl" if name.startswith("CL") else "C"))
    for i in range(3):
        mol.AddBond(i, i + 1, Chem.BondType.SINGLE)
    mol = mol.GetMol()
    Chem.SanitizeMol(mol)
    conformer = Chem.Conformer(4)
    for i, name in enumerate(order):
        conformer.SetAtomPosition(i, Point3D(*heavy[name]))
    mol.AddConformer(conformer)
    mol = Chem.AddHs(mol, addCoords=True)
    names, count = list(order), {1: 0, 2: 0}
    for atom in list(mol.GetAtoms())[4:]:
        carbon = atom.GetNeighbors()[0].GetIdx()
        count[carbon] += 1
        names.append(f"H{carbon}{count[carbon]}")
    ligand = app.Topology()
    residue = ligand.addResidue("DCE", ligand.addChain("L"), id="600")
    atoms = [
        ligand.addAtom(n, app.Element.getBySymbol(a.GetSymbol()), residue)
        for a, n in zip(mol.GetAtoms(), names)
    ]
    for bond in mol.GetBonds():
        ligand.addBond(atoms[bond.GetBeginAtomIdx()], atoms[bond.GetEndAtomIdx()])
    positions = mol.GetConformer().GetPositions() * 0.1  # Å -> nm
    modeller.add(ligand, [mm.Vec3(*p) for p in positions] * unit.nanometer)

    modeller.addSolvent(
        forcefield, model=recipe["solvent_model"], padding=recipe["padding_nm"] * unit.nanometer,
        neutralize=recipe["neutralize"], positiveIon=recipe["positive_ion"],
        negativeIon=recipe["negative_ion"], ionicStrength=recipe["ionic_strength_molar"] * unit.molar,
    )
    text = io.StringIO()
    app.PDBFile.writeFile(modeller.topology, modeller.positions, text, keepIds=True)
    atomic_write_text(cache.artifact, text.getvalue())
    pdb = app.PDBFile(str(cache.artifact))
    cache.store(prepared_observations(pdb))
    return pdb


def mm_system(topology, forcefield):
    return forcefield.createSystem(
        topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds, rigidWater=True,
    )


def equilibrate(outdir: Path, pdb, forcefield, platform):
    """MM minimization, 20 ps NVT, 50 ps NPT; cached as equilibrated.xml (State)."""
    system = mm_system(pdb.topology, forcefield)
    cache = equilibration_cache(outdir, pdb, system, platform)
    manifest = cache.load()
    if manifest is not None:
        state = mm.XmlSerializer.deserialize(cache.artifact.read_text())
        cache.check_observations(state_observations(state, pdb.topology), manifest)
        return state
    recipe = EQUILIBRATION_RECIPE
    barostat = mm.MonteCarloBarostat(
        recipe["pressure_bar"] * unit.bar, recipe["temperature_K"] * unit.kelvin,
        recipe["barostat_frequency"],
    )
    barostat.setRandomNumberSeed(recipe["barostat_seed"])
    system.addForce(barostat)
    barostat_frequency = barostat.getFrequency()
    barostat.setFrequency(0)  # off during NVT
    integrator = mm.LangevinMiddleIntegrator(
        recipe["temperature_K"] * unit.kelvin, recipe["friction_per_ps"] / unit.picosecond,
        recipe["step_fs"] * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(recipe["integrator_seed"])
    simulation = app.Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    t0 = time.perf_counter()
    simulation.minimizeEnergy(
        tolerance=recipe["min_tolerance_kj_mol_nm"] * unit.kilojoule_per_mole / unit.nanometer,
        maxIterations=recipe["min_iterations"],
    )
    simulation.context.setVelocitiesToTemperature(recipe["temperature_K"] * unit.kelvin, recipe["velocity_seed"])
    simulation.step(recipe["nvt_steps"])
    barostat.setFrequency(barostat_frequency)
    simulation.context.reinitialize(preserveState=True)
    simulation.step(recipe["npt_steps"])
    state = simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=False)
    atomic_write_text(cache.artifact, mm.XmlSerializer.serialize(state))
    # Record the serialized representation actually reused on the next run.
    state = mm.XmlSerializer.deserialize(cache.artifact.read_text())
    cache.store(state_observations(state, pdb.topology))
    box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    print(f"MM equilibration     : {time.perf_counter() - t0:.0f} s, box {np.diag(box).round(3)} nm", flush=True)
    return state


def qm_selection(topology, region: str):
    """(QM atom indices, boundary pairs (CB, CA), Asp124 oxygens, DCE C1, DCE Cl1)."""
    qm, pairs, asp_o = [], [], []
    c1 = cl1 = None
    for residue in topology.residues():
        atoms = {a.name: a.index for a in residue.atoms()}
        if residue.name == "DCE":
            qm.extend(atoms.values())
            c1, cl1 = atoms["C1"], atoms["CL1"]
        elif residue.name in ("ASP", "TRP", "HIS") and int(residue.id) in SIDE_CHAINS[region]:
            qm.extend(i for n, i in atoms.items() if n not in BACKBONE)
            pairs.append((atoms["CB"], atoms["CA"]))
            if int(residue.id) == 124:
                asp_o = [atoms["OD1"], atoms["OD2"]]
    return qm, pairs, asp_o, c1, cl1


def qm_bonds(topology, qm):
    qm_set = set(qm)
    return [(a.index, b.index) for a, b in topology.bonds() if a.index in qm_set and b.index in qm_set]


def partition_qm_bonds(topology, qm, reaction_pairs):
    bonds = qm_bonds(topology, qm)
    known = {tuple(sorted(pair)) for pair in bonds}
    reactions = []
    for pair in reaction_pairs:
        key = tuple(sorted(pair))
        if len(key) != 2 or key not in known:
            raise ValueError(f"reaction bond {pair} must be a covalent bond between two QM atoms")
        if key in reactions:
            raise ValueError(f"duplicate reaction bond {pair}")
        reactions.append(key)
    stable = [pair for pair in bonds if tuple(sorted(pair)) not in reactions]
    if not stable:
        raise ValueError("at least one QM covalent bond must remain in the stable-bond check")
    return stable, reactions


def validate_run_options(args):
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.min_iterations < 0:
        raise ValueError("min-iterations must be non-negative")
    if args.nprocs <= 0:
        raise ValueError("nprocs must be positive")
    if not math.isfinite(args.cutoff) or args.cutoff <= 0:
        raise ValueError("cutoff must be positive and finite")
    interval = getattr(args, "checkpoint_interval", DEFAULT_CHECKPOINT_INTERVAL)
    if interval <= 0 or interval % QMMM_RECIPE["dcd_interval"]:
        # Checkpoints on DCD frame steps keep each trajectory file's frames
        # aligned with the step ranges recorded in run.json.
        raise ValueError(
            f"checkpoint-interval must be a positive multiple of {QMMM_RECIPE['dcd_interval']}"
        )
    if getattr(args, "resume", None) not in (None, "checkpoint", "state"):
        raise ValueError("resume must be 'checkpoint' or 'state'")
    if getattr(args, "resume", None) and getattr(args, "archive_existing", False):
        raise ValueError("--resume and --archive-existing are mutually exclusive")


@contextmanager
def outdir_lock(outdir):
    """Refuse a second run in the same outdir while this one is alive."""
    with open(outdir / ".run.lock", "w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another run holds {outdir / '.run.lock'}") from None
        except OSError as exc:
            if exc.errno not in (errno.ENOLCK, errno.EOPNOTSUPP):
                raise
            print(f"warning: file locking unavailable in {outdir} ({exc}); "
                  "do not start two runs in this outdir", flush=True)
        yield


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def previous_run(outdir, resume, archive_existing):
    """The run record to resume, or None for a fresh run (after the overwrite rules)."""
    products = workflow.existing_run_products(outdir)
    if resume:
        if not (outdir / workflow.RUN_RECORD).is_file():
            raise RunMismatchError(f"--resume needs {outdir / workflow.RUN_RECORD}")
        return workflow.load_run_record(outdir)
    if products and archive_existing:
        target = workflow.archive_run_products(outdir, utc_now().replace("-", "").replace(":", ""))
        print(f"archived previous run products to {target}", flush=True)
    elif products:
        raise RunMismatchError(
            f"{outdir} already holds QM/MM run products ({', '.join(p.name for p in products)}); "
            "use --resume, --archive-existing or a new --outdir"
        )
    return None


def run(args):
    validate_run_options(args)  # Before creating files or loading optional dependencies.
    from openmmorca import ORCAPotential

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with outdir_lock(outdir):
        record = previous_run(outdir, getattr(args, "resume", None), getattr(args, "archive_existing", False))
        platform = mm.Platform.getPlatformByName(args.platform)
        forcefield = forcefield_with_dce()
        pdb = prepare(outdir, forcefield)
        state = equilibrate(outdir, pdb, forcefield, platform)
        topology = pdb.topology

        system = mm_system(topology, forcefield)  # no barostat: NVT
        box = state.getPeriodicBoxVectors()
        system.setDefaultPeriodicBoxVectors(*box)
        topology.setPeriodicBoxVectors(box)
        potential = ORCAPotential(
            method=args.method, basis=args.basis, charge=QMMM_RECIPE["charge"],
            multiplicity=QMMM_RECIPE["multiplicity"], nprocs=args.nprocs,
            extra_keywords=tuple(args.keyword),
        )
        try:
            return run_qmmm(args, outdir, platform, topology, state, system, potential, record)
        except RunMismatchError:
            raise  # A refused resume leaves the earlier run's files untouched.
        except BaseException as exc:
            # A stale PASS must never survive a new failed attempt in this outdir.
            atomic_write_text(outdir / "acceptance.json", json.dumps({
                "status": "FAILED", "task22_acceptance": "NOT_EVALUATED",
                "requested_steps": args.steps, "error_type": type(exc).__name__, "error": str(exc),
            }, indent=2) + "\n")
            raise
        finally:
            potential.close()


def run_environment():
    """Provenance of one attempt (best effort for git, which may be unavailable)."""
    versions = {"python": host_platform.python_version(), "openmm": mm.__version__}
    for name in ("openmm-orca", "orca-pi", "numpy"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    git = {"commit": None, "dirty": None}
    source = Path(__file__).resolve().parent.parent
    try:
        git["commit"] = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        ).stdout.strip()
        git["dirty"] = bool(subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain"], capture_output=True, text=True, check=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "versions": versions, "git": git, "argv": sys.argv, "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(), "opi_orca": os.environ.get("OPI_ORCA"),
    }


def run_identity(args, topology, state, system, qm, pairs, reaction_bonds):
    """Everything that defines the trajectory; nprocs and platform are per attempt."""
    serialized = lambda obj: hashlib.sha256(mm.XmlSerializer.serialize(obj).encode()).hexdigest()
    return {
        "start_state_sha256": serialized(state),
        "topology_sha256": topology_digest(topology),
        "mm_system_sha256": serialized(system),
        "qm": {
            "method": args.method, "basis": args.basis, "keywords": list(args.keyword),
            "region": args.region, "cutoff_nm": args.cutoff, "qm_atoms": list(qm),
            "boundary_pairs": [list(pair) for pair in pairs],
        },
        "reaction_bonds": [list(pair) for pair in reaction_bonds],
        "min_iterations": args.min_iterations,
        "recipe": QMMM_RECIPE,
    }


def save_checkpoint(outdir, record, simulation, step, csv_handle):
    """Write .chk + State, then commit them in run.json; drop the previous pair."""
    csv_handle.flush()
    os.fsync(csv_handle.fileno())
    csv_path = outdir / workflow.STEPS_CSV
    csv_bytes = csv_path.stat().st_size
    directory = outdir / workflow.CHECKPOINT_DIR
    directory.mkdir(exist_ok=True)
    chk = directory / f"step-{step:07d}.chk"
    xml = directory / f"step-{step:07d}.xml"
    atomic_write_bytes(chk, simulation.context.createCheckpoint())
    state = simulation.context.getState(
        getPositions=True, getVelocities=True, getParameters=True, enforcePeriodicBox=False,
    )
    atomic_write_text(xml, mm.XmlSerializer.serialize(state))
    previous = record.get("checkpoint")
    record["checkpoint"] = {
        "step": step, "time_ps": state.getTime().value_in_unit(unit.picosecond),
        "checkpoint_file": str(chk.relative_to(outdir)), "checkpoint_sha256": sha256_file(chk),
        "state_file": str(xml.relative_to(outdir)), "state_sha256": sha256_file(xml),
        "steps_csv_bytes": csv_bytes,
        "steps_csv_sha256": workflow.file_prefix_sha256(csv_path, csv_bytes),
        "platform": simulation.context.getPlatform().getName(), "openmm": mm.__version__,
    }
    record["attempts"][-1]["valid_through_step"] = step
    workflow.write_run_record(outdir, record)
    if previous and previous["step"] != step:
        for key in ("checkpoint_file", "state_file"):
            (outdir / previous[key]).unlink(missing_ok=True)


def restore_checkpoint(outdir, record, simulation, mode):
    """Load the committed checkpoint into *simulation*; return its step."""
    checkpoint = record.get("checkpoint")
    if checkpoint is None:
        raise RunMismatchError("run.json has no checkpoint yet; start again with --archive-existing")
    key = "checkpoint" if mode == "checkpoint" else "state"
    path = outdir / checkpoint[f"{key}_file"]
    if not path.is_file() or sha256_file(path) != checkpoint[f"{key}_sha256"]:
        raise RunMismatchError(f"{path} is missing or does not match run.json")
    if mode == "checkpoint":
        platform = simulation.context.getPlatform().getName()
        if (checkpoint["platform"], checkpoint["openmm"]) != (platform, mm.__version__):
            raise RunMismatchError(
                f"checkpoint was written by {checkpoint['platform']} / OpenMM {checkpoint['openmm']}, "
                f"this run uses {platform} / OpenMM {mm.__version__}; use --resume state"
            )
        simulation.context.loadCheckpoint(path.read_bytes())
    else:
        simulation.context.setState(mm.XmlSerializer.deserialize(path.read_text()))
    if simulation.currentStep != checkpoint["step"]:
        raise RunMismatchError(
            f"{path} holds step {simulation.currentStep}, run.json says {checkpoint['step']}"
        )
    return checkpoint["step"]


def run_qmmm(args, outdir, platform, topology, state, system, potential, record=None):
    qm, pairs, asp_o, c1, cl1 = qm_selection(topology, args.region)
    stable_bonds, reaction_bonds = partition_qm_bonds(
        topology, qm, getattr(args, "reaction_bond", ()),
    )
    identity = run_identity(args, topology, state, system, qm, pairs, reaction_bonds)
    resume = getattr(args, "resume", None)
    interval = getattr(args, "checkpoint_interval", DEFAULT_CHECKPOINT_INTERVAL)
    if record is None:
        record = {"schema_version": 1, "identity": identity, "status": "running",
                  "requested_steps": args.steps, "attempts": [], "checkpoint": None}
    else:
        workflow.check_run_identity(identity, record)
        checkpoint = record.get("checkpoint") or {}
        if args.steps <= checkpoint.get("step", 0):
            raise RunMismatchError(
                f"run is checkpointed at step {checkpoint['step']}; pass --steps beyond it to continue"
            )
    attempt_number = len(record["attempts"]) + 1
    if reaction_bonds:
        print(f"reaction bonds (0-based OpenMM indices, monitored separately): {reaction_bonds}", flush=True)
    mixed = potential.createMixedSystem(
        topology, system, qm, boundaryPairs=pairs, embeddingCutoff=args.cutoff * unit.nanometer
    )
    recipe = QMMM_RECIPE
    integrator = mm.LangevinMiddleIntegrator(
        recipe["temperature_K"] * unit.kelvin, recipe["friction_per_ps"] / unit.picosecond,
        recipe["step_fs"] * unit.femtosecond,
    )
    # A State restart must not replay the random stream of the first attempt;
    # a checkpoint restart overwrites the seed with the saved generator state.
    integrator.setRandomNumberSeed(recipe["integrator_seed"] + attempt_number - 1)
    simulation = app.Simulation(topology, mixed, integrator, platform)
    print(f"QM region {args.region}: {len(qm)} atoms + {len(pairs)} link H, method {args.method} {args.basis or ''}", flush=True)

    backend = potential.backends[0]
    bonds = qm_bonds(topology, qm)
    stable_indices = [k for k, pair in enumerate(bonds) if pair in stable_bonds]
    reaction_indices = [k for k, pair in enumerate(bonds) if tuple(sorted(pair)) in reaction_bonds]
    if resume:
        start = restore_checkpoint(outdir, record, simulation, resume)
        rows = workflow.truncate_steps_csv(outdir, record["checkpoint"], attempt_number - 1)
        reference = np.array(record["reference_bond_lengths_nm"])
        worst_bond = rows[-1]["max_bond_dev"]
        worst_stable = rows[-1]["max_stable_bond_dev"]
        worst_reaction = rows[-1]["max_reaction_bond_dev"]
        print(f"resuming from step {start} ({resume})", flush=True)
    else:
        simulation.context.setPositions(state.getPositions())
        simulation.context.setVelocities(state.getVelocities())
        if args.min_iterations > 0:
            t0 = time.perf_counter()
            simulation.minimizeEnergy(
                tolerance=recipe["min_tolerance_kj_mol_nm"] * unit.kilojoule_per_mole / unit.nanometer,
                maxIterations=args.min_iterations,
            )
            print(f"QM/MM minimization   : {time.perf_counter() - t0:.0f} s, {len(backend.timings.rows)} QM calls", flush=True)
            simulation.context.setVelocitiesToTemperature(
                recipe["temperature_K"] * unit.kelvin, recipe["min_velocity_seed"],
            )
        positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        reference = np.array([np.linalg.norm(positions[i] - positions[j]) for i, j in bonds])
        if not np.all(np.isfinite(reference)) or np.any(reference <= 0):
            raise ValueError("QM bond reference lengths must be finite and positive")
        record["reference_bond_lengths_nm"] = reference.tolist()
        record["qm_bonds"] = [list(pair) for pair in bonds]
        record["stable_bonds"] = [list(pair) for pair in stable_bonds]
        start, rows = 0, []
        worst_bond = worst_stable = worst_reaction = 0.0
    n_dof = 3 * mixed.getNumParticles() - mixed.getNumConstraints() - 3
    masses = np.array([mixed.getParticleMass(i).value_in_unit(unit.dalton) for i in range(mixed.getNumParticles())])
    k_b = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)

    trajectory = "trajectory.dcd" if attempt_number == 1 else f"trajectory.attempt-{attempt_number:03d}.dcd"
    attempt = {
        "attempt": attempt_number, "mode": resume or "fresh", "start_step": start,
        "valid_through_step": start, "trajectory": trajectory, "platform": platform.getName(),
        "nprocs": args.nprocs, "started": utc_now(), "status": "running",
        "environment": run_environment(),
    }
    record["attempts"].append(attempt)
    record["status"] = "running"
    record["requested_steps"] = args.steps
    workflow.write_run_record(outdir, record)
    atomic_write_text(outdir / "acceptance.json", json.dumps({
        "status": "RUNNING", "task22_acceptance": "NOT_EVALUATED", "requested_steps": args.steps,
    }, indent=2) + "\n")
    simulation.reporters.append(app.DCDReporter(str(outdir / trajectory), recipe["dcd_interval"]))

    n_rows_before = len(backend.timings.rows)
    t_md = time.perf_counter()
    try:
        with open(outdir / workflow.STEPS_CSV, "a" if resume else "w", newline="") as handle:
            writer = None
            for step in range(start + 1, args.steps + 1):
                simulation.step(1)
                # Fetching energy would pay for a second QM evaluation each step.
                s = simulation.context.getState(getPositions=True, getVelocities=True)
                pos = s.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                vel = s.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
                if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(vel)):
                    raise ValueError(f"non-finite positions or velocities at step {step}")
                kinetic = 0.5 * float(np.sum(masses[:, None] * vel**2))
                temperature = 2 * kinetic / (n_dof * k_b)
                lengths = np.array([np.linalg.norm(pos[i] - pos[j]) for i, j in bonds])
                deviations = np.abs(lengths / reference - 1)
                worst_bond = max(worst_bond, float(deviations.max()))
                worst_stable = max(worst_stable, float(deviations[stable_indices].max()))
                if reaction_indices:
                    worst_reaction = max(worst_reaction, float(deviations[reaction_indices].max()))
                timing = backend.timings.rows[-1]
                d_attack = min(np.linalg.norm(pos[o] - pos[c1]) for o in asp_o)
                row = {
                    "step": step, "temperature_K": temperature,
                    "t_write": timing["t_write"], "t_orca": timing["t_orca"], "t_read": timing["t_read"],
                    "t_total": timing["t_total"], "n_embed_groups": timing["n_embed_groups"],
                    "embed_changed": timing["embed_changed"], "scf_cycles": timing["scf_cycles"],
                    "restart_used": int(timing["restart_used"]), "fresh_retries": timing["fresh_retries"],
                    "d_OD_C1_nm": d_attack,
                    "d_C1_Cl1_nm": float(np.linalg.norm(pos[c1] - pos[cl1])), "max_bond_dev": worst_bond,
                    "max_stable_bond_dev": worst_stable, "max_reaction_bond_dev": worst_reaction,
                }
                for i, j in reaction_bonds:
                    row[f"reaction_{i}_{j}_nm"] = float(np.linalg.norm(pos[i] - pos[j]))
                rows.append(row)
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    if not resume:
                        writer.writeheader()
                writer.writerow(row)
                handle.flush()
                if step % interval == 0 or step == args.steps:
                    save_checkpoint(outdir, record, simulation, step, handle)
                if step % 10 == 0 or step == start + 1:
                    wall = (time.perf_counter() - t_md) / (step - start)
                    print(
                        f"step {step:5d}  T {temperature:6.1f} K  orca {timing['t_orca']:.2f} s  "
                        f"wall {wall:.2f} s/step  groups {timing['n_embed_groups']}  "
                        f"O–C1 {d_attack:.3f} nm  max stable bond dev {worst_stable * 100:.1f}%",
                        flush=True,
                    )
    except BaseException as exc:
        attempt.update(status="failed", ended=utc_now(), error_type=type(exc).__name__, error=str(exc),
                       fresh_retries=backend.n_fresh_retries)
        record["status"] = "failed"
        workflow.write_run_record(outdir, record)
        raise

    final = simulation.context.getState(
        getPositions=True, getVelocities=True, getParameters=True, enforcePeriodicBox=False,
    )
    atomic_write_text(outdir / "final_state.xml", mm.XmlSerializer.serialize(final))
    text = io.StringIO()
    app.PDBFile.writeFile(topology, final.getPositions(), text, keepIds=True)
    atomic_write_text(outdir / "final.pdb", text.getvalue())
    attempt.update(status="completed", ended=utc_now(), fresh_retries=backend.n_fresh_retries)
    record["status"] = "completed"
    workflow.write_run_record(outdir, record)

    md_rows = backend.timings.rows[n_rows_before:]
    n_steps = args.steps - start
    assert len(md_rows) == n_steps, f"{len(md_rows)} QM calls for {n_steps} steps"
    tail = np.array([r["temperature_K"] for r in rows[-500:]])
    print(f"\nsteps                : {args.steps} ({args.steps * recipe['step_fs'] / 1000:.3f} ps), "
          f"{n_steps} in this attempt")
    print(f"wall time (attempt)  : {time.perf_counter() - t_md:.0f} s")
    for field in ("t_write", "t_orca", "t_read", "t_total"):
        timing_rows = md_rows[1:] if len(md_rows) > 1 else md_rows
        values = np.array([r[field] for r in timing_rows])
        print(f"  {field:8s} mean {values.mean():.3f} s, p95 {np.percentile(values, 95):.3f} s")
    print(f"embedding groups     : {np.mean([r['n_embed_groups'] for r in md_rows]):.0f} mean, "
          f"{sum(r['embed_changed'] > 0 for r in md_rows[1:])} steps with crossings")
    print(f"n_fresh_retries      : {backend.n_fresh_retries}")
    print(f"T (last {len(tail)} steps)    : {tail.mean():.1f} ± {tail.std():.1f} K")
    print(f"max QM bond deviation: {worst_bond * 100:.1f} %")
    print(f"max stable bond dev  : {worst_stable * 100:.1f} %")
    print(f"max reaction bond dev: {worst_reaction * 100:.1f} %")
    summary = acceptance_summary(rows, args.steps, worst_stable, worst_bond)
    summary["reaction_bonds"] = reaction_bonds
    summary["max_reaction_bond_deviation"] = worst_reaction
    summary["attempts"] = attempt_number
    atomic_write_text(outdir / "acceptance.json", json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"{summary['kind']} checks          : {summary['status']}")
    print(f"Task 22 acceptance   : {summary['task22_acceptance']}")
    return summary["status"] == "PASS"


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", default="XTB")
    parser.add_argument("--basis", default=None)
    parser.add_argument("--keyword", action="append", default=[], help="extra ORCA keyword (repeatable)")
    parser.add_argument("--region", choices=sorted(SIDE_CHAINS), default="A")
    parser.add_argument("--steps", type=positive_int, default=2000)
    parser.add_argument("--min-iterations", type=int, default=0,
                        help="optional QM/MM minimization of the whole system (see module docstring)")
    parser.add_argument("--cutoff", type=float, default=1.2, help="embedding cutoff (nm)")
    parser.add_argument("--nprocs", type=positive_int, default=1)
    parser.add_argument("--reaction-bond", type=int, nargs=2, action="append", default=[],
                        metavar=("I", "J"), help="0-based QM atom indices of a reactive bond; repeatable")
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument("--outdir", default="dhla_out")
    parser.add_argument("--checkpoint-interval", type=positive_int, default=DEFAULT_CHECKPOINT_INTERVAL,
                        help="steps between checkpoints (multiple of the 10-step DCD interval)")
    parser.add_argument("--resume", choices=("checkpoint", "state"), default=None,
                        help="continue the run in --outdir from its last checkpoint (see module docstring)")
    parser.add_argument("--archive-existing", action="store_true",
                        help="move earlier QM/MM run products in --outdir to archive/ and start fresh")
    args = parser.parse_args()
    try:
        validate_run_options(args)
    except ValueError as exc:
        parser.error(str(exc))
    raise SystemExit(0 if run(args) else 1)


if __name__ == "__main__":
    main()
