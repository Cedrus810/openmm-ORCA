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

Acceptance (implementation plan Task 22): the whole run finishes without
intervention; the mean temperature of the last 500 steps is 300 ± 10 K; every
QM covalent bond stays within 20 % of its length after the QM/MM minimization.

    export OPI_ORCA=/home/ruigengji/ORCA611
    /home/ruigengji/miniforge3/envs/openmm_dev/bin/python examples/enzyme_qmmm.py \\
        --method XTB --steps 200 --outdir dhla_xtb          # smoke test
    PRTE_MCA_hwloc_default_cpu_list=0-39 ... --method r2SCAN-3c --nprocs 40 \\
        --steps 2000 --outdir dhla_r2scan3c   # 1 ps, ~3 h (validated 2026-10-01: PASS)

DCE lies entirely in the QM region (both regions), where its force-field
charges are zeroed and its internal terms removed, so its parameters only
matter for the MM equilibration. Sage + NAGL charges need no AmberTools.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit

DATA = Path(__file__).resolve().parent / "data" / "2DHC.pdb"
BACKBONE = {"N", "H", "CA", "HA", "C", "O", "OXT"}
SIDE_CHAINS = {"A": (124,), "C": (124, 125, 175, 289)}


def forcefield_with_dce():
    from openff.toolkit import Molecule
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator

    dce = Molecule.from_smiles("ClCCCl")
    dce.assign_partial_charges("openff-gnn-am1bcc-0.1.0-rc.3.pt")  # NAGL, no AmberTools
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
    forcefield.registerTemplateGenerator(
        SMIRNOFFTemplateGenerator(molecules=dce, forcefield="openff-2.2.0").generator
    )
    return forcefield


def prepare(outdir: Path, forcefield):
    """Protonated, solvated, neutralized 2DHC; cached as prepared.pdb."""
    cached = outdir / "prepared.pdb"
    if cached.exists():
        return app.PDBFile(str(cached))
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
    variants = [
        ("HID" if r.name == "HIS" and int(r.id) == 289 else None) for r in modeller.topology.residues()
    ]
    modeller.addHydrogens(forcefield, pH=7.0, variants=variants)

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
        forcefield, model="tip3p", padding=1.0 * unit.nanometer, neutralize=True,
        positiveIon="Na+", negativeIon="Cl-", ionicStrength=0.0 * unit.molar,
    )
    with open(cached, "w") as handle:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)
    return app.PDBFile(str(cached))


def mm_system(topology, forcefield):
    return forcefield.createSystem(
        topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds, rigidWater=True,
    )


def equilibrate(outdir: Path, pdb, forcefield, platform):
    """MM minimization, 20 ps NVT, 50 ps NPT; cached as equilibrated.xml (State)."""
    cached = outdir / "equilibrated.xml"
    if cached.exists():
        return mm.XmlSerializer.deserialize(cached.read_text())
    system = mm_system(pdb.topology, forcefield)
    barostat = mm.MonteCarloBarostat(1.0 * unit.bar, 300 * unit.kelvin)
    system.addForce(barostat)
    barostat_frequency = barostat.getFrequency()
    barostat.setFrequency(0)  # off during NVT
    integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond)
    simulation = app.Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    t0 = time.perf_counter()
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 2026)
    simulation.step(10_000)  # 20 ps NVT
    barostat.setFrequency(barostat_frequency)
    simulation.context.reinitialize(preserveState=True)
    simulation.step(25_000)  # 50 ps NPT
    state = simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=False)
    cached.write_text(mm.XmlSerializer.serialize(state))
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


def run(args):
    from openmmorca import ORCAPotential

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    platform = mm.Platform.getPlatformByName(args.platform)
    forcefield = forcefield_with_dce()
    pdb = prepare(outdir, forcefield)
    state = equilibrate(outdir, pdb, forcefield, platform)
    topology = pdb.topology

    system = mm_system(topology, forcefield)  # no barostat: NVT
    box = state.getPeriodicBoxVectors()
    system.setDefaultPeriodicBoxVectors(*box)
    topology.setPeriodicBoxVectors(box)
    qm, pairs, asp_o, c1, cl1 = qm_selection(topology, args.region)
    potential = ORCAPotential(
        method=args.method, basis=args.basis, charge=-1, multiplicity=1,
        nprocs=args.nprocs, extra_keywords=tuple(args.keyword),
    )
    mixed = potential.createMixedSystem(
        topology, system, qm, boundaryPairs=pairs, embeddingCutoff=args.cutoff * unit.nanometer
    )
    integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.5 * unit.femtosecond)
    simulation = app.Simulation(topology, mixed, integrator, platform)
    simulation.context.setPositions(state.getPositions())
    simulation.context.setVelocities(state.getVelocities())
    print(f"QM region {args.region}: {len(qm)} atoms + {len(pairs)} link H, method {args.method} {args.basis or ''}", flush=True)

    backend = potential.backends[0]
    if args.min_iterations > 0:
        t0 = time.perf_counter()
        simulation.minimizeEnergy(
            tolerance=10 * unit.kilojoule_per_mole / unit.nanometer, maxIterations=args.min_iterations
        )
        print(f"QM/MM minimization   : {time.perf_counter() - t0:.0f} s, {len(backend.timings.rows)} QM calls", flush=True)
        simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 2026)

    bonds = qm_bonds(topology, qm)
    positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    reference = np.array([np.linalg.norm(positions[i] - positions[j]) for i, j in bonds])
    n_dof = 3 * mixed.getNumParticles() - mixed.getNumConstraints() - 3
    masses = np.array([mixed.getParticleMass(i).value_in_unit(unit.dalton) for i in range(mixed.getNumParticles())])
    k_b = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)
    simulation.reporters.append(app.DCDReporter(str(outdir / "trajectory.dcd"), 10))

    rows = []
    n_rows_before = len(backend.timings.rows)
    worst_bond = 0.0
    t_md = time.perf_counter()
    for step in range(1, args.steps + 1):
        simulation.step(1)
        # Positions and velocities only: getEnergy would re-evaluate every force,
        # i.e. pay one extra QM calculation per step.
        s = simulation.context.getState(getPositions=True, getVelocities=True)
        pos = s.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        vel = s.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
        kinetic = 0.5 * float(np.sum(masses[:, None] * vel**2))
        temperature = 2 * kinetic / (n_dof * k_b)
        lengths = np.array([np.linalg.norm(pos[i] - pos[j]) for i, j in bonds])
        worst_bond = max(worst_bond, float(np.max(np.abs(lengths / reference - 1))))
        timing = backend.timings.rows[-1]
        d_attack = min(np.linalg.norm(pos[o] - pos[c1]) for o in asp_o)
        rows.append({
            "step": step, "temperature_K": temperature,
            "t_write": timing["t_write"], "t_orca": timing["t_orca"], "t_read": timing["t_read"],
            "t_total": timing["t_total"], "n_embed_groups": timing["n_embed_groups"],
            "embed_changed": timing["embed_changed"], "d_OD_C1_nm": d_attack,
            "d_C1_Cl1_nm": float(np.linalg.norm(pos[c1] - pos[cl1])), "max_bond_dev": worst_bond,
        })
        if step % 10 == 0 or step == 1:
            wall = (time.perf_counter() - t_md) / step
            print(
                f"step {step:5d}  T {temperature:6.1f} K  orca {timing['t_orca']:.2f} s  "
                f"wall {wall:.2f} s/step  groups {timing['n_embed_groups']}  "
                f"O–C1 {d_attack:.3f} nm  max bond dev {worst_bond * 100:.1f}%",
                flush=True,
            )
    with open(outdir / "steps.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    md_rows = backend.timings.rows[n_rows_before:]
    assert len(md_rows) == args.steps, f"{len(md_rows)} QM calls for {args.steps} steps"
    tail = np.array([r["temperature_K"] for r in rows[-500:]])
    print(f"\nsteps                : {args.steps} ({args.steps * 0.5 / 1000:.3f} ps)")
    print(f"wall time            : {time.perf_counter() - t_md:.0f} s")
    for field in ("t_write", "t_orca", "t_read", "t_total"):
        values = np.array([r[field] for r in md_rows[1:]])
        print(f"  {field:8s} mean {values.mean():.3f} s, p95 {np.percentile(values, 95):.3f} s")
    print(f"embedding groups     : {np.mean([r['n_embed_groups'] for r in md_rows]):.0f} mean, "
          f"{sum(r['embed_changed'] > 0 for r in md_rows[1:])} steps with crossings")
    print(f"n_fresh_retries      : {backend.n_fresh_retries}")
    print(f"T (last {len(tail)} steps)    : {tail.mean():.1f} ± {tail.std():.1f} K")
    print(f"max QM bond deviation: {worst_bond * 100:.1f} %")
    ok = abs(tail.mean() - 300) <= 10 and worst_bond <= 0.20
    print(f"acceptance           : {'PASS' if ok else 'FAIL'}")
    potential.close()
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", default="XTB")
    parser.add_argument("--basis", default=None)
    parser.add_argument("--keyword", action="append", default=[], help="extra ORCA keyword (repeatable)")
    parser.add_argument("--region", choices=sorted(SIDE_CHAINS), default="A")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--min-iterations", type=int, default=0,
                        help="optional QM/MM minimization of the whole system (see module docstring)")
    parser.add_argument("--cutoff", type=float, default=1.2, help="embedding cutoff (nm)")
    parser.add_argument("--nprocs", type=int, default=1)
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument("--outdir", default="dhla_out")
    raise SystemExit(0 if run(parser.parse_args()) else 1)


if __name__ == "__main__":
    main()
