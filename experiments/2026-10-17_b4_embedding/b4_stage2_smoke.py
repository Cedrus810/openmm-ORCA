"""Roadmap B4 stage 2 (smoke): ORCA cost of embedding the whole box (scheme S2) vs the 1.2 nm sphere (S0).

S2 is obtained from the unchanged ORCAPotential path with an embedding cutoff larger than the box, after
disabling the sphere/box check (`check_cutoff_against_box`, which S2 does not use: every group is placed
at its minimum image relative to the QM region, exactly as CutoffEmbedding.select places selected groups).
The check is disabled only inside this process.

Usage: b4_stage2_smoke.py T09_DIR OUTDIR --frames 0 1 [--nprocs 18] [--schemes S0 S2]
Writes OUTDIR/stage2.json (energy, n_groups, n_point_charges, t_orca, scf_cycles per scheme and frame)
and OUTDIR/qm_forces.npz (QM/MM-group forces on the QM atoms, kJ/mol/nm).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))
import enzyme_qmmm as enzyme  # noqa: E402

import openmmorca.potential as potential_mod  # noqa: E402
from openmmorca import ORCAPotential  # noqa: E402
from openmmorca.qmmm import embedding as embedding_mod  # noqa: E402

QM_GROUP = 1
FULL_BOX_NM = 100.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("t09", type=Path)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--nprocs", type=int, default=18)
    parser.add_argument("--schemes", nargs="+", default=["S0", "S2"], choices=["S0", "S2"])
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    frames = np.load(args.t09 / "frames.npz")
    run_files = args.t09 / "run_files"
    pdb = app.PDBFile(str(run_files / "prepared.pdb"))
    topology = pdb.topology
    state = mm.XmlSerializer.deserialize((run_files / "equilibrated.xml").read_text())
    system = enzyme.mm_system(topology, enzyme.forcefield_with_dce())
    system.setDefaultPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
    qm, pairs, *_ = enzyme.qm_selection(topology, "A")

    results = []
    qm_forces = {}
    for scheme, cutoff in (("S0", 1.2), ("S2", FULL_BOX_NM)):
        if scheme not in args.schemes:
            continue
        if scheme == "S2":
            embedding_mod.check_cutoff_against_box = lambda *a, **k: None
            potential_mod.check_cutoff_against_box = lambda *a, **k: None
        potential = ORCAPotential(method="r2SCAN-3c", charge=-1, multiplicity=1, nprocs=args.nprocs)
        try:
            mixed = potential.createMixedSystem(topology, system, qm, boundaryPairs=pairs, forceGroup=QM_GROUP,
                                                embeddingCutoff=cutoff * unit.nanometer)
            context = mm.Context(mixed, mm.VerletIntegrator(0.0005), mm.Platform.getPlatformByName("CPU"))
            backend = potential.backends[0]
            for f in args.frames:
                context.setPeriodicBoxVectors(*frames["box_nm"][f])
                context.setPositions(frames["positions_nm"][f])
                t0 = time.perf_counter()
                s = context.getState(getEnergy=True, getForces=True, groups={QM_GROUP})
                row = backend.timings.rows[-1]
                forces = s.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
                r = {"scheme": scheme, "frame": f, "step": int(frames["step"][f]),
                     "energy_kj_mol": s.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                     "n_groups": int(row["n_embed_groups"]),
                     "n_point_charges": int(np.count_nonzero(np.any(forces != 0, axis=1))) - len(qm) - 1,
                     "t_orca": float(row["t_orca"]), "t_write": float(row["t_write"]), "t_read": float(row["t_read"]),
                     "scf_cycles": row.get("scf_cycles"), "wall": time.perf_counter() - t0}
                results.append(r)
                qm_forces[f"{scheme}_f{f}"] = forces[qm]
                print(json.dumps(r), flush=True)
        finally:
            potential.close()
    (args.outdir / "stage2.json").write_text(json.dumps(results, indent=1) + "\n")
    np.savez(args.outdir / "qm_forces.npz", qm_atoms=np.array(qm), **qm_forces)


if __name__ == "__main__":
    main()
