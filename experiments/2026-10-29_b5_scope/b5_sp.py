"""B5 E2: whole-box (S2) QM/MM single points of region A or C on fixed snapshots.

Per frame: QM/MM force group (ORCA) and all other (MM) groups evaluated separately on the CPU platform,
so total = QM + MM is available for B3-style comparisons on common atoms. Forces are stored for the
atoms of regions A and C plus their boundary M1 atoms. Results: OUTDIR/<region>.npz, rewritten after
every frame; rerunning the same command skips finished frames (resumable).

Usage: b5_sp.py RUN_FILES_DIR FRAMES.npz OUTDIR --region A|C [--frames 0 1 ...] [--nprocs 18]
RUN_FILES_DIR holds prepared.pdb and equilibrated.xml of the run the frames come from.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
import enzyme_qmmm as enzyme  # noqa: E402

from openmmorca import ORCAPotential  # noqa: E402

QM_GROUP = 31
KJ = unit.kilojoule_per_mole
KEYS = ("e_qm", "e_mm", "f_qm", "f_mm", "n_embed_groups", "t_orca", "scf_cycles")


def load(path):
    if not path.is_file():
        return {}
    with np.load(path) as data:
        return {int(f): {k: data[f"{k}_{f}"] for k in KEYS} for f in data["frames_done"]}


def save(path, results, atoms):
    arrays = {"frames_done": np.array(sorted(results)), "atoms": atoms}
    for f, r in results.items():
        arrays.update({f"{k}_{f}": v for k, v in r.items()})
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_files", type=Path)
    parser.add_argument("frames_npz", type=Path)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--region", choices=sorted(enzyme.SIDE_CHAINS), required=True)
    parser.add_argument("--frames", type=int, nargs="*", default=None, help="indices into FRAMES.npz")
    parser.add_argument("--method", default="r2SCAN-3c")
    parser.add_argument("--nprocs", type=int, default=18)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    frames = np.load(args.frames_npz)
    picked = range(len(frames["step"])) if args.frames is None else args.frames
    pdb = app.PDBFile(str(args.run_files / "prepared.pdb"))
    topology = pdb.topology
    state = mm.XmlSerializer.deserialize((args.run_files / "equilibrated.xml").read_text())
    system = enzyme.mm_system(topology, enzyme.forcefield_with_dce())
    system.setDefaultPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
    qm, pairs, *_ = enzyme.qm_selection(topology, args.region)
    stored = set()
    for region in ("A", "C"):
        q, p, *_ = enzyme.qm_selection(topology, region)
        stored.update(q)
        stored.update(m1 for _, m1 in p)
    atoms = np.array(sorted(stored))
    meta = {"region": args.region, "qm_atoms": qm, "boundary_pairs": pairs, "method": args.method,
            "embedding": "whole box (embeddingCutoff=None)", "nprocs": args.nprocs,
            "frames_npz": str(args.frames_npz), "run_files": str(args.run_files)}
    (args.outdir / f"{args.region}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    path = args.outdir / f"{args.region}.npz"
    results = load(path)
    todo = [f for f in picked if f not in results]
    if not todo:
        return
    potential = ORCAPotential(method=args.method, charge=enzyme.QMMM_RECIPE["charge"],
                              multiplicity=enzyme.QMMM_RECIPE["multiplicity"], nprocs=args.nprocs)
    try:
        mixed = potential.createMixedSystem(topology, system, qm, boundaryPairs=pairs, forceGroup=QM_GROUP,
                                            embeddingCutoff=None)
        mm_groups = {f.getForceGroup() for f in mixed.getForces()} - {QM_GROUP}
        context = mm.Context(mixed, mm.VerletIntegrator(0.0005), mm.Platform.getPlatformByName("CPU"))
        backend = potential.backends[0]
        for f in todo:
            context.setPeriodicBoxVectors(*frames["box_nm"][f])
            context.setPositions(frames["positions_nm"][f])
            t0 = time.perf_counter()
            sq = context.getState(getEnergy=True, getForces=True, groups={QM_GROUP})
            sm = context.getState(getEnergy=True, getForces=True, groups=mm_groups)
            row = backend.timings.rows[-1]
            results[f] = {
                "e_qm": np.array(sq.getPotentialEnergy().value_in_unit(KJ)),
                "e_mm": np.array(sm.getPotentialEnergy().value_in_unit(KJ)),
                "f_qm": sq.getForces(asNumpy=True).value_in_unit(KJ / unit.nanometer)[atoms],
                "f_mm": sm.getForces(asNumpy=True).value_in_unit(KJ / unit.nanometer)[atoms],
                "n_embed_groups": np.array(int(row["n_embed_groups"])),
                "t_orca": np.array(float(row["t_orca"])),
                "scf_cycles": np.array(int(row["scf_cycles"])),
            }
            save(path, results, atoms)
            print(f"region {args.region} frame {f} step {int(frames['step'][f])}: E_QM {float(results[f]['e_qm']):.4f} "
                  f"kJ/mol, {int(row['n_embed_groups'])} groups, t_orca {float(row['t_orca']):.1f} s, "
                  f"SCF {int(row['scf_cycles'])}, wall {time.perf_counter() - t0:.1f} s", flush=True)
    finally:
        potential.close()


if __name__ == "__main__":
    main()
