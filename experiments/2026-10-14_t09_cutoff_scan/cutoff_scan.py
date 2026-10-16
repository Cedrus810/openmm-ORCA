"""T09 item 1: QM/MM energy and forces of fixed snapshots at several embedding cutoffs.

One ORCA backend per cutoff; each snapshot is a single point (only the QM/MM force group is
evaluated). Results go to OUTDIR/c<cutoff>.npz, one file per cutoff, written after every frame,
so rerunning the same command skips finished (cutoff, frame) pairs.

Usage: cutoff_scan.py RUN_FILES_DIR FRAMES.npz OUTDIR --cutoffs 1.2 2.0 [--frames 0 1 ...]
       [--method r2SCAN-3c] [--nprocs 32] [--region A]
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
from openmmorca.units import HARTREE_IN_KJ_MOL  # noqa: E402

QM_GROUP = 1


def load(path):
    if not path.is_file():
        return {}
    with np.load(path) as data:
        return {int(f): {k: data[f"{k}_{f}"] for k in ("energy_kj_mol", "forces", "n_embed_groups", "t_orca")}
                for f in data["frames_done"]}


def save(path, results):
    arrays = {"frames_done": np.array(sorted(results))}
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
    parser.add_argument("--cutoffs", type=float, nargs="+", required=True)
    parser.add_argument("--frames", type=int, nargs="*", default=None, help="indices into FRAMES.npz")
    parser.add_argument("--method", default="r2SCAN-3c")
    parser.add_argument("--nprocs", type=int, default=32)
    parser.add_argument("--region", choices=sorted(enzyme.SIDE_CHAINS), default="A")
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
    meta = {"region": args.region, "qm_atoms": qm, "boundary_pairs": pairs, "method": args.method,
            "nprocs": args.nprocs, "frames_npz": str(args.frames_npz), "run_files": str(args.run_files)}
    (args.outdir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    for cutoff in args.cutoffs:
        path = args.outdir / f"c{cutoff:.3f}.npz"
        results = load(path)
        todo = [f for f in picked if f not in results]
        if not todo:
            continue
        potential = ORCAPotential(method=args.method, charge=enzyme.QMMM_RECIPE["charge"],
                                  multiplicity=enzyme.QMMM_RECIPE["multiplicity"], nprocs=args.nprocs)
        try:
            mixed = potential.createMixedSystem(topology, system, qm, boundaryPairs=pairs, forceGroup=QM_GROUP,
                                                embeddingCutoff=cutoff * unit.nanometer)
            context = mm.Context(mixed, mm.VerletIntegrator(0.0005), mm.Platform.getPlatformByName("CPU"))
            backend = potential.backends[0]
            for f in todo:
                context.setPeriodicBoxVectors(*frames["box_nm"][f])
                context.setPositions(frames["positions_nm"][f])
                t0 = time.perf_counter()
                s = context.getState(getEnergy=True, getForces=True, groups={QM_GROUP})
                row = backend.timings.rows[-1]
                results[f] = {
                    "energy_kj_mol": np.array(s.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)),
                    "forces": s.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer),
                    "n_embed_groups": np.array(int(row["n_embed_groups"])),
                    "t_orca": np.array(float(row["t_orca"])),
                }
                save(path, results)
                e = float(results[f]["energy_kj_mol"]) / HARTREE_IN_KJ_MOL
                print(f"cutoff {cutoff:.3f} nm frame {f} step {int(frames['step'][f])}: E_QM {e:.8f} Eh, "
                      f"{int(row['n_embed_groups'])} groups, t_orca {float(row['t_orca']):.1f} s, "
                      f"wall {time.perf_counter() - t0:.1f} s", flush=True)
        finally:
            potential.close()


if __name__ == "__main__":
    main()
