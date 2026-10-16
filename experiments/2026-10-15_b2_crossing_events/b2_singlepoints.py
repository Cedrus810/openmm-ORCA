"""Roadmap B2, step 2: QM/MM single points with an explicitly chosen set of embedded groups.

Runs the jobs written by b2_events.py: each job is (frame, set of group indices). The selection
inside the PythonForce callback is overridden so that exactly that set is embedded; everything else
(link atoms, charge shift, group placement at the image nearest the QM region, force
redistribution) goes through the unchanged ORCAPotential path, as in the T09 scan (CPU platform,
only the QM/MM force group is evaluated).

Override: CutoffEmbedding.select is wrapped. It runs the original selection with a larger radius
(--reach, nm) and keeps only the requested groups. A group's placement depends only on its own
distances to the QM region, not on the cutoff, so the kept groups are placed exactly as the original
selection would place them. On the first job whose set equals the natural 1.2 nm selection this is
checked against the original output array by array.

Results: OUTDIR/sp_w<worker>.npz, rewritten after every job; rerunning the same command skips done
jobs (resumable). Stored per job: energy (kJ/mol), indices and forces (kJ/mol/nm) of atoms with a
nonzero QM/MM-group force, number of embedded groups, ORCA time.

Usage: b2_singlepoints.py JOBS.json FRAMES.npz RUN_DIR OUTDIR --worker I --n-workers N
       [--jobs ID ...] [--method r2SCAN-3c] [--nprocs 18] [--reach 2.0]
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

from openmmorca import ORCAPotential  # noqa: E402
from openmmorca.qmmm import embedding as embedding_mod  # noqa: E402
from openmmorca.units import HARTREE_IN_KJ_MOL  # noqa: E402

QM_GROUP = 1
OVERRIDE = {"groups": None, "reach": 2.0, "checked": False}
_original_select = embedding_mod.CutoffEmbedding.select


def _select_override(self, qm_positions_nm, positions_nm, box_vectors_nm):
    want = OVERRIDE["groups"]
    if want is None:
        return _original_select(self, qm_positions_nm, positions_nm, box_vectors_nm)
    cutoff = self.cutoff_nm
    natural_atoms, natural_pos, natural_q = _original_select(self, qm_positions_nm, positions_nm, box_vectors_nm)
    natural = set(self._previous)
    try:
        self.cutoff_nm = OVERRIDE["reach"]
        atoms, positions, _ = _original_select(self, qm_positions_nm, positions_nm, box_vectors_nm)
    finally:
        self.cutoff_nm = cutoff
    group_of_atom = dict(zip(self._atoms.tolist(), self._group_of.tolist()))
    keep = np.array([group_of_atom[int(a)] in want for a in atoms], dtype=bool)
    present = {group_of_atom[int(a)] for a in atoms[keep]}
    missing = set(want) - present
    if missing:
        raise RuntimeError(f"groups {sorted(missing)} are beyond the reach {OVERRIDE['reach']} nm")
    atoms, positions = atoms[keep], positions[keep]
    if set(want) == natural and not OVERRIDE["checked"]:
        assert np.array_equal(atoms, natural_atoms) and np.array_equal(positions, natural_pos), \
            "override placement differs from the original selection"
        OVERRIDE["checked"] = True
    self._previous = natural  # last_changed is then counted against the natural set
    self._finish(set(want), atoms, positions)
    return atoms, positions, self.charges_e[atoms]


embedding_mod.CutoffEmbedding.select = _select_override


def load(path):
    if not path.is_file():
        return {}
    with np.load(path) as data:
        return {int(j): {k: data[f"{k}_{j}"] for k in ("energy_kj_mol", "force_atoms", "forces", "n_groups",
                                                        "natural_equal", "t_orca")}
                for j in data["jobs_done"]}


def save(path, results):
    arrays = {"jobs_done": np.array(sorted(results), dtype=int)}
    for j, r in results.items():
        arrays.update({f"{k}_{j}": v for k, v in r.items()})
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs_json", type=Path)
    parser.add_argument("frames_npz", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--jobs", type=int, nargs="*", default=None)
    parser.add_argument("--method", default="r2SCAN-3c")
    parser.add_argument("--nprocs", type=int, default=18)
    parser.add_argument("--reach", type=float, default=2.0)
    parser.add_argument("--region", default="A")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    OVERRIDE["reach"] = args.reach

    spec = json.loads(args.jobs_json.read_text())
    jobs = spec["jobs"]
    if args.jobs is not None:
        jobs = [jobs[i] for i in args.jobs]
    jobs = [j for j in jobs if j["id"] % args.n_workers == args.worker]
    frames = np.load(args.frames_npz)
    row_of = {int(f): i for i, f in enumerate(frames["frame"])}

    path = args.outdir / f"sp_w{args.worker}.npz"
    results = load(path)
    todo = [j for j in jobs if j["id"] not in results]
    print(f"worker {args.worker}/{args.n_workers}: {len(jobs)} jobs, {len(todo)} to do", flush=True)
    if not todo:
        return

    pdb = app.PDBFile(str(args.run_dir / "prepared.pdb"))
    topology = pdb.topology
    state = mm.XmlSerializer.deserialize((args.run_dir / "equilibrated.xml").read_text())
    system = enzyme.mm_system(topology, enzyme.forcefield_with_dce())
    system.setDefaultPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
    qm, pairs, *_ = enzyme.qm_selection(topology, args.region)

    potential = ORCAPotential(method=args.method, charge=enzyme.QMMM_RECIPE["charge"],
                              multiplicity=enzyme.QMMM_RECIPE["multiplicity"], nprocs=args.nprocs)
    try:
        mixed = potential.createMixedSystem(topology, system, qm, boundaryPairs=pairs, forceGroup=QM_GROUP,
                                            embeddingCutoff=spec["cutoff_nm"] * unit.nanometer)
        context = mm.Context(mixed, mm.VerletIntegrator(0.0005), mm.Platform.getPlatformByName("CPU"))
        backend = potential.backends[0]
        for j in todo:
            r = row_of[j["frame"]]
            OVERRIDE["groups"] = set(j["groups"])
            context.setPeriodicBoxVectors(*frames["box_nm"][r])
            context.setPositions(frames["positions_nm"][r])
            t0 = time.perf_counter()
            s = context.getState(getEnergy=True, getForces=True, groups={QM_GROUP})
            row = backend.timings.rows[-1]
            forces = s.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
            nz = np.nonzero(np.any(forces != 0.0, axis=1))[0]
            results[j["id"]] = {
                "energy_kj_mol": np.array(s.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)),
                "force_atoms": nz.astype(np.int32), "forces": forces[nz],
                "n_groups": np.array(int(row["n_embed_groups"])),
                "natural_equal": np.array(int(row["embed_changed"]) == 0),
                "t_orca": np.array(float(row["t_orca"])),
            }
            assert int(row["n_embed_groups"]) == len(j["groups"])
            if j["kind"] == "base":
                assert int(row["embed_changed"]) == 0, f"job {j['id']}: base set differs from natural selection"
            save(path, results)
            e = float(results[j["id"]]["energy_kj_mol"]) / HARTREE_IN_KJ_MOL
            print(f"job {j['id']} {j['kind']} frame {j['frame']} step {j['step']} set_from {j['set_from']}: "
                  f"E_QM {e:.8f} Eh, {len(j['groups'])} groups, t_orca {float(row['t_orca']):.1f} s, "
                  f"wall {time.perf_counter() - t0:.1f} s", flush=True)
    finally:
        potential.close()


if __name__ == "__main__":
    main()
