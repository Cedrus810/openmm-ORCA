"""V3 plan Task 4: finite-difference check of S3 forces on the DhlA pilot frame (candidate 0).

Path: createMixedSystem(..., embeddingCutoff=None, ewaldCorrection=True), real ORCA r2SCAN-3c
VeryTightSCF, nprocs 1, fresh SCF; only the QM/MM force group (31) is evaluated.
Modes:
  base                 energy and forces at the frame          -> out/base.json
  fd ATOM AXIS SIGN    energy with x[ATOM, AXIS] += SIGN * H   -> out/fd_<atom>_<axis>_<p|m>.json
ATOM is a label: C1, Cl1, O (nucleophile OD2), Q1 (Asp CB), M1 (Asp CA), farO (farthest water O).
Existing outputs are skipped, so rerunning the same command resumes.  H = 1e-4 nm.
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
PILOT = ROOT / "experiments/2026-11-06_c2_matched_pilot"
FRAME, H = 0, 1e-4


def setup(tag):
    import openmm as mm
    import openmm.app as app
    from openmm import unit
    import enzyme_qmmm as enzyme
    from openmmorca import ORCAPotential
    from openmmorca.qmmm.imaging import minimum_image

    sel = json.loads((PILOT / "inputs/selection.json").read_text())
    meta = json.loads((Path(sel["scan"]) / "meta.json").read_text())
    pdb = app.PDBFile(str(Path(meta["run_dir"]) / "prepared.pdb"))
    topology = pdb.topology
    arrays = np.load(PILOT / "inputs/selected_frames.npz")
    pos = arrays["positions_nm"][FRAME].astype(float)
    box = arrays["box_nm"][FRAME].astype(float)
    system = enzyme.mm_system(topology, enzyme.forcefield_with_dce())
    system.setDefaultPeriodicBoxVectors(*box)
    topology.setPeriodicBoxVectors(box * unit.nanometer)
    qm, pairs, *_ = enzyme.qm_selection(topology, "A")
    q1, m1 = pairs[0][0], pairs[0][1]
    waters_o = [a.index for a in topology.atoms() if a.residue.name == "HOH" and a.name == "O"]
    d = np.linalg.norm(minimum_image(pos[waters_o] - pos[qm[0]], box), axis=1)
    atoms = dict(C1=meta["atoms"]["C1"], Cl1=meta["atoms"]["Cl1"], O=meta["atoms"]["O"], Q1=q1, M1=m1,
                 farO=int(waters_o[int(np.argmax(d))]))
    pot = ORCAPotential(method="r2SCAN-3c", charge=-1, nprocs=1, restart=False,
                        extra_keywords=("VeryTightSCF",), scratch_root=f"/dev/shm/openmmorca-s3fd-{tag}",
                        timeout_s=3600)
    mixed = pot.createMixedSystem(topology, system, qm, boundaryPairs=pairs, forceGroup=31,
                                  embeddingCutoff=None, ewaldCorrection=True)
    context = mm.Context(mixed, mm.VerletIntegrator(.0005), mm.Platform.getPlatformByName("CPU"))
    context.setPeriodicBoxVectors(*box)
    # Image changes need an atom within the displacement of a face of the anchor-centred cell:
    # report the smallest per-axis margin |L/2| - |rel| over all atoms (anchor = first QM atom).
    rel = minimum_image(pos - pos[qm[0]], box)
    margin = float((np.diag(box) / 2 - np.abs(rel)).min())
    return pot, context, pos, atoms, float(d.max()), margin


def main():
    from openmm import unit

    mode = sys.argv[1]
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    if mode == "base":
        tag = "base"
    else:
        label, axis, sign = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
        tag = f"fd_{label}_{axis}_{'p' if sign > 0 else 'm'}"
    path = out / f"{tag}.json"
    if path.exists():
        print(f"skip existing {path}", flush=True)
        return
    t0 = time.perf_counter()
    pot, context, pos, atoms, far_d, margin = setup(tag)
    t_setup = time.perf_counter() - t0
    trial = pos.copy()
    if mode != "base":
        trial[atoms[label], axis] += sign * H
    context.setPositions(trial)
    t = time.perf_counter()
    state = context.getState(getEnergy=True, getForces=(mode == "base"), groups={31})
    rec = dict(mode=mode, tag=tag, frame=FRAME, h_nm=H, atoms=atoms, far_water_o_dist_nm=far_d,
               min_face_margin_nm=margin,
               energy_kj_mol=state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
               wall_qm_s=time.perf_counter() - t, wall_setup_s=t_setup, host=platform.node(),
               timings_row={k: (v if isinstance(v, bool) else float(v))
                            for k, v in pot.backends[0].timings.rows[-1].items()},
               pid=os.getpid())
    if mode == "base":
        f = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        rec["forces"] = {k: f[i].tolist() for k, i in atoms.items()}
        rec["force_sum"] = f.sum(axis=0).tolist()
        rec["force_absmax"] = float(np.abs(f).max())
    pot.close()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=2) + "\n")
    tmp.replace(path)
    print(json.dumps({k: v for k, v in rec.items() if k != "forces"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
