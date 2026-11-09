"""V5 capture: S2 requests of every saved frame in the given C2 scan windows, plus the S3 pseudo charges.

Zero-force capture backend through createMixedSystem(..., embeddingCutoff=None), as prepare_pilot.py;
pseudo charges from the package EwaldCorrection (centre = mean of the QM request positions).
Writes requests/wXX.npz (skipped if present).  Usage: capture_windows.py W [W ...]   (openmm_dev)
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
SCAN = ROOT / "experiments/2026-11-01_c2_rc_scan/scan"


def main():
    windows = [int(w) for w in sys.argv[1:]]
    out = HERE / "requests"
    out.mkdir(exist_ok=True)
    todo = [w for w in windows if not (out / f"w{w:02d}.npz").exists()]
    if not todo:
        return
    import openmm as mm
    import openmm.app as app
    from openmm import unit
    import enzyme_qmmm as enzyme
    from openmmorca import ORCAPotential
    from openmmorca.backend.base import QMResult
    from openmmorca.qmmm.ewald_correction import EwaldCorrection

    class Capture:
        request = None

        def evaluate(self, request):
            self.request = request
            return QMResult(0., np.zeros((request.n_qm, 3)), np.zeros((request.n_mm, 3)))

        def close(self):
            pass

    meta = json.loads((SCAN / "meta.json").read_text())
    topology = app.PDBFile(str(Path(meta["run_dir"]) / "prepared.pdb")).topology
    system = enzyme.mm_system(topology, enzyme.forcefield_with_dce())
    with np.load(SCAN / "w00.npz") as d0:
        box0 = d0["box_nm"][0]
    system.setDefaultPeriodicBoxVectors(*box0)
    topology.setPeriodicBoxVectors(box0 * unit.nanometer)
    qm, pairs, *_ = enzyme.qm_selection(topology, "A")
    cap = Capture()
    pot = ORCAPotential(method="r2SCAN-3c", charge=-1, backend_factory=lambda: cap)
    mixed = pot.createMixedSystem(topology, system, qm, boundaryPairs=pairs, forceGroup=31, embeddingCutoff=None)
    context = mm.Context(mixed, mm.VerletIntegrator(.0005), mm.Platform.getPlatformByName("CPU"))
    corr = EwaldCorrection()
    for w in todo:
        with np.load(SCAN / f"w{w:02d}.npz") as d:
            pos, boxes, steps, rc = d["positions_nm"], d["box_nm"], d["step"], d["rc"]
        keep = {k: [] for k in ("qm_pos", "mm_pos", "mm_q", "sites", "Q", "c", "box")}
        for fr in range(len(steps)):
            context.setPeriodicBoxVectors(*boxes[fr])
            context.setPositions(pos[fr].astype(float))
            context.getState(getEnergy=True, groups={31})
            r = cap.request
            step = corr.prepare(r.qm_positions_nm.mean(0), r.mm_positions_nm, r.mm_charges_e, boxes[fr].astype(float))
            for k, v in (("qm_pos", r.qm_positions_nm), ("mm_pos", r.mm_positions_nm), ("mm_q", r.mm_charges_e),
                         ("sites", step.pseudo_positions_nm), ("Q", step.pseudo_charges_e), ("c", step.coefficients),
                         ("box", boxes[fr])):
                keep[k].append(np.asarray(v, float))
        np.savez(out / f"w{w:02d}.npz", step=steps, rc=rc, qm_elements=np.array(r.qm_elements),
                 **{k: np.array(v) for k, v in keep.items()})
        print(f"window {w:02d}: {len(steps)} frames, Qmax {np.abs(keep['Q']).max():.3f}", flush=True)
    pot.close()


if __name__ == "__main__":
    main()
