"""Classical S2-vs-Ewald field difference on every saved C2 scan frame (no QM).

For each frame: capture the S2 request through the existing createMixedSystem callback (zero-force
capture backend, as prepare_pilot.py), then fit dV = V_Ewald - V_S2 about the QM centroid (3rd
order) and record c0, the gradient g = grad dV, the S2 cluster dipole M about the centroid, and the
recomputed RC.  Output: OUTDIR/wXX.npz per window.

Usage: field_stats.py OUTDIR WINDOW [WINDOW ...]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(HERE))
from classical import KE, field_fit  # noqa: E402

SCAN = ROOT / "experiments/2026-11-01_c2_rc_scan/scan"


def main():
    outdir = Path(sys.argv[1])
    windows = [int(w) for w in sys.argv[2:]]
    outdir.mkdir(parents=True, exist_ok=True)

    import openmm as mm
    import openmm.app as app
    from openmm import unit
    import enzyme_qmmm as enzyme
    from openmmorca import ORCAPotential
    from openmmorca.backend.base import QMResult

    class Capture:
        request = None

        def evaluate(self, request):
            self.request = request
            return QMResult(0., np.zeros((request.n_qm, 3)), np.zeros((request.n_mm, 3)))

        def close(self):
            pass

    meta = json.loads((SCAN / "meta.json").read_text())
    pdb = app.PDBFile(str(Path(meta["run_dir"]) / "prepared.pdb"))
    topology = pdb.topology
    names = [a.name for a in topology.atoms()]
    o, c, cl = (meta["atoms"][k] for k in ("O", "C1", "Cl1"))
    system = enzyme.mm_system(topology, enzyme.forcefield_with_dce())
    with np.load(SCAN / "w00.npz") as d0:
        box0 = d0["box_nm"][0]
    system.setDefaultPeriodicBoxVectors(*box0)
    topology.setPeriodicBoxVectors(box0 * unit.nanometer)
    qm, pairs, *_ = enzyme.qm_selection(topology, "A")
    k_o, k_c, k_cl = (qm.index(i) for i in (o, c, cl))
    capture = Capture()
    pot = ORCAPotential(method="r2SCAN-3c", charge=-1, backend_factory=lambda: capture)
    mixed = pot.createMixedSystem(topology, system, qm, boundaryPairs=pairs, forceGroup=31, embeddingCutoff=None)
    context = mm.Context(mixed, mm.VerletIntegrator(.0005), mm.Platform.getPlatformByName("CPU"))
    t0 = time.perf_counter()
    for w in windows:
        with np.load(SCAN / f"w{w:02d}.npz") as data:
            pos_all, boxes, steps, rc_saved = data["positions_nm"], data["box_nm"], data["step"], data["rc"]
        rows = {k: [] for k in ("rc", "rc_saved", "step", "c0", "g", "hess", "cubic", "test_rms", "test_max",
                                "M", "centroid", "qm_pos", "dV_sites", "n_mm", "mm_charge")}
        for fr in range(len(steps)):
            box = boxes[fr].astype(float)
            context.setPeriodicBoxVectors(*box)
            context.setPositions(pos_all[fr].astype(float))
            capture.request = None
            context.getState(getEnergy=True, groups={31})
            r = capture.request
            qp, mp, q = r.qm_positions_nm, r.mm_positions_nm, r.mm_charges_e
            L = float(box[0, 0])
            assert np.allclose(box, np.diag([L] * 3))
            fit = field_fit(qp, mp, q, L, seed=1000 * w + fr)
            rows["rc"].append(np.linalg.norm(qp[k_cl] - qp[k_c]) - np.linalg.norm(qp[k_c] - qp[k_o]))
            rows["rc_saved"].append(rc_saved[fr])
            rows["step"].append(steps[fr])
            rows["c0"].append(fit["coef"][0])
            rows["g"].append(fit["coef"][1:4])
            rows["hess"].append(fit["coef"][4:10])
            rows["cubic"].append(fit["coef"][10:])
            rows["test_rms"].append(fit["test_rms"])
            rows["test_max"].append(fit["test_max"])
            rows["M"].append((q[:, None] * (mp - fit["centroid"])).sum(0))
            rows["centroid"].append(fit["centroid"])
            rows["qm_pos"].append(qp)
            rows["dV_sites"].append(fit["dV_sites"])
            rows["n_mm"].append(len(q))
            rows["mm_charge"].append(q.sum())
        np.savez(outdir / f"w{w:02d}.npz", box_L=L, KE=KE, qm_names=np.array([names[i] for i in qm] + ["HL"]),
                 **{k: np.array(v) for k, v in rows.items()})
        print(f"window {w:02d}: {len(steps)} frames, {time.perf_counter() - t0:.0f} s", flush=True)
    pot.close()


if __name__ == "__main__":
    main()
