"""V3 plan Task 5: the S3 API path on the three C2 pilot frames.

mode capture: createMixedSystem(..., embeddingCutoff=None, ewaldCorrection=True) with a zero-result
  capture backend (as prepare_pilot.py); saves the full requests (S2 charges + pseudo charges) and
  checks the real part against the pilot S2 requests and the pseudo charges against the Task 0
  prototype (taylor.py + the same template).  -> capture.npz, capture.json
mode orca FRAME: the same path with the real ORCA backend (r2SCAN-3c, nprocs 1, fresh SCF) for one
  frame: QM-group energy, forces, timings (t_vpot, t_ewald).  -> orca_XX.json
Run with the openmm_dev python.
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
PILOT = ROOT / "experiments/2026-11-06_c2_matched_pilot"
FRAMES = (0, 4, 8)


def build(backend_factory):
    import openmm as mm
    import openmm.app as app
    from openmm import unit
    import enzyme_qmmm as enzyme
    from openmmorca import ORCAPotential

    sel = json.loads((PILOT / "inputs/selection.json").read_text())
    scan_meta = json.loads((Path(sel["scan"]) / "meta.json").read_text())
    pdb = app.PDBFile(str(Path(scan_meta["run_dir"]) / "prepared.pdb"))
    topology = pdb.topology
    arrays = np.load(PILOT / "inputs/selected_frames.npz")
    system = enzyme.mm_system(topology, enzyme.forcefield_with_dce())
    box = arrays["box_nm"][0]
    system.setDefaultPeriodicBoxVectors(*box)
    topology.setPeriodicBoxVectors(box * unit.nanometer)
    qm, pairs, *_ = enzyme.qm_selection(topology, "A")
    pot = ORCAPotential(method="r2SCAN-3c", charge=-1, nprocs=1, restart=False,
                        scratch_root="/dev/shm/openmmorca-s3-api", backend_factory=backend_factory)
    mixed = pot.createMixedSystem(topology, system, qm, boundaryPairs=pairs, forceGroup=31,
                                  embeddingCutoff=None, ewaldCorrection=True)
    context = mm.Context(mixed, mm.VerletIntegrator(.0005), mm.Platform.getPlatformByName("CPU"))
    return pot, context, arrays


def capture():
    from openmmorca.backend.base import QMResult

    class Capture:
        request = None

        def evaluate(self, request):
            self.request = request
            return QMResult(0., np.zeros((request.n_qm, 3)), np.zeros((request.n_mm, 3)),
                            esp_kj_mol_e=np.zeros(request.n_esp))

        def close(self):
            pass

    cap = Capture()
    pot, context, arrays = build(lambda: cap)
    req = np.load(PILOT / "inputs/requests_A.npz")
    sys.path.insert(0, str(ROOT / "experiments/2026-11-07_s3_v3_prep"))
    from taylor import KE, monomials, taylor

    # Task 0 template (check_taylor_energy.template_W(0.6)); copied: that module imports PySCF
    rng = np.random.default_rng(0)
    d = rng.normal(size=(2000, 3)); d /= np.linalg.norm(d, axis=1)[:, None]
    x = d * (0.6 * rng.random(2000) ** (1 / 3))[:, None]
    i = np.arange(50) + 0.5
    phi, theta = np.arccos(1 - 2 * i / 50), np.pi * (1 + 5 ** 0.5) * i
    u = 2.0 * np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)
    G = KE / np.linalg.norm(x[:, None] - u[None], axis=-1)
    A, _ = monomials(x, 2)
    W = np.linalg.solve(G.T @ G + 1e-4 * 2000 * np.eye(50), G.T @ A)
    out, checks = {}, {}
    for f in FRAMES:
        context.setPeriodicBoxVectors(*arrays["box_nm"][f])
        context.setPositions(arrays["positions_nm"][f])
        context.getState(getEnergy=True, groups={31})
        r = cap.request
        n_real = r.n_mm - 50
        same = (np.array_equal(r.qm_positions_nm, req[f"S2_qm_pos_{f}"])
                and np.array_equal(r.mm_positions_nm[:n_real], req[f"S2_mm_pos_{f}"])
                and np.array_equal(r.mm_charges_e[:n_real], req[f"S2_mm_q_{f}"]))
        rc = r.qm_positions_nm.mean(0)
        c_proto, _ = taylor(rc, req[f"S2_mm_pos_{f}"].astype(float), req[f"S2_mm_q_{f}"].astype(float),
                            float(arrays["box_nm"][f][0, 0]), alpha=1.2, kmax=10)
        Q_proto = W @ c_proto
        checks[f] = dict(real_part_identical_to_pilot=bool(same),
                         pseudo_positions_max_diff_nm=float(np.abs(r.mm_positions_nm[n_real:] - (rc + u)).max()),
                         pseudo_charges_max_diff_e=float(np.abs(r.mm_charges_e[n_real:] - Q_proto).max()),
                         esp_points_are_pseudo=bool(np.array_equal(r.esp_points_nm, r.mm_positions_nm[n_real:])),
                         diagnostics={k: float(v) for k, v in r.diagnostics.items()})
        out[f"qm_pos_{f}"], out[f"mm_pos_{f}"], out[f"mm_q_{f}"] = r.qm_positions_nm, r.mm_positions_nm, r.mm_charges_e
        out["qm_elements"] = np.array(r.qm_elements)
    np.savez(HERE / "capture.npz", **out)
    (HERE / "capture.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=1))


def orca(f):
    from openmm import unit

    pot, context, arrays = build(None)
    context.setPeriodicBoxVectors(*arrays["box_nm"][f])
    context.setPositions(arrays["positions_nm"][f])
    t = time.perf_counter()
    state = context.getState(getEnergy=True, getForces=True, groups={31})
    wall = time.perf_counter() - t
    backend = pot.backends[0]
    row = backend.timings.rows[-1]
    rec = dict(candidate=f, energy_kj_mol=state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
               wall_s=wall, timings_row={k: (float(v) if not isinstance(v, bool) else v) for k, v in row.items()},
               n_qm_calls=len(backend.timings.rows))
    pot.close()
    (HERE / f"orca_{f:02d}.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "capture":
        capture()
    else:
        orca(int(sys.argv[2]))
