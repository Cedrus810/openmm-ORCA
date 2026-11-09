"""V4: S3 pseudo charges for the 20 B5/T09 frames with the package's EwaldCorrection (openmm_dev).

Centre = mean of the S2 QM request positions (link H included), as in QMMMCallback.
Writes pseudo_b5.npz (sites_f, Q_f, c_f for f = 0..19).
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from openmmorca.qmmm.ewald_correction import EwaldCorrection  # noqa: E402

req = np.load(ROOT / "experiments/2026-10-29_b5_scope/requests_t09_A.npz")
corr = EwaldCorrection()
out = {}
for f in range(20):
    qm, mm, q = (req[f"S2_{k}_{f}"].astype(float) for k in ("qm_pos", "mm_pos", "mm_q"))
    step = corr.prepare(qm.mean(0), mm, q, req["box_nm"][f].astype(float))
    out[f"sites_{f}"], out[f"Q_{f}"], out[f"c_{f}"] = step.pseudo_positions_nm, step.pseudo_charges_e, step.coefficients
    print(f, "Qmax", round(float(np.abs(step.pseudo_charges_e).max()), 3), "Qsum", round(float(step.pseudo_charges_e.sum()), 3),
          "|g|", round(float(np.linalg.norm(step.coefficients[1:4])), 1), flush=True)
np.savez(HERE / "pseudo_b5.npz", **out)
