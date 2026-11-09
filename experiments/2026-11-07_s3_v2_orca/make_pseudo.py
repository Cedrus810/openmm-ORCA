"""Regenerate the V1 regularized pseudo charges (K=50, R_s=2.0 nm, lam=1e-4) for the three pilot frames.

Same functions, seeds and fit as s3_v1.py --K 50 --Rs 2.0 --lam 1e-4; checks Q_absmax/Q_sum against
the V1 JSON records.  Run with the pyscf-env python.  Writes pseudo_K50_R2.0_lam1e-4.npz.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / "2026-11-07_s3_v1_pyscf"
sys.path.insert(0, str(V1))
from s3_v1 import KE, fit_dV, monomials, sphere  # noqa: E402

K, RS, LAM, NF = 50, 2.0, 1e-4, 500
req = np.load(HERE.parent / "2026-11-06_c2_matched_pilot/inputs/requests_A.npz")
out = {}
for f in (0, 4, 8):
    qm, mm, q = (req[f"S2_{k}_{f}"].astype(float) for k in ("qm_pos", "mm_pos", "mm_q"))
    L = float(req["box_nm"][f][0, 0])
    rc, pts, dV, fits = fit_dV(qm, mm, q, L, seed=20261107 + f)
    sites = sphere(K, RS, rc)
    A2, _ = monomials(pts - rc, 2)
    target = A2 @ fits[2]["coef"]
    G = KE / np.linalg.norm(pts[:, None, :] - sites[None], axis=-1)
    Q = np.linalg.solve(G[:NF].T @ G[:NF] + LAM * NF * np.eye(K), G[:NF].T @ target[:NF])
    v1 = json.loads((V1 / "out" / f"candidate_{f:02d}_P_K50_R2.0_lam0.0001.json").read_text())["pseudo"]
    print(f, "Qmax", np.abs(Q).max(), "V1", v1["Q_absmax"], "Qsum", Q.sum(), "V1", v1["Q_sum"], flush=True)
    assert abs(np.abs(Q).max() - v1["Q_absmax"]) < 1e-10 and abs(Q.sum() - v1["Q_sum"]) < 1e-10
    out[f"sites_{f}"], out[f"Q_{f}"] = sites, Q
np.savez(HERE / "pseudo_K50_R2.0_lam1e-4.npz", **out)
