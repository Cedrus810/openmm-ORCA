"""Classical check of the pseudo-charge fit (proposal option P): charge magnitudes vs fit error.

For each pilot frame: target = order-2 dV polynomial (coefficients from out/candidate_XX.json);
pseudo charges on a Fibonacci sphere (K, R_s) about the centroid; ridge fit with penalty lam*|Q|^2.
The harmonic part of the polynomial (trace of the quadratic part removed) is reported separately,
since charges outside the QM region can only produce harmonic potentials.
Usage: pseudo_conditioning.py   (writes pseudo_conditioning.json)
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-11-07_c2_error_decomp"))
sys.path.insert(0, str(HERE))
from classical import KE, monomials  # noqa: E402
from s3_v1 import sphere  # noqa: E402

req = np.load(HERE.parent / "2026-11-06_c2_matched_pilot/inputs/requests_A.npz")
out = []
for f in (0, 4, 8):
    rec = json.loads((HERE / "out" / f"candidate_{f:02d}.json").read_text())
    coef = np.array(rec["fits"]["2"]["coef"])
    rc = np.array(rec["centroid_nm"])
    qm = req[f"S2_qm_pos_{f}"]
    rng = np.random.default_rng(7 + f)
    d = rng.normal(size=(800, 3)); d /= np.linalg.norm(d, axis=1)[:, None]
    pts = qm[rng.integers(len(qm), size=800)] + d * (0.4 * rng.random(800) ** (1 / 3))[:, None]
    A, exps = monomials(pts - rc, 2)
    target = A @ coef
    trace = sum(coef[k] for k, e in enumerate(exps) if sorted(e) == [0, 0, 2])  # c_xx + c_yy + c_zz
    r2 = ((pts - rc) ** 2).sum(1)
    harmonic = target - trace / 3 * r2
    for K in (20, 50, 100):
        for R in (1.5, 2.0, 3.0):
            s = sphere(K, R, rc)
            G = KE / np.linalg.norm(pts[:, None] - s[None], axis=-1)
            for lam in (0.0, 1e-4, 1e-2):
                M = G.T @ G + lam * len(pts) * np.eye(K)
                Q = np.linalg.solve(M, G.T @ target) if lam else np.linalg.lstsq(G, target, rcond=1e-12)[0]
                out.append(dict(frame=f, K=K, R=R, lam=lam, Qmax=float(np.abs(Q).max()), Qsum=float(Q.sum()),
                                rms_vs_poly2=float(np.sqrt(((G @ Q - target) ** 2).mean())),
                                rms_vs_harmonic=float(np.sqrt(((G @ Q - harmonic) ** 2).mean())),
                                trace_part_rms=float(np.sqrt(((trace / 3 * (r2 - r2.mean())) ** 2).mean())),
                                laplacian_kj_mol_e_nm2=float(2 * trace)))
(HERE / "pseudo_conditioning.json").write_text(json.dumps(out, indent=1) + "\n")
print("frame K R lam | Qmax | rms vs poly2 | rms vs harmonic part | trace-part rms | laplacian")
for r in out:
    print(f"{r['frame']} {r['K']:3d} {r['R']:.1f} {r['lam']:.0e} | {r['Qmax']:8.2f} | {r['rms_vs_poly2']:.4f} | "
          f"{r['rms_vs_harmonic']:.4f} | {r['trace_part_rms']:.4f} | {r['laplacian_kj_mol_e_nm2']:.3f}")
