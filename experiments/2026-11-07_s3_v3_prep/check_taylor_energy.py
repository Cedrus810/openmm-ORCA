"""Task 0, item 1: S3 energies with analytic Taylor coefficients (order 2) on one pilot frame (PySCF).

Same comparison as V1 (R = E - E_ENV20, pilot ENV20): direct order-2 polynomial potential, and pseudo
charges Q = W c with W a fixed ridge fit (K=50, R_s=2.0 nm, lam=1e-4) on a uniform template ball of
radius 0.4 / 0.6 / 0.8 nm about rc (2000 points), as in the V3 plan Task 1.
Usage: check_taylor_energy.py FRAME [--threads N]   (writes out/candidate_XX.json)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from pyscf import gto, lib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "2026-11-07_s3_v1_pyscf"))
from s3_v1 import EH, KE, monomials, nuclear_term, poly_operator, run_scf, sphere  # noqa: E402
from taylor import EXPS2, taylor  # noqa: E402

K, RS, LAM, NT = 50, 2.0, 1e-4, 2000


def template_W(radius, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(NT, 3)); d /= np.linalg.norm(d, axis=1)[:, None]
    x = d * (radius * rng.random(NT) ** (1 / 3))[:, None]
    u = sphere(K, RS, np.zeros(3))
    G = KE / np.linalg.norm(x[:, None] - u[None], axis=-1)
    A, _ = monomials(x, 2)
    W = np.linalg.solve(G.T @ G + LAM * NT * np.eye(K), G.T @ A)    # (K, 10): Q = W c
    res = G @ W - A                                                   # per-monomial representation error
    return u, W, float(np.sqrt((res ** 2).mean(0)).max())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("frame", type=int)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    lib.num_threads(a.threads)
    f = a.frame
    req = np.load(HERE.parent / "2026-11-06_c2_matched_pilot/inputs/requests_A.npz")
    qm, mm, q = (req[f"S2_{k}_{f}"].astype(float) for k in ("qm_pos", "mm_pos", "mm_q"))
    L = float(req["box_nm"][f][0, 0])
    elements = req["qm_elements"].tolist()
    env20 = json.loads((HERE.parent / f"2026-11-06_c2_matched_pilot/pyscf/candidate_{f:02d}_ENV20.json")
                       .read_text())["energy_eh"]
    rc = qm.mean(0)
    c, _ = taylor(rc, mm, q, L, alpha=1.2, kmax=10)
    mol = gto.M(atom=list(zip(elements, (qm * 10).tolist())), basis="def2-svp", charge=-1, spin=0,
                unit="Angstrom", verbose=3)
    mol.max_memory = 6000
    E, wall = {}, {}
    t = time.perf_counter(); E["S2"] = run_scf(mol, mm, q); wall["S2"] = time.perf_counter() - t
    h = poly_operator(mol, rc, c, EXPS2)
    en = nuclear_term(mol, qm, rc, c, 2)
    t = time.perf_counter(); E["T2_direct"] = run_scf(mol, mm, q, h, en); wall["T2_direct"] = time.perf_counter() - t
    pseudo = {}
    for radius in (0.4, 0.6, 0.8):
        u, W, rep = template_W(radius)
        Q = W @ c
        sites = rc + u
        t = time.perf_counter()
        E[f"T2_P_r{radius}"] = run_scf(mol, np.vstack([mm, sites]), np.concatenate([q, Q]))
        wall[f"T2_P_r{radius}"] = time.perf_counter() - t
        pseudo[str(radius)] = dict(Q_absmax=float(np.abs(Q).max()), Q_sum=float(Q.sum()),
                                   max_monomial_rep_rms=rep)
    rec = dict(candidate=f, c=c.tolist(), energies_eh=E, env20_eh=env20,
               R_kj_mol={k: (v - env20) * EH for k, v in E.items()}, pseudo=pseudo, wall_s=wall,
               ewald=dict(alpha=1.2, kmax=10), template=dict(K=K, Rs=RS, lam=LAM, n_points=NT))
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / f"candidate_{f:02d}.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps({k: rec[k] for k in ("candidate", "R_kj_mol", "pseudo")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
