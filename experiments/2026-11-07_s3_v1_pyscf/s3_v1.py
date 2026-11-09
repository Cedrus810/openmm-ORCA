"""S3 proposal step V1: PySCF prototype of S2 + classical Ewald correction on the C2 pilot frames.

Quantity compared: R_v(f) = E_v(f) - E_ENV20(f) (kJ/mol), same QM geometry, method and S2 charges
as the 2026-11-07 pilot; the reference is the pilot ENV20 energy (PySCF periodic MM field without QM
self images).  The S2 baseline is R_S2 = D(f): relative changes -10.26 / -16.09 kJ/mol.

Variants (dV = V_Ewald - V_S2 of the S2 charges, fitted about the QM centroid, classical.py):
  S3_o1, S3_o2, S3_o3   dV_poly of order 1/2/3 added to the SCF as an external potential
                        (electrons: one-electron integrals of the polynomial; nuclei: constant)
  S3_P2                 order-2 dV_poly represented by K pseudo charges on a sphere of radius R_s
                        around the centroid, appended to the S2 point charges (proposal option P)

Usage: s3_v1.py REQUESTS.npz PILOT_PYSCF_DIR OUTDIR FRAME [--threads N] [--K 100] [--Rs 2.0]
       [--lam 0] [--only-pseudo]   (ridge penalty lam*n_points*|Q|^2; --only-pseudo runs S3_P2 only and
       writes candidate_XX_P_K{K}_R{Rs}_lam{lam}.json)
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from pyscf import dft, gto, lib, qmmm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-11-07_c2_error_decomp"))
from classical import KE, direct, ewald, monomials, units  # noqa: E402

EH = units.HARTREE_IN_KJ_MOL
BOHR_NM = units.BOHR_IN_ANGSTROM / 10.0


def fit_dV(qm, mm, q, L, seed):
    """Exact dV on 700 points in the QM region; fits of order 1-3 (500 fit, 200 test points)."""
    rng = np.random.default_rng(seed)
    rc = qm.mean(0)

    def cloud(n):
        d = rng.normal(size=(n, 3))
        d /= np.linalg.norm(d, axis=1)[:, None]
        return qm[rng.integers(len(qm), size=n)] + d * (0.4 * rng.random(n) ** (1 / 3))[:, None]

    pts = np.vstack([qm, cloud(485), cloud(200)])
    nf = 500
    dV = ewald(pts, mm, q, L) - direct(pts, mm, q)
    fits = {}
    for order in (1, 2, 3):
        A, exps = monomials(pts - rc, order)
        coef, *_ = np.linalg.lstsq(A[:nf], dV[:nf], rcond=None)
        res = A[nf:] @ coef - dV[nf:]
        fits[order] = dict(coef=coef, exps=exps, test_rms=float(np.sqrt((res ** 2).mean())),
                           test_max=float(np.abs(res).max()))
    return rc, pts, dV, fits


def poly_operator(mol, rc_nm, coef, exps):
    """One-electron matrix of -V_poly (electron charge -1), Eh; V_poly in kJ/mol/e with x in nm."""
    with mol.with_common_orig(rc_nm / BOHR_NM):
        ints = {0: mol.intor("int1e_ovlp"), 1: mol.intor("int1e_r"), 2: mol.intor("int1e_rr"),
                3: mol.intor("int1e_rrr")}
    h = np.zeros_like(ints[0])
    for c, e in zip(coef, exps):
        n = sum(e)
        axes = [a for a in range(3) for _ in range(e[a])]
        if n == 0:
            m = ints[0]
        else:
            idx = 0
            for a in axes:
                idx = 3 * idx + a
            m = ints[n][idx]
        h -= c * BOHR_NM ** n / EH * m
    return h


def nuclear_term(mol, qm_nm, rc_nm, coef, order):
    A, _ = monomials(qm_nm - rc_nm, order)
    return float(mol.atom_charges() @ (A @ coef)) / EH


def sphere(K, R, rc):
    i = np.arange(K) + 0.5
    phi = np.arccos(1 - 2 * i / K)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return rc + R * np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)


def run_scf(mol, mm_nm, q, extra_h=None, extra_nuc=0.0):
    mf = dft.RKS(mol)
    mf.xc = "r2scan"
    mf = qmmm.mm_charge(mf, mm_nm * 10, q, unit="Angstrom")
    if extra_h is not None:
        base_h, base_nuc = mf.get_hcore, mf.energy_nuc
        mf.get_hcore = lambda *a, **k: base_h(*a, **k) + extra_h
        mf.energy_nuc = lambda *a, **k: base_nuc(*a, **k) + extra_nuc
    mf.conv_tol = 1e-10
    mf.max_cycle = 100
    e = mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF not converged")
    return float(e)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("requests", type=Path)
    p.add_argument("pilot", type=Path)
    p.add_argument("outdir", type=Path)
    p.add_argument("frame", type=int)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--K", type=int, default=100)
    p.add_argument("--Rs", type=float, default=2.0)
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--only-pseudo", action="store_true")
    a = p.parse_args()
    lib.num_threads(a.threads)
    a.outdir.mkdir(parents=True, exist_ok=True)
    f = a.frame
    t0 = time.perf_counter()
    with np.load(a.requests) as z:
        qm = z[f"S2_qm_pos_{f}"].astype(float)
        mm = z[f"S2_mm_pos_{f}"].astype(float)
        q = z[f"S2_mm_q_{f}"].astype(float)
        box = z["box_nm"][f].astype(float)
        elements = z["qm_elements"].tolist()
    L = float(box[0, 0])
    assert np.allclose(box, np.diag([L] * 3))
    pilot = {s: json.loads((a.pilot / f"candidate_{f:02d}_{s}.json").read_text())["energy_eh"]
             for s in ("S2", "ENV20", "ENV30")}
    rc, pts, dV, fits = fit_dV(qm, mm, q, L, seed=20261107 + f)

    mol = gto.M(atom=list(zip(elements, (qm * 10).tolist())), basis="def2-svp", charge=-1, spin=0,
                unit="Angstrom", verbose=3)
    mol.max_memory = 6000
    energies, timing = {}, {}
    orders = () if a.only_pseudo else (1, 2, 3)
    if not a.only_pseudo:
        t = time.perf_counter()
        energies["S2"] = run_scf(mol, mm, q)
        timing["S2"] = time.perf_counter() - t
    for order in orders:
        fit = fits[order]
        h = poly_operator(mol, rc, fit["coef"], fit["exps"])
        en = nuclear_term(mol, qm, rc, fit["coef"], order)
        t = time.perf_counter()
        energies[f"S3_o{order}"] = run_scf(mol, mm, q, h, en)
        timing[f"S3_o{order}"] = time.perf_counter() - t

    # pseudo charges reproducing the order-2 polynomial on the fit points
    sites = sphere(a.K, a.Rs, rc)
    nf = 500
    A2, _ = monomials(pts - rc, 2)
    target = A2 @ fits[2]["coef"]
    G = KE / np.linalg.norm(pts[:, None, :] - sites[None], axis=-1)
    if a.lam:
        Q = np.linalg.solve(G[:nf].T @ G[:nf] + a.lam * nf * np.eye(a.K), G[:nf].T @ target[:nf])
    else:
        Q, *_ = np.linalg.lstsq(G[:nf], target[:nf], rcond=1e-12)
    pres = G @ Q - target
    pseudo = dict(K=a.K, Rs_nm=a.Rs, lam=a.lam, Q_sum=float(Q.sum()), Q_absmax=float(np.abs(Q).max()),
                  fit_rms_vs_poly2_fitpts=float(np.sqrt((pres[:nf] ** 2).mean())),
                  test_rms_vs_poly2=float(np.sqrt((pres[nf:] ** 2).mean())),
                  test_rms_vs_exact_dV=float(np.sqrt(((G @ Q - dV)[nf:] ** 2).mean())))
    t = time.perf_counter()
    energies["S3_P2"] = run_scf(mol, np.vstack([mm, sites]), np.concatenate([q, Q]))
    timing["S3_P2"] = time.perf_counter() - t

    rec = dict(candidate=f, host=platform.node(), threads=lib.num_threads(),
               quantity="R_v = E_v - E_ENV20 (kJ/mol); ENV20/ENV30/S2 energies from the 2026-11-07 pilot",
               energies_eh=energies, pilot_eh=pilot,
               s2_reproduction_eh=energies["S2"] - pilot["S2"] if "S2" in energies else None,
               R_kj_mol={k: (v - pilot["ENV20"]) * EH for k, v in energies.items()},
               env30_minus_env20_kj_mol=(pilot["ENV30"] - pilot["ENV20"]) * EH,
               fits={str(o): dict(test_rms_kj_mol_e=v["test_rms"], test_max_kj_mol_e=v["test_max"],
                                  coef=v["coef"].tolist()) for o, v in fits.items()},
               pseudo=pseudo, centroid_nm=rc.tolist(), wall_s=dict(timing, total=time.perf_counter() - t0))
    name = f"candidate_{f:02d}_P_K{a.K}_R{a.Rs}_lam{a.lam:g}.json" if a.only_pseudo else f"candidate_{f:02d}.json"
    (a.outdir / name).write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps({k: rec[k] for k in ("candidate", "s2_reproduction_eh", "R_kj_mol", "pseudo", "wall_s")},
                     indent=1), flush=True)


if __name__ == "__main__":
    main()
