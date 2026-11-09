"""C2 diagnosis: first-order decomposition of D(f) = E_S2(f) - E_ENV20(f) on the pilot frames.

Quantity: the S2 embedding error of the QM energy, D(f), from the 2026-11-07 pilot (PySCF
r2SCAN/def2-SVP, scheme A).  Reference values: pilot JSONs, D(0,4,8) = -48.15/-58.41/-64.24 kJ/mol.

Model: S2 and ENV differ only in the external potential from the *same* MM charges:
    dV(r) = V_Ewald(r) - V_S2(r),
V_S2 = direct Coulomb sum over the minimum-image S2 charges (exactly the request), V_Ewald = periodic
sum of the same charges with a uniform background (QM self images absent, as in ENV).  The nearest
missing source is ~L/2 away from the QM region, so dV is smooth there and is fitted by a low-order
polynomial about the QM centroid.  First order in dV:
    E_ENV - E_S2 ~ E1 = sum_a Z_a dV(R_a) - int n(r) dV(r) dr      (n = S2 electron density)
so D ~ -E1 + const; the constant (Ewald potential convention x fixed QM charge -1) cancels in
D(f) - D(0).  Polarization by the smooth dV is second order and is reported as the residual.

E1 is split by moments about the centroid: monopole q*c0, dipole mu.g, quadrupole, cubic.

Usage: decompose.py REQUESTS.npz PILOT_PYSCF_DIR OUTDIR FRAME [--threads N]
"""
import argparse
import importlib.util
import json
import platform
import time
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
from pyscf import dft, gto, lib, qmmm
from scipy.special import erfc

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("units", ROOT / "openmmorca" / "units.py")
units = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(units)
KE = units.COULOMB_KJ_MOL_NM            # kJ/mol nm e^-2
EH = units.HARTREE_IN_KJ_MOL
BOHR_NM = units.BOHR_IN_ANGSTROM / 10.0


def min_image(d, L):
    return d - L * np.round(d / L)


def direct(sites, pos, q, chunk=64):
    out = np.empty(len(sites))
    for i in range(0, len(sites), chunk):
        r = np.linalg.norm(sites[i:i + chunk, None, :] - pos[None], axis=-1)
        out[i:i + chunk] = (q[None] / r).sum(1)
    return KE * out


def ewald(sites, pos, q, L, alpha, nmax, chunk=64):
    """Periodic potential (kJ/mol/e) of point charges in a cubic box, uniform background."""
    V = L ** 3
    phi = np.empty(len(sites))
    for i in range(0, len(sites), chunk):
        r = np.linalg.norm(min_image(sites[i:i + chunk, None, :] - pos[None], L), axis=-1)
        phi[i:i + chunk] = (q[None] * erfc(alpha * r) / r).sum(1)
    n = np.arange(-nmax, nmax + 1)
    kk = 2 * np.pi * n / L
    ph = [np.exp(-1j * np.outer(pos[:, a], kk)) for a in range(3)]
    S = np.empty((len(n),) * 3, dtype=complex)
    for ia in range(len(n)):
        S[ia] = (ph[1] * (q * ph[0][:, ia])[:, None]).T @ ph[2]
    KX, KY, KZ = np.meshgrid(kk, kk, kk, indexing="ij")
    k2 = KX ** 2 + KY ** 2 + KZ ** 2
    f = np.zeros_like(k2)
    nz = k2 > 0
    f[nz] = np.exp(-k2[nz] / (4 * alpha ** 2)) / k2[nz]
    fS = (4 * np.pi / V) * f * S
    for i, s in enumerate(sites):
        phi[i] += (np.exp(1j * (KX * s[0] + KY * s[1] + KZ * s[2])) * fS).real.sum()
    phi -= np.pi * q.sum() / (V * alpha ** 2)
    return KE * phi


def monomials(x, order):
    """Columns: 1, x, y, z, xx, xy, ..., up to `order`; returns (matrix, list of exponent tuples)."""
    cols, exps = [np.ones(len(x))], [(0, 0, 0)]
    for o in range(1, order + 1):
        for c in combinations_with_replacement(range(3), o):
            e = tuple(c.count(a) for a in range(3))
            cols.append(np.prod([x[:, a] ** e[a] for a in range(3)], axis=0))
            exps.append(e)
    return np.stack(cols, 1), exps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("requests", type=Path)
    p.add_argument("pilot", type=Path)
    p.add_argument("outdir", type=Path)
    p.add_argument("frame", type=int)
    p.add_argument("--threads", type=int, default=8)
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
    pilot = {s: json.loads((a.pilot / f"candidate_{f:02d}_{s}.json").read_text()) for s in ("S2", "ENV20", "ENV30")}
    assert np.allclose(np.array(pilot["S2"]["qm_positions_nm"]), qm, atol=0, rtol=0)

    # --- exact dV on sample points in the QM region -------------------------------------------
    rng = np.random.default_rng(20261107 + f)
    centroid = qm.mean(0)

    def cloud(n):
        dirs = rng.normal(size=(n, 3))
        dirs /= np.linalg.norm(dirs, axis=1)[:, None]
        return qm[rng.integers(len(qm), size=n)] + dirs * (0.4 * rng.random(n) ** (1 / 3))[:, None]

    fit_pts = np.vstack([qm, cloud(500)])
    test_pts = cloud(200)
    pts = np.vstack([fit_pts, test_pts])
    dV = ewald(pts, mm, q, L, 1.2, 14) - direct(pts, mm, q)
    chk_idx = np.arange(0, len(pts), 10)
    dV_chk = ewald(pts[chk_idx], mm, q, L, 1.6, 20) - direct(pts[chk_idx], mm, q)
    ewald_dev = dV_chk - dV[chk_idx]
    ewald_check = {"alpha_nmax": [[1.2, 14], [1.6, 20]],
                   "max_abs_dev_after_constant_kj_mol_e": float(np.abs(ewald_dev - ewald_dev.mean()).max()),
                   "constant_kj_mol_e": float(ewald_dev.mean())}
    t_classical = time.perf_counter() - t0

    nf = len(fit_pts)
    fits = {}
    for order in (1, 2, 3, 4):
        A, exps = monomials(pts - centroid, order)
        coef, *_ = np.linalg.lstsq(A[:nf], dV[:nf], rcond=None)
        res = A[nf:] @ coef - dV[nf:]
        fits[order] = {"coef": coef, "exps": exps, "test_rms": float(np.sqrt((res ** 2).mean())),
                       "test_max": float(np.abs(res).max())}

    # --- S2 SCF, identical settings to the pilot --------------------------------------------------
    mol = gto.M(atom=list(zip(elements, (qm * 10).tolist())), basis="def2-svp", charge=-1, spin=0,
                unit="Angstrom", verbose=3)
    mol.max_memory = 6000
    mf = dft.RKS(mol)
    mf.xc = "r2scan"
    mf = qmmm.mm_charge(mf, mm * 10, q, unit="Angstrom")
    mf.conv_tol = 1e-10
    mf.max_cycle = 100
    e_s2 = mf.kernel()
    assert mf.converged
    dm = mf.make_rdm1()
    grids = mf.grids
    coords_nm = grids.coords * BOHR_NM
    rho = np.empty(len(grids.weights))
    ni = dft.numint.NumInt()
    for i in range(0, len(rho), 20000):
        ao = ni.eval_ao(mol, grids.coords[i:i + 20000], deriv=0)
        rho[i:i + 20000] = ni.eval_rho(mol, ao, dm, xctype="LDA")
    n_elec_grid = float((grids.weights * rho).sum())
    t_scf = time.perf_counter() - t0 - t_classical

    # --- first-order energy and moments ------------------------------------------------------------
    Z = mol.atom_charges().astype(float)
    dV_nuc = dV[:len(qm)]                      # exact dV at the nuclei (first rows of fit_pts)
    out_e1 = {}
    for rdom in (0.6, 0.8, 1.0):
        sel = np.linalg.norm(coords_nm - centroid, axis=1) < rdom
        w = grids.weights[sel] * rho[sel]
        row = {"electrons_in_domain": float(w.sum())}
        for order in (2, 3, 4):
            A, _ = monomials(coords_nm[sel] - centroid, order)
            row[f"E1_order{order}_kj_mol"] = float((Z * dV_nuc).sum() - w @ (A @ fits[order]["coef"]))
        out_e1[f"{rdom}"] = row

    # moments of the total QM charge (nuclei - electrons) about the centroid, in e nm^k
    sel = np.linalg.norm(coords_nm - centroid, axis=1) < 0.8
    w = grids.weights[sel] * rho[sel]
    A3_nuc, exps3 = monomials(qm - centroid, 3)
    A3_el, _ = monomials(coords_nm[sel] - centroid, 3)
    moments = Z @ A3_nuc - w @ A3_el               # sum over charges of monomial values
    coef3 = fits[3]["coef"]
    by_order = {}
    for k, e in enumerate(exps3):
        o = sum(e)
        by_order[o] = by_order.get(o, 0.0) + float(moments[k] * coef3[k])
    # exact E1 uses dV at nuclei; the moment split uses the fit at nuclei too, report both
    rec = dict(
        candidate=f, host=platform.node(), threads=lib.num_threads(),
        quantity="first-order E_ENV - E_S2 from classical dV = V_Ewald - V_S2 and the S2 density",
        e_s2_eh=float(e_s2), e_s2_pilot_eh=pilot["S2"]["energy_eh"],
        e_s2_reproduction_eh=float(e_s2 - pilot["S2"]["energy_eh"]),
        D_pilot_kj_mol=(pilot["S2"]["energy_eh"] - pilot["ENV20"]["energy_eh"]) * EH,
        D30_pilot_kj_mol=(pilot["S2"]["energy_eh"] - pilot["ENV30"]["energy_eh"]) * EH,
        ewald_check=ewald_check,
        fit={str(o): {"test_rms_kj_mol_e": v["test_rms"], "test_max_kj_mol_e": v["test_max"]} for o, v in fits.items()},
        n_grid=int(len(rho)), electrons_grid=n_elec_grid, electrons_expected=int(mol.nelectron),
        E1=out_e1,
        E1_by_order_kj_mol={str(k): v for k, v in sorted(by_order.items())},
        centroid_nm=centroid.tolist(),
        c0_kj_mol_e=float(coef3[0]),
        field_grad_kj_mol_nm_e=coef3[1:4].tolist(),       # g = grad dV ; field difference = -g
        hessian_terms=dict(zip([str(e) for e in exps3[4:10]], coef3[4:10].tolist())),
        charge_e=float(moments[0]), dipole_e_nm=moments[1:4].tolist(),
        second_moments_e_nm2=dict(zip([str(e) for e in exps3[4:10]], moments[4:10].tolist())),
        qm_positions_nm=qm.tolist(), qm_elements=elements,
        wall_s={"classical": t_classical, "scf_density": t_scf, "total": time.perf_counter() - t0},
    )
    path = a.outdir / f"candidate_{f:02d}.json"
    path.write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps({k: rec[k] for k in ("candidate", "e_s2_reproduction_eh", "D_pilot_kj_mol", "fit", "E1",
                                          "E1_by_order_kj_mol", "electrons_grid", "ewald_check", "wall_s")}, indent=1))


if __name__ == "__main__":
    main()
