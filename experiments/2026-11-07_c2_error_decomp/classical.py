"""Classical potential difference dV = V_Ewald - V_S2 for one S2 request (numpy only).

Same formulas as decompose.py (which ran the three pilot frames); shared here for field_stats.py.
"""
import importlib.util
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
from scipy.special import erfc

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("units", ROOT / "openmmorca" / "units.py")
units = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(units)
KE = units.COULOMB_KJ_MOL_NM


def min_image(d, L):
    return d - L * np.round(d / L)


def direct(sites, pos, q, chunk=64):
    out = np.empty(len(sites))
    for i in range(0, len(sites), chunk):
        r = np.linalg.norm(sites[i:i + chunk, None, :] - pos[None], axis=-1)
        out[i:i + chunk] = (q[None] / r).sum(1)
    return KE * out


def ewald(sites, pos, q, L, alpha=1.2, nmax=14, chunk=64):
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
    cols, exps = [np.ones(len(x))], [(0, 0, 0)]
    for o in range(1, order + 1):
        for c in combinations_with_replacement(range(3), o):
            e = tuple(c.count(a) for a in range(3))
            cols.append(np.prod([x[:, a] ** e[a] for a in range(3)], axis=0))
            exps.append(e)
    return np.stack(cols, 1), exps


def field_fit(qm, mm, q, L, seed, n_fit=250, n_test=60, order=3):
    """Fit dV about the QM centroid; returns dict with coefficients and held-out residuals."""
    rng = np.random.default_rng(seed)
    centroid = qm.mean(0)

    def cloud(n):
        d = rng.normal(size=(n, 3))
        d /= np.linalg.norm(d, axis=1)[:, None]
        return qm[rng.integers(len(qm), size=n)] + d * (0.4 * rng.random(n) ** (1 / 3))[:, None]

    pts = np.vstack([qm, cloud(n_fit), cloud(n_test)])
    nf = len(qm) + n_fit
    dV = ewald(pts, mm, q, L) - direct(pts, mm, q)
    A, exps = monomials(pts - centroid, order)
    coef, *_ = np.linalg.lstsq(A[:nf], dV[:nf], rcond=None)
    res = A[nf:] @ coef - dV[nf:]
    return dict(centroid=centroid, coef=coef, exps=exps, dV_sites=dV[:len(qm)],
                test_rms=float(np.sqrt((res ** 2).mean())), test_max=float(np.abs(res).max()))
