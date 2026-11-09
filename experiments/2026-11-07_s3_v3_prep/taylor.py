"""Task 0 prototype: analytic Taylor coefficients of dV = V_Ewald - V_S2 at a centre, and the MM forces
of the correction for given dE/dc (cubic box only; numpy).

Notation (units kJ/mol, nm, e; KE from units.py):
  dV(r) = sum_j q_j [ phi_E(r - r_j) - 1/|r - p_j| ] + background constant
  phi_E: Ewald kernel, real space erfc(a r)/r at the minimum image + reciprocal sum;
  p_j: the S2 placement of charge j (the request position).
  c_e = d^e dV(rc) / e!   for monomials e = (ex, ey, ez), |e| <= order, ordering as classical.monomials.
For dE/dc = m (10-vector): F_j = q_j sum_e m_e grad_d t_e(d_j), d_j = rc - r_j (images as above),
i.e. the MM force of the correction energy sum_e m_e c_e.
"""
import sys
from math import factorial
from pathlib import Path

import numpy as np
from scipy.special import erf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-11-07_c2_error_decomp"))
from classical import KE, min_image, monomials  # noqa: E402

EXPS2 = monomials(np.zeros((1, 3)), 2)[1]


def _radial(kind, r, a):
    """h, h', h'', h''' for kind 'erf' (-erf(ar)/r), 'erfc' (erfc(ar)/r), 'inv' (-1/r)."""
    if kind == "inv":
        return -1 / r, 1 / r ** 2, -2 / r ** 3, 6 / r ** 4
    u = erf(a * r)
    g = 2 * a / np.sqrt(np.pi) * np.exp(-(a * r) ** 2)       # u'
    g1 = -2 * a * a * r * g                                   # u''
    g2 = -2 * a * a * (g + r * g1)                            # u'''
    # w = u/r
    w = u / r
    w1 = g / r - u / r ** 2
    w2 = g1 / r - 2 * g / r ** 2 + 2 * u / r ** 3
    w3 = g2 / r - 3 * g1 / r ** 2 + 6 * g / r ** 3 - 6 * u / r ** 4
    if kind == "erf":
        return -w, -w1, -w2, -w3
    # erfc(ar)/r = 1/r - w
    return 1 / r - w, -1 / r ** 2 - w1, 2 / r ** 3 - w2, -6 / r ** 4 - w3


def _tensors(h, d, r):
    """Cartesian derivatives (value, grad, hess, third) of a radial function at vectors d (N,3)."""
    h0, h1, h2, h3 = h
    n = d / r[:, None]
    I = np.eye(3)
    grad = h1[:, None] * n
    hess = (h2 - h1 / r)[:, None, None] * n[:, :, None] * n[:, None, :] + (h1 / r)[:, None, None] * I
    A = h3 - 3 * h2 / r + 3 * h1 / r ** 2
    B = (h2 - h1 / r) / r
    nnn = n[:, :, None, None] * n[:, None, :, None] * n[:, None, None, :]
    sym = (I[None, :, :, None] * n[:, None, None, :] + I[None, :, None, :] * n[:, None, :, None]
           + I[None, None, :, :] * n[:, :, None, None])
    third = A[:, None, None, None] * nnn + B[:, None, None, None] * sym
    return h0, grad, hess, third


def _deriv(e, h0, grad, hess, third):
    """d^e of the radial function (per source) for |e| <= 2, plus its gradient (needs |e|+1)."""
    axes = [a for a in range(3) for _ in range(e[a])]
    if len(axes) == 0:
        return h0, grad
    if len(axes) == 1:
        return grad[:, axes[0]], hess[:, axes[0], :]
    return hess[:, axes[0], axes[1]], third[:, axes[0], axes[1], :]


def _efact(e):
    return factorial(e[0]) * factorial(e[1]) * factorial(e[2])


class Reciprocal:
    def __init__(self, L, alpha, kmax):
        self.L, self.alpha, self.n = L, alpha, np.arange(-kmax, kmax + 1)
        kk = 2 * np.pi * self.n / L
        self.kk = kk
        KX, KY, KZ = np.meshgrid(kk, kk, kk, indexing="ij")
        self.K = np.stack([KX, KY, KZ])
        k2 = KX ** 2 + KY ** 2 + KZ ** 2
        f = np.zeros_like(k2)
        nz = k2 > 0
        f[nz] = np.exp(-k2[nz] / (4 * alpha ** 2)) / k2[nz]
        self.f = (4 * np.pi / L ** 3) * f

    def phases(self, pos):
        return [np.exp(-1j * np.outer(pos[:, a], self.kk)) for a in range(3)]

    def structure(self, q, ph):
        S = np.empty((len(self.n),) * 3, dtype=complex)
        for ia in range(len(self.n)):
            S[ia] = (ph[1] * (q * ph[0][:, ia])[:, None]).T @ ph[2]
        return S

    def kfactor(self, e):
        out = np.ones_like(self.f, dtype=complex)
        for a in range(3):
            out = out * (1j * self.K[a]) ** e[a]
        return out

    def contract(self, A, ph, chunk=4000):
        """sum_k A(k) exp(-i k.r_j) for every j (A: (n,n,n) complex) -> (N,) complex."""
        n = len(self.n)
        out = np.empty(len(ph[0]), dtype=complex)
        A2 = A.reshape(n * n, n)
        for s in range(0, len(out), chunk):
            B = (A2 @ ph[2][s:s + chunk].T).T.reshape(-1, n, n)        # (c, a, b)
            out[s:s + chunk] = np.einsum("jab,jb,ja->j", B, ph[1][s:s + chunk], ph[0][s:s + chunk])
        return out


def taylor(rc, placed, q, L, alpha=1.2, kmax=14, want_force_m=None):
    """Return c (10,) and, if want_force_m (10,) is given, the correction MM forces (N,3)."""
    d_mi = min_image(rc - placed, L)
    d_s2 = rc - placed
    same = np.all(np.abs(d_mi - d_s2) < 1e-9, axis=1)
    c = np.zeros(len(EXPS2))
    F = np.zeros((len(q), 3)) if want_force_m is not None else None
    parts = []
    if same.any():
        d = d_mi[same]; r = np.linalg.norm(d, axis=1)
        parts.append((same, _tensors(_radial("erf", r, alpha), d, r)))
    if (~same).any():
        d = d_mi[~same]; r = np.linalg.norm(d, axis=1)
        parts.append((~same, _tensors(_radial("erfc", r, alpha), d, r)))
        d = d_s2[~same]; r = np.linalg.norm(d, axis=1)
        parts.append((~same, _tensors(_radial("inv", r, alpha), d, r)))
    for mask, T in parts:
        for k, e in enumerate(EXPS2):
            val, grad = _deriv(e, *T)
            c[k] += (q[mask] * val).sum() / _efact(e)
            if F is not None:
                F[mask] += want_force_m[k] / _efact(e) * q[mask][:, None] * grad
    rec = Reciprocal(L, alpha, kmax)
    ph = rec.phases(placed)
    S = rec.structure(q, ph)
    eikr = np.exp(1j * np.einsum("a,aijk->ijk", rc, rec.K))
    for k, e in enumerate(EXPS2):
        c[k] += (rec.f * (rec.kfactor(e) * eikr * S)).real.sum() / _efact(e)
    c[0] -= np.pi * q.sum() / (L ** 3 * alpha ** 2)
    if F is not None:
        P = sum(want_force_m[k] / _efact(e) * rec.kfactor(e) for k, e in enumerate(EXPS2))
        for a in range(3):
            # grad_d of t_e(d) with d = rc - r_j: factor i k_a; the j phase comes from contract()
            A = rec.f * P * 1j * rec.K[a] * eikr
            F[:, a] += q * rec.contract(A, ph).real
    return KE * c, (KE * F if F is not None else None)
