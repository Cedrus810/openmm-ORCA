"""Classical Ewald correction for whole-box (S2) embedding: scheme S3 (spec §1.2 exception, §11.6).

S2 places every MM charge at one image relative to the QM region. The missing
part of the periodic MM potential in the QM region,

    dV(r) = V_Ewald(r) - V_S2(r),

(V_Ewald: Ewald sum of the same charges with a uniform neutralising background,
no QM self images; V_S2: direct Coulomb sum over the placed charges) is smooth
there: its sources are the other images, at least about half a box away. S3
adds its second-order Taylor expansion about a centre r_c,

    dV(r) ~ sum_e c_e (r - r_c)^e,   |e| <= 2  (10 coefficients),

to the QM calculation. The polynomial is represented by K pseudo charges on a
sphere of radius R_s about r_c, Q = W c, with W a fixed ridge fit on a template
ball (pseudo charges only produce harmonic potentials; the trace part of the
quadratic term, which comes from the background, is approximated).

Energy derivatives (dE/dQ_k = QM electrostatic potential phi_k at the pseudo
charge k, for a variational SCF; m = W^T phi = dE/dc):

* MM charge j: F_j += -m . dc/dp_j            (``mm_forces``)
* centre:      dE/dr_c += sum_k pcgrad_k + sum_j F_j   (dc/dr_c = -sum_j dc/dp_j)

S3 is an option on top of S2, not PME-consistent QM/MM: QM charges never enter
OpenMM's PME and QM self images are absent. Validation:
``docs/validation/2026-11-07-s3-v1-pyscf.md``, ``…-s3-v2-orca.md``,
``…-s3-v3-task0.md``; plan ``docs/plans/2026-11-07-s3-v3-implementation-plan.md``.

Units: nm, e, kJ/mol; c_e in kJ/mol/e/nm^|e|.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations_with_replacement, product
from math import factorial

import numpy as np

from openmmorca.units import COULOMB_KJ_MOL_NM

ORDER = 2
EXPONENTS: tuple[tuple[int, int, int], ...] = tuple(
    [(0, 0, 0)]
    + [
        tuple(c.count(a) for a in range(3))
        for o in range(1, ORDER + 1)
        for c in combinations_with_replacement(range(3), o)
    ]
)
_FACT = np.array([factorial(e[0]) * factorial(e[1]) * factorial(e[2]) for e in EXPONENTS], float)
_AXES = [[a for a in range(3) for _ in range(e[a])] for e in EXPONENTS]


def monomials(x: np.ndarray) -> np.ndarray:
    """(n, 10) values of (x - 0)^e for the EXPONENTS ordering."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    return np.stack([np.prod([x[:, a] ** e[a] for a in range(3)], axis=0) for e in EXPONENTS], axis=1)


def _fibonacci_sphere(k: int) -> np.ndarray:
    i = np.arange(k) + 0.5
    phi = np.arccos(1 - 2 * i / k)
    theta = np.pi * (1 + 5**0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


_erf = np.vectorize(math.erf, otypes=[float])  # numpy has no erf; avoids a scipy dependency


def _radial(kind: str, r: np.ndarray, alpha: float):
    """h, h', h'', h''' of erfc(a r)/r ('erfc') or -erf(a r)/r ('erf')."""
    u = _erf(alpha * r)
    g = 2 * alpha / np.sqrt(np.pi) * np.exp(-((alpha * r) ** 2))
    g1 = -2 * alpha * alpha * r * g
    g2 = -2 * alpha * alpha * (g + r * g1)
    w = u / r
    w1 = g / r - u / r**2
    w2 = g1 / r - 2 * g / r**2 + 2 * u / r**3
    w3 = g2 / r - 3 * g1 / r**2 + 6 * g / r**3 - 6 * u / r**4
    if kind == "erf":
        return -w, -w1, -w2, -w3
    return 1 / r - w, -1 / r**2 - w1, 2 / r**3 - w2, -6 / r**4 - w3


def _tensors(h, d: np.ndarray):
    """Value, gradient, Hessian and third derivatives of a radial function at vectors d."""
    r = np.linalg.norm(d, axis=1)
    h0, h1, h2, h3 = h
    n = d / r[:, None]
    eye = np.eye(3)
    grad = h1[:, None] * n
    nn = n[:, :, None] * n[:, None, :]
    hess = (h2 - h1 / r)[:, None, None] * nn + (h1 / r)[:, None, None] * eye
    a3 = h3 - 3 * h2 / r + 3 * h1 / r**2
    b3 = (h2 - h1 / r) / r
    nnn = nn[:, :, :, None] * n[:, None, None, :]
    sym = (
        eye[None, :, :, None] * n[:, None, None, :]
        + eye[None, :, None, :] * n[:, None, :, None]
        + eye[None, None, :, :] * n[:, :, None, None]
    )
    third = a3[:, None, None, None] * nnn + b3[:, None, None, None] * sym
    return h0, grad, hess, third


@dataclass
class CorrectionStep:
    """Per-step S3 data: coefficients, pseudo charges and what ``mm_forces`` needs."""

    center_nm: np.ndarray
    coefficients: np.ndarray  # (10,), kJ/mol/e/nm^|e|
    pseudo_positions_nm: np.ndarray  # (K, 3)
    pseudo_charges_e: np.ndarray  # (K,)
    _charges: np.ndarray = field(repr=False, default=None)
    _real: tuple = field(repr=False, default=None)  # per-charge (grad, hess, third) summed over images
    _phases: list = field(repr=False, default=None)
    _recip: dict = field(repr=False, default=None)


class EwaldCorrection:
    """Second-order classical Ewald correction to S2 embedding, via pseudo charges.

    Parameters (defaults from the 2026-11-07 Task 0 checks): *n_pseudo* charges
    on a sphere of *radius_nm*; ridge penalty *ridge* (times the number of
    template points) for the fit on a uniform ball of *template_radius_nm*;
    Ewald splitting *alpha_nm* (nm^-1) and *kmax* reciprocal vectors per axis.
    Real-space images within ``5/alpha`` of the centre are summed over shells
    -2..2 of the given cell.
    """

    def __init__(
        self,
        n_pseudo: int = 50,
        radius_nm: float = 2.0,
        ridge: float = 1e-4,
        alpha_nm: float = 1.2,
        kmax: int = 10,
        template_radius_nm: float = 0.6,
        template_points: int = 2000,
        seed: int = 0,
    ) -> None:
        if n_pseudo < len(EXPONENTS) or radius_nm <= template_radius_nm or kmax < 1 or alpha_nm <= 0:
            raise ValueError("invalid EwaldCorrection parameters")
        self.n_pseudo, self.radius_nm, self.ridge = n_pseudo, float(radius_nm), float(ridge)
        self.alpha, self.kmax = float(alpha_nm), int(kmax)
        self.template_radius_nm = float(template_radius_nm)
        self.directions = _fibonacci_sphere(n_pseudo)
        rng = np.random.default_rng(seed)
        d = rng.normal(size=(template_points, 3))
        d /= np.linalg.norm(d, axis=1)[:, None]
        x = d * (template_radius_nm * rng.random(template_points) ** (1 / 3))[:, None]
        G = COULOMB_KJ_MOL_NM / np.linalg.norm(x[:, None, :] - radius_nm * self.directions[None], axis=-1)
        A = monomials(x)
        self.W = np.linalg.solve(G.T @ G + ridge * template_points * np.eye(n_pseudo), G.T @ A)
        # RMS over the template of (pseudo potential - monomial) for each unit coefficient
        self.template_error = np.sqrt(((G @ self.W - A) ** 2).mean(axis=0))
        self.r_cut = 5.0 / self.alpha

    # -- per step --------------------------------------------------------------------------
    def prepare(self, center_nm, mm_positions_nm, mm_charges_e, box_vectors_nm) -> CorrectionStep:
        """Coefficients and pseudo charges for S2 charges at their placed positions."""
        rc = np.asarray(center_nm, dtype=float)
        p = np.asarray(mm_positions_nm, dtype=float)
        q = np.asarray(mm_charges_e, dtype=float)
        box = np.asarray(box_vectors_nm, dtype=float)
        c = np.zeros(len(EXPONENTS))

        # real space: image 0 is the placed one, erfc/r - 1/r = -erf/r; other images erfc/r
        n = len(q)
        grad = np.zeros((n, 3))
        hess = np.zeros((n, 3, 3))
        third = np.zeros((n, 3, 3, 3))
        d0 = rc - p
        for shift in product(range(-2, 3), repeat=3):
            if shift == (0, 0, 0):
                idx = np.arange(n)
                d = d0
                kind = "erf"
            else:
                d_all = d0 + np.asarray(shift, float) @ box
                idx = np.flatnonzero(np.linalg.norm(d_all, axis=1) < self.r_cut)
                if len(idx) == 0:
                    continue
                d = d_all[idx]
                kind = "erfc"
            h0, g1, g2, g3 = _tensors(_radial(kind, np.linalg.norm(d, axis=1), self.alpha), d)
            qi = q[idx]
            for k, ax in enumerate(_AXES):
                if len(ax) == 0:
                    c[k] += qi @ h0
                elif len(ax) == 1:
                    c[k] += qi @ g1[:, ax[0]]
                else:
                    c[k] += qi @ g2[:, ax[0], ax[1]]
            grad[idx] += g1
            hess[idx] += g2
            third[idx] += g3

        # reciprocal space, separable in fractional coordinates
        inv = np.linalg.inv(box)
        volume = abs(float(np.linalg.det(box)))
        nvec = np.arange(-self.kmax, self.kmax + 1)
        recip = 2 * np.pi * inv.T  # rows: reciprocal vectors times 2 pi
        N1, N2, N3 = np.meshgrid(nvec, nvec, nvec, indexing="ij")
        K = N1[..., None] * recip[0] + N2[..., None] * recip[1] + N3[..., None] * recip[2]
        k2 = (K**2).sum(-1)
        f = np.zeros_like(k2)
        nz = k2 > 0
        f[nz] = np.exp(-k2[nz] / (4 * self.alpha**2)) / k2[nz]
        f *= 4 * np.pi / volume
        s = p @ inv
        phases = [np.exp(-2j * np.pi * np.outer(s[:, a], nvec)) for a in range(3)]
        S = np.empty(f.shape, dtype=complex)
        for ia in range(len(nvec)):
            S[ia] = (phases[1] * (q * phases[0][:, ia])[:, None]).T @ phases[2]
        eikc = np.exp(1j * (K @ rc))
        iK = [1j * K[..., a] for a in range(3)]
        kfac = []
        for ax in _AXES:
            t = np.ones_like(f, dtype=complex)
            for a in ax:
                t = t * iK[a]
            kfac.append(t)
        for k in range(len(EXPONENTS)):
            c[k] += float((f * kfac[k] * eikc * S).real.sum())
        c[0] -= np.pi * q.sum() / (volume * self.alpha**2)
        c = COULOMB_KJ_MOL_NM * c / _FACT

        Q = self.W @ c
        return CorrectionStep(
            center_nm=rc,
            coefficients=c,
            pseudo_positions_nm=rc + self.radius_nm * self.directions,
            pseudo_charges_e=Q,
            _charges=q,
            _real=(grad, hess, third),
            _phases=phases,
            _recip=dict(f=f, eikc=eikc, iK=iK, kfac=kfac),
        )

    def moments(self, esp_kj_mol_e: np.ndarray) -> np.ndarray:
        """dE/dc (10,) from the QM electrostatic potential at the pseudo charges."""
        return self.W.T @ np.asarray(esp_kj_mol_e, dtype=float)

    def mm_forces(self, step: CorrectionStep, m: np.ndarray, chunk: int = 4000) -> np.ndarray:
        """Correction forces on the S2 charges (kJ/mol/nm) for dE/dc = *m*.

        The centre gradient of the correction energy m.c is ``forces.sum(0)``.
        """
        m = np.asarray(m, dtype=float) / _FACT
        q = step._charges
        grad, hess, third = step._real
        F = np.zeros((len(q), 3))
        for k, ax in enumerate(_AXES):
            if len(ax) == 0:
                F += m[k] * grad
            elif len(ax) == 1:
                F += m[k] * hess[:, ax[0], :]
            else:
                F += m[k] * third[:, ax[0], ax[1], :]
        F *= q[:, None]
        r = step._recip
        P = sum(m[k] * r["kfac"][k] for k in range(len(EXPONENTS)))
        ph = step._phases
        n = len(ph[0][0])
        for a in range(3):
            A = (r["f"] * P * r["iK"][a] * r["eikc"]).reshape(n * n, n)
            for s0 in range(0, len(q), chunk):
                B = (A @ ph[2][s0 : s0 + chunk].T).T.reshape(-1, n, n)
                F[s0 : s0 + chunk, a] += q[s0 : s0 + chunk] * np.einsum(
                    "jab,jb,ja->j", B, ph[1][s0 : s0 + chunk], ph[0][s0 : s0 + chunk]
                ).real
        return COULOMB_KJ_MOL_NM * F
