"""Tests for the S3 classical Ewald correction (V3 plan Task 1; no ORCA)."""

from __future__ import annotations

import numpy as np
import pytest

from openmmorca.qmmm.ewald_correction import EXPONENTS, EwaldCorrection, monomials
from openmmorca.qmmm.imaging import minimum_image
from openmmorca.units import COULOMB_KJ_MOL_NM as KE


def random_system(box, n=60, net=1.0, seed=1):
    """Charges placed at the minimum image about the box centre (S2-like), net charge *net*."""
    rng = np.random.default_rng(seed)
    center = np.asarray(box).sum(0) / 2 + rng.normal(scale=0.05, size=3)
    frac = rng.random((n, 3))
    raw = frac @ np.asarray(box)
    placed = center + minimum_image(raw - center, box)
    # keep charges away from the centre (the QM region in real use)
    keep = np.linalg.norm(placed - center, axis=1) > 0.5
    placed = placed[keep]
    q = rng.normal(size=len(placed))
    q += (net - q.sum()) / len(q)
    return center, placed, q


def reference_dV(points, placed, q, box, alpha=1.4, kmax=8, shells=3):
    """Explicit image and k sums (no separable tricks) minus the direct sum over the placed charges."""
    box = np.asarray(box, float)
    volume = abs(np.linalg.det(box))
    recip = 2 * np.pi * np.linalg.inv(box).T
    n = np.array([v for v in np.ndindex(*(2 * kmax + 1,) * 3)]) - kmax
    n = n[np.any(n != 0, axis=1)]
    k = n @ recip
    k2 = (k**2).sum(1)
    fk = 4 * np.pi / volume * np.exp(-k2 / (4 * alpha**2)) / k2
    shifts = (np.array([v for v in np.ndindex(*(2 * shells + 1,) * 3)]) - shells) @ box
    out = np.zeros(len(points))
    for i, r in enumerate(points):
        d = np.linalg.norm(r - placed[None, :, :] - shifts[:, None, :], axis=-1)
        v = (q[None] * _erfc(alpha * d) / d).sum()
        v += (fk * (q[:, None] * np.cos((r - placed) @ k.T)).sum(0)).sum()
        v -= np.pi * q.sum() / (volume * alpha**2)
        v -= (q / np.linalg.norm(r - placed, axis=1)).sum()
        out[i] = v
    return KE * out


def _erfc(x):
    import math

    return np.vectorize(math.erfc, otypes=[float])(x)


def fd_coefficients(center, fn, h=2e-3):
    """Coefficients of the 2nd-order Taylor polynomial from central differences of fn(points)."""
    e = np.eye(3)
    pts = [center]
    for a in range(3):
        pts += [center + h * e[a], center - h * e[a]]
    pairs = [(a, b) for a in range(3) for b in range(a + 1, 3)]
    for a, b in pairs:
        pts += [center + h * (s1 * e[a] + s2 * e[b]) for s1 in (1, -1) for s2 in (1, -1)]
    v = fn(np.array(pts))
    g = [(v[1 + 2 * a] - v[2 + 2 * a]) / (2 * h) for a in range(3)]
    d2 = [(v[1 + 2 * a] + v[2 + 2 * a] - 2 * v[0]) / h**2 for a in range(3)]
    off = {}
    for i, (a, b) in enumerate(pairs):
        pp, pm, mp, mm = v[7 + 4 * i : 11 + 4 * i]
        off[(a, b)] = (pp - pm - mp + mm) / (4 * h * h)
    out = []
    for ex in EXPONENTS:
        ax = [a for a in range(3) for _ in range(ex[a])]
        if not ax:
            out.append(v[0])
        elif len(ax) == 1:
            out.append(g[ax[0]])
        elif ax[0] == ax[1]:
            out.append(d2[ax[0]] / 2)
        else:
            out.append(off[tuple(ax)])
    return np.array(out)


CUBIC = np.eye(3) * 3.0
TRICLINIC = np.array([[3.0, 0.0, 0.0], [0.6, 2.9, 0.0], [-0.5, 0.7, 2.8]])


@pytest.mark.parametrize("box", [CUBIC, TRICLINIC], ids=["cubic", "triclinic"])
def test_taylor_matches_reference_dV(box):
    center, placed, q = random_system(box)
    corr = EwaldCorrection(alpha_nm=2.0, kmax=12)
    c = corr.prepare(center, placed, q, box).coefficients
    ref = fd_coefficients(center, lambda pts: reference_dV(pts, placed, q, box))
    scale = np.abs(ref).max()
    assert np.abs(c[:4] - ref[:4]).max() < 1e-5 * scale
    assert np.abs(c[4:] - ref[4:]).max() < 1e-3 * scale  # O(h^2) error of the stencil


def test_ewald_self_consistency():
    center, placed, q = random_system(CUBIC)
    c1 = EwaldCorrection(alpha_nm=2.0, kmax=10).prepare(center, placed, q, CUBIC).coefficients
    c2 = EwaldCorrection(alpha_nm=2.8, kmax=18).prepare(center, placed, q, CUBIC).coefficients
    assert np.abs(c1 - c2).max() < 1e-5 * np.abs(c1).max()


def test_sheared_cell_same_lattice():
    """The same lattice in a sheared basis gives the same correction."""
    center, placed, q = random_system(CUBIC)
    sheared = CUBIC.copy()
    sheared[1] += sheared[0]
    c1 = EwaldCorrection(alpha_nm=2.0, kmax=12).prepare(center, placed, q, CUBIC).coefficients
    c2 = EwaldCorrection(alpha_nm=2.0, kmax=24).prepare(center, placed, q, sheared).coefficients
    assert np.abs(c1 - c2).max() < 1e-5 * np.abs(c1).max()


def test_mm_forces_fd():
    center, placed, q = random_system(CUBIC)
    corr = EwaldCorrection(alpha_nm=2.0, kmax=10)
    m = np.array([-1.0, 0.25, -0.2, -0.18, 0.02, 0.01, -0.01, 0.03, 0.005, -0.02])
    F = corr.mm_forces(corr.prepare(center, placed, q, CUBIC), m)
    dist = np.linalg.norm(placed - center, axis=1)
    probes = [int(np.argmin(dist)), int(np.argmax(dist)), len(q) // 2]
    h = 1e-5
    for j in probes:
        for a in range(3):
            p1, p2 = placed.copy(), placed.copy()
            p1[j, a] += h
            p2[j, a] -= h
            e1 = m @ corr.prepare(center, p1, q, CUBIC).coefficients
            e2 = m @ corr.prepare(center, p2, q, CUBIC).coefficients
            assert F[j, a] == pytest.approx(-(e1 - e2) / (2 * h), abs=1e-6 * np.abs(F).max())


def test_center_gradient_is_force_sum():
    center, placed, q = random_system(CUBIC)
    corr = EwaldCorrection(alpha_nm=2.0, kmax=10)
    m = np.array([-1.0, 0.3, 0.1, -0.2, 0.02, -0.01, 0.0, 0.01, 0.02, -0.03])
    F = corr.mm_forces(corr.prepare(center, placed, q, CUBIC), m)
    h = 1e-5
    for a in range(3):
        dc = np.zeros(3)
        dc[a] = h
        e1 = m @ corr.prepare(center + dc, placed, q, CUBIC).coefficients
        e2 = m @ corr.prepare(center - dc, placed, q, CUBIC).coefficients
        assert F.sum(0)[a] == pytest.approx((e1 - e2) / (2 * h), abs=1e-6 * np.abs(F).max())


def test_translation_invariance():
    center, placed, q = random_system(CUBIC)
    corr = EwaldCorrection(alpha_nm=2.0, kmax=10)
    c0 = corr.prepare(center, placed, q, CUBIC).coefficients
    for shift in (np.array([0.37, -0.21, 0.05]), CUBIC[1], CUBIC[0] - 2 * CUBIC[2]):
        c1 = corr.prepare(center + shift, placed + shift, q, CUBIC).coefficients
        assert np.abs(c1 - c0).max() < 1e-8 * np.abs(c0).max()


def test_replacing_one_image_changes_only_direct_term():
    """Moving one placed charge by a lattice vector changes c by the Taylor terms of its 1/r only."""
    center, placed, q = random_system(CUBIC)
    corr = EwaldCorrection(alpha_nm=2.0, kmax=10)
    c0 = corr.prepare(center, placed, q, CUBIC).coefficients
    j = int(np.argmax(np.linalg.norm(placed - center, axis=1)))
    moved = placed.copy()
    moved[j] += CUBIC[0]
    c1 = corr.prepare(center, moved, q, CUBIC).coefficients

    def direct(pts, p):
        return KE * q[j] / np.linalg.norm(pts - p, axis=1)

    expected = fd_coefficients(center, lambda pts: direct(pts, placed[j]) - direct(pts, moved[j]))
    assert np.abs((c1 - c0) - expected).max() < 1e-3 * np.abs(expected).max() + 1e-8


def test_pseudo_charges_reproduce_polynomial():
    """Harmonic part reproduced inside the template ball; charges stay small (V1 §3, Task 0 §1)."""
    corr = EwaldCorrection()
    c = np.array([-49.09, 39.13, -51.33, 29.77, 1.86, -9.91, 2.66, -1.07, -0.17, 1.96])  # pilot frame 0
    trace = c[4] + c[7] + c[9]
    harmonic = c.copy()
    harmonic[[4, 7, 9]] -= trace / 3
    rng = np.random.default_rng(5)
    d = rng.normal(size=(500, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    x = d * (0.6 * rng.random(500) ** (1 / 3))[:, None]
    sites = corr.radius_nm * corr.directions
    Q = corr.W @ harmonic
    pot = KE * (Q[None] / np.linalg.norm(x[:, None] - sites[None], axis=-1)).sum(1)
    assert np.sqrt(((pot - monomials(x) @ harmonic) ** 2).mean()) < 0.05
    assert np.abs(corr.W @ c).max() < 1.0


def test_neutral_mm_has_no_background_term():
    center, placed, q = random_system(CUBIC, net=0.0)
    c1 = EwaldCorrection(alpha_nm=2.0, kmax=10).prepare(center, placed, q, CUBIC).coefficients
    c2 = EwaldCorrection(alpha_nm=2.8, kmax=18).prepare(center, placed, q, CUBIC).coefficients
    assert np.abs(c1 - c2).max() < 1e-5 * np.abs(c1).max()


def test_invalid_parameters():
    with pytest.raises(ValueError):
        EwaldCorrection(n_pseudo=5)
    with pytest.raises(ValueError):
        EwaldCorrection(radius_nm=0.5, template_radius_nm=0.6)
