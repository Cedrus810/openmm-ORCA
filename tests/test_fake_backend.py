"""Tests for openmmorca.backend.fake (implementation plan Task 6)."""

from __future__ import annotations

import numpy as np
import pytest

from openmmorca.backend.base import QMRequest
from openmmorca.backend.fake import FakeBackend

ELEMENTS = ("O", "H", "H")
REF_NM = np.array([[0.0, 0.0, 0.0], [0.096, 0.0, 0.0], [-0.024, 0.093, 0.0]])
PC_CHARGES = np.array([-0.834, 0.417])
PC_NM = np.array([[0.30, 0.0, 0.0], [0.35, 0.08, 0.0]])


def make_request(positions, mm_positions=None, mm_charges=PC_CHARGES):
    return QMRequest(
        qm_elements=ELEMENTS,
        qm_positions_nm=positions,
        mm_positions_nm=mm_positions,
        mm_charges_e=mm_charges if mm_positions is not None else None,
    )


def test_equilibrium_has_zero_bond_energy():
    backend = FakeBackend(qm_charges_e=[0.0, 0.0, 0.0])
    result = backend.evaluate(make_request(REF_NM))
    assert result.energy_kj_mol == pytest.approx(0.0, abs=1e-12)


def test_fd_qm_and_mm():
    charges = [-0.834, 0.417, 0.417]
    backend = FakeBackend(qm_charges_e=charges)
    # First call sets r0 at the reference geometry.
    backend.evaluate(make_request(REF_NM, PC_NM))
    perturbed = REF_NM + np.array([[0.003, -0.002, 0.004], [0.001, 0.0, -0.002], [0.0, 0.002, 0.001]])
    result = backend.evaluate(make_request(perturbed, PC_NM))
    h = 1e-6
    # QM coordinates (9)
    for atom in range(3):
        for axis in range(3):
            plus, minus = perturbed.copy(), perturbed.copy()
            plus[atom, axis] += h
            minus[atom, axis] -= h
            e_plus = backend.evaluate(make_request(plus, PC_NM)).energy_kj_mol
            e_minus = backend.evaluate(make_request(minus, PC_NM)).energy_kj_mol
            fd = (e_plus - e_minus) / (2 * h)
            assert abs(fd - (-result.qm_forces_kj_mol_nm[atom, axis])) < 1e-5, (
                f"QM atom {atom} axis {axis}: fd={fd} force={-result.qm_forces_kj_mol_nm[atom, axis]}"
            )
    # MM coordinates (6)
    for charge in range(2):
        for axis in range(3):
            plus, minus = PC_NM.copy(), PC_NM.copy()
            plus[charge, axis] += h
            minus[charge, axis] -= h
            e_plus = backend.evaluate(make_request(perturbed, plus)).energy_kj_mol
            e_minus = backend.evaluate(make_request(perturbed, minus)).energy_kj_mol
            fd = (e_plus - e_minus) / (2 * h)
            assert abs(fd - (-result.mm_forces_kj_mol_nm[charge, axis])) < 1e-5, (
                f"charge {charge} axis {axis}"
            )


def test_newton_third_law():
    backend = FakeBackend(qm_charges_e=[-0.834, 0.417, 0.417])
    shifted = REF_NM + 0.01
    result = backend.evaluate(make_request(shifted, PC_NM, mm_charges=PC_CHARGES + 0.1))
    total = result.qm_forces_kj_mol_nm.sum(axis=0) + result.mm_forces_kj_mol_nm.sum(axis=0)
    assert np.linalg.norm(total) < 1e-9


def test_charge_count_mismatch():
    backend = FakeBackend(qm_charges_e=[0.0, 0.0])
    with pytest.raises(ValueError):
        backend.evaluate(make_request(REF_NM))
