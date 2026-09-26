"""Finite-difference validation of the ORCA backend (implementation plan Task 5).

This is the M0 acceptance gate: it proves that energies, unit conversion and
the sign convention (force = −gradient) close mathematically, for QM forces
(.engrad) as well as for point-charge reaction forces (.pcgrad).

All comparisons are done in Eh/bohr. h = 1e-3 Å = 1e-4 nm.
"""

from __future__ import annotations

import numpy as np
import pytest

from openmmorca.backend.base import QMRequest
from openmmorca.backend.orca_opi import ORCAConfig, ORCAOPIBackend
from openmmorca.units import BOHR_IN_ANGSTROM, HARTREE_PER_BOHR_IN_KJ_MOL_NM

BOHR_IN_NM = BOHR_IN_ANGSTROM / 10.0
H_NM = 1e-4  # 1e-3 Å
FD_THRESHOLD_EH_BOHR = 1e-4

WATER_ELEMENTS = ("O", "H", "H")
WATER_NM = np.array([[0.0, 0.0, 0.0], [0.096, 0.0, 0.0], [-0.024, 0.093, 0.0]])
PC_CHARGES = np.array([-0.834, 0.417])
PC_NM = np.array([[0.30, 0.0, 0.0], [0.35, 0.08, 0.0]])


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    config = ORCAConfig(
        method="HF",
        basis="def2-SVP",
        extra_keywords=("TightSCF",),
        scratch_root=str(tmp_path_factory.mktemp("fd_scratch")),
    )
    backend = ORCAOPIBackend(config, check_version=False)
    yield backend
    backend.close()


def energy_eh(bk: ORCAOPIBackend, qm_nm: np.ndarray, mm_nm: np.ndarray | None) -> float:
    request = QMRequest(
        qm_elements=WATER_ELEMENTS,
        qm_positions_nm=qm_nm,
        mm_positions_nm=None if mm_nm is None else mm_nm,
        mm_charges_e=None if mm_nm is None else PC_CHARGES,
    )
    result = bk.evaluate(request)
    return result.energy_kj_mol / 2625.4996394799


def fd_gradient_eh_bohr(
    bk: ORCAOPIBackend, qm_nm: np.ndarray, mm_nm: np.ndarray | None, which: tuple[int, int]
) -> float:
    """Central finite difference dE/dR (Eh/bohr) for QM atom `which[0]`, axis `which[1]`.

    For point charges, pass an index >= 0 beyond the QM atoms is not needed:
    use `fd_gradient_pc_eh_bohr` instead.
    """
    atom, axis = which
    plus, minus = qm_nm.copy(), qm_nm.copy()
    plus[atom, axis] += H_NM
    minus[atom, axis] -= H_NM
    e_plus = energy_eh(bk, plus, mm_nm)
    e_minus = energy_eh(bk, minus, mm_nm)
    return (e_plus - e_minus) / (2.0 * H_NM) * BOHR_IN_NM


def fd_gradient_pc_eh_bohr(
    bk: ORCAOPIBackend, qm_nm: np.ndarray, mm_nm: np.ndarray, which: tuple[int, int]
) -> float:
    """Central finite difference dE/dR_pc (Eh/bohr) for point charge `which[0]`."""
    charge, axis = which
    plus, minus = mm_nm.copy(), mm_nm.copy()
    plus[charge, axis] += H_NM
    minus[charge, axis] -= H_NM
    e_plus = energy_eh(bk, qm_nm, plus)
    e_minus = energy_eh(bk, qm_nm, minus)
    return (e_plus - e_minus) / (2.0 * H_NM) * BOHR_IN_NM


def analytic_gradient_eh_bohr(result) -> np.ndarray:
    """Convert QM forces (kJ/mol/nm) back to a gradient in Eh/bohr."""
    return -result.qm_forces_kj_mol_nm / HARTREE_PER_BOHR_IN_KJ_MOL_NM


def analytic_pc_gradient_eh_bohr(result) -> np.ndarray:
    return -result.mm_forces_kj_mol_nm / HARTREE_PER_BOHR_IN_KJ_MOL_NM


def evaluate(bk: ORCAOPIBackend, qm_nm: np.ndarray, mm_nm: np.ndarray | None):
    return bk.evaluate(
        QMRequest(
            qm_elements=WATER_ELEMENTS,
            qm_positions_nm=qm_nm,
            mm_positions_nm=mm_nm,
            mm_charges_e=None if mm_nm is None else PC_CHARGES,
        )
    )


@pytest.mark.orca
def test_fd_qm_atoms_full_qm(backend):
    result = evaluate(backend, WATER_NM, None)
    gradient = analytic_gradient_eh_bohr(result)
    for atom in range(3):
        for axis in range(3):
            fd = fd_gradient_eh_bohr(backend, WATER_NM, None, (atom, axis))
            assert abs(fd - gradient[atom, axis]) < FD_THRESHOLD_EH_BOHR, (
                f"atom {atom} axis {axis}: fd={fd:.10f} analytic={gradient[atom, axis]:.10f}"
            )


@pytest.mark.orca
def test_fd_qm_atoms_with_charges(backend):
    result = evaluate(backend, WATER_NM, PC_NM)
    gradient = analytic_gradient_eh_bohr(result)
    for atom in range(3):
        for axis in range(3):
            fd = fd_gradient_eh_bohr(backend, WATER_NM, PC_NM, (atom, axis))
            assert abs(fd - gradient[atom, axis]) < FD_THRESHOLD_EH_BOHR, (
                f"atom {atom} axis {axis}: fd={fd:.10f} analytic={gradient[atom, axis]:.10f}"
            )


@pytest.mark.orca
def test_fd_point_charges(backend):
    result = evaluate(backend, WATER_NM, PC_NM)
    pc_gradient = analytic_pc_gradient_eh_bohr(result)
    for charge in range(2):
        for axis in range(3):
            fd = fd_gradient_pc_eh_bohr(backend, WATER_NM, PC_NM, (charge, axis))
            assert abs(fd - pc_gradient[charge, axis]) < FD_THRESHOLD_EH_BOHR, (
                f"charge {charge} axis {axis}: fd={fd:.10f} analytic={pc_gradient[charge, axis]:.10f}"
            )


@pytest.mark.orca
def test_forces_sum_to_zero(backend):
    result = evaluate(backend, WATER_NM, PC_NM)
    total = result.qm_forces_kj_mol_nm.sum(axis=0) + result.mm_forces_kj_mol_nm.sum(axis=0)
    norm_total = np.linalg.norm(total)
    max_force = max(
        np.linalg.norm(result.qm_forces_kj_mol_nm, axis=1).max(),
        np.linalg.norm(result.mm_forces_kj_mol_nm, axis=1).max(),
    )
    assert norm_total < 1e-3 * max_force
