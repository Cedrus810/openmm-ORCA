"""Unit conversions for the openmmorca backend boundary.

All physical constants live here and only here (spec §6.4). The backend
boundary is: coordinates in nm, charges in e in; energy in kJ/mol and
forces in kJ/mol/nm out.
"""

from __future__ import annotations

import numpy as np

ANGSTROM_PER_NM: float = 10.0
BOHR_IN_ANGSTROM: float = 0.529177210903
HARTREE_IN_KJ_MOL: float = 2625.4996394799
# Eh/bohr -> kJ/mol/nm: Eh->kJ/mol, bohr->Å (×BOHR_IN_ANGSTROM), Å->nm (÷10)
HARTREE_PER_BOHR_IN_KJ_MOL_NM: float = HARTREE_IN_KJ_MOL * ANGSTROM_PER_NM / BOHR_IN_ANGSTROM
COULOMB_KJ_MOL_NM: float = 138.93545764438198


def nm_to_angstrom(x) -> np.ndarray:
    """Convert a position array (any shape, last unit nm) to Å."""
    return np.asarray(x, dtype=float) * ANGSTROM_PER_NM


def hartree_to_kj_mol(energy_eh: float) -> float:
    return float(energy_eh) * HARTREE_IN_KJ_MOL


def gradient_to_forces(gradient_eh_bohr) -> np.ndarray:
    """Convert a gradient in Eh/bohr to forces in kJ/mol/nm (F = -dE/dR)."""
    return -np.asarray(gradient_eh_bohr, dtype=float) * HARTREE_PER_BOHR_IN_KJ_MOL_NM
