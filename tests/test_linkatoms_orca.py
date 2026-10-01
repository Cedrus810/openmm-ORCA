"""ORCA end-to-end validation of H link atoms (Task 18, spec §10).

ACE-ALA-NME in vacuum, QM = ALA methyl side chain (CB, HB1-3) cut at CB–CA:
one link atom, QM + link H = methane, charge 0, singlet, HF/def2-SVP TightSCF.
"""

from __future__ import annotations

import numpy as np
import openmm as mm
import pytest
from openmm import unit

from openmmorca.potential import ORCAPotential
from test_linkatoms import atom_index, dipeptide, side_chain

pytestmark = pytest.mark.orca

KJ_PER_EH = 2625.4996394799


def orca_context(scratch_root):
    topology, system, positions = dipeptide()
    qm, pairs = side_chain(topology)
    potential = ORCAPotential(
        method="HF",
        basis="def2-SVP",
        extra_keywords=("TightSCF",),
        scratch_root=str(scratch_root),
    )
    mixed = potential.createMixedSystem(topology, system, qm, boundaryPairs=pairs)
    context = mm.Context(
        mixed, mm.VerletIntegrator(0.0005), mm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(positions * unit.nanometer)
    return topology, potential, context, positions


def energy_forces(context, positions):
    context.setPositions(positions * unit.nanometer)
    state = context.getState(getEnergy=True, getForces=True)
    return (
        state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
        state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer),
    )


def test_link_fd_boundary_atoms(scratch_root):
    topology, potential, context, positions = orca_context(scratch_root)
    _, forces = energy_forces(context, positions)
    atoms = {
        "CB (Q1)": atom_index(topology, "ALA", "CB"),
        "CA (M1)": atom_index(topology, "ALA", "CA"),
        "N (M2)": atom_index(topology, "ALA", "N"),
    }
    h = 1e-4
    errors = {}
    for label, particle in atoms.items():
        for axis in range(3):
            plus, minus = positions.copy(), positions.copy()
            plus[particle, axis] += h
            minus[particle, axis] -= h
            fd = (energy_forces(context, plus)[0] - energy_forces(context, minus)[0]) / (2 * h)
            errors[(label, axis)] = abs(fd + forces[particle, axis])
    worst = max(errors, key=errors.get)
    assert errors[worst] < 0.5, f"worst FD error {errors[worst]:.3f} kJ/mol/nm at {worst}"
    potential.close()


def test_link_translation_invariance(scratch_root):
    _, potential, context, positions = orca_context(scratch_root)
    e0, forces = energy_forces(context, positions)
    e1, _ = energy_forces(context, positions + np.array([1.0, 0.0, 0.0]))
    assert abs(e1 - e0) / KJ_PER_EH < 1e-7
    total = np.linalg.norm(forces.sum(axis=0))
    assert total < 1e-3 * np.linalg.norm(forces, axis=1).max()
    potential.close()
