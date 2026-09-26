"""Tests for openmmorca.units and openmmorca.errors (implementation plan Task 1)."""

from __future__ import annotations

import numpy as np
import openmm as mm
import openmm.app
from openmm import unit

from openmmorca import errors, units


def test_force_conversion_factor():
    expected = 49614.752589
    rel_err = abs(units.HARTREE_PER_BOHR_IN_KJ_MOL_NM - expected) / expected
    assert rel_err < 1e-9


def test_gradient_to_forces_flips_sign():
    gradient = np.array([[1.0, -2.0, 0.0]])
    forces = units.gradient_to_forces(gradient)
    factor = units.HARTREE_PER_BOHR_IN_KJ_MOL_NM
    np.testing.assert_allclose(forces, [[-factor, 2.0 * factor, 0.0]], rtol=0, atol=1e-6)


def test_nm_to_angstrom():
    result = units.nm_to_angstrom([[0.1, 0.2, 0.3]])
    np.testing.assert_allclose(result, [[1.0, 2.0, 3.0]], rtol=0, atol=1e-12)


def test_coulomb_constant_matches_openmm():
    system = mm.System()
    system.addParticle(12.0)
    system.addParticle(12.0)
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    for _ in range(2):
        nonbonded.addParticle(1.0, 0.1, 0.0)
    system.addForce(nonbonded)
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(1.0, 0, 0)])
    energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    assert abs(energy - units.COULOMB_KJ_MOL_NM) < 1e-9


def test_error_hierarchy():
    for exc_class in (errors.ORCACalculationError, errors.ORCAOutputError, errors.ORCATimeoutError):
        assert issubclass(exc_class, errors.OpenMMORCAError)
        assert issubclass(exc_class, RuntimeError)
