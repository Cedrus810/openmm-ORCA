"""Tests for the PythonForce callback with the fake backend (plan Task 8)."""

from __future__ import annotations

import numpy as np
import openmm as mm
import pytest
from openmm import unit

import helpers
from openmmorca.backend.fake import FakeBackend
from openmmorca.force import QMMMCallback, make_python_force
from openmmorca.qmmm import build_mixed_system
from openmmorca.units import COULOMB_KJ_MOL_NM

TIP3P_CHARGES = (-0.834, 0.417, 0.417)
QM_WATER = [0, 1, 2]


def build_mixed(force_group=1, n_waters=2, qm_atoms=QM_WATER):
    topology = helpers.water_topology(n_waters)
    system = helpers.flexible_tip3p_system(topology)
    positions = helpers.water_positions(n_waters)
    parts = build_mixed_system(topology, system, qm_atoms)
    backend = FakeBackend(qm_charges_e=TIP3P_CHARGES)
    callback = QMMMCallback(
        backend,
        parts.qm_atoms,
        parts.qm_elements,
        system.getNumParticles(),
        parts.mm_atoms,
        parts.mm_charges_e,
    )
    parts.system.addForce(make_python_force(callback, force_group=force_group))
    return topology, system, positions, parts, backend, callback


def make_context(system, positions):
    context = mm.Context(
        system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(positions * unit.nanometer)
    return context


def test_full_qm_energy_and_forces():
    topology = helpers.water_topology(1)
    backend = FakeBackend(qm_charges_e=TIP3P_CHARGES)
    callback = QMMMCallback(backend, (0, 1, 2), ("O", "H", "H"), 3)
    system = mm.System()
    for _ in range(3):
        system.addParticle(12.0)
    system.addForce(make_python_force(callback))
    positions = helpers.water_positions(1)
    context = make_context(system, positions)

    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)

    request = backend.last_request
    direct = backend.evaluate(request)
    assert energy == pytest.approx(direct.energy_kj_mol, abs=1e-10)
    np.testing.assert_allclose(forces, direct.qm_forces_kj_mol_nm, atol=1e-10)


def test_mixed_energy_decomposition():
    topology, system, positions, parts, backend, callback = build_mixed()
    context = make_context(parts.system, positions)

    # First evaluate directly so r0 is the equilibrium geometry.
    direct = backend.evaluate(backend.last_request or _request_from(callback, positions))

    energy_qm = context.getState(getEnergy=True, groups=1 << 1).getPotentialEnergy()
    energy_qm = energy_qm.value_in_unit(unit.kilojoule_per_mole)
    assert energy_qm == pytest.approx(direct.energy_kj_mol, abs=1e-10)

    energy_others = 0.0
    for group in range(4):
        if group == 1:
            continue
        e = context.getState(getEnergy=True, groups=1 << group).getPotentialEnergy()
        energy_others += e.value_in_unit(unit.kilojoule_per_mole)

    # Original system total minus the 9 hand-computed QM-MM Coulomb pairs.
    original_context = make_context(system, positions)
    energy_original = original_context.getState(getEnergy=True).getPotentialEnergy()
    energy_original = energy_original.value_in_unit(unit.kilojoule_per_mole)
    coulomb = 0.0
    original_charges = (-0.834, 0.417, 0.417, -0.834, 0.417, 0.417)
    for a in QM_WATER:
        for b in (3, 4, 5):
            r = np.linalg.norm(positions[a] - positions[b])
            coulomb += COULOMB_KJ_MOL_NM * original_charges[a] * original_charges[b] / r
    assert energy_others == pytest.approx(energy_original - coulomb, abs=1e-6)


def _request_from(callback, positions):
    from openmmorca.backend.base import QMRequest

    return QMRequest(
        qm_elements=callback.qm_elements,
        qm_positions_nm=positions[list(callback.qm_atoms)],
        mm_positions_nm=positions[list(callback.mm_atoms)],
        mm_charges_e=callback.mm_charges_e,
    )


def test_openmm_fd_total_energy():
    topology, system, positions, parts, backend, callback = build_mixed()
    context = make_context(parts.system, positions)

    def total_energy(pos):
        context.setPositions(pos * unit.nanometer)
        return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )

    state = context.getState(getForces=True)
    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)

    h = 1e-5
    for particle, axis in ((0, 0), (4, 1)):  # QM O x, MM H1 y
        plus, minus = positions.copy(), positions.copy()
        plus[particle, axis] += h
        minus[particle, axis] -= h
        fd = (total_energy(plus) - total_energy(minus)) / (2 * h)
        assert abs(fd - (-forces[particle, axis])) < 1e-3


def test_unsorted_qm_atoms_map_forces_correctly():
    topology = helpers.water_topology(3)
    system = helpers.flexible_tip3p_system(topology)
    positions = helpers.water_positions(3)
    parts = build_mixed_system(topology, system, [5, 3, 4])  # water 1, shuffled
    backend = FakeBackend(qm_charges_e=TIP3P_CHARGES)
    callback = QMMMCallback(
        backend, parts.qm_atoms, parts.qm_elements, 9, parts.mm_atoms, parts.mm_charges_e
    )
    parts.system.addForce(make_python_force(callback))
    context = make_context(parts.system, positions)

    forces = context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    expected = backend.evaluate(backend.last_request).qm_forces_kj_mol_nm
    np.testing.assert_allclose(forces[5], expected[0], atol=1e-10)
    np.testing.assert_allclose(forces[3], expected[1], atol=1e-10)
    np.testing.assert_allclose(forces[4], expected[2], atol=1e-10)


def test_step_counter():
    topology, system, positions, parts, backend, callback = build_mixed()
    context = make_context(parts.system, positions)
    for _ in range(3):
        context.getState(getForces=True)
    assert callback.step == 3
    assert backend.last_request.step == 2


def test_particle_count_mismatch():
    topology = helpers.water_topology(1)
    backend = FakeBackend(qm_charges_e=TIP3P_CHARGES)
    callback = QMMMCallback(backend, (0, 1, 2), ("O", "H", "H"), n_particles=6)

    # Direct invocation: a plain ValueError (OpenMM wraps it in
    # OpenMMException when it crosses the C++ boundary, see spec §2.8).
    class FakeState:
        def getPositions(self, asNumpy):
            return (helpers.water_positions(1) * unit.nanometer).value_in_unit(
                unit.nanometer
            ) * unit.nanometer

    with pytest.raises(ValueError):
        callback(FakeState())

    # Through a real Context the ValueError surfaces as OpenMMException.
    system = mm.System()
    for _ in range(3):
        system.addParticle(12.0)
    system.addForce(make_python_force(callback))
    context = make_context(system, helpers.water_positions(1))
    with pytest.raises(mm.OpenMMException, match="configured for 6"):
        context.getState(getEnergy=True)
