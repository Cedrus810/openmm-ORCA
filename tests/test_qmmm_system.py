"""Tests for OpenMM System modification (implementation plan Task 7)."""

from __future__ import annotations

import numpy as np
import openmm as mm
import openmm.app as app
import pytest
from openmm import unit

import helpers
from openmmorca.qmmm.system import (
    build_mixed_system,
    check_supported_forces,
    copy_system,
    get_nonbonded_force,
    particle_charges,
    zero_qm_bonded_terms,
)
from openmmorca.units import COULOMB_KJ_MOL_NM


def raw(quantity):
    """Strip an OpenMM Quantity down to its numeric value."""
    return quantity.value_in_unit(quantity.unit)


def two_water_parts(**kwargs):
    topology = helpers.water_topology(2)
    system = helpers.flexible_tip3p_system(topology)
    positions = helpers.water_positions(2)
    parts = build_mixed_system(topology, system, [0, 1, 2], **kwargs)
    return topology, system, positions, parts


# ------------------------------------------------------------------
# Bonded-term zeroing rules (spec §6.3: only all-QM terms)
# ------------------------------------------------------------------


def make_chain_system():
    system = mm.System()
    for _ in range(4):
        system.addParticle(12.0)
    bonds = mm.HarmonicBondForce()
    for a, b in ((0, 1), (1, 2), (2, 3)):
        bonds.addBond(a, b, 0.15, 100.0)
    angles = mm.HarmonicAngleForce()
    for a, b, c in ((0, 1, 2), (1, 2, 3)):
        angles.addAngle(a, b, c, 1.8, 50.0)
    torsions = mm.PeriodicTorsionForce()
    torsions.addTorsion(0, 1, 2, 3, 1, 0.0, 10.0)
    system.addForce(bonds)
    system.addForce(angles)
    system.addForce(torsions)
    return system


def test_bonded_rule_all_qm_only():
    system = make_chain_system()
    n_zeroed = zero_qm_bonded_terms(system, [0, 1, 2])
    assert n_zeroed == 3
    bonds = system.getForces()[0]
    angles = system.getForces()[1]
    torsions = system.getForces()[2]
    assert raw(bonds.getBondParameters(0)[3]) == 0.0  # bond (0,1) zeroed
    assert raw(bonds.getBondParameters(1)[3]) == 0.0  # bond (1,2) zeroed
    assert raw(bonds.getBondParameters(2)[3]) == 100.0  # bond (2,3) kept
    assert raw(angles.getAngleParameters(0)[4]) == 0.0  # angle (0,1,2) zeroed
    assert raw(angles.getAngleParameters(1)[4]) == 50.0  # angle (1,2,3) kept
    assert raw(torsions.getTorsionParameters(0)[6]) == 10.0  # torsion kept (has MM atom 3)


def test_rb_torsion_all_qm_zeroed():
    system = mm.System()
    for _ in range(4):
        system.addParticle(12.0)
    rb = mm.RBTorsionForce()
    rb.addTorsion(0, 1, 2, 3, 0.05, 0.1, -0.2, 0.3, 0.02, 0.01)
    system.addForce(rb)
    zero_qm_bonded_terms(system, [0, 1, 2, 3])
    params = system.getForces()[0].getTorsionParameters(0)
    assert all(raw(c) == 0.0 for c in params[4:])


# ------------------------------------------------------------------
# NonbondedForce modification (spec §6.2)
# ------------------------------------------------------------------


def test_qm_charges_zeroed_and_returned():
    _, system, _, parts = two_water_parts()
    charges = particle_charges(get_nonbonded_force(parts.system))
    for i in (0, 1, 2):
        assert charges[i] == 0.0
    assert parts.removed_qm_charges == {0: pytest.approx(-0.834), 1: pytest.approx(0.417), 2: pytest.approx(0.417)}


def test_qm_qm_exceptions():
    _, system, _, parts = two_water_parts()
    nb = get_nonbonded_force(parts.system)
    qm_pairs = {(0, 1), (0, 2), (1, 2)}
    found = set()
    for i in range(nb.getNumExceptions()):
        p1, p2, charge_prod, sigma, epsilon = nb.getExceptionParameters(i)
        pair = tuple(sorted((p1, p2)))
        if pair in qm_pairs:
            assert raw(charge_prod) == 0.0
            assert raw(epsilon) == 0.0
            found.add(pair)
    assert found == qm_pairs


def test_exception_count_scales_with_qm_only():
    topology = helpers.water_topology(50)
    system = helpers.flexible_tip3p_system(topology)
    parts = build_mixed_system(topology, system, [0, 1, 2])
    delta = get_nonbonded_force(parts.system).getNumExceptions() - get_nonbonded_force(
        system
    ).getNumExceptions()
    assert delta <= 3


def test_qm_mm_lj_retained():
    """Scan the QM–MM separation: the modified system's energy change equals the
    analytic Lennard-Jones change over all 9 QM×MM pairs (and nothing else)."""
    topology = helpers.water_topology(2)
    system = helpers.flexible_tip3p_system(topology)
    parts = build_mixed_system(topology, system, [0, 1, 2])
    nb = get_nonbonded_force(parts.system)

    pos_a = helpers.water_positions(2, spacing_nm=0.3)
    pos_b = pos_a.copy()
    pos_b[3:] += np.array([0.15, 0.0, 0.0])  # move MM water along x

    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(parts.system, integrator, mm.Platform.getPlatformByName("Reference"))

    def energy(positions):
        context.setPositions(positions * unit.nanometer)
        return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )

    delta_e = energy(pos_a) - energy(pos_b)

    # Analytic QM-MM LJ differential over all 9 pairs (Lorentz-Berthelot).
    def lj(r, sigma, epsilon):
        return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

    params = [tuple(raw(v) for v in nb.getParticleParameters(i)) for i in range(6)]
    expected = 0.0
    for a in (0, 1, 2):
        for b in (3, 4, 5):
            _, sigma_a, eps_a = params[a]
            _, sigma_b, eps_b = params[b]
            sigma = 0.5 * (sigma_a + sigma_b)
            epsilon = (eps_a * eps_b) ** 0.5
            if epsilon == 0.0:
                continue
            r_a = np.linalg.norm(pos_a[a] - pos_a[b])
            r_b = np.linalg.norm(pos_b[a] - pos_b[b])
            expected += lj(r_a, sigma, epsilon) - lj(r_b, sigma, epsilon)
    assert delta_e == pytest.approx(expected, abs=1e-6)


def test_mm_mm_unchanged():
    """With the QM water 100 nm away, the modified system must reproduce the
    original energy: any accidental change to MM–MM terms would show up."""
    topology = helpers.water_topology(3)
    system = helpers.flexible_tip3p_system(topology)
    parts = build_mixed_system(topology, system, [0, 1, 2])
    positions = helpers.water_positions(3)
    far = positions.copy()
    far[:3] += np.array([100.0, 0.0, 0.0])  # move the QM water away
    platform = mm.Platform.getPlatformByName("Reference")

    def energy(sys, pos):
        # Fresh integrator per Context: an Integrator cannot be bound to two
        # Contexts (this failed intermittently depending on GC timing).
        context = mm.Context(sys, mm.VerletIntegrator(0.001), platform)
        context.setPositions(pos * unit.nanometer)
        e = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )
        del context
        return e

    assert energy(parts.system, far) == pytest.approx(energy(system, far), abs=1e-6)


# ------------------------------------------------------------------
# Input validation
# ------------------------------------------------------------------


def test_unsupported_force_rejected():
    topology = helpers.water_topology(2)
    system = helpers.flexible_tip3p_system(topology)
    custom = mm.CustomExternalForce("0.5*k*x^2")
    custom.addGlobalParameter("k", 1.0)
    for i in range(3):
        custom.addParticle(i, [])
    system.addForce(custom)
    with pytest.raises(ValueError, match="CustomExternalForce"):
        build_mixed_system(topology, system, [0, 1, 2])


def test_parameter_offsets_rejected():
    topology = helpers.water_topology(2)
    system = helpers.flexible_tip3p_system(topology)
    nb = get_nonbonded_force(system)
    nb.addGlobalParameter("dq", 0.0)
    nb.addParticleParameterOffset("dq", 0, 0.1, 0.0, 0.0)
    with pytest.raises(ValueError):
        build_mixed_system(topology, system, [0, 1, 2])


def test_split_molecule_rejected():
    topology = helpers.water_topology(2)
    system = helpers.flexible_tip3p_system(topology)
    with pytest.raises(ValueError, match="whole molecules"):
        build_mixed_system(topology, system, [0, 1])


def test_bad_indices_rejected():
    topology = helpers.water_topology(2)
    system = helpers.flexible_tip3p_system(topology)
    for bad in ([], [0, 0, 1], [-1, 1, 2], [0, 1, 100]):
        with pytest.raises(ValueError):
            build_mixed_system(topology, system, bad)


def test_periodic_not_yet_supported():
    topology = helpers.water_topology(2)
    topology.setUnitCellDimensions((3.0, 3.0, 3.0) * unit.nanometer)
    forcefield = app.ForceField("tip3p.xml")
    system = forcefield.createSystem(topology, nonbondedMethod=app.PME, rigidWater=False)
    assert system.usesPeriodicBoundaryConditions()
    with pytest.raises(NotImplementedError, match="M5"):
        build_mixed_system(topology, system, [0, 1, 2])


# ------------------------------------------------------------------
# Constraints
# ------------------------------------------------------------------


def rigid_water_parts(remove_constraints):
    topology = helpers.water_topology(2)
    forcefield = app.ForceField("tip3p.xml")
    system = forcefield.createSystem(
        topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=True
    )
    assert system.getNumConstraints() == 6
    parts = build_mixed_system(
        topology, system, [0, 1, 2], remove_constraints=remove_constraints
    )
    return system, parts


def test_constraints_removed_inside_qm():
    system, parts = rigid_water_parts(remove_constraints=True)
    assert system.getNumConstraints() == 6
    assert parts.system.getNumConstraints() == 3
    for i in range(parts.system.getNumConstraints()):
        p1, p2, _ = parts.system.getConstraintParameters(i)
        assert p1 >= 3 and p2 >= 3  # only MM water constraints remain


def test_constraints_kept_with_warning():
    with pytest.warns(UserWarning):
        _, parts = rigid_water_parts(remove_constraints=False)
    assert parts.system.getNumConstraints() == 6


# ------------------------------------------------------------------
# TIP4P-Ew virtual sites (Review Focus 3)
# ------------------------------------------------------------------


def test_zero_charge_mm_atoms_are_not_embedded():
    topology = helpers.water_topology(2)
    positions = helpers.water_positions(2)
    system, topology = helpers.tip4pew_system(topology, positions)
    assert system.getNumParticles() == 8  # 2 waters × (O, H1, H2, M)
    parts = build_mixed_system(topology, system, [0, 1, 2])
    # MM water: O (particle 4) has zero charge and must not be embedded;
    # its M site (particle 7) must be.
    assert 4 not in parts.mm_atoms
    assert 7 in parts.mm_atoms
    assert 5 in parts.mm_atoms and 6 in parts.mm_atoms
    # The QM water's own M site (particle 3) belongs to the QM region.
    assert 3 not in parts.mm_atoms
    assert parts.removed_qm_charges.get(3) == pytest.approx(-1.04844)
    # MM charges sum to the MM water's net charge.
    assert np.sum(parts.mm_charges_e) == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------------
# Order preservation and small utilities
# ------------------------------------------------------------------


def test_qm_order_preserved():
    topology = helpers.water_topology(2)
    system = helpers.flexible_tip3p_system(topology)
    parts = build_mixed_system(topology, system, [2, 0, 1])
    assert parts.qm_atoms == (2, 0, 1)
    assert parts.qm_elements == ("H", "O", "H")


def test_copy_system_is_independent():
    topology = helpers.water_topology(2)
    system = helpers.flexible_tip3p_system(topology)
    copied = copy_system(system)
    nb = get_nonbonded_force(copied)
    nb.setParticleParameters(0, 0.5, 0.2, 0.1)
    original_charge = get_nonbonded_force(system).getParticleParameters(0)[0]
    assert raw(original_charge) == pytest.approx(-0.834)


def test_check_supported_forces_passes_tip3p():
    topology = helpers.water_topology(2)
    system = helpers.flexible_tip3p_system(topology)
    check_supported_forces(system)  # should not raise


def test_particle_charges():
    topology = helpers.water_topology(2)
    system = helpers.flexible_tip3p_system(topology)
    nb = get_nonbonded_force(system)
    np.testing.assert_allclose(
        particle_charges(nb), [-0.834, 0.417, 0.417, -0.834, 0.417, 0.417]
    )
