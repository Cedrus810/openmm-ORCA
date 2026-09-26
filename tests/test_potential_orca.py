"""End-to-end ORCA integration through ORCAPotential (plan Task 10)."""

from __future__ import annotations

import numpy as np
import openmm as mm
import pytest
from openmm import unit

import helpers
from openmmorca.backend.base import QMRequest
from openmmorca.backend.orca_opi import ORCAConfig, ORCAOPIBackend
from openmmorca.potential import ORCAPotential
from openmmorca.units import COULOMB_KJ_MOL_NM

REFERENCE = mm.Platform.getPlatformByName("Reference")


def make_orca_potential(scratch_root) -> ORCAPotential:
    # restart=False: the exactness comparisons below assume both ORCA calls
    # take the same SCF path; MORead from a previous geometry shifts energies
    # by ~1e-7 Eh, which exceeds the 1e-6 kJ/mol tolerances. Restart behaviour
    # itself is covered in test_orca_restart.py / test_restart_reproducibility.py.
    return ORCAPotential(
        method="HF",
        basis="def2-SVP",
        extra_keywords=("TightSCF",),
        scratch_root=str(scratch_root),
        restart=False,
    )


def total_energy(context) -> float:
    return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )


def total_forces(context) -> np.ndarray:
    return context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )


@pytest.mark.orca
def test_full_qm_water_openmm_equals_backend(scratch_root):
    potential = make_orca_potential(scratch_root)
    topology = helpers.water_topology(1)
    system = potential.createSystem(topology)
    positions = helpers.water_positions(1)
    context = mm.Context(system, mm.VerletIntegrator(0.001), REFERENCE)
    context.setPositions(positions * unit.nanometer)

    energy = total_energy(context)
    forces = total_forces(context)

    backend = potential.backends[0]
    direct = backend.evaluate(
        QMRequest(qm_elements=("O", "H", "H"), qm_positions_nm=positions)
    )
    assert energy == pytest.approx(direct.energy_kj_mol, abs=1e-6)
    np.testing.assert_allclose(forces, direct.qm_forces_kj_mol_nm, atol=1e-4)
    potential.close()


@pytest.mark.orca
def test_qmmm_dimer_decomposition(scratch_root):
    potential = make_orca_potential(scratch_root)
    topology = helpers.water_topology(2)
    system = helpers.flexible_tip3p_system(topology)
    positions = helpers.water_positions(2, spacing_nm=0.35)
    mixed = potential.createMixedSystem(topology, system, [0, 1, 2], forceGroup=1)
    context = mm.Context(mixed, mm.VerletIntegrator(0.001), REFERENCE)
    context.setPositions(positions * unit.nanometer)

    # ① Force group 1 (ORCA) must equal a direct backend call with the MM
    # water's three point charges.
    backend = potential.backends[0]
    request = QMRequest(
        qm_elements=("O", "H", "H"),
        qm_positions_nm=positions[:3],
        mm_positions_nm=positions[3:],
        mm_charges_e=np.array([-0.834, 0.417, 0.417]),
    )
    direct = backend.evaluate(request)
    energy_qm = context.getState(getEnergy=True, groups=1 << 1).getPotentialEnergy()
    energy_qm = energy_qm.value_in_unit(unit.kilojoule_per_mole)
    assert energy_qm == pytest.approx(direct.energy_kj_mol, abs=1e-6)

    # ② All other groups together: water 1 bonded (0 at equilibrium) plus
    # the 9 QM-MM Lennard-Jones pairs — nothing else may be left or missing.
    energy_others = 0.0
    for group in range(4):
        if group == 1:
            continue
        e = context.getState(getEnergy=True, groups=1 << group).getPotentialEnergy()
        energy_others += e.value_in_unit(unit.kilojoule_per_mole)

    nb = None
    for force in mixed.getForces():
        if isinstance(force, mm.NonbondedForce):
            nb = force
    params = [tuple(v.value_in_unit(v.unit) for v in nb.getParticleParameters(i)) for i in range(6)]
    lj = 0.0
    for a in (0, 1, 2):
        for b in (3, 4, 5):
            _, sigma_a, eps_a = params[a]
            _, sigma_b, eps_b = params[b]
            sigma = 0.5 * (sigma_a + sigma_b)
            epsilon = (eps_a * eps_b) ** 0.5
            if epsilon == 0.0:
                continue
            r = np.linalg.norm(positions[a] - positions[b])
            lj += 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
    assert energy_others == pytest.approx(lj, abs=1e-6)
    potential.close()


@pytest.mark.orca
def test_translation_invariance(scratch_root):
    potential = make_orca_potential(scratch_root)
    topology = helpers.water_topology(2)
    system = helpers.flexible_tip3p_system(topology)
    positions = helpers.water_positions(2, spacing_nm=0.35)
    mixed = potential.createMixedSystem(topology, system, [0, 1, 2])
    context = mm.Context(mixed, mm.VerletIntegrator(0.001), REFERENCE)
    context.setPositions(positions * unit.nanometer)
    energy_a = total_energy(context)
    forces_a = total_forces(context)

    shifted = positions + np.array([1.0, -2.0, 0.5])
    context.setPositions(shifted * unit.nanometer)
    energy_b = total_energy(context)
    forces_b = total_forces(context)

    eh_in_kj = 2625.4996394799
    assert abs(energy_a - energy_b) < 1e-7 * eh_in_kj

    for forces in (forces_a, forces_b):
        norm_total = np.linalg.norm(forces.sum(axis=0))
        max_force = np.linalg.norm(forces, axis=1).max()
        assert norm_total < 1e-3 * max_force
    potential.close()


@pytest.mark.orca
def test_openmm_fd_mm_atom(scratch_root):
    potential = make_orca_potential(scratch_root)
    topology = helpers.water_topology(2)
    system = helpers.flexible_tip3p_system(topology)
    positions = helpers.water_positions(2, spacing_nm=0.35)
    mixed = potential.createMixedSystem(topology, system, [0, 1, 2])
    context = mm.Context(mixed, mm.VerletIntegrator(0.001), REFERENCE)
    context.setPositions(positions * unit.nanometer)

    def energy(pos):
        context.setPositions(pos * unit.nanometer)
        return total_energy(context)

    forces = total_forces(context)  # at initial positions

    h = 1e-4
    particle, axis = 3, 0  # MM water O, x coordinate
    plus, minus = positions.copy(), positions.copy()
    plus[particle, axis] += h
    minus[particle, axis] -= h
    fd = (energy(plus) - energy(minus)) / (2 * h)
    assert abs(fd - (-forces[particle, axis])) < 0.5
    potential.close()


@pytest.mark.orca
def test_tip4p_virtual_site_embedding(scratch_root):
    potential = make_orca_potential(scratch_root)
    topology = helpers.water_topology(2)
    positions = helpers.water_positions(2, spacing_nm=0.35)
    system, topology = helpers.tip4pew_system(topology, positions)
    # Add M-site positions (tip4pew.xml average3 weights) so the array covers
    # all 8 particles; OpenMM recomputes virtual-site positions internally.
    full_positions = np.zeros((8, 3))
    for w in range(2):
        o, h1, h2 = positions[3 * w : 3 * w + 3]
        full_positions[4 * w + 0] = o
        full_positions[4 * w + 1] = h1
        full_positions[4 * w + 2] = h2
        full_positions[4 * w + 3] = (
            0.786646558 * o + 0.106676721 * h1 + 0.106676721 * h2
        )
    mixed = potential.createMixedSystem(topology, system, [0, 1, 2])
    context = mm.Context(mixed, mm.VerletIntegrator(0.001), REFERENCE)
    context.setPositions(full_positions * unit.nanometer)
    context.getState(getEnergy=True)  # trigger one QM evaluation

    # ① The embedding file holds exactly the MM water's H1, H2 and M site
    # (its O has zero charge and the QM water's own M site is QM-side).
    backend = potential.backends[0]
    pc_path = backend.scratch.current / "pc.pc"
    lines = pc_path.read_text().splitlines()
    assert lines[0].strip() == "3"
    charges = [float(line.split()[0]) for line in lines[1:]]
    assert sorted(charges) == pytest.approx(sorted([0.52422, 0.52422, -1.04844]))

    # ② FD w.r.t. translating the whole MM water unit (O, H1, H2 and M move
    # together — Context.setPositions does NOT recompute virtual-site
    # positions) must match the summed force on O/H1/H2. This proves forces
    # on the virtual site reach the parent atoms.
    def energy(pos):
        context.setPositions(pos * unit.nanometer)
        return total_energy(context)

    forces = total_forces(context)
    h = 1e-4
    plus, minus = full_positions.copy(), full_positions.copy()
    plus[4:8, 0] += h
    minus[4:8, 0] -= h
    fd = (energy(plus) - energy(minus)) / (2 * h)
    expected_force = forces[4:7].sum(axis=0)[0]  # M itself carries no force
    assert abs(fd - (-expected_force)) < 0.5
    potential.close()
