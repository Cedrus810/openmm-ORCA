"""Periodic QM/MM with cutoff embedding through the OpenMM layer (Task 21, spec §11).

FakeBackend, Reference platform. QM–MM electrostatics beyond the embedding
cutoff are neglected by design; these tests check the plumbing (imaging,
selection, force scatter, charge shift, box handling), not that physics.
"""

from __future__ import annotations

import numpy as np
import openmm as mm
import openmm.app as app
import pytest
from openmm import unit

from openmmorca.backend.fake import FakeBackend
from openmmorca.potential import ORCAPotential
from openmmorca.runtime.diagnostics import TIMING_FIELDS
from test_linkatoms import DIPEPTIDE_PDB, atom_index

OH = 0.09572
QM_WATER_FAKE_CHARGES = [-0.4, 0.2, 0.2]


def forcefield():
    return app.ForceField("amber14-all.xml", "amber14/tip3p.xml")


def water_box(box_nm: float = 2.5, method=app.PME, cutoff_nm: float = 1.0):
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(forcefield(), boxSize=mm.Vec3(box_nm, box_nm, box_nm) * unit.nanometer)
    system = forcefield().createSystem(
        modeller.topology,
        nonbondedMethod=method,
        nonbondedCutoff=cutoff_nm * unit.nanometer,
        constraints=None,
        rigidWater=False,
    )
    positions = modeller.getPositions().value_in_unit(unit.nanometer)
    return modeller.topology, system, np.array(positions)


def fake_potential(charges):
    charges = np.asarray(charges, dtype=float)
    return ORCAPotential("HF", basis="STO-3G", backend_factory=lambda: FakeBackend(charges))


def make_context(system, positions):
    context = mm.Context(
        system, mm.VerletIntegrator(0.0005), mm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(positions * unit.nanometer)
    return context


def box_of(system):
    return np.array(
        [v.value_in_unit(unit.nanometer) for v in system.getDefaultPeriodicBoxVectors()]
    )


def test_periodic_forces_consistent_without_crossings():
    topology, system, positions = water_box()
    potential = fake_potential(QM_WATER_FAKE_CHARGES)
    mixed = potential.createMixedSystem(
        topology, system, [0, 1, 2], embeddingCutoff=1.0 * unit.nanometer
    )
    python_force = next(f for f in mixed.getForces() if isinstance(f, mm.PythonForce))
    assert python_force.usesPeriodicBoundaryConditions()
    context = make_context(mixed, positions)
    context.getState(getEnergy=True)  # fixes the FakeBackend r0
    moved = positions.copy()
    moved[0] += np.array([0.004, -0.003, 0.002])  # distort the QM water

    # No group may sit within the finite-difference step of the cutoff, or the
    # selection would change between the two displaced evaluations.
    box = box_of(mixed)
    from openmmorca.qmmm.imaging import minimum_image

    rel = minimum_image(moved[3:] - moved[0], box)
    distances = np.linalg.norm(rel, axis=1)
    assert np.min(np.abs(distances - 1.0)) > 1e-3

    context.setPositions(moved * unit.nanometer)
    forces = context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    request = potential.backends[0].last_request
    assert request.n_mm > 0  # the embedding is really in play
    h = 1e-5
    for axis in range(3):
        energies = []
        for sign in (1, -1):
            trial = moved.copy()
            trial[0, axis] += sign * h
            context.setPositions(trial * unit.nanometer)
            energies.append(
                context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                    unit.kilojoule_per_mole
                )
            )
        fd = (energies[0] - energies[1]) / (2 * h)
        assert abs(fd + forces[0, axis]) < 1e-2, (axis, fd, -forces[0, axis])
    potential.close()


def test_qm_split_across_boundary():
    topology, system, positions = water_box()
    box = box_of(system)
    potential = fake_potential(QM_WATER_FAKE_CHARGES)
    mixed = potential.createMixedSystem(topology, system, [0, 1, 2], embeddingCutoff=1.0 * unit.nanometer)
    wrapped = positions.copy()
    wrapped[1] += box[0]  # QM H1 wrapped on its own to the other side of the box
    wrapped[2] -= box[1] + box[2]
    context = make_context(mixed, wrapped)
    context.getState(getEnergy=True)
    request = potential.backends[0].last_request
    qm = request.qm_positions_nm
    assert np.linalg.norm(qm[1] - qm[0]) == pytest.approx(
        np.linalg.norm(positions[1] - positions[0]), abs=1e-9
    )
    assert np.linalg.norm(qm[2] - qm[0]) == pytest.approx(
        np.linalg.norm(positions[2] - positions[0]), abs=1e-9
    )
    potential.close()


def test_embedding_counts_logged():
    topology, system, positions = water_box()
    potential = fake_potential(QM_WATER_FAKE_CHARGES)
    mixed = potential.createMixedSystem(topology, system, [0, 1, 2], embeddingCutoff=1.0 * unit.nanometer)
    context = make_context(mixed, positions)
    context.getState(getEnergy=True)
    assert "n_embed_groups" in TIMING_FIELDS and "embed_changed" in TIMING_FIELDS
    diagnostics = potential.backends[0].last_request.diagnostics
    assert diagnostics["n_embed_groups"] > 0
    assert diagnostics["embed_changed"] == diagnostics["n_embed_groups"]  # first step: all enter
    context.getState(getEnergy=True)
    assert potential.backends[0].last_request.diagnostics["embed_changed"] == 0
    potential.close()


def test_cutoff_periodic_rejected():
    topology, system, _ = water_box(method=app.CutoffPeriodic)
    potential = fake_potential(QM_WATER_FAKE_CHARGES)
    with pytest.raises(ValueError, match="CutoffPeriodic"):
        potential.createMixedSystem(topology, system, [0, 1, 2])
    potential.close()


def test_embedding_cutoff_checked_against_box():
    topology, system, _ = water_box()  # 2.5 nm box: half width 1.25 nm
    potential = fake_potential(QM_WATER_FAKE_CHARGES)
    with pytest.raises(ValueError, match="half"):
        potential.createMixedSystem(topology, system, [0, 1, 2], embeddingCutoff=1.3 * unit.nanometer)
    potential.close()


def solvated_dipeptide():
    pdb = app.PDBFile(str(DIPEPTIDE_PDB))
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(forcefield(), padding=1.2 * unit.nanometer)
    system = forcefield().createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=0.9 * unit.nanometer,
        constraints=None,
        rigidWater=False,
    )
    positions = np.array(modeller.getPositions().value_in_unit(unit.nanometer))
    return modeller.topology, system, positions


def test_charge_shift_with_cutoff_embedding():
    topology, system, positions = solvated_dipeptide()
    qm = [atom_index(topology, "ALA", n) for n in ("CB", "HB1", "HB2", "HB3")]
    cb, ca = atom_index(topology, "ALA", "CB"), atom_index(topology, "ALA", "CA")
    m2 = [atom_index(topology, "ALA", n) for n in ("N", "HA", "C")]
    nonbonded = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    charge = {
        i: nonbonded.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
        for i in [ca] + m2
    }
    potential = fake_potential(np.full(len(qm) + 1, 0.05))
    mixed = potential.createMixedSystem(
        topology, system, qm, boundaryPairs=[(cb, ca)], embeddingCutoff=0.8 * unit.nanometer
    )
    context = make_context(mixed, positions)
    context.getState(getEnergy=True)
    request = potential.backends[0].last_request
    assert request.qm_elements == ("C", "H", "H", "H", "H")  # methyl + link H
    # Backends see positions and charges only; find atoms by (minimum-image) position.
    from openmmorca.qmmm.imaging import minimum_image

    def embedded_rows(atom):
        rel = minimum_image(request.mm_positions_nm - positions[atom], box_of(mixed))
        return np.nonzero(np.linalg.norm(rel, axis=1) < 1e-9)[0]

    assert len(embedded_rows(ca)) == 0  # M1 is not embedded
    residual = sum(
        nonbonded.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
        for i in qm
    )  # −0.0016 e, corrected on ALA's M2 (all three are ALA atoms)
    for atom in m2:  # M2 always selected, with the shifted + corrected charge
        (row,) = embedded_rows(atom)
        assert request.mm_charges_e[row] == pytest.approx(
            charge[atom] + charge[ca] / 3 + residual / 3, abs=1e-12
        )
    potential.close()


def test_barostat_extra_evaluations():
    # 3 nm box: the soft FakeBackend water stretches under 300 K Langevin, and
    # the per-step cutoff + QM-extent guard needs room (2.5 nm is too tight).
    topology, system, positions = water_box(3.0)
    system.addForce(mm.MonteCarloBarostat(1.0 * unit.bar, 300 * unit.kelvin, 5))
    potential = fake_potential(QM_WATER_FAKE_CHARGES)
    mixed = potential.createMixedSystem(topology, system, [0, 1, 2], embeddingCutoff=1.0 * unit.nanometer)
    integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.0005)
    context = mm.Context(mixed, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions * unit.nanometer)
    integrator.step(50)
    assert potential.backends[0].n_calls == 50 + 2 * 10
    potential.close()


def test_nonperiodic_unchanged():
    """Non-periodic systems take the old path: no imaging, every charge embedded."""
    import helpers

    topology = helpers.water_topology(2)
    system = helpers.flexible_tip3p_system(topology)
    potential = fake_potential(QM_WATER_FAKE_CHARGES)
    mixed = potential.createMixedSystem(topology, system, [0, 1, 2])
    python_force = next(f for f in mixed.getForces() if isinstance(f, mm.PythonForce))
    assert not python_force.usesPeriodicBoundaryConditions()
    context = make_context(mixed, helpers.water_positions(2))
    context.getState(getEnergy=True)
    request = potential.backends[0].last_request
    assert request.n_mm == 3 and request.diagnostics == {}
    potential.close()
