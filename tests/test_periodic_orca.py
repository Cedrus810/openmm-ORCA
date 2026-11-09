"""Periodic QM/MM with cutoff embedding through the real ORCA backend (Task 21)."""

from __future__ import annotations

import numpy as np
import openmm as mm
import pytest
from openmm import unit

from openmmorca.potential import ORCAPotential
from test_periodic_fake import box_of, water_box

pytestmark = pytest.mark.orca

KJ_PER_EH = 2625.4996394799


def test_periodic_orca_timings_and_lattice_invariance(scratch_root):
    topology, system, positions = water_box()
    potential = ORCAPotential(
        method="HF",
        basis="STO-3G",
        extra_keywords=("TightSCF",),
        scratch_root=str(scratch_root),
    )
    mixed = potential.createMixedSystem(
        topology, system, [0, 1, 2], embeddingCutoff=1.0 * unit.nanometer
    )
    context = mm.Context(
        mixed, mm.VerletIntegrator(0.0005), mm.Platform.getPlatformByName("Reference")
    )
    box = box_of(mixed)

    def energy(pos):
        context.setPositions(pos * unit.nanometer)
        return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )

    e0 = energy(positions)
    e_shifted = energy(positions + box[0] + box[2])  # whole system, one lattice vector
    wrapped = positions.copy()
    wrapped[1] -= box[1]  # one QM H wrapped on its own
    e_wrapped = energy(wrapped)
    assert abs(e_shifted - e0) / KJ_PER_EH < 1e-7
    assert abs(e_wrapped - e0) / KJ_PER_EH < 1e-7

    rows = potential.backends[0].timings.rows
    assert len(rows) == 3
    assert all(row["n_embed_groups"] > 0 for row in rows)
    assert rows[0]["embed_changed"] == rows[0]["n_embed_groups"]
    assert rows[1]["embed_changed"] == 0 and rows[2]["embed_changed"] == 0
    potential.close()


def test_full_box_orca_invariance_and_finite_difference(scratch_root):
    """Whole-box embedding (embeddingCutoff=None) through ORCA (roadmap B4 scheme S2)."""
    topology, system, positions = water_box()
    potential = ORCAPotential(
        method="HF",
        basis="STO-3G",
        extra_keywords=("TightSCF",),
        scratch_root=str(scratch_root),
    )
    mixed = potential.createMixedSystem(topology, system, [0, 1, 2], embeddingCutoff=None)
    context = mm.Context(
        mixed, mm.VerletIntegrator(0.0005), mm.Platform.getPlatformByName("Reference")
    )
    box = box_of(mixed)
    backend = potential.backends[0]

    def energy(pos):
        context.setPositions(pos * unit.nanometer)
        return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )

    e0 = energy(positions)
    n_waters = topology.getNumResidues()
    assert backend.timings.rows[0]["n_embed_groups"] == n_waters - 1  # every MM water
    shifted = positions + box[0] + box[2]  # whole system, one lattice vector
    wrapped = positions.copy()
    wrapped[1] -= box[1]  # one QM H on its own
    wrapped[30:33] += box[0]  # one MM water on its own
    for trial in (shifted, wrapped):
        assert abs(energy(trial) - e0) / KJ_PER_EH < 1e-7
        assert backend.timings.rows[-1]["embed_changed"] == 0

    # Central differences of the Context total energy (QM O and an MM O),
    # h = 1e-4 nm, tolerance 0.5 kJ/mol/nm as in the link-atom test (Task 18).
    context.setPositions(positions * unit.nanometer)
    forces = context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    h = 1e-4
    for atom in (0, 21):
        for axis in range(3):
            e = []
            for sign in (1, -1):
                trial = positions.copy()
                trial[atom, axis] += sign * h
                e.append(energy(trial))
                assert backend.timings.rows[-1]["embed_changed"] == 0
            fd = (e[0] - e[1]) / (2 * h)
            assert abs(fd + forces[atom, axis]) < 0.5, (atom, axis, fd, -forces[atom, axis])
    potential.close()


def test_s3_orca_invariance_and_finite_difference(scratch_root):
    """S3 (whole box + classical Ewald correction, V3 plan Task 4) through ORCA and orca_vpot.

    Only the QM/MM force group is differentiated: the water box's own PME/LJ uses a 1.0 nm cutoff
    without a switch, whose finite-difference error at a far MM atom is unrelated to embedding
    (B4 §8). h = 1e-4 nm, tolerance 0.5 kJ/mol/nm as in the link-atom test (Task 18).
    """
    topology, system, positions = water_box()
    potential = ORCAPotential(
        method="HF", basis="STO-3G", extra_keywords=("TightSCF",), scratch_root=str(scratch_root)
    )
    mixed = potential.createMixedSystem(
        topology, system, [0, 1, 2], embeddingCutoff=None, ewaldCorrection=True, forceGroup=1
    )
    context = mm.Context(
        mixed, mm.VerletIntegrator(0.0005), mm.Platform.getPlatformByName("Reference")
    )
    box = box_of(mixed)
    backend = potential.backends[0]

    def energy(pos):
        context.setPositions(pos * unit.nanometer)
        return context.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )

    e0 = energy(positions)
    assert backend.timings.rows[0]["t_vpot"] > 0
    assert backend.timings.rows[0]["s3_q_absmax"] > 0
    shifted = positions + box[0] + box[2]
    wrapped = positions.copy()
    wrapped[1] -= box[1]
    wrapped[30:33] += box[0]
    for trial in (shifted, wrapped):
        assert abs(energy(trial) - e0) / KJ_PER_EH < 1e-7
        assert backend.timings.rows[-1]["embed_changed"] == 0

    context.setPositions(positions * unit.nanometer)
    forces = context.getState(getForces=True, groups={1}).getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    assert np.linalg.norm(forces.sum(axis=0)) < 1e-3 * np.abs(forces).max()
    from openmmorca.qmmm.imaging import minimum_image

    oxygens = np.arange(3, len(positions), 3)
    dist = np.linalg.norm(minimum_image(positions[oxygens] - positions[0], box), axis=1)
    far_o = int(oxygens[np.argmax(dist)])
    h = 1e-4
    for atom in (0, 1, 21, far_o):
        for axis in range(3):
            e = []
            for sign in (1, -1):
                trial = positions.copy()
                trial[atom, axis] += sign * h
                e.append(energy(trial))
                assert backend.timings.rows[-1]["embed_changed"] == 0
            fd = (e[0] - e[1]) / (2 * h)
            assert abs(fd + forces[atom, axis]) < 0.5, (atom, axis, fd, -forces[atom, axis])
    potential.close()
