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
