"""Restart reproducibility: same trajectory with restart on/off (plan Task 15).

50 NVE steps with identical initial velocities; per-step potential energies
must agree to < 1e-6 Eh between the MORead-restart run and the fresh-SCF run.
Roughly one minute of ORCA.
"""

from __future__ import annotations

import numpy as np
import openmm as mm
import pytest
from openmm import unit

import helpers
from openmmorca.potential import ORCAPotential

EH_TO_KJ_MOL = 2625.4996394799


def run_trajectory(scratch_root, restart: bool, n_steps: int = 50) -> list[float]:
    topology = helpers.water_topology(5)
    mm_system = helpers.flexible_tip3p_system(topology)
    positions = helpers.water_positions(5, spacing_nm=0.3)

    potential = ORCAPotential(
        method="HF",
        basis="STO-3G",
        extra_keywords=("TightSCF",),
        scratch_root=str(scratch_root),
        restart=restart,
    )
    mixed = potential.createMixedSystem(topology, mm_system, [0, 1, 2])
    integrator = mm.VerletIntegrator(0.25 * unit.femtosecond)
    context = mm.Context(mixed, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions * unit.nanometer)
    mm.LocalEnergyMinimizer.minimize(context)
    context.setVelocitiesToTemperature(300 * unit.kelvin, 20250926)

    potentials = []
    for step in range(n_steps + 1):
        state = context.getState(getEnergy=True)
        potentials.append(
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mol)
            / EH_TO_KJ_MOL
        )
        if step < n_steps:
            integrator.step(1)
    potential.close()
    return potentials


@pytest.mark.orca
@pytest.mark.slow
def test_restart_trajectory_reproducible(tmp_path):
    with_restart = run_trajectory(tmp_path / "on", restart=True)
    without_restart = run_trajectory(tmp_path / "off", restart=False)
    for step, (a, b) in enumerate(zip(with_restart, without_restart)):
        assert abs(a - b) < 1e-6, f"step {step}: {a} vs {b} Eh differ by {abs(a - b):.2e}"
