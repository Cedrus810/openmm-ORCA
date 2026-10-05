"""Restart reproducibility: same trajectory with restart on/off (plan Task 15).

50 NVE steps from one shared initial state (positions and velocities); per-step potential energies
must agree to < 1e-6 Eh between the MORead-restart run and the fresh-SCF run.
Roughly one minute of ORCA. With OPENMMORCA_EVIDENCE_DIR set, both energy
series are written to restart.csv there (T06).
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import openmm as mm
import pytest
from openmm import unit

import helpers
from openmmorca.potential import ORCAPotential

EH_TO_KJ_MOL = 2625.4996394799


def initial_state():
    """One shared starting point for both runs.

    Minimizing separately in each run is not reproducible: the SCF-guess noise
    (~1e-9 Eh) sends L-BFGS down different paths (2026-09-27: 205 vs 257 energy
    calls, step-0 energies 9e-5 Eh apart). So minimize once, MM only.
    """
    topology = helpers.water_topology(5)
    mm_system = helpers.flexible_tip3p_system(topology)
    context = mm.Context(
        mm_system, mm.VerletIntegrator(0.25 * unit.femtosecond), mm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(helpers.water_positions(5, spacing_nm=0.3) * unit.nanometer)
    mm.LocalEnergyMinimizer.minimize(context)
    context.setVelocitiesToTemperature(300 * unit.kelvin, 20250926)
    state = context.getState(getPositions=True, getVelocities=True)
    return topology, mm_system, state.getPositions(asNumpy=True), state.getVelocities(asNumpy=True)


def run_trajectory(scratch_root, restart: bool, start, n_steps: int = 50) -> list[float]:
    topology, mm_system, positions, velocities = start

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
    context.setPositions(positions)
    context.setVelocities(velocities)

    potentials = []
    for step in range(n_steps + 1):
        state = context.getState(getEnergy=True)
        potentials.append(
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            / EH_TO_KJ_MOL
        )
        if step < n_steps:
            integrator.step(1)
    potential.close()
    return potentials


@pytest.mark.orca
@pytest.mark.slow
def test_restart_trajectory_reproducible(tmp_path):
    start = initial_state()
    with_restart = run_trajectory(tmp_path / "on", restart=True, start=start)
    without_restart = run_trajectory(tmp_path / "off", restart=False, start=start)
    evidence = os.environ.get("OPENMMORCA_EVIDENCE_DIR")
    if evidence:
        with open(Path(evidence) / "restart.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step", "restart_eh", "fresh_eh"])
            writer.writerows((step, a, b) for step, (a, b) in enumerate(zip(with_restart, without_restart)))
    for step, (a, b) in enumerate(zip(with_restart, without_restart)):
        assert abs(a - b) < 1e-6, f"step {step}: {a} vs {b} Eh differ by {abs(a - b):.2e}"
