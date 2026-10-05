"""NVE drift test for the QM/MM water cluster (plan Task 11).

This is the M2 acceptance gate (spec §1.3 v0.1). Marked ``slow``: roughly
25 minutes of serial ORCA (4000 steps × ~0.35 s). With OPENMMORCA_EVIDENCE_DIR
set, the sampled energies are written to qmmm_nve.csv there (T06).
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


def run_nve(n_steps: int = 4000, report_every: int = 10):
    topology = helpers.water_topology(5)
    mm_system = helpers.flexible_tip3p_system(topology)
    positions = helpers.water_positions(5, spacing_nm=0.3)

    potential = ORCAPotential(
        method="HF",
        basis="STO-3G",
        extra_keywords=("TightSCF",),
        extra_blocks=("%scf maxiter 200 end",),
    )
    mixed = potential.createMixedSystem(topology, mm_system, [0, 1, 2])

    integrator = mm.VerletIntegrator(0.25 * unit.femtosecond)
    context = mm.Context(mixed, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions * unit.nanometer)
    mm.LocalEnergyMinimizer.minimize(context)
    context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    helpers.remove_com_velocity(context)

    times, potentials, totals = [], [], []
    for step in range(n_steps + 1):
        if step % report_every == 0:
            state = context.getState(getEnergy=True)
            pot = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            kin = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
            times.append(step * 0.25 * 1e-3)  # ps
            potentials.append(pot)
            totals.append(pot + kin)
        if step < n_steps:
            integrator.step(1)

    potential.close()
    evidence = os.environ.get("OPENMMORCA_EVIDENCE_DIR")
    if evidence:
        with open(Path(evidence) / "qmmm_nve.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_ps", "potential_kj_mol", "total_kj_mol"])
            writer.writerows(zip(times, potentials, totals))
    times = np.asarray(times)
    totals = np.asarray(totals)
    drift = np.polyfit(times, totals, 1)[0]
    return drift, float(np.std(totals))


@pytest.mark.orca
@pytest.mark.slow
def test_nve_drift():
    drift, scatter = run_nve()
    print(f"NVE drift {drift:+.4f} kJ/mol/ps, total-energy std dev {scatter:.4f} kJ/mol")
    # Thresholds: ~3x the values measured on 2026-09-27 (drift +0.0056,
    # std 0.0163; spec §1.3). The original spec target was 0.1 / 0.5.
    assert abs(drift) < 0.017, f"NVE drift {drift:+.4f} kJ/mol/ps exceeds 0.017"
    assert scatter < 0.05, f"NVE total-energy std dev {scatter:.4f} exceeds 0.05"
