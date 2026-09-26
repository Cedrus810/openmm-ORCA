"""NVE validation for a QM/MM water cluster (plan Task 11).

QM = 1 water (HF/STO-3G, TightSCF), MM = 4 flexible TIP3P waters, non-periodic
cluster. After minimization the cluster is propagated with a 0.25 fs Verlet
integrator and no thermostat for 1 ps; total energy drift (linear fit) and
scatter are reported.

    export OPI_ORCA=/home/ruigengji/ORCA611
    /home/ruigengji/miniforge3/envs/openmm_dev/bin/python examples/qmmm_water_cluster_nve.py [csv_path]
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit


def build_cluster(n_waters: int = 5, spacing_nm: float = 0.3):
    """Flexible TIP3P waters in a compact non-periodic cluster."""
    topology = app.Topology()
    chain = topology.addChain()
    for _ in range(n_waters):
        residue = topology.addResidue("HOH", chain)
        oxygen = topology.addAtom("O", app.element.oxygen, residue)
        h1 = topology.addAtom("H1", app.element.hydrogen, residue)
        h2 = topology.addAtom("H2", app.element.hydrogen, residue)
        topology.addBond(oxygen, h1)
        topology.addBond(oxygen, h2)

    oh = 0.09572
    angle = 1.82421813418
    positions = np.zeros((3 * n_waters, 3))
    for w in range(n_waters):
        base = np.array([w * spacing_nm, 0.0, 0.0])
        positions[3 * w + 0] = base
        positions[3 * w + 1] = base + np.array([oh, 0.0, 0.0])
        positions[3 * w + 2] = base + oh * np.array([np.cos(angle), np.sin(angle), 0.0])

    forcefield = app.ForceField("tip3p.xml")
    system = forcefield.createSystem(
        topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False
    )
    return topology, system, positions


def run_nve(n_steps: int = 4000, report_every: int = 10, csv_path: Path | None = None):
    from openmmorca.potential import ORCAPotential

    topology, mm_system, positions = build_cluster()
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

    samples: list[tuple[float, float, float]] = []  # (time_ps, pot, total)
    t_start = time.perf_counter()
    for step in range(n_steps + 1):
        if step % report_every == 0:
            state = context.getState(getEnergy=True)
            pot = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            kin = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
            samples.append((step * 0.25 * 1e-3, pot, pot + kin))
        if step < n_steps:
            integrator.step(1)
    wall = time.perf_counter() - t_start

    times = np.array([s[0] for s in samples])
    totals = np.array([s[2] for s in samples])
    drift = np.polyfit(times, totals, 1)[0]
    scatter = float(np.std(totals))

    if csv_path is not None:
        with open(csv_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_ps", "potential_kj_mol", "total_kj_mol"])
            writer.writerows(samples)

    print(f"steps simulated      : {n_steps}")
    print(f"wall time            : {wall:.1f} s ({wall / n_steps * 1000:.0f} ms/step)")
    print(f"total-energy drift   : {drift:+.4f} kJ/mol/ps")
    print(f"total-energy std dev : {scatter:.4f} kJ/mol")
    timings = potential.backends[0]
    print(f"scratch directory    : {timings.scratch.root}")
    potential.close()
    return drift, scatter


if __name__ == "__main__":
    csv_out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("nve_cluster.csv")
    drift, scatter = run_nve(csv_path=csv_out)
    print(f"\nCSV written to {csv_out}")
