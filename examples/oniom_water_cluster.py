"""ONIOM water cluster NVE (docs/plans/2026-09-28-oniom.md).

Two-layer subtractive ONIOM: E_ONIOM = E_high(model) + E_low(full) - E_low(model)
with HF/STO-3G on water 0 (model region) and xTB on the whole 5-water cluster
(low level). All force-field terms are absent — the OpenMM System is a pure
particle container driven by the ONIOM PythonForce.

    export OPI_ORCA=/path/to/orca
    python examples/oniom_water_cluster.py [csv_path] [--steps 4000]
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit

from qmmm_water_cluster_nve import build_cluster, remove_com_velocity


def run_nve(n_steps: int = 400, report_every: int = 10, csv_path: Path | None = None):
    from openmmorca import ONIOMPotential, ORCAPotential

    topology, ff_system, ff_positions = build_cluster()
    # Pre-minimize with the (cheap) force field: minimizing directly on the
    # ONIOM surface makes L-BFGS take large trial steps through geometries
    # where xTB refuses to converge. 0.25 fs NVE handles the residual QM
    # forces from the FF geometry (same methodology as qmmm_water_cluster_nve).
    ff_context = mm.Context(
        ff_system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference")
    )
    ff_context.setPositions(ff_positions * unit.nanometer)
    mm.LocalEnergyMinimizer.minimize(ff_context)
    positions = (
        ff_context.getState(getPositions=True)
        .getPositions(asNumpy=True)
        .value_in_unit(unit.nanometer)
    )
    del ff_context  # free the force-field Context before the ORCA-backed one

    oniom = ONIOMPotential(
        high=ORCAPotential(method="HF", basis="STO-3G", extra_keywords=("TightSCF",)),
        low=ORCAPotential(method="XTB"),
    )
    system = oniom.createONIOMSystem(topology, [0, 1, 2])

    integrator = mm.VerletIntegrator(0.25 * unit.femtosecond)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions * unit.nanometer)
    context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    remove_com_velocity(context)

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
    for summary in oniom.summarize_timings():
        print(f"  backend {summary['scratch']}: t_total mean {summary['t_total']['mean']:.2f} s")
    oniom.close()
    return drift, scatter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv_path", nargs="?", default="oniom_cluster.csv")
    parser.add_argument("--steps", type=int, default=400, help="0.25 fs steps (4000 = 1 ps)")
    args = parser.parse_args()
    csv_out = Path(args.csv_path)
    drift, scatter = run_nve(n_steps=args.steps, csv_path=csv_out)
    print(f"\nCSV written to {csv_out}")
