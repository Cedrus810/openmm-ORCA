"""Split the FD error of atom 1440 (z) seen in fd_orca_water_box.log into QM (PythonForce, group 1) and MM groups."""
import sys
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import unit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from test_periodic_fake import water_box  # noqa: E402

from openmmorca.potential import ORCAPotential  # noqa: E402

topology, system, positions = water_box()
potential = ORCAPotential(method="HF", basis="STO-3G", extra_keywords=("TightSCF",), scratch_root="/dev/shm/openmmorca-kasuga-b4s2")
mixed = potential.createMixedSystem(topology, system, [0, 1, 2], embeddingCutoff=None, forceGroup=1)
context = mm.Context(mixed, mm.VerletIntegrator(0.0005), mm.Platform.getPlatformByName("Reference"))
backend = potential.backends[0]
KJ = unit.kilojoule_per_mole
for atom, axis in ((1440, 2), (1440, 0), (21, 1)):
    for groups, label in (({1}, "QM"), ({0}, "MM")):
        context.setPositions(positions * unit.nanometer)
        f = context.getState(getForces=True, groups=groups).getForces(asNumpy=True).value_in_unit(KJ / unit.nanometer)
        e = []
        for h_sign in (1, -1):
            trial = positions.copy()
            trial[atom, axis] += h_sign * 1e-4
            context.setPositions(trial * unit.nanometer)
            e.append(context.getState(getEnergy=True, groups=groups).getPotentialEnergy().value_in_unit(KJ))
        fd = (e[0] - e[1]) / 2e-4
        print(f"atom {atom} axis {axis} {label}: fd {fd:12.4f} analytic {-f[atom, axis]:12.4f} err {abs(fd + f[atom, axis]):.3e}"
              f"  embed_changed {backend.timings.rows[-1]['embed_changed']}", flush=True)
potential.close()
