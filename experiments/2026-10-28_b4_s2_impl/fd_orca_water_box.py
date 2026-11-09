"""Whole-box embedding (embeddingCutoff=None) with real ORCA: lattice invariance and finite differences.

Same system as tests/test_periodic_orca.py (2.5 nm TIP3P box, QM water 0, HF/STO-3G TightSCF, Reference),
but prints the numbers: energy changes under lattice moves, and FD vs analytic force on the Context total
energy for QM O (atom 0), a QM H (atom 1) and MM O atoms 21 and the MM water O farthest from the QM O.
"""
import json
import sys
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import unit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from test_periodic_fake import box_of, water_box  # noqa: E402

from openmmorca.potential import ORCAPotential  # noqa: E402
from openmmorca.qmmm.imaging import minimum_image  # noqa: E402

KJ_PER_EH = 2625.4996394799
out = Path(__file__).resolve().parent / "fd_orca_water_box"
out.mkdir(exist_ok=True)
topology, system, positions = water_box()
potential = ORCAPotential(method="HF", basis="STO-3G", extra_keywords=("TightSCF",), scratch_root="/dev/shm/openmmorca-kasuga-b4s2")
mixed = potential.createMixedSystem(topology, system, [0, 1, 2], embeddingCutoff=None)
context = mm.Context(mixed, mm.VerletIntegrator(0.0005), mm.Platform.getPlatformByName("Reference"))
box = box_of(mixed)
backend = potential.backends[0]


def energy(pos):
    context.setPositions(pos * unit.nanometer)
    return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


e0 = energy(positions)
shifted = positions + box[0] + box[2]
wrapped = positions.copy()
wrapped[1] -= box[1]
wrapped[30:33] += box[0]
result = {"n_embed_groups": backend.timings.rows[0]["n_embed_groups"],
          "dE_lattice_shift_eh": (energy(shifted) - e0) / KJ_PER_EH,
          "dE_wrapped_eh": (energy(wrapped) - e0) / KJ_PER_EH}
o_atoms = np.arange(3, len(positions), 3)
far = int(o_atoms[np.argmax(np.linalg.norm(minimum_image(positions[o_atoms] - positions[0], box), axis=1))])
context.setPositions(positions * unit.nanometer)
forces = context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
h = 1e-4
rows = []
for atom in (0, 1, 21, far):
    for axis in range(3):
        e = []
        for sign in (1, -1):
            trial = positions.copy()
            trial[atom, axis] += sign * h
            e.append(energy(trial))
            assert backend.timings.rows[-1]["embed_changed"] == 0
        fd = (e[0] - e[1]) / (2 * h)
        rows.append({"atom": atom, "axis": axis, "fd": fd, "analytic": float(-forces[atom, axis]), "err": abs(fd + forces[atom, axis])})
result["far_mm_o"] = far
result["far_mm_o_distance_nm"] = float(np.linalg.norm(minimum_image(positions[far] - positions[0], box)))
result["fd"] = rows
result["fd_max_err_kj_mol_nm"] = max(r["err"] for r in rows)
result["t_orca_median_s"] = float(np.median([r["t_orca"] for r in backend.timings.rows[1:]]))
potential.close()
(out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "fd"}, indent=1))
for r in rows:
    print(f"atom {r['atom']:4d} axis {r['axis']}  fd {r['fd']:12.4f}  analytic {r['analytic']:12.4f}  err {r['err']:.2e}")
