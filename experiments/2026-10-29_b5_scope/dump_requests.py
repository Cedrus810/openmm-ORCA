"""B5 E1: dump the exact ORCA inputs (QM elements/positions incl. link H, embedded charges/positions, box)
of region A on the T09 frames for S0 (1.2 nm) and S2 (whole box), for the PySCF comparison on 183.

The requests come from the unchanged createMixedSystem path with a capturing backend (no QM run).
Usage: dump_requests.py T09_DIR OUT.npz
"""
import sys
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
import enzyme_qmmm as enzyme  # noqa: E402

from openmmorca import ORCAPotential  # noqa: E402
from openmmorca.backend.base import QMResult  # noqa: E402


class Capture:
    def __init__(self):
        self.requests = []

    def evaluate(self, request):
        self.requests.append(request)
        n_mm = request.n_mm
        return QMResult(0.0, np.zeros((request.n_qm, 3)), np.zeros((n_mm, 3)) if n_mm else None)

    def close(self):
        pass


t09, out = Path(sys.argv[1]), Path(sys.argv[2])
frames = np.load(t09 / "frames.npz")
pdb = app.PDBFile(str(t09 / "run_files" / "prepared.pdb"))
topology = pdb.topology
state = mm.XmlSerializer.deserialize((t09 / "run_files" / "equilibrated.xml").read_text())
system = enzyme.mm_system(topology, enzyme.forcefield_with_dce())
system.setDefaultPeriodicBoxVectors(*state.getPeriodicBoxVectors())
topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
qm, pairs, *_ = enzyme.qm_selection(topology, "A")
names = [a.name for a in topology.atoms()]
arrays = {"qm_atoms": np.array(qm), "qm_names": np.array([names[i] for i in qm]), "step": frames["step"],
          "box_nm": frames["box_nm"]}
for scheme, cutoff in (("S0", 1.2 * unit.nanometer), ("S2", None)):
    capture = Capture()
    potential = ORCAPotential(method="r2SCAN-3c", charge=-1, backend_factory=lambda: capture)
    mixed = potential.createMixedSystem(topology, system, qm, boundaryPairs=pairs, embeddingCutoff=cutoff)
    context = mm.Context(mixed, mm.VerletIntegrator(0.0005), mm.Platform.getPlatformByName("CPU"))
    for f in range(len(frames["step"])):
        context.setPeriodicBoxVectors(*frames["box_nm"][f])
        context.setPositions(frames["positions_nm"][f])
        context.getState(getForces=True, groups={mixed.getForce(mixed.getNumForces() - 1).getForceGroup()})
        r = capture.requests[-1]
        arrays[f"{scheme}_qm_pos_{f}"] = r.qm_positions_nm
        arrays[f"{scheme}_mm_pos_{f}"] = r.mm_positions_nm
        arrays[f"{scheme}_mm_q_{f}"] = r.mm_charges_e
    arrays["qm_elements"] = np.array(r.qm_elements)
    print(scheme, len(capture.requests), "requests; n_mm frame 0:", arrays[f"{scheme}_mm_q_0"].size,
          "sum q", round(float(arrays[f"{scheme}_mm_q_0"].sum()), 6), flush=True)
np.savez_compressed(out, **arrays)
print("->", out)
