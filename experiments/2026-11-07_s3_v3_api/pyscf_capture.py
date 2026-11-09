"""PySCF r2SCAN/def2-SVP energy of a captured S3 request (capture.npz), for comparison with Task 0.
Usage: pyscf_capture.py FRAME   (pyscf-env)  -> pyscf_XX.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from pyscf import gto, lib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-11-07_s3_v1_pyscf"))
from s3_v1 import EH, run_scf  # noqa: E402

f = int(sys.argv[1])
lib.num_threads(8)
z = np.load(HERE / "capture.npz")
qm, mm, q = z[f"qm_pos_{f}"], z[f"mm_pos_{f}"], z[f"mm_q_{f}"]
mol = gto.M(atom=list(zip(z["qm_elements"].tolist(), (qm * 10).tolist())), basis="def2-svp", charge=-1,
            spin=0, unit="Angstrom", verbose=3)
mol.max_memory = 6000
e = run_scf(mol, mm, q)
t0 = json.loads((HERE.parent / f"2026-11-07_s3_v3_prep/out/candidate_{f:02d}.json").read_text())
env20 = t0["env20_eh"]
rec = dict(candidate=f, energy_eh=e, R_kj_mol=(e - env20) * EH,
           task0_T2_P_r06_eh=t0["energies_eh"]["T2_P_r0.6"],
           diff_vs_task0_kj_mol=(e - t0["energies_eh"]["T2_P_r0.6"]) * EH)
(HERE / f"pyscf_{f:02d}.json").write_text(json.dumps(rec, indent=2) + "\n")
print(json.dumps(rec, indent=1))
