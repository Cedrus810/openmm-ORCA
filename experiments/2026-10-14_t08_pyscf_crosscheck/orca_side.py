"""T08: the same calculation through ORCAOPIBackend (openmm_dev, ORCA 6.1.1 local on 183).

Usage: orca_side.py OUT.json LABEL [extra ORCA keywords...]
Keeps qm.inp / qm.out next to OUT.json so the SCF and integral settings can be checked.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import opi

import openmmorca
from openmmorca import units
from openmmorca.backend.base import QMRequest
from openmmorca.backend.orca_opi import ORCAConfig, ORCAOPIBackend

from geometry import PC_ANG, PC_CHARGES, QM_ANG, QM_ELEMENTS

out, label, keywords = Path(sys.argv[1]), sys.argv[2], tuple(sys.argv[3:])
backend = ORCAOPIBackend(ORCAConfig(method="HF", basis="def2-SVP", extra_keywords=keywords, restart=False))
try:
    result = backend.evaluate(QMRequest(
        qm_elements=QM_ELEMENTS, qm_positions_nm=np.array(QM_ANG) / units.ANGSTROM_PER_NM,
        mm_positions_nm=np.array(PC_ANG) / units.ANGSTROM_PER_NM, mm_charges_e=np.array(PC_CHARGES)))
    for name in ("qm.inp", "qm.out"):
        shutil.copy(backend.scratch.current / name, out.with_name(f"{label}.{name}"))
finally:
    backend.close()
to_grad = -1.0 / units.HARTREE_PER_BOHR_IN_KJ_MOL_NM  # forces kJ/mol/nm -> gradient Eh/bohr
json.dump({
    "program": "ORCA via openmmorca", "label": label, "extra_keywords": list(keywords),
    "openmmorca": openmmorca.__version__, "opi": opi.__version__,
    "energy_eh": result.energy_kj_mol / units.HARTREE_IN_KJ_MOL,
    "qm_gradient_eh_bohr": (result.qm_forces_kj_mol_nm * to_grad).tolist(),
    "pc_gradient_eh_bohr": (result.mm_forces_kj_mol_nm * to_grad).tolist(),
}, open(out, "w"), indent=2)
print(f"{label}: E = {result.energy_kj_mol / units.HARTREE_IN_KJ_MOL:.12f} Eh -> {out}")
