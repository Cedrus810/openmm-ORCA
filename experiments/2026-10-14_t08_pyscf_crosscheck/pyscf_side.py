"""T08: PySCF HF/def2-SVP energy, QM gradient and point-charge gradient (pyscf-env on 183)."""
import json
import sys

import numpy as np
import pyscf
from pyscf import gto, qmmm, scf

from geometry import PC_ANG, PC_CHARGES, QM_ANG, QM_ELEMENTS

out = sys.argv[1]
mol = gto.M(atom=list(zip(QM_ELEMENTS, QM_ANG)), basis="def2-svp", unit="Angstrom",
            charge=0, spin=0, cart=False, verbose=0)
mf = qmmm.mm_charge(scf.RHF(mol), PC_ANG, PC_CHARGES, unit="Angstrom")
mf.conv_tol = 1e-12
mf.conv_tol_grad = 1e-8
energy = mf.kernel()
assert mf.converged
grad = mf.nuc_grad_method()
qm_grad = grad.kernel()
pc_grad = grad.grad_hcore_mm(mf.make_rdm1()) + grad.grad_nuc_mm()
json.dump({
    "program": "PySCF", "version": pyscf.__version__, "python": sys.version.split()[0],
    "settings": {"method": "RHF", "basis": "def2-svp (spherical)", "conv_tol": mf.conv_tol,
                 "conv_tol_grad": mf.conv_tol_grad, "integrals": "exact (no RI/COSX)",
                 "point_charges": "qmmm.mm_charge, point model"},
    "energy_eh": float(energy), "qm_gradient_eh_bohr": np.asarray(qm_grad).tolist(),
    "pc_gradient_eh_bohr": np.asarray(pc_grad).tolist(),
}, open(out, "w"), indent=2)
print(f"E = {energy:.12f} Eh -> {out}")
