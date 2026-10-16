"""T08: per-component differences, PySCF vs ORCA (Task 3 fixture and fresh ORCA runs).

Expected magnitude from implementation plan appendix C: energy ~1e-6 Eh, gradients ~1e-5 Eh/bohr.
Reported, not gated.
"""
import json
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
fixtures = here.parents[1] / "tests" / "data"


def read_fixture():
    lines = [l for l in (fixtures / "h2o_2pc.engrad").read_text().splitlines() if not l.startswith("#")]
    energy = float(lines[1])
    qm = np.array([float(x) for x in lines[2:11]]).reshape(3, 3)
    pc = np.loadtxt(fixtures / "h2o_2pc.pcgrad", skiprows=1)
    return {"label": "fixture (Task 3)", "energy_eh": energy, "qm_gradient_eh_bohr": qm, "pc_gradient_eh_bohr": pc}


ref = json.load(open(here / "pyscf.json"))
orca = [read_fixture()] + [json.load(open(p)) for p in sys.argv[1:]]
rows, report = [], {"pyscf": ref, "comparisons": []}
for o in orca:
    dE = o["energy_eh"] - ref["energy_eh"]
    dq = np.asarray(o["qm_gradient_eh_bohr"]) - np.asarray(ref["qm_gradient_eh_bohr"])
    dp = np.asarray(o["pc_gradient_eh_bohr"]) - np.asarray(ref["pc_gradient_eh_bohr"])
    report["comparisons"].append({"orca": o["label"], "dE_eh": dE, "dqm_grad": dq.tolist(), "dpc_grad": dp.tolist(),
                                  "max_abs_dqm": float(abs(dq).max()), "max_abs_dpc": float(abs(dp).max())})
    rows.append(f"| {o['label']} | {o['energy_eh']:.12f} | {dE:+.2e} | {abs(dq).max():.2e} | {abs(dp).max():.2e} |")
json.dump(report, open(here / "comparison.json", "w"), indent=2, default=lambda a: np.asarray(a).tolist())
print(f"PySCF {ref['version']}: E = {ref['energy_eh']:.12f} Eh\n")
print("| ORCA | E (Eh) | ΔE (Eh) | max|Δg_QM| (Eh/bohr) | max|Δg_pc| (Eh/bohr) |\n|---|---|---|---|---|")
print("\n".join(rows))
