"""T07: compare uninterrupted run A with interrupted-and-resumed run B (and B's superseded rows).

Reports per-step differences of the logged observables; no pass/fail gate. Usage: compare_ab.py DIR
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import unit

root = Path(sys.argv[1])
COLS = ["temperature_K", "d_OD_C1_nm", "d_C1_Cl1_nm", "max_bond_dev", "n_embed_groups", "scf_cycles", "restart_used"]


def rows(path):
    with open(path) as f:
        return {int(r["step"]): r for r in csv.DictReader(f)}


a, b = rows(root / "A/steps.csv"), rows(root / "B/steps.csv")
sup = rows(root / "B/steps.superseded.attempt-001.csv")
run_b = json.loads((root / "B/run.json").read_text())
report = {"run_b_status": run_b.get("status"), "attempts": run_b.get("attempts"),
          "steps_A": len(a), "steps_B": len(b), "superseded_steps": sorted(sup)}


def diff(x, y, steps):
    out = {}
    for c in COLS:
        d = [abs(float(x[s][c]) - float(y[s][c])) for s in steps]
        out[c] = {"max_abs": max(d) if d else None, "first_nonzero_step": next((s for s, v in zip(steps, d) if v), None)}
    return out


common = sorted(set(a) & set(b))
report["A_vs_B_prefix_1_20"] = diff(a, b, [s for s in common if s <= 20])
report["A_vs_B_resumed_21_end"] = diff(a, b, [s for s in common if s > 20])
report["A_vs_B_superseded"] = diff(a, sup, sorted(set(a) & set(sup)))
report["B_first_resumed_step"] = {c: b[21][c] for c in ("scf_cycles", "restart_used", "fresh_retries")}


def final_positions(d):
    s = mm.XmlSerializer.deserialize((root / d / "final_state.xml").read_text())
    return s.getPositions(asNumpy=True).value_in_unit(unit.nanometer), s.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)


(pa, va), (pb, vb) = final_positions("A"), final_positions("B")
report["final_state"] = {"max_abs_dx_nm": float(np.abs(pa - pb).max()), "rms_dx_nm": float(np.sqrt(((pa - pb) ** 2).sum(1).mean())),
                         "max_abs_dv_nm_ps": float(np.abs(va - vb).max())}
# Prefix rows must be kept byte for byte after the resume.
a_lines = (root / "A/steps.csv").read_text().splitlines()
b_lines = (root / "B/steps.csv").read_text().splitlines()
report["csv_prefix_header_plus_20_rows_identical_A_B"] = a_lines[:21] == b_lines[:21]
(root / "compare_ab.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
