"""Rewrite md_a_s2/acceptance.json for job 19798, which completed all 2000 steps but stopped at a
QM-call count assertion (2001 calls, removed 2026-10-29) before writing it. Uses the run's own
acceptance_summary on steps.csv with the run's arguments (2000 steps, no --reaction-bond, 1 attempt)."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))
import enzyme_workflow as workflow  # noqa: E402

outdir = Path(__file__).resolve().parent / "md_a_s2"
with open(outdir / workflow.STEPS_CSV, newline="") as handle:
    rows = [workflow._numeric_row(row) for row in csv.DictReader(handle)]
record = json.loads((outdir / workflow.RUN_RECORD).read_text())
assert record["status"] == "completed" and record["requested_steps"] == 2000
summary = workflow.acceptance_summary(rows, 2000, rows[-1]["max_stable_bond_dev"], rows[-1]["max_bond_dev"])
summary["reaction_bonds"] = []
summary["max_reaction_bond_deviation"] = rows[-1]["max_reaction_bond_dev"]
summary["attempts"] = len(record["attempts"])
workflow.atomic_write_text(outdir / "acceptance.json", json.dumps(summary, indent=2, allow_nan=False) + "\n")
print(json.dumps(summary, indent=1))
