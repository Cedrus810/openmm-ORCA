"""Recompute NVE and restart-reproducibility results from saved CSVs (T06).

Reads, from one evidence directory, whichever of these exist:

* qmmm_nve.csv  — tests/test_nve.py (OPENMMORCA_EVIDENCE_DIR set); gate:
  |drift| < 0.017 kJ/mol/ps and std dev < 0.05 kJ/mol (spec §1.3, v0.1)
* oniom_nve.csv — examples/oniom_water_cluster.py; no gate of its own, it is
  compared with the QM/MM thresholds (docs/plans/2026-09-28-oniom.md)
* restart.csv   — tests/test_restart_reproducibility.py; gate: every step's
  restart/fresh potential energies agree to < 1e-6 Eh

Drift is the slope of a linear least-squares fit of total energy against time.
Writes summary.json (with SHA-256 of every input) next to the CSVs. Exit code 1
if a gated result fails.

    python examples/analyze_nve.py <evidence-dir>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

DRIFT_GATE, STD_GATE = 0.017, 0.05  # kJ/mol/ps, kJ/mol
RESTART_GATE_EH = 1e-6


def read(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}


def nve(path: Path) -> dict:
    data = read(path)
    times, totals = data["time_ps"], data["total_kj_mol"]
    drift = float(np.polyfit(times, totals, 1)[0])
    std = float(np.std(totals))
    return {
        "samples": len(times), "duration_ps": float(times[-1] - times[0]),
        "drift_kj_mol_ps": drift, "std_kj_mol": std,
        "max_minus_min_kj_mol": float(totals.max() - totals.min()),
        "within_qmmm_thresholds": bool(abs(drift) < DRIFT_GATE and std < STD_GATE),
    }


def restart(path: Path) -> dict:
    data = read(path)
    difference = np.abs(data["restart_eh"] - data["fresh_eh"])
    return {
        "steps": len(difference) - 1, "max_abs_difference_eh": float(difference.max()),
        "worst_step": int(data["step"][int(difference.argmax())]),
        "pass": bool(difference.max() < RESTART_GATE_EH),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("evidence_dir")
    directory = Path(parser.parse_args().evidence_dir)
    summary, failed = {"inputs": {}}, []
    for name, analyse in (("qmmm_nve.csv", nve), ("oniom_nve.csv", nve), ("restart.csv", restart)):
        path = directory / name
        if not path.is_file():
            continue
        summary["inputs"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        result = summary[path.stem] = analyse(path)
        if name == "qmmm_nve.csv" and not result["within_qmmm_thresholds"]:
            failed.append("QM/MM NVE exceeds |drift| < 0.017 kJ/mol/ps or std < 0.05 kJ/mol")
        if name == "restart.csv" and not result["pass"]:
            failed.append(f"restart energies differ by {result['max_abs_difference_eh']:.2e} Eh")
    summary["failed"] = failed
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for key in ("qmmm_nve", "oniom_nve"):
        if key in summary:
            r = summary[key]
            print(f"{key:10s}: {r['samples']} samples / {r['duration_ps']:.2f} ps, drift "
                  f"{r['drift_kj_mol_ps']:+.4f} kJ/mol/ps, std {r['std_kj_mol']:.4f} kJ/mol, "
                  f"within QM/MM thresholds: {r['within_qmmm_thresholds']}")
    if "restart" in summary:
        r = summary["restart"]
        print(f"restart   : {r['steps']} steps, max |ΔE| {r['max_abs_difference_eh']:.2e} Eh "
              f"(step {r['worst_step']}), pass: {r['pass']}")
    for problem in failed:
        print(f"FAILED: {problem}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
