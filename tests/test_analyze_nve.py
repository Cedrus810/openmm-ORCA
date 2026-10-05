"""examples/analyze_nve.py on synthetic evidence CSVs (no ORCA)."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parent.parent / "examples" / "analyze_nve.py"


def write(path, header, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def analyze(directory):
    result = subprocess.run([sys.executable, str(SCRIPT), str(directory)], capture_output=True, text=True)
    return result.returncode, json.loads((directory / "summary.json").read_text())


def nve_rows(drift, noise=0.01):
    times = np.linspace(0, 1, 401)
    wiggle = noise * np.sin(40 * times)
    return [(t, -100.0, -50.0 + drift * t + w) for t, w in zip(times, wiggle)]


def test_gates_pass_and_inputs_are_hashed(tmp_path):
    write(tmp_path / "qmmm_nve.csv", ["time_ps", "potential_kj_mol", "total_kj_mol"], nve_rows(0.005))
    write(tmp_path / "oniom_nve.csv", ["time_ps", "potential_kj_mol", "total_kj_mol"], nve_rows(0.03))
    write(tmp_path / "restart.csv", ["step", "restart_eh", "fresh_eh"],
          [(k, -75.0 + 1e-3 * k, -75.0 + 1e-3 * k + 2e-9) for k in range(51)])
    code, summary = analyze(tmp_path)
    assert code == 0 and summary["failed"] == []
    assert abs(summary["qmmm_nve"]["drift_kj_mol_ps"] - 0.005) < 2e-3
    # ONIOM has no gate of its own; exceeding the QM/MM thresholds is reported only.
    assert summary["oniom_nve"]["within_qmmm_thresholds"] is False
    assert summary["restart"]["steps"] == 50 and summary["restart"]["pass"] is True
    assert abs(summary["restart"]["max_abs_difference_eh"] - 2e-9) < 1e-12
    assert set(summary["inputs"]) == {"qmmm_nve.csv", "oniom_nve.csv", "restart.csv"}


def test_failed_gates_exit_nonzero(tmp_path):
    write(tmp_path / "qmmm_nve.csv", ["time_ps", "potential_kj_mol", "total_kj_mol"], nve_rows(0.05))
    write(tmp_path / "restart.csv", ["step", "restart_eh", "fresh_eh"], [(0, -75.0, -75.0), (1, -75.0, -75.00001)])
    code, summary = analyze(tmp_path)
    assert code == 1
    assert len(summary["failed"]) == 2
    assert summary["restart"]["worst_step"] == 1
