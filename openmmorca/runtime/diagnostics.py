"""Timing log and failure bundles (spec §9.3, §9.4)."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np

TIMING_FIELDS = (
    "step",
    "t_write",
    "t_orca",
    "t_read",
    "t_total",
    "scf_cycles",
    "restart_used",
    "fresh_retries",
)

BUNDLE_FILES = (
    "qm.inp",
    "qm.out",
    "qm.err",
    "pc.pc",
    "qm.engrad",
    "qm.pcgrad",
    "guess.gbw",
)


class TimingLog:
    """CSV log of per-step timings; first row of summarize() skips cold start."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.rows: list[dict] = []

    def append(self, row: dict) -> None:
        missing = [field for field in TIMING_FIELDS if field not in row]
        if missing:
            raise ValueError(f"timing row is missing fields: {missing}")
        write_header = not self.path.exists()
        self.rows.append({field: row[field] for field in TIMING_FIELDS})
        with open(self.path, "a", newline="") as handle:
            writer = csv.writer(handle)
            if write_header:
                writer.writerow(TIMING_FIELDS)
            writer.writerow([row[field] for field in TIMING_FIELDS])

    def summarize(self, skip_first: bool = True) -> dict[str, dict[str, float]]:
        rows = self.rows[1:] if (skip_first and len(self.rows) > 1) else self.rows
        summary: dict[str, dict[str, float]] = {}
        for field in ("t_write", "t_orca", "t_read", "t_total"):
            values = np.asarray([float(row[field]) for row in rows])
            if values.size == 0:
                continue
            summary[field] = {
                "mean": float(values.mean()),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "n": int(values.size),
            }
        return summary


def write_failure_bundle(
    failures_dir: Path,
    step: int,
    current_dir: Path,
    qm_elements,
    qm_positions_ang,
    metadata: dict,
) -> Path:
    """Copy everything needed to reproduce a failed step into a bundle directory."""
    bundle = Path(failures_dir) / f"failure_step_{step:06d}"
    bundle.mkdir(parents=True, exist_ok=True)
    current_dir = Path(current_dir)
    for name in BUNDLE_FILES:
        source = current_dir / name
        if source.is_file():
            shutil.copy2(source, bundle / name)
    if qm_elements is not None:
        with open(bundle / "geometry.xyz", "w") as handle:
            handle.write(f"{len(qm_elements)}\n\n")
            for element, row in zip(
                qm_elements,
                np.asarray(qm_positions_ang if qm_positions_ang is not None else []).reshape(-1, 3),
            ):
                handle.write(f"{element} {row[0]:.10f} {row[1]:.10f} {row[2]:.10f}\n")
    with open(bundle / "metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2, default=str)
    return bundle
