"""Recompute a DhlA QM/MM run's results from its products (T06).

Reads run.json, steps.csv, acceptance.json and the trajectory files written by
examples/enzyme_qmmm.py, and recomputes the run length, temperature, bond
deviations, embedding changes, SCF cycles/retries and timings without reusing
the run's own acceptance code. QM bond deviations are recomputed independently
from the DCD frames (every 10 steps; needs mdtraj) and must not exceed the
per-step maxima logged in steps.csv. The recomputed Task 22 verdict must match
acceptance.json. Any inconsistency is reported and gives exit code 1.

    python examples/analyze_enzyme_run.py dhla_r2scan3c [--json report.json]

Runs written before run.json existed (2026-10-01 and earlier) are analysed
with --legacy-region A|C: the QM bonds are rebuilt from prepared.pdb with the
example's region selection, the reference lengths come from the starting
positions in equilibrated.xml (those runs used no QM/MM minimization), and the
verdict to compare is the "acceptance" line of run.log. Columns those runs did
not log (SCF cycles, retries per step, stable/reaction split) are reported as
unavailable.

The verdict concerns the run's implementation gates (length, temperature,
bond stability); it is not evidence of reaction accuracy or equilibrium
sampling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

TAIL = 500
FULL_STEPS = 2000
TEMPERATURE_K, TEMPERATURE_TOL_K = 300.0, 10.0
BOND_GATE = 0.20


def percentile95(values):
    return float(np.percentile(values, 95)) if values else None


def legacy_record(outdir: Path, region: str, n_steps: int):
    """Reconstruct the run.json fields an older run did not write."""
    import openmm as mm
    import openmm.app as app
    from openmm import unit

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import enzyme_qmmm

    topology = app.PDBFile(str(outdir / "prepared.pdb")).topology
    qm = enzyme_qmmm.qm_selection(topology, region)[0]
    bonds = enzyme_qmmm.qm_bonds(topology, qm)
    state = mm.XmlSerializer.deserialize((outdir / "equilibrated.xml").read_text())
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    reference = [float(np.linalg.norm(positions[i] - positions[j])) for i, j in bonds]
    return {
        "legacy": True, "status": "completed", "requested_steps": n_steps,
        "identity": {"recipe": {"step_fs": 0.5, "dcd_interval": 10}, "qm": {"region": region}},
        "qm_bonds": [list(pair) for pair in bonds], "stable_bonds": [list(pair) for pair in bonds],
        "reference_bond_lengths_nm": reference,
        "attempts": [{"attempt": 1, "mode": "fresh", "start_step": 0, "valid_through_step": n_steps,
                      "trajectory": "trajectory.dcd", "status": "completed"}],
    }


def legacy_acceptance(outdir: Path) -> dict:
    """Verdict printed by older runs: 'acceptance : PASS' (2000-step runs only)."""
    for line in reversed((outdir / "run.log").read_text().splitlines()):
        if line.startswith("acceptance"):
            verdict = line.split(":", 1)[1].strip()
            return {"status": verdict, "task22_acceptance": verdict}
    return {}


class Analysis:
    def __init__(self, outdir: Path, legacy_region: str | None = None):
        self.outdir = Path(outdir)
        self.legacy_region = legacy_region
        self.problems: list[str] = []
        self.report: dict = {"outdir": str(self.outdir)}

    def check(self, ok: bool, message: str) -> bool:
        if not ok:
            self.problems.append(message)
        return ok

    def load(self):
        csv_path = self.outdir / "steps.csv"
        with open(csv_path, newline="") as handle:
            self.rows = [
                {key: int(value) if key == "step" else float(value) for key, value in row.items()}
                for row in csv.DictReader(handle)
            ]
        if self.legacy_region:
            logged = [line for line in (self.outdir / "run.log").read_text().splitlines()
                      if line.startswith("steps ") and ":" in line]
            if not self.check(bool(logged), "run.log has no 'steps' summary line"):
                raise SystemExit("\n".join(self.problems))
            requested = int(logged[-1].split(":", 1)[1].split()[0])
            self.record = legacy_record(self.outdir, self.legacy_region, requested)
            self.acceptance = legacy_acceptance(self.outdir)
            self.report["legacy"] = True
            self.report["artifact_sha256"] = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(self.outdir.iterdir()) if path.is_file()
            }
            # No stable/reaction split was logged: every QM bond was "stable".
            for row in self.rows:
                row.setdefault("max_stable_bond_dev", row["max_bond_dev"])
                row.setdefault("max_reaction_bond_dev", 0.0)
            return
        self.record = json.loads((self.outdir / "run.json").read_text())
        self.acceptance = json.loads((self.outdir / "acceptance.json").read_text())
        checkpoint = self.record.get("checkpoint") or {}
        data = csv_path.read_bytes()
        self.check(
            checkpoint.get("steps_csv_bytes") == len(data)
            and checkpoint.get("steps_csv_sha256") == hashlib.sha256(data).hexdigest(),
            "steps.csv differs from the final checkpoint recorded in run.json",
        )

    def attempt_of_step(self):
        """Attempt number that produced each committed step."""
        owner = {}
        for attempt in self.record["attempts"]:
            for step in range(attempt["start_step"] + 1, attempt["valid_through_step"] + 1):
                owner[step] = attempt["attempt"]
        return owner

    def length(self):
        requested = self.record["requested_steps"]
        steps = [row["step"] for row in self.rows]
        step_fs = self.record["identity"]["recipe"]["step_fs"]
        complete = self.check(steps == list(range(1, requested + 1)),
                              f"steps.csv does not hold steps 1..{requested} exactly once")
        self.check(self.record["status"] == "completed", f"run status is {self.record['status']}")
        self.report["length"] = {
            "requested_steps": requested, "logged_steps": len(steps), "complete": complete,
            "duration_ps": len(steps) * step_fs / 1000, "attempts": len(self.record["attempts"]),
        }
        return complete

    def temperature(self):
        values = [row["temperature_K"] for row in self.rows]
        finite = self.check(all(math.isfinite(t) and t > 0 for t in values),
                            "non-finite or non-positive temperatures in steps.csv")
        tail = values[-TAIL:]
        self.report["temperature"] = {
            "finite": finite, "tail_samples": len(tail),
            "tail_mean_K": statistics.fmean(tail) if finite else None,
            "tail_std_K": statistics.pstdev(tail) if finite else None,
            "min_K": min(values), "max_K": max(values),
        }
        return finite

    def bonds_from_log(self):
        result = {}
        for column in ("max_bond_dev", "max_stable_bond_dev", "max_reaction_bond_dev"):
            values = [row[column] for row in self.rows]
            self.check(all(b >= a for a, b in zip(values, values[1:])),
                       f"{column} in steps.csv is not a running maximum")
            result[column] = values[-1]
        self.report["bonds"] = result

    def bonds_from_trajectory(self):
        """Independent bond deviations at DCD frames (positions in Å, float32)."""
        try:
            import mdtraj
        except ImportError:
            self.report["bonds"]["trajectory"] = "skipped: mdtraj not installed"
            return
        bonds = np.array(self.record["qm_bonds"], dtype=int)
        stable = {tuple(pair) for pair in self.record["stable_bonds"]}
        stable_mask = np.array([tuple(pair) in stable for pair in bonds.tolist()])
        reference = np.array(self.record["reference_bond_lengths_nm"])
        interval = self.record["identity"]["recipe"]["dcd_interval"]
        by_step = {row["step"]: row for row in self.rows}
        atoms = np.unique(bonds)
        local = {atom: k for k, atom in enumerate(atoms)}
        i = np.array([local[a] for a in bonds[:, 0]])
        j = np.array([local[b] for b in bonds[:, 1]])
        worst_all = worst_stable = 0.0
        n_frames = 0
        for attempt in self.record["attempts"]:
            path = self.outdir / attempt["trajectory"]
            if not self.check(path.is_file(), f"missing trajectory {path.name}"):
                continue
            with mdtraj.formats.DCDTrajectoryFile(str(path)) as handle:
                xyz, lengths, angles = handle.read(atom_indices=atoms)
            first = (attempt["start_step"] // interval + 1) * interval
            frame_steps = range(first, first + interval * len(xyz), interval)
            for frame, step in enumerate(frame_steps):
                if step > attempt["valid_through_step"]:
                    break  # Superseded by a later attempt.
                vectors = (xyz[frame, i] - xyz[frame, j]) / 10.0  # Å -> nm
                if lengths is not None and np.all(lengths[frame] > 0):
                    if not np.allclose(angles[frame], 90.0):
                        self.problems.append(f"{path.name}: non-rectangular box not supported")
                        return
                    box = lengths[frame] / 10.0
                    vectors -= box * np.round(vectors / box)
                deviation = np.abs(np.linalg.norm(vectors, axis=1) / reference - 1)
                worst_all = max(worst_all, float(deviation.max()))
                worst_stable = max(worst_stable, float(deviation[stable_mask].max()))
                # float32 Å coordinates: allow ~1e-4 relative rounding.
                self.check(deviation.max() <= by_step[step]["max_bond_dev"] + 1e-4,
                           f"step {step}: DCD bond deviation exceeds the logged maximum")
                n_frames += 1
        self.report["bonds"]["trajectory"] = {
            "frames": n_frames, "max_bond_dev": worst_all, "max_stable_bond_dev": worst_stable,
        }

    def qm_calls(self):
        owner = self.attempt_of_step()
        first_steps = {a["start_step"] + 1 for a in self.record["attempts"]}
        # The first step of every attempt starts from a fresh SCF guess.
        steady = [row for row in self.rows if row["step"] not in first_steps]
        timing = {
            field: {"mean_s": statistics.fmean(r[field] for r in steady) if steady else None,
                    "p95_s": percentile95([r[field] for r in steady])}
            for field in ("t_write", "t_orca", "t_read", "t_total")
        }
        logged = "scf_cycles" in self.rows[0]  # Not logged before 2026-10-05.
        retries = {}
        for row in self.rows if logged else ():
            attempt = owner.get(row["step"])
            retries[attempt] = max(retries.get(attempt, 0), int(row["fresh_retries"]))
        self.report["qm_calls"] = {
            "timing_excluding_first_step_of_each_attempt": timing,
            "scf_cycles_mean": statistics.fmean(r["scf_cycles"] for r in self.rows) if logged else None,
            "scf_cycles_max": max(r["scf_cycles"] for r in self.rows) if logged else None,
            "restart_used_fraction": statistics.fmean(r["restart_used"] for r in self.rows) if logged else None,
            "fresh_retries_by_attempt": {str(k): v for k, v in sorted(retries.items())} if logged else None,
            "embedding_groups_mean": statistics.fmean(r["n_embed_groups"] for r in self.rows),
            "steps_with_embedding_changes": sum(r["embed_changed"] > 0 for r in steady),
        }

    def verdict(self, complete, finite):
        steps = self.record["requested_steps"]
        bonds = self.report["bonds"]
        stable_ok = 0 <= bonds["max_stable_bond_dev"] <= BOND_GATE
        smoke = complete and finite and stable_ok
        if steps < FULL_STEPS:
            task22 = "NOT_EVALUATED"
            status = "PASS" if smoke else "FAIL"
        else:
            temperature = self.report["temperature"]
            temperature_ok = (temperature["tail_samples"] == TAIL
                              and abs(temperature["tail_mean_K"] - TEMPERATURE_K) <= TEMPERATURE_TOL_K)
            status = "PASS" if smoke and temperature_ok else "FAIL"
            all_ok = 0 <= bonds["max_bond_dev"] <= BOND_GATE
            task22 = "PASS" if status == "PASS" and all_ok else "FAIL"
        self.report["verdict"] = {"status": status, "task22_acceptance": task22}
        for key, value in self.report["verdict"].items():
            self.check(self.acceptance.get(key) == value,
                       f"acceptance.json {key}={self.acceptance.get(key)!r}, recomputed {value!r}")

    def run(self) -> dict:
        self.load()
        complete = self.length()
        finite = self.temperature()
        self.bonds_from_log()
        self.bonds_from_trajectory()
        self.qm_calls()
        self.verdict(complete, finite)
        if not self.legacy_region:
            self.report["provenance"] = {
                "identity": self.record["identity"],
                "attempts": [{k: a.get(k) for k in ("attempt", "mode", "start_step", "valid_through_step",
                                                     "status", "platform", "nprocs", "environment")}
                             for a in self.record["attempts"]],
            }
        self.report["problems"] = self.problems
        return self.report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("outdir")
    parser.add_argument("--json", help="also write the full report to this file")
    parser.add_argument("--legacy-region", choices=("A", "C"),
                        help="analyse a run written before run.json existed (see module docstring)")
    args = parser.parse_args()
    report = Analysis(Path(args.outdir), args.legacy_region).run()
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
    length, temperature, bonds = report["length"], report["temperature"], report["bonds"]
    print(f"steps        : {length['logged_steps']}/{length['requested_steps']} "
          f"({length['duration_ps']:.3f} ps, {length['attempts']} attempt(s))")
    if temperature["finite"]:
        print(f"T (last {temperature['tail_samples']:<4d}): {temperature['tail_mean_K']:.1f} ± "
              f"{temperature['tail_std_K']:.1f} K")
    print(f"bond dev     : all {bonds['max_bond_dev'] * 100:.1f} %, stable "
          f"{bonds['max_stable_bond_dev'] * 100:.1f} %, reaction {bonds['max_reaction_bond_dev'] * 100:.1f} %")
    if isinstance(bonds.get("trajectory"), dict):
        print(f"  from DCD   : all {bonds['trajectory']['max_bond_dev'] * 100:.1f} % "
              f"over {bonds['trajectory']['frames']} frames")
    calls = report["qm_calls"]
    t_orca = calls["timing_excluding_first_step_of_each_attempt"]["t_orca"]
    if t_orca["mean_s"] is not None:
        print(f"t_orca       : mean {t_orca['mean_s']:.3f} s, p95 {t_orca['p95_s']:.3f} s "
              "(first step of each attempt excluded)")
    if calls["scf_cycles_mean"] is not None:
        print(f"SCF cycles   : mean {calls['scf_cycles_mean']:.1f}, max {calls['scf_cycles_max']:.0f}; "
              f"fresh retries {calls['fresh_retries_by_attempt']}")
    else:
        print("SCF cycles   : not logged by this run")
    print(f"embedding    : {calls['embedding_groups_mean']:.0f} groups mean, "
          f"{calls['steps_with_embedding_changes']} steps with crossings")
    print(f"verdict      : {report['verdict']['status']}, Task 22 {report['verdict']['task22_acceptance']}")
    for problem in report["problems"]:
        print(f"PROBLEM: {problem}")
    raise SystemExit(1 if report["problems"] else 0)


if __name__ == "__main__":
    main()
