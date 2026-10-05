"""Cache identity, run record and acceptance checks for the DhlA example (no QM calls)."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import statistics
import tempfile
from pathlib import Path


class CacheMismatchError(ValueError):
    """An existing stage artifact cannot safely be reused."""


def sha256_file(path: Path) -> str:
    with open(path, "rb") as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """Replace a file only after its complete contents have been written."""
    _atomic_write(path, text, "w")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    _atomic_write(path, data, "wb")


def _atomic_write(path: Path, data, mode: str) -> None:
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode=mode, dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def different_fields(expected, actual, prefix="") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        fields = []
        for key in sorted(set(expected) | set(actual)):
            name = f"{prefix}.{key}" if prefix else key
            if key not in expected or key not in actual:
                fields.append(name)
            else:
                fields.extend(different_fields(expected[key], actual[key], name))
        return fields
    return [] if expected == actual else [prefix]


class StageCache:
    """Artifact + sidecar manifest. Legacy or mismatched caches fail closed."""

    def __init__(self, artifact: Path, identity: dict):
        self.artifact = Path(artifact)
        self.manifest = self.artifact.with_name(self.artifact.name + ".manifest.json")
        # Freeze the identity, so later changes to recipe dictionaries cannot
        # silently change the identity associated with an in-progress stage.
        self.identity = json.loads(json.dumps(identity, allow_nan=False))

    def reject(self, reason: str) -> None:
        raise CacheMismatchError(
            f"cache mismatch for {self.artifact}: {reason}. "
            "Use a new --outdir to rebuild; existing artifacts have been preserved"
        )

    def load(self) -> dict | None:
        if not self.artifact.exists() and not self.manifest.exists():
            return None
        if not self.artifact.is_file() or not self.manifest.is_file():
            self.reject("artifact or manifest missing (legacy/incomplete cache)")
        try:
            data = json.loads(self.manifest.read_text())
        except (OSError, ValueError):
            self.reject("manifest is unreadable or invalid JSON")
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            self.reject("unsupported manifest schema")
        if not isinstance(data.get("identity"), dict) or not isinstance(data.get("observations"), dict):
            self.reject("manifest identity or observations missing")
        different = different_fields(self.identity, data["identity"], "identity")
        if different:
            self.reject("changed fields: " + ", ".join(different))
        if data.get("artifact_sha256") != sha256_file(self.artifact):
            self.reject("artifact_sha256 changed")
        return data

    def check_observations(self, actual: dict, manifest: dict) -> None:
        different = different_fields(actual, manifest["observations"], "observations")
        if different:
            self.reject("changed fields: " + ", ".join(different))

    def store(self, observations: dict) -> None:
        data = {
            "schema_version": 1,
            "identity": self.identity,
            "artifact_sha256": sha256_file(self.artifact),
            "observations": observations,
        }
        atomic_write_text(self.manifest, json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n")


def acceptance_summary(rows, expected_steps: int, worst_stable_deviation: float,
                       worst_all_deviation: float | None = None) -> dict:
    """Short runs check finite data/stable bonds; Task 22 also needs 1 ps/500 samples.

    The example's integration step is fixed at 0.5 fs. A successful smoke run
    does not evaluate the 300 +/- 10 K full-run temperature gate.
    """
    if expected_steps <= 0:
        raise ValueError("steps must be positive")
    temperatures = [float(row["temperature_K"]) for row in rows]
    valid_temperatures = bool(temperatures) and all(math.isfinite(t) and t > 0 for t in temperatures)
    tail = temperatures[-500:]
    mean = statistics.fmean(tail) if valid_temperatures else None
    std = statistics.pstdev(tail) if valid_temperatures else None
    complete = len(rows) == expected_steps and all(
        row["step"] == step for step, row in enumerate(rows, 1)
    )
    stable = math.isfinite(worst_stable_deviation) and 0 <= worst_stable_deviation <= 0.20
    full = expected_steps >= 2000
    ok = complete and valid_temperatures and stable
    if full:
        ok = ok and len(tail) == 500 and abs(mean - 300) <= 10
    status = "PASS" if ok else "FAIL"
    if worst_all_deviation is None:
        worst_all_deviation = worst_stable_deviation
    original_bond_gate = math.isfinite(worst_all_deviation) and 0 <= worst_all_deviation <= 0.20
    task22 = "PASS" if ok and original_bond_gate else "FAIL"
    return {
        "kind": "full" if full else "smoke",
        "status": status,
        "task22_acceptance": task22 if full else "NOT_EVALUATED",
        "completed_steps": len(rows),
        "requested_steps": expected_steps,
        "duration_ps": len(rows) * 0.5 / 1000,
        "temperature_samples": len(tail),
        "temperature_mean_K": mean,
        "temperature_std_K": std,
        "max_stable_bond_deviation": worst_stable_deviation,
        "max_qm_bond_deviation": worst_all_deviation,
    }


# ---------------------------------------------------------------------------
# QM/MM run record: status, checkpoints and resume rules (T04)
# ---------------------------------------------------------------------------

RUN_RECORD = "run.json"
STEPS_CSV = "steps.csv"
CHECKPOINT_DIR = "checkpoints"
# Run products that a fresh run must not silently overwrite.
RUN_PRODUCT_PATTERNS = (
    RUN_RECORD, STEPS_CSV, "steps.superseded.*.csv", "trajectory*.dcd",
    CHECKPOINT_DIR, "final_state.xml", "final.pdb",
)


class RunMismatchError(ValueError):
    """An existing QM/MM run cannot safely be resumed or replaced."""


def existing_run_products(outdir: Path) -> list[Path]:
    return sorted({path for pattern in RUN_PRODUCT_PATTERNS for path in Path(outdir).glob(pattern)})


def archive_run_products(outdir: Path, stamp: str) -> Path:
    """Move earlier run products (and their acceptance.json) to archive/run-<stamp>/."""
    outdir = Path(outdir)
    target = outdir / "archive" / f"run-{stamp}"
    target.mkdir(parents=True, exist_ok=False)
    for path in existing_run_products(outdir) + [outdir / "acceptance.json"]:
        if path.exists():
            path.rename(target / path.name)
    return target


def load_run_record(outdir: Path) -> dict:
    path = Path(outdir) / RUN_RECORD
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise RunMismatchError(f"cannot read {path}: {exc}") from exc
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise RunMismatchError(f"{path} has an unsupported schema")
    for key in ("identity", "attempts", "status"):
        if key not in record:
            raise RunMismatchError(f"{path} lacks {key!r}")
    return record


def write_run_record(outdir: Path, record: dict) -> None:
    atomic_write_text(
        Path(outdir) / RUN_RECORD, json.dumps(record, indent=2, allow_nan=False) + "\n"
    )


def check_run_identity(expected: dict, record: dict) -> None:
    changed = different_fields(json.loads(json.dumps(expected)), record["identity"], "identity")
    if changed:
        raise RunMismatchError(
            "cannot resume: changed fields " + ", ".join(changed)
            + ". Resume with the original settings, or start again with --archive-existing"
        )


def file_prefix_sha256(path: Path, n_bytes: int) -> str:
    with open(path, "rb") as handle:
        data = handle.read(n_bytes)
    if len(data) != n_bytes:
        raise RunMismatchError(f"{path} is shorter than its checkpointed {n_bytes} bytes")
    return hashlib.sha256(data).hexdigest()


def _numeric_row(row: dict) -> dict:
    return {key: int(value) if key == "step" else float(value) for key, value in row.items()}


def truncate_steps_csv(outdir: Path, checkpoint: dict, attempt: int) -> list[dict]:
    """Keep the checkpointed CSV prefix byte for byte; preserve later rows separately.

    Rows written after the checkpoint by attempt *attempt* are moved to
    steps.superseded.attempt-NNN.csv, since the resumed run recomputes them.
    """
    outdir = Path(outdir)
    path = outdir / STEPS_CSV
    n_bytes = checkpoint["steps_csv_bytes"]
    if file_prefix_sha256(path, n_bytes) != checkpoint["steps_csv_sha256"]:
        raise RunMismatchError(f"{path} was modified before checkpoint step {checkpoint['step']}")
    data = path.read_bytes()
    prefix, tail = data[:n_bytes], data[n_bytes:]
    rows = [_numeric_row(row) for row in csv.DictReader(io.StringIO(prefix.decode()))]
    if [row["step"] for row in rows] != list(range(1, checkpoint["step"] + 1)):
        raise RunMismatchError(f"{path} does not hold steps 1..{checkpoint['step']}")
    if tail:
        header = prefix.split(b"\n", 1)[0] + b"\n"
        atomic_write_bytes(outdir / f"steps.superseded.attempt-{attempt:03d}.csv", header + tail)
        atomic_write_bytes(path, prefix)
    return rows
