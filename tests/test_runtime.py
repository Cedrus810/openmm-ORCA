"""Tests for runtime restart state and diagnostics (plan Task 12)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from openmmorca.runtime.diagnostics import (
    TIMING_FIELDS,
    TimingLog,
    write_failure_bundle,
)
from openmmorca.runtime.restart import GUESS_NAME, RestartState


def test_restart_no_guess_initially(tmp_path):
    state = RestartState(tmp_path)
    assert state.last_good == tmp_path / "last_good.gbw"
    assert state.has_guess is False
    assert state.stage_guess(tmp_path / "current") is None


def test_restart_commit_and_stage(tmp_path):
    state = RestartState(tmp_path)
    current = tmp_path / "current"
    current.mkdir()
    source = current / "qm.gbw"
    source.write_bytes(b"abc")
    state.commit(source)
    assert state.has_guess is True
    staged = state.stage_guess(current)
    assert staged == current / GUESS_NAME
    assert staged.read_bytes() == b"abc"


def test_restart_commit_is_atomic(tmp_path):
    state = RestartState(tmp_path)
    current = tmp_path / "current"
    current.mkdir()
    source = current / "qm.gbw"
    source.write_bytes(b"abc")
    state.commit(source)
    assert list(tmp_path.glob("*.tmp")) == []
    assert list((tmp_path).glob("last_good.gbw.tmp*")) == []


def test_restart_disabled(tmp_path):
    state = RestartState(tmp_path, enabled=False)
    current = tmp_path / "current"
    current.mkdir()
    (current / "qm.gbw").write_bytes(b"abc")
    state.commit(current / "qm.gbw")
    assert state.has_guess is False


def test_restart_invalidate(tmp_path):
    state = RestartState(tmp_path)
    current = tmp_path / "current"
    current.mkdir()
    (current / "qm.gbw").write_bytes(b"abc")
    state.commit(current / "qm.gbw")
    assert state.has_guess is True
    state.invalidate()
    assert state.has_guess is False


def test_timing_log_csv(tmp_path):
    log = TimingLog(tmp_path / "timings.csv")
    row = {field: 1.0 if field.startswith("t_") else 0 for field in TIMING_FIELDS}
    row["step"] = 0
    log.append(row)
    row2 = dict(row, step=1)
    log.append(row2)
    lines = (tmp_path / "timings.csv").read_text().splitlines()
    assert lines[0].split(",") == list(TIMING_FIELDS)
    assert len(lines) == 3


def test_timing_log_missing_field(tmp_path):
    log = TimingLog(tmp_path / "timings.csv")
    row = {field: 0.0 for field in TIMING_FIELDS}
    del row["t_orca"]
    with pytest.raises(ValueError):
        log.append(row)


def test_timing_summary_skips_cold_start(tmp_path):
    log = TimingLog(tmp_path / "timings.csv")
    for i, t_total in enumerate([10.0, 1.0, 1.0, 1.0]):
        row = {field: 0.0 for field in TIMING_FIELDS}
        row["step"] = i
        row["t_total"] = t_total
        log.append(row)
    summary = log.summarize()
    assert summary["t_total"]["mean"] == pytest.approx(1.0)
    assert summary["t_total"]["n"] == 3


def test_failure_bundle_contents(tmp_path):
    current = tmp_path / "current"
    current.mkdir()
    (current / "qm.inp").write_text("inp")
    (current / "qm.out").write_text("out")
    (current / "pc.pc").write_text("pc")
    bundle = write_failure_bundle(
        tmp_path / "failures",
        step=0,
        current_dir=current,
        qm_elements=("O", "H", "H"),
        qm_positions_ang=np.zeros((3, 3)),
        metadata={"error": "SCF not converged"},
    )
    assert (bundle / "qm.inp").read_text() == "inp"
    assert (bundle / "qm.out").read_text() == "out"
    geometry = (bundle / "geometry.xyz").read_text().splitlines()
    assert geometry[0].strip() == "3"
    metadata = json.loads((bundle / "metadata.json").read_text())
    assert metadata["error"] == "SCF not converged"


def test_failure_bundle_overwrites_same_step(tmp_path):
    current = tmp_path / "current"
    current.mkdir()
    (current / "qm.inp").write_text("first")
    failures = tmp_path / "failures"
    bundle = write_failure_bundle(failures, 7, current, ("O",), None, {"a": 1})
    assert bundle.name == "failure_step_000007"
    (current / "qm.inp").write_text("second")
    bundle2 = write_failure_bundle(failures, 7, current, ("O",), None, {"a": 2})
    assert bundle2 == bundle
    assert (bundle / "qm.inp").read_text() == "second"
    assert json.loads((bundle / "metadata.json").read_text())["a"] == 2
