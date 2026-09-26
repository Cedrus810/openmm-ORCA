"""Tests for openmmorca.runtime.scratch (implementation plan Task 4, no ORCA needed)."""

from __future__ import annotations

from pathlib import Path

from openmmorca.runtime.scratch import ScratchDir


def test_create_makes_uuid_dirs(scratch_root: Path):
    s1 = ScratchDir.create(scratch_root)
    s2 = ScratchDir.create(scratch_root)
    assert s1.root != s2.root
    assert s1.root.parent == scratch_root.resolve()
    for sub in (s1.current, s1.restart, s1.failures):
        assert sub.is_dir()


def test_reset_current_removes_files_and_subdirs(scratch_root: Path):
    s = ScratchDir.create(scratch_root)
    (s.current / "qm.engrad").write_text("junk")
    subdir = s.current / "sub"
    subdir.mkdir()
    (subdir / "inner").write_text("junk")
    s.reset_current()
    assert s.current.is_dir()
    assert list(s.current.iterdir()) == []


def test_cleanup_keeps_failures_and_timings(scratch_root: Path):
    s = ScratchDir.create(scratch_root)
    (s.failures / "failure_step_000000").mkdir()
    (s.failures / "failure_step_000000" / "qm.inp").write_text("junk")
    (s.root / "timings.csv").write_text("step,t_total\n")
    (s.current / "qm.out").write_text("junk")
    (s.restart / "last_good.gbw").write_text("junk")
    s.cleanup()
    assert (s.failures / "failure_step_000000" / "qm.inp").is_file()
    assert (s.root / "timings.csv").is_file()
    assert not s.current.exists()
    assert not s.restart.exists()
    s.cleanup()  # idempotent
    assert s.root.is_dir()


def test_cleanup_removes_empty_root(scratch_root: Path):
    s = ScratchDir.create(scratch_root)
    s.cleanup()
    assert not s.root.exists()
    s.cleanup()  # idempotent
