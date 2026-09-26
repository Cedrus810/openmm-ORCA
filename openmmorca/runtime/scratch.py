"""Per-backend scratch directory (spec §9.1).

One ScratchDir instance belongs to exactly one backend instance (one OpenMM
Context). ``current/`` is wiped before every ORCA run so that any result file
found after the run was produced by *this* run; ``restart/`` holds the
last-good wavefunction (Task 13); ``failures/`` and ``timings.csv`` survive
cleanup for post-mortem analysis.
"""

from __future__ import annotations

import getpass
import shutil
import tempfile
import uuid
from pathlib import Path


class ScratchDir:
    """Directory tree ``<root>/{current,restart,failures}`` for one backend."""

    root: Path
    current: Path
    restart: Path
    failures: Path

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.current = self.root / "current"
        self.restart = self.root / "restart"
        self.failures = self.root / "failures"
        for directory in (self.root, self.current, self.restart, self.failures):
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(cls, scratch_root: str | Path | None = None) -> "ScratchDir":
        """Create a unique instance directory below *scratch_root*."""
        if scratch_root is None:
            scratch_root = Path(tempfile.gettempdir()) / f"openmmorca-{getpass.getuser()}"
        instance_dir = Path(scratch_root) / uuid.uuid4().hex
        return cls(instance_dir)

    def reset_current(self) -> None:
        """Remove everything inside ``current/`` (keeping the directory itself)."""
        self.current.mkdir(parents=True, exist_ok=True)
        for child in self.current.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)

    def cleanup(self) -> None:
        """Delete ``current/`` and ``restart/``; keep ``failures/`` and ``timings.csv``.

        Removes ``failures/`` (if empty) and the whole root (if empty) as well,
        so a backend that never failed leaves no trace. Idempotent.
        """
        shutil.rmtree(self.current, ignore_errors=True)
        shutil.rmtree(self.restart, ignore_errors=True)
        if self.failures.is_dir() and not any(self.failures.iterdir()):
            self.failures.rmdir()
        if self.root.is_dir() and not any(self.root.iterdir()):
            self.root.rmdir()
