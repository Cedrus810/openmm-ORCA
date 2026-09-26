"""SCF restart state: last-good .gbw bookkeeping (spec §9.2).

The last-good wavefunction lives *outside* ``current/`` because a failed ORCA
run may overwrite or truncate ``qm.gbw``. Staging copies it to
``current/guess.gbw`` (never the basename — ORCA overwrites ``qm.gbw`` before
reading it, spec §2.6).
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

GUESS_NAME = "guess.gbw"


class RestartState:
    """Tracks the last-good ``.gbw`` file for MORead restarts."""

    def __init__(self, restart_dir: Path, enabled: bool = True) -> None:
        self.restart_dir = Path(restart_dir)
        self.enabled = enabled
        self.restart_dir.mkdir(parents=True, exist_ok=True)

    @property
    def last_good(self) -> Path:
        return self.restart_dir / "last_good.gbw"

    @property
    def has_guess(self) -> bool:
        return self.enabled and self.last_good.is_file()

    def stage_guess(self, current_dir: Path) -> Path | None:
        """Copy the last-good wavefunction into *current_dir* as guess.gbw."""
        if not self.has_guess:
            return None
        target = Path(current_dir) / GUESS_NAME
        shutil.copy2(self.last_good, target)
        return target

    def commit(self, gbw_path: Path) -> None:
        """Atomically promote *gbw_path* to the last-good wavefunction."""
        if not self.enabled:
            return
        tmp = self.restart_dir / f"last_good.gbw.tmp-{uuid.uuid4().hex}"
        shutil.copy2(gbw_path, tmp)
        os.replace(tmp, self.last_good)

    def invalidate(self) -> None:
        """Drop the last-good wavefunction (next step runs a fresh SCF)."""
        if self.last_good.exists():
            self.last_good.unlink()
