"""Pytest configuration for openmmorca tests.

ORCA-dependent tests are marked ``@pytest.mark.orca`` and are skipped
automatically when no ORCA installation can be found (either via the
``OPI_ORCA`` environment variable or on ``PATH``).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest


def _is_elf(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            return handle.read(4) == b"\x7fELF"
    except OSError:
        return False


def _orca_available() -> bool:
    """True if a real ORCA binary can be located.

    ``shutil.which("orca")`` alone is not trustworthy: on desktop Linux the
    GNOME screen reader is also called ``orca`` (a Python script in
    /usr/bin), so we additionally require an ELF binary.
    """
    orca_dir = os.environ.get("OPI_ORCA")
    if orca_dir and (Path(orca_dir) / "orca").exists():
        return True
    which_hit = shutil.which("orca")
    return bool(which_hit) and _is_elf(Path(which_hit))


def pytest_collection_modifyitems(config, items) -> None:
    if _orca_available():
        return
    skip_orca = pytest.mark.skip(reason="ORCA not found: set OPI_ORCA")
    for item in items:
        if "orca" in item.keywords:
            item.add_marker(skip_orca)


@pytest.fixture
def scratch_root(tmp_path: Path) -> Path:
    """Scratch root for backend tests, so tests never write to /tmp/openmmorca-<user>."""
    return tmp_path / "scratch"
