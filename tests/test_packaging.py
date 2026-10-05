"""Version consistency between source, project metadata and the installed distribution."""

from __future__ import annotations

import tomllib
from importlib import metadata
from pathlib import Path

import pytest

import openmmorca

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_project_version_is_read_from_the_package():
    if not PYPROJECT.is_file():
        pytest.skip("not a source checkout")
    project = tomllib.loads(PYPROJECT.read_text())
    assert "version" not in project["project"]
    assert "version" in project["project"]["dynamic"]
    assert project["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "openmmorca.__version__"}


def test_installed_metadata_matches_imported_package():
    """Stale editable installs (or a stale openmm_orca.egg-info) fail here.

    Fix with ``pip install -e . --no-deps`` in the test environment.
    """
    try:
        installed = metadata.version("openmm-orca")
    except metadata.PackageNotFoundError:
        pytest.skip("openmm-orca is not installed (source tree on sys.path only)")
    assert installed == openmmorca.__version__
