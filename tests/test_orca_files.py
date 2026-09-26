"""Tests for openmmorca.backend.orca_files (implementation plan Task 3)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from openmmorca.backend.orca_files import read_engrad, read_pcgrad, write_pointcharges
from openmmorca.errors import ORCAOutputError

DATA_DIR = Path(__file__).parent / "data"


def test_read_engrad_fixture():
    energy, gradient = read_engrad(DATA_DIR / "h2o_2pc.engrad", n_atoms=3)
    assert energy == pytest.approx(-75.977160671564, abs=0.0)
    assert gradient.shape == (3, 3)
    np.testing.assert_allclose(
        gradient[0], [-0.003408773893, -0.019449988281, 0.0], atol=1e-15
    )


def test_read_engrad_atom_count_mismatch():
    with pytest.raises(ORCAOutputError):
        read_engrad(DATA_DIR / "h2o_2pc.engrad", n_atoms=4)


def test_read_engrad_missing_file():
    missing = DATA_DIR / "does_not_exist.engrad"
    with pytest.raises(ORCAOutputError, match=str(missing)):
        read_engrad(missing, n_atoms=3)


def test_read_engrad_nan(tmp_path):
    copy = tmp_path / "nan.engrad"
    text = (DATA_DIR / "h2o_2pc.engrad").read_text()
    text = text.replace("-0.003408773893", "nan", 1)
    copy.write_text(text)
    with pytest.raises(ORCAOutputError):
        read_engrad(copy, n_atoms=3)


def test_read_pcgrad_fixture():
    gradient = read_pcgrad(DATA_DIR / "h2o_2pc.pcgrad", n_charges=2)
    assert gradient.shape == (2, 3)
    np.testing.assert_allclose(
        gradient[0], [0.012533896858, -0.001642805287, 0.0], atol=1e-15
    )


def test_read_pcgrad_count_mismatch():
    with pytest.raises(ORCAOutputError):
        read_pcgrad(DATA_DIR / "h2o_2pc.pcgrad", n_charges=3)


def test_write_pointcharges_format(tmp_path):
    path = tmp_path / "pc.pc"
    charges = [-0.834, 0.417]
    positions = [[3.000, 0.000, 0.000], [3.500, 0.800, 0.000]]
    write_pointcharges(path, charges, positions)
    lines = path.read_text().splitlines()
    assert lines[0] == "2"
    assert len(lines) == 3
    parsed = np.array([[float(v) for v in line.split()] for line in lines[1:]])
    np.testing.assert_allclose(parsed[:, 0], charges, atol=1e-10)
    np.testing.assert_allclose(parsed[:, 1:], positions, atol=1e-10)


def test_write_pointcharges_count_mismatch(tmp_path):
    with pytest.raises(ValueError):
        write_pointcharges(tmp_path / "pc.pc", [-0.834, 0.417], np.zeros((3, 3)))
