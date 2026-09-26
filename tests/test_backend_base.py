"""Tests for openmmorca.backend.base (implementation plan Task 2)."""

from __future__ import annotations

import numpy as np
import pytest

from openmmorca.backend.base import QMRequest, QMResult, check_result


def make_request(n_qm=3, n_mm=2, **overrides):
    kwargs = dict(
        qm_elements=("O", "H", "H"),
        qm_positions_nm=np.zeros((n_qm, 3)),
    )
    if n_mm > 0:
        kwargs["mm_positions_nm"] = np.zeros((n_mm, 3))
        kwargs["mm_charges_e"] = np.zeros(n_mm)
    kwargs.update(overrides)
    return QMRequest(**kwargs)


def make_result(n_qm=3, n_mm=2, **overrides):
    kwargs = dict(
        energy_kj_mol=-1.0,
        qm_forces_kj_mol_nm=np.zeros((n_qm, 3)),
    )
    kwargs["mm_forces_kj_mol_nm"] = None if n_mm == 0 else np.zeros((n_mm, 3))
    kwargs.update(overrides)
    return QMResult(**kwargs)


def test_request_shapes():
    request = make_request(n_qm=3, n_mm=2)
    assert request.n_qm == 3
    assert request.n_mm == 2


def test_request_rejects_wrong_qm_shape():
    with pytest.raises(ValueError):
        make_request(qm_positions_nm=np.zeros((2, 3)))


def test_request_requires_positions_and_charges_together():
    with pytest.raises(ValueError):
        QMRequest(
            qm_elements=("O", "H", "H"),
            qm_positions_nm=np.zeros((3, 3)),
            mm_positions_nm=np.zeros((2, 3)),
        )


def test_request_rejects_count_mismatch():
    with pytest.raises(ValueError):
        QMRequest(
            qm_elements=("O", "H", "H"),
            qm_positions_nm=np.zeros((3, 3)),
            mm_positions_nm=np.zeros((2, 3)),
            mm_charges_e=np.zeros(3),
        )


def test_check_result_accepts_full_qm():
    check_result(make_request(n_mm=0), make_result(n_mm=0))


def test_check_result_requires_mm_forces():
    with pytest.raises(ValueError, match="MM forces"):
        check_result(make_request(n_mm=2), make_result(n_mm=2, mm_forces_kj_mol_nm=None))


def test_check_result_rejects_nan():
    forces = np.zeros((3, 3))
    forces[1, 0] = np.nan
    with pytest.raises(ValueError):
        check_result(make_request(n_mm=0), make_result(n_mm=0, qm_forces_kj_mol_nm=forces))


def test_check_result_rejects_wrong_shape():
    with pytest.raises(ValueError):
        check_result(make_request(n_mm=2), make_result(n_mm=2, mm_forces_kj_mol_nm=np.zeros((3, 3))))
