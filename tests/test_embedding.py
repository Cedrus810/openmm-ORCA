"""Tests for cutoff embedding charge selection (Task 20, spec §11.1)."""

from __future__ import annotations

import numpy as np
import pytest

import helpers
from openmmorca.qmmm.embedding import (
    CutoffEmbedding,
    check_cutoff_against_box,
    groups_from_topology,
)
from test_imaging import orthorhombic, truncated_octahedron

TIP3P = np.array([-0.834, 0.417, 0.417])


def two_waters(mm_offset):
    """QM water 0 at the origin region, MM water 1 displaced by *mm_offset*."""
    positions = helpers.water_positions(1)
    mm_water = helpers.water_positions(1) + np.asarray(mm_offset, dtype=float)
    positions = np.vstack([positions, mm_water])
    charges = np.concatenate([np.zeros(3), TIP3P])
    return positions, charges


def embedding(charges, cutoff=1.2):
    return CutoffEmbedding([(3, 4, 5)], charges, cutoff_nm=cutoff)


def test_groups_from_topology():
    topology = helpers.water_topology(3)
    groups = groups_from_topology(topology, [3, 4, 5, 6, 8])  # atom 7 not embedded
    assert groups == [(3, 4, 5), (6, 8)]


def test_group_inclusion_all_or_nothing():
    # MM water O at 0.45 nm from the QM O, its H1 pointing back toward the QM water.
    positions, charges = two_waters([0.0, 0.0, 0.0])
    positions[3:6] = np.array([[0.45, 0.0, 0.0], [0.36, 0.0, 0.0], [0.47, 0.09, 0.0]])
    emb = embedding(charges, cutoff=0.3)
    qm = positions[:3]
    nearest = {a: min(np.linalg.norm(positions[a] - q) for q in qm) for a in (3, 4, 5)}
    assert nearest[4] < 0.3 < min(nearest[3], nearest[5])  # only H1 is inside
    atoms, embedded_positions, embedded_charges = emb.select(qm, positions, orthorhombic(3.0))
    assert list(atoms) == [3, 4, 5]
    np.testing.assert_allclose(embedded_charges, TIP3P)
    np.testing.assert_allclose(embedded_positions, positions[3:6], atol=1e-12)
    assert emb.last_n_groups == 1


def test_minimum_image_positions():
    box = orthorhombic(3.0)
    positions, charges = two_waters([2.75, 0.0, 0.0])  # near the right edge
    positions[:3] += np.array([0.05, 1.0, 1.0])  # QM water near the left edge
    positions[3:6] += np.array([0.0, 1.0, 1.0])
    expected = positions[3:6] - box[0]  # the image just left of the QM water
    positions[4] -= box[1]  # and one MM H wrapped on its own
    emb = embedding(charges)
    atoms, embedded_positions, _ = emb.select(positions[:3], positions, box)
    assert list(atoms) == [3, 4, 5]
    np.testing.assert_allclose(embedded_positions, expected, atol=1e-12)


def test_cutoff_excludes_far_groups():
    positions, charges = two_waters([2.0, 0.0, 0.0])
    emb = embedding(charges, cutoff=1.2)
    atoms, embedded_positions, embedded_charges = emb.select(
        positions[:3], positions, orthorhombic(6.0)
    )
    assert len(atoms) == 0
    assert embedded_positions.shape == (0, 3) and embedded_charges.shape == (0,)
    assert emb.last_n_groups == 0


def test_change_counter():
    positions, charges = two_waters([0.4, 0.0, 0.0])
    emb = embedding(charges, cutoff=1.2)
    box = orthorhombic(6.0)
    emb.select(positions[:3], positions, box)
    assert (emb.last_n_groups, emb.last_changed) == (1, 1)  # entering counts too
    emb.select(positions[:3], positions, box)
    assert emb.last_changed == 0
    moved = positions.copy()
    moved[3:6] += np.array([2.0, 0.0, 0.0])
    emb.select(moved[:3], moved, box)
    assert (emb.last_n_groups, emb.last_changed) == (0, 1)


def test_cutoff_larger_than_half_box_rejected():
    check_cutoff_against_box(1.4, orthorhombic(3.0))
    with pytest.raises(ValueError, match="half"):
        check_cutoff_against_box(1.5, orthorhombic(3.0))
    # Truncated octahedron d = 3 nm: perpendicular width d·sqrt(2/3) ≈ 2.449 nm,
    # so 1.3 nm is rejected although it is below d / 2 = 1.5 nm.
    with pytest.raises(ValueError, match="half"):
        check_cutoff_against_box(1.3, truncated_octahedron(3.0))
    check_cutoff_against_box(1.2, truncated_octahedron(3.0))


def test_cutoff_plus_qm_extent_rejected():
    positions, charges = two_waters([0.4, 0.0, 0.0])
    emb = embedding(charges, cutoff=1.2)
    emb.select(positions[:3], positions, orthorhombic(2.9))  # 1.2 + D(0.15) < 1.45
    with pytest.raises(ValueError, match="QM extent"):
        emb.select(positions[:3], positions, orthorhombic(2.6))  # 1.2 + 0.15 ≥ 1.3


def test_invalid_groups_rejected():
    charges = np.zeros(6)
    with pytest.raises(ValueError, match="more than one group"):
        CutoffEmbedding([(3, 4), (4, 5)], charges)
    with pytest.raises(ValueError, match="empty"):
        CutoffEmbedding([(3, 4), ()], charges)
    with pytest.raises(ValueError, match="cutoff"):
        CutoffEmbedding([(3, 4)], charges, cutoff_nm=0.0)
